#!/usr/bin/env python3
"""
NRMS (Wu et al., 2019) — implementazione fedele al paradigma originale:

NEWS ENCODER:  token embedding -> multi-head self-attention (mascherata sui
               token di PAD) -> additive attention -> news vector
USER ENCODER:  sequenza di news vector della history -> multi-head
               self-attention (mascherata sugli slot di history vuoti) ->
               additive attention -> user vector
SCORING:       dot product tra user vector e ciascun candidate news vector

Semplificazioni dichiarate (si veda anche config.py e NRMS_REPORT.md):
- Solo il titolo come input testuale (niente abstract/categoria/entità).
- Un solo layer di self-attention per encoder (come nel paper originale).
- Nessun embedding pretrained: appreso da zero, inizializzazione Xavier.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    WORD_EMBEDDING_DIM, NUM_ATTENTION_HEADS, HEAD_DIM, NEWS_VECTOR_DIM,
    ADDITIVE_ATTN_DIM, DROPOUT, PAD_IDX,
)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention multi-head standard (Vaswani et al.), con mascheratura
    opzionale per ignorare posizioni di padding."""

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(dim, inner_dim)
        self.k_proj = nn.Linear(dim, inner_dim)
        self.v_proj = nn.Linear(dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.scale = head_dim ** -0.5

    def forward(self, x, mask=None):
        # x: (B, L, dim); mask: (B, L) 1=valido, 0=padding
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        if mask is not None:
            key_mask = mask[:, None, None, :].bool()  # (B,1,1,L)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # righe interamente mascherate -> 0
        out = torch.matmul(attn, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        return self.out_proj(out)


class AdditiveAttention(nn.Module):
    """Attention additiva (Bahdanau-style) per aggregare una sequenza in un
    singolo vettore, con mascheratura per gli slot non validi."""

    def __init__(self, dim: int, attn_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, attn_dim)
        self.query = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: (B, L, dim); mask: (B, L) 1=valido, 0=padding
        scores = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, dim)
        return out, weights


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, WORD_EMBEDDING_DIM, padding_idx=PAD_IDX)
        nn.init.xavier_uniform_(self.embedding.weight.data[1:])  # PAD resta a 0
        self.dropout = nn.Dropout(DROPOUT)
        self.self_attn = MultiHeadSelfAttention(WORD_EMBEDDING_DIM, NUM_ATTENTION_HEADS, HEAD_DIM)
        self.additive_attn = AdditiveAttention(NUM_ATTENTION_HEADS * HEAD_DIM, ADDITIVE_ATTN_DIM)

    def forward(self, title_ids):
        # title_ids: (B, T) id di token, PAD_IDX=0
        mask = (title_ids != PAD_IDX).float()
        x = self.dropout(self.embedding(title_ids))
        x = self.self_attn(x, mask=mask)
        x = self.dropout(x)
        news_vec, _ = self.additive_attn(x, mask=mask)
        return news_vec  # (B, NEWS_VECTOR_DIM)


class UserEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(NEWS_VECTOR_DIM, NUM_ATTENTION_HEADS, HEAD_DIM)
        self.additive_attn = AdditiveAttention(NUM_ATTENTION_HEADS * HEAD_DIM, ADDITIVE_ATTN_DIM)

    def forward(self, history_news_vecs, history_mask):
        # history_news_vecs: (B, H, NEWS_VECTOR_DIM); history_mask: (B, H)
        x = self.self_attn(history_news_vecs, mask=history_mask)
        user_vec, attn_weights = self.additive_attn(x, mask=history_mask)
        return user_vec, attn_weights


class NRMS(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size)
        self.user_encoder = UserEncoder()

    def encode_news(self, title_ids_flat):
        return self.news_encoder(title_ids_flat)

    def forward(self, history_title_ids, history_mask, candidate_title_ids):
        """
        history_title_ids:  (B, H, T)
        history_mask:       (B, H)
        candidate_title_ids:(B, C, T)
        Ritorna scores (B, C).
        """
        B, H, T = history_title_ids.shape
        _, C, _ = candidate_title_ids.shape

        hist_flat = history_title_ids.view(B * H, T)
        hist_vecs = self.news_encoder(hist_flat).view(B, H, -1)
        user_vec, _ = self.user_encoder(hist_vecs, history_mask)  # (B, NEWS_VECTOR_DIM)

        cand_flat = candidate_title_ids.view(B * C, T)
        cand_vecs = self.news_encoder(cand_flat).view(B, C, -1)

        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)  # (B, C)
        return scores

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
