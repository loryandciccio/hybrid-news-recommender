#!/usr/bin/env python3
"""
Inferenza batched di NRMS su record di valutazione (candidate set completo,
dimensione variabile). Per efficienza, gli score sono calcolati candidato
per candidato tramite un batch di history+candidati con padding al
candidate-count massimo del batch (mascherato) — evita un forward per
singola impression mantenendo la correttezza (i punteggi per candidati di
padding vengono scartati, non influenzano softmax/self-attention perché il
padding e' solo lato "numero di candidati", non lato token, e ogni
candidato è comunque incapsulato/valutato indipendentemente dagli altri nel
dot product finale — l'unica interazione cross-candidato di NRMS è nella
self-attention della *history*, condivisa e non influenzata dal numero di
candidati).
"""

from __future__ import annotations

import numpy as np
import torch

from config import MAX_HISTORY_LEN, MAX_TITLE_LEN, PAD_IDX, BATCH_SIZE as EVAL_BATCH
from metrics import evaluate_scores


@torch.no_grad()
def score_records(model, records, device, batch_size=64):
    model.eval()
    all_scores = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        max_c = max(r.candidate_title_ids.shape[0] for r in batch)

        hist = torch.stack([r.history_title_ids for r in batch]).to(device)
        hist_mask = torch.stack([r.history_mask for r in batch]).to(device)

        cand_padded = torch.zeros(len(batch), max_c, MAX_TITLE_LEN, dtype=torch.long)
        for i, r in enumerate(batch):
            c = r.candidate_title_ids.shape[0]
            cand_padded[i, :c] = r.candidate_title_ids
        cand_padded = cand_padded.to(device)

        scores = model(hist, hist_mask, cand_padded).cpu().numpy()
        for i, r in enumerate(batch):
            c = r.candidate_title_ids.shape[0]
            all_scores.append(scores[i, :c])
    return all_scores


def evaluate_model_on_records(model, records, device, item_popularity=None, pop_total=None):
    scores = score_records(model, records, device, batch_size=EVAL_BATCH)
    y_true_list = [r.y_true for r in records]
    cands_list = [r.cands for r in records]
    return evaluate_scores(y_true_list, scores, cands_list, item_popularity, pop_total)
