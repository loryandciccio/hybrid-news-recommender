#!/usr/bin/env python3
"""
Tokenizzazione deterministica e vocabolario costruito SOLO sui titoli delle
news presenti in `dataset/MINDsmall_train/news.tsv` (il catalogo del
periodo TRAIN, 9-14 novembre — mai `dev/news.tsv`, per evitare esattamente
il bias di catalogo documentato in VALIDAZIONE_METODOLOGICA.md sezione 10.2).

Pipeline:
    lowercase -> split su tutto cio' che non e' [a-z0-9] (regex) -> token

Nessun dato di VALIDATION/INTERNAL_TEST entra nella costruzione del
vocabolario: e' costruito una sola volta da train/news.tsv, che per
costruzione MIND-small copre il 100% dei candidati di ogni giorno TRAIN
(verificato in `hybrid/experiments/run_cf_training_source_audit.py` e
sezione 10.2 di VALIDAZIONE_METODOLOGICA.md: 100% di copertura candidati in
train/news.tsv, contro 38-86% di dev/news.tsv a seconda del giorno).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from config import (
    TRAIN_NEWS_PATH, RESULTS_DIR, MAX_TITLE_LEN, MIN_TOKEN_FREQ,
    PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX,
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
NEWS_COLS = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, token_to_idx: dict[str, int]):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {i: t for t, i in token_to_idx.items()}

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, tokens: list[str], max_len: int = MAX_TITLE_LEN) -> list[int]:
        ids = [self.token_to_idx.get(t, UNK_IDX) for t in tokens[:max_len]]
        ids = ids + [PAD_IDX] * (max_len - len(ids))
        return ids

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.token_to_idx, f)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with open(path) as f:
            token_to_idx = json.load(f)
        return cls(token_to_idx)


def build_vocab_from_train_news() -> Vocab:
    df = pd.read_csv(TRAIN_NEWS_PATH, sep="\t", header=None, names=NEWS_COLS)
    counter = Counter()
    for title in df["Title"].fillna(""):
        counter.update(tokenize(title))

    token_to_idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    next_idx = 2
    for token, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if freq < MIN_TOKEN_FREQ:
            continue
        token_to_idx[token] = next_idx
        next_idx += 1

    vocab = Vocab(token_to_idx)
    print(f"Vocabolario costruito da {len(df)} titoli TRAIN: {len(counter)} token distinti osservati, "
          f"{len(vocab)} nel vocabolario (soglia freq>={MIN_TOKEN_FREQ}, incluso PAD/UNK)")
    return vocab


def build_title_lookup(news_path: Path, vocab: Vocab) -> dict[str, list[int]]:
    """Precalcola l'encoding del titolo per ogni News_ID di un file news.tsv."""
    df = pd.read_csv(news_path, sep="\t", header=None, names=NEWS_COLS)
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.News_ID] = vocab.encode(tokenize(row.Title))
    return lookup


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab_from_train_news()
    vocab.save(RESULTS_DIR / "vocab.json")
    print(f"Scritto {RESULTS_DIR / 'vocab.json'}")


if __name__ == "__main__":
    main()
