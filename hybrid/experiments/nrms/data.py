#!/usr/bin/env python3
"""
Caricamento dati, catalog coverage e dataset PyTorch per NRMS.

Catalog coverage: il catalogo usato per il
lookup dei titoli e' `dataset/MINDsmall_train/news.tsv`, che copre il 100%
dei candidati di ogni giorno TRAIN (verificato di seguito e gia' in
`VALIDAZIONE_METODOLOGICA.md` sezione 10.2) — a differenza di
`dev/news.tsv`, che copre solo il 38-86% a seconda del giorno e che NON
viene usato qui come catalogo per candidati/history di periodi TRAIN.
Per l'eventuale, rarissimo candidato assente anche da train/news.tsv, la
strategia deterministica e': titolo vuoto -> tutti i token sono PAD ->
il news encoder produce un vettore "neutro" (nessun crash, nessuna
euristica nascosta).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    TRAIN_NEWS_PATH, DEV_NEWS_PATH, TRAIN_BEHAVIORS_PATH,
    MAX_HISTORY_LEN, MAX_TITLE_LEN, NUM_NEGATIVES, NEGATIVE_SAMPLING_SEED,
    PAD_IDX,
)
from tokenizer import Vocab, build_title_lookup, tokenize

BEHAV_COLS = ["imp_id", "user_id", "time", "history", "impressions"]
NEWS_COLS = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]

EMPTY_TITLE_IDS = [PAD_IDX] * MAX_TITLE_LEN


def load_behaviors_with_date():
    df = pd.read_csv(TRAIN_BEHAVIORS_PATH, sep="\t", header=None, names=BEHAV_COLS)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["date_str"] = df["time"].dt.strftime("%Y-%m-%d")
    return df


def parse_history(hist):
    if pd.isna(hist):
        return []
    return str(hist).split()


def parse_impressions(impr):
    if pd.isna(impr):
        return [], np.array([], dtype=int)
    cands, labels = [], []
    for tok in str(impr).split():
        nid, lab = tok.rsplit("-", 1)
        cands.append(nid)
        labels.append(int(lab))
    return cands, np.asarray(labels)


def compute_catalog_coverage(behav: pd.DataFrame, dates: list[str], train_ids: set[str], dev_ids: set[str]) -> dict:
    n_cand_total, n_cand_train_cov, n_cand_dev_cov = 0, 0, 0
    n_clicked_total, n_clicked_train_cov, n_clicked_dev_cov = 0, 0, 0
    sub = behav[behav["date_str"].isin(dates)]
    for imp in sub["impressions"].dropna():
        for tok in str(imp).split():
            nid, lab = tok.rsplit("-", 1)
            n_cand_total += 1
            if nid in train_ids:
                n_cand_train_cov += 1
            if nid in dev_ids:
                n_cand_dev_cov += 1
            if lab == "1":
                n_clicked_total += 1
                if nid in train_ids:
                    n_clicked_train_cov += 1
                if nid in dev_ids:
                    n_clicked_dev_cov += 1
    return {
        "dates": dates,
        "pct_candidati_con_titolo_train_news": round(100 * n_cand_train_cov / n_cand_total, 2) if n_cand_total else None,
        "pct_candidati_con_titolo_dev_news": round(100 * n_cand_dev_cov / n_cand_total, 2) if n_cand_total else None,
        "pct_clicked_con_titolo_train_news": round(100 * n_clicked_train_cov / n_clicked_total, 2) if n_clicked_total else None,
        "pct_clicked_con_titolo_dev_news": round(100 * n_clicked_dev_cov / n_clicked_total, 2) if n_clicked_total else None,
        "n_candidati_totali": n_cand_total,
        "n_clicked_totali": n_clicked_total,
    }


class NRMSTrainDataset(Dataset):
    """Un sample per (impression, candidato positivo): NUM_NEGATIVES negativi
    campionati dagli stessi candidati negativi dell'impression. Se
    l'impression ha meno di NUM_NEGATIVES negativi disponibili, si
    ricampiona CON ripetizione (documentato) per raggiungere la dimensione
    fissa richiesta dal batching. Il seed e' fisso; il campionamento e'
    ri-eseguito (nuovo seed derivato) a ogni chiamata di `resample_epoch`,
    cosi' come previsto per un resampling per epoch.
    """

    def __init__(self, behav, dates, title_lookup, seed=NEGATIVE_SAMPLING_SEED):
        self.title_lookup = title_lookup
        self.base_seed = seed
        self.samples = []  # (history_ids_list, pos_id, list_of_impression_neg_ids)
        sub = behav[behav["date_str"].isin(dates)]
        for row in sub.itertuples(index=False):
            h = parse_history(row.history)
            cands, y = parse_impressions(row.impressions)
            if not cands or y.sum() == 0:
                continue
            pos_ids = [cands[i] for i in np.where(y == 1)[0]]
            neg_ids = [cands[i] for i in np.where(y == 0)[0]]
            if not neg_ids:
                continue
            h = h[-MAX_HISTORY_LEN:]
            for pos_id in pos_ids:
                self.samples.append((h, pos_id, neg_ids))
        self.epoch = 0

    def resample_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)

    def _title(self, nid):
        return self.title_lookup.get(nid, EMPTY_TITLE_IDS)

    def __getitem__(self, idx):
        h, pos_id, neg_ids = self.samples[idx]
        rng = random.Random(hash((self.base_seed, self.epoch, idx)) & 0xFFFFFFFF)
        if len(neg_ids) >= NUM_NEGATIVES:
            sampled_negs = rng.sample(neg_ids, NUM_NEGATIVES)
        else:
            sampled_negs = [rng.choice(neg_ids) for _ in range(NUM_NEGATIVES)]

        cand_ids = [pos_id] + sampled_negs  # positivo sempre in posizione 0 (target fisso)
        cand_titles = [self._title(c) for c in cand_ids]

        hist_titles = [self._title(nid) for nid in h]
        n_pad = MAX_HISTORY_LEN - len(hist_titles)
        hist_titles = [EMPTY_TITLE_IDS] * n_pad + hist_titles  # left-padding (history vuota -> tutta padding)
        hist_mask = [0.0] * n_pad + [1.0] * len(h)

        return {
            "history_title_ids": torch.tensor(hist_titles, dtype=torch.long),
            "history_mask": torch.tensor(hist_mask, dtype=torch.float32),
            "candidate_title_ids": torch.tensor(cand_titles, dtype=torch.long),
            "target": torch.tensor(0, dtype=torch.long),  # il positivo e' sempre in posizione 0
        }


class NRMSEvalRecord:
    __slots__ = ("history_title_ids", "history_mask", "candidate_title_ids", "y_true", "cands")


def build_eval_records(behav, dates, title_lookup):
    """Costruisce record di valutazione (candidate set COMPLETO, nessun
    sampling) per un elenco di date. Usato per VALIDATION e INTERNAL_TEST."""
    records = []
    sub = behav[behav["date_str"].isin(dates)]
    for row in sub.itertuples(index=False):
        h = parse_history(row.history)
        cands, y = parse_impressions(row.impressions)
        if not cands or y.sum() == 0 or y.sum() == len(y):
            continue
        h = h[-MAX_HISTORY_LEN:]
        hist_titles = [title_lookup.get(nid, EMPTY_TITLE_IDS) for nid in h]
        n_pad = MAX_HISTORY_LEN - len(hist_titles)
        hist_titles = [EMPTY_TITLE_IDS] * n_pad + hist_titles
        hist_mask = [0.0] * n_pad + [1.0] * len(h)

        rec = NRMSEvalRecord()
        rec.history_title_ids = torch.tensor(hist_titles, dtype=torch.long)
        rec.history_mask = torch.tensor(hist_mask, dtype=torch.float32)
        rec.candidate_title_ids = torch.tensor([title_lookup.get(c, EMPTY_TITLE_IDS) for c in cands], dtype=torch.long)
        rec.y_true = y
        rec.cands = cands
        records.append(rec)
    return records
