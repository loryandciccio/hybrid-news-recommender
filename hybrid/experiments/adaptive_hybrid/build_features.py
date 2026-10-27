#!/usr/bin/env python3
"""
Costruisce, per ciascuna impression valutabile (>=1 positivo, >=1 negativo,
history non vuota, candidate set non vuoto), gli score CF/CBF grezzi e
normalizzati (stessa normalizzazione di hybrid_recommender.py: min-max per
candidate set) e un vettore di feature PRE-CLICK (nessuna dipende dal
candidato cliccato o dalla sua label).

Due stage, mai mescolati:
  - "dev":  CF_enriched fittato su FIT (9-11 nov). Produce le feature per
            GATING_TRAIN (12 nov) e GATING_VALIDATION (13 nov).
  - "test": CF_enriched fittato su FIT+VALIDATION (9-13 nov). Produce le
            feature per INTERNAL_TEST (14 nov). Da eseguire UNA SOLA VOLTA,
            dopo aver congelato feature/soglie/architettura sullo stage dev.

NON modifica hybrid_recommender.py. Nessun modello di produzione alterato.

Uso (dalla root del progetto):
    python3 hybrid/experiments/adaptive_hybrid/build_features.py --stage dev
    python3 hybrid/experiments/adaptive_hybrid/build_features.py --stage test
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import (
    RESULTS_DIR, EPSILON_CF, GATING_TRAIN_DATES, GATING_VALIDATION_DATES,
    INTERNAL_TEST_DATES, DEV_CF_FIT_DATES, TEST_CF_FIT_DATES,
    load_news_universe, build_cf_from_dates, load_train_behaviors,
    parse_history, parse_impressions, cf_scores, cbf_scores, norm,
    cf_available_mask,
)


def build_record(row, sim, idx_map, item_counts, pop, pop_total, embeddings, news_id_to_idx, news_metadata):
    h = parse_history(row.history)
    cands, y_true = parse_impressions(row.impressions)
    if not h or not cands or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None

    cf_raw = cf_scores(h, cands, sim, idx_map)
    cbf_raw = cbf_scores(h, cands, embeddings, news_id_to_idx, news_metadata)
    cf_norm = norm(cf_raw.copy())
    cbf_norm = norm(cbf_raw.copy())

    cf_available = bool(cf_available_mask(cf_raw))

    n_candidates = len(cands)
    history_length = len(h)
    # "Warm" = ha almeno una co-occorrenza osservata in FIT (item_counts>0), NON
    # semplice appartenenza al catalogo news.tsv usato come universo (si veda
    # la nota in common.py:build_cf_from_dates).
    warm_mask = np.array([item_counts.get(c, 0) > 0 for c in cands])
    warm_candidate_ratio = float(warm_mask.mean())
    pct_cf_representation = warm_candidate_ratio  # stessa quantita', nome esplicito per chiarezza

    cf_std = float(np.std(cf_raw))
    cbf_std = float(np.std(cbf_raw))
    cf_max = float(cf_raw.max())
    cbf_max = float(cbf_raw.max())
    cf_sorted = np.sort(cf_raw)[::-1]
    cbf_sorted = np.sort(cbf_raw)[::-1]
    cf_gap = float(cf_sorted[0] - cf_sorted[1]) if len(cf_sorted) > 1 else 0.0
    cbf_gap = float(cbf_sorted[0] - cbf_sorted[1]) if len(cbf_sorted) > 1 else 0.0

    k_top = min(5, n_candidates)
    top_cf_idx = set(np.argsort(-cf_raw)[:k_top].tolist())
    top_cbf_idx = set(np.argsort(-cbf_raw)[:k_top].tolist())
    top5_overlap = len(top_cf_idx & top_cbf_idx) / k_top if k_top > 0 else np.nan

    if n_candidates > 1 and np.std(cf_raw) > 0 and np.std(cbf_raw) > 0:
        spearman_corr, _ = spearmanr(cf_raw, cbf_raw)
    else:
        spearman_corr = 0.0

    train_pops = np.array([pop.get(c, 0) for c in cands], dtype=float)
    pop_mean = float(train_pops.mean())
    pop_median = float(np.median(train_pops))

    features = {
        "history_length": history_length,
        "candidate_count": n_candidates,
        "CF_AVAILABLE": int(cf_available),
        "cf_score_std": cf_std,
        "cf_score_max": cf_max,
        "cf_gap_top1_top2": cf_gap,
        "cbf_score_std": cbf_std,
        "cbf_score_max": cbf_max,
        "cbf_gap_top1_top2": cbf_gap,
        "top5_overlap_cf_cbf": top5_overlap,
        "spearman_cf_cbf": float(spearman_corr) if spearman_corr is not None and not np.isnan(spearman_corr) else 0.0,
        "warm_candidate_ratio": warm_candidate_ratio,
        "pct_cf_representation": pct_cf_representation,
        "popularity_mean_fit": pop_mean,
        "popularity_median_fit": pop_median,
    }

    return {
        "date": row.date_str,
        "cands": cands,
        "y_true": y_true.astype(np.int8),
        "cf_norm": cf_norm.astype(np.float32),
        "cbf_norm": cbf_norm.astype(np.float32),
        "features": features,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "test"], required=True)
    args = ap.parse_args()

    behav = load_train_behaviors()
    news_id_to_idx, news_metadata, embeddings, news_ids_universe = load_news_universe()

    if args.stage == "dev":
        fit_dates = DEV_CF_FIT_DATES
        eval_groups = {"gating_train": GATING_TRAIN_DATES, "gating_validation": GATING_VALIDATION_DATES}
    else:
        fit_dates = TEST_CF_FIT_DATES
        eval_groups = {"internal_test": INTERNAL_TEST_DATES}

    print(f"[{args.stage}] Costruzione CF_enriched da FIT={fit_dates}...")
    sim, idx_map, pop, item_counts, n_fit_rows = build_cf_from_dates(behav, fit_dates, news_ids_universe, enriched=True)
    pop_total = float(sum(pop.values())) or 1.0
    print(f"  Righe FIT: {n_fit_rows}, item con rappresentazione CF reale (item_counts>0): {len(item_counts)}")

    for group_name, dates in eval_groups.items():
        sub = behav[behav["date_str"].isin(dates)]
        records = []
        for row in sub.itertuples(index=False):
            rec = build_record(row, sim, idx_map, item_counts, pop, pop_total, embeddings, news_id_to_idx, news_metadata)
            if rec is not None:
                records.append(rec)
        out_path = RESULTS_DIR / f"features_{group_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(records, f)
        n_cf_avail = sum(r["features"]["CF_AVAILABLE"] for r in records)
        print(f"  [{group_name}] {len(records)} impression valutabili, CF_AVAILABLE=1: {n_cf_avail} ({100*n_cf_avail/len(records):.2f}%)  -> {out_path}")


if __name__ == "__main__":
    main()
