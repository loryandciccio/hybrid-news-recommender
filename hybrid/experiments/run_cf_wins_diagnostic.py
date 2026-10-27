#!/usr/bin/env python3
"""
Costruzione del dataset DIAGNOSTICO per la domanda "possiamo prevedere, prima
del click, quando il CF e' piu' utile del CBF?" (Fase 4 della validazione
metodologica). NON addestra alcun classificatore: calcola solo la label
diagnostica CF_WINS e le feature pre-raccomandazione, per un'ispezione
descrittiva (correlazioni, medie per gruppo).

NON modifica hybrid_recommender.py: riusa gli stessi metodi privati
(_get_baseline_scores, _get_cf_scores, _get_cbf_scores, _norm) gia' usati da
run_complementarity_analysis.py.

Label diagnostica (definita esplicitamente, solo su impression a singolo
positivo, per evitare ambiguita' su quale candidato cliccato considerare):

    rank_CF  = posizione (1-based, tie-break su news_id crescente) del
               candidato cliccato nel ranking indotto da CF puro (0/1/0)
    rank_CBF = idem, per CBF puro (0/0/1)
    CF_WINS  = 1 se rank_CF < rank_CBF (il CF piazza il click cliccato piu'
               in alto del CBF), 0 se rank_CF >= rank_CBF (include i pareggi,
               trattati conservativamente come "non vince il CF")

Tutte le feature elencate sono calcolabili PRIMA di conoscere il click:
dipendono solo dalla cronologia dell'utente e dal candidate set
dell'impression, mai dalla label.

Uso (dalla root del progetto):
    python3 hybrid/experiments/run_cf_wins_diagnostic.py [--limit N] [--offset N]

Output:
    hybrid/experiments/cf_wins_diagnostic.csv
"""

from __future__ import annotations

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "hybrid"))

from hybrid_recommender import (
    HybridRecommender,
    DEFAULT_EMBEDDINGS,
    DEFAULT_NEWS,
    DEFAULT_TRAIN_BEHAVIORS,
    DEFAULT_DEV_BEHAVIORS,
)

OUT_DIR = Path(__file__).resolve().parent


def _resolve_path(rel: str) -> str:
    p = PROJECT_ROOT / rel
    return str(p) if p.exists() else rel


def rank_of_clicked(y_score: np.ndarray, cands: list[str], clicked_id: str) -> int:
    """Rank 1-based del candidato cliccato, tie-break su news_id crescente (coerente col resto del progetto)."""
    order = np.lexsort((np.asarray(cands, dtype=str), -y_score))
    ranked_ids = [cands[i] for i in order]
    return ranked_ids.index(clicked_id) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    model = HybridRecommender()
    model.load_data(_resolve_path(DEFAULT_EMBEDDINGS), _resolve_path(DEFAULT_NEWS), _resolve_path(DEFAULT_TRAIN_BEHAVIORS))

    cols = ["imp_id", "user_id", "time", "history", "impressions"]
    df_full = pl.read_csv(_resolve_path(DEFAULT_DEV_BEHAVIORS), separator="\t", has_header=False, new_columns=cols, quote_char=None)
    df = df_full.slice(args.offset, args.limit) if args.limit else df_full

    rows = []
    n_processed = 0

    for row in df.iter_rows(named=True):
        h = str(row["history"]).split() if row["history"] else []
        imps = str(row["impressions"]).split() if row["impressions"] else []
        if not h or not imps:
            continue

        cands = [imp.split("-")[0] for imp in imps]
        y_true = np.array([int(imp.split("-")[1]) for imp in imps])

        if y_true.sum() != 1:
            continue  # solo impression a singolo positivo, per una label CF_WINS non ambigua

        clicked_id = cands[int(np.argmax(y_true))]

        baseline_raw = model._get_baseline_scores(cands)
        cf_raw = model._get_cf_scores(h, cands)
        cbf_raw = model._get_cbf_scores(h, cands)

        cf_norm = model._norm(cf_raw)
        cbf_norm = model._norm(cbf_raw)

        rank_cf = rank_of_clicked(cf_norm, cands, clicked_id)
        rank_cbf = rank_of_clicked(cbf_norm, cands, clicked_id)
        cf_wins = int(rank_cf < rank_cbf)

        # --- Feature IMPRESSION-level, tutte disponibili prima del click ---
        n_candidates = len(cands)
        history_length = len(h)

        train_pops = np.array([model.item_popularity.get(c, 0) for c in cands], dtype=float)
        pct_unseen = float(np.mean(train_pops == 0))
        pct_warm = 1.0 - pct_unseen
        mean_pop = float(train_pops.mean())
        median_pop = float(np.median(train_pops))
        max_pop = float(train_pops.max())

        cf_std = float(np.std(cf_raw))
        cbf_std = float(np.std(cbf_raw))
        cf_max = float(cf_raw.max())
        cbf_max = float(cbf_raw.max())

        cf_sorted = np.sort(cf_raw)[::-1]
        cbf_sorted = np.sort(cbf_raw)[::-1]
        cf_gap_top1_top2 = float(cf_sorted[0] - cf_sorted[1]) if len(cf_sorted) > 1 else 0.0
        cbf_gap_top1_top2 = float(cbf_sorted[0] - cbf_sorted[1]) if len(cbf_sorted) > 1 else 0.0

        k_top = min(5, n_candidates)
        top_cf_set = set(np.argsort(-cf_raw)[:k_top].tolist())
        top_cbf_set = set(np.argsort(-cbf_raw)[:k_top].tolist())
        topk_agreement = len(top_cf_set & top_cbf_set) / k_top if k_top > 0 else np.nan

        rows.append({
            "history_length": history_length,
            "n_candidates": n_candidates,
            "pct_unseen_candidates": pct_unseen,
            "pct_warm_candidates": pct_warm,
            "mean_train_pop_candidates": mean_pop,
            "median_train_pop_candidates": median_pop,
            "max_train_pop_candidates": max_pop,
            "cf_score_std": cf_std,
            "cbf_score_std": cbf_std,
            "cf_score_max": cf_max,
            "cbf_score_max": cbf_max,
            "cf_gap_top1_top2": cf_gap_top1_top2,
            "cbf_gap_top1_top2": cbf_gap_top1_top2,
            "topk5_agreement_cf_cbf": topk_agreement,
            "rank_CF": rank_cf,
            "rank_CBF": rank_cbf,
            "CF_WINS": cf_wins,
        })

        n_processed += 1
        if n_processed % 10000 == 0:
            print(f"  processate {n_processed} impression a singolo positivo...", flush=True)

    print(f"\nImpression a singolo positivo processate: {n_processed}")
    with open(OUT_DIR / "cf_wins_diagnostic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Scritto cf_wins_diagnostic.csv ({len(rows)} righe)")


if __name__ == "__main__":
    main()
