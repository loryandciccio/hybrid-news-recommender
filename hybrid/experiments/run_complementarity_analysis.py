#!/usr/bin/env python3
"""
Analisi della complementarita' tra i segnali Baseline / CF / CBF dell'Hybrid Recommender.

IMPORTANTE: questo script NON modifica in alcun modo hybrid_recommender.py.
Riusa direttamente i metodi esistenti (_get_baseline_scores, _get_cf_scores,
_get_cbf_scores, _norm, get_item_novelty, ndcg_at_k) importandoli dal modulo
originale, cosi' che ogni numero prodotto qui sia calcolato con ESATTAMENTE
la stessa logica di scoring del progetto principale.

Ottimizzazione chiave: i tre segnali grezzi (baseline, CF, CBF) vengono
calcolati UNA SOLA VOLTA per impression (il calcolo del segnale CF, che
materializza righe dense dalla matrice sparsa ItemKNN, e' il piu' costoso),
poi normalizzati e ricombinati con pesi diversi per ottenere le 5
configurazioni di interesse SENZA ricalcolare nulla. Questo rende possibile
analizzare l'intero DEV set in un'unica passata, invece dei consueti
campioni di 2000 impression usati per la weight sweep.

Configurazioni analizzate:
    Baseline pura      (1, 0, 0)
    CF puro            (0, 1, 0)
    CBF puro           (0, 0, 1)
    Hybrid attuale      (0, 0.03, 0.97)
    Hybrid CF 10%       (0, 0.10, 0.90)

Uso (dalla root del progetto):
    python3 hybrid/experiments/run_complementarity_analysis.py [--limit N] [--offset N]

Di default analizza l'INTERO DEV set (nessun limit/offset).

Output:
    hybrid/experiments/impression_level_results.csv
        Una riga per impression valida, con covariate (history_length,
        n_candidates, popolarita' TRAIN dell'item cliccato, flag cold-start)
        e per ciascuna delle 5 configurazioni: AUC (se disponibile), MRR,
        nDCG@5/10, Precision@5, Recall@5, Novelty@5/10, hit@1, hit@5.
    hybrid/experiments/history_length_distribution.csv
        Distribuzione grezza della lunghezza della history nel DEV.
    hybrid/experiments/ranking_correlation.csv
        Per impression: correlazione di Spearman tra score CF e CBF (sui
        candidati), overlap e Jaccard dei rispettivi Top-5.

NOTA SUL LEAKAGE (si veda anche ANALISI_COMPLEMENTARITA.md, sezione 12):
tutte le covariate usate per segmentare (history_length, popolarita' TRAIN
dei candidati, flag cold-start) sono calcolate esclusivamente da
informazioni disponibili PRIMA della valutazione (history dell'utente,
item_popularity del solo TRAIN). L'unica cosa mai usata come input ai
modelli e' la label clicked/non-clicked, che qui viene usata solo per
calcolare le metriche di valutazione (AUC/MRR/nDCG/hit@K), MAI come feature.
"""

from __future__ import annotations

import sys
import csv
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "hybrid"))

from hybrid_recommender import (
    HybridRecommender,
    DEFAULT_EMBEDDINGS,
    DEFAULT_NEWS,
    DEFAULT_TRAIN_BEHAVIORS,
    DEFAULT_DEV_BEHAVIORS,
    ndcg_at_k,
)

OUT_DIR = Path(__file__).resolve().parent

CONFIGS = {
    "baseline_pure": (1.0, 0.0, 0.0),
    "cf_pure": (0.0, 1.0, 0.0),
    "cbf_pure": (0.0, 0.0, 1.0),
    "hybrid_0_03_97": (0.0, 0.03, 0.97),
    "hybrid_0_10_90": (0.0, 0.10, 0.90),
}


def _resolve_path(rel: str) -> str:
    p = PROJECT_ROOT / rel
    return str(p) if p.exists() else rel


def combine(baseline_norm, cf_norm, cbf_norm, weights):
    bw, cfw, cbfw = weights
    total = bw + cfw + cbfw
    return (bw / total) * baseline_norm + (cfw / total) * cf_norm + (cbfw / total) * cbf_norm


def mrr_first_hit(y_sorted: np.ndarray) -> float:
    hits = np.where(y_sorted == 1)[0]
    return 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Numero di impression DEV da analizzare (default: intero DEV set)")
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    model = HybridRecommender()  # pesi di default irrilevanti: usiamo solo i _get_*_scores grezzi
    model.load_data(_resolve_path(DEFAULT_EMBEDDINGS), _resolve_path(DEFAULT_NEWS), _resolve_path(DEFAULT_TRAIN_BEHAVIORS))

    cols = ["imp_id", "user_id", "time", "history", "impressions"]
    df_full = pl.read_csv(_resolve_path(DEFAULT_DEV_BEHAVIORS), separator="\t", has_header=False, new_columns=cols, quote_char=None)
    df = df_full.slice(args.offset, args.limit) if args.limit else df_full

    # --- Distribuzione grezza della history length (su tutte le righe del DEV, anche quelle poi scartate) ---
    hist_lengths_all = []
    for row in df.iter_rows(named=True):
        h = str(row["history"]).split() if row["history"] else []
        hist_lengths_all.append(len(h))
    hist_counter = Counter(hist_lengths_all)
    with open(OUT_DIR / "history_length_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["history_length", "n_impressions"])
        for length in sorted(hist_counter):
            w.writerow([length, hist_counter[length]])
    print(f"Distribuzione history length salvata ({len(hist_counter)} valori distinti, {len(hist_lengths_all)} impression totali nel DEV analizzato)")

    imp_rows = []
    corr_rows = []

    n_total = len(df)
    n_processed = 0
    n_skipped_no_hist_or_cand = 0
    n_skipped_single_class = 0

    for row in df.iter_rows(named=True):
        h = str(row["history"]).split() if row["history"] else []
        imps = str(row["impressions"]).split() if row["impressions"] else []
        if not h or not imps:
            n_skipped_no_hist_or_cand += 1
            continue

        cands = [imp.split("-")[0] for imp in imps]
        y_true = np.array([int(imp.split("-")[1]) for imp in imps])

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            n_skipped_single_class += 1
            continue

        # --- Segnali grezzi, calcolati UNA SOLA VOLTA (stessa logica di hybrid_recommender.py) ---
        baseline_raw = model._get_baseline_scores(cands)
        cf_raw = model._get_cf_scores(h, cands)
        cbf_raw = model._get_cbf_scores(h, cands)

        baseline_norm = model._norm(baseline_raw)
        cf_norm = model._norm(cf_raw)
        cbf_norm = model._norm(cbf_raw)

        n_pos = int(y_true.sum())
        has_both_classes = True  # garantito dal filtro sopra

        # --- Covariate (nessun leakage: solo history e popolarita' TRAIN) ---
        history_length = len(h)
        n_candidates = len(cands)
        clicked_idx = np.where(y_true == 1)[0]
        single_positive = len(clicked_idx) == 1
        clicked_id = cands[clicked_idx[0]] if single_positive else None
        clicked_train_pop = model.item_popularity.get(clicked_id, 0) if single_positive else None
        clicked_cold = (clicked_train_pop == 0) if single_positive else None

        record = {
            "history_length": history_length,
            "n_candidates": n_candidates,
            "n_clicks": n_pos,
            "single_positive": single_positive,
            "clicked_id": clicked_id,
            "clicked_train_pop": clicked_train_pop,
            "clicked_cold": clicked_cold,
        }

        for cfg_name, weights in CONFIGS.items():
            y_score = combine(baseline_norm, cf_norm, cbf_norm, weights)

            auc = roc_auc_score(y_true, y_score) if has_both_classes else np.nan

            order = np.lexsort((np.asarray(cands, dtype=str), -y_score))
            y_sorted = y_true[order]

            mrr = mrr_first_hit(y_sorted)
            ndcg5 = ndcg_at_k(y_true, y_score, k=5)
            ndcg10 = ndcg_at_k(y_true, y_score, k=10)
            hits_at_5 = int(np.sum(y_sorted[:5]))
            p5 = hits_at_5 / 5.0
            r5 = hits_at_5 / n_pos
            hit1 = bool(y_sorted[0] == 1)
            hit5 = bool(hits_at_5 > 0)

            cand_sorted = [cands[i] for i in order]
            nov5 = float(np.mean([model.get_item_novelty(nid) for nid in cand_sorted[:5]]))
            nov10 = float(np.mean([model.get_item_novelty(nid) for nid in cand_sorted[:10]]))

            record[f"{cfg_name}__AUC"] = auc
            record[f"{cfg_name}__MRR"] = mrr
            record[f"{cfg_name}__nDCG5"] = ndcg5
            record[f"{cfg_name}__nDCG10"] = ndcg10
            record[f"{cfg_name}__P5"] = p5
            record[f"{cfg_name}__R5"] = r5
            record[f"{cfg_name}__Novelty5"] = nov5
            record[f"{cfg_name}__Novelty10"] = nov10
            record[f"{cfg_name}__hit1"] = hit1
            record[f"{cfg_name}__hit5"] = hit5

        imp_rows.append(record)

        # --- Correlazione/overlap dei ranking CF vs CBF (Fase 10) ---
        if len(cands) >= 2 and np.std(cf_norm) > 0 and np.std(cbf_norm) > 0:
            rho, _ = spearmanr(cf_norm, cbf_norm)
        else:
            rho = np.nan
        k_top = min(5, len(cands))
        top_cf = set(np.argsort(-cf_norm)[:k_top].tolist())
        top_cbf = set(np.argsort(-cbf_norm)[:k_top].tolist())
        inter = len(top_cf & top_cbf)
        union = len(top_cf | top_cbf)
        jaccard = inter / union if union > 0 else np.nan
        overlap_frac = inter / k_top if k_top > 0 else np.nan
        corr_rows.append({
            "history_length": history_length,
            "spearman_cf_cbf": rho,
            "top5_overlap_frac": overlap_frac,
            "top5_jaccard": jaccard,
        })

        n_processed += 1
        if n_processed % 5000 == 0:
            print(f"  processate {n_processed} impression valide...", flush=True)

    print(f"\nRighe DEV totali nel range analizzato: {n_total}")
    print(f"Scartate (history o candidati mancanti): {n_skipped_no_hist_or_cand}")
    print(f"Scartate (una sola classe presente, AUC non definita): {n_skipped_single_class}")
    print(f"Impression valide analizzate: {n_processed}")

    # --- Scrive impression_level_results.csv ---
    if imp_rows:
        fieldnames = list(imp_rows[0].keys())
        with open(OUT_DIR / "impression_level_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in imp_rows:
                w.writerow(r)
        print(f"Scritto impression_level_results.csv ({len(imp_rows)} righe, {len(fieldnames)} colonne)")

    if corr_rows:
        with open(OUT_DIR / "ranking_correlation.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
            w.writeheader()
            for r in corr_rows:
                w.writerow(r)
        print(f"Scritto ranking_correlation.csv ({len(corr_rows)} righe)")


if __name__ == "__main__":
    main()
