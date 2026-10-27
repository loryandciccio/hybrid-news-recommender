#!/usr/bin/env python3
"""
Audit della costruzione del training set del Collaborative Filtering.

Determina, per ciascun modello del progetto (Baseline, ItemKNN standalone,
ALS latente, CF interno all'Hybrid), quali interazioni user-item vengono
usate per costruirlo (history vs click positivi nelle impression vs
negativi vs timestamp), e quantifica quanta informazione positiva
disponibile nel TRAIN viene ignorata dal CF usato dall'Hybrid.

NON modifica alcun algoritmo esistente: legge solo i file TSV grezzi e
replica (in sola lettura) la logica di costruzione già presente nei
moduli del progetto, per contarne gli insiemi risultanti.

Uso (dalla root del progetto):
    python3 hybrid/experiments/run_cf_training_source_audit.py

Output:
    hybrid/experiments/cf_training_source_audit.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

BEHAV_COLS = ["imp_id", "user_id", "time", "history", "impressions"]


def main():
    behav = pd.read_csv(
        PROJECT_ROOT / "dataset/MINDsmall_train/behaviors.tsv",
        sep="\t", header=None, names=BEHAV_COLS,
    )
    behav["time"] = pd.to_datetime(behav["time"], format="%m/%d/%Y %I:%M:%S %p")

    # --- Interazioni da history (usate da: item_popularity dell'Hybrid,
    #     _build_itemknn_matrix dell'Hybrid, create_interaction_matrix.py/ALS) ---
    history_interactions = 0
    history_item_counter = Counter()
    for h in behav["history"].dropna():
        toks = str(h).split()
        history_interactions += len(toks)
        history_item_counter.update(toks)
    history_items = set(history_item_counter.keys())

    # --- Interazioni positive dalle impression (usate da: baseline.ipynb,
    #     cf_itemknn/itemknn_recency_pure.py e itemknn_recency_fallback.py
    #     per popolarita' E per il grafo di co-click) ---
    n_rows_with_impressions = 0
    positive_interactions = 0
    negative_interactions = 0
    clicked_item_counter = Counter()
    all_candidate_counter = Counter()
    for imp in behav["impressions"].dropna():
        n_rows_with_impressions += 1
        for tok in str(imp).split():
            nid, lab = tok.rsplit("-", 1)
            all_candidate_counter[nid] += 1
            if lab == "1":
                positive_interactions += 1
                clicked_item_counter[nid] += 1
            else:
                negative_interactions += 1
    clicked_items = set(clicked_item_counter.keys())
    candidate_items = set(all_candidate_counter.keys())

    # --- Utenti distinti in ciascuna fonte ---
    n_users_total = behav["user_id"].nunique()
    n_rows_with_history = behav["history"].notna().sum()

    # --- Overlap tra history-items e clicked-impression-items ---
    only_history = history_items - clicked_items
    only_clicked = clicked_items - history_items
    both = history_items & clicked_items
    union_items = history_items | clicked_items

    # --- Interazioni "aggiuntive" (positive click su item MAI comparsi in history) ---
    additional_positive_interactions_novel_items = sum(
        clicked_item_counter[i] for i in only_clicked
    )
    # Interazioni positive totali che il CF attuale (history-only) NON vede affatto
    # come segnale di interazione (anche se l'item è già "warm" via history, il
    # CLICK STESSO come evento non entra mai nella costruzione del grafo/popolarita')
    total_ignored_positive_events = positive_interactions

    # --- Coverage CF attuale (item con almeno 1 occorrenza in history, quindi
    #     rappresentati nella matrice ItemKNN dell'Hybrid) vs coverage potenziale
    #     se si aggiungessero anche i click positivi delle impression ---
    coverage_current = len(history_items)
    coverage_potential = len(union_items)

    result = {
        "train_rows_total": int(len(behav)),
        "train_rows_with_history": int(n_rows_with_history),
        "train_rows_with_impressions": int(n_rows_with_impressions),
        "train_users_distinct": int(n_users_total),
        "history": {
            "total_interactions": int(history_interactions),
            "distinct_items": int(len(history_items)),
        },
        "impressions": {
            "positive_interactions": int(positive_interactions),
            "negative_interactions": int(negative_interactions),
            "distinct_clicked_items": int(len(clicked_items)),
            "distinct_candidate_items": int(len(candidate_items)),
        },
        "overlap_history_vs_clicked_items": {
            "both": int(len(both)),
            "only_history_never_clicked_in_impressions": int(len(only_history)),
            "only_clicked_never_in_history": int(len(only_clicked)),
            "union_distinct_items": int(len(union_items)),
            "pct_history_items_also_clicked": round(100 * len(both) / len(history_items), 2),
            "pct_clicked_items_never_in_history": round(100 * len(only_clicked) / len(clicked_items), 2),
        },
        "ignored_by_current_cf": {
            "total_positive_click_events_in_train": int(total_ignored_positive_events),
            "positive_click_events_on_items_absent_from_history": int(additional_positive_interactions_novel_items),
            "distinct_items_absent_from_history_but_clicked": int(len(only_clicked)),
        },
        "coverage": {
            "current_cf_item_coverage_history_only": int(coverage_current),
            "potential_cf_item_coverage_history_plus_clicks": int(coverage_potential),
            "coverage_gain_items": int(coverage_potential - coverage_current),
            "coverage_gain_pct": round(100 * (coverage_potential - coverage_current) / coverage_current, 2),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cf_training_source_audit.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
