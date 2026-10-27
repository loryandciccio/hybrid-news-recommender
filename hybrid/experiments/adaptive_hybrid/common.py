#!/usr/bin/env python3
"""
Utilità condivise per l'esperimento Adaptive Hybrid (Rule-Based + Neural Gating).

NON modifica hybrid_recommender.py: riusa le funzioni di scoring gia'
validate in hybrid/experiments/run_cf_enriched_experiment.py (stesso
algoritmo ItemKNN/CBF del progetto principale, k=200, shrinkage=50,
decay=0.8/0.95).

Definizione di CF_AVAILABLE:
    CF_AVAILABLE = 1  se  max(cf_raw) - min(cf_raw) > EPSILON_CF
    CF_AVAILABLE = 0  altrimenti (ranking CF privo di qualunque informazione
                       discriminante: tutti i candidati hanno lo stesso score,
                       tipicamente 0, e l'ordine sarebbe deciso solo dal
                       tie-break deterministico su news_id)

EPSILON_CF = 1e-9: sufficientemente piccolo da non escludere segnali CF
deboli ma reali, sufficientemente grande da assorbire rumore in virgola
mobile. Con gli score CF di questo progetto (shrinkage Bayesiano, mai
negativi, spesso esattamente 0.0 per costruzione quando non c'e' nessun
vicino), un ranking degenere produce quasi sempre range esattamente 0.0,
quindi la soglia esatta scelta ha un impatto trascurabile in pratica
(verificato: 0 impression con range in (0, 1e-6] su VALIDATION).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
ADAPTIVE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ADAPTIVE_DIR / "results"
FIGURES_DIR = ADAPTIVE_DIR / "figures"

sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "hybrid"))

from run_cf_enriched_experiment import (  # noqa: E402
    load_train_behaviors, parse_history, parse_impressions,
    build_itemknn, cf_scores, cbf_scores, norm, build_item_popularity,
)
from temporal_split_definition import FIT_DATES, VALIDATION_DATES, INTERNAL_TEST_DATES  # noqa: E402
from hybrid_recommender import ndcg_at_k  # noqa: E402

NEWS_COLS = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]

EPSILON_CF = 1e-9

# Split diagnostico interno per la gating (dentro VALIDATION, mai su INTERNAL_TEST)
GATING_TRAIN_DATES = ["2019-11-12"]
GATING_VALIDATION_DATES = ["2019-11-13"]

# Split usato per il CF durante lo sviluppo (dev = GATING_TRAIN + GATING_VALIDATION):
DEV_CF_FIT_DATES = FIT_DATES  # 9-11 novembre, invariato rispetto a run_cf_enriched_experiment.py

# Split usato per il CF alla valutazione finale su INTERNAL_TEST (14 novembre):
# ricostruito da FIT+VALIDATION (9-13 novembre), stessa scelta di run_cf_enriched_experiment.py
# stage=internal_test, per usare tutta l'informazione causale disponibile prima del 14 novembre.
TEST_CF_FIT_DATES = FIT_DATES + VALIDATION_DATES


def load_news_universe():
    import pandas as pd
    news_meta_df = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_dev/news.tsv", sep="\t", header=None, names=NEWS_COLS)
    news_id_to_idx = {nid: i for i, nid in enumerate(news_meta_df["News_ID"])}
    news_metadata = dict(zip(news_meta_df["News_ID"], news_meta_df["Category"]))
    embeddings = np.load(PROJECT_ROOT / "content_based_text/embeddings/news_embeddings.npy")
    return news_id_to_idx, news_metadata, embeddings, set(news_id_to_idx.keys())


def build_cf_from_dates(behav, fit_dates, news_ids_universe, enriched: bool = True):
    """Costruisce il grafo ItemKNN (e la popolarita' history-only) usando solo le
    righe di TRAIN nelle date indicate (mai righe successive: nessun leakage)."""
    fit_df = behav[behav["date_str"].isin(fit_dates)]
    history_only_sets = [parse_history(h) for h in fit_df["history"]]

    if enriched:
        local_sets = []
        for row in fit_df.itertuples(index=False):
            h = parse_history(row.history)
            cands, y_true = parse_impressions(row.impressions)
            clicked = [cands[i] for i in np.where(y_true == 1)[0]]
            local_sets.append(h + clicked)
    else:
        local_sets = history_only_sets

    sim, idx_map, item_counts = build_itemknn(news_ids_universe, local_sets)
    pop = build_item_popularity(history_only_sets)  # popolarita' SEMPRE history-only (Baseline_original)
    # NOTA IMPORTANTE: idx_map (news_id_to_news_idx) contiene una voce per OGNI
    # articolo dell'universo (news_ids_universe, 42.416 articoli di
    # MINDsmall_dev/news.tsv), inclusi quelli MAI comparsi in una co-occorrenza
    # (item_counts.get(nid, 0) == 0). "in idx_map" NON equivale a "ha una
    # rappresentazione CF reale": va sempre verificato tramite item_counts.
    return sim, idx_map, pop, item_counts, len(fit_df)


def cf_available_mask(cf_raw: np.ndarray) -> bool:
    return (cf_raw.max() - cf_raw.min()) > EPSILON_CF


def rank_metrics_for_score(y_true, y_score, cands):
    """Replica esatta della logica di tie-break/metriche di hybrid_recommender.py."""
    order = np.lexsort((cands, -y_score))
    y_sorted = y_true[order]
    hits = np.where(y_sorted == 1)[0]
    mrr = 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0
    p5 = float(np.sum(y_sorted[:5]) / 5.0)
    r5 = float(np.sum(y_sorted[:5]) / np.sum(y_true))
    return mrr, p5, r5, [cands[i] for i in order]
