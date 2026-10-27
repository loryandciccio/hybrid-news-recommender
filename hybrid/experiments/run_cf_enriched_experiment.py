#!/usr/bin/env python3
"""
Confronto sperimentale CF_original (history-only, replica esatta della logica
di hybrid_recommender.py) vs CF_enriched (history + click positivi delle
impression TRAIN disponibili temporalmente prima della valutazione) vs CBF
puro, sullo split temporale interno CONGELATO in temporal_split_definition.py.

NON modifica hybrid_recommender.py. NON sostituisce alcun modello di
produzione. E' uno script sperimentale a se stante in hybrid/experiments/.

L'algoritmo di similarita' (ItemKNN con shrinkage Bayesiano, k_neighbors=200,
shrinkage=50, cf_decay=0.8) e' IDENTICO a quello di hybrid_recommender.py:
l'unica variabile e' la fonte delle interazioni usate per costruire il grafo
di co-click e la popolarita' (history-only vs history+click positivi
disponibili nel periodo di FIT).

Uso (dalla root del progetto):
    python3 hybrid/experiments/run_cf_enriched_experiment.py --stage validation
    python3 hybrid/experiments/run_cf_enriched_experiment.py --stage internal_test

Output:
    hybrid/experiments/cf_enriched_validation_results.json
    hybrid/experiments/cf_enriched_internal_test_results.json (solo dopo freeze)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "hybrid"))

from temporal_split_definition import FIT_DATES, VALIDATION_DATES, INTERNAL_TEST_DATES  # noqa: E402
from hybrid_recommender import ndcg_at_k, cosine_similarity_1d  # noqa: E402

BEHAV_COLS = ["imp_id", "user_id", "time", "history", "impressions"]
NEWS_COLS = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]

K_NEIGHBORS = 200
SHRINKAGE = 50
CF_DECAY = 0.8
CBF_DECAY = 0.95
CBF_TEXT_WEIGHT = 0.7
CBF_TOPIC_WEIGHT = 0.3


def load_train_behaviors() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_train/behaviors.tsv", sep="\t", header=None, names=BEHAV_COLS)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["date_str"] = df["time"].dt.strftime("%Y-%m-%d")
    return df


def parse_impressions(impr) -> Tuple[List[str], np.ndarray]:
    if pd.isna(impr):
        return [], np.array([], dtype=int)
    cands, labels = [], []
    for tok in str(impr).split():
        nid, lab = tok.rsplit("-", 1)
        cands.append(nid)
        labels.append(int(lab))
    return cands, np.asarray(labels)


def parse_history(hist) -> List[str]:
    if pd.isna(hist):
        return []
    return str(hist).split()


def build_itemknn(news_ids_universe: set, local_sets_source: List[List[str]]):
    """Replica esatta di _build_itemknn_matrix di hybrid_recommender.py.

    `local_sets_source`: lista di liste di news_id (una per "sessione" usata
    per contare co-occorrenze). Per CF_original e' la sola `history` di ogni
    riga; per CF_enriched e' `history + click positivi disponibili`.
    """
    co_clicks = defaultdict(Counter)
    item_counts = Counter()

    for local_raw in local_sets_source:
        local = list(dict.fromkeys([nid for nid in local_raw if nid in news_ids_universe]))
        if len(local) < 2:
            if len(local) == 1:
                item_counts[local[0]] += 1
            continue
        for nid in local:
            item_counts[nid] += 1
        for i, nid1 in enumerate(local):
            for nid2 in local[i + 1:]:
                co_clicks[nid1][nid2] += 1
                co_clicks[nid2][nid1] += 1

    sorted_news_ids = sorted(news_ids_universe)
    news_id_to_news_idx = {nid: i for i, nid in enumerate(sorted_news_ids)}
    n_items = len(sorted_news_ids)

    data, rows, cols = [], [], []
    for i, nid1 in enumerate(sorted_news_ids):
        neighbors = co_clicks.get(nid1, Counter())
        top_neighbors = neighbors.most_common(K_NEIGHBORS)
        c1 = item_counts.get(nid1, 1)
        for nid2, count in top_neighbors:
            j = news_id_to_news_idx[nid2]
            c2 = item_counts.get(nid2, 1)
            shrunk_sim = (count + SHRINKAGE) / (math.sqrt(c1 * c2) + SHRINKAGE)
            if shrunk_sim > 0:
                data.append(shrunk_sim)
                rows.append(i)
                cols.append(j)

    sim = sparse.csr_matrix((data, (rows, cols)), shape=(n_items, n_items))
    return sim, news_id_to_news_idx, item_counts


def cf_scores(history, candidates, sim, news_id_to_news_idx):
    n = len(candidates)
    scores = np.zeros(n)
    if sim is None:
        return scores
    w_hist = np.array([CF_DECAY ** i for i in range(len(history))][::-1])
    user_vec = np.zeros(sim.shape[0])
    for h_item, w in zip(history, w_hist):
        idx = news_id_to_news_idx.get(h_item)
        if idx is not None:
            user_vec += sim[idx].toarray().flatten() * w
    if user_vec.max() > 0:
        user_vec = user_vec / user_vec.max()
    for i, nid in enumerate(candidates):
        idx = news_id_to_news_idx.get(nid)
        if idx is not None:
            scores[i] = user_vec[idx]
    return scores


def norm(s: np.ndarray) -> np.ndarray:
    m, M = s.min(), s.max()
    return (s - m) / (M - m) if M > m else s


def baseline_scores(candidates, item_popularity, pop_total):
    return np.array([item_popularity.get(nid, 0) / pop_total if pop_total > 0 else 0.0 for nid in candidates])


def cbf_scores(history, candidates, embeddings, news_id_to_idx, news_metadata):
    n = len(candidates)
    w_hist = np.array([CBF_DECAY ** i for i in range(len(history))][::-1])
    u_vecs = []
    for h_item, w in zip(history, w_hist):
        if h_item in news_id_to_idx:
            u_vecs.append(embeddings[news_id_to_idx[h_item]] * w)
    u_vec = np.sum(u_vecs, axis=0).reshape(1, -1) if u_vecs else None
    if u_vec is not None:
        nrm = np.linalg.norm(u_vec)
        if nrm > 0:
            u_vec /= nrm
    u_cat_pref = defaultdict(float)
    for h_item, w in zip(history, w_hist):
        cat = news_metadata.get(h_item)
        if cat:
            u_cat_pref[cat] += w
    s_text = np.zeros(n)
    s_topic = np.zeros(n)
    for i, nid in enumerate(candidates):
        if u_vec is not None and nid in news_id_to_idx:
            s_text[i] = cosine_similarity_1d(u_vec.ravel(), embeddings[news_id_to_idx[nid]].ravel())
        cat = news_metadata.get(nid)
        if cat:
            s_topic[i] = u_cat_pref.get(cat, 0.0)
    return CBF_TEXT_WEIGHT * norm(s_text) + CBF_TOPIC_WEIGHT * norm(s_topic)


def build_item_popularity(local_sets_source: List[List[str]]) -> Counter:
    c = Counter()
    for local in local_sets_source:
        c.update(local)
    return c


def rank_metrics(y_true, y_score, cands):
    order = np.lexsort((cands, -y_score))
    y_sorted = y_true[order]
    hits = np.where(y_sorted == 1)[0]
    mrr = 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0
    p5 = float(np.sum(y_sorted[:5]) / 5.0)
    r5 = float(np.sum(y_sorted[:5]) / np.sum(y_true))
    return mrr, p5, r5, [cands[i] for i in order]


def evaluate_config(rows_df, score_fn, news_id_to_news_idx_cf, item_pop_for_novelty, pop_total_for_novelty, label):
    metrics = defaultdict(list)
    n_warm_candidates, n_total_candidates = 0, 0
    for row in rows_df.itertuples(index=False):
        h = parse_history(row.history)
        cands, y_true = parse_impressions(row.impressions)
        if not h or not cands or y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        y_score = score_fn(h, cands)
        try:
            metrics["AUC"].append(roc_auc_score(y_true, y_score))
        except Exception:
            pass
        metrics["nDCG@5"].append(ndcg_at_k(y_true, y_score, 5))
        metrics["nDCG@10"].append(ndcg_at_k(y_true, y_score, 10))
        mrr, p5, r5, ranked = rank_metrics(y_true, y_score, cands)
        metrics["MRR"].append(mrr)
        metrics["P@5"].append(p5)
        metrics["R@5"].append(r5)

        def nov(nid):
            c = item_pop_for_novelty.get(nid, 1)
            p = c / pop_total_for_novelty
            return -math.log2(p) if p > 0 else 0.0

        metrics["Novelty@5"].append(float(np.mean([nov(nid) for nid in ranked[:5]])))

        if news_id_to_news_idx_cf is not None:
            warm = [nid for nid in cands if nid in news_id_to_news_idx_cf]
            n_warm_candidates += len(warm)
            n_total_candidates += len(cands)

    out = {k: float(np.mean(v)) for k, v in metrics.items() if len(v) > 0}
    out["n_impressions_evaluated"] = int(len(metrics["AUC"]))
    if n_total_candidates > 0:
        out["pct_candidates_warm_in_cf"] = round(100 * n_warm_candidates / n_total_candidates, 2)
    print(f"[{label}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validation", "internal_test"], required=True)
    args = ap.parse_args()

    behav = load_train_behaviors()

    fit_mask = behav["date_str"].isin(FIT_DATES)
    val_mask = behav["date_str"].isin(VALIDATION_DATES)
    test_mask = behav["date_str"].isin(INTERNAL_TEST_DATES)

    if args.stage == "validation":
        fit_df = behav[fit_mask]
        eval_df = behav[val_mask]
        stage_label = "VALIDATION (fit=Nov9-11, eval=Nov12-13)"
    else:
        fit_df = behav[fit_mask | val_mask]
        eval_df = behav[test_mask]
        stage_label = "INTERNAL_TEST (fit=Nov9-13, eval=Nov14, frozen config)"

    print("=" * 80)
    print(f"STAGE: {stage_label}")
    print(f"Righe FIT: {len(fit_df)}   Righe EVAL: {len(eval_df)}")
    print("=" * 80)

    # Universo di articoli: metadata news del DEV (stesso vocabolario usato da hybrid_recommender.py)
    news_meta_df = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_dev/news.tsv", sep="\t", header=None, names=NEWS_COLS)
    news_id_to_idx = {nid: i for i, nid in enumerate(news_meta_df["News_ID"])}
    news_metadata = dict(zip(news_meta_df["News_ID"], news_meta_df["Category"]))
    embeddings = np.load(PROJECT_ROOT / "content_based_text/embeddings/news_embeddings.npy")
    news_ids_universe = set(news_id_to_idx.keys())

    # --- Fonti di interazione, ristrette a FIT (nessun dato da EVAL entra nel fitting) ---
    history_only_sets = [parse_history(h) for h in fit_df["history"]]

    enriched_sets = []
    for row in fit_df.itertuples(index=False):
        h = parse_history(row.history)
        cands, y_true = parse_impressions(row.impressions)
        clicked = [cands[i] for i in np.where(y_true == 1)[0]]
        enriched_sets.append(h + clicked)

    print("\n[CF_original] Costruzione grafo ItemKNN (solo history, FIT)...")
    sim_orig, idx_orig, item_counts_orig = build_itemknn(news_ids_universe, history_only_sets)
    print(f"  Item con rappresentazione CF (CF_original): {len(item_counts_orig)}")

    print("[CF_enriched] Costruzione grafo ItemKNN (history + click positivi FIT)...")
    sim_enr, idx_enr, item_counts_enr = build_itemknn(news_ids_universe, enriched_sets)
    print(f"  Item con rappresentazione CF (CF_enriched): {len(item_counts_enr)}")

    pop_orig = build_item_popularity(history_only_sets)
    pop_orig_total = float(sum(pop_orig.values())) or 1.0
    pop_enr = build_item_popularity(enriched_sets)
    pop_enr_total = float(sum(pop_enr.values())) or 1.0

    configs = {
        "cf_original": lambda h, c: cf_scores(h, c, sim_orig, idx_orig),
        "cf_enriched": lambda h, c: cf_scores(h, c, sim_enr, idx_enr),
        "cbf_pure": lambda h, c: cbf_scores(h, c, embeddings, news_id_to_idx, news_metadata),
        "baseline_original": lambda h, c: baseline_scores(c, pop_orig, pop_orig_total),
        "baseline_enriched": lambda h, c: baseline_scores(c, pop_enr, pop_enr_total),
    }

    results = {}
    for name, fn in configs.items():
        idx_map = idx_orig if name == "cf_original" else (idx_enr if name == "cf_enriched" else None)
        results[name] = evaluate_config(eval_df, fn, idx_map, pop_orig, pop_orig_total, name)

    results["_meta"] = {
        "stage": args.stage,
        "fit_rows": int(len(fit_df)),
        "eval_rows": int(len(eval_df)),
        "cf_original_item_coverage": int(len(item_counts_orig)),
        "cf_enriched_item_coverage": int(len(item_counts_enr)),
        "coverage_gain_pct": round(100 * (len(item_counts_enr) - len(item_counts_orig)) / len(item_counts_orig), 2),
    }

    out_path = OUT_DIR / f"cf_enriched_{args.stage}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScritto {out_path}")


if __name__ == "__main__":
    main()
