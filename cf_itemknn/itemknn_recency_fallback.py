#!/usr/bin/env python3
"""
ItemKNN (co-click) per MIND-small con scoring pesato per recency + FALLBACK DI POPOLARITA'.

E' un "hybrid leggero": CF ItemKNN + prior di popolarità globale.

Proprietà principali:
- La similarità item-item è addestrata SOLO sul TRAIN (nessun leakage dal DEV).
- Tie-breaking deterministico per le metriche di rank: (-score, news_id).
- Metriche standard MIND: AUC, MRR, nDCG@5/10, P@5, R@5.
- Metrica beyond-accuracy: Novelty@5/10, calcolata dalla popolarità di click sul TRAIN.
- Scoring pesato per recency: gli articoli più recenti della cronologia contribuiscono di più.
- Score finale: alpha * CF + (1-alpha) * popolarità normalizzata.

Esempio d'uso con i parametri di default (eseguire dalla root del progetto):
  python3 cf_itemknn/itemknn_recency_fallback.py \
    --train-behaviors dataset/MINDsmall_train/behaviors.tsv \
    --dev-behaviors   dataset/MINDsmall_dev/behaviors.tsv \
    --window 30 --k 200 --shrink 50 --decay 0.8 --alpha 0.7

Esempio d'uso con la configurazione a più alta shrinkage/finestra esplorata durante
il tuning (window=80, k=800, shrink=10, decay=0.5, alpha=0.5):
  python3 cf_itemknn/itemknn_recency_fallback.py \
    --train-behaviors dataset/MINDsmall_train/behaviors.tsv \
    --dev-behaviors   dataset/MINDsmall_dev/behaviors.tsv \
    --window 80 --k 800 --shrink 10 --decay 0.5 --max-hist 20 --alpha 0.5
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# --------------------------
# Funzioni di supporto per l'nDCG (implementazione locale)
# --------------------------

def dcg_at_k(y_true, y_score, k):
    """Calcola il DCG@k usando gli score predetti per ordinare gli item."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="mergesort")[:k]
    gains = y_true[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))

def ideal_dcg_at_k(y_true, k):
    """Calcola l'IDCG@k ordinando gli item in base alla label vera (ranking ideale)."""
    y_true = np.asarray(y_true)
    ideal = np.sort(y_true)[::-1][:k]
    discounts = np.log2(np.arange(2, len(ideal) + 2))
    return float(np.sum(ideal / discounts))

def ndcg_at_k(y_true, y_score, k):
    """Calcola nDCG@k = DCG@k / IDCG@k."""
    idcg = ideal_dcg_at_k(y_true, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(y_true, y_score, k) / idcg


def parse_impressions(impr: str) -> Tuple[List[str], np.ndarray]:
    parts = str(impr).split()
    cands, labels = [], []
    for p in parts:
        try:
            nid, lab = p.rsplit("-", 1)
        except ValueError:
            continue
        cands.append(nid)
        labels.append(1 if lab == "1" else 0)
    return cands, np.asarray(labels, dtype=np.int32)

def parse_history(hist: str) -> List[str]:
    if hist is None or (isinstance(hist, float) and np.isnan(hist)):
        return []
    s = str(hist).strip()
    if not s:
        return []
    return s.split()


@dataclass
class ItemKNNModel:
    neighbors: Dict[str, List[Tuple[str, float]]]
    item_clicks: Counter
    pop_total: float
    pop_max: float


def build_popularity(train_behav: pd.DataFrame) -> Counter:
    clicked = (
        train_behav["impressions"]
        .dropna()
        .astype(str)
        .str.split()
        .explode()
    )
    clicked = clicked[clicked.str.endswith("-1")].str.rsplit("-", n=1).str[0]
    return Counter(clicked)


def build_coclick_counts(train_behav: pd.DataFrame, window: int):
    co_counts: Dict[str, Counter] = defaultdict(Counter)
    item_counts: Counter = Counter()

    for _, row in train_behav.iterrows():
        hist = parse_history(row.get("history", ""))
        if window > 0 and len(hist) > window:
            hist = hist[-window:]

        cands, labels = parse_impressions(row.get("impressions", ""))
        clicked = [cands[i] for i in np.where(labels == 1)[0]]

        local = list(dict.fromkeys(hist + clicked))
        m = len(local)
        if m < 2:
            if m == 1:
                item_counts[local[0]] += 1
            continue

        for nid in local:
            item_counts[nid] += 1

        for a in range(m):
            ia = local[a]
            ca = co_counts[ia]
            for b in range(a + 1, m):
                ib = local[b]
                ca[ib] += 1
                co_counts[ib][ia] += 1

    return co_counts, item_counts


def compute_neighbors(co_counts: Dict[str, Counter], item_counts: Counter, k: int, shrink: float):
    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    for i, ctr in co_counts.items():
        ci = float(item_counts.get(i, 0))
        if ci <= 0:
            continue
        sims = []
        for j, cij in ctr.items():
            cj = float(item_counts.get(j, 0))
            if cj <= 0:
                continue
            sim = float(cij) / (math.sqrt(ci * cj) + float(shrink))
            if sim > 0:
                sims.append((j, sim))
        if not sims:
            continue
        sims.sort(key=lambda x: (-x[1], x[0]))
        if k > 0:
            sims = sims[:k]
        neighbors[i] = sims
    return neighbors


def train_itemknn(train_behav: pd.DataFrame, window: int, k: int, shrink: float) -> ItemKNNModel:
    pop = build_popularity(train_behav)
    pop_total = float(sum(pop.values())) if len(pop) else 1.0
    pop_max = float(max(pop.values())) if len(pop) else 1.0

    co_counts, item_counts = build_coclick_counts(train_behav, window=window)
    neigh = compute_neighbors(co_counts, item_counts, k=k, shrink=shrink)

    return ItemKNNModel(neighbors=neigh, item_clicks=pop, pop_total=pop_total, pop_max=pop_max)


def score_candidates_recency(model: ItemKNNModel, history: List[str], candidates: List[str], decay: float, max_hist: int):
    if not candidates:
        return np.asarray([], dtype=np.float64)

    if max_hist > 0 and len(history) > max_hist:
        history = history[-max_hist:]

    cand_index = {nid: idx for idx, nid in enumerate(candidates)}
    scores = np.zeros(len(candidates), dtype=np.float64)

    t = 0
    for h in reversed(history):
        w = decay ** t
        t += 1
        neigh = model.neighbors.get(h)
        if not neigh:
            continue
        for j, sim in neigh:
            idx = cand_index.get(j)
            if idx is not None:
                scores[idx] += w * sim
    return scores


def popularity_score(model: ItemKNNModel, news_id: str) -> float:
    # Normalizza rispetto al massimo per restare nella scala [0,1] (semplice e stabile)
    return float(model.item_clicks.get(news_id, 0.0)) / float(model.pop_max)


def blend_scores(cf_scores: np.ndarray, candidates: List[str], model: ItemKNNModel, alpha: float) -> np.ndarray:
    if len(candidates) == 0:
        return cf_scores
    pop = np.asarray([popularity_score(model, nid) for nid in candidates], dtype=np.float64)
    return alpha * cf_scores + (1.0 - alpha) * pop


def deterministic_order(y_score: np.ndarray, cand_ids: List[str]) -> np.ndarray:
    y_score = np.asarray(y_score, dtype=np.float64)
    cand_ids = np.asarray(cand_ids, dtype=str)
    return np.lexsort((cand_ids, -y_score))


def mrr_first_hit(y_true: np.ndarray, y_score: np.ndarray, cand_ids: List[str]) -> float:
    order = deterministic_order(y_score, cand_ids)
    y_sorted = y_true[order]
    hits = np.where(y_sorted == 1)[0]
    return 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, cand_ids: List[str], k: int) -> float:
    order = deterministic_order(y_score, cand_ids)[:k]
    return float(np.mean(y_true[order])) if len(order) else 0.0


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, cand_ids: List[str], k: int) -> float:
    denom = int(np.sum(y_true))
    if denom <= 0:
        return 0.0
    order = deterministic_order(y_score, cand_ids)[:k]
    return float(np.sum(y_true[order]) / denom) if len(order) else 0.0


def item_novelty(model: ItemKNNModel, news_id: str) -> float:
    c = model.item_clicks.get(news_id, 1)
    p = float(c) / float(model.pop_total)
    return float(-np.log2(p))


def novelty_at_k(model: ItemKNNModel, y_score: np.ndarray, cand_ids: List[str], k: int) -> float:
    order = deterministic_order(y_score, cand_ids)[:k]
    if len(order) == 0:
        return 0.0
    top_ids = [cand_ids[i] for i in order]
    return float(np.mean([item_novelty(model, nid) for nid in top_ids]))


def load_behaviors(path: Path) -> pd.DataFrame:
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    return pd.read_csv(path, sep="\t", header=None, names=cols)


def evaluate(model: ItemKNNModel, dev_behav: pd.DataFrame, decay: float, max_hist: int, alpha: float, ks=(5, 10), limit: int | None = None):
    metrics = {
        "auc": [],
        "mrr": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "p@5": [],
        "r@5": [],
        "novelty@5": [],
        "novelty@10": [],
    }

    n_rows = len(dev_behav) if limit is None else min(len(dev_behav), limit)

    for idx in range(n_rows):
        row = dev_behav.iloc[idx]
        history = parse_history(row.get("history", ""))
        candidates, y_true = parse_impressions(row.get("impressions", ""))

        if len(candidates) == 0:
            continue
        if int(y_true.sum()) == 0:
            continue

        cf = score_candidates_recency(model, history, candidates, decay=decay, max_hist=max_hist)
        y_score = blend_scores(cf, candidates, model, alpha=alpha)

        if len(np.unique(y_true)) == 2:
            metrics["auc"].append(roc_auc_score(y_true, y_score))

        metrics["mrr"].append(mrr_first_hit(y_true, y_score, candidates))
        metrics["ndcg@5"].append(ndcg_at_k(y_true, y_score, k=ks[0]))
        metrics["ndcg@10"].append(ndcg_at_k(y_true, y_score, k=ks[1]))
        metrics["p@5"].append(precision_at_k(y_true, y_score, candidates, k=ks[0]))
        metrics["r@5"].append(recall_at_k(y_true, y_score, candidates, k=ks[0]))
        metrics["novelty@5"].append(novelty_at_k(model, y_score, candidates, k=ks[0]))
        metrics["novelty@10"].append(novelty_at_k(model, y_score, candidates, k=ks[1]))

    return {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-behaviors", type=Path, required=True)
    ap.add_argument("--dev-behaviors", type=Path, required=True)

    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--shrink", type=float, default=50.0)

    ap.add_argument("--decay", type=float, default=0.8)
    ap.add_argument("--max-hist", type=int, default=50)

    ap.add_argument("--alpha", type=float, default=0.7, help="Peso del blending per il CF; (1-alpha) per la popolarità")
    ap.add_argument("--limit-dev", type=int, default=None)

    args = ap.parse_args()

    train_behav = load_behaviors(args.train_behaviors)
    dev_behav = load_behaviors(args.dev_behaviors)

    print("Training ItemKNN (scoring pesato per recency) + fallback di popolarità...")
    model = train_itemknn(train_behav, window=args.window, k=args.k, shrink=args.shrink)

    print("Valutazione sul DEV...")
    results = evaluate(model, dev_behav, decay=args.decay, max_hist=args.max_hist, alpha=args.alpha, limit=args.limit_dev)

    print("\nRiepilogo (DEV):")
    print(
        f"AUC={results['auc']:.4f}, MRR={results['mrr']:.4f}, "
        f"nDCG@5={results['ndcg@5']:.4f}, nDCG@10={results['ndcg@10']:.4f}, "
        f"P@5={results['p@5']:.4f}, R@5={results['r@5']:.4f}, "
        f"Novelty@5={results['novelty@5']:.4f}, Novelty@10={results['novelty@10']:.4f}"
    )


if __name__ == "__main__":
    main()
