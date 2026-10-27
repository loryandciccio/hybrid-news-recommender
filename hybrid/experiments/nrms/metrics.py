#!/usr/bin/env python3
"""Metriche IDENTICHE a quelle di hybrid_recommender.py (stesso tie-break,
stessa formula nDCG). Nessuna reimplementazione divergente."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def dcg_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="mergesort")[:k]
    gains = y_true[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ideal_dcg_at_k(y_true, k):
    y_true = np.asarray(y_true)
    ideal = np.sort(y_true)[::-1][:k]
    discounts = np.log2(np.arange(2, len(ideal) + 2))
    return float(np.sum(ideal / discounts))


def ndcg_at_k(y_true, y_score, k):
    idcg = ideal_dcg_at_k(y_true, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(y_true, y_score, k) / idcg


def rank_metrics(y_true, y_score, cands):
    order = np.lexsort((cands, -y_score))
    y_sorted = y_true[order]
    hits = np.where(y_sorted == 1)[0]
    mrr = 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0
    p5 = float(np.sum(y_sorted[:5]) / 5.0)
    r5 = float(np.sum(y_sorted[:5]) / np.sum(y_true))
    return mrr, p5, r5, [cands[i] for i in order]


def evaluate_scores(y_true_list, y_score_list, cands_list, item_popularity=None, pop_total=None):
    metrics = defaultdict(list)
    for y_true, y_score, cands in zip(y_true_list, y_score_list, cands_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
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
        if item_popularity is not None:
            def nov(nid):
                c = item_popularity.get(nid, 1)
                p = c / pop_total
                return -math.log2(p) if p > 0 else 0.0
            metrics["Novelty@5"].append(float(np.mean([nov(n) for n in ranked[:5]])))
            metrics["Novelty@10"].append(float(np.mean([nov(n) for n in ranked[:10]])))
    out = {k: float(np.mean(v)) for k, v in metrics.items() if len(v) > 0}
    out["n"] = int(len(metrics["AUC"]))
    return out
