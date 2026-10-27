#!/usr/bin/env python3
"""
Funzioni di valutazione condivise da rule_based_hybrid.py, train_gating.py ed
evaluate_adaptive.py. Le metriche e il tie-break sono IDENTICHE a quelle di
hybrid_recommender.py (nessuna modifica all'algoritmo di valutazione).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from common import ndcg_at_k, rank_metrics_for_score

W_CF_STATIC = 0.03
W_CBF_STATIC = 0.97


def evaluate_records(records, score_fn, item_popularity=None, pop_total=None):
    """score_fn(rec) -> np.ndarray di score per i candidati di rec (stesso ordine di rec['cands'])."""
    metrics = defaultdict(list)
    from sklearn.metrics import roc_auc_score

    for rec in records:
        y_true = rec["y_true"].astype(int)
        cands = rec["cands"]
        y_score = score_fn(rec)

        try:
            metrics["AUC"].append(roc_auc_score(y_true, y_score))
        except Exception:
            pass

        metrics["nDCG@5"].append(ndcg_at_k(y_true, y_score, 5))
        metrics["nDCG@10"].append(ndcg_at_k(y_true, y_score, 10))

        mrr, p5, r5, ranked = rank_metrics_for_score(y_true, y_score, cands)
        metrics["MRR"].append(mrr)
        metrics["P@5"].append(p5)
        metrics["R@5"].append(r5)

        if item_popularity is not None:
            def nov(nid):
                c = item_popularity.get(nid, 1)
                p = c / pop_total
                return -math.log2(p) if p > 0 else 0.0
            metrics["Novelty@5"].append(float(np.mean([nov(nid) for nid in ranked[:5]])))
            metrics["Novelty@10"].append(float(np.mean([nov(nid) for nid in ranked[:10]])))

    out = {k: float(np.mean(v)) for k, v in metrics.items() if len(v) > 0}
    out["n"] = int(len(metrics["AUC"]))
    return out


def per_impression_metric(records, score_fn, metric="AUC"):
    """Restituisce un array 1D di valori per-impression della metrica scelta
    (per il bootstrap paired)."""
    from sklearn.metrics import roc_auc_score
    vals = []
    for rec in records:
        y_true = rec["y_true"].astype(int)
        cands = rec["cands"]
        y_score = score_fn(rec)
        if metric == "AUC":
            try:
                vals.append(roc_auc_score(y_true, y_score))
            except Exception:
                vals.append(np.nan)
        elif metric == "MRR":
            mrr, _, _, _ = rank_metrics_for_score(y_true, y_score, cands)
            vals.append(mrr)
        elif metric == "nDCG@5":
            vals.append(ndcg_at_k(y_true, y_score, 5))
        else:
            raise ValueError(metric)
    return np.array(vals, dtype=float)


def paired_bootstrap_ci(records, score_fn_a, score_fn_b, metric="AUC", n_boot=2000, seed=42):
    """CI 95% bootstrap sulla differenza (A - B) della metrica, appaiata per impression."""
    vals_a = per_impression_metric(records, score_fn_a, metric)
    vals_b = per_impression_metric(records, score_fn_b, metric)
    mask = ~np.isnan(vals_a) & ~np.isnan(vals_b)
    vals_a, vals_b = vals_a[mask], vals_b[mask]
    diff = vals_a - vals_b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
    return {
        "metric": metric,
        "n": int(n),
        "mean_diff": float(diff.mean()),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "ci_excludes_zero": bool(np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0),
    }
