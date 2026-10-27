#!/usr/bin/env python3
"""
Bootstrap paired, segment analysis ed error analysis (contingenza top-5)
per il confronto NRMS vs CBF, a partire dagli score grezzi salvati da
`run_nrms_experiment.py --stage compare_cbf` (`raw_scores_<split>.npz`).

NON modifica hybrid_recommender.py.

Uso (dalla root del progetto):
    python3 hybrid/experiments/nrms/analyze_comparison.py --split validation
    python3 hybrid/experiments/nrms/analyze_comparison.py --split internal_test
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from config import RESULTS_DIR
from metrics import ndcg_at_k, rank_metrics


def load_raw(split):
    data = np.load(RESULTS_DIR / f"raw_scores_{split}.npz", allow_pickle=True)
    return (list(data["nrms_scores"]), list(data["cbf_scores"]),
            list(data["y_true"]), list(data["cands"]), list(data["history_len"]))


def per_impression_metric(y_true_list, score_list, cands_list, metric):
    vals = []
    from sklearn.metrics import roc_auc_score
    for y_true, y_score, cands in zip(y_true_list, score_list, cands_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            vals.append(np.nan)
            continue
        if metric == "AUC":
            try:
                vals.append(roc_auc_score(y_true, y_score))
            except Exception:
                vals.append(np.nan)
        elif metric == "MRR":
            mrr, _, _, _ = rank_metrics(y_true, y_score, cands)
            vals.append(mrr)
        elif metric == "nDCG@5":
            vals.append(ndcg_at_k(y_true, y_score, 5))
        elif metric == "nDCG@10":
            vals.append(ndcg_at_k(y_true, y_score, 10))
    return np.array(vals)


def bootstrap_ci(y_true_list, nrms_scores, cbf_scores, cands_list, metric, n_boot=2000, seed=42):
    a = per_impression_metric(y_true_list, nrms_scores, cands_list, metric)
    b = per_impression_metric(y_true_list, cbf_scores, cands_list, metric)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
    return {
        "metric": metric, "n": int(n), "n_boot": n_boot, "seed": seed,
        "mean_diff_NRMS_minus_CBF": float(diff.mean()),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "ci_excludes_zero": bool(np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0),
    }


def rank_of_clicked(y_score, cands, clicked_id):
    order = np.lexsort((np.asarray(cands, dtype=str), -np.asarray(y_score)))
    ranked = [cands[i] for i in order]
    return ranked.index(clicked_id) + 1


def top5_contingency_and_rank(y_true_list, nrms_scores, cbf_scores, cands_list):
    both, only_nrms, only_cbf, neither = 0, 0, 0, 0
    nrms_better, cbf_better, tie = 0, 0, 0
    n_single_pos = 0
    for y_true, s_nrms, s_cbf, cands in zip(y_true_list, nrms_scores, cbf_scores, cands_list):
        y_true = np.asarray(y_true)
        if y_true.sum() != 1:
            continue
        n_single_pos += 1
        clicked_id = cands[int(np.argmax(y_true))]
        r_nrms = rank_of_clicked(s_nrms, cands, clicked_id)
        r_cbf = rank_of_clicked(s_cbf, cands, clicked_id)
        nrms_top5 = r_nrms <= 5
        cbf_top5 = r_cbf <= 5
        if nrms_top5 and cbf_top5:
            both += 1
        elif nrms_top5:
            only_nrms += 1
        elif cbf_top5:
            only_cbf += 1
        else:
            neither += 1
        if r_nrms < r_cbf:
            nrms_better += 1
        elif r_cbf < r_nrms:
            cbf_better += 1
        else:
            tie += 1
    n = n_single_pos
    return {
        "n_single_positive": n,
        "top5_solo_NRMS_pct": round(100 * only_nrms / n, 2),
        "top5_solo_CBF_pct": round(100 * only_cbf / n, 2),
        "top5_entrambi_pct": round(100 * both / n, 2),
        "top5_nessuno_pct": round(100 * neither / n, 2),
        "pct_impression_NRMS_rank_migliore": round(100 * nrms_better / n, 2),
        "pct_impression_CBF_rank_migliore": round(100 * cbf_better / n, 2),
        "pct_tie": round(100 * tie / n, 2),
    }


def segment_analysis(y_true_list, nrms_scores, cbf_scores, cands_list, history_len):
    history_len = np.array(history_len)
    # NOTA: quasi la meta' delle impression ha history_length == MAX_HISTORY_LEN
    # (20, il tetto di troncamento), il che rende degenere una segmentazione
    # per tercili (il 66-esimo percentile coincide col tetto). Si usano quindi
    # soglie fisse e documentate, non stimate sui dati.
    segments = {
        "history_corta_(<=5)": history_len <= 5,
        "history_media_(6-19)": (history_len > 5) & (history_len < 20),
        "history_lunga_(>=20_troncata)": history_len >= 20,
    }

    def auc_for_mask(mask, scores):
        from sklearn.metrics import roc_auc_score
        vals = []
        for i in np.where(mask)[0]:
            yt, ys = np.asarray(y_true_list[i]), np.asarray(scores[i])
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            try:
                vals.append(roc_auc_score(yt, ys))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else None, len(vals)

    out = {}
    for name, mask in segments.items():
        nrms_auc, n = auc_for_mask(mask, nrms_scores)
        cbf_auc, _ = auc_for_mask(mask, cbf_scores)
        out[name] = {"n": n, "NRMS_AUC": nrms_auc, "CBF_AUC": cbf_auc,
                     "delta": (nrms_auc - cbf_auc) if (nrms_auc is not None and cbf_auc is not None) else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "internal_test"], required=True)
    args = ap.parse_args()

    nrms_scores, cbf_scores, y_true_list, cands_list, history_len = load_raw(args.split)
    print(f"[{args.split}] {len(nrms_scores)} impression comuni caricate")

    bootstrap_results = {}
    for metric in ["AUC", "MRR", "nDCG@5"]:
        ci = bootstrap_ci(y_true_list, nrms_scores, cbf_scores, cands_list, metric)
        bootstrap_results[metric] = ci
        print(f"NRMS-CBF | {metric}: mean_diff={ci['mean_diff_NRMS_minus_CBF']:.5f} IC95%=[{ci['ci95_low']:.5f},{ci['ci95_high']:.5f}] escludes_zero={ci['ci_excludes_zero']}")

    seg = segment_analysis(y_true_list, nrms_scores, cbf_scores, cands_list, history_len)
    print("\nSegment analysis (history length):")
    print(json.dumps(seg, indent=2))

    err = top5_contingency_and_rank(y_true_list, nrms_scores, cbf_scores, cands_list)
    print("\nError analysis (top-5 contingency + rank):")
    print(json.dumps(err, indent=2))

    out = {"split": args.split, "bootstrap": bootstrap_results, "segment_analysis": seg, "error_analysis": err}
    with open(RESULTS_DIR / f"analysis_{args.split}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nScritto analysis_{args.split}.json")


if __name__ == "__main__":
    main()
