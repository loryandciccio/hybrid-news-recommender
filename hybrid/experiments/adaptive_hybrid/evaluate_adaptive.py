#!/usr/bin/env python3
"""
Confronto finale, sulla stessa identica popolazione di impression: CBF puro,
Hybrid statico (0/0.03/0.97), Rule-Based Adaptive (regola congelata da
rule_based_hybrid.py) e Neural Gating (modello congelato da train_gating.py).

Due stage:
  --stage validation     -> GATING_VALIDATION (13 nov), usato per scegliere/
                             confermare la configurazione (gia' fatto per
                             rule-based e gating separatamente; qui si
                             confrontano exhaustive tutti e 4 gli score sulla
                             stessa popolazione).
  --stage internal_test  -> INTERNAL_TEST (14 nov), UNA SOLA VALUTAZIONE,
                             a configurazione congelata. Richiede
                             `build_features.py --stage test` gia' eseguito.

Per ciascuno stage riporta le metriche su: tutte le impression, solo
CF_AVAILABLE=1, solo CF_AVAILABLE=0. Calcola inoltre bootstrap paired
(Gating-CBF, Rule-Based-CBF) su AUC/MRR/nDCG@5, e l'analisi dei pesi
appresi dalla gating (solo per lo stage in cui e' disponibile).

NON modifica hybrid_recommender.py.

Uso (dalla root del progetto):
    python3 hybrid/experiments/adaptive_hybrid/evaluate_adaptive.py --stage validation
    python3 hybrid/experiments/adaptive_hybrid/evaluate_adaptive.py --stage internal_test
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from common import RESULTS_DIR
from gating_network import GatingNet
from evaluate_common import evaluate_records, paired_bootstrap_ci, W_CF_STATIC, W_CBF_STATIC
from train_gating import FEATURE_COLS, featurize, compute_gating_scores


def load_rule_config():
    with open(RESULTS_DIR / "rule_based_selection.json") as f:
        return json.load(f)


def load_gating_model():
    with open(RESULTS_DIR / "gating_best_arch.json") as f:
        arch = json.load(f)
    with open(RESULTS_DIR / "gating_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = GatingNet(n_features=len(FEATURE_COLS), hidden=arch["hidden"])
    model.load_state_dict(torch.load(RESULTS_DIR / "gating_model_best.pt"))
    model.eval()
    return model, scaler, arch


def score_cbf_pure(rec):
    return rec["cbf_norm"]


def score_hybrid_static(rec):
    return W_CF_STATIC * rec["cf_norm"] + W_CBF_STATIC * rec["cbf_norm"]


def make_rule_based_score_fn(rule_name, cbf_std_median, cbf_max_median):
    def score_fn(rec):
        feats = rec["features"]
        if not feats["CF_AVAILABLE"]:
            return rec["cbf_norm"]
        if rule_name == "R3_max_threshold_0.10_0.03":
            w_cf = 0.10 if feats["cbf_score_max"] < cbf_max_median else 0.03
        elif rule_name == "R2_std_threshold_0.10_0.03":
            w_cf = 0.10 if feats["cbf_score_std"] < cbf_std_median else 0.03
        elif rule_name == "R4_std_threshold_strong_0.20_0.03":
            w_cf = 0.20 if feats["cbf_score_std"] < cbf_std_median else 0.03
        elif rule_name == "R5_combined_std_and_gap":
            w_cf = 0.10 if (feats["cbf_score_std"] < cbf_std_median and feats["cf_gap_top1_top2"] > 0.0) else 0.03
        else:
            w_cf = W_CF_STATIC
        return w_cf * rec["cf_norm"] + (1 - w_cf) * rec["cbf_norm"]
    return score_fn


def subset(records, cf_available_value):
    return [r for r in records if r["features"]["CF_AVAILABLE"] == cf_available_value]


def weight_analysis(model, records, X, cf_avail):
    with torch.no_grad():
        x = torch.tensor(X)
        cf_a = torch.tensor(cf_avail)
        w = model(x, cf_a).numpy()
    w_cf = w[:, 0]

    out = {
        "w_cf_mean_all": float(w_cf.mean()),
        "w_cf_mean_when_CF_AVAILABLE_1": float(w_cf[cf_avail == 1].mean()) if (cf_avail == 1).any() else None,
        "w_cf_mean_when_CF_AVAILABLE_0": float(w_cf[cf_avail == 0].mean()) if (cf_avail == 0).any() else None,
        "w_cf_std_all": float(w_cf.std()),
        "w_cf_percentiles": {p: float(np.percentile(w_cf, p)) for p in [5, 25, 50, 75, 95]},
    }

    avail_mask = cf_avail == 1
    if avail_mask.sum() > 20:
        cbf_std_vals = np.array([r["features"]["cbf_score_std"] for r in records])[avail_mask]
        hist_len_vals = np.array([r["features"]["history_length"] for r in records])[avail_mask]
        warm_vals = np.array([r["features"]["warm_candidate_ratio"] for r in records])[avail_mask]
        w_cf_avail = w_cf[avail_mask]

        def decile_table(feature_vals, w_vals):
            try:
                bins = np.quantile(feature_vals, np.linspace(0, 1, 6))
                bins[0] -= 1e-9
                idx = np.digitize(feature_vals, bins) - 1
                idx = np.clip(idx, 0, 4)
                return [float(w_vals[idx == b].mean()) if (idx == b).any() else None for b in range(5)]
            except Exception:
                return None

        out["w_cf_by_cbf_score_std_quintile"] = decile_table(cbf_std_vals, w_cf_avail)
        out["w_cf_by_history_length_quintile"] = decile_table(hist_len_vals, w_cf_avail)
        out["w_cf_by_warm_candidate_ratio_quintile"] = decile_table(warm_vals, w_cf_avail)

    return out


def run_stage(stage: str):
    group_name = "gating_validation" if stage == "validation" else "internal_test"
    with open(RESULTS_DIR / f"features_{group_name}.pkl", "rb") as f:
        records = pickle.load(f)
    print(f"[{stage}] {len(records)} impression ({group_name})")

    rule_cfg = load_rule_config()
    rule_name = rule_cfg["best_rule_name"]
    cbf_std_median = rule_cfg["cbf_std_median_from_gating_train"]
    cbf_max_median = rule_cfg["cbf_max_median_from_gating_train"]
    rule_score_fn = make_rule_based_score_fn(rule_name, cbf_std_median, cbf_max_median)

    model, scaler, arch = load_gating_model()
    X, cf_avail, _ = featurize(records, scaler=scaler)
    gating_scores = compute_gating_scores(model, records, X, cf_avail, torch.device("cpu"))
    idx_ref = {id(r): i for i, r in enumerate(records)}

    def gating_score_fn(rec):
        return gating_scores[idx_ref[id(rec)]]

    scorers = {
        "CBF_puro": score_cbf_pure,
        "Hybrid_statico_0.03_0.97": score_hybrid_static,
        f"Rule_Based_{rule_name}": rule_score_fn,
        "Neural_Gating": gating_score_fn,
    }

    results = {}
    for pop_name, pop_subset in [("all", records), ("CF_AVAILABLE_1", subset(records, 1)), ("CF_AVAILABLE_0", subset(records, 0))]:
        results[pop_name] = {}
        print(f"\n--- Popolazione: {pop_name} (n={len(pop_subset)}) ---")
        for name, fn in scorers.items():
            if name == "Neural_Gating" and pop_name != "all":
                fn_local = (lambda rec, _fn=gating_score_fn: _fn(rec))
            else:
                fn_local = fn
            m = evaluate_records(pop_subset, fn_local)
            results[pop_name][name] = m
            print(f"  {name}: AUC={m.get('AUC', float('nan')):.4f} MRR={m.get('MRR', float('nan')):.4f} nDCG5={m.get('nDCG@5', float('nan')):.4f} nDCG10={m.get('nDCG@10', float('nan')):.4f} P5={m.get('P@5', float('nan')):.4f} R5={m.get('R@5', float('nan')):.4f} n={m.get('n')}")

    print("\n--- Bootstrap paired (2000 resample) su tutte le impression, vs CBF_puro ---")
    bootstrap_results = {}
    for name, fn in scorers.items():
        if name == "CBF_puro":
            continue
        bootstrap_results[name] = {}
        for metric in ["AUC", "MRR", "nDCG@5"]:
            ci = paired_bootstrap_ci(records, fn, score_cbf_pure, metric=metric, n_boot=2000, seed=42)
            bootstrap_results[name][metric] = ci
            print(f"  {name} - CBF_puro | {metric}: mean_diff={ci['mean_diff']:.5f}, IC95%=[{ci['ci95_low']:.5f}, {ci['ci95_high']:.5f}], escludes_zero={ci['ci_excludes_zero']}")

    weights = None
    if stage in ("validation", "internal_test"):
        weights = weight_analysis(model, records, X, cf_avail)
        print("\n--- Analisi pesi appresi (Neural Gating) ---")
        print(json.dumps(weights, indent=2))

    out = {
        "stage": stage, "group_name": group_name, "n_impressions": len(records),
        "rule_name": rule_name, "gating_arch": arch,
        "metrics": results, "bootstrap_vs_cbf": bootstrap_results, "gating_weight_analysis": weights,
    }
    with open(RESULTS_DIR / f"final_comparison_{stage}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nScritto final_comparison_{stage}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validation", "internal_test"], required=True)
    args = ap.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
