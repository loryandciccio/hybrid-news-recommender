#!/usr/bin/env python3
"""
Rule-Based Adaptive Hybrid: un numero limitato (5) di regole interpretabili,
tutte basate esclusivamente su feature pre-click e sulla maschera
CF_AVAILABLE. Le soglie (mediane di cbf_score_std/cbf_score_max) sono
stimate SOLO su GATING_TRAIN (12 novembre) e poi congelate; la scelta della
regola migliore viene fatta guardando le metriche su GATING_VALIDATION
(13 novembre), mai su INTERNAL_TEST.

Regola comune a tutte le varianti: se CF_AVAILABLE=0, w_CF=0 (CBF puro) per
costruzione — non e' un'opzione scelta dai dati, e' imposta esplicitamente
per evitare che un ranking CF degenere (score tutti uguali, deciso dal solo
tie-break su news_id) influenzi il punteggio finale.

NON modifica hybrid_recommender.py.

Uso (dalla root del progetto):
    python3 hybrid/experiments/adaptive_hybrid/rule_based_hybrid.py
"""

from __future__ import annotations

import json
import pickle

import numpy as np

from common import RESULTS_DIR
from evaluate_common import evaluate_records, W_CF_STATIC, W_CBF_STATIC


def score_static(rec, w_cf=W_CF_STATIC, w_cbf=W_CBF_STATIC):
    if not rec["features"]["CF_AVAILABLE"]:
        return rec["cbf_norm"]
    return w_cf * rec["cf_norm"] + w_cbf * rec["cbf_norm"]


def make_threshold_rule(threshold_feature, w_cf_low_disp, w_cf_high_disp, low_means):
    """Regola: se la feature di dispersione del CBF e' 'bassa confidenza'
    (sotto soglia se low_means=True, sopra se low_means=False) assegna un
    peso CF maggiore; altrimenti quasi solo CBF. Sempre 0 se CF_AVAILABLE=0.
    """
    def score_fn(rec, threshold):
        feats = rec["features"]
        if not feats["CF_AVAILABLE"]:
            return rec["cbf_norm"]
        val = feats[threshold_feature]
        use_high_cf = (val < threshold) if low_means else (val > threshold)
        w_cf = w_cf_low_disp if use_high_cf else w_cf_high_disp
        w_cbf = 1.0 - w_cf
        return w_cf * rec["cf_norm"] + w_cbf * rec["cbf_norm"]
    return score_fn


def make_combined_rule(std_threshold, gap_threshold, w_cf_boost):
    def score_fn(rec):
        feats = rec["features"]
        if not feats["CF_AVAILABLE"]:
            return rec["cbf_norm"]
        if feats["cbf_score_std"] < std_threshold and feats["cf_gap_top1_top2"] > gap_threshold:
            w_cf = w_cf_boost
        else:
            w_cf = W_CF_STATIC
        return w_cf * rec["cf_norm"] + (1 - w_cf) * rec["cbf_norm"]
    return score_fn


def main():
    with open(RESULTS_DIR / "features_gating_train.pkl", "rb") as f:
        train_recs = pickle.load(f)
    with open(RESULTS_DIR / "features_gating_validation.pkl", "rb") as f:
        val_recs = pickle.load(f)

    cf_avail_train = [r for r in train_recs if r["features"]["CF_AVAILABLE"]]
    cbf_std_median = float(np.median([r["features"]["cbf_score_std"] for r in cf_avail_train]))
    cbf_max_median = float(np.median([r["features"]["cbf_score_max"] for r in cf_avail_train]))
    print(f"Soglie stimate su GATING_TRAIN (12 nov, n_CF_AVAILABLE={len(cf_avail_train)}):")
    print(f"  mediana cbf_score_std = {cbf_std_median:.4f}")
    print(f"  mediana cbf_score_max = {cbf_max_median:.4f}")

    rule_std = make_threshold_rule("cbf_score_std", w_cf_low_disp=0.10, w_cf_high_disp=0.03, low_means=True)
    rule_max = make_threshold_rule("cbf_score_max", w_cf_low_disp=0.10, w_cf_high_disp=0.03, low_means=True)
    rule_std_strong = make_threshold_rule("cbf_score_std", w_cf_low_disp=0.20, w_cf_high_disp=0.03, low_means=True)
    rule_combined = make_combined_rule(cbf_std_median, 0.0, w_cf_boost=0.10)

    configs = {
        "R0_hybrid_static_0.03_0.97": lambda rec: score_static(rec),
        "R1_cf_available_only_0.03_0.97": lambda rec: score_static(rec),  # identico a R0 (CF_AVAILABLE gia' forzato in score_static)
        "R2_std_threshold_0.10_0.03": lambda rec: rule_std(rec, cbf_std_median),
        "R3_max_threshold_0.10_0.03": lambda rec: rule_max(rec, cbf_max_median),
        "R4_std_threshold_strong_0.20_0.03": lambda rec: rule_std_strong(rec, cbf_std_median),
        "R5_combined_std_and_gap": lambda rec: rule_combined(rec),
    }

    print("\n--- Metriche su GATING_VALIDATION (13 nov), per regola ---")
    val_results = {}
    for name, fn in configs.items():
        metrics = evaluate_records(val_recs, fn)
        val_results[name] = metrics
        print(f"{name}: AUC={metrics['AUC']:.4f} MRR={metrics['MRR']:.4f} nDCG5={metrics['nDCG@5']:.4f} P5={metrics['P@5']:.4f}")

    cbf_pure_metrics = evaluate_records(val_recs, lambda rec: rec["cbf_norm"])
    print(f"\nCBF_puro (riferimento): AUC={cbf_pure_metrics['AUC']:.4f} MRR={cbf_pure_metrics['MRR']:.4f} nDCG5={cbf_pure_metrics['nDCG@5']:.4f}")

    best_name = max(
        [n for n in val_results if n != "R0_hybrid_static_0.03_0.97" and n != "R1_cf_available_only_0.03_0.97"],
        key=lambda n: val_results[n]["AUC"],
    )
    print(f"\nRegola adattiva migliore su GATING_VALIDATION (per AUC): {best_name}")

    frozen_config = {
        "cbf_std_median_from_gating_train": cbf_std_median,
        "cbf_max_median_from_gating_train": cbf_max_median,
        "best_rule_name": best_name,
        "gating_validation_results": val_results,
        "cbf_pure_reference": cbf_pure_metrics,
    }
    with open(RESULTS_DIR / "rule_based_selection.json", "w") as f:
        json.dump(frozen_config, f, indent=2)
    print(f"\nScritto rule_based_selection.json (regola congelata: {best_name})")


if __name__ == "__main__":
    main()
