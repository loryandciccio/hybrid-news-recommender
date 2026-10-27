#!/usr/bin/env python3
"""
Complementarita' CF_enriched vs CBF, CF_WINS diagnostico, e classificatore
diagnostico minimo (Logistic Regression + Decision Tree) ricalcolati
INTERAMENTE sul protocollo temporale interno (FIT=Nov9-11,
VALIDATION=Nov12-13), senza mai toccare INTERNAL_TEST (Nov14) ne' il DEV
(Nov15). Sostituisce i numeri ottenuti sul protocollo DEV per qualunque
decisione GO/NO-GO.

NON modifica hybrid_recommender.py. Nessun modello di produzione alterato.
Nessuna rete neurale: solo Logistic Regression e un Decision Tree poco
profondo, usati in modo puramente diagnostico.

Split diagnostico interno (temporale, dentro VALIDATION):
    TRAIN_DIAGNOSTIC      = 12 novembre
    VALIDATION_DIAGNOSTIC = 13 novembre

Uso (dalla root del progetto):
    python3 hybrid/experiments/run_validation_complementarity_v2.py

Output:
    hybrid/experiments/v2_cf_wins_diagnostic.csv
    hybrid/experiments/v2_topk_contingency.json
    hybrid/experiments/v2_rank_delta.csv
    hybrid/experiments/v2_segment_complementarity.csv
    hybrid/experiments/v2_classifier_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "hybrid"))

from run_cf_enriched_experiment import (  # noqa: E402
    load_train_behaviors, parse_history, parse_impressions,
    build_itemknn, cf_scores, cbf_scores,
)
from temporal_split_definition import FIT_DATES, VALIDATION_DATES  # noqa: E402

NEWS_COLS = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]


def rank_of_clicked(y_score, cands, clicked_id):
    order = np.lexsort((np.asarray(cands, dtype=str), -y_score))
    ranked_ids = [cands[i] for i in order]
    return ranked_ids.index(clicked_id) + 1


def topk_bucket(rank, k):
    return rank <= k


def main():
    behav = load_train_behaviors()
    fit_df = behav[behav["date_str"].isin(FIT_DATES)]
    val_df = behav[behav["date_str"].isin(VALIDATION_DATES)]
    print(f"FIT righe: {len(fit_df)}  VALIDATION righe: {len(val_df)}")

    news_meta_df = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_dev/news.tsv", sep="\t", header=None, names=NEWS_COLS)
    news_id_to_idx = {nid: i for i, nid in enumerate(news_meta_df["News_ID"])}
    news_metadata = dict(zip(news_meta_df["News_ID"], news_meta_df["Category"]))
    embeddings = np.load(PROJECT_ROOT / "content_based_text/embeddings/news_embeddings.npy")
    news_ids_universe = set(news_id_to_idx.keys())

    enriched_sets = []
    for row in fit_df.itertuples(index=False):
        h = parse_history(row.history)
        cands, y_true = parse_impressions(row.impressions)
        clicked = [cands[i] for i in np.where(y_true == 1)[0]]
        enriched_sets.append(h + clicked)

    print("Costruzione grafo ItemKNN CF_enriched (history + click positivi FIT)...")
    sim_enr, idx_enr, item_counts_enr = build_itemknn(news_ids_universe, enriched_sets)
    print(f"  Item con rappresentazione CF_enriched: {len(item_counts_enr)}")

    rows = []
    for row in val_df.itertuples(index=False):
        h = parse_history(row.history)
        cands, y_true = parse_impressions(row.impressions)
        if not h or not cands or y_true.sum() != 1:
            continue
        clicked_id = cands[int(np.argmax(y_true))]

        cf_raw = cf_scores(h, cands, sim_enr, idx_enr)
        cbf_raw = cbf_scores(h, cands, embeddings, news_id_to_idx, news_metadata)

        rank_cf = rank_of_clicked(cf_raw, cands, clicked_id)
        rank_cbf = rank_of_clicked(cbf_raw, cands, clicked_id)

        n_candidates = len(cands)
        history_length = len(h)
        warm_mask = np.array([c in idx_enr for c in cands])
        warm_candidate_ratio = float(warm_mask.mean())

        cf_std = float(np.std(cf_raw))
        cbf_std = float(np.std(cbf_raw))
        cf_max = float(cf_raw.max())
        cbf_max = float(cbf_raw.max())
        cf_sorted = np.sort(cf_raw)[::-1]
        cbf_sorted = np.sort(cbf_raw)[::-1]
        cf_gap = float(cf_sorted[0] - cf_sorted[1]) if len(cf_sorted) > 1 else 0.0
        cbf_gap = float(cbf_sorted[0] - cbf_sorted[1]) if len(cbf_sorted) > 1 else 0.0
        k_top = min(5, n_candidates)
        top_cf_set = set(np.argsort(-cf_raw)[:k_top].tolist())
        top_cbf_set = set(np.argsort(-cbf_raw)[:k_top].tolist())
        topk_agreement = len(top_cf_set & top_cbf_set) / k_top if k_top > 0 else np.nan

        rows.append({
            "date": row.date_str,
            "history_length": history_length,
            "n_candidates": n_candidates,
            "warm_candidate_ratio": warm_candidate_ratio,
            "cf_score_std": cf_std,
            "cbf_score_std": cbf_std,
            "cf_score_max": cf_max,
            "cbf_score_max": cbf_max,
            "cf_gap_top1_top2": cf_gap,
            "cbf_gap_top1_top2": cbf_gap,
            "topk5_agreement_cf_cbf": topk_agreement,
            "rank_CF": rank_cf,
            "rank_CBF": rank_cbf,
            "CF_WINS": int(rank_cf < rank_cbf),
            "CF_TIE": int(rank_cf == rank_cbf),
            "CBF_WINS": int(rank_cbf < rank_cf),
            "delta_rank": rank_cf - rank_cbf,
            "cf_top1": rank_cf == 1,
            "cbf_top1": rank_cbf == 1,
            "cf_top5": rank_cf <= 5,
            "cbf_top5": rank_cbf <= 5,
            "cf_top10": rank_cf <= 10,
            "cbf_top10": rank_cbf <= 10,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "v2_cf_wins_diagnostic.csv", index=False)
    print(f"\nImpression a singolo positivo su VALIDATION: {len(df)}")
    print(f"CF_WINS: {100*df['CF_WINS'].mean():.2f}%  CBF_WINS: {100*df['CBF_WINS'].mean():.2f}%  TIE: {100*df['CF_TIE'].mean():.2f}%")

    # --- Sezione 5: contingenza TOP-1/5/10 ---
    contingency = {}
    for k, cf_col, cbf_col in [(1, "cf_top1", "cbf_top1"), (5, "cf_top5", "cbf_top5"), (10, "cf_top10", "cbf_top10")]:
        both = (df[cf_col] & df[cbf_col]).mean()
        only_cf = (df[cf_col] & ~df[cbf_col]).mean()
        only_cbf = (~df[cf_col] & df[cbf_col]).mean()
        neither = (~df[cf_col] & ~df[cbf_col]).mean()
        contingency[f"top{k}"] = {
            "solo_CF": round(100 * only_cf, 2),
            "solo_CBF": round(100 * only_cbf, 2),
            "entrambi": round(100 * both, 2),
            "nessuno": round(100 * neither, 2),
        }
        print(f"TOP-{k}: solo_CF={contingency[f'top{k}']['solo_CF']}%  solo_CBF={contingency[f'top{k}']['solo_CBF']}%  entrambi={contingency[f'top{k}']['entrambi']}%  nessuno={contingency[f'top{k}']['nessuno']}%")

    with open(OUT_DIR / "v2_topk_contingency.json", "w") as f:
        json.dump(contingency, f, indent=2)

    # --- Sezione 6: distribuzione Delta rank ---
    delta_summary = {
        "pct_CF_migliore": round(100 * (df["delta_rank"] < 0).mean(), 2),
        "pct_CBF_migliore": round(100 * (df["delta_rank"] > 0).mean(), 2),
        "pct_tie": round(100 * (df["delta_rank"] == 0).mean(), 2),
        "delta_rank_mean": round(float(df["delta_rank"].mean()), 3),
        "delta_rank_median": round(float(df["delta_rank"].median()), 3),
        "delta_rank_p10": round(float(df["delta_rank"].quantile(0.10)), 2),
        "delta_rank_p90": round(float(df["delta_rank"].quantile(0.90)), 2),
    }
    print("\nDelta rank (rank_CF - rank_CBF):", delta_summary)
    df[["date", "delta_rank", "rank_CF", "rank_CBF"]].to_csv(OUT_DIR / "v2_rank_delta.csv", index=False)
    with open(OUT_DIR / "v2_delta_rank_summary.json", "w") as f:
        json.dump(delta_summary, f, indent=2)

    # --- Sezione 7: complementarita' per segmento (solo feature pre-click) ---
    seg_rows = []

    def add_seg(name, mask):
        sub = df[mask]
        if len(sub) == 0:
            return
        seg_rows.append({
            "segment": name,
            "n": len(sub),
            "CF_WINS_pct": round(100 * sub["CF_WINS"].mean(), 2),
            "CBF_WINS_pct": round(100 * sub["CBF_WINS"].mean(), 2),
            "cf_top5_pct": round(100 * sub["cf_top5"].mean(), 2),
            "cbf_top5_pct": round(100 * sub["cbf_top5"].mean(), 2),
        })

    h_q = df["history_length"].quantile([0.33, 0.66]).values
    add_seg("history_corta", df["history_length"] <= h_q[0])
    add_seg("history_media", (df["history_length"] > h_q[0]) & (df["history_length"] <= h_q[1]))
    add_seg("history_lunga", df["history_length"] > h_q[1])

    w_q = df["warm_candidate_ratio"].quantile([0.33, 0.66]).values
    add_seg("candidate_set_prevalentemente_cold", df["warm_candidate_ratio"] <= w_q[0])
    add_seg("candidate_set_quota_warm_elevata", df["warm_candidate_ratio"] > w_q[1])

    c_med = df["cbf_score_std"].median()
    add_seg("bassa_cbf_score_std", df["cbf_score_std"] <= c_med)
    add_seg("alta_cbf_score_std", df["cbf_score_std"] > c_med)

    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "v2_segment_complementarity.csv", index=False)
    print("\nComplementarita' per segmento:\n", seg_df.to_string(index=False))

    # --- Sezione 9: split diagnostico temporale (12 nov = train, 13 nov = val) ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler

    feature_cols = [
        "history_length", "n_candidates", "warm_candidate_ratio",
        "cf_score_std", "cbf_score_std", "cf_score_max", "cbf_score_max",
        "cf_gap_top1_top2", "cbf_gap_top1_top2", "topk5_agreement_cf_cbf",
    ]

    train_diag = df[df["date"] == "2019-11-12"].dropna(subset=feature_cols + ["CF_WINS"])
    val_diag = df[df["date"] == "2019-11-13"].dropna(subset=feature_cols + ["CF_WINS"])
    print(f"\nTRAIN_DIAGNOSTIC (12 nov): {len(train_diag)}  VALIDATION_DIAGNOSTIC (13 nov): {len(val_diag)}")
    print(f"Prevalenza CF_WINS: train={train_diag['CF_WINS'].mean():.4f}  val={val_diag['CF_WINS'].mean():.4f}")

    X_train, y_train = train_diag[feature_cols].values, train_diag["CF_WINS"].values
    X_val, y_val = val_diag[feature_cols].values, val_diag["CF_WINS"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    results = {}

    majority_pred = np.zeros(len(y_val))
    prevalence = y_train.mean()
    results["baseline_majority"] = {
        "roc_auc": 0.5,
        "pr_auc": round(float(prevalence), 4),
        "balanced_accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
    }

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    p_val = lr.predict_proba(X_val_s)[:, 1]
    pred_val = (p_val >= 0.5).astype(int)
    results["logistic_regression"] = {
        "roc_auc": round(float(roc_auc_score(y_val, p_val)), 4),
        "pr_auc": round(float(average_precision_score(y_val, p_val)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_val, pred_val)), 4),
        "precision": round(float(precision_score(y_val, pred_val, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, pred_val, zero_division=0)), 4),
        "coefficients": dict(zip(feature_cols, [round(float(c), 4) for c in lr.coef_[0]])),
    }

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, class_weight="balanced", random_state=42)
    tree.fit(X_train, y_train)
    p_val_tree = tree.predict_proba(X_val)[:, 1]
    pred_val_tree = tree.predict(X_val)
    results["decision_tree"] = {
        "roc_auc": round(float(roc_auc_score(y_val, p_val_tree)), 4),
        "pr_auc": round(float(average_precision_score(y_val, p_val_tree)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_val, pred_val_tree)), 4),
        "precision": round(float(precision_score(y_val, pred_val_tree, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, pred_val_tree, zero_division=0)), 4),
        "feature_importances": dict(zip(feature_cols, [round(float(i), 4) for i in tree.feature_importances_])),
    }

    results["_meta"] = {
        "n_train_diagnostic": int(len(train_diag)),
        "n_val_diagnostic": int(len(val_diag)),
        "prevalence_train": round(float(prevalence), 4),
        "prevalence_val": round(float(y_val.mean()), 4),
        "feature_cols": feature_cols,
    }

    print("\n--- Risultati classificatore diagnostico (CF_WINS, VALIDATION_DIAGNOSTIC=13 nov) ---")
    for name, r in results.items():
        if name == "_meta":
            continue
        print(f"{name}: ROC-AUC={r['roc_auc']}, PR-AUC={r['pr_auc']}, bal_acc={r['balanced_accuracy']}, precision={r['precision']}, recall={r['recall']}")

    with open(OUT_DIR / "v2_classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nScritto v2_classifier_results.json")


if __name__ == "__main__":
    main()
