#!/usr/bin/env python3
"""
Analisi di segmentazione a partire da impression_level_results.csv e
ranking_correlation.csv (prodotti da run_complementarity_analysis.py).

NON modifica alcun algoritmo del progetto: opera esclusivamente sui valori
gia' calcolati per impression (AUC/MRR/nDCG/hit@K per le 5 configurazioni,
piu' le covariate history_length, clicked_train_pop, clicked_cold).

Copre le Fasi 4-10 dell'analisi di complementarita':
- Fase 4-6: performance per bucket di lunghezza della history utente
  (bucket: 0-2, 3-5, 6-10, 11-20, >20; aggiunta una suddivisione
  estesa della coda lunga, documentata in ANALISI_COMPLEMENTARITA.md §3).
- Fase 7: performance per stato di cold-start dell'item cliccato
  (popolarita' di training == 0), solo su impression a singolo positivo.
- Fase 8: performance per bucket di popolarita' di training dell'item
  cliccato (quantili 33/66 tra gli item con popolarita' > 0).
- Fase 9: matrice di complementarita' CF/CBF (hit@1, hit@5), globale e per
  bucket di history length e per stato cold-start.
- Fase 10: riepilogo di correlazione di Spearman e overlap/Jaccard Top-5
  tra i ranking CF e CBF.

Uso (dalla root del progetto, dopo aver eseguito run_complementarity_analysis.py):
    python3 hybrid/experiments/run_segment_analysis.py

Output (tutti in questa cartella):
    results_by_history_length.csv
    results_by_history_length_extended.csv
    results_by_coldstart.csv
    results_by_candidate_popularity.csv
    complementarity_matrix.json
    complementarity_by_history_length.csv
    ranking_correlation_summary.csv
"""

from pathlib import Path
import json

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

CONFIGS = ["baseline_pure", "cf_pure", "cbf_pure", "hybrid_0_03_97", "hybrid_0_10_90"]
METRICS = ["AUC", "MRR", "nDCG5", "nDCG10", "P5", "R5", "Novelty5", "Novelty10"]

HIST_BINS = [0, 2, 5, 10, 20, 10_000]
HIST_LABELS = ["0-2", "3-5", "6-10", "11-20", ">20"]

HIST_BINS_EXT = [0, 2, 5, 10, 20, 50, 100, 10_000]
HIST_LABELS_EXT = ["0-2", "3-5", "6-10", "11-20", "21-50", "51-100", ">100"]


def summarize_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for bucket, g in df.groupby(group_col, observed=True):
        row = {
            "bucket": bucket,
            "n_impressions": len(g),
            "mean_n_candidates": g["n_candidates"].mean(),
            "mean_history_length": g["history_length"].mean(),
        }
        for cfg in CONFIGS:
            for m in METRICS:
                row[f"{cfg}__{m}"] = g[f"{cfg}__{m}"].mean()
            row[f"{cfg}__hit1_rate"] = g[f"{cfg}__hit1"].mean()
            row[f"{cfg}__hit5_rate"] = g[f"{cfg}__hit5"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def add_deltas(t: pd.DataFrame) -> pd.DataFrame:
    for m in ["AUC", "MRR", "nDCG5", "Novelty5"]:
        t[f"delta_{m}_CF_CBF"] = t[f"cf_pure__{m}"] - t[f"cbf_pure__{m}"]
        t[f"delta_{m}_Hybrid_CBF"] = t[f"hybrid_0_03_97__{m}"] - t[f"cbf_pure__{m}"]
    return t


def contingency(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    both = int(((df[col_a]) & (df[col_b])).sum())
    only_a = int(((df[col_a]) & (~df[col_b])).sum())
    only_b = int(((~df[col_a]) & (df[col_b])).sum())
    neither = int(((~df[col_a]) & (~df[col_b])).sum())
    n = int(len(df))
    return {
        "both": both, "only_CF": only_a, "only_CBF": only_b, "neither": neither, "n": n,
        "both_pct": round(100 * both / n, 2), "only_CF_pct": round(100 * only_a / n, 2),
        "only_CBF_pct": round(100 * only_b / n, 2), "neither_pct": round(100 * neither / n, 2),
    }


def main():
    df = pd.read_csv(OUT_DIR / "impression_level_results.csv")
    corr = pd.read_csv(OUT_DIR / "ranking_correlation.csv")

    df["hist_bucket"] = pd.cut(df["history_length"], bins=HIST_BINS, labels=HIST_LABELS)
    df["hist_bucket_ext"] = pd.cut(df["history_length"], bins=HIST_BINS_EXT, labels=HIST_LABELS_EXT)

    # --- Fase 4-6: history length ---
    t1 = add_deltas(summarize_by(df, "hist_bucket"))
    t1.to_csv(OUT_DIR / "results_by_history_length.csv", index=False)
    t2 = summarize_by(df, "hist_bucket_ext")
    t2.to_csv(OUT_DIR / "results_by_history_length_extended.csv", index=False)
    print("Scritto results_by_history_length.csv e _extended.csv")

    # --- Fase 7: cold-start (solo impression a singolo positivo) ---
    single = df[df["single_positive"] == True].copy()  # noqa: E712
    print(f"Impression a singolo positivo: {len(single)} / {len(df)} ({100*len(single)/len(df):.1f}%)")

    rows = []
    for cold, g in single.groupby("clicked_cold"):
        row = {"clicked_cold": cold, "n_impressions": len(g)}
        for cfg in CONFIGS:
            for m in ["AUC", "MRR", "nDCG5", "R5", "Novelty5"]:
                row[f"{cfg}__{m}"] = g[f"{cfg}__{m}"].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "results_by_coldstart.csv", index=False)
    print("Scritto results_by_coldstart.csv")

    # --- Fase 8: popolarita' del candidato cliccato (solo TRAIN) ---
    warm = single[single["clicked_train_pop"] > 0]
    q_low, q_high = warm["clicked_train_pop"].quantile([0.33, 0.66]).values

    def pop_bucket(pop):
        if pop == 0:
            return "unseen"
        elif pop <= q_low:
            return "bassa"
        elif pop <= q_high:
            return "media"
        return "alta"

    single = single.copy()
    single["pop_bucket"] = single["clicked_train_pop"].apply(pop_bucket)
    rows = []
    for b in ["unseen", "bassa", "media", "alta"]:
        g = single[single["pop_bucket"] == b]
        if len(g) == 0:
            continue
        row = {"pop_bucket": b, "n_impressions": len(g), "mean_train_pop": g["clicked_train_pop"].mean()}
        for cfg in CONFIGS:
            for m in ["AUC", "MRR", "nDCG5", "R5", "Novelty5"]:
                row[f"{cfg}__{m}"] = g[f"{cfg}__{m}"].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "results_by_candidate_popularity.csv", index=False)
    print(f"Scritto results_by_candidate_popularity.csv (quantili pop>0: 33%={q_low}, 66%={q_high})")

    # --- Fase 9: matrice di complementarita' ---
    summary = {
        "hit1": contingency(df, "cf_pure__hit1", "cbf_pure__hit1"),
        "hit5": contingency(df, "cf_pure__hit5", "cbf_pure__hit5"),
    }
    json.dump(summary, open(OUT_DIR / "complementarity_matrix.json", "w"), indent=2)
    print("Scritto complementarity_matrix.json:", json.dumps(summary, indent=2))

    rows = []
    for b, g in df.groupby("hist_bucket", observed=True):
        n = len(g)
        both = int(((g["cf_pure__hit5"]) & (g["cbf_pure__hit5"])).sum())
        only_cf = int(((g["cf_pure__hit5"]) & (~g["cbf_pure__hit5"])).sum())
        only_cbf = int(((~g["cf_pure__hit5"]) & (g["cbf_pure__hit5"])).sum())
        neither = int(((~g["cf_pure__hit5"]) & (~g["cbf_pure__hit5"])).sum())
        rows.append({"bucket": b, "n": n, "both_%": 100 * both / n, "only_CF_%": 100 * only_cf / n,
                     "only_CBF_%": 100 * only_cbf / n, "neither_%": 100 * neither / n})
    pd.DataFrame(rows).to_csv(OUT_DIR / "complementarity_by_history_length.csv", index=False)
    print("Scritto complementarity_by_history_length.csv")

    # --- Fase 10: correlazione/overlap dei ranking ---
    desc = corr[["spearman_cf_cbf", "top5_overlap_frac", "top5_jaccard"]].describe()
    desc.to_csv(OUT_DIR / "ranking_correlation_summary.csv")
    print("Scritto ranking_correlation_summary.csv")
    print(desc)


if __name__ == "__main__":
    main()
