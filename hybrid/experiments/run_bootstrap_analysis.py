#!/usr/bin/env python3
"""
Bootstrap paired sulle differenze di metrica tra configurazioni, a livello di
impression, usando i risultati gia' calcolati da run_complementarity_analysis.py
(hybrid/experiments/impression_level_results.csv, 70.938 impression valide
sull'intero DEV set).

NON modifica alcun algoritmo: opera esclusivamente sui valori di AUC/MRR/nDCG5
gia' calcolati per ciascuna impression e per ciascuna configurazione.

Metodologia: per ciascun confronto (es. Hybrid 0/3/97 vs CBF puro), si calcola
la differenza di metrica IMPRESSION PER IMPRESSION (accoppiata, dato che le due
configurazioni sono valutate esattamente sulle stesse impression), poi si
ricampiona con reinserimento l'insieme delle impression B volte, calcolando la
media della differenza ad ogni ricampionamento. L'intervallo di confidenza al
95% e' il 2.5-97.5 percentile della distribuzione bootstrap delle medie.

Se l'intervallo di confidenza include lo 0, si scrive esplicitamente che non
emerge evidenza di una differenza consistente sul campione analizzato -
NON si usa la parola "significativo" in senso statistico stretto.

Uso:
    python3 hybrid/experiments/run_bootstrap_analysis.py
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

SEED = 42
N_BOOTSTRAP = 10000

OUT_DIR = Path(__file__).resolve().parent

def paired_bootstrap(diff: np.ndarray, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_impressions": int(n),
        "observed_mean_diff": float(diff.mean()),
        "bootstrap_mean_diff": float(means.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "ci_includes_zero": bool(lo <= 0 <= hi),
        "n_bootstrap_samples": n_boot,
        "seed": seed,
    }


def main():
    df = pd.read_csv(OUT_DIR / "impression_level_results.csv")

    comparisons = [
        ("hybrid_0_03_97", "cbf_pure", "Hybrid 0/3/97 vs CBF puro"),
        ("cf_pure", "cbf_pure", "CF puro vs CBF puro"),
        ("hybrid_0_10_90", "cbf_pure", "Hybrid 0/10/90 vs CBF puro"),
        ("baseline_pure", "cbf_pure", "Baseline pura vs CBF puro"),
    ]
    metrics = ["AUC", "MRR", "nDCG5"]

    results = {}
    for cfg_a, cfg_b, label in comparisons:
        results[label] = {}
        for m in metrics:
            col_a = f"{cfg_a}__{m}"
            col_b = f"{cfg_b}__{m}"
            valid = df[[col_a, col_b]].dropna()
            diff = (valid[col_a] - valid[col_b]).to_numpy()
            res = paired_bootstrap(diff)
            results[label][m] = res
            verdict = "l'intervallo include lo 0: non emerge evidenza di una differenza consistente nel campione analizzato" if res["ci_includes_zero"] else "l'intervallo NON include lo 0"
            print(f"{label} | Δ{m} = {res['observed_mean_diff']:+.5f}  CI95=[{res['ci95_low']:+.5f}, {res['ci95_high']:+.5f}]  -> {verdict}")
        print()

    with open(OUT_DIR / "bootstrap_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Risultati salvati in {OUT_DIR / 'bootstrap_results.json'}")


if __name__ == "__main__":
    main()
