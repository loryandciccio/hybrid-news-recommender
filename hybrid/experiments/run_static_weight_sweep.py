#!/usr/bin/env python3
"""
Weight sweep statica dell'Hybrid Recommender.

Riesegue ESATTAMENTE la pipeline esistente in hybrid/hybrid_recommender.py
(nessuna modifica agli algoritmi), variando solo i pesi baseline/CF/CBF passati
al costruttore di HybridRecommender, su tre porzioni disgiunte del DEV set
(--offset 0/2000/40000, --limit 2000 ciascuna).

I dati sono caricati (embedding, matrice ItemKNN, popolarita') UNA SOLA VOLTA
e riutilizzati per tutte le configurazioni, per evitare 48 ricostruzioni
identiche della stessa pipeline e per garantire che ogni configurazione veda
esattamente gli stessi dati.

Uso (dalla root del progetto):
    python3 hybrid/experiments/run_static_weight_sweep.py

Output:
    hybrid/experiments/static_weight_sweep.json   (48 righe, una per configurazione x offset)
    hybrid/experiments/static_weight_sweep.csv     (stesso contenuto, formato tabellare)
    hybrid/experiments/static_weight_summary.csv   (media/deviazione standard per configurazione, sulle 3 porzioni)

Nota: questo script E' la versione pulita e riproducibile dello script
temporaneo (/tmp/weight_grid_experiment.py) usato per generare i risultati
riportati in questa cartella. Rieseguirlo produce numeri identici, perche'
la pipeline sottostante (popolarita', similarita' ItemKNN, embedding SBERT)
e' interamente deterministica e non dipende da alcun seed casuale.
"""

import sys
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

# Rende importabile hybrid_recommender.py indipendentemente dalla cwd da cui si lancia lo script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "hybrid"))

from hybrid_recommender import (
    HybridRecommender,
    DEFAULT_EMBEDDINGS,
    DEFAULT_NEWS,
    DEFAULT_TRAIN_BEHAVIORS,
    DEFAULT_DEV_BEHAVIORS,
)

OUT_DIR = Path(__file__).resolve().parent

# Configurazioni testate: (baseline, cf, cbf). HybridRecommender rinormalizza
# internamente a somma 1, quindi passare 3/97 o 0.03/0.97 e' equivalente.
MAIN_GRID = [
    (0, 0, 100), (0, 1, 99), (0, 2, 98), (0, 3, 97), (0, 5, 95),
    (0, 7.5, 92.5), (0, 10, 90), (0, 15, 85), (0, 20, 80),
    (0, 30, 70), (0, 40, 60), (0, 50, 50),
]
SECONDARY_GRID = [
    (5, 5, 90), (5, 10, 85), (10, 10, 80), (10, 20, 70),
]
CONFIGS = MAIN_GRID + SECONDARY_GRID

OFFSETS = [0, 2000, 40000]
LIMIT = 2000

METRICS = ["AUC", "MRR", "nDCG@5", "nDCG@10", "P@5", "R@5", "Novelty@5", "Novelty@10"]


def _resolve_path(rel: str) -> str:
    p = PROJECT_ROOT / rel
    return str(p) if p.exists() else rel


def run_sweep():
    base = HybridRecommender()
    base.load_data(
        _resolve_path(DEFAULT_EMBEDDINGS),
        _resolve_path(DEFAULT_NEWS),
        _resolve_path(DEFAULT_TRAIN_BEHAVIORS),
    )

    results = []
    total = len(CONFIGS) * len(OFFSETS)
    count = 0

    for bw, cfw, cbfw in CONFIGS:
        for offset in OFFSETS:
            count += 1
            m = HybridRecommender(baseline_weight=bw, cf_weight=cfw, cbf_weight=cbfw)
            # Riusa le strutture dati gia' caricate (embedding, ItemKNN, popolarita'):
            # sono indipendenti dai pesi, quindi identiche per ogni configurazione.
            m.news_id_to_idx = base.news_id_to_idx
            m.news_metadata = base.news_metadata
            m.embeddings = base.embeddings
            m.item_popularity = base.item_popularity
            m.pop_total = base.pop_total
            m.itemknn_similarity = base.itemknn_similarity
            m.news_ids_array = base.news_ids_array
            m.news_id_to_news_idx = base.news_id_to_news_idx

            res = m.evaluate(_resolve_path(DEFAULT_DEV_BEHAVIORS), limit=LIMIT, offset=offset)
            row = {"baseline": bw, "cf": cfw, "cbf": cbfw, "offset": offset, "limit": LIMIT, **res}
            results.append(row)
            print(f"[{count}/{total}] b={bw} cf={cfw} cbf={cbfw} offset={offset} -> "
                  f"AUC={res['AUC']:.5f} MRR={res['MRR']:.5f} nDCG5={res['nDCG@5']:.5f} "
                  f"Nov5={res['Novelty@5']:.4f}", flush=True)

    return results


def write_outputs(results):
    with open(OUT_DIR / "static_weight_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT_DIR / "static_weight_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline_weight", "cf_weight", "cbf_weight", "offset", "limit",
                    "AUC", "MRR", "nDCG@5", "nDCG@10", "Precision@5", "Recall@5",
                    "Novelty@5", "Novelty@10"])
        for r in results:
            w.writerow([r["baseline"], r["cf"], r["cbf"], r["offset"], r["limit"],
                        r["AUC"], r["MRR"], r["nDCG@5"], r["nDCG@10"], r["P@5"], r["R@5"],
                        r["Novelty@5"], r["Novelty@10"]])

    groups = defaultdict(list)
    for r in results:
        groups[(r["baseline"], r["cf"], r["cbf"])].append(r)

    rows = []
    for key, rs in groups.items():
        row = {"baseline_weight": key[0], "cf_weight": key[1], "cbf_weight": key[2], "n_offsets": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs]
            row[m + "_mean"] = st.mean(vals)
            row[m + "_std"] = st.pstdev(vals) if len(vals) > 1 else 0.0
        rows.append(row)
    rows.sort(key=lambda r: -r["AUC_mean"])

    with open(OUT_DIR / "static_weight_summary.csv", "w", newline="") as f:
        header = ["baseline_weight", "cf_weight", "cbf_weight", "n_offsets"] + \
                 [m + "_mean" for m in METRICS] + [m + "_std" for m in METRICS]
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nScritti: static_weight_sweep.{{json,csv}}, static_weight_summary.csv in {OUT_DIR}")


if __name__ == "__main__":
    write_outputs(run_sweep())
