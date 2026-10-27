#!/usr/bin/env python3
"""
Grid search sui pesi principali dell'Hybrid Recommender.

Individua i pesi ottimali per:
- baseline_weight
- cf_weight
- cbf_weight

Ogni combinazione viene valutata lanciando hybrid_recommender.py come
sottoprocesso indipendente (ricostruendo l'intera pipeline dati a ogni run).
Salva i risultati in grid_search_results.csv.
"""

import subprocess
import csv
import sys
from itertools import product


def extract_metric(output, metric_name):
    """Estrae il valore di una metrica dall'output testuale del modello."""
    try:
        for line in output.split('\n'):
            if metric_name in line:
                # Effettua il parsing di righe del tipo "AUC         : 0.6384"
                parts = line.split(':')
                if len(parts) >= 2:
                    return float(parts[1].strip())
    except Exception:
        pass
    return None


# Parametri della grid search
BASELINE_WEIGHTS = [0.0, 0.05, 0.1, 0.15]
CF_WEIGHTS = [0.0, 0.05, 0.1, 0.15, 0.2]
LIMIT = 500  # 500 per un'iterazione rapida; 2000 per la validazione finale

# Costruisce la griglia di configurazioni
results = []

total_configs = len(BASELINE_WEIGHTS) * len(CF_WEIGHTS)
current = 0

print("=" * 80)
print("GRID SEARCH: PESI DELL'HYBRID RECOMMENDER")
print("=" * 80)
print(f"Test di {total_configs} configurazioni con --limit {LIMIT}")
print("=" * 80)

for baseline_w in BASELINE_WEIGHTS:
    for cf_w in CF_WEIGHTS:
        cbf_w = 1.0 - baseline_w - cf_w

        # Scarta le combinazioni non valide (peso CBF fuori da [0, 1])
        if cbf_w < 0 or cbf_w > 1.0:
            continue

        current += 1

        print(f"\n[{current}/{total_configs}] Test: Baseline={baseline_w:.2f}, CF={cf_w:.2f}, CBF={cbf_w:.2f}")

        # Esegue la valutazione come sottoprocesso
        cmd = [
            "python3",
            "hybrid/hybrid_recommender.py",
            f"--baseline-weight={baseline_w}",
            f"--cf-weight={cf_w}",
            f"--cbf-weight={cbf_w}",
            f"--limit={LIMIT}"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Effettua il parsing dell'output per estrarre le metriche
            output = result.stdout

            # Estrae le metriche finali
            auc = extract_metric(output, "AUC")
            mrr = extract_metric(output, "MRR")
            ndcg5 = extract_metric(output, "nDCG@5")
            ndcg10 = extract_metric(output, "nDCG@10")
            p5 = extract_metric(output, "P@5")
            r5 = extract_metric(output, "R@5")
            novelty5 = extract_metric(output, "Novelty@5")
            novelty10 = extract_metric(output, "Novelty@10")
            
            config = {
                'baseline_weight': baseline_w,
                'cf_weight': cf_w,
                'cbf_weight': cbf_w,
                'auc': auc,
                'mrr': mrr,
                'ndcg5': ndcg5,
                'ndcg10': ndcg10,
                'p5': p5,
                'r5': r5,
                'novelty5': novelty5,
                'novelty10': novelty10,
            }
            
            results.append(config)

            # Stampa il risultato della configurazione corrente
            if auc is not None:
                print(f"  ✓ AUC: {auc:.4f} | MRR: {mrr:.4f} | nDCG@5: {ndcg5:.4f}")
            else:
                print(f"  ✗ Impossibile effettuare il parsing delle metriche")

        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout dopo 600 secondi")
        except Exception as e:
            print(f"  ✗ Errore: {e}")

# Salva i risultati
print("\n" + "=" * 80)
print("SALVATAGGIO RISULTATI")
print("=" * 80)

if results:
    # Ordina per AUC decrescente
    results_sorted = sorted(results, key=lambda x: x.get('auc', 0) or 0, reverse=True)

    csv_path = "hybrid/grid_search_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_sorted[0].keys())
        writer.writeheader()
        writer.writerows(results_sorted)

    print(f"Risultati salvati in {csv_path}")
    print(f"\nMigliori 10 configurazioni:\n")

    for i, config in enumerate(results_sorted[:10], 1):
        print(f"{i}. Baseline={config['baseline_weight']:.2f}, CF={config['cf_weight']:.2f}, CBF={config['cbf_weight']:.2f}")
        print(f"   AUC={config['auc']:.4f}, MRR={config['mrr']:.4f}, nDCG@5={config['ndcg5']:.4f}")
        print()
else:
    print("Nessun risultato da salvare!")

print("=" * 80)
