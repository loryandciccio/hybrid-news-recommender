#!/usr/bin/env python3
"""
Script di tuning per il decadimento temporale del CBF e il bilanciamento
interno semantico/topical dell'Hybrid Recommender (grid search su
cbf_decay × cbf_text_weight, a pesi principali baseline/CF/CBF fissi).
"""

import os
import polars as pl
import numpy as np
import pandas as pd
from hybrid_recommender import HybridRecommender, DEFAULT_EMBEDDINGS, DEFAULT_NEWS, DEFAULT_TRAIN_BEHAVIORS, DEFAULT_DEV_BEHAVIORS

def grid_search(limit=500):
    # Corregge i percorsi a seconda che lo script sia lanciato da hybrid/ o dalla root
    emb_path = DEFAULT_EMBEDDINGS if os.path.exists(DEFAULT_EMBEDDINGS) else "../" + DEFAULT_EMBEDDINGS
    news_path = DEFAULT_NEWS if os.path.exists(DEFAULT_NEWS) else "../" + DEFAULT_NEWS
    train_path = DEFAULT_TRAIN_BEHAVIORS if os.path.exists(DEFAULT_TRAIN_BEHAVIORS) else "../" + DEFAULT_TRAIN_BEHAVIORS
    dev_path = DEFAULT_DEV_BEHAVIORS if os.path.exists(DEFAULT_DEV_BEHAVIORS) else "../" + DEFAULT_DEV_BEHAVIORS

    # Intervalli di iperparametri da testare
    decays = [0.8, 0.9, 0.95, 0.99]
    text_weights = [0.3, 0.4, 0.5, 0.6, 0.7]

    # Carica i dati (embedding, metadata, popolarità, matrice ItemKNN) una sola volta:
    # questi artefatti non dipendono dai pesi/iperparametri testati nel ciclo,
    # quindi vengono riutilizzati per ogni combinazione invece di essere ricaricati.
    hybrid = HybridRecommender()
    hybrid.load_data(emb_path, news_path, train_path)

    results_list = []

    print(f"\nAvvio Grid Search (combinazioni totali: {len(decays) * len(text_weights)})\n")

    for d in decays:
        for tw in text_weights:
            print(f">>> Test: Decay={d}, Text Weight={tw}")

            # Nuova istanza "leggera" con i soli iperparametri da testare;
            # le strutture dati pesanti vengono copiate dall'istanza già caricata
            # sopra, evitando di ricostruire embedding e matrice ItemKNN ad ogni iterazione.
            current_hybrid = HybridRecommender(
                cbf_decay=d,
                cbf_text_weight=tw,
                cbf_topic_weight=1.0 - tw
            )
            current_hybrid.news_id_to_idx = hybrid.news_id_to_idx
            current_hybrid.news_metadata = hybrid.news_metadata
            current_hybrid.embeddings = hybrid.embeddings
            current_hybrid.item_popularity = hybrid.item_popularity
            current_hybrid.pop_total = hybrid.pop_total
            current_hybrid.itemknn_similarity = hybrid.itemknn_similarity
            current_hybrid.news_ids_array = hybrid.news_ids_array
            current_hybrid.news_id_to_news_idx = hybrid.news_id_to_news_idx

            metrics = current_hybrid.evaluate(dev_path, limit=limit)

            res = {
                "Decay": d,
                "Text_Weight": tw,
                "Topic_Weight": 1.0 - tw,
                **metrics
            }
            results_list.append(res)

    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by="AUC", ascending=False)

    output_file = "tuning_results.csv"
    df_results.to_csv(output_file, index=False)

    print("\n" + "="*50)
    print("GRID SEARCH COMPLETATA")
    print(f"Migliori 3 configurazioni (per AUC):")
    print(df_results.head(3).to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    grid_search(limit=500)
