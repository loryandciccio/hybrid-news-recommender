#!/usr/bin/env python3
"""
Audit rigoroso della definizione di "cold-start" usata (implicitamente) in
hybrid_recommender.py e nel report ANALISI_COMPLEMENTARITA.md.

NON modifica alcun algoritmo. Calcola, per ciascun articolo cliccato nelle
impression DEV a singolo positivo, cinque categorie NON equivalenti:

    A. Assente completamente da MINDsmall_train/news.tsv
    B. Presente in train/news.tsv ma mai comparso nel campo `history` di
       alcuna riga di TRAIN/behaviors.tsv (= la definizione di popolarita'
       usata da hybrid_recommender.py, item_popularity)
    C. Presente in train/news.tsv ma mai cliccato (etichetta "-1") nelle
       impression del TRAIN (= la definizione di popolarita' usata da
       baseline_model/baseline.ipynb e cf_itemknn/*)
    D. Mai comparso come candidato (click o no) nelle impression del TRAIN
    E. Nessuna rappresentazione CF utilizzabile (0 occorrenze nel conteggio
       item_counts usato da _build_itemknn_matrix per costruire il grafo
       di co-click)

Uso (dalla root del progetto, richiede che
hybrid/experiments/impression_level_results.csv sia gia' stato generato da
run_complementarity_analysis.py):
    python3 hybrid/experiments/run_coldstart_audit.py

Output:
    hybrid/experiments/coldstart_category_breakdown.csv  (una riga per
        impression a singolo positivo, con le 5 colonne booleane A-E)
    hybrid/experiments/coldstart_manual_sample.csv        (campione di 25
        articoli con titolo, per verifica manuale del matching degli ID)
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

NEWS_COLS = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]
BEHAV_COLS = ["imp_id", "user_id", "time", "history", "impressions"]


def main():
    train_news = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_train/news.tsv", sep="\t", header=None, names=NEWS_COLS)
    train_news_ids = set(train_news["News_ID"].astype(str).str.strip())
    print(f"A-universo: {len(train_news_ids)} articoli distinti in train/news.tsv")

    train_behav = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_train/behaviors.tsv", sep="\t", header=None, names=BEHAV_COLS)

    history_ids = set()
    for h in train_behav["history"].dropna():
        history_ids.update(str(h).split())
    print(f"B-universo: {len(history_ids)} articoli distinti nelle history del TRAIN")

    clicked_ids, all_candidate_ids = set(), set()
    for imp in train_behav["impressions"].dropna():
        for tok in str(imp).split():
            nid, lab = tok.rsplit("-", 1)
            all_candidate_ids.add(nid)
            if lab == "1":
                clicked_ids.add(nid)
    print(f"C-universo: {len(clicked_ids)} articoli cliccati almeno una volta nel TRAIN")
    print(f"D-universo: {len(all_candidate_ids)} articoli comparsi come candidati nel TRAIN")

    dev_news = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_dev/news.tsv", sep="\t", header=None, names=NEWS_COLS)
    dev_news_ids = set(dev_news["News_ID"].astype(str).str.strip())
    dev_news_indexed = dev_news.set_index(dev_news["News_ID"].astype(str).str.strip())

    # Replica esatta della logica di _build_itemknn_matrix (hybrid_recommender.py) per item_counts
    item_counts = Counter()
    for h in train_behav["history"]:
        history = str(h).split() if pd.notna(h) else []
        history = [nid for nid in history if nid in dev_news_ids]
        if len(history) < 2:
            continue
        for nid in history:
            item_counts[nid] += 1
    cf_represented_ids = set(item_counts.keys())
    print(f"E-universo: {len(cf_represented_ids)} articoli con rappresentazione CF (item_counts>0)")

    imp = pd.read_csv(OUT_DIR / "impression_level_results.csv")
    single = imp[imp["single_positive"] == True].copy()  # noqa: E712
    clicked_dev_ids = single["clicked_id"].astype(str).str.strip()
    print(f"\nImpression DEV a singolo positivo: {len(single)}")

    def categorize(nid):
        return pd.Series({
            "A_absent_from_train_news": nid not in train_news_ids,
            "B_never_in_train_history": nid not in history_ids,
            "C_never_clicked_in_train": nid not in clicked_ids,
            "D_never_candidate_in_train": nid not in all_candidate_ids,
            "E_no_cf_representation": nid not in cf_represented_ids,
        })

    results = clicked_dev_ids.apply(categorize)
    results["clicked_id"] = clicked_dev_ids.values
    n = len(results)

    print(f"\nQuota per categoria (su {n} impression a singolo positivo):")
    for col in results.columns[:-1]:
        cnt = int(results[col].sum())
        print(f"  {col}: {cnt} ({100*cnt/n:.2f}%)")

    results.to_csv(OUT_DIR / "coldstart_category_breakdown.csv", index=False)
    print(f"\nScritto coldstart_category_breakdown.csv ({n} righe)")

    # --- Campione per verifica manuale ---
    sample = results.sample(25, random_state=42).copy()
    titles = []
    for nid in sample["clicked_id"]:
        titles.append(dev_news_indexed.loc[nid]["Title"] if nid in dev_news_indexed.index else "???")
    sample["title_dev"] = titles
    sample.to_csv(OUT_DIR / "coldstart_manual_sample.csv", index=False)
    print(f"Scritto coldstart_manual_sample.csv (25 righe per verifica manuale)")


if __name__ == "__main__":
    main()
