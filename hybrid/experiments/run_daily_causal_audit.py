#!/usr/bin/env python3
"""
Audit causale giorno-per-giorno (TRAIN, 9-14 novembre 2019).

Obiettivo: spiegare quantitativamente il salto di AUC del CBF osservato tra
VALIDATION (Nov 12-13, AUC~0.5157) e INTERNAL_TEST (Nov 14, AUC~0.6104), e
fornire statistiche descrittive giorno-per-giorno rispettando la causalita'
temporale (un giorno D non usa mai informazioni di comportamento di giorni
successivi).

Il CBF (`cbf_scores`) usa solo: (a) embedding SBERT statici (proprieta' del
testo, non del comportamento, quindi non soggetti a vincoli di causalita'
temporale), (b) il campo `history` della riga stessa, che e' STATICO per
utente durante tutta la finestra TRAIN (verificato in
run_cf_enriched_experiment.py / VALIDAZIONE_METODOLOGICA.md sezione 9.5) e
quindi precede l'intera finestra di log delle impression. Per il CBF non
serve percio' alcun "fitting" su giorni precedenti: ogni riga e' gia'
valutabile causalmente con i soli dati della riga stessa.

Le uniche statistiche che richiedono un vincolo esplicito di causalita' sono
quelle "aggregate nel tempo" (cold-start rate, quota di candidati mai visti
prima), che qui sono sempre calcolate usando SOLO i giorni strettamente
precedenti al giorno valutato (cumulative, mai lo stesso giorno o successivi).

NON modifica hybrid_recommender.py. Nessun modello di produzione alterato.

Uso (dalla root del progetto):
    python3 hybrid/experiments/run_daily_causal_audit.py

Output:
    hybrid/experiments/daily_cbf_audit.csv
    hybrid/experiments/daily_stratified_comparison.csv
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "hybrid"))

from run_cf_enriched_experiment import cbf_scores, parse_history, parse_impressions  # noqa: E402
from hybrid_recommender import ndcg_at_k  # noqa: E402

NEWS_COLS = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]
BEHAV_COLS = ["imp_id", "user_id", "time", "history", "impressions"]

DAYS = ["2019-11-09", "2019-11-10", "2019-11-11", "2019-11-12", "2019-11-13", "2019-11-14"]


def rank_metrics(y_true, y_score, cands):
    order = np.lexsort((cands, -y_score))
    y_sorted = y_true[order]
    hits = np.where(y_sorted == 1)[0]
    mrr = 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0
    return mrr


def main():
    behav = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_train/behaviors.tsv", sep="\t", header=None, names=BEHAV_COLS)
    behav["time"] = pd.to_datetime(behav["time"], format="%m/%d/%Y %I:%M:%S %p")
    behav["date_str"] = behav["time"].dt.strftime("%Y-%m-%d")

    news_meta_df = pd.read_csv(PROJECT_ROOT / "dataset/MINDsmall_dev/news.tsv", sep="\t", header=None, names=NEWS_COLS)
    news_id_to_idx = {nid: i for i, nid in enumerate(news_meta_df["News_ID"])}
    news_metadata = dict(zip(news_meta_df["News_ID"], news_meta_df["Category"]))
    embeddings = np.load(PROJECT_ROOT / "content_based_text/embeddings/news_embeddings.npy")

    # Stato cumulativo CAUSALE: aggiornato SOLO dopo aver valutato il giorno corrente,
    # cosi' che il giorno D venga sempre valutato con lo stato accumulato dai soli
    # giorni < D (mai D stesso).
    seen_history_items_before = set()
    seen_candidate_items_before = set()
    seen_users_before = set()

    daily_rows = []
    per_impression_records = []  # per la comparazione stratificata Nov12-13 vs Nov14

    for day in DAYS:
        day_df = behav[behav["date_str"] == day]

        history_lens, cand_counts, n_pos_list = [], [], []
        aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
        cold_start_flags = []  # solo per impression a singolo positivo
        new_candidate_flags = []
        users_today = set()

        this_day_history_items = set()
        this_day_candidate_items = set()

        for row in day_df.itertuples(index=False):
            h = parse_history(row.history)
            cands, y_true = parse_impressions(row.impressions)
            users_today.add(row.user_id)
            this_day_history_items.update(h)
            this_day_candidate_items.update(cands)

            if not h or not cands:
                continue

            history_lens.append(len(h))
            cand_counts.append(len(cands))
            n_pos_list.append(int(y_true.sum()))

            # quota di candidati mai visti come candidati nei giorni PRECEDENTI (causale)
            if cands:
                n_new = sum(1 for c in cands if c not in seen_candidate_items_before)
                new_candidate_flags.append(n_new / len(cands))

            if y_true.sum() == 1:
                clicked_id = cands[int(np.argmax(y_true))]
                cold_start_flags.append(clicked_id not in seen_history_items_before)

            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue

            y_score = cbf_scores(h, cands, embeddings, news_id_to_idx, news_metadata)
            try:
                auc = roc_auc_score(y_true, y_score)
                aucs.append(auc)
            except Exception:
                auc = None
            mrrs.append(rank_metrics(y_true, y_score, cands))
            ndcg5s.append(ndcg_at_k(y_true, y_score, 5))
            ndcg10s.append(ndcg_at_k(y_true, y_score, 10))

            if auc is not None:
                per_impression_records.append({
                    "date": day,
                    "auc": auc,
                    "history_length": len(h),
                    "n_candidates": len(cands),
                })

        n_users_overlap_with_before = len(users_today & seen_users_before)

        daily_rows.append({
            "date": day,
            "n_impressions_total": int(len(day_df)),
            "n_impressions_evaluable_auc": int(len(aucs)),
            "n_impressions_single_positive": int(len(cold_start_flags)),
            "cbf_AUC": float(np.mean(aucs)) if aucs else None,
            "cbf_MRR": float(np.mean(mrrs)) if mrrs else None,
            "cbf_nDCG5": float(np.mean(ndcg5s)) if ndcg5s else None,
            "cbf_nDCG10": float(np.mean(ndcg10s)) if ndcg10s else None,
            "history_length_mean": float(np.mean(history_lens)) if history_lens else None,
            "history_length_median": float(np.median(history_lens)) if history_lens else None,
            "candidate_count_mean": float(np.mean(cand_counts)) if cand_counts else None,
            "candidate_count_median": float(np.median(cand_counts)) if cand_counts else None,
            "n_positive_mean": float(np.mean(n_pos_list)) if n_pos_list else None,
            "cold_start_rate_causal": float(np.mean(cold_start_flags)) if cold_start_flags else None,
            "pct_new_candidates_causal": float(np.mean(new_candidate_flags)) * 100 if new_candidate_flags else None,
            "n_distinct_users": int(len(users_today)),
            "n_users_overlap_with_previous_days": int(n_users_overlap_with_before),
            "pct_users_overlap_with_previous_days": round(100 * n_users_overlap_with_before / len(users_today), 2) if users_today else None,
            "n_distinct_history_items_today": int(len(this_day_history_items)),
            "n_distinct_candidate_items_today": int(len(this_day_candidate_items)),
        })

        # aggiorna lo stato cumulativo SOLO ora, per il giorno successivo
        seen_history_items_before.update(this_day_history_items)
        seen_candidate_items_before.update(this_day_candidate_items)
        seen_users_before.update(users_today)

        print(f"{day}: " + ", ".join(f"{k}={v}" for k, v in daily_rows[-1].items() if k != "date"))

    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(OUT_DIR / "daily_cbf_audit.csv", index=False)
    print(f"\nScritto daily_cbf_audit.csv ({len(daily_df)} righe)")

    # --- Comparazione stratificata Nov12-13 vs Nov14, per isolare composizione vs drift ---
    imp_df = pd.DataFrame(per_impression_records)
    imp_df["period"] = imp_df["date"].map(lambda d: "VALIDATION_12_13" if d in ("2019-11-12", "2019-11-13") else ("INTERNAL_TEST_14" if d == "2019-11-14" else "other"))
    imp_df = imp_df[imp_df["period"] != "other"]

    strat_rows = []
    for feature in ["history_length", "n_candidates"]:
        try:
            imp_df[f"{feature}_decile"] = pd.qcut(imp_df[feature], 5, duplicates="drop")
        except Exception:
            continue
        g = imp_df.groupby([f"{feature}_decile", "period"], observed=True)["auc"].agg(["mean", "count"]).reset_index()
        g["stratify_by"] = feature
        strat_rows.append(g)

    strat_df = pd.concat(strat_rows, ignore_index=True) if strat_rows else pd.DataFrame()
    strat_df.to_csv(OUT_DIR / "daily_stratified_comparison.csv", index=False)
    print(f"Scritto daily_stratified_comparison.csv ({len(strat_df)} righe)")

    print("\n--- Confronto grezzo (non stratificato) ---")
    for period in ["VALIDATION_12_13", "INTERNAL_TEST_14"]:
        sub = imp_df[imp_df["period"] == period]
        print(f"{period}: n={len(sub)}, AUC medio={sub['auc'].mean():.4f}, hist_len medio={sub['history_length'].mean():.2f}, n_cand medio={sub['n_candidates'].mean():.2f}")


if __name__ == "__main__":
    main()
