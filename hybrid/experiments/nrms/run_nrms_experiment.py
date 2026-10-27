#!/usr/bin/env python3
"""
Orchestratore dell'esperimento NRMS. Fasi, in ordine:
  1. Costruzione/caricamento vocabolario e lookup titoli (train/news.tsv).
  2. Catalog coverage report (train/news.tsv vs dev/news.tsv, per split).
  3. Costruzione dataset di training (TRAIN=9-11 nov) e record di
     valutazione nativi NRMS (VALIDATION=12-13 nov, INTERNAL_TEST=14 nov,
     candidate set completo, catalogo train/news.tsv al 100% di copertura).
  4. Sanity check obbligatori.
  5. Training (con early stopping su VALIDATION).
  6. Valutazione NRMS "nativa" (100% copertura, catalogo train/news.tsv).
  7. Costruzione del "common evaluation set" per il confronto equo con CBF
     (restrizione ai candidati presenti anche in dev/news.tsv, unico
     catalogo per cui esistono embedding SBERT precalcolati per il CBF) e
     valutazione di NRMS e CBF sulla stessa identica popolazione.
  8. Bootstrap paired, segment analysis, error analysis (contingenza top-5).
  9. Salvataggio di tutti gli output in results/.

Uso (dalla root del progetto):
    python3 hybrid/experiments/nrms/run_nrms_experiment.py --stage smoke
    python3 hybrid/experiments/nrms/run_nrms_experiment.py --stage full
    python3 hybrid/experiments/nrms/run_nrms_experiment.py --stage compare_cbf --split validation
    python3 hybrid/experiments/nrms/run_nrms_experiment.py --stage compare_cbf --split internal_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hybrid/experiments
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hybrid"))

from config import (
    RESULTS_DIR, CHECKPOINTS_DIR, TRAIN_NEWS_PATH, DEV_NEWS_PATH,
    TRAIN_DATES, VALIDATION_DATES, INTERNAL_TEST_DATES,
    MAX_EPOCHS, WORD_EMBEDDING_DIM, NUM_ATTENTION_HEADS, NEWS_VECTOR_DIM,
    MAX_TITLE_LEN, MAX_HISTORY_LEN, NUM_NEGATIVES, BATCH_SIZE, LEARNING_RATE,
)
from tokenizer import Vocab, build_vocab_from_train_news, build_title_lookup
from data import (
    load_behaviors_with_date, compute_catalog_coverage,
    NRMSTrainDataset, build_eval_records, parse_history, parse_impressions,
)
from model import NRMS
from train import run_sanity_checks, train_nrms
from evaluate import evaluate_model_on_records, score_records
from metrics import evaluate_scores

NEWS_COLS = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]


def setup_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    vocab_path = RESULTS_DIR / "vocab.json"
    if vocab_path.exists():
        vocab = Vocab.load(vocab_path)
        print(f"Vocabolario caricato da {vocab_path} ({len(vocab)} token)")
    else:
        vocab = build_vocab_from_train_news()
        vocab.save(vocab_path)

    title_lookup = build_title_lookup(TRAIN_NEWS_PATH, vocab)
    print(f"Title lookup (train/news.tsv): {len(title_lookup)} articoli")

    behav = load_behaviors_with_date()
    return vocab, title_lookup, behav


def report_catalog_coverage(behav):
    train_news = pd.read_csv(TRAIN_NEWS_PATH, sep="\t", header=None, names=NEWS_COLS)
    dev_news = pd.read_csv(DEV_NEWS_PATH, sep="\t", header=None, names=NEWS_COLS)
    train_ids, dev_ids = set(train_news["News_ID"]), set(dev_news["News_ID"])

    coverage = {}
    for name, dates in [("TRAIN", TRAIN_DATES), ("VALIDATION", VALIDATION_DATES), ("INTERNAL_TEST", INTERNAL_TEST_DATES)]:
        coverage[name] = compute_catalog_coverage(behav, dates, train_ids, dev_ids)
        print(f"[{name}] {coverage[name]}")

    with open(RESULTS_DIR / "catalog_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)
    return coverage


def run_smoke(vocab, title_lookup, behav, device):
    train_ds_full = NRMSTrainDataset(behav, TRAIN_DATES, title_lookup)
    print(f"Training dataset completo: {len(train_ds_full)} sample (informativo, per lo smoke test si usa solo un sottoinsieme)")
    run_sanity_checks(len(vocab), train_ds_full, device)


def run_full(vocab, title_lookup, behav, device):
    t0 = time.time()
    train_ds = NRMSTrainDataset(behav, TRAIN_DATES, title_lookup)
    val_records = build_eval_records(behav, VALIDATION_DATES, title_lookup)
    preprocessing_time = time.time() - t0
    print(f"Training samples: {len(train_ds)}  Validation impressions: {len(val_records)}  "
          f"(preprocessing: {preprocessing_time:.1f}s)")

    run_sanity_checks(len(vocab), train_ds, device)

    t0 = time.time()
    model, training_log = train_nrms(len(vocab), train_ds, val_records, device)
    training_time = time.time() - t0

    t0 = time.time()
    val_metrics = evaluate_model_on_records(model, val_records, device)
    inference_time_val = time.time() - t0

    efficiency = {
        "preprocessing_time_sec": round(preprocessing_time, 1),
        "training_time_sec": round(training_time, 1),
        "inference_time_validation_sec": round(inference_time_val, 1),
        "n_validation_impressions": len(val_records),
        "inference_ms_per_impression": round(1000 * inference_time_val / len(val_records), 3),
        "checkpoint_size_mb": round((CHECKPOINTS_DIR / "nrms_best.pt").stat().st_size / (1024 * 1024), 2),
        "n_params": training_log["n_params"],
    }
    print("\nEfficienza:", json.dumps(efficiency, indent=2))
    with open(RESULTS_DIR / "efficiency.json", "w") as f:
        json.dump(efficiency, f, indent=2)
    with open(RESULTS_DIR / "nrms_native_validation_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    print("\nMetriche NRMS native (catalogo train/news.tsv, 100% copertura) su VALIDATION:")
    print(json.dumps(val_metrics, indent=2))


def build_common_eval_set(behav, dates, dev_ids, title_lookup):
    """Filtra i candidati di ogni impression al sottoinsieme presente anche
    in dev/news.tsv (unico catalogo con embedding SBERT per il CBF), e
    ricostruisce i record di valutazione per NRMS su questo stesso
    insieme ristretto. Riporta la quota di impression/candidati esclusi."""
    sub = behav[behav["date_str"].isin(dates)]
    n_impressions_original = 0
    n_impressions_kept = 0
    n_candidates_original = 0
    n_candidates_kept = 0

    records = []
    from data import EMPTY_TITLE_IDS
    import torch as _torch

    for row in sub.itertuples(index=False):
        h = parse_history(row.history)
        cands, y = parse_impressions(row.impressions)
        if not cands or y.sum() == 0:
            continue
        n_impressions_original += 1
        n_candidates_original += len(cands)

        keep_idx = [i for i, c in enumerate(cands) if c in dev_ids]
        if not keep_idx:
            continue
        cands_f = [cands[i] for i in keep_idx]
        y_f = y[keep_idx]
        if y_f.sum() == 0 or y_f.sum() == len(y_f):
            continue

        n_impressions_kept += 1
        n_candidates_kept += len(cands_f)

        h = h[-MAX_HISTORY_LEN:]
        hist_titles = [title_lookup.get(nid, EMPTY_TITLE_IDS) for nid in h]
        n_pad = MAX_HISTORY_LEN - len(hist_titles)
        hist_titles = [EMPTY_TITLE_IDS] * n_pad + hist_titles
        hist_mask = [0.0] * n_pad + [1.0] * len(h)

        class Rec:
            pass
        rec = Rec()
        rec.history_title_ids = _torch.tensor(hist_titles, dtype=_torch.long)
        rec.history_mask = _torch.tensor(hist_mask, dtype=_torch.float32)
        rec.candidate_title_ids = _torch.tensor([title_lookup.get(c, EMPTY_TITLE_IDS) for c in cands_f], dtype=_torch.long)
        rec.y_true = y_f
        rec.cands = cands_f
        rec.history_raw = h
        records.append(rec)

    stats = {
        "n_impressions_original": n_impressions_original,
        "n_impressions_kept": n_impressions_kept,
        "pct_impressions_excluded": round(100 * (1 - n_impressions_kept / n_impressions_original), 2) if n_impressions_original else None,
        "n_candidates_original": n_candidates_original,
        "n_candidates_kept": n_candidates_kept,
        "pct_candidates_excluded": round(100 * (1 - n_candidates_kept / n_candidates_original), 2) if n_candidates_original else None,
    }
    return records, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["setup", "smoke", "full", "compare_cbf"], required=True)
    ap.add_argument("--split", choices=["validation", "internal_test"], default="validation")
    args = ap.parse_args()

    device = torch.device("cpu")
    vocab, title_lookup, behav = setup_data()

    if args.stage == "setup":
        report_catalog_coverage(behav)
        return

    if args.stage == "smoke":
        run_smoke(vocab, title_lookup, behav, device)
        return

    if args.stage == "full":
        run_full(vocab, title_lookup, behav, device)
        return

    if args.stage == "compare_cbf":
        from run_cf_enriched_experiment import cbf_scores as cbf_scores_fn, load_train_behaviors as load_train_behaviors_cf
        import numpy as np_

        dates = VALIDATION_DATES if args.split == "validation" else INTERNAL_TEST_DATES
        dev_news = pd.read_csv(DEV_NEWS_PATH, sep="\t", header=None, names=NEWS_COLS)
        news_id_to_idx = {nid: i for i, nid in enumerate(dev_news["News_ID"])}
        news_metadata = dict(zip(dev_news["News_ID"], dev_news["Category"]))
        embeddings = np.load(Path(__file__).resolve().parents[3] / "content_based_text/embeddings/news_embeddings.npy")
        dev_ids = set(news_id_to_idx.keys())

        print(f"Costruzione common evaluation set per split={args.split} (dates={dates})...")
        records, stats = build_common_eval_set(behav, dates, dev_ids, title_lookup)
        print("Common evaluation set stats:", json.dumps(stats, indent=2))

        arch_path = RESULTS_DIR / "training_log.json"
        with open(arch_path) as f:
            tlog = json.load(f)
        model = NRMS(len(vocab)).to(device)
        model.load_state_dict(torch.load(CHECKPOINTS_DIR / "nrms_best.pt", map_location=device))
        model.eval()

        t0 = time.time()
        nrms_scores = score_records(model, records, device, batch_size=BATCH_SIZE)
        nrms_inference_time = time.time() - t0

        cbf_score_list = []
        for rec in records:
            h = rec.history_raw
            cands = rec.cands
            s = cbf_scores_fn(h, cands, embeddings, news_id_to_idx, news_metadata)
            cbf_score_list.append(s)

        y_true_list = [r.y_true for r in records]
        cands_list = [r.cands for r in records]

        nrms_metrics = evaluate_scores(y_true_list, nrms_scores, cands_list)
        cbf_metrics = evaluate_scores(y_true_list, cbf_score_list, cands_list)

        print(f"\n--- Confronto NRMS vs CBF, common evaluation set ({args.split}, n={len(records)}) ---")
        print("NRMS:", json.dumps(nrms_metrics, indent=2))
        print("CBF :", json.dumps(cbf_metrics, indent=2))

        out = {
            "split": args.split, "common_eval_stats": stats, "n_common": len(records),
            "nrms_metrics": nrms_metrics, "cbf_metrics": cbf_metrics,
            "nrms_inference_time_sec": round(nrms_inference_time, 2),
        }
        with open(RESULTS_DIR / f"comparison_{args.split}.json", "w") as f:
            json.dump(out, f, indent=2)

        # salva anche gli score grezzi per bootstrap/segment/error analysis successive
        np_.savez(
            RESULTS_DIR / f"raw_scores_{args.split}.npz",
            nrms_scores=np_.array(nrms_scores, dtype=object),
            cbf_scores=np_.array(cbf_score_list, dtype=object),
            y_true=np_.array(y_true_list, dtype=object),
            cands=np_.array(cands_list, dtype=object),
            history_len=np_.array([len(r.history_raw) for r in records]),
        )
        print(f"\nScritto comparison_{args.split}.json e raw_scores_{args.split}.npz")


if __name__ == "__main__":
    main()
