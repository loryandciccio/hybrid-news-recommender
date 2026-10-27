#!/usr/bin/env python3
"""
Addestra la Neural Gating Network con una loss pairwise BPR-style
(clicked candidate vs non-clicked candidate della stessa impression), MAI
sulla label diagnostica CF_WINS.

GATING_TRAIN = 12 novembre, GATING_VALIDATION = 13 novembre (mai
INTERNAL_TEST = 14 novembre, congelato). Lo scaler delle feature e' fittato
SOLO su GATING_TRAIN.

Confronta al massimo 4 configurazioni predefinite (hidden in {[8],[16,8]} x
lr in {1e-3,1e-2}), nessuna grid search estesa. Seed fisso, early stopping
sulla AUC di GATING_VALIDATION, numero massimo di epoch limitato.

NON modifica hybrid_recommender.py.

Uso (dalla root del progetto):
    python3 hybrid/experiments/adaptive_hybrid/train_gating.py
"""

from __future__ import annotations

import json
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from common import RESULTS_DIR
from gating_network import GatingNet
from evaluate_common import evaluate_records

FEATURE_COLS = [
    "history_length", "candidate_count", "CF_AVAILABLE",
    "cf_score_std", "cf_score_max", "cf_gap_top1_top2",
    "cbf_score_std", "cbf_score_max", "cbf_gap_top1_top2",
    "top5_overlap_cf_cbf", "spearman_cf_cbf",
    "warm_candidate_ratio", "popularity_mean_fit", "popularity_median_fit",
]

MAX_EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 512
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def featurize(records, scaler=None):
    X = np.array([[r["features"][c] for c in FEATURE_COLS] for r in records], dtype=np.float32)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X).astype(np.float32)
    cf_avail = np.array([r["features"]["CF_AVAILABLE"] for r in records], dtype=np.float32)
    return X_s, cf_avail, scaler


def bpr_loss_for_batch(model, batch_records, X_batch, cf_avail_batch, device):
    x = torch.tensor(X_batch, device=device)
    cf_a = torch.tensor(cf_avail_batch, device=device)
    w = model(x, cf_a)  # (B, 2)

    losses = []
    for i, rec in enumerate(batch_records):
        y = rec["y_true"]
        cf_n = torch.tensor(rec["cf_norm"], device=device)
        cbf_n = torch.tensor(rec["cbf_norm"], device=device)
        score = w[i, 0] * cf_n + w[i, 1] * cbf_n
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pos_scores = score[pos_idx].unsqueeze(1)   # (P,1)
        neg_scores = score[neg_idx].unsqueeze(0)   # (1,N)
        diff = pos_scores - neg_scores
        losses.append(-torch.nn.functional.logsigmoid(diff).mean())

    if not losses:
        return None
    return torch.stack(losses).mean()


def train_one_config(hidden, lr, train_recs, val_recs, X_train, cf_a_train, X_val, cf_a_val, device):
    set_seed(SEED)
    model = GatingNet(n_features=len(FEATURE_COLS), hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []

    n = len(train_recs)
    indices = np.arange(n)

    for epoch in range(MAX_EPOCHS):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        rng.shuffle(indices)
        epoch_losses = []
        for start in range(0, n, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            batch_records = [train_recs[i] for i in batch_idx]
            X_batch = X_train[batch_idx]
            cf_a_batch = cf_a_train[batch_idx]
            loss = bpr_loss_for_batch(model, batch_records, X_batch, cf_a_batch, device)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = bpr_loss_for_batch(model, val_recs, X_val, cf_a_val, device)
            val_loss_val = float(val_loss.item()) if val_loss is not None else float("nan")

        val_metrics = evaluate_gating_on_records(model, val_recs, X_val, cf_a_val, device)
        train_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss_mean, "val_loss": val_loss_val, "val_AUC": val_metrics["AUC"]})
        print(f"    epoch {epoch}: train_loss={train_loss_mean:.4f} val_loss={val_loss_val:.4f} val_AUC={val_metrics['AUC']:.4f}")

        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"    early stopping all'epoch {epoch} (best val_AUC={best_val_auc:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val_auc, history


def compute_gating_scores(model, records, X, cf_avail, device):
    """Score gating pre-calcolati per ogni record, nello stesso ordine di `records`."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, device=device)
        cf_a = torch.tensor(cf_avail, device=device)
        w = model(x, cf_a).cpu().numpy()
    return [w[i, 0] * rec["cf_norm"] + w[i, 1] * rec["cbf_norm"] for i, rec in enumerate(records)]


def evaluate_gating_on_records(model, records, X, cf_avail, device):
    precomputed = compute_gating_scores(model, records, X, cf_avail, device)
    idx_ref = {id(rec): i for i, rec in enumerate(records)}

    def score_fn(rec):
        return precomputed[idx_ref[id(rec)]]

    return evaluate_records(records, score_fn)


def main():
    device = torch.device("cpu")
    with open(RESULTS_DIR / "features_gating_train.pkl", "rb") as f:
        train_recs = pickle.load(f)
    with open(RESULTS_DIR / "features_gating_validation.pkl", "rb") as f:
        val_recs = pickle.load(f)

    X_train, cf_a_train, scaler = featurize(train_recs, scaler=None)
    X_val, cf_a_val, _ = featurize(val_recs, scaler=scaler)

    configs = [
        {"hidden": [8], "lr": 1e-3},
        {"hidden": [8], "lr": 1e-2},
        {"hidden": [16, 8], "lr": 1e-3},
        {"hidden": [16, 8], "lr": 1e-2},
    ]

    all_results = []
    best_overall = None
    for cfg in configs:
        print(f"\n=== Config: hidden={cfg['hidden']} lr={cfg['lr']} ===")
        model, best_val_auc, history = train_one_config(
            cfg["hidden"], cfg["lr"], train_recs, val_recs, X_train, cf_a_train, X_val, cf_a_val, device
        )
        n_params = model.n_params()
        result = {
            "hidden": cfg["hidden"], "lr": cfg["lr"], "best_val_AUC": best_val_auc,
            "n_params": n_params, "n_epochs_run": len(history), "history": history,
        }
        all_results.append(result)
        if best_overall is None or best_val_auc > best_overall["best_val_AUC"]:
            best_overall = result
            torch.save(model.state_dict(), RESULTS_DIR / "gating_model_best.pt")

    print("\n--- Riepilogo configurazioni (selezione SOLO su GATING_VALIDATION) ---")
    for r in all_results:
        print(f"hidden={r['hidden']} lr={r['lr']}: best_val_AUC={r['best_val_AUC']:.4f}, n_params={r['n_params']}, epoch={r['n_epochs_run']}")
    print(f"\nConfig scelta: hidden={best_overall['hidden']} lr={best_overall['lr']} (val_AUC={best_overall['best_val_AUC']:.4f})")

    with open(RESULTS_DIR / "gating_training_log.json", "w") as f:
        json.dump({
            "seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
            "feature_cols": FEATURE_COLS, "configs": all_results,
            "best_config": {"hidden": best_overall["hidden"], "lr": best_overall["lr"], "n_params": best_overall["n_params"], "best_val_AUC": best_overall["best_val_AUC"]},
        }, f, indent=2)

    with open(RESULTS_DIR / "gating_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(RESULTS_DIR / "gating_best_arch.json", "w") as f:
        json.dump({"hidden": best_overall["hidden"], "lr": best_overall["lr"], "feature_cols": FEATURE_COLS}, f, indent=2)

    print("\nScritto gating_model_best.pt, gating_training_log.json, gating_scaler.pkl, gating_best_arch.json")


if __name__ == "__main__":
    main()
