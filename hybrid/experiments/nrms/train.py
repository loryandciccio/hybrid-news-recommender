#!/usr/bin/env python3
"""
Training NRMS: softmax cross-entropy su (1 positivo + NUM_NEGATIVES
negativi) per training sample — NON MSE. Seed fisso, early stopping su
AUC di VALIDATION, gradient clipping, checkpoint del best model.

Sanity check obbligatori, eseguiti PRIMA del
training completo da `run_nrms_experiment.py`:
    A) un batch passa nel modello senza errori;
    B) le dimensioni dei tensori sono coerenti con l'architettura attesa;
    C) su input casuali (pesi non addestrati) gli score differenziano
       candidati diversi (non sono tutti identici per costruzione);
    D)/E) la loss diminuisce e il modello arriva a overfittare quasi
       perfettamente un piccolo sottoinsieme (accuracy training vicina a 1).
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    SEED, BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE, GRAD_CLIP_NORM,
    MAX_HISTORY_LEN, MAX_TITLE_LEN, CHECKPOINTS_DIR, RESULTS_DIR,
)
from model import NRMS
from data import NRMSTrainDataset
from evaluate import evaluate_model_on_records


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_sanity_checks(vocab_size, train_dataset, device):
    print("=" * 70)
    print("SANITY CHECK OBBLIGATORI (pre-training)")
    print("=" * 70)
    set_seed(SEED)
    model = NRMS(vocab_size).to(device)

    loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    hist = batch["history_title_ids"].to(device)
    hist_mask = batch["history_mask"].to(device)
    cand = batch["candidate_title_ids"].to(device)
    target = batch["target"].to(device)

    assert hist.shape == (8, MAX_HISTORY_LEN, MAX_TITLE_LEN), f"shape history errata: {hist.shape}"
    assert cand.shape[0] == 8 and cand.shape[2] == MAX_TITLE_LEN, f"shape candidati errata: {cand.shape}"
    scores = model(hist, hist_mask, cand)
    assert scores.shape == (8, cand.shape[1]), f"shape score errata: {scores.shape}"
    print(f"[A/B] Batch OK. history={tuple(hist.shape)} candidate={tuple(cand.shape)} scores={tuple(scores.shape)}")

    with torch.no_grad():
        std_across_candidates = scores.std(dim=1).mean().item()
    assert std_across_candidates > 1e-4, "gli score non differenziano i candidati (std troppo bassa)"
    print(f"[C] Deviazione standard media degli score tra candidati (pesi non addestrati): {std_across_candidates:.4f} > 0 -> OK")

    print("\n[D/E] Test di overfitting su un piccolo sottoinsieme (64 sample, 200 step)...")
    tiny_indices = list(range(min(64, len(train_dataset))))
    tiny_subset = torch.utils.data.Subset(train_dataset, tiny_indices)
    tiny_loader = DataLoader(tiny_subset, batch_size=16, shuffle=True)

    overfit_model = NRMS(vocab_size).to(device)
    optimizer = torch.optim.Adam(overfit_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for step in range(200):
        for batch in tiny_loader:
            hist = batch["history_title_ids"].to(device)
            hist_mask = batch["history_mask"].to(device)
            cand = batch["candidate_title_ids"].to(device)
            target = batch["target"].to(device)
            scores = overfit_model(hist, hist_mask, cand)
            loss = loss_fn(scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    with torch.no_grad():
        correct, total = 0, 0
        for batch in DataLoader(tiny_subset, batch_size=16, shuffle=False):
            hist = batch["history_title_ids"].to(device)
            hist_mask = batch["history_mask"].to(device)
            cand = batch["candidate_title_ids"].to(device)
            target = batch["target"].to(device)
            scores = overfit_model(hist, hist_mask, cand)
            pred = scores.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += len(target)
    train_acc = correct / total
    print(f"    loss iniziale={losses[0]:.4f} -> loss finale={losses[-1]:.4f}")
    print(f"    training accuracy su overfit-set: {train_acc:.4f}")
    assert losses[-1] < losses[0] * 0.5, "la loss non e' diminuita a sufficienza: bloccare il training completo"
    assert train_acc > 0.9, f"il modello non riesce a overfittare un piccolo campione (acc={train_acc:.2f} < 0.90): bloccare il training completo"
    print("\nTUTTI I SANITY CHECK SUPERATI. Procedo con il training completo.")
    print("=" * 70)
    return {
        "batch_shapes_ok": True,
        "score_std_untrained": std_across_candidates,
        "overfit_initial_loss": losses[0],
        "overfit_final_loss": losses[-1],
        "overfit_train_accuracy": train_acc,
    }


def train_nrms(vocab_size, train_dataset, val_records, device):
    set_seed(SEED)
    model = NRMS(vocab_size).to(device)
    n_params = model.n_params()
    print(f"Modello NRMS: {n_params:,} parametri totali")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        train_dataset.resample_epoch(epoch)
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model.train()
        t0 = time.time()
        epoch_losses = []
        for batch in loader:
            hist = batch["history_title_ids"].to(device)
            hist_mask = batch["history_mask"].to(device)
            cand = batch["candidate_title_ids"].to(device)
            target = batch["target"].to(device)

            scores = model(hist, hist_mask, cand)
            loss = loss_fn(scores, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_time = time.time() - t0
        val_metrics = evaluate_model_on_records(model, val_records, device)
        train_loss_mean = float(np.mean(epoch_losses))

        entry = {
            "epoch": epoch, "train_loss": train_loss_mean, "train_time_sec": round(train_time, 1),
            "val_AUC": val_metrics.get("AUC"), "val_MRR": val_metrics.get("MRR"),
            "val_nDCG@5": val_metrics.get("nDCG@5"), "val_nDCG@10": val_metrics.get("nDCG@10"),
        }
        history.append(entry)
        print(f"Epoch {epoch}: train_loss={train_loss_mean:.4f} ({train_time:.1f}s) "
              f"val_AUC={val_metrics.get('AUC'):.4f} val_MRR={val_metrics.get('MRR'):.4f} "
              f"val_nDCG@5={val_metrics.get('nDCG@5'):.4f}")

        if val_metrics.get("AUC", 0) > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "nrms_best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping all'epoch {epoch} (best val_AUC={best_val_auc:.4f} all'epoch {best_epoch})")
                break

    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "nrms_best.pt"))

    training_log = {
        "seed": SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "grad_clip_norm": GRAD_CLIP_NORM,
        "n_params": n_params, "best_epoch": best_epoch, "best_val_AUC": best_val_auc,
        "history": history,
    }
    with open(RESULTS_DIR / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    checkpoint_size_mb = (CHECKPOINTS_DIR / "nrms_best.pt").stat().st_size / (1024 * 1024)
    print(f"\nTraining completato. Best epoch={best_epoch}, best val_AUC={best_val_auc:.4f}")
    print(f"Checkpoint: {checkpoint_size_mb:.2f} MB")

    return model, training_log
