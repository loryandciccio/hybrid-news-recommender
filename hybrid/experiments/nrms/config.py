#!/usr/bin/env python3
"""
Configurazione NRMS. Un'unica fonte di verità per iperparametri, path e
protocollo temporale — nessun valore duplicato altrove.

Framework: PyTorch (gia' presente nell'ambiente, usato anche dalla fase
Adaptive Hybrid; non e' stata aggiunta alcuna dipendenza a
requirements.txt principale). Verificato prima di procedere: torch 2.11.0
disponibile.

Semplificazioni dichiarate rispetto al paper NRMS originale (Wu et al.,
2019), per mantenere il modello minimale e interpretabile:
- Solo il campo `title` come input testuale (no abstract/categoria/entità).
- Embedding delle parole appresi da zero,
  NON pretrained: evita di scaricare GloVe/altri embedding esterni non
  già presenti nel repository, mantenendo l'esperimento riproducibile
  offline. Dichiarato esplicitamente come scelta, non default silenzioso.
- Un solo layer di self-attention multi-head per il news encoder e per lo
  user encoder (il paper originale usa la stessa profondita', quindi qui
  non e' una riduzione: e' l'architettura standard NRMS a singolo layer).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NRMS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = NRMS_DIR / "results"
CHECKPOINTS_DIR = NRMS_DIR / "checkpoints"
FIGURES_DIR = NRMS_DIR / "figures"

TRAIN_NEWS_PATH = PROJECT_ROOT / "dataset/MINDsmall_train/news.tsv"
DEV_NEWS_PATH = PROJECT_ROOT / "dataset/MINDsmall_dev/news.tsv"
TRAIN_BEHAVIORS_PATH = PROJECT_ROOT / "dataset/MINDsmall_train/behaviors.tsv"

# --- Split temporale (riutilizzato invariato dalle fasi precedenti) ---
TRAIN_DATES = ["2019-11-09", "2019-11-10", "2019-11-11"]
VALIDATION_DATES = ["2019-11-12", "2019-11-13"]
INTERNAL_TEST_DATES = ["2019-11-14"]

# --- Tokenizzazione / vocabolario ---
MAX_TITLE_LEN = 20          # troncamento/padding titolo, in token
MIN_TOKEN_FREQ = 2          # frequenza minima nel TRAIN per entrare nel vocabolario
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# --- History utente ---
MAX_HISTORY_LEN = 20         # troncamento alle MAX_HISTORY_LEN news piu' recenti

# --- Architettura ---
WORD_EMBEDDING_DIM = 128     # divisibile per NUM_ATTENTION_HEADS
NUM_ATTENTION_HEADS = 8
HEAD_DIM = WORD_EMBEDDING_DIM // NUM_ATTENTION_HEADS  # 16
NEWS_VECTOR_DIM = WORD_EMBEDDING_DIM  # dimensione output del news encoder (= query/key/value dim totale)
ADDITIVE_ATTN_DIM = 128
DROPOUT = 0.2

# --- Negative sampling ---
NUM_NEGATIVES = 4            # 1 positivo + 4 negativi per training sample
NEGATIVE_SAMPLING_SEED = 42

# --- Training ---
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 15
PATIENCE = 3
GRAD_CLIP_NORM = 5.0
