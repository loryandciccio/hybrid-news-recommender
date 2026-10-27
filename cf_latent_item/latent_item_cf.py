from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares


@dataclass
class LatentFactorItemCFConfig:
    factors: int = 50
    regularization: float = 0.01
    iterations: int = 20
    alpha_confidence: float = 20.0
    similarity_metric: str = "dot"  # "dot" oppure "cosine"
    decay: float = 0.8
    max_hist: int = 50


class LatentFactorItemCFRecommender:
    """
    Collaborative Filtering item-based a fattori latenti, con un semplice fallback per il cold-start.

    - Apprende i fattori latenti degli articoli tramite ALS (libreria implicit)
      sulla matrice di interazione utente-item.
    - Assegna un punteggio ai candidati aggregando, nello spazio latente, le
      similarità tra ciascun candidato e gli articoli nella cronologia dell'utente.
    - Gli articoli "freddi" (assenti dal training ALS) ricevono un vettore proxy
      calcolato come centroide del bucket (Categoria, Sottocategoria).
    """

    def __init__(self, config: Optional[LatentFactorItemCFConfig] = None) -> None:
        self.config = config or LatentFactorItemCFConfig()
        self.item_factors: Optional[np.ndarray] = None  # shape: (n_items, k)
        self.item_index_to_id: Optional[np.ndarray] = None  # News_ID originali come stringhe
        self.item_id_to_index: Dict[str, int] = {}
        self.category_centroids: Dict[Tuple[str, str], np.ndarray] = {}
        self.global_centroid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Training e caricamento dei fattori
    # ------------------------------------------------------------------

    def fit_from_matrix(
        self,
        matrix_path: Path,
        user_ids_path: Path,
        item_ids_path: Path,
    ) -> None:
        """
        Allena ALS sulla matrice di interazione implicita utente-item salvata su disco.

        Si aspetta:
        - matrix_path: matrice COO o CSR salvata con scipy.sparse.save_npz
        - user_ids_path: array npy degli ID utente (non usato direttamente qui)
        - item_ids_path: array npy di stringhe News_ID, allineate alle colonne della matrice
        """
        interaction_matrix = sparse.load_npz(str(matrix_path))
        item_ids = np.load(str(item_ids_path), allow_pickle=True)

        self.item_index_to_id = np.asarray(item_ids, dtype=str)
        self.item_id_to_index = {nid: i for i, nid in enumerate(self.item_index_to_id)}

        # random_state fissato per riproducibilità: senza un seed esplicito,
        # l'inizializzazione casuale di ALS produce fattori latenti (e quindi
        # metriche) leggermente diversi a ogni esecuzione.
        model = AlternatingLeastSquares(
            factors=self.config.factors,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            alpha=self.config.alpha_confidence,
            random_state=42,
        )

        # La libreria implicit si aspetta la matrice item-user (trasposta)
        model.fit(interaction_matrix.T.tocsr())
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)

        if self.config.similarity_metric == "cosine":
            norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self.item_factors = self.item_factors / norms

        self.global_centroid = self.item_factors.mean(axis=0)

    # ------------------------------------------------------------------
    # Calcolo dei centroidi per il cold-start
    # ------------------------------------------------------------------

    def build_category_centroids(self, news_tsv_path: Path) -> None:
        """
        Costruisce i centroidi (Categoria, Sottocategoria) come media non pesata
        dei fattori latenti degli articoli "caldi" (già presenti in ALS).

        Gestisce lo schema di `news.tsv` di MIND, le cui colonne sono:
        News_ID, Category, SubCategory, Title, Abstract, URL, Title_Entities, Abstract_Entities
        (lette senza intestazione, quindi si usano gli indici posizionali).
        """
        import pandas as pd

        if self.item_factors is None or self.item_index_to_id is None:
            raise RuntimeError("fit_from_matrix deve essere chiamato prima di build_category_centroids.")

        df = pd.read_csv(news_tsv_path, sep="\t", header=None)
        # Formato MIND: col0=News_ID, col1=Category, col2=SubCategory, ...
        df = df[[0, 1, 2]].dropna()

        nid_to_bucket: Dict[str, Tuple[str, str]] = {}
        for _, row in df.iterrows():
            nid_to_bucket[str(row[0])] = (str(row[1]), str(row[2]))

        bucket_to_vecs: Dict[Tuple[str, str], List[np.ndarray]] = {}
        for idx, nid in enumerate(self.item_index_to_id):
            bucket = nid_to_bucket.get(str(nid))
            if bucket is None:
                continue
            bucket_to_vecs.setdefault(bucket, []).append(self.item_factors[idx])

        centroids: Dict[Tuple[str, str], np.ndarray] = {}
        for bucket, vecs in bucket_to_vecs.items():
            mat = np.stack(vecs, axis=0)
            centroids[bucket] = mat.mean(axis=0)

        self.category_centroids = centroids

    # ------------------------------------------------------------------
    # Funzioni di accesso ai vettori
    # ------------------------------------------------------------------

    def _get_warm_index(self, item_id: str) -> Optional[int]:
        return self.item_id_to_index.get(str(item_id))

    def is_warm(self, item_id: str) -> bool:
        return self._get_warm_index(item_id) is not None

    def get_item_vector(
        self,
        item_id: str,
        bucket: Optional[Tuple[str, str]] = None,
    ) -> np.ndarray:
        """
        Restituisce il vettore latente di item_id, con questa priorità:
        - il fattore ALS, se l'articolo è "caldo" (visto in training);
        - il centroide Categoria/Sottocategoria, se il bucket è fornito e disponibile;
        - il centroide globale, come fallback di ultima istanza.
        """
        if self.item_factors is None:
            raise RuntimeError("Il modello non è stato ancora addestrato (fit_from_matrix).")

        idx = self._get_warm_index(item_id)
        if idx is not None:
            return self.item_factors[idx]

        if bucket is not None and bucket in self.category_centroids:
            return self.category_centroids[bucket]

        if self.global_centroid is not None:
            return self.global_centroid

        return np.zeros(self.item_factors.shape[1], dtype=self.item_factors.dtype)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        history_item_ids: Iterable[str],
        candidate_item_ids: List[str],
        history_buckets: Optional[Dict[str, Tuple[str, str]]] = None,
        candidate_buckets: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> np.ndarray:
        """
        Scoring vettorializzato per una singola impression.

        - history_item_ids: lista ordinata di ID articolo; l'ultimo è il più recente.
        - candidate_item_ids: lista degli ID degli articoli candidati dell'impression.
        - history_buckets / candidate_buckets: mappe opzionali item_id -> (Categoria, Sottocategoria)

        Restituisce:
        - scores: np.ndarray di shape (len(candidate_item_ids),)
        """
        history = list(history_item_ids)
        if not history or not candidate_item_ids:
            return np.zeros(len(candidate_item_ids), dtype=np.float32)

        if self.config.max_hist > 0 and len(history) > self.config.max_hist:
            history = history[-self.config.max_hist :]

        # Costruisce la matrice di cronologia H (L x k), a partire dall'articolo più recente
        hist_vecs: List[np.ndarray] = []
        for h in reversed(history):
            bucket = history_buckets.get(h) if history_buckets is not None else None
            hist_vecs.append(self.get_item_vector(h, bucket=bucket))

        H = np.stack(hist_vecs, axis=0)  # shape: (L, k)
        L = H.shape[0]

        # Costruisce la matrice dei candidati C (M x k)
        cand_vecs: List[np.ndarray] = []
        for c in candidate_item_ids:
            bucket = candidate_buckets.get(c) if candidate_buckets is not None else None
            cand_vecs.append(self.get_item_vector(c, bucket=bucket))

        C = np.stack(cand_vecs, axis=0)  # shape: (M, k)

        # Similarità candidato-storia: M x L, calcolate in un'unica moltiplicazione matriciale
        sims = C @ H.T

        # Pesi di recency: t=0 per l'articolo più recente (peso massimo)
        t = np.arange(L, dtype=np.float32)
        w = np.power(self.config.decay, t)  # shape: (L,)
        w_sum = float(w.sum())
        if w_sum == 0.0:
            return np.zeros(len(candidate_item_ids), dtype=np.float32)

        weighted = sims * w[None, :]
        scores = weighted.sum(axis=1) / w_sum
        return scores.astype(np.float32)

