#!/usr/bin/env python3
"""
HYBRID RECOMMENDER - combina Baseline + Collaborative Filtering + Content-Based Filtering.

Il modello finale del progetto: combina TRE strategie complementari in un'unica
funzione di ranking pesata:
1. BASELINE: popolarità globale degli articoli (click totali in training).
2. COLLABORATIVE FILTERING: similarità ItemKNN basata sui co-click, pesata per recency.
3. CONTENT-BASED FILTERING: embedding semantici SBERT + metadata categoriali.

I pesi ottimali (default di questo script) sono stati individuati tramite grid search
sistematica (si veda grid_search_weights.py e weight_optimization.md).
Valuta tutte le metriche standard: AUC, MRR, nDCG@5/10, P@5, R@5, Novelty@5/10.

Per eseguire lo script, posizionarsi nella root del progetto e lanciare:
python3 hybrid/hybrid_recommender.py --limit 2000
"""

import os
import argparse
import numpy as np
import polars as pl
import scipy.sparse as sparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set
import math


# --------------------------
# Funzioni di supporto (implementazione locale, solo NumPy)
# --------------------------

def cosine_similarity_1d(a, b):
    """Similarità del coseno tra due vettori 1D, senza dipendenze esterne."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def dcg_at_k(y_true, y_score, k):
    """Calcola il DCG@k: ordina gli item per score predetto e applica lo sconto posizionale log2."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="mergesort")[:k]
    gains = y_true[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))

def ideal_dcg_at_k(y_true, k):
    """Calcola l'IDCG@k: DCG ottenuto ordinando gli item per la label vera (ranking ideale)."""
    y_true = np.asarray(y_true)
    ideal = np.sort(y_true)[::-1][:k]
    discounts = np.log2(np.arange(2, len(ideal) + 2))
    return float(np.sum(ideal / discounts))

def ndcg_at_k(y_true, y_score, k):
    """Calcola nDCG@k = DCG@k / IDCG@k (normalizzato rispetto al ranking ideale)."""
    idcg = ideal_dcg_at_k(y_true, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(y_true, y_score, k) / idcg


# Percorsi di default (relativi alla root del progetto)
DEFAULT_TRAIN_BEHAVIORS = "dataset/MINDsmall_train/behaviors.tsv"
DEFAULT_DEV_BEHAVIORS = "dataset/MINDsmall_dev/behaviors.tsv"
DEFAULT_NEWS = "dataset/MINDsmall_dev/news.tsv"
DEFAULT_EMBEDDINGS = "content_based_text/embeddings/news_embeddings.npy"


class HybridRecommender:
    """
    Combina Baseline (popolarità), CF (ItemKNN) e CBF (semantico + topical).

    Pesi:
    - baseline_weight: peso del segnale di popolarità globale
    - cf_weight: peso del segnale collaborativo (co-click)
    - cbf_weight: peso del segnale content-based (semantico + topical)
    """
    
    def __init__(
        self,
        baseline_weight: float = 0.0,
        cf_weight: float = 0.03,
        cbf_weight: float = 0.97,
        cf_decay: float = 0.8,
        cbf_decay: float = 0.95,
        cbf_text_weight: float = 0.7,
        cbf_topic_weight: float = 0.3,
        k_neighbors: int = 200,
        shrinkage: int = 50
    ):
        """
        Inizializza il recommender ibrido con pesi e iperparametri configurabili.

        Args:
            baseline_weight: peso della baseline di popolarità [0, 1]
            cf_weight: peso del Collaborative Filtering [0, 1]
            cbf_weight: peso del Content-Based Filtering [0, 1]
            cf_decay: decadimento temporale della history per il CF (0-1, più vicino a 1 = memoria più lunga)
            cbf_decay: decadimento temporale della history per il CBF (0-1)
            cbf_text_weight: peso del segnale semantico all'interno del CBF (0-1)
            cbf_topic_weight: peso del segnale topical all'interno del CBF (0-1)
            k_neighbors: numero di vicini per l'ItemKNN
            shrinkage: shrinkage Bayesiano per l'ItemKNN
        """
        # Pesi dei tre segnali principali, rinormalizzati a somma 1.
        # Guardia esplicita contro la divisione per zero nel caso limite
        # (raro ma possibile da riga di comando) in cui tutti i pesi siano nulli.
        total_weight = baseline_weight + cf_weight + cbf_weight
        if total_weight <= 0:
            raise ValueError("La somma di baseline_weight, cf_weight e cbf_weight deve essere > 0")
        self.baseline_weight = baseline_weight / total_weight
        self.cf_weight = cf_weight / total_weight
        self.cbf_weight = cbf_weight / total_weight

        # Pesi interni al CBF (segnale semantico vs topical), stessa guardia
        cbf_total = cbf_text_weight + cbf_topic_weight
        if cbf_total <= 0:
            raise ValueError("La somma di cbf_text_weight e cbf_topic_weight deve essere > 0")
        self.cbf_text_weight = cbf_text_weight / cbf_total
        self.cbf_topic_weight = cbf_topic_weight / cbf_total
        
        # Parametri di decadimento temporale
        self.cf_decay = cf_decay
        self.cbf_decay = cbf_decay

        # Parametri dell'ItemKNN
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage

        # Strutture dati caricate da load_data()
        self.news_id_to_idx: Dict[str, int] = {}
        self.news_metadata: Dict[str, str] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.item_popularity: Dict[str, int] = {}
        self.pop_total: float = 0.0

        # CF: matrice di similarità item-item
        self.itemknn_similarity: Optional[np.ndarray] = None
        self.news_ids_array: Optional[np.ndarray] = None
        self.news_id_to_news_idx: Dict[str, int] = {}
        
    def load_data(self, emb_path: str, news_path: str, train_behav_path: str):
        """Carica tutti i dati necessari: embedding, metadata, popolarità, e costruisce la matrice ItemKNN."""
        print("=" * 80)
        print("CARICAMENTO DATI PER L'HYBRID RECOMMENDER")
        print("=" * 80)

        # Carica gli embedding SBERT precalcolati
        print(f"[1/4] Caricamento embedding SBERT da {emb_path}...")
        self.embeddings = np.load(emb_path)

        # Carica i metadata degli articoli (categoria usata dal segnale topical del CBF)
        print(f"[2/4] Caricamento metadata news da {news_path}...")
        news_cols = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]
        df_news = pl.read_csv(news_path, separator="\t", has_header=False, new_columns=news_cols, quote_char=None)
        for i, row in enumerate(df_news.iter_rows(named=True)):
            nid = row["News_ID"]
            self.news_id_to_idx[nid] = i
            self.news_metadata[nid] = row["Category"]

        # Calcola la popolarità esclusivamente dai dati di training (nessun leakage dal DEV)
        print(f"[3/4] Calcolo popolarità da {train_behav_path}...")
        cols = ["imp_id", "user_id", "time", "history", "impressions"]
        behav = pl.read_csv(train_behav_path, separator="\t", has_header=False, new_columns=cols, quote_char=None)

        # Conteggio dei click storici (cronologia utente) per articolo
        history_clicks = (
            behav.select("history")
            .filter(pl.col("history").is_not_null())
            .with_columns(pl.col("history").str.split(" "))
            .explode("history")
            .rename({"history": "article_id"})
        )
        self.item_popularity = Counter(history_clicks["article_id"].to_list())
        self.pop_total = float(sum(self.item_popularity.values())) if self.item_popularity else 1.0

        # Costruisce la matrice di similarità ItemKNN (co-click)
        print(f"[4/4] Costruzione della matrice di similarità ItemKNN (co-click)...")
        self._build_itemknn_matrix(train_behav_path)

        print("Caricamento dati completato!")
        print(f"  - Shape embedding: {self.embeddings.shape}")
        print(f"  - Articoli: {len(self.news_id_to_idx)}")
        print(f"  - Voci di popolarità: {len(self.item_popularity)}")
        print(f"  - Matrice di similarità ItemKNN: {self.itemknn_similarity.shape if self.itemknn_similarity is not None else 'None'}")
        print("=" * 80)

    def _build_itemknn_matrix(self, train_behav_path: str):
        """Costruisce la matrice sparsa di similarità item-item basata sui co-click (approccio ItemKNN)."""
        cols = ["imp_id", "user_id", "time", "history", "impressions"]
        behav = pl.read_csv(train_behav_path, separator="\t", has_header=False, new_columns=cols, quote_char=None)

        # Universo di articoli considerato: quelli presenti nel file di metadata caricato in load_data
        news_ids_set = set(self.news_id_to_idx.keys())

        # Conta i co-click: per ogni cronologia utente, ogni coppia di articoli letti insieme
        # viene considerata "co-cliccata" una volta (relazione simmetrica).
        co_clicks = defaultdict(Counter)
        item_counts = Counter()

        for row in behav.iter_rows(named=True):
            history = str(row["history"]).split() if row["history"] else []
            history = [nid for nid in history if nid in news_ids_set]

            if len(history) < 2:
                continue

            # Conta le occorrenze di ciascun articolo
            for nid in history:
                item_counts[nid] += 1

            # Conta i co-click per ogni coppia di articoli nella stessa cronologia
            for i, nid1 in enumerate(history):
                for nid2 in history[i+1:]:
                    co_clicks[nid1][nid2] += 1
                    co_clicks[nid2][nid1] += 1

        # Mappa ordinata news_id -> indice di matrice
        sorted_news_ids = sorted(news_ids_set)
        self.news_ids_array = np.array(sorted_news_ids)
        for idx, nid in enumerate(sorted_news_ids):
            self.news_id_to_news_idx[nid] = idx

        # Costruisce la matrice sparsa di similarità (coseno con shrinkage Bayesiano)
        n_items = len(sorted_news_ids)
        similarity_data = []
        row_indices = []
        col_indices = []

        for i, nid1 in enumerate(sorted_news_ids):
            if i % 500 == 0:
                print(f"  Costruzione matrice: {i}/{n_items}...")

            neighbors_count = co_clicks.get(nid1, Counter())

            # Mantiene solo i top-k vicini più simili (grafo di similarità sparso)
            top_neighbors = neighbors_count.most_common(self.k_neighbors)

            for nid2, count in top_neighbors:
                j = self.news_id_to_news_idx[nid2]

                # Similarità del coseno con shrinkage Bayesiano: penalizza le
                # co-occorrenze con poco supporto statistico (pochi co-click osservati)
                c1 = item_counts.get(nid1, 1)
                c2 = item_counts.get(nid2, 1)

                shrunk_sim = (count + self.shrinkage) / (math.sqrt(c1 * c2) + self.shrinkage)

                if shrunk_sim > 0:
                    similarity_data.append(shrunk_sim)
                    row_indices.append(i)
                    col_indices.append(j)

        # Costruisce la matrice sparsa finale
        self.itemknn_similarity = sparse.csr_matrix(
            (similarity_data, (row_indices, col_indices)),
            shape=(n_items, n_items)
        )
    
    def _norm(self, s: np.ndarray) -> np.ndarray:
        """Normalizzazione Min-Max in [0, 1], calcolata sul singolo candidate set."""
        m, M = s.min(), s.max()
        return (s - m) / (M - m) if M > m else s

    def get_item_novelty(self, news_id: str) -> float:
        """Novelty come self-information: -log2(P(item)), P stimata dal training."""
        c = self.item_popularity.get(news_id, 1)
        p = float(c) / self.pop_total
        return -math.log2(p) if p > 0 else 0.0

    def _get_baseline_scores(self, candidates: List[str]) -> np.ndarray:
        """Segnale baseline: popolarità globale nel training."""
        scores = np.zeros(len(candidates))
        for i, nid in enumerate(candidates):
            scores[i] = float(self.item_popularity.get(nid, 0)) / self.pop_total if self.pop_total > 0 else 0.0
        return scores

    def _get_cf_scores(self, history: List[str], candidates: List[str]) -> np.ndarray:
        """Segnale Collaborative Filtering: ItemKNN pesato per recency."""
        n = len(candidates)
        cf_scores = np.zeros(n)

        if self.itemknn_similarity is None:
            return cf_scores

        # Applica il decadimento temporale alla cronologia (il click più recente pesa di più)
        w_hist = np.array([self.cf_decay ** i for i in range(len(history))][::-1])

        # Costruisce il "profilo CF" dell'utente sommando i vicini ItemKNN di ogni
        # articolo in cronologia, pesati per recency
        user_cf_vector = np.zeros(self.itemknn_similarity.shape[0])

        for h_item, weight in zip(history, w_hist):
            if h_item in self.news_id_to_news_idx:
                item_idx = self.news_id_to_news_idx[h_item]
                # Estrae gli articoli simili a questo item storico
                similar = self.itemknn_similarity[item_idx].toarray().flatten()
                user_cf_vector += similar * weight

        # Normalizza il profilo CF rispetto al suo massimo
        if np.max(user_cf_vector) > 0:
            user_cf_vector = user_cf_vector / np.max(user_cf_vector)

        # Assegna il punteggio ai candidati
        for i, nid in enumerate(candidates):
            if nid in self.news_id_to_news_idx:
                item_idx = self.news_id_to_news_idx[nid]
                cf_scores[i] = user_cf_vector[item_idx]

        return cf_scores

    def _get_cbf_scores(self, history: List[str], candidates: List[str]) -> np.ndarray:
        """Segnale Content-Based Filtering: semantico (SBERT) + topical (categoria)."""
        n = len(candidates)
        cbf_scores = np.zeros(n)

        # Applica il decadimento temporale alla cronologia
        w_hist = np.array([self.cbf_decay ** i for i in range(len(history))][::-1])

        # Segnale semantico: costruisce l'embedding utente come somma pesata per recency
        u_vecs = []
        for h_item, weight in zip(history, w_hist):
            if h_item in self.news_id_to_idx:
                embedding = self.embeddings[self.news_id_to_idx[h_item]]
                u_vecs.append(embedding * weight)

        u_vec_semantic = np.sum(u_vecs, axis=0).reshape(1, -1) if u_vecs else None
        if u_vec_semantic is not None:
            norm = np.linalg.norm(u_vec_semantic)
            if norm > 0:
                u_vec_semantic /= norm

        # Segnale topical: costruisce le preferenze di categoria dell'utente, pesate per recency
        u_cat_pref = defaultdict(float)
        for h_item, weight in zip(history, w_hist):
            cat = self.news_metadata.get(h_item)
            if cat:
                u_cat_pref[cat] += weight

        # Assegna il punteggio ai candidati su entrambi i segnali
        s_text = np.zeros(n)
        s_topic = np.zeros(n)

        for i, nid in enumerate(candidates):
            # Similarità semantica
            if u_vec_semantic is not None and nid in self.news_id_to_idx:
                c_vec = self.embeddings[self.news_id_to_idx[nid]].ravel()
                s_text[i] = cosine_similarity_1d(u_vec_semantic.ravel(), c_vec)

            # Corrispondenza tematica (categoria)
            cat = self.news_metadata.get(nid)
            if cat:
                s_topic[i] = u_cat_pref.get(cat, 0.0)

        # Normalizza e combina i due sotto-segnali (70% semantico, 30% topical di default)
        s_text_norm = self._norm(s_text)
        s_topic_norm = self._norm(s_topic)

        cbf_scores = (self.cbf_text_weight * s_text_norm +
                      self.cbf_topic_weight * s_topic_norm)

        return cbf_scores

    def get_scores(self, history: List[str], candidates: List[str]) -> np.ndarray:
        """Calcola lo score finale combinando i tre segnali (baseline, CF, CBF)."""
        # Calcola i segnali individuali
        baseline_scores = self._get_baseline_scores(candidates)
        cf_scores = self._get_cf_scores(history, candidates)
        cbf_scores = self._get_cbf_scores(history, candidates)

        # Normalizza ciascun segnale indipendentemente, all'interno del candidate set
        baseline_norm = self._norm(baseline_scores)
        cf_norm = self._norm(cf_scores)
        cbf_norm = self._norm(cbf_scores)

        # Combina i segnali con i pesi configurati
        final_scores = (
            self.baseline_weight * baseline_norm +
            self.cf_weight * cf_norm +
            self.cbf_weight * cbf_norm
        )

        return final_scores

    def evaluate(self, dev_path: str, limit: int = 2000, offset: int = 0):
        """Valuta l'hybrid recommender sul set DEV.

        Nota metodologica: per limiti di efficienza (si veda _get_cf_scores,
        che materializza una riga densa della matrice sparsa ItemKNN per ogni
        articolo di cronologia), la valutazione di default copre le prime
        `limit` impression del DEV set (2000 di default, su un totale di
        73.152) anziché l'intero set, a differenza dei moduli standalone
        (baseline, ItemKNN, CBF) che valutano sull'intero DEV. I risultati
        riportati nella documentazione del progetto usano `limit=2000`,
        `offset=0` (comportamento di default, invariato).

        Il parametro `offset` (default 0, quindi nessun cambiamento rispetto
        al comportamento originale) permette di valutare su una porzione
        DIVERSA e disgiunta del DEV set rispetto a quella usata per la grid
        search dei pesi (`grid_search_weights.py`/`parameter_search.py`, che
        usano sempre le impression a partire dall'indice 0). Serve per
        verificare che i pesi scelti tramite grid search non siano
        overfittati alle prime impression del DEV set su cui sono stati
        selezionati (si veda REVISIONE_PROGETTO.md, sezione sul data split).
        """
        print("\n" + "=" * 80)
        print(f"VALUTAZIONE HYBRID RECOMMENDER (Limit: {limit})")
        print("=" * 80)
        print(f"Configurazione:")
        print(f"  Peso baseline: {self.baseline_weight:.3f}")
        print(f"  Peso CF: {self.cf_weight:.3f}")
        print(f"  Peso CBF: {self.cbf_weight:.3f}")
        print(f"    - Testo: {self.cbf_text_weight:.3f}")
        print(f"    - Categoria: {self.cbf_topic_weight:.3f}")
        print(f"  Decay CF: {self.cf_decay:.2f}")
        print(f"  Decay CBF: {self.cbf_decay:.2f}")
        print("=" * 80 + "\n")

        cols = ["imp_id", "user_id", "time", "history", "impressions"]
        df_full = pl.read_csv(dev_path, separator="\t", has_header=False, new_columns=cols, quote_char=None)
        df = df_full.slice(offset, limit) if offset else df_full.head(limit)

        metrics = defaultdict(list)

        for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Valutazione"):
            h = str(row["history"]).split() if row["history"] else []
            imps = str(row["impressions"]).split() if row["impressions"] else []

            if not h or not imps:
                continue

            cands = [imp.split("-")[0] for imp in imps]
            y_true = np.array([int(imp.split("-")[1]) for imp in imps])

            # Salta le impression senza alcun negativo o alcun positivo: l'AUC
            # non è definita in questi casi (una sola classe presente)
            if sum(y_true) == 0 or sum(y_true) == len(y_true):
                continue

            # Calcola gli score dell'impression corrente
            y_score = self.get_scores(h, cands)

            # AUC
            try:
                metrics["AUC"].append(roc_auc_score(y_true, y_score))
            except Exception:
                pass

            # nDCG
            metrics["nDCG@5"].append(ndcg_at_k(y_true, y_score, k=5))
            metrics["nDCG@10"].append(ndcg_at_k(y_true, y_score, k=10))

            # Metriche basate sul rank. Tie-break deterministico: a parità di
            # score, ordina per news_id crescente (coerente con gli altri
            # moduli del progetto: baseline e ItemKNN).
            order = np.lexsort((cands, -y_score))
            y_sorted = y_true[order]
            cand_sorted = [cands[i] for i in order]
            
            # MRR: reciproco della posizione del primo click in classifica
            hits = np.where(y_sorted == 1)[0]
            metrics["MRR"].append(1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0)

            # P@5, R@5
            hits_at_5 = np.sum(y_sorted[:5])
            metrics["P@5"].append(hits_at_5 / 5.0)
            metrics["R@5"].append(hits_at_5 / np.sum(y_true))

            # Novelty (self-information media sui primi K raccomandati)
            metrics["Novelty@5"].append(np.mean([self.get_item_novelty(nid) for nid in cand_sorted[:5]]))
            metrics["Novelty@10"].append(np.mean([self.get_item_novelty(nid) for nid in cand_sorted[:10]]))

        # Stampa i risultati aggregati
        final_results = {k: float(np.mean(v)) for k, v in metrics.items()}

        print("\n" + "=" * 80)
        print("RISULTATI FINALI HYBRID")
        print("=" * 80)
        for k in ["AUC", "MRR", "nDCG@5", "nDCG@10", "P@5", "R@5", "Novelty@5", "Novelty@10"]:
            if k in final_results:
                print(f"{k:12s}: {final_results[k]:.4f}")
        print("=" * 80)
        
        return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Recommender: combina Baseline + CF + CBF")
    parser.add_argument("--limit", type=int, default=2000, help="Numero di impression del DEV set da valutare")
    parser.add_argument("--offset", type=int, default=0, help="Indice di impression da cui iniziare la valutazione (per verifiche su una porzione del DEV disgiunta da quella usata in fase di tuning)")
    parser.add_argument("--baseline-weight", type=float, default=0.0, help="Peso della baseline (popolarità)")
    parser.add_argument("--cf-weight", type=float, default=0.03, help="Peso del Collaborative Filtering")
    parser.add_argument("--cbf-weight", type=float, default=0.97, help="Peso del Content-Based Filtering")
    parser.add_argument("--cf-decay", type=float, default=0.8, help="Decadimento temporale per il CF")
    parser.add_argument("--cbf-decay", type=float, default=0.95, help="Decadimento temporale per il CBF")
    parser.add_argument("--cbf-text-weight", type=float, default=0.7, help="Peso del segnale semantico nel CBF")
    parser.add_argument("--cbf-topic-weight", type=float, default=0.3, help="Peso del segnale topical nel CBF")
    parser.add_argument("--k-neighbors", type=int, default=200, help="Numero di vicini per l'ItemKNN")
    parser.add_argument("--shrinkage", type=int, default=50, help="Shrinkage Bayesiano per l'ItemKNN")
    
    args = parser.parse_args()
    
    hybrid = HybridRecommender(
        baseline_weight=args.baseline_weight,
        cf_weight=args.cf_weight,
        cbf_weight=args.cbf_weight,
        cf_decay=args.cf_decay,
        cbf_decay=args.cbf_decay,
        cbf_text_weight=args.cbf_text_weight,
        cbf_topic_weight=args.cbf_topic_weight,
        k_neighbors=args.k_neighbors,
        shrinkage=args.shrinkage
    )
    
    hybrid.load_data(
        DEFAULT_EMBEDDINGS,
        DEFAULT_NEWS,
        DEFAULT_TRAIN_BEHAVIORS
    )
    
    hybrid.evaluate(DEFAULT_DEV_BEHAVIORS, limit=args.limit, offset=args.offset)
