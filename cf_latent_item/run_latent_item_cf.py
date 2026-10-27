"""
Pipeline di valutazione da riga di comando per il Collaborative Filtering
item-based a fattori latenti (ALS): allena il modello sulla matrice di
interazione TRAIN, applica il fallback a centroidi di categoria per gli
articoli cold-start e valuta le metriche standard sulle impression del DEV.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from .latent_item_cf import LatentFactorItemCFConfig, LatentFactorItemCFRecommender
except ImportError:
    from latent_item_cf import LatentFactorItemCFConfig, LatentFactorItemCFRecommender


BEHAV_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def parse_impressions(impr: str) -> Tuple[List[str], np.ndarray]:
    parts = str(impr).split()
    cands: List[str] = []
    labels: List[int] = []
    for p in parts:
        try:
            nid, lab = p.rsplit("-", 1)
        except ValueError:
            continue
        cands.append(nid)
        labels.append(1 if lab == "1" else 0)
    return cands, np.asarray(labels, dtype=np.int32)


def parse_history(hist: str) -> List[str]:
    if hist is None or (isinstance(hist, float) and np.isnan(hist)):
        return []
    s = str(hist).strip()
    if not s:
        return []
    return s.split()


def build_train_popularity(train_behav: pd.DataFrame) -> Counter:
    clicked = (
        train_behav["impressions"]
        .dropna()
        .astype(str)
        .str.split()
        .explode()
    )
    clicked = clicked[clicked.str.endswith("-1")].str.rsplit("-", n=1).str[0]
    return Counter(clicked)


def deterministic_order(y_score: np.ndarray, cand_ids: List[str]) -> np.ndarray:
    y_score = np.asarray(y_score, dtype=np.float64)
    cand_ids_arr = np.asarray(cand_ids, dtype=str)
    return np.lexsort((cand_ids_arr, -y_score))


def mrr_first_hit(y_true: np.ndarray, y_score: np.ndarray, cand_ids: List[str]) -> float:
    order = deterministic_order(y_score, cand_ids)
    y_sorted = y_true[order]
    hits = np.where(y_sorted == 1)[0]
    return 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, cand_ids: List[str], k: int) -> float:
    order = deterministic_order(y_score, cand_ids)[:k]
    if len(order) == 0:
        return 0.0
    return float(np.mean(y_true[order]))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, cand_ids: List[str], k: int) -> float:
    denom = int(np.sum(y_true))
    if denom <= 0:
        return 0.0
    order = deterministic_order(y_score, cand_ids)[:k]
    if len(order) == 0:
        return 0.0
    return float(np.sum(y_true[order]) / denom)


def item_novelty(clicks: Counter, pop_total: float, news_id: str) -> float:
    c = clicks.get(news_id, 1)
    p = float(c) / float(pop_total) if pop_total > 0 else 0.0
    if p <= 0.0:
        return 0.0
    return float(-np.log2(p))


def novelty_at_k(clicks: Counter, pop_total: float, y_score: np.ndarray, cand_ids: List[str], k: int) -> float:
    order = deterministic_order(y_score, cand_ids)[:k]
    if len(order) == 0:
        return 0.0
    top_ids = [cand_ids[i] for i in order]
    vals = [item_novelty(clicks, pop_total, nid) for nid in top_ids]
    return float(np.mean(vals)) if vals else 0.0


@dataclass
class EvalConfig:
    k_list: Tuple[int, int] = (5, 10)
    limit_dev: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent factor item-based CF on MIND (impression-level).")
    parser.add_argument(
        "--train-behaviors",
        type=Path,
        default=Path("dataset/MINDsmall_train/behaviors.tsv"),
    )
    parser.add_argument(
        "--dev-behaviors",
        type=Path,
        default=Path("dataset/MINDsmall_dev/behaviors.tsv"),
    )
    parser.add_argument(
        "--news-train",
        type=Path,
        default=Path("dataset/MINDsmall_train/news.tsv"),
    )
    parser.add_argument("--factors", type=int, default=50)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--alpha-conf", type=float, default=20.0)
    parser.add_argument(
        "--similarity-metric",
        type=str,
        default="dot",
        choices=["dot", "cosine"],
    )
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--max-hist", type=int, default=50)
    parser.add_argument("--limit-dev", type=int, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    matrix_path = base_dir / "sparse_user_item_interaction.npz"
    user_ids_path = base_dir / "user_ids.npy"
    item_ids_path = base_dir / "news_ids.npy"

    if not matrix_path.exists() or not user_ids_path.exists() or not item_ids_path.exists():
        raise SystemExit(
            "Interaction matrix artifacts not found in cf_latent_item/. "
            "Run cf_latent_item/create_interaction_matrix.py first."
        )

    train_behav = pd.read_csv(args.train_behaviors, sep="\t", header=None, names=BEHAV_COLS)
    dev_behav = pd.read_csv(args.dev_behaviors, sep="\t", header=None, names=BEHAV_COLS)

    pop_clicks = build_train_popularity(train_behav)
    pop_total = float(sum(pop_clicks.values())) if len(pop_clicks) else 1.0

    config = LatentFactorItemCFConfig(
        factors=args.factors,
        regularization=args.reg,
        iterations=args.iterations,
        alpha_confidence=args.alpha_conf,
        similarity_metric=args.similarity_metric,
        decay=args.decay,
        max_hist=args.max_hist,
    )
    model = LatentFactorItemCFRecommender(config=config)
    model.fit_from_matrix(matrix_path, user_ids_path, item_ids_path)

    # Costruisce i centroidi (Categoria, Sottocategoria) per gli articoli "caldi"
    model.build_category_centroids(args.news_train)

    # Precostruisce i bucket per tutti i News_ID del news.tsv di training
    news_df = pd.read_csv(args.news_train, sep="\t", header=None)
    # col0=News_ID, col1=Category, col2=SubCategory
    id_to_bucket = {
        str(row[0]): (str(row[1]), str(row[2]))
        for _, row in news_df.iterrows()
    }

    eval_conf = EvalConfig(limit_dev=args.limit_dev)
    metrics = {
        "auc": [],
        "mrr": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "p@5": [],
        "r@5": [],
        "novelty@5": [],
        "novelty@10": [],
    }

    n_rows = len(dev_behav)
    if eval_conf.limit_dev is not None:
        n_rows = min(n_rows, eval_conf.limit_dev)

    for idx in range(n_rows):
        row = dev_behav.iloc[idx]
        history_ids = parse_history(row.get("history", ""))
        cand_ids, y_true = parse_impressions(row.get("impressions", ""))

        if len(cand_ids) == 0 or int(y_true.sum()) == 0:
            continue

        history_buckets = {nid: id_to_bucket.get(nid) for nid in history_ids}
        cand_buckets = {nid: id_to_bucket.get(nid) for nid in cand_ids}

        # Score CF latente dai fattori ALS degli articoli
        scores_cf = model.score_candidates(
            history_item_ids=history_ids,
            candidate_item_ids=cand_ids,
            history_buckets=history_buckets,
            candidate_buckets=cand_buckets,
        )

        # Combina con la popolarità: 0.8 * latente + 0.2 * popolarità normalizzata (radice quadrata)
        pop_arr = np.asarray([pop_clicks.get(nid, 0) for nid in cand_ids], dtype=np.float64)
        sqrt_pop = np.sqrt(pop_arr)
        max_sqrt = float(sqrt_pop.max()) if sqrt_pop.size > 0 else 0.0
        if max_sqrt > 0.0:
            sqrt_pop = sqrt_pop / max_sqrt
        scores = 0.8 * scores_cf + 0.2 * sqrt_pop.astype(scores_cf.dtype)

        # AUC (solo se sono presenti entrambe le classi)
        if len(np.unique(y_true)) == 2:
            try:
                metrics["auc"].append(roc_auc_score(y_true, scores))
            except Exception:
                pass

        metrics["mrr"].append(mrr_first_hit(y_true, scores, cand_ids))
        metrics["ndcg@5"].append(_ndcg_at_k(y_true, scores, cand_ids, k=5))
        metrics["ndcg@10"].append(_ndcg_at_k(y_true, scores, cand_ids, k=10))
        metrics["p@5"].append(precision_at_k(y_true, scores, cand_ids, k=5))
        metrics["r@5"].append(recall_at_k(y_true, scores, cand_ids, k=5))
        metrics["novelty@5"].append(novelty_at_k(pop_clicks, pop_total, scores, cand_ids, k=5))
        metrics["novelty@10"].append(novelty_at_k(pop_clicks, pop_total, scores, cand_ids, k=10))

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    print("Latent factor item-based CF (impression-level) metrics on DEV:")
    for key in ["auc", "mrr", "ndcg@5", "ndcg@10", "p@5", "r@5", "novelty@5", "novelty@10"]:
        print(f"{key}: {_mean(metrics[key]):.6f}")


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, cand_ids: List[str], k: int) -> float:
    order = deterministic_order(y_score, cand_ids)[:k]
    if len(order) == 0:
        return 0.0
    gains = y_true[order]

    def dcg(arr: np.ndarray) -> float:
        return float(
            np.sum(arr / np.log2(np.arange(2, len(arr) + 2, dtype=np.float64)))
        )

    ideal = np.sort(y_true)[::-1][:k]
    idcg = dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return dcg(gains) / idcg


if __name__ == "__main__":
    main()

