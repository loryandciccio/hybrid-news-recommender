#!/usr/bin/env python3
"""Offline synthetic-only hybrid recommender demonstration.

Synthetic embeddings are random numeric features, NOT SBERT vectors. No download,
network request, real-user input, article content, or model checkpoint is used.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "demo" / "generate_synthetic_behaviors.py"
HYBRID = ROOT / "hybrid" / "hybrid_recommender.py"


def prepare_news_and_vectors(folder: Path) -> None:
    import numpy as np

    article_ids = [f"SYN_{topic}{i:02}" for topic in "ABC" for i in range(1, 11)]
    data_dir = folder / "dataset" / "MINDsmall_dev"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "news.tsv").open("w", encoding="utf8") as fh:
        for name in article_ids:
            category = name.split("_")[1][0]
            fh.write(
                "\t".join([
                    name, f"synthetic_{category}", "synthetic_subcategory",
                    f"synthetic title for {name}", "fabricated placeholder summary",
                    "https://example.invalid/synthetic-only", "[]", "[]"
                ]) + "\n"
            )
    vectors = np.random.default_rng(2026).normal(
        size=(len(article_ids), 16)
    ).astype(np.float32)
    path = folder / "content_based_text" / "embeddings"
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "news_embeddings.npy", vectors, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local, synthetic-only hybrid smoke demo.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 20:
        parser.error("--limit must be between 1 and 20")
    with tempfile.TemporaryDirectory(prefix="synthetic-hybrid-") as folder_name:
        folder = Path(folder_name)
        behavior_dir = folder / "dataset" / "MINDsmall_train"
        behavior_dir.mkdir(parents=True)
        # Original fixture generator writes train and dev files; no MIND input.
        generated = folder / "demo" / "generated"
        subprocess.run(
            [sys.executable, "-I", str(GENERATOR), "--output", str(generated)],
            check=True, cwd=folder, timeout=25
        )
        (behavior_dir / "behaviors.tsv").write_bytes(
            (generated / "train_behaviors.tsv").read_bytes()
        )
        dev_dir = folder / "dataset" / "MINDsmall_dev"
        dev_dir.mkdir(parents=True)
        (dev_dir / "behaviors.tsv").write_bytes(
            (generated / "dev_behaviors.tsv").read_bytes()
        )
        prepare_news_and_vectors(folder)
        safe_env = dict(os.environ)
        safe_env["PYTHONDONTWRITEBYTECODE"] = "1"
        print("SYNTHETIC DATA ONLY — random embeddings, not SBERT; metrics are NOT MIND results.", flush=True)
        subprocess.run(
            [sys.executable, "-I", str(HYBRID), "--limit", str(args.limit)],
            check=True, cwd=folder, env=safe_env, timeout=90
        )


if __name__ == "__main__":
    main()
