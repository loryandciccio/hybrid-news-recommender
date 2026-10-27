#!/usr/bin/env python3
"""Generate synthetic MIND-shaped behavior rows; NO real articles, users, or requests."""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path

def build_rows(count: int, seed: int, phase: str) -> list[list[str]]:
    rng = random.Random(seed)
    topics = [
        [f"SYN_A{i:02}" for i in range(1, 11)],
        [f"SYN_B{i:02}" for i in range(1, 11)],
        [f"SYN_C{i:02}" for i in range(1, 11)],
    ]
    rows = []
    for row_id in range(1, count + 1):
        topic = (row_id - 1) % len(topics)
        pool = topics[topic]
        history = rng.sample(pool, 5)
        clicked = rng.choice(pool)
        negatives = rng.sample(topics[(topic + 1) % 3] + topics[(topic + 2) % 3], 4)
        items = [(clicked, 1)] + [(n, 0) for n in negatives]
        rng.shuffle(items)
        rows.append([
            str(row_id),
            f"SYN_USER_{(row_id - 1) % 12:03}",
            "11/14/2019 10:00:00 AM" if phase == "train" else "11/15/2019 10:00:00 AM",
            " ".join(history),
            " ".join(f"{article}-{label}" for article, label in items),
        ])
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description="Write deterministic, fully synthetic behavior fixtures.")
    parser.add_argument("--output", type=Path, default=Path("demo/generated"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, n, seed, phase in [
        ("train_behaviors.tsv", 60, 42, "train"),
        ("dev_behaviors.tsv", 20, 43, "dev"),
    ]:
        path = args.output / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="\t", lineterminator="\n").writerows(build_rows(n, seed, phase))
        print(f"{path}: {n} SYNTHETIC impressions")

if __name__ == "__main__":
    main()
