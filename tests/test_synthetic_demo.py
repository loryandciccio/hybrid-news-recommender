"""Offline smoke test for the existing ItemKNN CLI using synthetic input only."""
from __future__ import annotations
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class SyntheticItemKNNSmokeTest(unittest.TestCase):
    def test_generator_is_deterministic_and_itemknn_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthetic-news-") as directory:
            dest = Path(directory)
            generate = [
                sys.executable, "-I", str(ROOT / "demo/generate_synthetic_behaviors.py"),
                "--output", str(dest),
            ]
            proc = subprocess.run(generate, check=True, capture_output=True, text=True, timeout=20)
            first = (dest / "train_behaviors.tsv").read_bytes()
            self.assertIn("SYNTHETIC", proc.stdout)
            subprocess.run(generate, check=True, capture_output=True, text=True, timeout=20)
            self.assertEqual(first, (dest / "train_behaviors.tsv").read_bytes())
            result = subprocess.run([
                sys.executable, "-I", str(ROOT / "cf_itemknn/itemknn_recency_fallback.py"),
                "--train-behaviors", str(dest / "train_behaviors.tsv"),
                "--dev-behaviors", str(dest / "dev_behaviors.tsv"),
                "--limit-dev", "20",
            ], check=True, capture_output=True, text=True, timeout=45)
            self.assertIn("AUC=", result.stdout)
            self.assertIn("nDCG@5=", result.stdout)

    def test_hybrid_runs_with_synthetic_vectors(self) -> None:
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "demo/run_hybrid_synthetic.py"), "--limit", "20"],
            check=True, capture_output=True, text=True, timeout=60
        )
        self.assertIn("SYNTHETIC DATA ONLY", result.stdout)
        self.assertIn("RISULTATI FINALI HYBRID", result.stdout)
        self.assertIn("AUC", result.stdout)


if __name__ == "__main__":
    unittest.main()
