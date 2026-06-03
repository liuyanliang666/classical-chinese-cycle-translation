from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccnlp.evaluate import chrf_score, exact_match_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction/reference CSV files.")
    parser.add_argument("--input", required=True, help="CSV with prediction and reference columns")
    args = parser.parse_args()

    predictions: list[str] = []
    references: list[str] = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not {"prediction", "reference"}.issubset(reader.fieldnames or set()):
            raise ValueError("CSV must contain prediction and reference columns")
        for row in reader:
            predictions.append(row["prediction"])
            references.append(row["reference"])

    print(f"Exact Match: {exact_match_rate(predictions, references):.4f}")
    print(f"ChrF: {chrf_score(predictions, references):.4f}")


if __name__ == "__main__":
    main()
