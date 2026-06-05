from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_train_record(row: dict) -> dict:
    return {
        "task": "modern_to_luxun_style",
        "source": f"鲁迅风格化：{row['source_plain'].strip()}",
        "target": row["target_luxun"].strip(),
        "id": row["id"],
        "title": row.get("title", ""),
        "book": row.get("book", ""),
        "source_id": row.get("source_id", ""),
        "date": row.get("date", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/validation/test JSONL for Lu Xun style transfer.")
    parser.add_argument(
        "--input",
        default="data/processed/luxun_style/luxun_plain_pairs.filtered.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/luxun_style",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    groups = defaultdict(list)
    for row in rows:
        key = row.get("title") or row.get("source_id") or row["id"]
        groups[key].append(row)

    group_items = list(groups.items())
    rng = random.Random(args.seed)
    rng.shuffle(group_items)

    n_total = len(rows)
    n_test_target = int(n_total * args.test_ratio)
    n_val_target = int(n_total * args.val_ratio)

    splits = {"test": [], "validation": [], "train": []}

    for _, group_rows in group_items:
        if len(splits["test"]) < n_test_target:
            splits["test"].extend(group_rows)
        elif len(splits["validation"]) < n_val_target:
            splits["validation"].extend(group_rows)
        else:
            splits["train"].extend(group_rows)

    out_dir = Path(args.output_dir)
    for name, split_rows in splits.items():
        records = [to_train_record(row) for row in split_rows]
        records.sort(key=lambda row: row["id"])
        write_jsonl(records, out_dir / f"{name}.jsonl")

    print(f"Input rows: {n_total}")
    print(f"Groups by title: {len(group_items)}")
    for name in ["train", "validation", "test"]:
        print(f"{name}: {len(splits[name])} rows -> {out_dir / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
