from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccnlp.preprocess import (
    build_task_examples,
    dedup_examples,
    load_niutrans_parallel_dir,
    load_parallel_csv,
    save_jsonl,
    split_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bidirectional prompt data with train/val/test split.")
    parser.add_argument("--input", required=True, help="CSV file or NiuTrans 双语数据 directory")
    parser.add_argument("--output_dir", default="data/processed", help="Directory for train/validation/test jsonl")
    parser.add_argument(
        "--format",
        choices=["csv", "niutrans"],
        default="csv",
        help="Input data format",
    )
    parser.add_argument("--max_examples", type=int, default=None, help="Optional limit on parallel pairs before split")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Validation fraction of parallel pairs")
    parser.add_argument("--test_ratio", type=float, default=0.01, help="Test fraction of parallel pairs")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducible splits")
    parser.add_argument("--no_dedup", action="store_true", help="Disable exact-duplicate removal")
    args = parser.parse_args()

    if args.format == "niutrans":
        examples = load_niutrans_parallel_dir(args.input, max_examples=args.max_examples)
    else:
        examples = load_parallel_csv(args.input)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]

    n_loaded = len(examples)
    if not args.no_dedup:
        examples = dedup_examples(examples)
    n_unique = len(examples)

    splits = split_examples(examples, args.val_ratio, args.test_ratio, args.seed)

    output_dir = Path(args.output_dir)
    print(f"Loaded {n_loaded} pairs; {n_unique} unique after dedup. Writing to {output_dir}/")
    for name, split_examples_list in splits.items():
        records = build_task_examples(split_examples_list)
        save_jsonl(records, output_dir / f"{name}.jsonl")
        print(f"  {name}.jsonl: {len(split_examples_list)} pairs -> {len(records)} records")


if __name__ == "__main__":
    main()
