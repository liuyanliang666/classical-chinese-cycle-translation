from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ParallelExample:
    classical: str
    modern: str
    book: str = ""


_SPACE_RE = re.compile(r"\s+")
_PUNCT_SPACING = {
    " ，": "，",
    " 。": "。",
    " ！": "！",
    " ？": "？",
    " ；": "；",
    " ：": "：",
    " 、": "、",
    "“ ": "“",
    " ”": "”",
}


def clean_text(text: str) -> str:
    """Normalize whitespace and common Chinese punctuation spacing."""
    normalized = _SPACE_RE.sub("", text.strip())
    for old, new in _PUNCT_SPACING.items():
        normalized = normalized.replace(old, new)
    return normalized


def split_long_text(text: str, max_chars: int = 128) -> list[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in "。！？；" or len(current) >= max_chars:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def load_parallel_csv(path: str | Path) -> list[ParallelExample]:
    """Load a CSV with classical/modern columns."""
    examples: list[ParallelExample] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"classical", "modern"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("CSV must contain classical and modern columns")
        for row in reader:
            classical = clean_text(row.get("classical", ""))
            modern = clean_text(row.get("modern", ""))
            if classical and modern:
                examples.append(ParallelExample(classical=classical, modern=modern))
    return examples


def load_niutrans_parallel_dir(
    root_dir: str | Path,
    max_examples: int | None = None,
) -> list[ParallelExample]:
    """Load NiuTrans/Classical-Modern source.txt and target.txt pairs recursively."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"NiuTrans directory not found: {root}")

    examples: list[ParallelExample] = []
    for source_path in sorted(root.rglob("source.txt")):
        target_path = source_path.with_name("target.txt")
        if not target_path.exists():
            continue

        # 取「双语数据」根目录下的第一层目录名作为来源书籍（如 论语 / 史记）。
        rel_parts = source_path.relative_to(root).parts
        book = rel_parts[0] if len(rel_parts) > 1 else ""

        source_lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        target_lines = target_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for classical, modern in zip(source_lines, target_lines):
            classical = clean_text(classical)
            modern = clean_text(modern)
            if not classical or not modern or classical == modern:
                continue
            examples.append(ParallelExample(classical=classical, modern=modern, book=book))
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples


def build_task_examples(examples: Iterable[ParallelExample]) -> list[dict]:
    """Convert parallel pairs into bidirectional seq2seq prompt examples.

    每条记录附带 book / length 元信息，供阶段2按来源书籍、文本长度分组分析。
    训练脚本会用 remove_columns 删除这些额外列，因此不影响训练。
    """
    tasks: list[dict] = []
    for example in examples:
        meta = {"book": example.book, "length": len(example.classical)}
        tasks.append(
            {
                "task": "classical_to_modern",
                "source": f"古文翻今：{example.classical}",
                "target": example.modern,
                **meta,
            }
        )
        tasks.append(
            {
                "task": "modern_to_classical",
                "source": f"今文翻古：{example.modern}",
                "target": example.classical,
                **meta,
            }
        )
    return tasks


def dedup_examples(examples: Iterable[ParallelExample]) -> list[ParallelExample]:
    """Drop exact-duplicate (classical, modern) pairs, keeping first occurrence."""
    seen: set[tuple[str, str]] = set()
    unique: list[ParallelExample] = []
    for example in examples:
        key = (example.classical, example.modern)
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def split_examples(
    examples: Iterable[ParallelExample],
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
) -> dict[str, list[ParallelExample]]:
    """Shuffle (fixed seed) and split parallel pairs into train/validation/test.

    在「句对」层级划分，保证同一句对的两个翻译方向落在同一 split，避免泄漏。
    """
    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    return {
        "test": items[:n_test],
        "validation": items[n_test : n_test + n_val],
        "train": items[n_test + n_val :],
    }


def save_jsonl(records: Iterable[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
