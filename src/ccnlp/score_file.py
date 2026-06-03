"""对已保存的预测文件（eval_runner 的 --output）重新计算指标，按方向报告。

用途：当评估时跳过了 BERTScore（如服务器无法下 bert-base-chinese），
可把 test_predictions.jsonl 拷到有网的机器上补算 BERTScore，无需重跑生成。

用法：
    PYTHONPATH=src python -m ccnlp.score_file --input test_predictions.jsonl
    PYTHONPATH=src python -m ccnlp.score_file --input test_predictions.jsonl --no_bertscore
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccnlp.evaluate import bertscore_f1, bleu_score, chrf_score, exact_match_rate

_DIRECTION_NAME = {
    "classical_to_modern": "古→今 (classical→modern)",
    "modern_to_classical": "今→古 (modern→classical)",
}


def _report(label: str, predictions: list[str], references: list[str], use_bertscore: bool) -> None:
    print(f"\n=== {label} (n={len(predictions)}) ===")
    print(f"  BLEU        : {bleu_score(predictions, references)}")
    print(f"  ChrF        : {chrf_score(predictions, references)}")
    print(f"  ExactMatch  : {round(exact_match_rate(predictions, references), 4)}")
    if use_bertscore:
        print(f"  BERTScore F1: {bertscore_f1(predictions, references)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="对预测文件重新计算指标")
    parser.add_argument("--input", required=True, help="eval_runner 输出的 jsonl（含 task/prediction/reference）")
    parser.add_argument("--no_bertscore", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    with Path(args.input).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    use_bertscore = not args.no_bertscore
    all_preds: list[str] = []
    all_refs: list[str] = []
    for task in ("classical_to_modern", "modern_to_classical"):
        subset = [r for r in rows if r.get("task") == task]
        if not subset:
            continue
        preds = [r["prediction"] for r in subset]
        refs = [r["reference"] for r in subset]
        _report(_DIRECTION_NAME.get(task, task), preds, refs, use_bertscore)
        all_preds.extend(preds)
        all_refs.extend(refs)

    _report("总体 (both directions)", all_preds, all_refs, use_bertscore)


if __name__ == "__main__":
    main()
