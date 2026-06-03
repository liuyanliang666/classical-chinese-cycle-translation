"""在 test 集上评估微调模型，按翻译方向报告 BLEU / ChrF / ExactMatch / BERTScore。

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m ccnlp.eval_runner \
        --model_dir /data2/lyl/outputs/randeng-bart-niutrans \
        --test_file data/processed/test.jsonl \
        --output /data2/lyl/outputs/randeng-bart-niutrans/test_predictions.jsonl

预测结果会连同 book/length 元信息一起存盘，供任务9 的分组分析直接复用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccnlp.evaluate import bertscore_f1, bleu_score, chrf_score, exact_match_rate
from ccnlp.inference import TASK_PREFIX, ModelGenerator, TaskType

_DIRECTION_NAME = {
    "classical_to_modern": "古→今 (classical→modern)",
    "modern_to_classical": "今→古 (modern→classical)",
}


def _load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _strip_prefix(source: str, task: str) -> str:
    prefix = TASK_PREFIX[TaskType(task)]
    return source[len(prefix) :] if source.startswith(prefix) else source


def _report(label: str, predictions: list[str], references: list[str], use_bertscore: bool) -> dict:
    metrics = {
        "n": len(predictions),
        "exact_match": round(exact_match_rate(predictions, references), 4),
        "chrf": chrf_score(predictions, references),
        "bleu": bleu_score(predictions, references),
    }
    if use_bertscore:
        metrics["bertscore_f1"] = bertscore_f1(predictions, references)

    print(f"\n=== {label} (n={metrics['n']}) ===")
    print(f"  BLEU        : {metrics['bleu']}")
    print(f"  ChrF        : {metrics['chrf']}")
    print(f"  ExactMatch  : {metrics['exact_match']}")
    if use_bertscore:
        print(f"  BERTScore F1: {metrics['bertscore_f1']}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="评估微调模型在 test 集上的翻译质量")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_file", default="data/processed/test.jsonl")
    parser.add_argument("--output", default=None, help="可选：保存逐句预测的 jsonl，供分组分析")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_bertscore", action="store_true", help="跳过 BERTScore（不联网/省时）")
    args = parser.parse_args()

    records = _load_jsonl(args.test_file)
    generator = ModelGenerator(args.model_dir, num_beams=args.num_beams)
    print(f"已加载模型：{args.model_dir}（设备：{generator.device}），测试样本 {len(records)} 条")

    use_bertscore = not args.no_bertscore
    rows: list[dict] = []
    all_preds: list[str] = []
    all_refs: list[str] = []
    summary: dict[str, dict] = {}

    # 按翻译方向分别生成与评估。
    for task in ("classical_to_modern", "modern_to_classical"):
        subset = [r for r in records if r.get("task") == task]
        if not subset:
            continue
        inputs = [_strip_prefix(r["source"], task) for r in subset]
        refs = [r["target"] for r in subset]
        preds = generator.generate_batch(inputs, task, batch_size=args.batch_size)

        summary[task] = _report(_DIRECTION_NAME.get(task, task), preds, refs, use_bertscore)
        all_preds.extend(preds)
        all_refs.extend(refs)
        for r, inp, pred in zip(subset, inputs, preds):
            rows.append(
                {
                    "task": task,
                    "input": inp,
                    "prediction": pred,
                    "reference": r["target"],
                    "book": r.get("book", ""),
                    "length": r.get("length"),
                }
            )

    summary["overall"] = _report("总体 (both directions)", all_preds, all_refs, use_bertscore)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n逐句预测已保存到 {out}（{len(rows)} 条），可用于任务9 分组分析。")


if __name__ == "__main__":
    main()
