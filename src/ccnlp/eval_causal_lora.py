"""Evaluate a Qwen Causal LM + LoRA adapter on Lu Xun style transfer data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccnlp.evaluate import bertscore_f1, bleu_score, chrf_score, exact_match_rate
from ccnlp.inference import CausalLoraGenerator


LUXUN_PREFIX = "鲁迅风格化："


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估 Qwen LoRA 鲁迅风格改写模型")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--test_file", default="data/processed/luxun_style/test.jsonl")
    parser.add_argument("--output", default=None, help="可选：保存逐句预测 jsonl")
    parser.add_argument("--task", default="luxun_style", choices=["luxun_style"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--no_bertscore", action="store_true", help="跳过 BERTScore（不联网/省时）")
    parser.add_argument("--no_4bit", action="store_true", help="不用 4-bit 量化加载基础模型")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = _load_jsonl(args.test_file)
    inputs = [_input_text(record) for record in records]
    references = [_reference_text(record) for record in records]

    generator = CausalLoraGenerator(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        load_in_4bit=not args.no_4bit,
    )
    print(f"已加载模型：{args.base_model} + {args.adapter_dir}（设备：{generator.device}）")
    predictions = generator.generate_batch(
        inputs,
        args.task,
        max_source_length=args.max_source_length,
        batch_size=args.batch_size,
    )

    metrics = {
        "n": len(predictions),
        "exact_match": round(exact_match_rate(predictions, references), 4),
        "chrf": chrf_score(predictions, references),
        "bleu": bleu_score(predictions, references),
    }
    if not args.no_bertscore:
        metrics["bertscore_f1"] = bertscore_f1(predictions, references)

    print(f"\n=== 鲁迅风格改写 (n={metrics['n']}) ===")
    print(f"  BLEU        : {metrics['bleu']}")
    print(f"  ChrF        : {metrics['chrf']}")
    print(f"  ExactMatch  : {metrics['exact_match']}")
    if "bertscore_f1" in metrics:
        print(f"  BERTScore F1: {metrics['bertscore_f1']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for record, inp, pred, ref in zip(records, inputs, predictions, references):
                row = {
                    "task": args.task,
                    "input": inp,
                    "prediction": pred,
                    "reference": ref,
                    "book": record.get("book", ""),
                    "title": record.get("title", ""),
                    "source_id": record.get("source_id", ""),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n逐句预测已保存到 {out}（{len(predictions)} 条）。")


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _input_text(record: dict) -> str:
    if "source_plain" in record:
        return str(record["source_plain"]).strip()
    if "source" in record:
        source = str(record["source"]).strip()
        return source[len(LUXUN_PREFIX) :].strip() if source.startswith(LUXUN_PREFIX) else source
    if "context" in record:
        context = str(record["context"])
        marker = "Input:"
        answer_marker = "Answer:"
        if marker in context and answer_marker in context:
            return context.split(marker, 1)[1].split(answer_marker, 1)[0].strip()
    raise KeyError("Cannot find input text in record")


def _reference_text(record: dict) -> str:
    if "target_luxun" in record:
        return str(record["target_luxun"]).strip()
    return str(record["target"]).strip()


if __name__ == "__main__":
    main()
