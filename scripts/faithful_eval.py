"""抗幻觉解码评测（服务器端跑，checkpoint 在 /data2/lyl）。

对每个 mode（baseline / copybias / rerank / both）在 test 集上生成译文，
报告翻译质量（BLEU/ChrF/ExactMatch，可选 BERTScore）+ 幻觉指标，并逐句存盘。
逐句 jsonl 含 source/reference 与各 mode 的 hyp，便于拷回本地补算 BERTScore / 幻觉率。

用法：
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/faithful_eval.py \
      --model_dir /data2/lyl/outputs/randeng-bart-niutrans \
      --test_file data/processed/test.jsonl \
      --modes baseline copybias rerank both \
      --no_bertscore --output faithful.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccnlp.evaluate import bertscore_f1, bleu_score, chrf_score, exact_match_rate
from ccnlp.faithful_decode import faithful_translate
from ccnlp.hallucination import aggregate_triples
from ccnlp.inference import TASK_PREFIX, ModelGenerator, TaskType

_DIRECTION_NAME = {"classical_to_modern": "古→今", "modern_to_classical": "今→古"}


def _load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _strip_prefix(source: str, task: str) -> str:
    prefix = TASK_PREFIX[TaskType(task)]
    return source[len(prefix) :] if source.startswith(prefix) else source


def main() -> None:
    parser = argparse.ArgumentParser(description="抗幻觉解码评测")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_file", default="data/processed/test.jsonl")
    parser.add_argument(
        "--modes", nargs="+", default=["baseline", "copybias", "rerank", "both"],
        choices=["baseline", "copybias", "rerank", "both"],
    )
    parser.add_argument("--direction", default="classical_to_modern",
                        choices=["classical_to_modern", "modern_to_classical"],
                        help="幻觉高发于古→今（默认）")
    parser.add_argument("--k", type=int, default=5, help="N-best 候选数（重排用）")
    parser.add_argument("--bias", type=float, default=3.0, help="copy 偏置强度")
    parser.add_argument("--scope", default="content", choices=["numbers", "content", "all"])
    parser.add_argument("--no_roundtrip", action="store_true", help="重排不使用回环一致性打分")
    parser.add_argument("--no_bertscore", action="store_true")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 句（调试用）")
    parser.add_argument("--output", default=None, help="逐句结果 jsonl")
    args = parser.parse_args()

    task = args.direction
    records = [r for r in _load_jsonl(args.test_file) if r.get("task") == task]
    if args.limit:
        records = records[: args.limit]
    sources = [_strip_prefix(r["source"], task) for r in records]
    refs = [r["target"] for r in records]

    generator = ModelGenerator(args.model_dir, num_beams=args.num_beams)
    print(f"已加载：{args.model_dir}（{generator.device}），{_DIRECTION_NAME[task]} {len(records)} 句"
          f"，modes={args.modes}，k={args.k}，bias={args.bias}，scope={args.scope}")

    use_bert = not args.no_bertscore
    per_mode: dict[str, list[str]] = {}
    header = f"{'mode':<10} {'BLEU':>7} {'ChrF':>7} {'EM':>6}" + (f" {'BERT':>7}" if use_bert else "") \
        + f" {'数字幻觉句%':>10} {'数字保留%':>9} {'内容字幻觉':>9}"
    print("\n" + header)
    print("-" * len(header))
    for mode in args.modes:
        hyps = faithful_translate(
            generator, sources, task, mode=mode, k=args.k, bias=args.bias,
            scope=args.scope, use_roundtrip=not args.no_roundtrip,
        )
        per_mode[mode] = hyps
        h = aggregate_triples(list(zip(sources, hyps, refs)))
        line = (f"{mode:<10} {bleu_score(hyps, refs):>7.2f} {chrf_score(hyps, refs):>7.2f} "
                f"{exact_match_rate(hyps, refs):>6.3f}")
        if use_bert:
            line += f" {bertscore_f1(hyps, refs):>7.2f}"
        line += (f" {h['num_halluc_sent_rate']*100:>10.1f} {h['ref_num_recall']*100:>9.1f} "
                 f"{h['unsupported_char_rate']:>9.4f}")
        print(line)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for i, r in enumerate(records):
                row = {"task": task, "source": sources[i], "reference": refs[i],
                       "book": r.get("book", "")}
                for mode in args.modes:
                    row[mode] = per_mode[mode][i]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n逐句结果已存 {out}（{len(records)} 句），可拷回本地补算 BERTScore / 幻觉率。")


if __name__ == "__main__":
    main()
