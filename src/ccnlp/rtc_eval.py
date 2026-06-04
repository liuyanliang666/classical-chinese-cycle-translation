"""回环一致性（RTC）评估：A→B→A' 双向回环，三层一致性 + 分组分析。

RTC = 0.2×字面(ChrF/编辑距离) + 0.6×语义(BERTScore) + 0.2×结构(长度比/句数)

两种用法：
  1) 服务器（有 GPU、无外网）——生成回环文本并存盘，算字面+结构，跳过语义：
     CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m ccnlp.rtc_eval \
         --model_dir /data2/lyl/outputs/randeng-bart-niutrans \
         --test_file data/processed/test.jsonl \
         --output /data2/lyl/outputs/randeng-bart-niutrans/rtc_roundtrip.jsonl \
         --no_bertscore

  2) 本地 Mac（有外网）——读回环文本，补算 BERTScore 与完整 RTC + 分组：
     PYTHONPATH=src python -m ccnlp.rtc_eval --from_file rtc_roundtrip.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ccnlp.evaluate import (
    bertscore_f1_list,
    function_word_ratio,
    literal_consistency,
    rtc_score,
    structural_consistency,
)
from ccnlp.inference import TASK_PREFIX, TaskType

_DIRECTION_NAME = {"c2m2c": "古→今→古", "m2c2m": "今→古→今"}


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _strip_prefix(source: str, task: str) -> str:
    prefix = TASK_PREFIX[TaskType(task)]
    return source[len(prefix) :] if source.startswith(prefix) else source


def _length_bucket(length: int) -> str:
    if length < 20:
        return "短句(<20)"
    if length <= 50:
        return "中句(20-50)"
    return "长句(>50)"


def build_roundtrips(model_dir: str, test_file: str, batch_size: int, num_beams: int) -> list[dict]:
    """加载模型，对每个句对做双向回环，返回带 original/mid/roundtrip 的行。"""
    from ccnlp.inference import ModelGenerator

    records = _load_jsonl(test_file)
    # 从 classical_to_modern 记录还原唯一句对（同时拿到 classical/modern/book）。
    pairs = []
    for r in records:
        if r.get("task") == "classical_to_modern":
            pairs.append(
                {
                    "classical": _strip_prefix(r["source"], "classical_to_modern"),
                    "modern": r["target"],
                    "book": r.get("book", ""),
                }
            )

    generator = ModelGenerator(model_dir, num_beams=num_beams)
    print(f"已加载模型：{model_dir}（设备：{generator.device}），句对 {len(pairs)} 个")

    classicals = [p["classical"] for p in pairs]
    moderns = [p["modern"] for p in pairs]

    # 古→今→古
    c2m = generator.generate_batch(classicals, "classical_to_modern", batch_size=batch_size)
    c2m2c = generator.generate_batch(c2m, "modern_to_classical", batch_size=batch_size)
    # 今→古→今
    m2c = generator.generate_batch(moderns, "modern_to_classical", batch_size=batch_size)
    m2c2m = generator.generate_batch(m2c, "classical_to_modern", batch_size=batch_size)

    rows: list[dict] = []
    for i, p in enumerate(pairs):
        rows.append(
            {
                "direction": "c2m2c",
                "book": p["book"],
                "classical": p["classical"],          # 该句对的古文（用于虚词/长度分组）
                "original": p["classical"],
                "mid": c2m[i],
                "roundtrip": c2m2c[i],
            }
        )
        rows.append(
            {
                "direction": "m2c2m",
                "book": p["book"],
                "classical": p["classical"],
                "original": p["modern"],
                "mid": m2c[i],
                "roundtrip": m2c2m[i],
            }
        )
    return rows


def score_rows(rows: list[dict], use_bertscore: bool) -> list[dict]:
    """对每行计算字面/结构/(语义)/RTC，写回行内。"""
    for row in rows:
        a, ap = row["original"], row["roundtrip"]
        row["literal"] = round(literal_consistency(a, ap), 4)
        row["structural"] = round(structural_consistency(a, ap), 4)
        row["func_ratio"] = round(function_word_ratio(row["classical"]), 4)
        row["length"] = len(row["classical"])

    if use_bertscore:
        originals = [r["original"] for r in rows]
        roundtrips = [r["roundtrip"] for r in rows]
        semantics = bertscore_f1_list(roundtrips, originals)
        for row, sem in zip(rows, semantics):
            row["semantic"] = round(sem, 4)
            row["rtc"] = round(rtc_score(row["literal"], sem, row["structural"]), 4)
    return rows


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _report_group(title: str, rows: list[dict], key_fn, use_bertscore: bool) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is not None:
            groups[str(key)].append(row)

    print(f"\n### {title}")
    header = "  {:<16} {:>5} {:>8} {:>8}".format("组", "n", "字面", "结构")
    if use_bertscore:
        header += " {:>8} {:>8}".format("语义", "RTC")
    print(header)
    for key in sorted(groups):
        g = groups[key]
        line = "  {:<16} {:>5} {:>8} {:>8}".format(
            key, len(g), _mean([r["literal"] for r in g]), _mean([r["structural"] for r in g])
        )
        if use_bertscore:
            line += " {:>8} {:>8}".format(
                _mean([r["semantic"] for r in g]), _mean([r["rtc"] for r in g])
            )
        print(line)


def report(rows: list[dict], use_bertscore: bool) -> None:
    print("\n" + "=" * 60)
    print("回环一致性（RTC）评估" + ("" if use_bertscore else "  [仅字面+结构，语义待补]"))
    _report_group("总体", rows, lambda r: "ALL", use_bertscore)
    _report_group("按翻译方向", rows, lambda r: _DIRECTION_NAME.get(r["direction"], r["direction"]), use_bertscore)
    _report_group("按文本长度", rows, lambda r: _length_bucket(r["length"]), use_bertscore)
    _report_group(
        "按虚词比例（仅古→今→古）",
        [r for r in rows if r["direction"] == "c2m2c"],
        lambda r: "高虚词" if r["func_ratio"] >= 0.15 else "低虚词",
        use_bertscore,
    )
    # 书籍太多，只看样本量最大的前若干本。
    book_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        book_counts[r["book"]] += 1
    top_books = {b for b, _ in sorted(book_counts.items(), key=lambda x: -x[1])[:6]}
    _report_group(
        "按来源书籍（Top6）",
        [r for r in rows if r["book"] in top_books],
        lambda r: r["book"] or "(未知)",
        use_bertscore,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="回环一致性 RTC 评估")
    parser.add_argument("--model_dir", default=None, help="生成模式：模型目录")
    parser.add_argument("--test_file", default="data/processed/test.jsonl")
    parser.add_argument("--from_file", default=None, help="评分模式：已存的回环 jsonl")
    parser.add_argument("--output", default=None, help="生成模式下保存回环文本的 jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_bertscore", action="store_true")
    args = parser.parse_args()

    use_bertscore = not args.no_bertscore

    if args.from_file:
        rows = _load_jsonl(args.from_file)
    elif args.model_dir:
        rows = build_roundtrips(args.model_dir, args.test_file, args.batch_size, args.num_beams)
    else:
        raise SystemExit("需指定 --model_dir（生成）或 --from_file（评分）其一")

    rows = score_rows(rows, use_bertscore)
    report(rows, use_bertscore)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n回环文本与逐句分数已保存到 {out}（{len(rows)} 行）。")


if __name__ == "__main__":
    main()
