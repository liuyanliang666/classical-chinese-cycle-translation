"""幻觉 / 忠实度度量（无新依赖）。

针对**古→今**方向（幻觉高发处），在 (源 S=古文, 译文 H, 参考 R=今文黄金) 上量化：

  1) 数字忠实度：H 中「源与参考都没有」的数字 = 凭空数字（幻觉）；
     另报参考数字在 H 中的保留率。
  2) 不被支持的内容字率：H 的内容字（去标点 / 空白 / 数字 / 文言虚词）中，
     「既不在 S 也不在 R」的占比——凭空人名（如「张温」的「张」）会被计入。

BLEU / BERTScore 对幻觉不敏感，本模块作为它们的补充诊断；也被
`faithful_decode.py` 复用为 N-best 重排的忠实度打分。

用法（仿 `ccnlp.copy_rate`）：
  PYTHONPATH=src python -m ccnlp.hallucination --from_file rtc_roundtrip.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ccnlp.evaluate import FUNCTION_WORDS, normalized_edit_distance

# 阿拉伯数字串 + 中文数字串。
_ARABIC_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CJK_NUMERALS = "〇零一二三四五六七八九十百千万亿兆两廿卅壹贰叁肆伍陆柒捌玖拾佰仟"
_CJK_NUM_RE = re.compile(f"[{_CJK_NUMERALS}]+")

_PUNCT = set("，。！？；：、“”‘’（）《》〈〉「」『』…—·,.!?;:\"'()[]{}<>-—~`@#$%^&*_+=|\\/")
_NUMERAL_CHARS = set("0123456789.,") | set(_CJK_NUMERALS)
# 内容字停用集合：文言虚词 + 标点 + 数字字符。
_STOP_CHARS = set(FUNCTION_WORDS) | _PUNCT | _NUMERAL_CHARS

_DIRECTION_NAME = {"c2m2c": "古→今", "m2c2m": "今→古"}


def extract_numbers(text: str) -> list[str]:
    """抽取文本中的数字串（阿拉伯 + 中文数字）。"""
    return _ARABIC_RE.findall(text) + _CJK_NUM_RE.findall(text)


def _is_content_char(ch: str) -> bool:
    return ch not in _STOP_CHARS and not ch.isspace()


def number_faithfulness(source: str, hyp: str, reference: str | None = None) -> dict:
    """H 中不被 源∪参考 支持的数字 = 幻觉数字；并统计参考数字的保留率。"""
    support = source + "" + (reference or "")
    hyp_nums = extract_numbers(hyp)
    halluc = [n for n in hyp_nums if n not in support]
    ref_nums = extract_numbers(reference) if reference else []
    preserved = [n for n in ref_nums if n in hyp]
    return {
        "n_hyp_nums": len(hyp_nums),
        "n_halluc": len(halluc),
        "has_halluc": bool(halluc),
        "halluc_examples": halluc,
        "n_ref_nums": len(ref_nums),
        "n_preserved": len(preserved),
    }


def unsupported_content_char_rate(source: str, hyp: str, reference: str | None = None) -> float:
    """H 的内容字中「既不在 S 也不在 R」的占比，[0,1]（越低越忠实）。"""
    supported = set(source) | set(reference or "")
    content = [c for c in hyp if _is_content_char(c)]
    if not content:
        return 0.0
    unsupported = sum(1 for c in content if c not in supported)
    return unsupported / len(content)


def faithfulness_penalty(source: str, hyp: str, reference: str | None = None) -> float:
    """综合幻觉惩罚（越小越忠实），供重排使用：
    数字幻觉率 + 内容字幻觉率（等权）。"""
    nf = number_faithfulness(source, hyp, reference)
    num_rate = nf["n_halluc"] / nf["n_hyp_nums"] if nf["n_hyp_nums"] else 0.0
    char_rate = unsupported_content_char_rate(source, hyp, reference)
    return num_rate + char_rate


def _load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _pair_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """拆出 c2m2c / m2c2m 两组并按 classical 字段对齐配对。"""
    c = [r for r in rows if r.get("direction") == "c2m2c"]
    m = [r for r in rows if r.get("direction") == "m2c2m"]
    if len(c) == len(m) and any(
        x.get("classical") != y.get("classical") for x, y in zip(c, m)
    ):
        idx = {r["classical"]: r for r in m}
        m = [idx.get(x["classical"], x) for x in c]
    return c, m


def aggregate_triples(triples: list[tuple[str, str, str | None]]) -> dict:
    """对一批 (源 S, 译文 H, 参考 R) 聚合幻觉指标。R 可为 None。"""
    if not triples:
        return {"n": 0}

    n_sent_halluc = 0
    tot_hyp_nums = tot_halluc = tot_ref_nums = tot_preserved = 0
    char_rates: list[float] = []
    for s, h, r in triples:
        nf = number_faithfulness(s, h, r)
        n_sent_halluc += int(nf["has_halluc"])
        tot_hyp_nums += nf["n_hyp_nums"]
        tot_halluc += nf["n_halluc"]
        tot_ref_nums += nf["n_ref_nums"]
        tot_preserved += nf["n_preserved"]
        char_rates.append(unsupported_content_char_rate(s, h, r))

    n = len(triples)
    return {
        "n": n,
        "num_halluc_sent_rate": round(n_sent_halluc / n, 4),
        "num_halluc_token_rate": round(tot_halluc / tot_hyp_nums, 4) if tot_hyp_nums else 0.0,
        "ref_num_recall": round(tot_preserved / tot_ref_nums, 4) if tot_ref_nums else 0.0,
        "unsupported_char_rate": round(sum(char_rates) / n, 4),
    }


def halluc_stats(
    rows: list[dict], direction: str = "c2m2c", exclude_copy_threshold: float | None = None
) -> dict:
    """从回环 jsonl 聚合幻觉指标。

    古→今（c2m2c）：S = original(古文)，H = mid(今译)，R = 配对 m2c2m 行的 original(今文黄金)。
    今→古（m2c2m）：S = original(今文)，H = mid(古译)，R = 配对 c2m2c 行的 original(古文黄金)。

    exclude_copy_threshold：若给定（如 0.8），剔除「译文照抄源句」的句子
    （相似度 = 1−归一化编辑距离(S,H) > 阈值），用于**去除复制对幻觉指标的混淆**。
    """
    c, m = _pair_rows(rows)
    if direction == "c2m2c":
        triples = [(ci["original"], ci["mid"], mi["original"]) for ci, mi in zip(c, m)]
    else:
        triples = [(mi["original"], mi["mid"], ci["original"]) for ci, mi in zip(c, m)]

    n_excluded = 0
    if exclude_copy_threshold is not None:
        kept = [
            (s, h, r) for (s, h, r) in triples
            if (1.0 - normalized_edit_distance(s, h)) <= exclude_copy_threshold
        ]
        n_excluded = len(triples) - len(kept)
        triples = kept

    stats = aggregate_triples(triples)
    stats["n_excluded_copies"] = n_excluded
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="幻觉 / 忠实度诊断")
    parser.add_argument("--from_file", required=True, help="rtc_roundtrip.jsonl")
    parser.add_argument(
        "--direction",
        choices=["c2m2c", "m2c2m", "both"],
        default="c2m2c",
        help="c2m2c=古→今（默认，幻觉高发）；both=两向都算",
    )
    parser.add_argument(
        "--exclude_copies",
        type=float,
        default=None,
        metavar="THRESH",
        help="剔除照抄句（相似度>THRESH，如 0.8）后再算，用于去除复制对幻觉的混淆",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.from_file)
    dirs = ["c2m2c", "m2c2m"] if args.direction == "both" else [args.direction]
    print(f"文件：{args.from_file}")
    for d in dirs:
        s = halluc_stats(rows, d, exclude_copy_threshold=args.exclude_copies)
        print(f"\n=== {_DIRECTION_NAME[d]}（n={s.get('n', 0)}）===")
        if not s.get("n"):
            print("  （无该方向样本）")
            continue
        if args.exclude_copies is not None:
            print(f"  （已剔除照抄句 {s['n_excluded_copies']} 条，阈值>{args.exclude_copies}）")
        print(f"  含幻觉数字句占比      : {s['num_halluc_sent_rate']:.1%}")
        print(f"  词级数字幻觉率        : {s['num_halluc_token_rate']:.1%}")
        print(f"  参考数字保留率        : {s['ref_num_recall']:.1%}")
        print(f"  不被支持的内容字率    : {s['unsupported_char_rate']:.4f}")


if __name__ == "__main__":
    main()
