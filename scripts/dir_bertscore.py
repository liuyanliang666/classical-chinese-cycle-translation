"""补算「单向译文 vs 参考译文」的语义一致性（BERTScore F1）。

回环文件里 `mid` 就是单向译文，配对行的 `original` 就是黄金参考：
  古→今：pred = c2m2c.mid，ref = m2c2m.original（黄金今文）
  今→古：pred = m2c2m.mid，ref = c2m2c.original（黄金古文）
两方向的 c2m2c / m2c2m 行按生成顺序成对出现，用 `classical` 字段校验配对。

同时重算 BLEU / ChrF 作为一致性交叉核对（应与 CYCLE_CONSISTENCY_RESULTS.md 对得上）。

用法：PYTHONPATH=src python scripts/dir_bertscore.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ccnlp.evaluate import bertscore_f1, bleu_score, chrf_score

DL = Path("/Users/mac/Downloads")
RUNS = [
    ("baseline (A, 3ep, λ=0)", "rtc_roundtrip_baseline.jsonl"),
    ("matched λ=0 (Aplus1)", "rtc_roundtrip_Aplus1.jsonl"),
    ("λ=0.1", "rtc_roundtrip_L01.jsonl"),
    ("λ=0.2", "rtc_roundtrip_L02.jsonl"),
    ("λ=0.3", "rtc_roundtrip_L03.jsonl"),
    ("λ=0.4", "rtc_roundtrip_L04.jsonl"),
    ("λ=0.5 (L05)", "rtc_roundtrip_L05.jsonl"),
    ("matched λ=0.5 (L05m)", "rtc_roundtrip_L05m.jsonl"),
]


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _pairs(rows: list[dict]):
    """返回 (c2m_rows, m2c_rows) 并校验逐对的 classical 一致。"""
    c = [r for r in rows if r.get("direction") == "c2m2c"]
    m = [r for r in rows if r.get("direction") == "m2c2m"]
    assert len(c) == len(m), f"方向行数不等: c2m2c={len(c)} m2c2m={len(m)}"
    mism = sum(1 for x, y in zip(c, m) if x.get("classical") != y.get("classical"))
    if mism:  # 顺序对不上则按 classical 重排 m
        idx = {r["classical"]: r for r in m}
        m = [idx[x["classical"]] for x in c]
    return c, m


def _score(preds: list[str], refs: list[str]) -> dict:
    return {
        "bleu": bleu_score(preds, refs),
        "chrf": chrf_score(preds, refs),
        "bert": bertscore_f1(preds, refs),  # 均值 ×100
    }


def main() -> None:
    print(f"{'运行':<24} {'方向':<6} {'BLEU':>7} {'ChrF':>7} {'BERTScore-F1':>13}")
    print("-" * 64)
    for label, fname in RUNS:
        path = DL / fname
        if not path.exists():
            print(f"{label:<24} [缺文件 {fname}]")
            continue
        c, m = _pairs(_load(path))

        c2m_pred = [r["mid"] for r in c]            # 古→今 译文
        c2m_ref = [r["original"] for r in m]        # 黄金今文
        m2c_pred = [r["mid"] for r in m]            # 今→古 译文
        m2c_ref = [r["original"] for r in c]        # 黄金古文

        s_c2m = _score(c2m_pred, c2m_ref)
        s_m2c = _score(m2c_pred, m2c_ref)
        s_all = _score(c2m_pred + m2c_pred, c2m_ref + m2c_ref)

        for dname, s in (("古→今", s_c2m), ("今→古", s_m2c), ("总体", s_all)):
            tag = label if dname == "古→今" else ""
            print(f"{tag:<24} {dname:<6} {s['bleu']:>7.2f} {s['chrf']:>7.2f} {s['bert']:>13.2f}")
        print("-" * 64)


if __name__ == "__main__":
    main()
