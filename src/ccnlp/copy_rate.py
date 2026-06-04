"""复制坍缩诊断：度量「今译（中间步）」与古文原句的相似度。

回环约束权重过大时，模型可能在翻译这步直接照抄古文原句来骗高回环一致性。
本脚本在古→今方向上量化这种「复制坍缩」：
  相似度 = 1 − 归一化编辑距离(古文原句, 今译)
  复制率 = 相似度 > 阈值 的句子占比（默认阈值 0.8）

用法：
  PYTHONPATH=src python -m ccnlp.copy_rate --from_file rtc_roundtrip.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccnlp.evaluate import normalized_edit_distance


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def copy_stats(rows: list[dict], threshold: float = 0.8) -> dict:
    """在 c2m2c（古→今→古）行上，比较古文原句(original) 与今译(mid)。"""
    c2m = [r for r in rows if r.get("direction") == "c2m2c"]
    sims = [1.0 - normalized_edit_distance(r["original"], r["mid"]) for r in c2m]
    if not sims:
        return {"n": 0, "mean_sim": 0.0, "copy_rate": 0.0}
    copy_rate = sum(1 for s in sims if s > threshold) / len(sims)
    return {
        "n": len(sims),
        "mean_sim": round(sum(sims) / len(sims), 4),
        "copy_rate": round(copy_rate, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="复制坍缩诊断（今译 vs 古文原句相似度）")
    parser.add_argument("--from_file", required=True, help="rtc_roundtrip.jsonl")
    parser.add_argument("--threshold", type=float, default=0.8, help="判定为复制的相似度阈值")
    args = parser.parse_args()

    rows = _load_jsonl(args.from_file)
    stats = copy_stats(rows, args.threshold)
    print(f"文件：{args.from_file}")
    print(f"  古→今 样本数        : {stats['n']}")
    print(f"  今译≈古文 平均相似度 : {stats['mean_sim']}")
    print(f"  复制率(相似度>{args.threshold}) : {stats['copy_rate']:.1%}")


if __name__ == "__main__":
    main()
