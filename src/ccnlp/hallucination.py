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
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

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

DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
_SEVERITIES = {"none", "low", "medium", "high"}
_TASK_PREFIXES = {
    "classical_to_modern": "古文翻今：",
    "modern_to_classical": "今文翻古：",
}


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


def _build_content_hallucination_messages(
    source: str, hyp: str, reference: str | None = None
) -> list[dict[str, str]]:
    ref_text = reference.strip() if reference else "未提供"
    return [
        {
            "role": "system",
            "content": (
                "你是一个严谨的中文翻译忠实度评审。只判断译文是否相对源文"
                "或参考译文新增、篡改、误解了事实内容；不要因为表达更通顺、"
                "同义改写或必要补足主语就判为幻觉。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请判断候选译文是否存在内容幻觉。\n\n"
                "判定标准：\n"
                "1. 若候选译文加入源文/参考都没有的人物、地点、时间、数字、事件、因果或评价，判为幻觉。\n"
                "2. 若候选译文改变了事实关系、主体客体、数量、时间或语气立场，判为幻觉。\n"
                "3. 合理意译、语序调整、文言到白话的必要补全，不应判为幻觉。\n\n"
                "必须只输出 JSON，格式如下：\n"
                "{\n"
                '  "has_hallucination": true 或 false,\n'
                '  "severity": "none|low|medium|high",\n'
                '  "unsupported_claims": [{"text": "可疑内容", "reason": "原因"}],\n'
                '  "explanation": "一句话说明"\n'
                "}\n\n"
                f"源文：\n{source.strip()}\n\n"
                f"参考译文：\n{ref_text}\n\n"
                f"候选译文：\n{hyp.strip()}"
            ),
        },
    ]


def _extract_chat_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("API response missing choices[0].message.content") from exc

    if not isinstance(content, str):
        raise RuntimeError("API response content is not a string")

    content = content.strip()
    if not content:
        raise RuntimeError("API returned empty content")
    return content


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("LLM response is not a JSON object")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response JSON must be an object")
    return parsed


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "是", "有"}:
            return True
        if normalized in {"false", "no", "n", "0", "否", "无"}:
            return False
    return None


def _normalize_llm_hallucination_result(parsed: dict[str, Any]) -> dict[str, Any]:
    has_hallucination = _coerce_bool(parsed.get("has_hallucination"))
    if has_hallucination is None:
        raise RuntimeError("LLM JSON missing boolean has_hallucination")

    severity = str(parsed.get("severity") or ("high" if has_hallucination else "none")).lower()
    if severity not in _SEVERITIES:
        severity = "high" if has_hallucination else "none"

    unsupported_claims = parsed.get("unsupported_claims", [])
    if unsupported_claims is None:
        unsupported_claims = []
    elif not isinstance(unsupported_claims, list):
        unsupported_claims = [{"text": str(unsupported_claims), "reason": ""}]

    return {
        "has_hallucination": has_hallucination,
        "severity": severity,
        "unsupported_claims": unsupported_claims,
        "explanation": str(parsed.get("explanation") or ""),
    }


def llm_content_hallucination(
    source: str,
    hyp: str,
    reference: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_tokens: int | None = 512,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """调用 OpenAI-compatible Chat Completions API 判断译文是否有内容幻觉。

    用户可显式传入 api_key/base_url/model，也可通过环境变量提供：
    LLM_API_KEY、LLM_API_BASE_URL、LLM_MODEL（兼容 OPENAI_API_KEY、OPENAI_BASE_URL）。
    """
    resolved_api_key = (
        api_key if api_key is not None else os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError("api_key is required, or set LLM_API_KEY/OPENAI_API_KEY")

    resolved_base_url = (
        base_url
        if base_url is not None
        else os.getenv("LLM_API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_LLM_BASE_URL
    )
    resolved_model = model if model is not None else os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL

    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": _build_content_hallucination_messages(source, hyp, reference),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{resolved_base_url.rstrip('/')}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        response = (opener or urllib.request.urlopen)(request, timeout=timeout)
        try:
            raw_body = response.read()
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc

    try:
        api_response = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned invalid JSON: {exc}") from exc

    content = _extract_chat_content(api_response)
    return _normalize_llm_hallucination_result(_parse_json_object(content))


def _load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(rows: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_known_task_prefix(text: str, task: str | None = None) -> str:
    value = text.strip()
    if task and task in _TASK_PREFIXES and value.startswith(_TASK_PREFIXES[task]):
        return value[len(_TASK_PREFIXES[task]) :].strip()
    for prefix in _TASK_PREFIXES.values():
        if value.startswith(prefix):
            return value[len(prefix) :].strip()
    return value


def _select_text_field(
    row: dict,
    preferred_field: str | None,
    fallback_fields: list[str],
    label: str,
) -> str:
    fields: list[str] = []
    if preferred_field:
        fields.append(preferred_field)
    fields.extend(field for field in fallback_fields if field not in fields)

    for field in fields:
        if field in row and row[field] is not None:
            value = str(row[field]).strip()
            if value:
                return value
    raise ValueError(f"{label} field not found; tried {fields}")


def _next_reference_for_prediction(
    reference_by_task: dict[str, list[dict]],
    reference_offsets: dict[str, int],
    references: list[dict],
    global_offset: int,
    prediction: dict,
) -> tuple[dict, int]:
    task = prediction.get("task")
    if task is not None and str(task) in reference_by_task:
        key = str(task)
        offset = reference_offsets.get(key, 0)
        if offset >= len(reference_by_task[key]):
            raise ValueError(f"Not enough reference rows for task={key}")
        reference_offsets[key] = offset + 1
        return reference_by_task[key][offset], global_offset

    if global_offset >= len(references):
        raise ValueError("Not enough reference rows for prediction rows")
    return references[global_offset], global_offset + 1


def _coerce_hallucination_flag(result: dict[str, Any] | bool) -> bool:
    if isinstance(result, dict):
        value = result.get("has_hallucination")
    else:
        value = result
    flag = _coerce_bool(value)
    if flag is None:
        raise RuntimeError("LLM result missing boolean has_hallucination")
    return flag


def judge_llm_content_hallucination_jsonl(
    reference_file: str | Path,
    prediction_file: str | Path,
    output_file: str | Path | None = None,
    *,
    source_field: str = "source",
    reference_field: str = "target",
    hyp_field: str = "prediction",
    task: str | None = None,
    limit: int | None = None,
    judge_one: Callable[[str, str, str | None], dict[str, Any] | bool] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_tokens: int | None = 512,
) -> dict[str, float | int]:
    """批量调用 LLM 判断 prediction_file 中每句译文是否有内容幻觉。

    reference_file 是标准答案 jsonl，默认读取 source/target；
    prediction_file 是待判断输出 jsonl，默认读取 prediction。
    输出 jsonl 每行只包含 line_no/task/has_hallucination，便于统计幻觉率。
    """
    references = _load_jsonl(reference_file)
    predictions = _load_jsonl(prediction_file)
    if task is not None:
        references = [row for row in references if row.get("task") == task]
        predictions = [row for row in predictions if row.get("task") == task]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        predictions = predictions[:limit]

    reference_by_task: dict[str, list[dict]] = {}
    for row in references:
        if row.get("task") is not None:
            reference_by_task.setdefault(str(row["task"]), []).append(row)

    def call_judge(source: str, hyp: str, reference: str | None) -> dict[str, Any]:
        return llm_content_hallucination(
            source=source,
            hyp=hyp,
            reference=reference,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )

    judge = judge_one or call_judge
    reference_offsets: dict[str, int] = {}
    global_offset = 0
    output_rows: list[dict] = []

    for prediction in predictions:
        reference_row, global_offset = _next_reference_for_prediction(
            reference_by_task, reference_offsets, references, global_offset, prediction
        )
        row_task = prediction.get("task") or reference_row.get("task")
        task_name = str(row_task) if row_task is not None else None
        source = _strip_known_task_prefix(
            _select_text_field(reference_row, source_field, ["input", "original"], "source"),
            task_name,
        )
        reference = _select_text_field(
            reference_row, reference_field, ["reference", "target"], "reference"
        )
        hyp = _select_text_field(
            prediction,
            hyp_field,
            ["hyp", "output", "mid", "translation", "candidate", "target"],
            "prediction",
        )
        has_hallucination = _coerce_hallucination_flag(judge(source, hyp, reference))

        out_row: dict[str, Any] = {"line_no": len(output_rows) + 1}
        if task_name is not None:
            out_row["task"] = task_name
        out_row["has_hallucination"] = has_hallucination
        output_rows.append(out_row)

    if output_file is not None:
        _write_jsonl(output_rows, output_file)

    n_hallucination = sum(1 for row in output_rows if row["has_hallucination"])
    n = len(output_rows)
    return {
        "n": n,
        "n_hallucination": n_hallucination,
        "hallucination_rate": round(n_hallucination / n, 4) if n else 0.0,
    }


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
    parser.add_argument("--from_file", default=None, help="rtc_roundtrip.jsonl")
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
    parser.add_argument("--reference_file", default=None, help="LLM 判断模式：标准答案 jsonl")
    parser.add_argument("--prediction_file", default=None, help="LLM 判断模式：待判断输出 jsonl")
    parser.add_argument("--output", default=None, help="LLM 判断模式：逐句布尔结果 jsonl")
    parser.add_argument("--source_field", default="source", help="标准答案 jsonl 中的源文字段")
    parser.add_argument("--reference_field", default="target", help="标准答案 jsonl 中的参考答案字段")
    parser.add_argument("--hyp_field", default="prediction", help="待判断输出 jsonl 中的译文字段")
    parser.add_argument("--task", default=None, help="可选：只判断指定 task，如 classical_to_modern")
    parser.add_argument("--limit", type=int, default=None, help="可选：只判断前 N 条，调试 API 用")
    parser.add_argument("--api_key", default=None, help="LLM API key；也可用 LLM_API_KEY/OPENAI_API_KEY")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible base URL；也可用环境变量")
    parser.add_argument("--model", default=None, help="LLM 模型名；也可用 LLM_MODEL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    if args.reference_file or args.prediction_file:
        if not args.reference_file or not args.prediction_file:
            parser.error("--reference_file and --prediction_file must be used together")
        summary = judge_llm_content_hallucination_jsonl(
            reference_file=args.reference_file,
            prediction_file=args.prediction_file,
            output_file=args.output,
            source_field=args.source_field,
            reference_field=args.reference_field,
            hyp_field=args.hyp_field,
            task=args.task,
            limit=args.limit,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        print(f"标准答案：{args.reference_file}")
        print(f"待判断输出：{args.prediction_file}")
        if args.output:
            print(f"逐句结果：{args.output}")
        print(f"LLM 内容幻觉率：{summary['hallucination_rate']:.1%}")
        print(f"幻觉句数：{summary['n_hallucination']} / {summary['n']}")
        return

    if not args.from_file:
        parser.error("--from_file is required unless using --reference_file/--prediction_file")

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
