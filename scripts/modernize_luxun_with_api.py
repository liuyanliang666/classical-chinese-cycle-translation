from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


DEFAULT_INPUT = "/Users/mac/Downloads/LuXun_7000_zawen_xiaoshuo_long_style_texts.json"
DEFAULT_OUTPUT = "data/processed/luxun_style/luxun_modernized_openai.jsonl"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def load_texts(path: str | Path) -> list[str]:
    source = Path(path)
    with source.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or any(not isinstance(item, str) for item in data):
        raise ValueError(f"{source} must contain a JSON list of strings")
    return data


def record_id(index: int) -> str:
    if index <= 0:
        raise ValueError("index must be 1-based")
    return f"luxun_api_{index:06d}"


def load_seen_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()

    seen: set[str] = set()
    with output.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {output}:{line_no}: {exc}") from exc
            if isinstance(row, dict) and row.get("id"):
                seen.add(str(row["id"]))
    return seen


def build_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是一个严谨的中文改写助手，擅长把二十世纪现代文学文本"
                "改写成今天自然、日常的现代白话。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请将下面这段鲁迅原文改写成现代日常白话文。\n\n"
                "要求：\n"
                "1. 保留原意、人物关系、事实和情绪强度。\n"
                "2. 用今天自然、通顺、日常的现代中文表达。\n"
                "3. 不要添加原文没有的信息。\n"
                "4. 不要解释、评论、加标题或输出列表。\n"
                "5. 只输出改写后的文本。\n\n"
                f"原文：\n{text.strip()}"
            ),
        },
    ]


def build_payload(text: str, model: str, temperature: float) -> dict[str, Any]:
    return {
        "model": model,
        "messages": build_messages(text),
        "temperature": temperature,
    }


def extract_chat_content(response: dict[str, Any]) -> str:
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


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_tokens: int | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.opener = opener or urllib.request.urlopen

    def modernize(self, text: str) -> str:
        payload = build_payload(text=text, model=self.model, temperature=self.temperature)
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            response = self.opener(request, timeout=self.timeout)
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
            parsed = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API returned invalid JSON: {exc}") from exc

        return extract_chat_content(parsed)


def modernize_with_retries(
    modernize_one: Callable[[str], str],
    text: str,
    retries: int,
    retry_sleep: float = 1.0,
) -> str:
    attempts = max(1, retries)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return modernize_one(text)
        except Exception as exc:  # noqa: BLE001 - keep batch jobs alive and report row-level errors.
            last_error = exc
            if attempt < attempts and retry_sleep > 0:
                time.sleep(retry_sleep)

    if last_error is None:
        raise RuntimeError("modernization failed")
    raise last_error


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_batch(
    input_path: str | Path,
    output_path: str | Path,
    modernize_one: Callable[[str], str],
    model: str,
    start: int = 1,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    retries: int = 3,
    retry_sleep: float = 1.0,
) -> dict[str, int]:
    if start <= 0:
        raise ValueError("start must be 1-based")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    texts = load_texts(input_path)
    seen_ids = load_seen_ids(output_path)
    summary = {"processed": 0, "skipped": 0, "errors": 0}
    considered = 0

    for index, source_text in enumerate(texts, start=1):
        if index < start:
            continue
        if limit is not None and considered >= limit:
            break
        considered += 1

        item_id = record_id(index)
        if item_id in seen_ids:
            summary["skipped"] += 1
            continue

        record = {
            "id": item_id,
            "source_luxun": source_text,
            "target_modern": "",
            "model": model,
        }

        try:
            record["target_modern"] = modernize_with_retries(
                modernize_one=modernize_one,
                text=source_text,
                retries=retries,
                retry_sleep=retry_sleep,
            )
        except Exception as exc:  # noqa: BLE001 - write the failure and continue the batch.
            record["error"] = str(exc)
            summary["errors"] += 1

        append_jsonl(record, output_path)
        seen_ids.add(item_id)
        summary["processed"] += 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modernize Lu Xun source records with an OpenAI-compatible chat completions API."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON list of Lu Xun source texts.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI-compatible chat model name.",
    )
    parser.add_argument(
        "--base-url",
        "--base_url",
        dest="base_url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL prefix, for example https://api.openai.com/v1.",
    )
    parser.add_argument("--start", type=int, default=1, help="1-based input record index to start from.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of input records to consider.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each written row.")
    parser.add_argument("--retries", type=int, default=3, help="Total attempts per row.")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="Seconds to sleep between retry attempts.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens value for the API.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required in the environment.")

    client = OpenAIChatClient(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    summary = run_batch(
        input_path=args.input,
        output_path=args.output,
        modernize_one=client.modernize,
        model=args.model,
        start=args.start,
        limit=args.limit,
        sleep_seconds=args.sleep,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    print(
        "Processed: {processed}, skipped: {skipped}, errors: {errors}, output: {output}".format(
            **summary,
            output=Path(args.output),
        )
    )


if __name__ == "__main__":
    main()
