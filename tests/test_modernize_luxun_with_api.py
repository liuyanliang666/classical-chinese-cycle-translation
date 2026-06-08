import json
from pathlib import Path

import pytest

from scripts import modernize_luxun_with_api as modernize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_load_texts_requires_json_list_of_strings(tmp_path: Path):
    input_path = tmp_path / "luxun.json"
    input_path.write_text(json.dumps(["原文一", "原文二"], ensure_ascii=False), encoding="utf-8")

    assert modernize.load_texts(input_path) == ["原文一", "原文二"]

    input_path.write_text(json.dumps([{"text": "不是字符串"}], ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="list of strings"):
        modernize.load_texts(input_path)


def test_extract_chat_content_strips_openai_compatible_response():
    response = {"choices": [{"message": {"content": "  现代白话文本\n"}}]}

    assert modernize.extract_chat_content(response) == "现代白话文本"


def test_run_batch_skips_existing_ids_and_writes_successes_and_errors(tmp_path: Path):
    input_path = tmp_path / "luxun.json"
    output_path = tmp_path / "modernized.jsonl"
    input_path.write_text(json.dumps(["甲原文", "乙原文", "丙原文"], ensure_ascii=False), encoding="utf-8")
    output_path.write_text(
        json.dumps(
            {
                "id": "luxun_api_000002",
                "source_luxun": "乙原文",
                "target_modern": "已有改写",
                "model": "test-model",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_modernize(text: str) -> str:
        calls.append(text)
        if text == "丙原文":
            raise RuntimeError("api failed")
        return f"现代{text}"

    summary = modernize.run_batch(
        input_path=input_path,
        output_path=output_path,
        modernize_one=fake_modernize,
        model="test-model",
        retries=1,
    )

    assert calls == ["甲原文", "丙原文"]
    assert summary == {"processed": 2, "skipped": 1, "errors": 1}
    rows = read_jsonl(output_path)
    assert [row["id"] for row in rows] == ["luxun_api_000002", "luxun_api_000001", "luxun_api_000003"]
    assert rows[1]["target_modern"] == "现代甲原文"
    assert rows[2]["target_modern"] == ""
    assert rows[2]["error"] == "api failed"
