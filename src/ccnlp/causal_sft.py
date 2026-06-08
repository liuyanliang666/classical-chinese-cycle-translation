from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "你是一个鲁迅风格改写助手。请在保持原意和关键信息的前提下，"
    "将现代白话文本改写成鲁迅的语言风格。只输出改写结果，不要解释。"
)
USER_TEMPLATE = "请将下面文本改写成鲁迅风格，保持原意，不要解释。\n{source}"
ASSISTANT_PREFIX = "<|assistant|>\n"
LUXUN_SOURCE_PREFIX = "鲁迅风格化："


def build_luxun_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    source = _source_text(example)
    target = _target_text(example)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(source=source)},
        {"role": "assistant", "content": target},
    ]


def build_generation_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(source=text.strip())},
    ]


def build_prompt(example: dict[str, Any], tokenizer: Any) -> str:
    return _render_chat(build_luxun_messages(example), tokenizer, add_generation_prompt=False)


def build_generation_prompt(text: str, tokenizer: Any) -> str:
    return _render_chat(build_generation_messages(text), tokenizer, add_generation_prompt=True)


def preprocess_example(
    example: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    messages = build_luxun_messages(example)
    full_text = _render_chat(messages, tokenizer, add_generation_prompt=False)
    target_text = _target_text(example)

    full_ids = _encode(tokenizer, full_text, add_special_tokens=True)
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]

    label_start = _target_label_start(tokenizer, full_text, target_text, full_ids)
    labels = [-100] * label_start + full_ids[label_start:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _render_chat(
    messages: list[dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    rendered = [
        f"System: {messages[0]['content']}",
        f"User: {messages[1]['content']}",
    ]
    if add_generation_prompt:
        rendered.append("Assistant: ")
    elif len(messages) > 2:
        rendered.append(f"Assistant: {messages[2]['content']}")
    return "\n".join(rendered)


def _encode(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return list(encoded["input_ids"])


def _target_label_start(
    tokenizer: Any,
    full_text: str,
    target_text: str,
    full_ids: list[int],
) -> int:
    target_start = full_text.rfind(target_text)
    if target_start < 0:
        raise ValueError("target text was not found in rendered chat template")

    offset_start = _target_label_start_from_offsets(tokenizer, full_text, target_start)
    if offset_start is not None:
        return min(offset_start, len(full_ids))

    prefix_ids = _encode(tokenizer, full_text[:target_start])
    return min(len(prefix_ids), len(full_ids))


def _target_label_start_from_offsets(
    tokenizer: Any,
    full_text: str,
    target_start: int,
) -> int | None:
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError):
        return None

    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None

    for index, offset in enumerate(offsets):
        start, end = offset
        if end > target_start:
            return index
    return len(offsets)


def _source_text(example: dict[str, Any]) -> str:
    if "source_plain" in example:
        return str(example["source_plain"]).strip()
    source = str(example["source"]).strip()
    if source.startswith(LUXUN_SOURCE_PREFIX):
        source = source[len(LUXUN_SOURCE_PREFIX) :]
    return source.strip()


def _target_text(example: dict[str, Any]) -> str:
    if "target_luxun" in example:
        return str(example["target_luxun"]).strip()
    return str(example["target"]).strip()
