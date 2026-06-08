from ccnlp.causal_sft import (
    ASSISTANT_PREFIX,
    build_luxun_messages,
    build_prompt,
    preprocess_example,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        rendered = "\n".join(parts)
        if tokenize:
            return self(rendered)["input_ids"]
        return rendered

    def __call__(self, text, max_length=None, truncation=False, add_special_tokens=False):
        ids = [ord(ch) for ch in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_build_luxun_messages_uses_plain_source_and_target():
    example = {
        "source_plain": "街上很多人沉默地走过。",
        "target_luxun": "街上走着许多人，沉默便像铁一样压着。",
    }

    messages = build_luxun_messages(example)

    assert messages[0]["role"] == "system"
    assert "鲁迅" in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "请将下面文本改写成鲁迅风格，保持原意，不要解释。\n街上很多人沉默地走过。",
    }
    assert messages[2] == {
        "role": "assistant",
        "content": "街上走着许多人，沉默便像铁一样压着。",
    }


def test_build_prompt_supports_existing_source_target_records():
    tokenizer = DummyTokenizer()
    example = {
        "source": "鲁迅风格化：街上很多人沉默地走过。",
        "target": "街上走着许多人，沉默便像铁一样压着。",
    }

    prompt = build_prompt(example, tokenizer)

    assert "鲁迅风格化：" not in prompt
    assert "街上很多人沉默地走过。" in prompt
    assert "街上走着许多人，沉默便像铁一样压着。" in prompt
    assert ASSISTANT_PREFIX in prompt


def test_preprocess_example_masks_prompt_tokens_and_keeps_answer_labels():
    tokenizer = DummyTokenizer()
    example = {
        "source_plain": "街上很多人沉默地走过。",
        "target_luxun": "街上走着许多人。",
    }

    features = preprocess_example(example, tokenizer, max_seq_length=512)
    labels = features["labels"]

    first_label = next(i for i, label in enumerate(labels) if label != -100)
    assert first_label > 0
    assert labels[:first_label] == [-100] * first_label
    assert labels[first_label:] == features["input_ids"][first_label:]
    assert features["input_ids"][-1] == tokenizer.eos_token_id


def test_preprocess_example_masks_until_target_inside_full_chat_template():
    tokenizer = ThinkingPromptTokenizer()
    example = {
        "source_plain": "街上很多人沉默地走过。",
        "target_luxun": "街上走着许多人。",
    }

    features = preprocess_example(example, tokenizer, max_seq_length=512)
    first_label = next(i for i, label in enumerate(features["labels"]) if label != -100)
    labeled_text = "".join(chr(i) for i in features["input_ids"][first_label:-1])

    assert labeled_text == "街上走着许多人。"


def test_preprocess_example_truncates_to_max_length():
    tokenizer = DummyTokenizer()
    example = {
        "source_plain": "甲" * 200,
        "target_luxun": "乙" * 200,
    }

    features = preprocess_example(example, tokenizer, max_seq_length=64)

    assert len(features["input_ids"]) == 64
    assert len(features["attention_mask"]) == 64
    assert len(features["labels"]) == 64


class ThinkingPromptTokenizer(DummyTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n<think>\n\n</think>\n\n")
        rendered = "\n".join(parts)
        if tokenize:
            return self(rendered)["input_ids"]
        return rendered
