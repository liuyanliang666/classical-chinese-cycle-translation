import json
from pathlib import Path

import pytest

from ccnlp.evaluate import (
    bleu_score,
    chrf_score,
    exact_match_rate,
    function_word_ratio,
    literal_consistency,
    normalized_edit_distance,
    rtc_score,
    structural_consistency,
)
from ccnlp.hallucination import (
    extract_numbers,
    faithfulness_penalty,
    judge_llm_content_hallucination_jsonl,
    llm_content_hallucination,
    number_faithfulness,
    unsupported_content_char_rate,
)
from ccnlp.inference import BaselineGenerator, TASK_PREFIX, TaskType
from ccnlp.preprocess import (
    ParallelExample,
    build_task_examples,
    clean_text,
    dedup_examples,
    load_niutrans_parallel_dir,
    load_parallel_csv,
    split_examples,
)


def test_task_type_behaves_like_string_enum():
    assert TaskType.CLASSICAL_TO_MODERN == "classical_to_modern"
    assert TaskType("classical_to_modern") is TaskType.CLASSICAL_TO_MODERN


def test_clean_text_normalizes_spaces_and_punctuation():
    assert clean_text(" 学而时习之 ，  不亦说乎！ ") == "学而时习之，不亦说乎！"


def test_load_parallel_csv_and_build_bidirectional_examples(tmp_path: Path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "classical,modern\n学而时习之，不亦说乎,学习后按时温习，不也很快乐吗\n",
        encoding="utf-8",
    )

    pairs = load_parallel_csv(csv_path)
    tasks = build_task_examples(pairs)

    assert pairs == [
        ParallelExample(
            classical="学而时习之，不亦说乎",
            modern="学习后按时温习，不也很快乐吗",
        )
    ]
    assert tasks[0]["source"] == "古文翻今：学而时习之，不亦说乎"
    assert tasks[0]["target"] == "学习后按时温习，不也很快乐吗"
    assert tasks[1]["source"] == "今文翻古：学习后按时温习，不也很快乐吗"
    assert tasks[1]["target"] == "学而时习之，不亦说乎"


def test_load_niutrans_parallel_dir_reads_source_target_pairs(tmp_path: Path):
    chapter = tmp_path / "双语数据" / "论语" / "学而篇"
    chapter.mkdir(parents=True)
    (chapter / "source.txt").write_text("学而时习之，不亦说乎\n有朋自远方来，不亦乐乎\n", encoding="utf-8")
    (chapter / "target.txt").write_text(
        "学习后按时温习，不也很快乐吗\n有朋友从远方来，不也很快乐吗\n",
        encoding="utf-8",
    )

    examples = load_niutrans_parallel_dir(tmp_path / "双语数据", max_examples=1)

    assert examples == [
        ParallelExample(
            classical="学而时习之，不亦说乎",
            modern="学习后按时温习，不也很快乐吗",
            book="论语",
        )
    ]


def test_build_task_examples_attaches_book_and_length_metadata():
    pairs = [ParallelExample(classical="学而时习之", modern="学习并按时温习", book="论语")]
    tasks = build_task_examples(pairs)

    assert tasks[0]["book"] == "论语"
    assert tasks[0]["length"] == len("学而时习之")
    assert tasks[1]["book"] == "论语"
    assert tasks[1]["length"] == len("学而时习之")


def test_dedup_examples_removes_exact_duplicates():
    pairs = [
        ParallelExample(classical="甲", modern="一"),
        ParallelExample(classical="甲", modern="一"),
        ParallelExample(classical="乙", modern="二"),
    ]
    assert dedup_examples(pairs) == [
        ParallelExample(classical="甲", modern="一"),
        ParallelExample(classical="乙", modern="二"),
    ]


def test_split_examples_is_deterministic_and_keeps_all_pairs():
    pairs = [ParallelExample(classical=str(i), modern=str(i)) for i in range(100)]
    splits = split_examples(pairs, val_ratio=0.1, test_ratio=0.1, seed=42)

    assert len(splits["test"]) == 10
    assert len(splits["validation"]) == 10
    assert len(splits["train"]) == 80

    # 同种子可复现，且三个 split 无交集、并集覆盖全部
    again = split_examples(pairs, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert [e.classical for e in splits["train"]] == [e.classical for e in again["train"]]
    all_back = splits["train"] + splits["validation"] + splits["test"]
    assert {e.classical for e in all_back} == {e.classical for e in pairs}


def test_baseline_generator_supports_translation_and_styles():
    generator = BaselineGenerator()

    modern = generator.generate("学而时习之，不亦说乎", TaskType.CLASSICAL_TO_MODERN)
    classical = generator.generate("学习后按时温习，不也很快乐吗", TaskType.MODERN_TO_CLASSICAL)
    luxun = generator.generate("我看见很多人沉默", TaskType.LUXUN_STYLE, style_strength=0.8)
    poem = generator.generate("春天来了，花开了", TaskType.TANG_POEM_STYLE)

    assert "学习" in modern
    assert "学" in classical and "乎" in classical
    assert "我向来" in luxun
    assert "春" in poem and "兮" in poem


def test_model_task_prefix_supports_luxun_style_training_prefix():
    assert TASK_PREFIX[TaskType.LUXUN_STYLE] == "鲁迅风格化："


def test_evaluation_metrics_return_reasonable_values():
    predictions = ["学习后按时温习，不也很快乐吗", "春风又绿江南岸"]
    references = ["学习后按时温习，不也很快乐吗", "春风吹绿江南岸"]

    assert exact_match_rate(predictions, references) == 0.5
    assert 60.0 <= chrf_score(predictions, references) <= 100.0


def test_bleu_score_is_high_for_identical_and_perfect_for_match():
    pytest.importorskip("sacrebleu")
    predictions = ["学习后按时温习，不也很快乐吗", "春风又绿江南岸"]
    references = ["学习后按时温习，不也很快乐吗", "春风吹绿江南岸"]

    score = bleu_score(predictions, references)
    assert 0.0 < score <= 100.0
    # 完全一致应得满分
    assert bleu_score(["温故而知新"], ["温故而知新"]) == 100.0


def test_rtc_component_metrics():
    # 完全相同 → 编辑距离为 0，结构一致性满分
    assert normalized_edit_distance("学而时习之", "学而时习之") == 0.0
    assert structural_consistency("学而时习之。", "学而时习之。") == 1.0
    # 字面一致性：≥6 字的相同串内置 ChrF 可达满分 → literal=1.0
    assert literal_consistency("学而时习之乎", "学而时习之乎") == 1.0
    # 相同串字面一致性必为各串的高值
    assert literal_consistency("学而时习之", "学而时习之") >= 0.7
    # 完全不同 → 编辑距离为 1
    assert normalized_edit_distance("甲乙丙", "丁戊己") == 1.0


def test_function_word_ratio_counts_classical_particles():
    # “之乎者也”全是虚词 → 比例为 1
    assert function_word_ratio("之乎者也") == 1.0
    # 无虚词
    assert function_word_ratio("山高水长") == 0.0


def test_rtc_score_weighting():
    # 三项都为 1 → RTC=1；语义权重 0.6 最大
    assert rtc_score(1.0, 1.0, 1.0) == 1.0
    assert abs(rtc_score(0.0, 1.0, 0.0) - 0.6) < 1e-9


# ---- 幻觉 / 忠实度度量 ----

def test_extract_numbers_arabic_and_chinese():
    nums = extract_numbers("世延弹劾十三条罪，又记 13 次")
    assert "十三" in nums
    assert "13" in nums


def test_number_faithfulness_flags_unsupported_numbers():
    # 源与参考都没有「四」→ 译文凭空数字算幻觉
    nf = number_faithfulness(source="十三条罪", hyp="四条罪", reference="十三条罪")
    assert nf["n_halluc"] == 1 and nf["has_halluc"]
    # 译文数字被源支持 → 不算幻觉
    nf2 = number_faithfulness(source="十三条罪", hyp="十三条罪行", reference="十三条")
    assert nf2["n_halluc"] == 0


def test_unsupported_content_char_rate_flags_hallucinated_name():
    # 源/参考都无「张」，译文凭空人名「张温」→「张」被计入
    rate = unsupported_content_char_rate(
        source="温故而知新", hyp="张温温习旧知识", reference="温习旧知识获得新理解"
    )
    assert rate > 0.0
    # 完全被支持 → 0
    assert unsupported_content_char_rate("温故知新", "温故知新", "温故知新") == 0.0


def test_faithfulness_penalty_lower_is_better():
    # 仅用源（重排口径，不看参考）：凭空数字应使惩罚更大
    faithful = faithfulness_penalty("十三条罪", "十三条罪", reference=None)
    halluc = faithfulness_penalty("十三条罪", "四十条罪", reference=None)
    assert halluc > faithful


def test_llm_content_hallucination_posts_to_openai_compatible_api_and_parses_json():
    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "has_hallucination": True,
                                        "severity": "high",
                                        "unsupported_claims": [
                                            {
                                                "text": "张温拜见皇帝",
                                                "reason": "源文和参考均未提到皇帝",
                                            }
                                        ],
                                        "explanation": "译文新增了拜见皇帝这一事件。",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                },
                ensure_ascii=False,
            ).encode("utf-8")

        def close(self):
            captured["closed"] = True

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    result = llm_content_hallucination(
        source="温故而知新",
        hyp="张温温习旧知识后拜见皇帝",
        reference="温习旧知识获得新理解",
        api_key="sk-test",
        base_url="https://example.test/v1",
        model="deepseek-chat",
        timeout=7.0,
        opener=fake_opener,
    )

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["timeout"] == 7.0
    assert captured["closed"] is True
    assert captured["payload"]["model"] == "deepseek-chat"
    user_prompt = captured["payload"]["messages"][1]["content"]
    assert "温故而知新" in user_prompt
    assert "张温温习旧知识后拜见皇帝" in user_prompt
    assert "温习旧知识获得新理解" in user_prompt
    assert result["has_hallucination"] is True
    assert result["severity"] == "high"
    assert result["unsupported_claims"][0]["text"] == "张温拜见皇帝"


def test_llm_content_hallucination_requires_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="api_key"):
        llm_content_hallucination(
            source="温故而知新",
            hyp="温习旧知识获得新理解",
            base_url="https://example.test/v1",
            model="deepseek-chat",
        )


def test_llm_content_hallucination_defaults_to_deepseek_base_url(monkeypatch):
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "has_hallucination": False,
                                        "severity": "none",
                                        "unsupported_claims": [],
                                        "explanation": "无幻觉。",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                },
                ensure_ascii=False,
            ).encode("utf-8")

        def close(self):
            pass

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    llm_content_hallucination(
        source="温故而知新",
        hyp="温习旧知识获得新理解",
        reference="温习旧知识获得新理解",
        api_key="sk-test",
        opener=fake_opener,
    )

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["payload"]["model"] == "deepseek-v4-flash"


def test_judge_llm_content_hallucination_jsonl_outputs_sentence_flags(tmp_path: Path):
    reference_file = tmp_path / "test.jsonl"
    prediction_file = tmp_path / "predictions.jsonl"
    output_file = tmp_path / "hallucination_flags.jsonl"
    reference_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "classical_to_modern",
                        "source": "古文翻今：温故而知新",
                        "target": "温习旧知识获得新理解",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "task": "modern_to_classical",
                        "source": "今文翻古：学习后按时温习",
                        "target": "学而时习之",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "task": "classical_to_modern",
                        "source": "古文翻今：十三条罪",
                        "target": "十三条罪行",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "classical_to_modern",
                        "prediction": "张温温习旧知识后拜见皇帝",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "task": "classical_to_modern",
                        "prediction": "十三条罪行",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_judge(source, hyp, reference):
        calls.append((source, hyp, reference))
        return {"has_hallucination": "皇帝" in hyp}

    summary = judge_llm_content_hallucination_jsonl(
        reference_file=reference_file,
        prediction_file=prediction_file,
        output_file=output_file,
        judge_one=fake_judge,
    )

    assert calls == [
        ("温故而知新", "张温温习旧知识后拜见皇帝", "温习旧知识获得新理解"),
        ("十三条罪", "十三条罪行", "十三条罪行"),
    ]
    rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"line_no": 1, "task": "classical_to_modern", "has_hallucination": True},
        {"line_no": 2, "task": "classical_to_modern", "has_hallucination": False},
    ]
    assert summary == {"n": 2, "n_hallucination": 1, "hallucination_rate": 0.5}


def test_source_copy_bias_processor_boosts_source_tokens():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ccnlp.faithful_decode import make_logits_processor

    class FakeTok:
        all_special_ids: list[int] = []
        _vocab = {"三": 3, "人": 7, "行": 9}
        _inv = {3: "三", 7: "人", 9: "行"}

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [self._vocab[c] for c in text]}

        def decode(self, ids):
            return "".join(self._inv[i] for i in ids)

    class FakeGen:
        tokenizer = FakeTok()

    # content：数字字符「三」+ 内容字「人」都被抬高，未出现的「行」不变
    proc = make_logits_processor(FakeGen(), "三人", bias=5.0, scope="content")
    out = proc(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 10))
    assert out[0, 3].item() == 5.0
    assert out[0, 7].item() == 5.0
    assert out[0, 9].item() == 0.0

    # numbers：只抬数字字符「三」，内容字「人」不动
    proc2 = make_logits_processor(FakeGen(), "三人", bias=5.0, scope="numbers")
    out2 = proc2(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 10))
    assert out2[0, 3].item() == 5.0
    assert out2[0, 7].item() == 0.0
