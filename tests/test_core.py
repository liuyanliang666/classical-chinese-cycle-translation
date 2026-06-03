from pathlib import Path

import pytest

from ccnlp.evaluate import bleu_score, chrf_score, exact_match_rate
from ccnlp.inference import BaselineGenerator, TaskType
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
