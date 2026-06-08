from importlib import import_module

import pytest

from ccnlp.inference import TaskType


def load_ui_config():
    try:
        return import_module("ccnlp.ui_config")
    except ModuleNotFoundError as exc:
        pytest.fail(f"UI config module is missing: {exc}")


def test_demo_ui_only_exposes_final_presentation_tasks():
    ui_config = load_ui_config()

    labels = [task.label for task in ui_config.STYLE_TASKS]

    assert labels == ["文言文风格", "鲁迅风格"]
    assert ui_config.STYLE_TASKS[0].task == TaskType.MODERN_TO_CLASSICAL
    assert ui_config.STYLE_TASKS[1].task == TaskType.LUXUN_STYLE


def test_demo_examples_match_available_styles():
    ui_config = load_ui_config()
    available_tasks = {task.task for task in ui_config.STYLE_TASKS}

    assert ui_config.DEMO_EXAMPLES
    assert all(example.task in available_tasks for example in ui_config.DEMO_EXAMPLES)


def test_style_cards_describe_real_model_plan():
    ui_config = load_ui_config()
    titles = {card.title for card in ui_config.STYLE_CARDS}

    assert "文言文风格" in titles
    assert "鲁迅风格" in titles
    assert any("BART" in card.backend for card in ui_config.STYLE_CARDS)
    assert any("Qwen" in card.backend for card in ui_config.STYLE_CARDS)
