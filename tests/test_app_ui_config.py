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


def test_ui_config_has_no_prefilled_demo_examples():
    ui_config = load_ui_config()

    assert not hasattr(ui_config, "DEFAULT_INPUT")
    assert not hasattr(ui_config, "DEMO_EXAMPLES")


def test_style_cards_describe_remote_model_backends():
    ui_config = load_ui_config()
    titles = {card.title for card in ui_config.STYLE_CARDS}

    assert "文言文风格" in titles
    assert "鲁迅风格" in titles
    assert any("BART" in card.backend for card in ui_config.STYLE_CARDS)
    assert any("Qwen3-4B" in card.backend for card in ui_config.STYLE_CARDS)


def test_luxun_backend_names_specific_qwen3_4b_model():
    ui_config = load_ui_config()
    luxun_task = next(task for task in ui_config.STYLE_TASKS if task.label == "鲁迅风格")
    luxun_card = next(card for card in ui_config.STYLE_CARDS if card.title == "鲁迅风格")

    assert luxun_task.backend == "Qwen3-4B 鲁迅风格微调模型"
    assert luxun_card.backend == "Qwen3-4B LoRA"
