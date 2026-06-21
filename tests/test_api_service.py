import pytest

from ccnlp.api_service import GeneratorRegistry, map_luxun_style_strength
from ccnlp.inference import GenerationResult, TaskType


class FakeSeq2SeqGenerator:
    calls = []

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.calls.append(("init", model_dir))

    def generate(self, text: str, task: str):
        self.calls.append(("generate", text, task))
        return f"{task}:{text}"


class FakeLoraGenerator:
    calls = []

    def __init__(self, base_model: str, adapter_dir: str):
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.calls.append(("init", base_model, adapter_dir))

    def generate(self, text: str, task: str, style_strength: float):
        self.calls.append(("generate", text, task, style_strength))
        return f"{style_strength}:{text}"


def test_generator_registry_uses_seq2seq_model_for_translation_tasks():
    FakeSeq2SeqGenerator.calls = []
    registry = GeneratorRegistry(
        seq2seq_model="/models/seq2seq",
        model_generator_cls=FakeSeq2SeqGenerator,
    )

    result = registry.generate("学习后按时温习", "modern_to_classical")

    assert result.output_text == "modern_to_classical:学习后按时温习"
    assert result.note == "云端 Seq2Seq 模型输出。"
    assert FakeSeq2SeqGenerator.calls == [
        ("init", "/models/seq2seq"),
        ("generate", "学习后按时温习", "modern_to_classical"),
    ]


def test_generator_registry_uses_lora_model_for_luxun_style():
    FakeLoraGenerator.calls = []
    registry = GeneratorRegistry(
        qwen_base_model="/models/qwen",
        luxun_adapter="/models/lora",
        lora_generator_cls=FakeLoraGenerator,
    )

    result = registry.generate("街上很多人沉默", "luxun_style", style_strength=0.75)

    assert result.output_text == "1.225:街上很多人沉默"
    assert result.note == "云端 Qwen LoRA 鲁迅风格模型输出。"
    assert FakeLoraGenerator.calls == [
        ("init", "/models/qwen", "/models/lora"),
        ("generate", "街上很多人沉默", "luxun_style", 1.225),
    ]


@pytest.mark.parametrize(
    ("frontend_strength", "expected_inference_strength"),
    [
        (-1.0, 1.0),
        (0.0, 1.0),
        (0.5, 1.15),
        (1.0, 1.3),
        (2.0, 1.3),
    ],
)
def test_map_luxun_style_strength_scales_frontend_range_to_inference_range(
    frontend_strength,
    expected_inference_strength,
):
    assert map_luxun_style_strength(frontend_strength) == pytest.approx(expected_inference_strength)


def test_generator_registry_requires_configured_model_path():
    registry = GeneratorRegistry()

    with pytest.raises(RuntimeError, match="CCNLP_SEQ2SEQ_MODEL"):
        registry.generate("学习后按时温习", "modern_to_classical")


def test_api_generate_accepts_json_body():
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    from fastapi.testclient import TestClient
    from ccnlp.api_server import create_app

    class FakeRegistry:
        def generate(self, text: str, task: str, style_strength: float):
            assert text == "学习后按时温习"
            assert task == "modern_to_classical"
            assert style_strength == 1.0
            return GenerationResult(
                task=TaskType.MODERN_TO_CLASSICAL,
                input_text=text,
                output_text="学而时习之",
                note="云端 Seq2Seq 模型输出。",
            )

    client = TestClient(create_app(FakeRegistry()))

    response = client.post(
        "/generate",
        json={
            "text": "学习后按时温习",
            "task": "modern_to_classical",
            "style_strength": 1.0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "output": "学而时习之",
        "note": "云端 Seq2Seq 模型输出。",
    }
