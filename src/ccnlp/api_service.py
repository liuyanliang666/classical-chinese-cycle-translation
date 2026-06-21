from __future__ import annotations

import os
from typing import Any

from ccnlp.inference import CausalLoraGenerator, GenerationResult, ModelGenerator, TaskType


SEQ2SEQ_ENV = "CCNLP_SEQ2SEQ_MODEL"
QWEN_BASE_ENV = "CCNLP_QWEN_BASE_MODEL"
LUXUN_ADAPTER_ENV = "CCNLP_LUXUN_ADAPTER"
LUXUN_INFERENCE_STRENGTH_MIN = 1.0
LUXUN_INFERENCE_STRENGTH_MAX = 1.3


def map_luxun_style_strength(frontend_strength: float) -> float:
    """Map frontend slider range [0, 1] to LoRA inference strength [1.0, 1.3]."""
    clamped = min(1.0, max(0.0, float(frontend_strength)))
    span = LUXUN_INFERENCE_STRENGTH_MAX - LUXUN_INFERENCE_STRENGTH_MIN
    return LUXUN_INFERENCE_STRENGTH_MIN + clamped * span


class GeneratorRegistry:
    """Lazy model registry shared by the API server.

    The API process keeps loaded models in memory across requests. Translation tasks
    use a seq2seq checkpoint; Lu Xun style transfer uses Qwen + LoRA.
    """

    def __init__(
        self,
        *,
        seq2seq_model: str | None = None,
        qwen_base_model: str | None = None,
        luxun_adapter: str | None = None,
        model_generator_cls: type[Any] = ModelGenerator,
        lora_generator_cls: type[Any] = CausalLoraGenerator,
    ) -> None:
        self.seq2seq_model = seq2seq_model
        self.qwen_base_model = qwen_base_model
        self.luxun_adapter = luxun_adapter
        self.model_generator_cls = model_generator_cls
        self.lora_generator_cls = lora_generator_cls
        self._seq2seq_generator: Any | None = None
        self._lora_generator: Any | None = None

    @classmethod
    def from_env(cls) -> "GeneratorRegistry":
        return cls(
            seq2seq_model=os.getenv(SEQ2SEQ_ENV),
            qwen_base_model=os.getenv(QWEN_BASE_ENV),
            luxun_adapter=os.getenv(LUXUN_ADAPTER_ENV),
        )

    def generate(
        self,
        text: str,
        task: str | TaskType,
        *,
        style_strength: float = 1.0,
    ) -> GenerationResult:
        task_type = TaskType(task)
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text must not be empty")

        if task_type in {TaskType.CLASSICAL_TO_MODERN, TaskType.MODERN_TO_CLASSICAL}:
            generator = self._get_seq2seq_generator()
            output = generator.generate(cleaned, task_type.value)
            return GenerationResult(
                task=task_type,
                input_text=cleaned,
                output_text=str(output),
                note="云端 Seq2Seq 模型输出。",
            )

        if task_type is TaskType.LUXUN_STYLE:
            generator = self._get_lora_generator()
            inference_strength = map_luxun_style_strength(style_strength)
            output = generator.generate(
                cleaned,
                task_type.value,
                style_strength=inference_strength,
            )
            return GenerationResult(
                task=task_type,
                input_text=cleaned,
                output_text=str(output),
                note="云端 Qwen LoRA 鲁迅风格模型输出。",
            )

        raise ValueError(f"Unsupported task: {task}")

    def _get_seq2seq_generator(self) -> Any:
        if not self.seq2seq_model:
            raise RuntimeError(f"{SEQ2SEQ_ENV} is required for translation tasks")
        if self._seq2seq_generator is None:
            self._seq2seq_generator = self.model_generator_cls(self.seq2seq_model)
        return self._seq2seq_generator

    def _get_lora_generator(self) -> Any:
        if not self.qwen_base_model:
            raise RuntimeError(f"{QWEN_BASE_ENV} is required for luxun_style")
        if not self.luxun_adapter:
            raise RuntimeError(f"{LUXUN_ADAPTER_ENV} is required for luxun_style")
        if self._lora_generator is None:
            self._lora_generator = self.lora_generator_cls(
                base_model=self.qwen_base_model,
                adapter_dir=self.luxun_adapter,
            )
        return self._lora_generator
