from __future__ import annotations

from dataclasses import dataclass

from ccnlp.inference import TaskType


@dataclass(frozen=True)
class StyleTask:
    label: str
    task: TaskType
    description: str
    backend: str


@dataclass(frozen=True)
class StyleCard:
    title: str
    backend: str
    description: str


STYLE_TASKS = [
    StyleTask(
        label="文言文风格",
        task=TaskType.MODERN_TO_CLASSICAL,
        description="面向现代文输入的文言化表达。",
        backend="BART 双向古今转换模型",
    ),
    StyleTask(
        label="鲁迅风格",
        task=TaskType.LUXUN_STYLE,
        description="面向现代文输入的文学风格改写。",
        backend="Qwen3-4B 鲁迅风格微调模型",
    ),
]

STYLE_CARDS = [
    StyleCard(
        title="文言文风格",
        backend="BART",
        description="使用双向训练过的古今转换模型，展示阶段固定调用现代文到文言文方向。",
    ),
    StyleCard(
        title="鲁迅风格",
        backend="Qwen3-4B LoRA",
        description="使用鲁迅语料微调后的生成模型，突出句式、语气和批判性表达特征。",
    ),
]
