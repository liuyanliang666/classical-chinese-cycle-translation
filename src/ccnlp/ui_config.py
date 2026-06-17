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
class DemoExample:
    title: str
    text: str
    task: TaskType


@dataclass(frozen=True)
class StyleCard:
    title: str
    backend: str
    description: str


DEFAULT_INPUT = "今天的天气很好，我想出去走走，也想把最近发生的事情认真想一想。"

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

DEMO_EXAMPLES = [
    DemoExample(
        title="日常叙述",
        text="今天的天气很好，我想出去走走，也想把最近发生的事情认真想一想。",
        task=TaskType.MODERN_TO_CLASSICAL,
    ),
    DemoExample(
        title="社会观察",
        text="街上很多人沉默地走过，没有人愿意先开口，好像每个人都在等待别人做决定。",
        task=TaskType.LUXUN_STYLE,
    ),
    DemoExample(
        title="学习计划",
        text="我准备每天复习一点旧知识，再把新的内容整理出来，这样才能慢慢进步。",
        task=TaskType.MODERN_TO_CLASSICAL,
    ),
    DemoExample(
        title="人群与沉默",
        text="大家都知道问题存在，可是会议结束以后，所有人还是像往常一样安静地离开了。",
        task=TaskType.LUXUN_STYLE,
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
