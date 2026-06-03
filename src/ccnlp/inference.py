from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    CLASSICAL_TO_MODERN = "classical_to_modern"
    MODERN_TO_CLASSICAL = "modern_to_classical"
    LUXUN_STYLE = "luxun_style"
    TANG_POEM_STYLE = "tang_poem_style"


@dataclass
class GenerationResult:
    task: TaskType
    input_text: str
    output_text: str
    note: str


class BaselineGenerator:
    """Rule-based fallback generator used before fine-tuned models are ready."""

    classical_map = {
        "学而时习之，不亦说乎": "学习了知识并且按时温习，不也是很快乐的事吗？",
        "有朋自远方来，不亦乐乎": "有朋友从远方来，不也是很令人高兴的吗？",
        "温故而知新": "温习旧知识，从而获得新的理解。",
        "己所不欲，勿施于人": "自己不愿承受的事情，也不要强加给别人。",
    }
    modern_map = {
        "学习后按时温习，不也很快乐吗": "学而时习之，不亦说乎？",
        "有朋友从远方来，不也很快乐吗": "有朋自远方来，不亦乐乎？",
        "自己不想要的，不要施加给别人": "己所不欲，勿施于人。",
    }

    def generate(
        self,
        text: str,
        task: TaskType | str,
        style_strength: float = 0.5,
    ) -> str:
        task_type = TaskType(task)
        cleaned = text.strip()
        if task_type == TaskType.CLASSICAL_TO_MODERN:
            return self.classical_map.get(cleaned, self._classical_to_modern(cleaned))
        if task_type == TaskType.MODERN_TO_CLASSICAL:
            return self.modern_map.get(cleaned, self._modern_to_classical(cleaned))
        if task_type == TaskType.LUXUN_STYLE:
            return self._luxun_style(cleaned, style_strength)
        if task_type == TaskType.TANG_POEM_STYLE:
            return self._tang_poem_style(cleaned)
        raise ValueError(f"Unsupported task: {task}")

    def generate_with_metadata(
        self,
        text: str,
        task: TaskType | str,
        style_strength: float = 0.5,
    ) -> GenerationResult:
        task_type = TaskType(task)
        return GenerationResult(
            task=task_type,
            input_text=text,
            output_text=self.generate(text, task_type, style_strength),
            note="当前为规则基线输出；训练完成后可替换为 Hugging Face 模型。",
        )

    def _classical_to_modern(self, text: str) -> str:
        replacements = {
            "吾": "我",
            "汝": "你",
            "之": "它",
            "曰": "说",
            "不亦": "不也是",
            "乎": "吗",
            "者": "的人",
            "也": "。",
        }
        output = text
        for old, new in replacements.items():
            output = output.replace(old, new)
        if not output.endswith(("。", "！", "？")):
            output += "。"
        return output

    def _modern_to_classical(self, text: str) -> str:
        replacements = {
            "我": "吾",
            "你": "汝",
            "他说": "其曰",
            "说": "曰",
            "的": "之",
            "吗": "乎",
            "快乐": "乐",
            "学习": "学",
            "温习": "习",
            "不要": "勿",
        }
        output = text
        for old, new in replacements.items():
            output = output.replace(old, new)
        output = output.rstrip("。！？")
        if not output.endswith(("乎", "也", "矣", "。")):
            output += "乎"
        return output + ("。" if not output.endswith("。") else "")

    def _luxun_style(self, text: str, style_strength: float) -> str:
        prefix = "我向来是不惮以最坏的恶意来推测这世事的。"
        if style_strength < 0.35:
            return f"{text}，这话里似乎另有一层冷意。"
        if style_strength < 0.7:
            return f"{text}。我向来觉得，沉默里也有铁一样的声响。"
        return f"{prefix}{text}，许多人沉默着，仿佛这沉默便是他们唯一的回答。"

    def _tang_poem_style(self, text: str) -> str:
        compact = text.replace("，", "").replace("。", "").replace("！", "").replace("？", "")
        first = compact[:5] or "清风入古道"
        second = compact[5:10] or "明月照归人"
        return f"{first}兮{second}，一川烟雨入诗心。"


# 任务前缀必须与训练时 build_task_examples 使用的完全一致，否则模型不认。
TASK_PREFIX = {
    TaskType.CLASSICAL_TO_MODERN: "古文翻今：",
    TaskType.MODERN_TO_CLASSICAL: "今文翻古：",
}


class ModelGenerator:
    """Loads a fine-tuned seq2seq checkpoint for bidirectional translation."""

    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> None:
        # 延迟导入，使得仅用 BaselineGenerator 时无需安装 torch/transformers。
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def generate(self, text: str, task: TaskType | str, max_source_length: int = 128) -> str:
        task_type = TaskType(task)
        prefix = TASK_PREFIX.get(task_type)
        if prefix is None:
            raise ValueError(f"ModelGenerator 仅支持双向翻译任务，收到：{task}")

        prompt = f"{prefix}{text.strip()}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_source_length,
        ).to(self.device)
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def generate_batch(
        self,
        texts: list[str],
        task: TaskType | str,
        max_source_length: int = 128,
        batch_size: int = 32,
    ) -> list[str]:
        """批量翻译（评估时用，比逐句快得多）。"""
        task_type = TaskType(task)
        prefix = TASK_PREFIX.get(task_type)
        if prefix is None:
            raise ValueError(f"ModelGenerator 仅支持双向翻译任务，收到：{task}")

        results: list[str] = []
        for start in range(0, len(texts), batch_size):
            chunk = [f"{prefix}{t.strip()}" for t in texts[start : start + batch_size]]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length,
                padding=True,
            ).to(self.device)
            with self._torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                )
            results.extend(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        return [r.strip() for r in results]

    def generate_with_metadata(self, text: str, task: TaskType | str) -> GenerationResult:
        task_type = TaskType(task)
        return GenerationResult(
            task=task_type,
            input_text=text,
            output_text=self.generate(text, task_type),
            note="微调模型输出。",
        )
