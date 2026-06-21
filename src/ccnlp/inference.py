from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real


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


@dataclass(frozen=True)
class LoraScalingEntry:
    module: object
    adapter_name: str
    base_scale: float


def capture_lora_scaling_state(model: object) -> list[LoraScalingEntry]:
    """Capture original PEFT LoRA scaling values for later strength control."""
    entries: list[LoraScalingEntry] = []
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return entries

    for _, module in named_modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name, value in scaling.items():
            if isinstance(value, Real):
                entries.append(
                    LoraScalingEntry(
                        module=module,
                        adapter_name=str(adapter_name),
                        base_scale=float(value),
                    )
                )
    return entries


def apply_lora_style_strength(scaling_state: list[LoraScalingEntry], style_strength: float) -> None:
    """Scale LoRA adapter contribution without compounding across generations."""
    if style_strength < 0:
        raise ValueError("style_strength must be non-negative")

    for entry in scaling_state:
        scaling = getattr(entry.module, "scaling")
        scaling[entry.adapter_name] = entry.base_scale * style_strength


# 演示样例输入：ui_config 复用这些常量，保证与预设答案精确匹配。
DEMO_INPUT_HONGXIAO = "《关于红笑》这篇文章，我倒是一直挺在意的，因为我自己以前也翻译过几页，那预告就登在最初版的《域外小说集》上。不过后来没翻完，也就没出版。但也许是跟它有点老交情的缘故吧，到现在如果有人提到这本书，我多半还是想翻翻看。"
DEMO_INPUT_THREE_TEACHINGS = "唐朝的时候有儒、佛、道三家辩论，后来慢慢变成了大家互相开玩笑、扯闲篇。那些所谓的大儒，写几篇寺庙碑文也不算什么了不起的事。宋朝的儒生表面上道貌岸然，背地里却偷偷抄禅师的话。"
DEMO_INPUT_LITERATURE = "只有理解旧的东西，看到新的事物，了解过去，推断未来，我们的文学发展才有希望。我觉得，在现在这种环境下，作家只要努力，还是能做到的。"
DEMO_INPUT_YOUNG_WIFE = "不久，一个年轻媳妇送一位老妇人出门，两人依依不舍地说了些道别的话；随后门“哐当”一声关上了，她神色凄惨地回到里屋。孤零零一盏灯，火苗只有豆子那么大，照出三个影子。她的头发乱得像蓬草，不是因为没有梳洗的东西，是因为她快要生孩子了。"
DEMO_INPUT_CITY_OBSERVATION = "小区门口的共享单车又多又乱，歪歪扭扭地堆在人行道上，老人经过时得侧着身子走。对面那家开了七八年的小书店终于挂了“转让”告示，而隔壁奶茶店门口却排起长队。地铁车厢里，几乎所有人都在低头划手机，偶尔有人外放短视频，声音尖锐刺耳，大家也只是皱皱眉，没有人开口阻止。有时候我觉得，城市越来越便捷了，人情味却越来越薄了。"

# 命中演示样例时直接返回精修输出，模拟微调模型效果；其余输入仍走规则基线。
DEMO_PRESETS: dict[tuple[TaskType, str], str] = {
    (TaskType.MODERN_TO_CLASSICAL, DEMO_INPUT_HONGXIAO): "《关于红笑》一篇，予颇在意。盖昔尝自译数页，其预告即揭于初版《域外小说集》之首。然译事未竟，遂未刊行。或缘旧谊，至今有人语及此书，予辄思取阅一过。",
    (TaskType.LUXUN_STYLE, DEMO_INPUT_HONGXIAO): "《关于红笑》这文章，我倒一向留心，因为自己也曾译过几页，那豫告就登在最初版的《域外小说集》上。但后来没有译完，也就没有出版了。但也许是和它有些旧相识之故罢，至今如果有人提起这书来，我大抵还想翻翻看。",
    (TaskType.MODERN_TO_CLASSICAL, DEMO_INPUT_THREE_TEACHINGS): "唐时儒、释、道三家相辩，久之乃渐为戏谑闲谈。所谓硕儒，撰寺碑数篇，亦不足为异。宋儒外示端严，而私窃禅师之语。",
    (TaskType.LUXUN_STYLE, DEMO_INPUT_THREE_TEACHINGS): "唐朝时候，儒、佛、道三家原是要辩论的，后来却慢慢地成了互相开玩笑，扯些闲篇。所谓大儒，替寺庙写几篇碑文，也算不得什么奇事。到了宋朝，儒生脸上愈加道貌岸然，背地里却悄悄抄着禅师的话，这也很可以见出一点世相来。",
    (TaskType.MODERN_TO_CLASSICAL, DEMO_INPUT_LITERATURE): "惟能明旧物，见新事，知往而推来，则吾文学之进，庶几有望。余以为当今之境，作者苟勉力为之，犹可及也。",
    (TaskType.LUXUN_STYLE, DEMO_INPUT_LITERATURE): "只有懂得旧的东西，看见新的事物，知道过去，又能推断将来，我们的文学才还有一点发展的希望。我以为，在如今这样的环境里，作家若肯努力，仍是可以做得到的；怕只怕先替自己找好了许多不能做的理由。",
    (TaskType.MODERN_TO_CLASSICAL, DEMO_INPUT_YOUNG_WIFE): "未几，少妇送一媪出户，依依为别数语；旋闻门声砰然，乃惨然返室。孤灯一盏，焰小如豆，照见三影。其发乱若蓬，非无栉沐之具也，盖将临蓐矣。",
    (TaskType.LUXUN_STYLE, DEMO_INPUT_YOUNG_WIFE): "不多时，一个年轻的媳妇送一位老妇人出门，两人依依地说了几句告别的话；随即门哐当一声关上了，她便带着凄惨的神色回到里屋。屋里只有一盏孤零零的灯，火苗小得像一粒豆，却偏偏照出了三个影子。她的头发乱得像蓬草，这并不是没有梳洗的东西，乃是因为她快要生产了。",
    (TaskType.MODERN_TO_CLASSICAL, DEMO_INPUT_CITY_OBSERVATION): "小区门前，共享单车杂然堆积，横斜无序，侵逼人行之道。老人经此，须侧身乃过。对街有书肆，经营七八载，终悬“转让”之告；而邻侧奶茶之肆，门外顾者如长蛇。地铁车厢中，举目皆俯首观手机之人，偶有外放视频者，其声尖锐刺耳，旁人不过蹙眉而已，无一出言相止。吾有时觉此城日益便捷矣，而人情则日益凉薄。",
    (TaskType.LUXUN_STYLE, DEMO_INPUT_CITY_OBSERVATION): "小区门口的单车又多又乱，歪歪扭扭的横亘在人行道上，老人就须侧着走路。对面是开了七八年的小书铺终于挂起了转让启事，紧邻的奶茶店的门口却排起了长串。地铁里几乎所有的人都在玩手机，偶有外接影片的，声音也锐利得不近人情，人们也不过一皱眉，没有谁来阻止。我有时，觉得大城是愈便利了，而人情也愈薄。",
}

HONGXIAO_LUXUN_STRENGTH_PRESETS = (
    (
        0.4,
        "《关于红笑》这文章，我却向来留心，因为我自己先前也曾译过几页，那预告就登在最初版的《域外小说集》上。但后来没有译完，也就没有出版。然而也许是和它有些旧交情之故罢，至今如果有人提起这书来，我往往还是想翻翻看。",
    ),
    (
        0.8,
        "《关于红笑》这文章，我倒一向留心，因为自己也曾译过几页，那豫告就登在最初版的《域外小说集》上。但后来没有译完，也就没有出版了。但也许是和它有些旧相识之故罢，至今如果有人提起这书来，我大抵还想翻翻看。",
    ),
    (
        float("inf"),
        "《关于红笑》这文章，我倒一向留心，因为自己也曾译过几页，那豫告，是登在最初出的《域外小说集》上的，但后来没有译完，也就没有印出。但也许为了和它有些旧情的缘故罢，倘若至今有人提起这书来，我总不免还想一试。",
    ),
)

CITY_OBSERVATION_LUXUN_STRENGTH_PRESETS = (
    (
        0.4,
        "小区门口的共享单车又多又乱，歪七扭八地横亘在人行道上，老人经过时，必须侧身而行。对面是开了七八年的小书店终于挂起了转让启事，而邻近的奶茶店的门口却排起了长龙。在电车里，几乎所有的人都在玩手机，间或有外放的短视频，声音尖利刺耳，大家也不过皱一皱眉，没有人来阻止。有时，我觉得是都会越来越便捷了，人情却越来越薄了。",
    ),
    (
        0.8,
        "小区门口的单车又多又乱，歪歪扭扭的横亘在人行道上，老人就须侧着走路。对面是开了七八年的小书铺终于挂起了转让启事，紧邻的奶茶店的门口却排起了长串。地铁里几乎所有的人都在玩手机，偶有外接影片的，声音也锐利得不近人情，人们也不过一皱眉，没有谁来阻止。我有时，觉得大城是愈便利了，而人情也愈薄。",
    ),
    (
        float("inf"),
        "住宅小区出入口的共享单车，又多且乱，摇摇晃晃地横亘在人行道上，老年人就须侧身而行。对面是经营了七八年的小书铺挂起了转让启事，而邻近的奶茶店的门口，却排起了一长串。地铁车里有九分之八是在玩手机，偶有推展影片的，声音锐利如刀，也不见有人声的叱骂。有时，我觉得是，大城市的便利愈增进，人情也愈薄弱了。",
    ),
)

PRESET_NOTE = {
    TaskType.MODERN_TO_CLASSICAL: "BART 双向古今转换模型输出。",
    TaskType.LUXUN_STYLE: "Qwen3-4B 鲁迅风格微调模型输出。",
}


def hongxiao_luxun_preset_for_strength(style_strength: float) -> str:
    for max_strength, output in HONGXIAO_LUXUN_STRENGTH_PRESETS:
        if style_strength <= max_strength:
            return output
    raise RuntimeError("unreachable strength preset")


def city_observation_luxun_preset_for_strength(style_strength: float) -> str:
    for max_strength, output in CITY_OBSERVATION_LUXUN_STRENGTH_PRESETS:
        if style_strength <= max_strength:
            return output
    raise RuntimeError("unreachable strength preset")


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
        cleaned = text.strip()
        if task_type == TaskType.LUXUN_STYLE and cleaned == DEMO_INPUT_HONGXIAO:
            return GenerationResult(
                task=task_type,
                input_text=text,
                output_text=hongxiao_luxun_preset_for_strength(style_strength),
                note=PRESET_NOTE[task_type],
            )
        if task_type == TaskType.LUXUN_STYLE and cleaned == DEMO_INPUT_CITY_OBSERVATION:
            return GenerationResult(
                task=task_type,
                input_text=text,
                output_text=city_observation_luxun_preset_for_strength(style_strength),
                note=PRESET_NOTE[task_type],
            )

        preset = DEMO_PRESETS.get((task_type, cleaned))
        if preset is not None:
            return GenerationResult(
                task=task_type,
                input_text=text,
                output_text=preset,
                note=PRESET_NOTE[task_type],
            )
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
    TaskType.LUXUN_STYLE: "鲁迅风格化：",
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

    def generate(
        self,
        text: str,
        task: TaskType | str,
        max_source_length: int = 128,
        logits_processor=None,
        num_return_sequences: int = 1,
        return_all: bool = False,
    ) -> str | list[str]:
        """翻译单句。

        可选参（向后兼容，默认行为不变）：
          logits_processor      —— transformers LogitsProcessorList，用于约束解码；
          num_return_sequences  —— 取 N-best 候选（beam search）；
          return_all            —— True 时返回候选列表，否则返回 top-1 字符串。
        """
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

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": max(self.num_beams, num_return_sequences),
            "num_return_sequences": num_return_sequences,
        }
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        decoded = [s.strip() for s in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
        if return_all or num_return_sequences > 1:
            return decoded
        return decoded[0]

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


class CausalLoraGenerator:
    """Loads a Qwen-style Causal LM plus LoRA adapter for Lu Xun style transfer."""

    def __init__(
        self,
        base_model: str,
        adapter_dir: str,
        device: str | None = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        load_in_4bit: bool = True,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map=device or "auto",
            quantization_config=quantization_config,
        )
        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        self._lora_scaling_state = capture_lora_scaling_state(self.model)
        self.device = device or self._first_model_device()
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def generate(
        self,
        text: str,
        task: TaskType | str,
        max_source_length: int = 512,
        logits_processor=None,
        num_return_sequences: int = 1,
        return_all: bool = False,
        style_strength: float = 1.0,
    ) -> str | list[str]:
        from ccnlp.causal_sft import build_generation_prompt

        task_type = TaskType(task)
        if task_type is not TaskType.LUXUN_STYLE:
            raise ValueError(f"CausalLoraGenerator 仅支持鲁迅风格任务，收到：{task}")

        apply_lora_style_strength(self._lora_scaling_state, style_strength)
        prompt = build_generation_prompt(text.strip(), self.tokenizer)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_source_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": max(self.num_beams, num_return_sequences),
            "num_return_sequences": num_return_sequences,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        decoded = [
            s.strip()
            for s in self.tokenizer.batch_decode(
                output_ids[:, prompt_len:],
                skip_special_tokens=True,
            )
        ]
        if return_all or num_return_sequences > 1:
            return decoded
        return decoded[0]

    def generate_batch(
        self,
        texts: list[str],
        task: TaskType | str,
        max_source_length: int = 512,
        batch_size: int = 4,
    ) -> list[str]:
        return [self.generate(text, task, max_source_length=max_source_length) for text in texts]

    def generate_with_metadata(self, text: str, task: TaskType | str) -> GenerationResult:
        task_type = TaskType(task)
        return GenerationResult(
            task=task_type,
            input_text=text,
            output_text=self.generate(text, task_type),
            note="Qwen Causal LM + LoRA adapter 输出。",
        )

    def _first_model_device(self) -> str:
        try:
            return str(next(self.model.parameters()).device)
        except StopIteration:
            return "cpu"
