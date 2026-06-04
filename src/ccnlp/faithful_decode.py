"""推理期抗幻觉解码（无需重训）。

两种机制，可单用或合用：
  1) source-copy 偏置解码（`SourceCopyBiasProcessor`）：对「源句中出现的数字 /
     内容字」对应的词表 token 抬高 logits，鼓励数字、专名被「抄」而非「编」。
  2) N-best 忠实度重排：beam 取 k 个候选，按「源端忠实度 + 回环一致性 − 照抄惩罚」
     选最优。重排**只用源句**（不碰参考，避免测试集泄漏），其中回环一致性把
     训练期失败的 cycle 信号重新用作解码期选择准则。

注意：重排打分**不使用参考译文**，参考仅用于事后评测（见 `hallucination.py`）。

mode ∈ {baseline, copybias, rerank, both}。逐句运行（batch_size=1），仅 498 句、开销可接受。
"""

from __future__ import annotations

from ccnlp.evaluate import normalized_edit_distance
from ccnlp.hallucination import _is_content_char, _NUMERAL_CHARS, faithfulness_penalty
from ccnlp.inference import TASK_PREFIX, ModelGenerator, TaskType

_REVERSE_TASK = {
    "classical_to_modern": "modern_to_classical",
    "modern_to_classical": "classical_to_modern",
}


def make_logits_processor(generator: ModelGenerator, source_text: str, bias: float, scope: str = "content"):
    """构造一个对「源句相关 token」加偏置的 LogitsProcessorList。

    scope: "numbers"=仅数字字符 token；"content"=数字+内容字 token（默认）；"all"=全部源 token。
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    special = set(generator.tokenizer.all_special_ids)
    ids = generator.tokenizer(source_text, add_special_tokens=False)["input_ids"]
    keep: set[int] = set()
    for tid in ids:
        if tid in special:
            continue
        tok = generator.tokenizer.decode([tid]).strip()
        if not tok:
            continue
        if scope == "all":
            keep.add(tid)
        elif scope == "numbers":
            if any(c in _NUMERAL_CHARS for c in tok):
                keep.add(tid)
        else:  # content
            if any(c in _NUMERAL_CHARS or _is_content_char(c) for c in tok):
                keep.add(tid)

    bias_ids = torch.tensor(sorted(keep), dtype=torch.long)

    class SourceCopyBiasProcessor(LogitsProcessor):
        def __init__(self, token_ids, bias_value: float) -> None:
            self.token_ids = token_ids
            self.bias = bias_value

        def __call__(self, input_ids, scores):
            if self.token_ids.numel():
                scores[:, self.token_ids.to(scores.device)] += self.bias
            return scores

    return LogitsProcessorList([SourceCopyBiasProcessor(bias_ids, bias)])


def faithfulness_score(
    source: str,
    candidate: str,
    generator: ModelGenerator | None = None,
    reverse_task: str | None = None,
    w_halluc: float = 1.0,
    w_roundtrip: float = 1.0,
    w_copy: float = 1.0,
) -> float:
    """候选译文的忠实度打分（越大越好，**不使用参考**）。

    score = w_rt·回环相似度(译回源 vs 源) − w_halluc·幻觉惩罚 − w_copy·照抄相似度
    照抄相似度 = 1−归一化编辑距离(源, 候选)，惩罚「直接抄原文」防止重排选回复制坍缩。
    """
    penalty = w_halluc * faithfulness_penalty(source, candidate, reference=None)
    copy_sim = 1.0 - normalized_edit_distance(source, candidate)
    score = -penalty - w_copy * copy_sim
    if generator is not None and reverse_task is not None:
        back = generator.generate(candidate, reverse_task)
        rt_sim = 1.0 - normalized_edit_distance(source, back)
        score += w_roundtrip * rt_sim
    return score


def rerank(
    source: str,
    candidates: list[str],
    generator: ModelGenerator | None = None,
    reverse_task: str | None = None,
    **weights,
) -> str:
    """从候选里挑忠实度最高者；同分时保留 beam 顺序（模型偏好）。"""
    best, best_score = candidates[0], float("-inf")
    for cand in candidates:
        s = faithfulness_score(source, cand, generator, reverse_task, **weights)
        if s > best_score:
            best, best_score = cand, s
    return best


def faithful_translate(
    generator: ModelGenerator,
    texts: list[str],
    task: TaskType | str,
    mode: str = "rerank",
    k: int = 5,
    bias: float = 3.0,
    scope: str = "content",
    use_roundtrip: bool = True,
) -> list[str]:
    """按 mode 翻译一批句子（逐句）。mode ∈ {baseline, copybias, rerank, both}。"""
    task = TaskType(task).value if isinstance(task, TaskType) else str(task)
    if task not in TASK_PREFIX and TaskType(task) not in TASK_PREFIX:
        raise ValueError(f"不支持的任务：{task}")
    reverse_task = _REVERSE_TASK[task] if use_roundtrip else None

    out: list[str] = []
    for text in texts:
        proc = (
            make_logits_processor(generator, text, bias, scope)
            if mode in ("copybias", "both")
            else None
        )
        if mode == "baseline":
            out.append(generator.generate(text, task))
        elif mode == "copybias":
            out.append(generator.generate(text, task, logits_processor=proc))
        else:  # rerank / both
            cands = generator.generate(
                text, task, logits_processor=proc, num_return_sequences=k, return_all=True
            )
            out.append(rerank(text, cands, generator, reverse_task))
    return out
