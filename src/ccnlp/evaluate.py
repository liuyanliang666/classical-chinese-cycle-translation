from __future__ import annotations

from collections import Counter
from typing import Sequence


def exact_match_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    _validate_lengths(predictions, references)
    if not predictions:
        return 0.0
    matches = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
    return matches / len(predictions)


def chrf_score(predictions: Sequence[str], references: Sequence[str], n: int = 6, beta: float = 2.0) -> float:
    """Small dependency-free ChrF implementation for quick coursework baselines."""
    _validate_lengths(predictions, references)
    if not predictions:
        return 0.0

    scores = []
    for prediction, reference in zip(predictions, references):
        precisions = []
        recalls = []
        for order in range(1, n + 1):
            pred_counts = _char_ngrams(prediction, order)
            ref_counts = _char_ngrams(reference, order)
            overlap = sum((pred_counts & ref_counts).values())
            pred_total = sum(pred_counts.values())
            ref_total = sum(ref_counts.values())
            precisions.append(overlap / pred_total if pred_total else 0.0)
            recalls.append(overlap / ref_total if ref_total else 0.0)

        precision = sum(precisions) / n
        recall = sum(recalls) / n
        if precision == 0.0 and recall == 0.0:
            scores.append(0.0)
        else:
            beta_sq = beta * beta
            scores.append((1 + beta_sq) * precision * recall / (beta_sq * precision + recall))
    return round(100 * sum(scores) / len(scores), 4)


def bleu_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Corpus BLEU via sacrebleu with Chinese (character) tokenization."""
    _validate_lengths(predictions, references)
    if not predictions:
        return 0.0
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(list(predictions), [list(references)], tokenize="zh")
    return round(bleu.score, 4)


def bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str = "zh",
    model_type: str | None = None,
    batch_size: int = 64,
) -> float:
    """Mean BERTScore F1 (×100). Downloads a Chinese BERT on first use."""
    _validate_lengths(predictions, references)
    if not predictions:
        return 0.0
    from bert_score import score as _bert_score

    _, _, f1 = _bert_score(
        list(predictions),
        list(references),
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
    )
    return round(float(f1.mean()) * 100, 4)


def bertscore_f1_list(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str = "zh",
    model_type: str | None = None,
    batch_size: int = 64,
) -> list[float]:
    """每条样本的 BERTScore F1（0~1），用于分组分析。"""
    _validate_lengths(predictions, references)
    if not predictions:
        return []
    from bert_score import score as _bert_score

    _, _, f1 = _bert_score(
        list(predictions),
        list(references),
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
    )
    return [float(x) for x in f1]


# ---- 回环一致性（RTC）的三层组件，均归一化到 [0,1] ----

_SENTENCE_ENDINGS = "。！？；!?;"
# 常见文言虚词，用于按虚词密度分组。
FUNCTION_WORDS = set("之乎者也矣焉哉而以于其为则乃且夫盖故所与於兮耳邪欤乎哉")


def normalized_edit_distance(a: str, b: str) -> float:
    """字符级 Levenshtein 距离 / 较长串长度，返回 [0,1]（0=完全相同）。"""
    a = "".join(a.split())
    b = "".join(b.split())
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[len(b)] / max(len(a), len(b))


def literal_consistency(a: str, b: str) -> float:
    """字面一致性 = 0.5×ChrF + 0.5×(1−归一化编辑距离)，[0,1]。"""
    chrf = chrf_score([a], [b]) / 100.0
    return 0.5 * chrf + 0.5 * (1.0 - normalized_edit_distance(a, b))


def count_sentences(text: str) -> int:
    return max(sum(1 for ch in text if ch in _SENTENCE_ENDINGS), 1)


def structural_consistency(a: str, b: str) -> float:
    """结构一致性 = 0.5×长度比 + 0.5×句数相似度，[0,1]。"""
    la, lb = len(a), len(b)
    length_ratio = min(la, lb) / max(la, lb) if max(la, lb) else 1.0
    sa, sb = count_sentences(a), count_sentences(b)
    sentence_sim = 1.0 - abs(sa - sb) / max(sa, sb)
    return 0.5 * length_ratio + 0.5 * sentence_sim


def function_word_ratio(text: str) -> float:
    """文言虚词占比，[0,1]。"""
    compact = "".join(text.split())
    if not compact:
        return 0.0
    return sum(1 for ch in compact if ch in FUNCTION_WORDS) / len(compact)


def rtc_score(literal: float, semantic: float, structural: float) -> float:
    """综合回环一致性：0.2×字面 + 0.6×语义 + 0.2×结构。"""
    return 0.2 * literal + 0.6 * semantic + 0.2 * structural


def _char_ngrams(text: str, n: int) -> Counter[str]:
    compact = "".join(text.split())
    if len(compact) < n:
        return Counter()
    return Counter(compact[index : index + n] for index in range(len(compact) - n + 1))


def _validate_lengths(predictions: Sequence[str], references: Sequence[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
