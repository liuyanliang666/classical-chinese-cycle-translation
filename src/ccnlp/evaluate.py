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


def _char_ngrams(text: str, n: int) -> Counter[str]:
    compact = "".join(text.split())
    if len(compact) < n:
        return Counter()
    return Counter(compact[index : index + n] for index in range(len(compact) - n + 1))


def _validate_lengths(predictions: Sequence[str], references: Sequence[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
