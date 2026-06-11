"""Simple OCR evaluation metrics: CER and WER."""

from __future__ import annotations


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance without external dependencies."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def character_error_rate(predicted: str, ground_truth: str) -> float:
    """CER = character edit distance / ground-truth character count."""
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    return levenshtein_distance(predicted, ground_truth) / len(ground_truth)


def word_error_rate(predicted: str, ground_truth: str) -> float:
    """WER = word edit distance / ground-truth word count."""
    gt_words = ground_truth.split()
    pred_words = predicted.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return levenshtein_distance("\n".join(pred_words), "\n".join(gt_words)) / len(gt_words)
