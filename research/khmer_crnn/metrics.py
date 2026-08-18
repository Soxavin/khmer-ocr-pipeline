"""Greedy CTC decoding + Levenshtein character error rate (CER).

Vendored rather than pulled from a dependency: both algorithms are short,
well understood, and keeping them here means this experiment has zero extra
metric-library deps (per CLAUDE.md dep-hygiene — don't add a package for
15 lines of code).
"""
from __future__ import annotations

import torch


def greedy_ctc_decode(log_probs: torch.Tensor, i2c: dict[int, str], blank: int = 0) -> list[str]:
    """Greedy CTC decode: argmax per timestep, collapse repeats, drop blanks.

    Args:
        log_probs: (T, B, V) log-softmax output from the model.
        i2c: index -> character map (vocab indices start at 1; 0 is blank).
        blank: the CTC blank index (default 0, matching training).
    Returns:
        List of length B, one decoded string per batch item.
    """
    # (T, B, V) -> (B, T) best-path indices
    best_path = log_probs.argmax(dim=2).transpose(0, 1).tolist()
    decoded = []
    for path in best_path:
        chars = []
        prev = None
        for idx in path:
            if idx != prev and idx != blank:
                chars.append(i2c.get(idx, ""))
            prev = idx
        decoded.append("".join(chars))
    return decoded


def levenshtein(a: str, b: str) -> int:
    """Classic O(len(a)*len(b)) edit distance with a rolling 1-D DP row."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            curr_row[j] = min(insert_cost, delete_cost, replace_cost)
        prev_row = curr_row
    return prev_row[-1]


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    """Corpus-level CER = sum(edit distances) / sum(reference lengths).

    OOV characters in a reference (chars dropped because they weren't in the
    train-only vocab) simply can never be produced by the decoder, so they
    naturally count as errors here — no special-casing needed.
    """
    total_edits = 0
    total_len = 0
    for pred, ref in zip(predictions, references):
        total_edits += levenshtein(pred, ref)
        total_len += len(ref)
    if total_len == 0:
        return 0.0
    return total_edits / total_len
