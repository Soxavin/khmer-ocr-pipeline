"""Kiri OCR recognizer wrapper — per-cell Otsu binarization + CTC recognition.

Provides the thin integration layer between the vendored Kiri model and the
surya_kiri hybrid engine. Exposes three public functions:

  otsu_cell(rgb)             — Otsu-threshold a cell crop, auto-inverting dark backgrounds
  recognize_cell(crop_rgb)   — Full pipeline for a single crop: Otsu → Kiri CTC decode → clean text
  recognize_cells(crops_rgb) — Batched equivalent of recognize_cell over a list of crops

Model weights are lazy-loaded on first call and cached for the process lifetime.
"""
from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .kiri_vendor.loader import load_kiri_model
from .kiri_vendor.model import preprocess_pil

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

_OTSU_INVERT_MEAN_THRESHOLD = 127  # mean below this → invert binary (auto-polarity)
_BATCH_SIZE = 64  # cap on cells stacked into a single model forward pass

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_kiri: Optional[tuple] = None
_kiri_load_failed = False  # cache load failure so we don't re-download per cell


def _get_kiri():
    """Lazy-load the Kiri recognizer (model + cfg + tokenizer), cached process-wide.

    A load failure is cached too: without this, a bad load would re-trigger the
    HuggingFace download for every one of the ~240 cells on a page.
    """
    global _kiri, _kiri_load_failed
    if _kiri_load_failed:
        raise RuntimeError("Kiri model load previously failed; not retrying this run.")
    if _kiri is None:
        try:
            _kiri = load_kiri_model(device="cpu", verbose=True)
        except Exception:
            _kiri_load_failed = True
            raise
    return _kiri


# ---------------------------------------------------------------------------
# Otsu helper
# ---------------------------------------------------------------------------

def otsu_cell(rgb: np.ndarray) -> np.ndarray:
    """Binarize an RGB cell crop with Otsu thresholding + auto-polarity.

    If the mean pixel value of the binarized image is below *OTSU_INVERT_MEAN_THRESHOLD*
    (i.e. the background came out dark), the image is inverted so the background
    stays white — this handles both dark-on-light and light-on-dark table cells.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if binary.mean() < _OTSU_INVERT_MEAN_THRESHOLD:
        binary = 255 - binary
    return binary


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognize_cell(crop_rgb: np.ndarray) -> str:
    """Recognize text in a single table-cell crop.

    Pipeline: Otsu binarization → Kiri CTC decode (fast path) → clean text.
    Delegates to `recognize_cells` so there is a single recognition code path.
    """
    return recognize_cells([crop_rgb])[0]


def recognize_cells(crops_rgb: list[np.ndarray]) -> list[str]:
    """Recognize text for a batch of table-cell crops in one or few model passes.

    Equivalent to ``[recognize_cell(c) for c in crops_rgb]`` but batches the
    Otsu-binarized crops into ``(N,1,IMG_H,IMG_W)`` tensors (chunked by
    `_BATCH_SIZE`) so `encode`/`ctc_head` run once per chunk instead of once
    per cell. Returns one string per input crop, in order. Any failure warns
    and falls back to `""` for the affected crops rather than raising.
    """
    if not crops_rgb:
        return []

    try:
        model, cfg, tokenizer = _get_kiri()
    except Exception as exc:
        warnings.warn(f"Kiri recognition failed: {exc}")
        return [""] * len(crops_rgb)

    results: list[str] = [""] * len(crops_rgb)
    for start in range(0, len(crops_rgb), _BATCH_SIZE):
        chunk = crops_rgb[start:start + _BATCH_SIZE]
        try:
            tensors = []
            for crop_rgb in chunk:
                binary = otsu_cell(crop_rgb)
                tensors.append(preprocess_pil(cfg, Image.fromarray(binary)))
            batch = torch.cat(tensors, dim=0)

            with torch.inference_mode():
                mem = model.encode(batch)
                logits = model.ctc_head(mem)
                pred_ids = logits.argmax(dim=-1)

            for i in range(len(chunk)):
                text = tokenizer.decode_ctc(pred_ids[i].tolist())
                if cfg.COLLAPSE_WHITESPACE:
                    text = " ".join(text.split())
                # Post-processing: strip whitespace, trailing dots (Kiri often
                # emits a '.' after numbers).
                results[start + i] = text.strip().rstrip(".").strip()
        except Exception as exc:
            warnings.warn(f"Kiri recognition failed: {exc}")
            # Affected chunk falls back to "" (already the default in results).

    return results
