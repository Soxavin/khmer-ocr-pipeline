"""Kiri OCR recognizer wrapper — per-cell Otsu binarization + CTC recognition.

Provides the thin integration layer between the vendored Kiri model and the
surya_kiri hybrid engine. Exposes two public functions:

  otsu_cell(rgb)        — Otsu-threshold a cell crop, auto-inverting dark backgrounds
  recognize_cell(crop_rgb) — Full pipeline: Otsu → Kiri CTC decode → clean text

Model weights are lazy-loaded on first call and cached for the process lifetime.
"""
from __future__ import annotations

import os
import tempfile
import warnings
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .kiri_vendor.loader import load_kiri_model
from .kiri_vendor.model import preprocess_pil, greedy_ctc_decode

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

_OTSU_INVERT_MEAN_THRESHOLD = 127  # mean below this → invert binary (auto-polarity)

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

    Pipeline: Otsu binarization → save as temp PNG → Kiri CTC decode (fast path).
    The temp file is cleaned up immediately after recognition.
    Returns the recognised string (stripped, trailing '.' removed).
    """
    binary = otsu_cell(crop_rgb)

    # Kiri's recognize_single_line_image takes a file path, so save to a temp PNG.
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(binary).save(tmp_path)

        model, cfg, tokenizer = _get_kiri()
        pil_img = Image.open(tmp_path).convert("L")
        tensor = preprocess_pil(cfg, pil_img)
        text, _confidence = greedy_ctc_decode(model, tensor, tokenizer, cfg)
    except Exception as exc:
        warnings.warn(f"Kiri recognition failed: {exc}")
        text = ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Post-processing: strip whitespace, trailing dots (Kiri often emits a '.' after numbers)
    return text.strip().rstrip(".").strip()
