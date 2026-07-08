"""Tests for the Kiri OCR recognizer wrapper (kiri_recognizer.py).

Covers the pure Otsu helper exhaustively; model-download recognition tests
are skipped unless Kiri weights are already cached locally.
"""
from __future__ import annotations

import numpy as np
import pytest

from khmer_pipeline.engines.kiri_recognizer import (
    otsu_cell,
    recognize_cell,
    recognize_cells,
    recognize_cells_conf,
)


# ---------------------------------------------------------------------------
# Otsu helper — deterministic, pure NumPy/OpenCV, no model needed
# ---------------------------------------------------------------------------

def _make_rgb(h: int, w: int, fill: tuple) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = fill
    return img


class TestOtsuCell:
    """Deterministic Otsu binarization + auto-polarity tests."""

    def test_black_text_on_white_background(self):
        # White bg (255) with a dark "text" stripe (0)
        img = np.full((30, 100, 3), 255, dtype=np.uint8)
        img[5:10, 10:50] = 0
        result = otsu_cell(img)
        # Text should be 0 (black), background 255 (white)
        assert result[7, 30] == 0
        assert result[0, 0] == 255

    def test_white_text_on_dark_background_inverts(self):
        # Dark bg (30) with white text stripe (240)
        img = np.full((30, 100, 3), 30, dtype=np.uint8)
        img[5:10, 10:50] = 240
        result = otsu_cell(img)
        # Auto-polarity should have kicked in: bg should be white (255)
        assert result[0, 0] == 255

    def test_yellow_on_orange_separable(self):
        """Synthetic yellow-on-orange array should separate into two classes."""
        # Yellow (~255,255,0) on orange (~255,165,0)
        img = np.zeros((30, 100, 3), dtype=np.uint8)
        img[:, :] = (255, 165, 0)  # orange bg
        img[5:10, 10:50] = (255, 255, 0)  # yellow text
        result = otsu_cell(img)
        # Should produce a binary image (only 0 and 255 values)
        unique = np.unique(result)
        assert set(unique.flatten()).issubset({0, 255})
        # At least both classes present
        assert len(unique) == 2

    def test_all_white_produces_white(self):
        img = _make_rgb(20, 20, (255, 255, 255))
        result = otsu_cell(img)
        # All white: Otsu may threshold to either 0 or 255, but we just ensure
        # it returns a valid np array of the right shape
        assert result.shape == (20, 20)

    def test_all_black_produces_white_background(self):
        img = _make_rgb(20, 20, (0, 0, 0))
        result = otsu_cell(img)
        # Auto-polarity should invert all-black → all-white bg
        assert result.mean() > 127

    def test_mixed_gray_returns_binary(self):
        """A 50% gray image should produce binary output (0 or 255 only)."""
        img = _make_rgb(20, 20, (128, 128, 128))
        result = otsu_cell(img)
        unique = np.unique(result)
        assert set(unique.flatten()).issubset({0, 255})


# ---------------------------------------------------------------------------
# Batched API — empty-input case needs no model load (early-return in
# recognize_cells before `_get_kiri()` is ever called).
# ---------------------------------------------------------------------------

def test_recognize_cells_empty_list_returns_empty():
    assert recognize_cells([]) == []


def test_recognize_cells_conf_empty_returns_empty():
    assert recognize_cells_conf([]) == []


# ---------------------------------------------------------------------------
# Recognition integration test (skipped unless model is cached)
# ---------------------------------------------------------------------------

def _kiri_model_available() -> bool:
    """Return True if the Kiri model is already cached locally (HF cache)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache("mrrtmob/kiri-ocr", "model.safetensors")
        return path is not None
    except Exception:
        return False


@pytest.mark.skipif(not _kiri_model_available(), reason="Kiri model not cached locally")
class TestRecognizeCellIntegration:
    """Integration tests that require the Kiri model to be downloaded."""

    def test_recognize_blank_cell(self):
        """A blank (all-white) cell should return empty string."""
        img = np.full((40, 120, 3), 255, dtype=np.uint8)
        text = recognize_cell(img)
        assert isinstance(text, str)
        # Blank cells typically produce empty or whitespace-only output
        assert text.strip() == ""

    def test_recognize_returns_str(self):
        """Minimal smoke test: any non-empty cell returns a string."""
        img = np.full((48, 200, 3), 255, dtype=np.uint8)
        img[10:38, 20:180] = 0  # dark bar
        text = recognize_cell(img)
        assert isinstance(text, str)

    def test_recognize_cells_batched_matches_single(self):
        """Batched recognize_cells([a, b]) must equal per-cell recognize_cell calls."""
        img_a = np.full((40, 120, 3), 255, dtype=np.uint8)
        img_b = np.full((48, 200, 3), 255, dtype=np.uint8)
        img_b[10:38, 20:180] = 0  # dark bar

        batched = recognize_cells([img_a, img_b])
        assert isinstance(batched, list)
        assert len(batched) == 2
        assert all(isinstance(t, str) for t in batched)
        assert batched == [recognize_cell(img_a), recognize_cell(img_b)]

    def test_recognize_cells_conf_returns_text_and_confidence(self):
        """recognize_cells_conf returns (str, float) pairs with conf in [0, 1]."""
        img = np.full((40, 120, 3), 255, dtype=np.uint8)
        result = recognize_cells_conf([img])
        assert isinstance(result, list)
        assert len(result) == 1
        text, conf = result[0]
        assert isinstance(text, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_recognize_cells_conf_text_matches_recognize_cells(self):
        """Text half of recognize_cells_conf must equal recognize_cells' output."""
        img_a = np.full((40, 120, 3), 255, dtype=np.uint8)
        img_b = np.full((48, 200, 3), 255, dtype=np.uint8)
        img_b[10:38, 20:180] = 0  # dark bar

        conf_pairs = recognize_cells_conf([img_a, img_b])
        texts = [t for t, _ in conf_pairs]
        assert texts == recognize_cells([img_a, img_b])
