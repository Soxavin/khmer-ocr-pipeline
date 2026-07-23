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
# A5 regression: HF downloads must be pinned to a known-good revision so an
# upstream re-push of model.safetensors cannot silently swap the weights.
# ---------------------------------------------------------------------------

def test_hf_download_pins_revision():
    from unittest.mock import patch
    import khmer_pipeline.engines.kiri_vendor.loader as loader

    calls = []

    def _fake_download(repo_id, filename, revision=None):
        calls.append((filename, revision))
        if filename == "model.safetensors":
            return "/fake/path/model.safetensors"
        raise FileNotFoundError  # skip the vocab candidates

    with patch("huggingface_hub.hf_hub_download", _fake_download):
        path = loader._download_from_hf(loader._HF_REPO)

    assert path == "/fake/path/model.safetensors"
    assert calls, "expected hf_hub_download to be called"
    # model AND vocab downloads must all use the same pinned revision
    assert all(rev == loader._HF_REVISION for _, rev in calls)
    assert "model.safetensors" in {fn for fn, _ in calls}


# ---------------------------------------------------------------------------
# A3: failure visibility (warning_sink) + per-run latch reset
# ---------------------------------------------------------------------------

def _boom():
    raise RuntimeError("no model")


def test_recognize_cells_conf_sink_collects_load_failure(monkeypatch):
    import khmer_pipeline.engines.kiri_recognizer as kr
    monkeypatch.setattr(kr, "_get_kiri", _boom)
    sink: list[str] = []
    crops = [np.full((10, 20, 3), 255, dtype=np.uint8)]
    result = kr.recognize_cells_conf(crops, warning_sink=sink)
    assert result == [("", 0.0)]
    assert sink and "unavailable" in sink[0]


def test_recognize_cells_conf_warns_when_no_sink(monkeypatch):
    import warnings as _w
    import khmer_pipeline.engines.kiri_recognizer as kr
    monkeypatch.setattr(kr, "_get_kiri", _boom)
    crops = [np.full((10, 20, 3), 255, dtype=np.uint8)]
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        kr.recognize_cells_conf(crops)  # no sink → warnings.warn path (unchanged)
    assert any("unavailable" in str(x.message) for x in caught)


def test_reset_kiri_failure_clears_latch():
    import khmer_pipeline.engines.kiri_recognizer as kr
    kr._kiri_load_failed = True
    kr.reset_kiri_failure()
    assert kr._kiri_load_failed is False


def test_latch_then_reset_allows_retry(monkeypatch):
    """After a load failure the latch short-circuits; reset must re-enable a retry."""
    import khmer_pipeline.engines.kiri_recognizer as kr
    calls = {"n": 0}

    def _fail_load(device="cpu", verbose=False):
        calls["n"] += 1
        raise RuntimeError("net down")

    monkeypatch.setattr(kr, "load_kiri_model", _fail_load)
    monkeypatch.setattr(kr, "_kiri", None)
    monkeypatch.setattr(kr, "_kiri_load_failed", False)

    import pytest
    with pytest.raises(Exception):
        kr._get_kiri()          # first attempt loads → fails, sets latch
    with pytest.raises(RuntimeError, match="previously failed"):
        kr._get_kiri()          # latched: no second load attempt
    assert calls["n"] == 1

    kr.reset_kiri_failure()
    with pytest.raises(Exception):
        kr._get_kiri()          # retry after reset → loads again
    assert calls["n"] == 2


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


# --- local fine-tuned weights override (KHMER_KIRI_WEIGHTS, Track B) ---

def test_local_weights_unset_no_default_returns_none(monkeypatch, tmp_path):
    import khmer_pipeline.engines.kiri_vendor.loader as loader
    monkeypatch.delenv("KHMER_KIRI_WEIGHTS", raising=False)
    monkeypatch.setattr(loader, "_DEFAULT_WEIGHTS_DIR", tmp_path / "absent")
    assert loader._local_weights_path() is None


def test_local_weights_unset_uses_default_dir_when_present(monkeypatch, tmp_path):
    import khmer_pipeline.engines.kiri_vendor.loader as loader
    monkeypatch.delenv("KHMER_KIRI_WEIGHTS", raising=False)
    (tmp_path / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(loader, "_DEFAULT_WEIGHTS_DIR", tmp_path)
    assert loader._local_weights_path() == tmp_path / "model.safetensors"


def test_stock_sentinel_forces_hf_snapshot(monkeypatch, tmp_path):
    import khmer_pipeline.engines.kiri_vendor.loader as loader
    (tmp_path / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(loader, "_DEFAULT_WEIGHTS_DIR", tmp_path)
    monkeypatch.setenv("KHMER_KIRI_WEIGHTS", "stock")
    assert loader._local_weights_path() is None


def test_local_weights_dir_resolves_to_safetensors(monkeypatch, tmp_path):
    import khmer_pipeline.engines.kiri_vendor.loader as loader
    (tmp_path / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("KHMER_KIRI_WEIGHTS", str(tmp_path))
    assert loader._local_weights_path() == tmp_path / "model.safetensors"


def test_local_weights_file_used_directly(monkeypatch, tmp_path):
    import khmer_pipeline.engines.kiri_vendor.loader as loader
    f = tmp_path / "finetuned.safetensors"
    f.write_bytes(b"x")
    monkeypatch.setenv("KHMER_KIRI_WEIGHTS", str(f))
    assert loader._local_weights_path() == f


def test_local_weights_missing_fails_loud(monkeypatch, tmp_path):
    import khmer_pipeline.engines.kiri_vendor.loader as loader
    import pytest
    monkeypatch.setenv("KHMER_KIRI_WEIGHTS", str(tmp_path / "ghost"))
    with pytest.raises(FileNotFoundError, match="KHMER_KIRI_WEIGHTS"):
        loader._local_weights_path()


# ---------------------------------------------------------------------------
# Wide-cell chunking — pure NumPy, no model needed
# ---------------------------------------------------------------------------
# Kiri's input is a fixed IMG_H x IMG_W (48x640) and ResizeKeepRatioPadNoCrop
# scales by HEIGHT then crops the overflow, so anything past ~13.3:1 was silently
# discarded before the model ever ran. Measured on the MoC gas notification: 21%
# of cells exceeded the cap, and a 73-char cell came back as 42 chars (58%).

def _bars(h: int, w: int, bars: list[tuple[int, int]]) -> np.ndarray:
    """White (255) binary canvas with black (0) ink bars at the given x-ranges."""
    img = np.full((h, w), 255, dtype=np.uint8)
    for x0, x1 in bars:
        img[:, x0:x1] = 0
    return img


class TestSplitWideBinary:
    def test_narrow_cell_returned_unchanged(self):
        from khmer_pipeline.engines.kiri_recognizer import _split_wide_binary
        img = _bars(48, 200, [(10, 60), (80, 150)])  # 4.2:1, well under the cap
        parts = _split_wide_binary(img, max_aspect=13.33)
        assert len(parts) == 1
        assert parts[0] is img  # identity: the sub-cap path must not copy or alter

    def test_exactly_at_cap_not_split(self):
        from khmer_pipeline.engines.kiri_recognizer import _split_wide_binary
        img = _bars(48, int(48 * 13.33), [(10, 60)])
        assert len(_split_wide_binary(img, max_aspect=13.33)) == 1

    def test_over_cap_is_split_into_fitting_chunks(self):
        from khmer_pipeline.engines.kiri_recognizer import _split_wide_binary
        h, w = 48, 48 * 40  # 40:1, needs >= 3 chunks to fit under 13.33:1
        img = _bars(h, w, [(100, 300), (700, 900), (1400, 1600)])
        parts = _split_wide_binary(img, max_aspect=13.33)
        assert len(parts) >= 3
        for p in parts:
            assert p.shape[1] / p.shape[0] <= 13.33 + 1e-6

    def test_split_is_lossless_in_width(self):
        from khmer_pipeline.engines.kiri_recognizer import _split_wide_binary
        img = _bars(48, 48 * 30, [(100, 300), (900, 1100)])
        parts = _split_wide_binary(img, max_aspect=13.33)
        assert sum(p.shape[1] for p in parts) == img.shape[1]

    def test_cuts_land_in_whitespace_not_through_ink(self):
        from khmer_pipeline.engines.kiri_recognizer import _split_wide_binary
        # Ink bars separated by wide blank gutters; every cut must fall in a gutter,
        # because slicing through a Khmer glyph corrupts it in BOTH chunks.
        h = 48
        bars = [(i * 400, i * 400 + 300) for i in range(6)]  # gutters at 300-400 mod 400
        img = _bars(h, 2400, bars)
        parts = _split_wide_binary(img, max_aspect=13.33)
        offset = 0
        for p in parts[:-1]:
            offset += p.shape[1]
            column = img[:, offset]          # first column of the NEXT chunk
            assert column.min() == 255, f"cut at x={offset} sliced through ink"

    def test_all_blank_cell_still_splits_without_error(self):
        from khmer_pipeline.engines.kiri_recognizer import _split_wide_binary
        img = np.full((48, 48 * 30), 255, dtype=np.uint8)
        parts = _split_wide_binary(img, max_aspect=13.33)
        assert sum(p.shape[1] for p in parts) == img.shape[1]

    def test_solid_ink_cell_falls_back_to_hard_cuts(self):
        from khmer_pipeline.engines.kiri_recognizer import _split_wide_binary
        # No gutter anywhere: there is no good cut, but it must still fit the cap
        # rather than raise or return an over-wide chunk.
        img = np.full((48, 48 * 30), 0, dtype=np.uint8)
        parts = _split_wide_binary(img, max_aspect=13.33)
        for p in parts:
            assert p.shape[1] / p.shape[0] <= 13.33 + 1e-6

    def test_zero_height_is_safe(self):
        from khmer_pipeline.engines.kiri_recognizer import _split_wide_binary
        img = np.zeros((0, 10), dtype=np.uint8)
        assert len(_split_wide_binary(img, max_aspect=13.33)) == 1


class TestChunkedReassembly:
    """Wide cells are split, decoded piecewise, then rejoined — with a fake model
    so the wiring is tested without downloading weights."""

    @staticmethod
    def _fake_kiri(monkeypatch, piece_texts, piece_confs):
        """Patch _get_kiri + the decode path to return scripted per-piece results."""
        import khmer_pipeline.engines.kiri_recognizer as kr
        import torch

        state = {"n": 0}

        class _Tok:
            blank_id = 0

            def decode_ctc(self, row):
                i = state["n"]
                state["n"] += 1
                return piece_texts[i]

        class _Model:
            def encode(self, batch):
                return batch

            def ctc_head(self, mem):
                # (N, T=1, V=2); softmax max later reads the scripted confidence.
                n = mem.shape[0]
                out = torch.zeros(n, 1, 2)
                for i in range(n):
                    c = piece_confs[state["n"] + i]
                    # logits chosen so softmax max ~= c is not needed; we only
                    # assert ordering/joining, so make argmax != blank.
                    out[i, 0, 1] = 10.0 * c
                return out

        from khmer_pipeline.engines.kiri_vendor.model import CFG
        monkeypatch.setattr(kr, "_get_kiri", lambda: (_Model(), CFG(), _Tok()))
        return state

    def test_wide_cell_pieces_are_joined_in_order(self, monkeypatch):
        import khmer_pipeline.engines.kiri_recognizer as kr
        wide = np.full((48, 48 * 30, 3), 255, dtype=np.uint8)
        wide[10:20, 100:200] = 0
        self._fake_kiri(monkeypatch, ["AAA", "BBB", "CCC"], [0.9, 0.9, 0.9])
        out = kr.recognize_cells_conf([wide])
        assert len(out) == 1
        assert out[0][0] == "AAA BBB CCC"

    def test_confidence_is_driven_by_the_worst_piece_not_the_mean(self, monkeypatch):
        import khmer_pipeline.engines.kiri_recognizer as kr
        wide = np.full((48, 48 * 30, 3), 255, dtype=np.uint8)
        wide[10:20, 100:200] = 0

        # Identical except ONE chunk is poor. Under min-semantics that single bad
        # chunk drives the cell's confidence down; under mean-semantics the two
        # good chunks would mask it. Comparing the two runs asserts the semantics
        # without hardcoding the fake model's softmax arithmetic.
        self._fake_kiri(monkeypatch, ["A", "B", "C"], [0.9, 0.9, 0.9])
        all_good = kr.recognize_cells_conf([wide])[0][1]

        self._fake_kiri(monkeypatch, ["A", "B", "C"], [0.9, 0.02, 0.9])
        one_bad = kr.recognize_cells_conf([wide])[0][1]

        assert one_bad < all_good

    def test_chunking_emits_a_warning_to_the_sink(self, monkeypatch):
        import khmer_pipeline.engines.kiri_recognizer as kr
        wide = np.full((48, 48 * 30, 3), 255, dtype=np.uint8)
        self._fake_kiri(monkeypatch, ["A"] * 10, [0.9] * 10)
        sink = []
        kr.recognize_cells_conf([wide], warning_sink=sink)
        assert any("split into chunks" in w for w in sink)

    def test_narrow_cells_emit_no_chunking_warning(self, monkeypatch):
        import khmer_pipeline.engines.kiri_recognizer as kr
        narrow = np.full((48, 200, 3), 255, dtype=np.uint8)
        self._fake_kiri(monkeypatch, ["A"], [0.9])
        sink = []
        kr.recognize_cells_conf([narrow], warning_sink=sink)
        assert not any("split into chunks" in w for w in sink)

    def test_mixed_batch_keeps_cell_order(self, monkeypatch):
        import khmer_pipeline.engines.kiri_recognizer as kr
        narrow = np.full((48, 200, 3), 255, dtype=np.uint8)
        wide = np.full((48, 48 * 30, 3), 255, dtype=np.uint8)
        # narrow -> 1 piece, wide -> 3 pieces, narrow -> 1 piece
        self._fake_kiri(monkeypatch, ["N1", "W1", "W2", "W3", "N2"], [0.9] * 5)
        out = kr.recognize_cells_conf([narrow, wide, narrow])
        assert [t for t, _ in out] == ["N1", "W1 W2 W3", "N2"]
