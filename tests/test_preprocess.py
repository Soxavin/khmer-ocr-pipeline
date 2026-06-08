from __future__ import annotations
import numpy as np
import pytest
from khmer_pipeline.models import IngestResult, PreprocessResult
from khmer_pipeline.preprocess import PreprocessConfig, preprocess


def _make_ingest_result(n_pages: int = 1, h: int = 100, w: int = 100) -> IngestResult:
    """Creates an IngestResult with gradient pages (non-flat, so processing has effect)."""
    row = np.arange(w, dtype=np.uint8).reshape(1, w)
    channel = np.tile(row, (h, 1))
    img = np.stack([channel, channel, channel], axis=2)
    return IngestResult(
        source_name="test.pdf",
        page_images=[img.copy() for _ in range(n_pages)],
        dpi=200,
        page_count=n_pages,
    )


def test_preprocess_returns_preprocess_result():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert isinstance(r, PreprocessResult)


def test_preprocess_preserves_source_name():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert r.source_name == "test.pdf"


def test_preprocess_preserves_dpi():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert r.dpi == 200


def test_preprocess_preserves_page_count():
    r = preprocess(_make_ingest_result(3), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert r.page_count == 3
    assert len(r.page_images) == 3


def test_preprocess_image_shape_unchanged():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert r.page_images[0].shape == (100, 100, 3)


def test_preprocess_images_are_rgb_uint8():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    arr = r.page_images[0]
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3


def test_preprocess_all_flags_false_is_passthrough():
    ingest_r = _make_ingest_result()
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert np.array_equal(r.page_images[0], original)


def test_preprocess_default_config_does_not_raise():
    # Smoke test: default config (all True) must not raise
    preprocess(_make_ingest_result())


def _make_red_blob_image() -> IngestResult:
    """100x100 off-white image with a 20x20 red square in the centre (RGB)."""
    img = np.full((100, 100, 3), 240, dtype=np.uint8)
    img[40:60, 40:60] = [255, 0, 0]  # red in RGB
    return IngestResult(
        source_name="stamp_test.pdf",
        page_images=[img],
        dpi=200,
        page_count=1,
    )


def test_stamp_removal_changes_red_region():
    ingest_r = _make_red_blob_image()
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=True, sharpen=False, normalise=False))
    output = r.page_images[0]
    # Image overall changed
    assert not np.array_equal(output, original)
    # The red blob region (rows 40:60, cols 40:60) is no longer red.
    # Inpainting replaces it with the surrounding background (~240,240,240).
    # Check mean red channel in the blob area is below 242 (was 255 before).
    blob_region = output[40:60, 40:60]
    mean_red_channel = blob_region[:, :, 0].mean()
    assert mean_red_channel < 242, f"Red channel mean in blob region is {mean_red_channel:.1f}, expected < 242 after inpainting"


def test_sharpen_changes_pixels():
    ingest_r = _make_ingest_result()  # gradient image — not flat, so sharpening has effect
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=True, normalise=False))
    assert not np.array_equal(r.page_images[0], original)
