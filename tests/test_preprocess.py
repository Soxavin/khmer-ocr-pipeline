from __future__ import annotations
import cv2
import numpy as np
import pytest
from khmer_pipeline.models import IngestResult, PreprocessResult
from khmer_pipeline.preprocess import PreprocessConfig, preprocess, _deskew, _skew_angle, _normalise_table_backgrounds, _crop_margins, _cap_resolution, _geometric_preprocess


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
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
    assert isinstance(r, PreprocessResult)


def test_preprocess_preserves_source_name():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
    assert r.source_name == "test.pdf"


def test_preprocess_preserves_dpi():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
    assert r.dpi == 200


def test_preprocess_preserves_page_count():
    r = preprocess(_make_ingest_result(3), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
    assert r.page_count == 3
    assert len(r.page_images) == 3


def test_preprocess_image_shape_unchanged():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
    assert r.page_images[0].shape == (100, 100, 3)


def test_preprocess_images_are_rgb_uint8():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
    arr = r.page_images[0]
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3


def test_preprocess_all_flags_false_is_passthrough():
    ingest_r = _make_ingest_result()
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
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
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=True, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
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
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=True, normalise=False, deskew=False, normalise_table_backgrounds=False))
    assert not np.array_equal(r.page_images[0], original)


def test_normalise_changes_pixels():
    ingest_r = _make_ingest_result()  # gradient image — CLAHE has something to work with
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=False, normalise=True, deskew=False, normalise_table_backgrounds=False))
    assert not np.array_equal(r.page_images[0], original)


def _make_rotated_rect_image(angle_degrees: float, size: int = 200) -> np.ndarray:
    """White background with a single black rectangle rotated by angle_degrees."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    rect = ((size / 2, size / 2), (size / 2, size / 6), angle_degrees)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [box], (0, 0, 0))
    return img


def test_deskew_preserves_shape_and_dtype():
    img = _make_rotated_rect_image(10)
    out = _deskew(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_deskew_is_noop_on_axis_aligned_image():
    img = _make_rotated_rect_image(0)
    out = _deskew(img)
    assert np.array_equal(out, img)


def test_deskew_rotates_skewed_image():
    img = _make_rotated_rect_image(10)
    out = _deskew(img)
    assert not np.array_equal(out, img)
    # Correcting the detected skew should leave a near-zero residual angle,
    # smaller than the original image's skew.
    assert abs(_skew_angle(out)) < abs(_skew_angle(img))


def test_preprocess_deskew_flag_controls_step():
    ingest_r = IngestResult(
        source_name="skew.pdf",
        page_images=[_make_rotated_rect_image(10)],
        dpi=200,
        page_count=1,
    )
    r_deskewed = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=True, normalise_table_backgrounds=False))
    r_raw = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
    # The two outputs share the same crop but differ in rotation correction
    assert not np.array_equal(r_deskewed.page_images[0], r_raw.page_images[0])


def _make_colored_bg_image() -> IngestResult:
    """100x100 white page with a 40x40 light-blue shaded region (simulating a
    table header fill) containing a 10x10 dark 'text' block."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[20:60, 20:60] = [180, 220, 255]  # light blue shading (RGB)
    img[35:45, 35:45] = [30, 30, 30]     # dark text pixels inside the shaded region
    return IngestResult(
        source_name="bg_test.pdf",
        page_images=[img],
        dpi=200,
        page_count=1,
    )


def test_background_normalise_desaturates_colored_fill():
    ingest_r = _make_colored_bg_image()
    r = preprocess(ingest_r, PreprocessConfig(
        remove_stamps=False, sharpen=False, normalise=False, deskew=False,
        normalise_table_backgrounds=True,
    ))
    output = r.page_images[0]
    # The light-blue shaded region (away from the dark text block) should now
    # be near-neutral — R, G, B channels much closer together than before.
    bg_pixel = output[25, 25].astype(int)
    assert max(bg_pixel) - min(bg_pixel) < 15, f"expected near-neutral color, got {bg_pixel}"


def test_background_normalise_preserves_dark_text():
    ingest_r = _make_colored_bg_image()
    r = preprocess(ingest_r, PreprocessConfig(
        remove_stamps=False, sharpen=False, normalise=False, deskew=False,
        normalise_table_backgrounds=True,
    ))
    output = r.page_images[0]
    text_pixel = output[40, 40]
    assert text_pixel.max() < 60


def test_background_normalise_disabled_is_passthrough():
    ingest_r = _make_colored_bg_image()
    r = preprocess(ingest_r, PreprocessConfig(
        remove_stamps=False, sharpen=False, normalise=False, deskew=False,
        normalise_table_backgrounds=False,
    ))
    # Light-blue region should NOT be desaturated when the flag is off
    bg_pixel = r.page_images[0][25, 25].astype(int)
    assert max(bg_pixel) - min(bg_pixel) >= 15, f"expected colored region preserved, got {bg_pixel}"


# --- _crop_margins ---

def test_crop_margins_trims_white_border():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[45:55, 45:55] = 0  # small black square at center
    result = _crop_margins(img)
    assert result.shape[0] < 100 and result.shape[1] < 100

def test_crop_margins_blank_image_returns_original():
    img = np.full((80, 80, 3), 255, dtype=np.uint8)
    result = _crop_margins(img)
    assert result.shape == img.shape

def test_crop_margins_preserves_dtype():
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[10:50, 10:50] = 200
    result = _crop_margins(img)
    assert result.dtype == np.uint8


# --- _cap_resolution ---

def test_cap_resolution_downscales_large_image():
    img = np.zeros((3000, 2000, 3), dtype=np.uint8)
    result = _cap_resolution(img, max_dim=2048)
    assert max(result.shape[:2]) == 2048

def test_cap_resolution_does_not_upscale_small_image():
    img = np.zeros((800, 600, 3), dtype=np.uint8)
    result = _cap_resolution(img, max_dim=2048)
    assert result.shape == img.shape

def test_cap_resolution_preserves_aspect_ratio():
    img = np.zeros((3000, 1500, 3), dtype=np.uint8)
    result = _cap_resolution(img, max_dim=2048)
    h, w = result.shape[:2]
    assert abs(w / h - 0.5) < 0.01

def test_cap_resolution_preserves_dtype():
    img = np.zeros((4000, 3000, 3), dtype=np.uint8)
    result = _cap_resolution(img, max_dim=2048)
    assert result.dtype == np.uint8


# --- recognition_page_images / _geometric_preprocess ---

def test_preprocess_populates_recognition_page_images():
    r = preprocess(_make_ingest_result(3), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False, deskew=False, normalise_table_backgrounds=False))
    assert r.recognition_page_images is not None
    assert len(r.recognition_page_images) == 3


def test_preprocess_recognition_images_gated_off():
    """B5: with_recognition_images=False skips the second pass entirely."""
    r = preprocess(_make_ingest_result(3), PreprocessConfig(with_recognition_images=False))
    assert r.recognition_page_images is None


def test_preprocess_recognition_images_shapes_match_page_images():
    """B5: each recognition image shares its page image's H×W (geometry-compatible)."""
    r = preprocess(_make_ingest_result(2), PreprocessConfig(deskew=True))
    assert r.recognition_page_images is not None
    for full, geo in zip(r.page_images, r.recognition_page_images):
        assert full.shape[:2] == geo.shape[:2]


def test_geometric_preprocess_skips_photometric_changes():
    """_geometric_preprocess must not apply photometric changes (normalise/sharpen/
    remove_stamps/normalise_table_backgrounds) even when those flags are on —
    only crop + resolution cap + optional deskew run."""
    ingest_r = _make_colored_bg_image()
    img = ingest_r.page_images[0]
    cfg = PreprocessConfig(remove_stamps=True, sharpen=True, normalise=True, deskew=False, normalise_table_backgrounds=True)
    geo = _geometric_preprocess(img, cfg)

    # Manually compute the crop+cap-only baseline (deskew off here, no photometric steps).
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgr = _crop_margins(bgr)
    bgr = _cap_resolution(bgr)
    expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    assert np.array_equal(geo, expected)
