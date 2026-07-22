from __future__ import annotations
import cv2
import numpy as np
import pytest
from khmer_pipeline.models import IngestResult, PreprocessResult
from khmer_pipeline.preprocess import PreprocessConfig, preprocess, suggest_preprocess_settings, _deskew, _skew_angle, _normalise_table_backgrounds, _crop_margins, _cap_resolution, _geometric_preprocess


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


# --- Shape-gated stamp removal -------------------------------------------------
# The colour mask alone cannot tell a stamp from coloured body text, so removal is
# gated on component SHAPE. These fixtures are BGR (what _remove_stamps takes).
# Measured on a real Ministry notification, the ungated version erased 8.61% of the
# page from a 1.66% colour mask — mostly blue paragraphs and blue table figures.

_BLUE_BGR = (255, 0, 0)


def _blank_page(h: int = 400, w: int = 400) -> np.ndarray:
    return np.full((h, w, 3), 240, dtype=np.uint8)


def _page_with_text_block(connector_px: int = 10) -> np.ndarray:
    """A roughly SQUARE block of blue text lines, fused into one component.

    This is the review edge case: with tight line spacing, a whole paragraph can
    merge into a single large component that clears both the size and the aspect
    gate. `connector_px` bridges the line gaps and is wide enough to survive the
    opening, so this really does reach the multi-line-text check."""
    img = _blank_page()
    for i in range(6):
        top = 40 + i * 50
        cv2.rectangle(img, (60, top), (340, top + 20), _BLUE_BGR, -1)
    # Vertical bridges through the gaps → one connected component.
    cv2.rectangle(img, (200, 40), (200 + connector_px, 330), _BLUE_BGR, -1)
    return img


def _page_with_thin_text_lines() -> np.ndarray:
    """Separate thin blue lines — ordinary coloured body text, no stamp anywhere."""
    img = _blank_page()
    for i in range(6):
        top = 40 + i * 50
        cv2.rectangle(img, (60, top), (340, top + 3), _BLUE_BGR, -1)
    return img


def _page_with_ring_stamp() -> np.ndarray:
    """A hollow ring seal: large bbox, near-square, but very LOW fill density —
    which is exactly why fill density alone cannot be the gate."""
    img = _blank_page()
    cv2.circle(img, (200, 200), 120, _BLUE_BGR, thickness=10)
    return img


def test_stamp_gate_keeps_merged_multiline_text_block():
    from khmer_pipeline.preprocess import _remove_stamps
    img = _page_with_text_block()
    out = _remove_stamps(img.copy())
    assert np.array_equal(out, img), (
        "A square block of coloured multi-line text must survive untouched — it is "
        "body text, not a stamp."
    )


def test_stamp_gate_keeps_plain_coloured_text():
    from khmer_pipeline.preprocess import _remove_stamps
    img = _page_with_thin_text_lines()
    out = _remove_stamps(img.copy())
    assert np.array_equal(out, img), "Thin coloured text lines must never be erased."


def test_stamp_gate_still_removes_hollow_ring_seal():
    from khmer_pipeline.preprocess import _remove_stamps
    img = _page_with_ring_stamp()
    out = _remove_stamps(img.copy())
    assert not np.array_equal(out, img), (
        "A hollow ring seal must still be removed — proving the multi-line-text "
        "guard did not cost us real stamps."
    )


def test_stamp_gate_leaves_page_untouched_when_nothing_qualifies():
    """The honesty property: identify nothing → destroy nothing."""
    from khmer_pipeline.preprocess import _remove_stamps
    img = _page_with_thin_text_lines()
    assert np.array_equal(_remove_stamps(img.copy()), img)


def test_suggest_reports_no_stamps_for_coloured_text_only_page():
    """Coloured text must not trigger a 'stamps found' recommendation."""
    rgb = cv2.cvtColor(_page_with_thin_text_lines(), cv2.COLOR_BGR2RGB)
    ing = IngestResult(source_name="t.pdf", page_images=[rgb], dpi=200, page_count=1)
    s = suggest_preprocess_settings(ing.page_images)
    stamp = next(c for c in s["checks"] if c["field"] == "remove_stamps")
    assert stamp["active"] is False
    assert stamp["reason"] == "no_stamps"


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


# --- suggest_preprocess_settings ---

def _checkerboard(size: int = 100) -> np.ndarray:
    """1px black/white checkerboard: extreme Laplacian variance AND contrast."""
    tile = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    ch = np.tile(tile, (size // 2, size // 2))
    return np.stack([ch, ch, ch], axis=2)


def _flat_gray(size: int = 100) -> np.ndarray:
    return np.full((size, size, 3), 128, dtype=np.uint8)


def _full_gradient(size: int = 256) -> np.ndarray:
    """Smooth 0→255 horizontal gradient: high contrast_std, ~zero Laplacian."""
    row = np.arange(size, dtype=np.uint8).reshape(1, size)
    ch = np.tile(row, (size, 1))
    return np.stack([ch, ch, ch], axis=2)


def test_suggest_empty_page_list():
    out = suggest_preprocess_settings([])
    # Shape extended in §2.47 with skew/stamp signals; zeros for unreadable uploads.
    assert out["scores"] == {"laplacian_var": 0.0, "contrast_std": 0.0,
                             "skew_deg": 0.0, "stamp_ink_ratio": 0.0}
    assert out["suggested"] == {}
    assert out["rationale"] == {}


def test_suggest_flat_gray_suggests_nothing():
    # Flat gray: LOW contrast and LOW sharpness — both defaults stay on.
    out = suggest_preprocess_settings([_flat_gray()])
    assert out["suggested"] == {}
    assert out["rationale"] == {}


def test_suggest_sharp_image_disables_sharpen():
    out = suggest_preprocess_settings([_checkerboard()])
    assert out["suggested"].get("sharpen") is False
    assert out["scores"]["laplacian_var"] > 500


def test_suggest_high_contrast_disables_normalise():
    out = suggest_preprocess_settings([_full_gradient()])
    assert out["suggested"] == {"normalise": False}
    assert out["scores"]["contrast_std"] > 60
    # Linear gradient has near-zero Laplacian: sharpen must NOT be suggested.
    assert "sharpen" not in out["suggested"]


def test_suggest_rationale_keys_mirror_suggested():
    out = suggest_preprocess_settings([_checkerboard()])
    assert set(out["rationale"]) == set(out["suggested"])
    assert all(isinstance(v, str) and v for v in out["rationale"].values())


def test_suggest_only_touches_v1_fields():
    out = suggest_preprocess_settings([_checkerboard(), _full_gradient()])
    assert set(out["suggested"]) <= {"sharpen", "normalise"}


def test_suggest_aggregates_with_median():
    # Two flat pages + one checkerboard: the median is the flat score, so the
    # single extreme page must not trigger a suggestion.
    out = suggest_preprocess_settings([_flat_gray(), _flat_gray(), _checkerboard()])
    assert out["suggested"] == {}


def test_suggest_scores_are_plain_floats():
    out = suggest_preprocess_settings([_checkerboard()])
    assert all(type(v) is float for v in out["scores"].values())


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


def _tilted_stripes(size: int = 200) -> np.ndarray:
    """White page with black bars rotated ~3°: clear skew signal."""
    img = np.full((size, size), 255, dtype=np.uint8)
    for y in range(30, size - 30, 24):
        img[y:y + 6, 20:size - 20] = 0
    m = cv2.getRotationMatrix2D((size / 2, size / 2), 3.0, 1.0)
    img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_NEAREST,
                         borderValue=255)
    return np.stack([img, img, img], axis=2)


def _red_stamp_page(size: int = 200) -> np.ndarray:
    """White page with a saturated red disc: stamp-ink signal."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), 30, (220, 20, 20), -1)  # RGB red
    return img


def test_suggest_checks_cover_all_user_toggles():
    out = suggest_preprocess_settings([_flat_gray()])
    fields = [c["field"] for c in out["checks"]]
    assert fields == ["deskew", "remove_stamps", "sharpen", "normalise",
                      "normalise_table_backgrounds"]
    for c in out["checks"]:
        assert isinstance(c["active"], bool)
        assert isinstance(c["reason"], str) and c["reason"]
        assert isinstance(c["detail"], str) and c["detail"]


def test_suggest_tilted_page_reports_tilt():
    out = suggest_preprocess_settings([_tilted_stripes()])
    deskew = next(c for c in out["checks"] if c["field"] == "deskew")
    assert deskew["active"] is True
    assert deskew["reason"] == "tilted"
    assert out["scores"]["skew_deg"] > 0.5


def test_suggest_straight_page_reports_straight():
    out = suggest_preprocess_settings([_full_gradient()])
    deskew = next(c for c in out["checks"] if c["field"] == "deskew")
    assert deskew["reason"] in ("straight",)


def test_suggest_stamp_ink_detected():
    out = suggest_preprocess_settings([_red_stamp_page()])
    stamps = next(c for c in out["checks"] if c["field"] == "remove_stamps")
    assert stamps["active"] is True
    assert stamps["reason"] == "stamps_found"
    assert out["scores"]["stamp_ink_ratio"] > 0.002


def test_suggest_clean_page_no_stamps():
    out = suggest_preprocess_settings([_flat_gray()])
    stamps = next(c for c in out["checks"] if c["field"] == "remove_stamps")
    assert stamps["active"] is False
    assert stamps["reason"] == "no_stamps"


def test_suggest_empty_list_has_empty_checks():
    assert suggest_preprocess_settings([])["checks"] == []
