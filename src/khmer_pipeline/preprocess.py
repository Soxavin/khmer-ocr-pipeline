from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from .models import IngestResult, PreprocessResult


@dataclass
class PreprocessConfig:
    remove_stamps: bool = True
    sharpen: bool = True
    normalise: bool = True
    deskew: bool = True
    normalise_table_backgrounds: bool = True
    # INTERNAL flag (NOT a user-facing knob): orchestrators set this False for
    # engines that never read recognition_page_images, so the second
    # (geometric-only) preprocessing pass — a full deskew Otsu+minAreaRect per
    # page — is skipped. Deliberately EXEMPT from the 4-point sidebar/CLI pattern:
    # do NOT add a checkbox or --flag for it. Default True preserves prior behavior.
    with_recognition_images: bool = True


def preprocess(result: IngestResult, config: PreprocessConfig | None = None) -> PreprocessResult:
    if config is None:
        config = PreprocessConfig()
    processed = [_preprocess_image(img, config) for img in result.page_images]

    # Geometric-only pages (crop + deskew, no photometric changes) so the
    # surya_kiri engine can recognise cells with deskew applied without the
    # photometric normalisation that degrades Kiri — see the engine. Gated so the
    # default `surya` engine (which never reads them) doesn't pay for a second pass.
    recognition_page_images = None
    if config.with_recognition_images:
        recognition_page_images = [_geometric_preprocess(img, config) for img in result.page_images]
        # The two frame sets are coordinate-compatible ONLY because the geometric
        # steps (_crop_margins, _cap_resolution, _deskew) run BEFORE all photometric
        # steps in _preprocess_image: both frames therefore share identical geometry
        # (same H×W, same deskew) and differ only photometrically. Reordering those
        # steps would silently desynchronise table bboxes from text bboxes, so pin
        # the invariant with a per-page shape check.
        for full, geo in zip(processed, recognition_page_images):
            assert full.shape[:2] == geo.shape[:2], (
                "recognition image geometry diverged from the page image — geometric "
                "preprocessing must precede all photometric steps in _preprocess_image"
            )

    return PreprocessResult(
        source_name=result.source_name,
        page_images=processed,
        dpi=result.dpi,
        page_count=result.page_count,
        recognition_page_images=recognition_page_images,
        low_res_scan=result.low_res_scan,
    )


_CROP_MARGINS_BORDER_THRESH = 240   # pixels above this value are treated as empty border
_CROP_MARGINS_PAD = 20              # px of content margin kept after crop
# Longest edge cap before downscaling. 2900 (not 2048) because Kiri scales each
# cell crop by HEIGHT to CFG.IMG_H=48: at 2048 a large scan (budget p3, 4400px
# native) is squeezed 0.465x, putting 97% of its cell crops BELOW 48px so Kiri
# upsamples blur. 2900 keeps those crops at ~55px. Measured §2.42: budget p3
# numeric_cell_accuracy 0.279→0.550, CER 0.256→0.164; ARDB (2000px native, under
# both caps) is bit-identical. Raising further mainly costs memory — the 24GB box
# runs PyTorch + MLX co-resident.
_CAP_RESOLUTION_MAX_DIM = 2900


def _crop_margins(bgr: np.ndarray, border_thresh: int = _CROP_MARGINS_BORDER_THRESH) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, border_thresh, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return bgr
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = gray.shape
    x1 = max(0, x - _CROP_MARGINS_PAD)
    y1 = max(0, y - _CROP_MARGINS_PAD)
    x2 = min(w_img, x + w + _CROP_MARGINS_PAD)
    y2 = min(h_img, y + h + _CROP_MARGINS_PAD)
    return bgr[y1:y2, x1:x2]


def _cap_resolution(bgr: np.ndarray, max_dim: int = _CAP_RESOLUTION_MAX_DIM) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _preprocess_image(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgr = _crop_margins(bgr)
    bgr = _cap_resolution(bgr)
    if cfg.deskew:
        bgr = _deskew(bgr)
    if cfg.remove_stamps:
        bgr = _remove_stamps(bgr)
    if cfg.normalise_table_backgrounds:
        bgr = _normalise_table_backgrounds(bgr)
    if cfg.sharpen:
        bgr = _sharpen(bgr)
    if cfg.normalise:
        bgr = _normalise(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def _geometric_preprocess(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Geometric-only preprocessing: crop + resolution cap + optional deskew, with
    NO photometric changes (CLAHE / desaturation / sharpen / stamp removal). Used
    by per-cell-binarizing recognizers (surya_kiri) that need geometric correction
    (deskew) but are degraded by photometric normalization."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgr = _crop_margins(bgr)
    bgr = _cap_resolution(bgr)
    if cfg.deskew:
        bgr = _deskew(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


# Above this variance-of-Laplacian the scan is already crisp, so software
# sharpening adds ringing for no gain. Heuristic starting point pending
# calibration on the GDDE corpus.
_SUGGEST_SHARP_LAPLACIAN = 500.0
# Above this grayscale std the page is already well contrasted, so CLAHE
# mostly amplifies scan noise. Same caveat: heuristic pending calibration.
_SUGGEST_CONTRAST_STD = 60.0
# Above this fraction of saturated red/blue pixels a stamp/signature is likely
# present (a typical circular stamp covers ~0.2–2% of an A4 page).
_SUGGEST_STAMP_INK_RATIO = 0.002


def suggest_preprocess_settings(page_images: list[np.ndarray]) -> dict:
    """Cheap image-quality scores + conservative suggested PreprocessConfig toggles.

    Scores blur (variance of Laplacian) and contrast (grayscale std) per page,
    aggregated with the median (robust to one odd page). Returns
    ``{"scores": {...}, "suggested": {field: bool}, "rationale": {field: str}}``
    where `suggested` holds ONLY fields deviating from the dataclass defaults —
    usually empty. Suggestions are advisory: the UI shows them, the user decides.
    Note: with the surya_kiri engine these photometric toggles influence
    layout/table detection only (recognition reads geometric-only frames).
    """
    if not page_images:
        return {"scores": {"laplacian_var": 0.0, "contrast_std": 0.0,
                           "skew_deg": 0.0, "stamp_ink_ratio": 0.0},
                "suggested": {}, "rationale": {}, "checks": []}
    blur_scores, contrast_scores, skews, ink_ratios = [], [], [], []
    for img in page_images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast_scores.append(float(gray.std()))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        skews.append(abs(_skew_angle(bgr)))
        # Score the SHAPE-GATED mask, not raw colour: blue body text trivially
        # exceeds a colour-ratio threshold, which is how the UI ended up
        # recommending stamp removal on documents that have no stamp at all.
        mask = _stamp_regions_mask(bgr)
        ink_ratios.append(float(cv2.countNonZero(mask)) / mask.size)
    laplacian_var = float(np.median(blur_scores))
    contrast_std = float(np.median(contrast_scores))
    # Skew/ink use the MAX: one tilted or stamped page is enough to matter.
    skew_deg = float(np.max(skews))
    stamp_ink_ratio = float(np.max(ink_ratios))

    suggested: dict[str, bool] = {}
    rationale: dict[str, str] = {}
    if laplacian_var > _SUGGEST_SHARP_LAPLACIAN:
        suggested["sharpen"] = False
        rationale["sharpen"] = "The scan is already sharp, so extra sharpening would only add noise."
    if contrast_std > _SUGGEST_CONTRAST_STD:
        suggested["normalise"] = False
        rationale["normalise"] = "The pages are already well contrasted, so contrast enhancement is unnecessary."

    # Per-toggle assessment for the UI's "scan check" story. `active` means
    # "this cleanup is useful for THIS document"; `reason` is a stable key the
    # frontend localizes; `detail` keeps the measured evidence in English.
    tilted = skew_deg >= _DESKEW_MIN_ANGLE_DEG
    stamped = stamp_ink_ratio >= _SUGGEST_STAMP_INK_RATIO
    soft = laplacian_var <= _SUGGEST_SHARP_LAPLACIAN
    faded = contrast_std <= _SUGGEST_CONTRAST_STD
    checks = [
        {"field": "deskew", "active": tilted,
         "reason": "tilted" if tilted else "straight",
         "detail": f"largest page tilt {skew_deg:.1f}\u00b0"},
        {"field": "remove_stamps", "active": stamped,
         "reason": "stamps_found" if stamped else "no_stamps",
         "detail": (f"stamp-shaped marks on {stamp_ink_ratio * 100:.2f}% of the worst page"
                    if stamped else
                    "no stamp-shaped marks found; any colored text is left intact")},
        {"field": "sharpen", "active": soft,
         "reason": "soft_scan" if soft else "already_sharp",
         "detail": f"sharpness score {laplacian_var:.0f} (threshold {_SUGGEST_SHARP_LAPLACIAN:.0f})"},
        {"field": "normalise", "active": faded,
         "reason": "faded" if faded else "good_contrast",
         "detail": f"contrast score {contrast_std:.0f} (threshold {_SUGGEST_CONTRAST_STD:.0f})"},
        {"field": "normalise_table_backgrounds", "active": True,
         "reason": "table_shading_default",
         "detail": "always applied; harmless when tables have no shading"},
    ]
    return {"scores": {"laplacian_var": laplacian_var, "contrast_std": contrast_std,
                       "skew_deg": skew_deg, "stamp_ink_ratio": stamp_ink_ratio},
            "suggested": suggested, "rationale": rationale, "checks": checks}


def _stamp_ink_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
    return cv2.bitwise_or(mask_red, mask_blue)


# --- Stamp SHAPE gate ---------------------------------------------------------
# Colour alone cannot separate a stamp from coloured body text: Cambodian ministry
# documents routinely print body text and headers in blue, and financial tables use
# red figures. Removing every red/blue pixel erased 8.61% of a real MoC notification
# (5.18x the colour mask, once dilated) including table values. So a colour-masked
# component is only removed when its SHAPE says "stamp".
#
# Opening kernel: clears specks and hairline bridges before component analysis.
# Deliberately SMALL — measured on a real MoC seal, a 5px kernel erased the stamp's
# own ring (stroke thinner than 5px) and the whole seal stopped being detected.
# 3px keeps real rings intact; the multi-line-text check below is the actual guard
# against coloured paragraphs, not this kernel.
_STAMP_OPEN_KERNEL_PX = 3
# A stamp's bbox spans at least this fraction of the page's shorter side (both
# dimensions). Fraction, not pixels, so it holds at any DPI.
_STAMP_MIN_BOX_FRAC = 0.05
# Stamps are round/oval; text LINES are strongly elongated.
_STAMP_MAX_ASPECT = 3.0
# Fill density above which a component is an unambiguous solid/heavy seal — far
# above any text block, so it is a safe fast-accept. Deliberately NOT used as the
# general gate: a hollow ring seal scores ~0.05, BELOW a merged text block (~0.1-0.2),
# so a density floor would reject real stamps and a ceiling would miss the text.
_STAMP_SOLID_EXTENT = 0.55
# A row counts as blank when its ink falls below this fraction of the component's
# mean row ink. Multi-line text alternates dense rows with blank gaps; a ring has
# ink on its left/right arcs in every row, so it shows no blank bands at all.
_STAMP_BLANK_ROW_FRAC = 0.25
# Two or more blank bands means multi-line text, not a stamp.
_STAMP_MAX_LINE_GAPS = 1


def _has_text_line_gaps(component: np.ndarray) -> bool:
    """True when a component's horizontal profile alternates like multi-line text.

    Guards the case where tight line spacing fuses a whole paragraph into one
    square-ish component that clears the size and aspect gates."""
    rows = component.sum(axis=1, dtype=np.float64) / 255.0
    if rows.size == 0:
        return False
    mean_ink = rows.mean()
    if mean_ink <= 0:
        return False
    blank = rows < (mean_ink * _STAMP_BLANK_ROW_FRAC)
    # Count contiguous runs of blank rows (bands), not individual rows.
    bands = int(np.count_nonzero(blank[1:] & ~blank[:-1])) + int(blank[0])
    return bands > _STAMP_MAX_LINE_GAPS


def _stamp_regions_mask(bgr: np.ndarray) -> np.ndarray:
    """Colour-masked components that actually look like stamps.

    Returns an all-zero mask when nothing qualifies — the honesty property: what we
    cannot positively identify as a stamp is left alone."""
    coloured = _stamp_ink_mask(bgr)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_STAMP_OPEN_KERNEL_PX,) * 2)
    opened = cv2.morphologyEx(coloured, cv2.MORPH_OPEN, k)

    h, w = opened.shape[:2]
    min_box = min(h, w) * _STAMP_MIN_BOX_FRAC
    keep = np.zeros_like(opened)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    for i in range(1, count):  # 0 is the background
        x, y, bw, bh, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                              stats[i, cv2.CC_STAT_AREA])
        if bw < min_box or bh < min_box:
            continue  # glyph-sized: never a stamp
        if max(bw / bh, bh / bw) > _STAMP_MAX_ASPECT:
            continue  # a text line, not a seal
        component = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255
        extent = area / float(bw * bh)
        if extent < _STAMP_SOLID_EXTENT and _has_text_line_gaps(component):
            continue  # multi-line text fused into one block
        # Confirmed stamp: take ALL coloured ink inside its box, not just this
        # component. A seal fragments into an outer ring, an inner text ring and an
        # emblem; individually those fail the gates, so removing only the winning
        # component leaves most of the stamp behind. Colour filtering stays strictly
        # LOCALIZED to a region already proven to be a stamp.
        keep[y:y + bh, x:x + bw] |= coloured[y:y + bh, x:x + bw]
    return keep


def _remove_stamps(bgr: np.ndarray) -> np.ndarray:
    stamps = _stamp_regions_mask(bgr)
    if cv2.countNonZero(stamps) == 0:
        return bgr  # nothing identifiable as a stamp: change nothing

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(stamps, kernel, iterations=2)
    return cv2.inpaint(bgr, dilated, 5, cv2.INPAINT_TELEA)


def _sharpen(bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(bgr, ddepth=-1, kernel=kernel)


def _normalise(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# Thresholds for detecting shaded table cell backgrounds (e.g. header rows or
# alternating-row fills) so they can be desaturated toward neutral gray/white,
# leaving dark text pixels untouched, for a cleaner grid for Surya's table detector.
_TABLE_BG_MIN_VALUE = 150
_TABLE_BG_MIN_SATURATION = 30


def _normalise_table_backgrounds(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = (v > _TABLE_BG_MIN_VALUE) & (s > _TABLE_BG_MIN_SATURATION)
    s[mask] = 0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# Below this angle (in degrees) a rotation correction is treated as a no-op,
# since the visual difference is negligible and avoids needless resampling.
_DESKEW_MIN_ANGLE_DEG = 0.5


def _skew_angle(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # np.where on a (rows, cols) array yields (row, col) = (y, x) pairs, but
    # minAreaRect expects (x, y) points. Swapping the axes like this mirrors
    # the point set through y=x, which negates the angle minAreaRect returns;
    # the final `return -angle` below cancels that out to give the correct
    # sign, so don't "fix" this ordering without also removing the negation.
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # Normalise to (-45, 45] so the result represents the smallest rotation
    # needed to align the dominant content with the axes.
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    # See the (row, col) vs (x, y) note above: this negation is required to
    # correct for the axis swap in `coords`.
    return -angle


def _deskew(bgr: np.ndarray) -> np.ndarray:
    angle = _skew_angle(bgr)
    if abs(angle) < _DESKEW_MIN_ANGLE_DEG:
        return bgr
    h, w = bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # INTER_CUBIC for smooth resampling; BORDER_REPLICATE avoids introducing
    # black borders at the rotated edges, which would otherwise hurt OCR.
    return cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
