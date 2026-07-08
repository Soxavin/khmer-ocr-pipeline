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


def preprocess(result: IngestResult, config: PreprocessConfig | None = None) -> PreprocessResult:
    if config is None:
        config = PreprocessConfig()
    processed = [_preprocess_image(img, config) for img in result.page_images]
    return PreprocessResult(
        source_name=result.source_name,
        page_images=processed,
        dpi=result.dpi,
        page_count=result.page_count,
        # Geometric-only pages (crop + deskew, no photometric changes) so the
        # surya_kiri engine can recognise cells with deskew applied without the
        # photometric normalisation that degrades Kiri — see the engine.
        recognition_page_images=[_geometric_preprocess(img, config) for img in result.page_images],
    )


_CROP_MARGINS_BORDER_THRESH = 240   # pixels above this value are treated as empty border
_CROP_MARGINS_PAD = 20              # px of content margin kept after crop
_CAP_RESOLUTION_MAX_DIM = 2048      # longest edge cap before downscaling


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


def _remove_stamps(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))

    combined = cv2.bitwise_or(mask_red, mask_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(combined, kernel, iterations=2)

    if cv2.countNonZero(dilated) == 0:
        return bgr

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
