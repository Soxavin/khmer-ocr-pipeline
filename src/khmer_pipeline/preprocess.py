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


def preprocess(result: IngestResult, config: PreprocessConfig | None = None) -> PreprocessResult:
    if config is None:
        config = PreprocessConfig()
    processed = [_preprocess_image(img, config) for img in result.page_images]
    return PreprocessResult(
        source_name=result.source_name,
        page_images=processed,
        dpi=result.dpi,
        page_count=result.page_count,
    )


def _preprocess_image(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if cfg.deskew:
        bgr = _deskew(bgr)
    if cfg.remove_stamps:
        bgr = _remove_stamps(bgr)
    if cfg.sharpen:
        bgr = _sharpen(bgr)
    if cfg.normalise:
        bgr = _normalise(bgr)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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
