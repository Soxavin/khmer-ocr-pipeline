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
    raise NotImplementedError


def _normalise(bgr: np.ndarray) -> np.ndarray:
    raise NotImplementedError
