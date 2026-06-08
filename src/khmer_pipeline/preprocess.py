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
    raise NotImplementedError


def _sharpen(bgr: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def _normalise(bgr: np.ndarray) -> np.ndarray:
    raise NotImplementedError
