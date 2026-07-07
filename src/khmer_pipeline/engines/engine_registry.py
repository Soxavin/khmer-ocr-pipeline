from __future__ import annotations
import os
from .protocols import OCREngine, CorrectionEngine
from .surya import run_surya
from .tesseract_engine import run_tesseract
from .hybrid_engine import run_hybrid
from .surya_kiri_engine import run_surya_kiri
from ..postprocess import postprocess

_OCR_ENGINES: dict[str, OCREngine] = {
    "surya": run_surya,
    "tesseract": run_tesseract,
    "hybrid": run_hybrid,
    "surya_kiri": run_surya_kiri,
}
ACTIVE_OCR_ENGINE: OCREngine = _OCR_ENGINES.get(
    os.environ.get("OCR_ENGINE", "surya"), run_surya
)
ACTIVE_CORRECTION_ENGINE: CorrectionEngine = postprocess


def get_ocr_engine(name: str) -> OCREngine:
    """Return the OCR engine registered under *name*, or run_surya if unknown."""
    return _OCR_ENGINES.get(name, run_surya)

