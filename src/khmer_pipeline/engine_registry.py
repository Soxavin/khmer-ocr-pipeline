from __future__ import annotations
import os
from .protocols import OCREngine, CorrectionEngine
from .surya import run_surya
from .tesseract_engine import run_tesseract
from .hybrid_engine import run_hybrid
from .postprocess import postprocess

_OCR_ENGINES: dict[str, OCREngine] = {
    "surya": run_surya,
    "tesseract": run_tesseract,
    "hybrid": run_hybrid,
}
ACTIVE_OCR_ENGINE: OCREngine = _OCR_ENGINES.get(
    os.environ.get("OCR_ENGINE", "surya"), run_surya
)
ACTIVE_CORRECTION_ENGINE: CorrectionEngine = postprocess

