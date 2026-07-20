from __future__ import annotations
import os
from .protocols import OCREngine, CorrectionEngine
from .surya import run_surya
from .tesseract_engine import run_tesseract
from .hybrid_engine import run_hybrid
from .surya_kiri_engine import run_surya_kiri
from .surya_kiri_vlm_engine import run_surya_kiri_vlm
from .auto_engine import run_auto
from ..postprocess import postprocess

_OCR_ENGINES: dict[str, OCREngine] = {
    "surya": run_surya,
    "tesseract": run_tesseract,
    "hybrid": run_hybrid,
    "surya_kiri": run_surya_kiri,
    "surya_kiri_vlm": run_surya_kiri_vlm,
    "auto": run_auto,
}


def get_ocr_engine(name: str) -> OCREngine:
    """Return the OCR engine registered under *name*.

    Raises ValueError (listing the valid names) on an unknown name — a typo must
    never silently fall back to Surya and benchmark the wrong engine."""
    try:
        return _OCR_ENGINES[name]
    except KeyError:
        raise ValueError(
            f"Unknown OCR engine {name!r}. Valid engines: {sorted(_OCR_ENGINES)}."
        ) from None


# Resolve the active engine from OCR_ENGINE at import — an unknown value raises
# here (fail loudly) rather than silently running Surya.
ACTIVE_OCR_ENGINE: OCREngine = get_ocr_engine(os.environ.get("OCR_ENGINE", "surya"))
ACTIVE_CORRECTION_ENGINE: CorrectionEngine = postprocess

