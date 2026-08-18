from __future__ import annotations
import os
from .protocols import OCREngine, CorrectionEngine
from .surya import run_surya
from .tesseract_engine import run_tesseract
from .hybrid_engine import run_hybrid
from .surya_kiri_engine import run_surya_kiri
from .surya_kiri_vlm_engine import run_surya_kiri_vlm
from .auto_engine import run_auto
from .gemini_engine import run_gemini
from .gemma_ardb_engine import run_gemma_ardb
from .qwen_ardb_engine import run_qwen_ardb
from ..postprocess import postprocess

_OCR_ENGINES: dict[str, OCREngine] = {
    "surya": run_surya,
    "tesseract": run_tesseract,
    "hybrid": run_hybrid,
    "surya_kiri": run_surya_kiri,
    "surya_kiri_vlm": run_surya_kiri_vlm,
    "auto": run_auto,
    "gemini": run_gemini,  # cloud (opt-in); never selected by `auto`
    # Both fine-tune engines are always registered (reachable via get_ocr_engine
    # for CLI/direct/test use) regardless of whether the UI can select them —
    # see apps/api/api.py's _ENGINES for the actual UI-visibility gate. Neither is
    # ever selected by `auto`.
    "gemma_ardb": run_gemma_ardb,
    "qwen_ardb": run_qwen_ardb,  # not in apps.api _ENGINES yet — no config has passed real-doc eval
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

