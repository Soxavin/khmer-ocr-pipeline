from __future__ import annotations
from .protocols import OCREngine, CorrectionEngine

# Import current implementations
from .surya import run_surya
from .postprocess import postprocess

# Register the active engines.
# To swap a model in the future, simply change the assigned function here.
ACTIVE_OCR_ENGINE: OCREngine = run_surya
ACTIVE_CORRECTION_ENGINE: CorrectionEngine = postprocess
