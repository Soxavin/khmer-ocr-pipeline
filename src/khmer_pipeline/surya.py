from __future__ import annotations
from .models import PreprocessResult, SuryaResult


def _get_predictors():
    """Returns (layout_pred, rec_pred, table_pred) tuple."""
    raise NotImplementedError("Stage 3 (Surya 2 OCR) not yet implemented.")


def run_surya(result: PreprocessResult) -> SuryaResult:
    raise NotImplementedError("Stage 3 (Surya 2 OCR) not yet implemented.")
