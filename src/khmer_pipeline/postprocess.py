from __future__ import annotations
from .models import SuryaResult, PostprocessResult


def postprocess(result: SuryaResult) -> PostprocessResult:
    raise NotImplementedError("Stage 4 (post-processing) not yet implemented.")
