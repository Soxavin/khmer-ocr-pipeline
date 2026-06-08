from __future__ import annotations
from .models import IngestResult, PreprocessResult


def preprocess(result: IngestResult) -> PreprocessResult:
    raise NotImplementedError("Stage 2 (preprocessing) not yet implemented.")
