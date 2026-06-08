from __future__ import annotations
from .models import PostprocessResult, ExportResult


def export(result: PostprocessResult) -> ExportResult:
    raise NotImplementedError("Stage 5 (export) not yet implemented.")
