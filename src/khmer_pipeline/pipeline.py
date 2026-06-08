from __future__ import annotations
from .ingest import ingest
from .preprocess import preprocess
from .surya import run_surya
from .postprocess import postprocess
from .export import export
from .models import ExportResult


def run(source: bytes, source_name: str, dpi: int = 200) -> ExportResult:
    ingest_result = ingest(source, source_name, dpi=dpi)
    preprocess_result = preprocess(ingest_result)
    surya_result = run_surya(preprocess_result)
    postprocess_result = postprocess(surya_result)
    return export(postprocess_result)
