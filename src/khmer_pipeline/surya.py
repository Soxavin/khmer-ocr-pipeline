from __future__ import annotations
from typing import Any
import numpy as np
from PIL import Image
from .models import PreprocessResult, SuryaResult, SuryaPageResult

_layout_pred = None
_rec_pred = None
_table_pred = None


def _get_predictors():
    global _layout_pred, _rec_pred, _table_pred
    if _layout_pred is None:
        from surya.inference import SuryaInferenceManager
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.table_rec import TableRecPredictor
        manager = SuryaInferenceManager()
        _layout_pred = LayoutPredictor(manager)
        _rec_pred = RecognitionPredictor(manager)
        _table_pred = TableRecPredictor(manager)
    return _layout_pred, _rec_pred, _table_pred


def run_surya(result: PreprocessResult) -> SuryaResult:
    layout_pred, rec_pred, table_pred = _get_predictors()
    pil_images = [Image.fromarray(img) for img in result.page_images]
    pages = [
        _process_page(idx, pil_img, layout_pred, rec_pred, table_pred)
        for idx, pil_img in enumerate(pil_images)
    ]
    return SuryaResult(source_name=result.source_name, pages=pages)


def _process_page(
    page_index: int,
    pil_img: Image.Image,
    layout_pred,
    rec_pred,
    table_pred,
) -> SuryaPageResult:
    layout_result = layout_pred([pil_img])[0]
    ocr_result = rec_pred([pil_img], [layout_result])[0]

    text_blocks = [_serialize_block(b) for b in ocr_result.blocks]
    ocr_text = _build_ocr_text(ocr_result.blocks)

    table_bboxes = [b for b in layout_result.bboxes if b.label in ("Table", "TableOfContents")]
    if table_bboxes:
        crops = [pil_img.crop(tuple(map(int, b.bbox))) for b in table_bboxes]
        table_results = table_pred(crops, mode="full")
        tables = [_serialize_table(t) for t in table_results]
    else:
        tables = []

    return SuryaPageResult(
        page_index=page_index,
        text_blocks=text_blocks,
        tables=tables,
        ocr_text=ocr_text,
    )


def _serialize_block(b) -> dict[str, Any]:
    return {
        "label": b.label,
        "html": b.html,
        "bbox": b.bbox,
        "polygon": b.polygon,
        "reading_order": b.reading_order,
        "confidence": b.confidence,
        "skipped": b.skipped,
        "error": b.error,
    }


def _serialize_table(t) -> dict[str, Any]:
    return {
        "rows": [r.model_dump() for r in t.rows],
        "cols": [c.model_dump() for c in t.cols],
        "cells": [cell.model_dump() for cell in t.cells],
        "html": t.html,
        "error": t.error,
        "mode": t.mode,
        "image_bbox": t.image_bbox,
    }


def _build_ocr_text(blocks) -> str:
    active = [b for b in blocks if not b.skipped and not b.error]
    ordered = sorted(active, key=lambda b: b.reading_order)
    return "\n\n".join(b.html for b in ordered)
