from __future__ import annotations
import warnings
from typing import Any, Callable, Optional
from PIL import Image
from .models import PreprocessResult, SuryaResult, SuryaPageResult

_layout_pred = None
_rec_pred = None
_table_pred = None


def models_loaded() -> bool:
    return _layout_pred is not None


def preload_models() -> None:
    _get_predictors()


def _get_predictors():
    global _layout_pred, _rec_pred, _table_pred
    if _layout_pred is None:
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.table_rec import TableRecPredictor
        from surya.settings import settings
        layout = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
        rec = RecognitionPredictor(FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT))
        table = TableRecPredictor()
        _layout_pred, _rec_pred, _table_pred = layout, rec, table
    return _layout_pred, _rec_pred, _table_pred


def run_surya(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> SuryaResult:
    layout_pred, rec_pred, table_pred = _get_predictors()
    pil_images = [Image.fromarray(img) for img in result.page_images]
    total = len(pil_images)
    pages = []
    for idx, pil_img in enumerate(pil_images):
        if on_page is not None:
            on_page(idx, total)
        pages.append(_process_page(idx, pil_img, layout_pred, rec_pred, table_pred))
    return SuryaResult(source_name=result.source_name, pages=pages)


def _process_page(
    page_index: int,
    pil_img: Image.Image,
    layout_pred,
    rec_pred,
    table_pred,
) -> SuryaPageResult:
    layout_result = layout_pred([pil_img])[0]

    # Layout boxes are the text_blocks (used for overlay drawing with labels)
    text_blocks = [_serialize_layout_box(b) for b in layout_result.bboxes]

    # OCR: pass non-Table layout bboxes to recognition predictor
    # Filter degenerate bboxes (zero/negative width or height) before rec_pred
    non_table_bboxes = [
        b for b in layout_result.bboxes
        if b.label != "Table" and b.bbox[2] > b.bbox[0] and b.bbox[3] > b.bbox[1]
    ]
    if non_table_bboxes:
        page_bboxes = [[list(map(int, b.bbox)) for b in non_table_bboxes]]
        try:
            ocr_result = rec_pred([pil_img], bboxes=page_bboxes)[0]
            ocr_text = _build_ocr_text(ocr_result.text_lines)
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index}: {e}")
            ocr_text = ""
    else:
        ocr_text = ""

    table_bboxes = [b for b in layout_result.bboxes if b.label == "Table"]
    if table_bboxes:
        crops = [pil_img.crop(tuple(map(int, b.bbox))) for b in table_bboxes]
        table_results = table_pred(crops)
        for t, crop in zip(table_results, crops):
            if t.cells:
                try:
                    cell_bboxes = [list(map(int, c.bbox)) for c in t.cells]
                    cell_ocr = rec_pred([crop], bboxes=[cell_bboxes])[0]
                    for cell, line in zip(t.cells, cell_ocr.text_lines):
                        cell.text_lines = [{"text": line.text, "bbox": line.bbox}]
                except Exception as e:
                    warnings.warn(f"Cell OCR failed: {e}")
        tables = []
        for t in table_results:
            tbl = _serialize_table(t)
            tbl["cells"] = _filter_phantom_cells(tbl["cells"], tbl["image_bbox"])
            tables.append(tbl)
    else:
        tables = []

    return SuryaPageResult(
        page_index=page_index,
        text_blocks=text_blocks,
        tables=tables,
        ocr_text=ocr_text,
    )


def _serialize_layout_box(b) -> dict[str, Any]:
    return {
        "label": b.label,
        "bbox": b.bbox,
        "polygon": b.polygon,
        "reading_order": b.position,
    }


def _serialize_table(t) -> dict[str, Any]:
    return {
        "rows": [r.model_dump() for r in t.rows],
        "cols": [c.model_dump() for c in t.cols],
        "cells": [cell.model_dump() for cell in t.cells],
        "image_bbox": t.image_bbox,
    }


def _filter_phantom_cells(cells: list[dict], parent_bbox: list[float]) -> list[dict]:
    px0, py0, px1, py1 = parent_bbox
    return [
        c for c in cells
        if not (c["bbox"][2] <= px0 or c["bbox"][0] >= px1 or
                c["bbox"][3] <= py0 or c["bbox"][1] >= py1)
    ]


def _build_ocr_text(text_lines) -> str:
    return "\n\n".join(line.text for line in text_lines if line.text)
