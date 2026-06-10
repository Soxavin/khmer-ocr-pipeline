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

    # Per-region OCR: crop each non-Table layout region, run rec_pred on the crop
    text_blocks: list[dict] = []
    for layout_bbox in layout_result.bboxes:
        if layout_bbox.label == "Table":
            continue
        x0, y0, x1, y1 = layout_bbox.bbox
        if (x1 - x0) < 50 or (y1 - y0) < 20:
            continue
        crop = pil_img.crop((int(x0), int(y0), int(x1), int(y1)))
        crop_w, crop_h = crop.size
        try:
            region_ocr = rec_pred([crop], bboxes=[[[0, 0, crop_w, crop_h]]])[0]
            for line in region_ocr.text_lines:
                block = _serialize_text_line(line)
                block = _adjust_coordinates(block, x0, y0)
                block["label"] = layout_bbox.label
                block["region_label"] = layout_bbox.label
                block["reading_order"] = layout_bbox.position
                text_blocks.append(block)
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index}: {e}")

    # Sort blocks: primary by reading_order (if set), fallback top-to-bottom left-to-right
    def _sort_key(block: dict) -> tuple:
        ro = block.get("reading_order") or 0
        bbox = block.get("bbox") or [0, 0, 0, 0]
        if ro > 0:
            return (0, ro, 0.0, 0.0)
        return (1, 0, bbox[1], bbox[0])

    sorted_blocks = sorted(text_blocks, key=_sort_key)

    # Plain text — no region labels embedded
    ocr_text = "\n\n".join(b["text"] for b in sorted_blocks if b.get("text"))

    # Table recognition (unchanged)
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
        text_blocks=sorted_blocks,
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


def _serialize_text_line(line) -> dict[str, Any]:
    return {
        "text": line.text,
        "bbox": list(line.bbox),
        "polygon": [list(p) for p in line.polygon],
        "confidence": line.confidence,
    }


def _adjust_coordinates(block_dict: dict, offset_x: float, offset_y: float) -> dict:
    if block_dict.get("bbox"):
        b = block_dict["bbox"]
        block_dict["bbox"] = [b[0] + offset_x, b[1] + offset_y, b[2] + offset_x, b[3] + offset_y]
    if block_dict.get("polygon"):
        block_dict["polygon"] = [
            [p[0] + offset_x, p[1] + offset_y]
            for p in block_dict["polygon"]
        ]
    return block_dict


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


