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


def _new_table_predictor():
    from surya.table_rec import TableRecPredictor
    return TableRecPredictor()


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

    # Collect page-space bboxes for all non-Table, non-degenerate layout regions
    text_regions = [
        b for b in layout_result.bboxes
        if b.label != "Table"
        and (b.bbox[2] - b.bbox[0]) >= 50
        and (b.bbox[3] - b.bbox[1]) >= 20
    ]

    # OCR all text regions in one call: one page image, one bbox per region (page-space)
    text_blocks: list[dict] = []
    if text_regions:
        region_bboxes = [list(map(int, b.bbox)) for b in text_regions]
        try:
            region_ocr = rec_pred([pil_img], bboxes=[region_bboxes])[0]
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index}: {e}")
            region_ocr = None
        if region_ocr is not None:
            for line, layout_bbox in zip(region_ocr.text_lines, text_regions):
                block = _serialize_text_line(line)
                block["label"] = layout_bbox.label
                block["region_label"] = layout_bbox.label
                block["reading_order"] = layout_bbox.position
                text_blocks.append(block)

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

    # Table recognition: one table_pred call per table (batch size 1).
    # Batching multiple differently-structured table crops in a single
    # table_pred(crops) call crashes surya 0.17.1's TableRecPredictor
    # (decoder position_ids/cache sized from the first image only).
    table_bboxes = [b for b in layout_result.bboxes if b.label == "Table"]
    tables: list[dict] = []
    for b in table_bboxes:
        crop = pil_img.crop(tuple(map(int, b.bbox)))
        try:
            t = table_pred([crop])[0]
        except Exception:
            # Retry once with a freshly-constructed TableRecPredictor: a
            # stale decoder cache/position-id state on the shared singleton
            # can otherwise cause a one-off tensor-shape mismatch on this crop.
            try:
                t = _new_table_predictor()([crop])[0]
            except Exception as e:
                warnings.warn(f"Table recognition failed on page {page_index}: {e}")
                continue
        if t.cells:
            try:
                cell_bboxes = [list(map(int, c.bbox)) for c in t.cells]
                cell_ocr = rec_pred([crop], bboxes=[cell_bboxes])[0]
                for cell, line in zip(t.cells, cell_ocr.text_lines):
                    cell.text_lines = [{"text": line.text, "bbox": line.bbox}]
            except Exception as e:
                warnings.warn(f"Cell OCR failed: {e}")
        tbl = _serialize_table(t)
        tbl["cells"] = _filter_phantom_cells(tbl["cells"], tbl["image_bbox"])
        tables.append(tbl)

    return SuryaPageResult(
        page_index=page_index,
        text_blocks=sorted_blocks,
        tables=tables,
        ocr_text=ocr_text,
    )



def _serialize_text_line(line) -> dict[str, Any]:
    return {
        "text": line.text,
        "bbox": list(line.bbox),
        "polygon": [list(p) for p in (line.polygon or [])],
        "confidence": line.confidence,
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


