from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from PIL import Image

from .models import PreprocessResult, SuryaResult, SuryaPageResult
from .table_stitch import merge_table_regions
from .slanet_structure import predict_cells
from .surya import run_surya, _get_predictors
from .memory import clear_device_cache

# Hybrid OCR engine: SLANet for table STRUCTURE (unified grid + per-cell coords),
# Surya for RECOGNITION (read each small cell). Solves the fragmentation bottleneck
# that pure-geometric stitching could not (see PROJECT_LOG 2.12-2.14): small cell
# crops keep recall high while SLANet supplies correct row<->value structure.
# Paragraph/header text + table DETECTION are reused from run_surya unchanged.


def _ocr_cells(rec_pred, crop_rgb: np.ndarray, cells: list[dict]) -> list[str]:
    # OCR each SLANet cell via Surya block-mode recognition: one LayoutBox per cell
    # (coords relative to the crop), recognition crops+reads each, blocks come back
    # in the same order as the boxes. Returns text per cell, aligned to `cells`.
    from surya.layout.schema import LayoutResult, LayoutBox
    from .surya import _html_to_text

    h, w = crop_rgb.shape[:2]
    boxes = []
    for i, c in enumerate(cells):
        x0, y0, x1, y1 = c["bbox"]
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        boxes.append(LayoutBox(polygon=poly, label="Text", raw_label="Text",
                               position=i, count=0, confidence=1.0))
    layout = LayoutResult(bboxes=boxes, image_bbox=[0.0, 0.0, float(w), float(h)], error=False)
    page_ocr = rec_pred([Image.fromarray(crop_rgb)], layout_results=[layout], full_page=False)[0]
    return [_html_to_text(b.html) for b in page_ocr.blocks]


def _build_table(cells: list[dict], texts: list[str], master_bbox) -> dict:
    # Assemble SLANet cells + per-cell text into our standard table dict shape.
    n_rows = max((c["row_id"] + c["row_span"] for c in cells), default=0)
    n_cols = max((c["col_id"] + c["col_span"] for c in cells), default=0)
    out_cells = []
    for cid, (c, txt) in enumerate(zip(cells, texts)):
        out_cells.append({
            "row_id": c["row_id"], "col_id": c["col_id"], "cell_id": cid,
            "bbox": [], "polygon": [],
            "text_lines": [{"text": txt, "bbox": []}] if txt else [],
        })
    return {
        "rows": [{"row_id": i} for i in range(n_rows)],
        "cols": [{"col_id": j} for j in range(n_cols)],
        "cells": out_cells,
        "image_bbox": list(master_bbox),
        "bbox": list(master_bbox),
    }


def run_hybrid(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> SuryaResult:
    # Reuse Surya for page text + table detection, then rebuild each table with
    # SLANet structure + per-cell Surya recognition.
    base = run_surya(result, on_page)
    _, rec_pred = _get_predictors()

    pages: list[SuryaPageResult] = []
    for idx, page in enumerate(base.pages):
        boxes = [tuple(float(v) for v in t["bbox"]) for t in page.tables if t.get("bbox")]
        if not boxes:
            pages.append(page)
            continue
        img = result.page_images[idx]
        h, w = img.shape[:2]
        new_tables: list[dict] = []
        for mb in merge_table_regions(boxes):
            x0, y0, x1, y1 = (max(0, int(mb[0])), max(0, int(mb[1])),
                              min(w, int(mb[2])), min(h, int(mb[3])))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1]
            cells = predict_cells(crop)
            if not cells:
                continue
            texts = _ocr_cells(rec_pred, crop, cells)
            new_tables.append(_build_table(cells, texts, (x0, y0, x1, y1)))
        clear_device_cache()
        if not new_tables:
            pages.append(page)
            continue
        pages.append(SuryaPageResult(
            page_index=page.page_index,
            text_blocks=page.text_blocks,
            tables=new_tables,
            ocr_text=page.ocr_text,
        ))

    return SuryaResult(source_name=base.source_name, pages=pages, warnings=base.warnings)
