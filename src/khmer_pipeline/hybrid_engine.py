from __future__ import annotations
import os
from typing import Callable, Optional
import numpy as np
from PIL import Image

from .models import PreprocessResult, SuryaResult, SuryaPageResult
from .table_stitch import merge_table_regions
from .slanet_structure import predict_cells
from .surya import run_surya, _get_predictors, _parse_html_table, _build_table_from_grid
from .memory import clear_device_cache

# Hybrid OCR engine: SLANet for table STRUCTURE (unified grid + per-cell coords),
# Surya for RECOGNITION. Two modes (KHMER_HYBRID_MODE):
#   "rowband" (default) — read each SLANet row as one full-width strip with
#               label="Table"; Surya's VLM reads the natural line AND emits <td>
#               columns itself, parsed back with the pure-Surya _parse_html_table.
#               On the fragmented real table it lifts Cell_Accuracy 0.024→0.393 and
#               beats cell mode on every metric (PROJECT_LOG 2.17).
#   "cell"    — read each SLANet cell as its own crop. Reliable structure but the
#               VLM hallucinates on tiny isolated crops (PROJECT_LOG 2.15); kept for
#               comparison only.
# Paragraph/header text + table DETECTION are reused from run_surya unchanged.

_ROW_STRIP_Y_PAD_PX = 8


def _hybrid_mode() -> str:
    return os.environ.get("KHMER_HYBRID_MODE", "rowband")


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


def _row_bands(cells: list[dict], crop_w: int, crop_h: int) -> list[dict]:
    # One full-width strip per SLANet row. x always spans the whole table crop so a
    # missing/short cell can't narrow the strip and rob the VLM of column context;
    # y is padded so ascenders/descenders and the grid lines (which the VLM uses to
    # emit <td>s) are not clipped.
    by_row: dict[int, list[dict]] = {}
    for c in cells:
        by_row.setdefault(c["row_id"], []).append(c)
    bands = []
    for row_id in sorted(by_row):
        ys = [v for c in by_row[row_id] for v in (c["bbox"][1], c["bbox"][3])]
        y0 = max(0, int(min(ys)) - _ROW_STRIP_Y_PAD_PX)
        y1 = min(crop_h, int(max(ys)) + _ROW_STRIP_Y_PAD_PX)
        bands.append({"row_id": row_id, "bbox": [0, y0, crop_w, y1]})
    return bands


def _ocr_rowbands(rec_pred, crop_rgb: np.ndarray, bands: list[dict]) -> dict[tuple[int, int], str]:
    # Recognise each full-width row strip as a one-row "Table"; Surya emits the row's
    # <td> columns. Bands are concatenated into one grid, local row indices offset so
    # each band occupies its own global row (blank bands still reserve a row slot).
    from surya.layout.schema import LayoutResult, LayoutBox

    h, w = crop_rgb.shape[:2]
    boxes = []
    for i, b in enumerate(bands):
        x0, y0, x1, y1 = b["bbox"]
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        boxes.append(LayoutBox(polygon=poly, label="Table", raw_label="Table",
                               position=i, count=0, confidence=1.0))
    layout = LayoutResult(bboxes=boxes, image_bbox=[0.0, 0.0, float(w), float(h)], error=False)
    page_ocr = rec_pred([Image.fromarray(crop_rgb)], layout_results=[layout], full_page=False)[0]

    grid: dict[tuple[int, int], str] = {}
    row_offset = 0
    for block in page_ocr.blocks:
        local = _parse_html_table(block.html or "")
        local_rows = (max(r for r, _ in local) + 1) if local else 0
        for (r, c), text in local.items():
            grid[(row_offset + r, c)] = text
        row_offset += max(local_rows, 1)
    return grid


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
    rowband = _hybrid_mode() == "rowband"

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
            if rowband:
                ch, cw = crop.shape[:2]
                grid = _ocr_rowbands(rec_pred, crop, _row_bands(cells, cw, ch))
                new_tables.append(_build_table_from_grid(grid, "", [x0, y0, x1, y1]))
            else:
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
