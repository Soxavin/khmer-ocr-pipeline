"""Surya + Kiri + Otsu hybrid OCR engine.

Combines Surya's layout detection and table-structure prediction (TableRecPredictor)
with per-cell Otsu binarization and Kiri CTC recognition. Registered as
``OCR_ENGINE=surya_kiri``.

Architecture
------------
1. ``run_surya(skip_tables=True)`` for the page-level text layer only — Table
   regions are dropped before recognition, so Surya's expensive table-HTML VLM
   never runs (this engine rebuilds tables from raw-image structure instead).
2. A dedicated layout + ``TableRecPredictor`` pass on the RAW page for cell
   polygons + (row_id, col_id) structure.
3. Per cell: Otsu threshold → Kiri CTC decode → strip trailing dots.
4. ``_build_table_from_grid`` to produce standard pipeline ``Table`` dicts.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..models import PreprocessResult, SuryaResult, SuryaPageResult
from ..utils.memory import clear_device_cache
from .surya import run_surya, get_manager, _get_predictors, _build_table_from_grid
from .table_stitch import merge_table_regions
from .kiri_recognizer import recognize_cell


def run_surya_kiri(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> SuryaResult:
    """Run the Surya + Kiri hybrid pipeline over every page in *result*.

    Returns a ``SuryaResult`` whose tables are built from TableRecPredictor
    structure + Kiri CTC recognition instead of Surya's VLM HTML output.
    """
    # TODO(speed): the table-VLM waste is now resolved via `skip_tables=True`
    # below. The only remaining minor redundancy is the second layout pass on
    # the raw page (below) to locate table regions for TableRecPredictor.

    # 1. Reuse Surya for page-level text blocks + ocr_text; skip_tables=True
    #    drops Table regions before recognition (we rebuild tables from
    #    raw-image structure below, so Surya's table VLM would be wasted work).
    base = run_surya(result, on_page, skip_tables=True)

    # 2. Lazy-init predictors on Surya's shared inference manager.
    manager = get_manager()
    layout_pred, _ = _get_predictors()
    from surya.table_rec import TableRecPredictor
    _tbl_pred = TableRecPredictor(manager=manager)

    # Recognise from RAW pixels: preprocessing (CLAHE/desaturation) helps Surya's
    # structure but degrades Kiri recognition. Because preprocessing also deskews/
    # crops, preprocessed-space bboxes don't map onto raw pixels — so the WHOLE
    # table pipeline (layout → TableRec → crop) runs on the raw page here.
    raw_imgs = result.raw_page_images if result.raw_page_images is not None else result.page_images

    pages: list[SuryaPageResult] = []
    for idx, page in enumerate(base.pages):
        img = raw_imgs[idx]
        h, w = img.shape[:2]

        # Detect table regions on the raw page (own layout pass, not base.tables).
        layout = layout_pred([Image.fromarray(img)])[0]
        table_bboxes = [tuple(float(v) for v in b.bbox) for b in layout.bboxes if b.label == "Table"]
        if not table_bboxes:
            pages.append(page)
            continue

        merged = merge_table_regions(table_bboxes)
        if not merged:
            pages.append(page)
            continue

        new_tables: list[dict] = []
        for mb in merged:
            x0, y0, x1, y1 = (
                max(0, int(mb[0])), max(0, int(mb[1])),
                min(w, int(mb[2])), min(h, int(mb[3])),
            )
            if x1 <= x0 or y1 <= y0:
                continue

            crop = img[y0:y1, x0:x1]
            try:
                cells = _tbl_pred([Image.fromarray(crop)])[0].cells
            except Exception:
                continue

            if not cells:
                continue

            # Build a (row, col) → text grid from TableRecPredictor cells.
            grid: dict[tuple[int, int], str] = {}
            for cell in cells:
                # Polygon → axis-aligned bounding box within the crop.
                xs = [p[0] for p in cell.polygon]
                ys = [p[1] for p in cell.polygon]
                cx0, cy0, cx1, cy1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                cx0, cy0 = max(0, cx0), max(0, cy0)
                cx1, cy1 = min(crop.shape[1], cx1), min(crop.shape[0], cy1)

                if cx1 - cx0 < 3 or cy1 - cy0 < 3:
                    text = ""
                else:
                    cell_crop = crop[cy0:cy1, cx0:cx1]
                    text = recognize_cell(cell_crop)

                grid[(cell.row_id, cell.col_id)] = text

            if not grid:
                continue

            table = _build_table_from_grid(
                grid, "",
                [float(x0), float(y0), float(x1), float(y1)],
            )
            # run_surya sets a top-level "bbox" on each table (surya.py); the
            # layout overlay in app.py reads t["bbox"]. _build_table_from_grid only
            # sets "image_bbox", so mirror run_surya and set "bbox" here too.
            table["bbox"] = [float(x0), float(y0), float(x1), float(y1)]
            new_tables.append(table)

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

    return SuryaResult(
        source_name=base.source_name,
        pages=pages,
        warnings=base.warnings,
    )
