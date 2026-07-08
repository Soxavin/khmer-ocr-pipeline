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
from .kiri_recognizer import recognize_cells_conf, reset_kiri_failure

# Cells below this per-cell recognizer confidence trigger a page-level warning.
_LOW_CONF_THRESHOLD = 0.80


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

    # Retry the Kiri load once per run: clear a prior within-run failure latch so
    # a transient first-run HF blip doesn't leave Kiri disabled for the whole
    # (long-lived Streamlit) process. The latch still guards against ~240 repeated
    # download attempts within this run.
    reset_kiri_failure()

    # 1. Reuse Surya for page-level text blocks + ocr_text; skip_tables=True
    #    drops Table regions before recognition (we rebuild tables from
    #    raw-image structure below, so Surya's table VLM would be wasted work).
    base = run_surya(result, on_page, skip_tables=True)

    # 2. Lazy-init predictors on Surya's shared inference manager.
    manager = get_manager()
    layout_pred, _ = _get_predictors()
    from surya.table_rec import TableRecPredictor
    _tbl_pred = TableRecPredictor(manager=manager)

    pages: list[SuryaPageResult] = []
    extra_warnings: list[str] = []
    # Per-run sink for Kiri recognizer failures (unavailable / per-batch), so they
    # reach SuryaResult.warnings instead of being lost to warnings.warn after
    # run_surya's capture window has closed.
    kiri_warnings: list[str] = []

    # Recognise from the geometric-only image: deskew/crop are applied (deskew is
    # needed for skewed scans) but photometric normalization (CLAHE/desaturation)
    # is skipped, since it degrades Kiri's per-cell Otsu binarization. The
    # geometric-only and fully-preprocessed frames share IDENTICAL geometry (same
    # crop+cap+deskew, run before any photometric step — asserted in preprocess()),
    # so table bboxes map 1:1 between them; they differ only photometrically. We
    # still run the whole table pipeline (layout → TableRec → crop) on the
    # geometric-only page so the per-cell crops keep their un-normalized pixels.
    if result.recognition_page_images is not None:
        raw_imgs = result.recognition_page_images
    else:
        # Falling back to the photometrically-preprocessed pages is a MEASURED
        # accuracy loss (0.79→0.675, PROJECT_LOG §2.30) — surface it, never silent.
        raw_imgs = result.page_images
        extra_warnings.append(
            "Surya+Kiri: recognition images unavailable — falling back to "
            "photometrically-preprocessed pages; accuracy may be reduced."
        )

    for idx, page in enumerate(base.pages):
        img = raw_imgs[idx]
        h, w = img.shape[:2]
        page_low_conf_count = 0

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
        for m_idx, mb in enumerate(merged, start=1):
            x0, y0, x1, y1 = (
                max(0, int(mb[0])), max(0, int(mb[1])),
                min(w, int(mb[2])), min(h, int(mb[3])),
            )
            if x1 <= x0 or y1 <= y0:
                continue

            crop = img[y0:y1, x0:x1]
            try:
                cells = _tbl_pred([Image.fromarray(crop)])[0].cells
            except Exception as exc:
                # Never drop a table silently: the analyst would see "no table" on
                # a page that visibly has one. Surface it via the warnings channel.
                extra_warnings.append(
                    f"Surya+Kiri page {page.page_index + 1}: table structure prediction "
                    f"failed for region {m_idx} ({exc}) — table omitted; try the Surya "
                    f"engine for this page."
                )
                continue

            if not cells:
                extra_warnings.append(
                    f"Surya+Kiri page {page.page_index + 1}: table structure prediction "
                    f"returned no cells for region {m_idx} — table omitted; try the Surya "
                    f"engine for this page."
                )
                continue

            # Build a (row, col) → text grid from TableRecPredictor cells, plus a
            # parallel confidence grid.
            # Pass 1: compute axis-aligned crops, applying the size guard directly
            # so tiny cells get "" without ever reaching the batched recognizer.
            grid: dict[tuple[int, int], str] = {}
            conf_grid: dict[tuple[int, int], float] = {}
            pending_keys: list[tuple[int, int]] = []
            pending_crops: list[np.ndarray] = []
            for cell in cells:
                # Polygon → axis-aligned bounding box within the crop.
                xs = [p[0] for p in cell.polygon]
                ys = [p[1] for p in cell.polygon]
                cx0, cy0, cx1, cy1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                cx0, cy0 = max(0, cx0), max(0, cy0)
                cx1, cy1 = min(crop.shape[1], cx1), min(crop.shape[0], cy1)

                key = (cell.row_id, cell.col_id)
                if cx1 - cx0 < 3 or cy1 - cy0 < 3:
                    # Intentionally blank (too small to hold text), not low-confidence.
                    grid[key] = ""
                    conf_grid[key] = 1.0
                else:
                    pending_keys.append(key)
                    pending_crops.append(crop[cy0:cy1, cx0:cx1])

            # Pass 2: recognize all qualifying cells for this table in one batch.
            texts_confs = recognize_cells_conf(pending_crops, warning_sink=kiri_warnings)
            for key, (text, conf) in zip(pending_keys, texts_confs):
                grid[key] = text
                conf_grid[key] = conf

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
            for tcell in table["cells"]:
                conf = conf_grid.get((tcell["row_id"], tcell["col_id"]), 1.0)
                tcell["confidence"] = conf
                if conf < _LOW_CONF_THRESHOLD:
                    page_low_conf_count += 1
            new_tables.append(table)

        clear_device_cache()

        if page_low_conf_count:
            extra_warnings.append(
                f"Surya+Kiri page {page.page_index + 1}: {page_low_conf_count} table "
                f"cell(s) below {_LOW_CONF_THRESHOLD:.0%} confidence — verify those cells."
            )

        if not new_tables:
            pages.append(page)
            continue

        pages.append(SuryaPageResult(
            page_index=page.page_index,
            text_blocks=page.text_blocks,
            tables=new_tables,
            ocr_text=page.ocr_text,
        ))

    # Merge unique Kiri warnings once (a load failure repeats per table crop).
    extra_warnings.extend(dict.fromkeys(kiri_warnings))

    return SuryaResult(
        source_name=base.source_name,
        pages=pages,
        warnings=base.warnings + extra_warnings,
    )
