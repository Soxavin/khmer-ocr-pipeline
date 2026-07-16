"""Surya-VLM + Kiri hybrid: plain Surya's structure+text, Kiri re-reads Khmer cells.

Registered as ``OCR_ENGINE=surya_kiri_vlm``. The §2.36 "Surya-structure + Kiri-text"
variant, corrected by §2.37 and enabled by the §2.39 fine-tuned Kiri weights.

Why this shape (PROJECT_LOG §2.36→§2.41): plain Surya's table strength — correct
structure, placement, and SPANS — lives in its table VLM (joint structure+text HTML
inference). `surya_kiri` skips that VLM and inherits TableRec's span-blindness;
post-hoc span repair failed in production (§2.40). This engine keeps the VLM as the
single source of structure AND text, then upgrades ONLY the Khmer-heavy cells with
Kiri (numbers/Latin stay Surya — its strength).

Safety contract — the engine's floor is plain Surya:
- Kiri re-reads happen only when TableRec's grid shape EXACTLY equals the VLM grid
  shape (a discrete integer gate, not a pixel threshold — the §2.40 lesson).
- Every fallback (gate mismatch, TableRec failure, low Kiri confidence, empty read)
  keeps Surya's text untouched. There is no path that corrupts a cell.
- Re-read cells carry `confidence`; untouched cells don't — the UI confidence view
  shows exactly which cells Kiri touched.

Cost: the slowest engine (the table VLM runs, unlike surya_kiri's skip_tables path)
plus a TableRec pass per table that contains Khmer-heavy cells.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..models import PreprocessResult, SuryaResult, SuryaPageResult
from ..utils.memory import clear_device_cache
from .surya import run_surya, get_manager
from .kiri_recognizer import recognize_cells_conf, reset_kiri_failure

# A cell is "Khmer-heavy" (re-read candidate) when at least this fraction of its
# non-space characters are in the Khmer block.
_KHMER_HEAVY_MIN_RATIO = 0.5
# Kiri's read replaces Surya's text only at/above this confidence; below it (or on
# an empty read) Surya's text is kept — the safe direction.
_KIRI_REPLACE_MIN_CONF = 0.5
# Pad around each re-read crop: TableRec's cell-tight boxes can clip Khmer
# ascenders/descenders (diacritics above/below the baseline); a small symmetric
# buffer is safe — Otsu binarization tolerates the sliver of neighbouring
# gridline it may include.
_CROP_PAD_PX = 2


def _khmer_ratio(text: str) -> float:
    """Fraction of non-space characters in the Khmer Unicode block (U+1780–U+17FF)."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if "\u1780" <= ch <= "\u17ff") / len(chars)


def _cell_text(cell: dict) -> str:
    return " ".join(t["text"] for t in (cell.get("text_lines") or []) if t.get("text")).strip()


def run_surya_kiri_vlm(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> SuryaResult:
    """Run plain Surya in full, then re-OCR Khmer-heavy table cells with Kiri.

    Returns a ``SuryaResult`` whose tables keep the VLM's structure and non-Khmer
    text verbatim; Khmer-heavy cells are replaced by Kiri reads when the TableRec
    geometry gate and the confidence floor both pass.
    """
    reset_kiri_failure()

    # 1. Plain Surya, VLM included — structure, spans, and all text.
    base = run_surya(result, on_page)

    manager = get_manager()
    from surya.table_rec import TableRecPredictor
    tbl_pred = TableRecPredictor(manager=manager)

    extra_warnings: list[str] = []
    kiri_warnings: list[str] = []

    # Kiri recognizes from the geometric-only frame (§2.30: photometric steps hurt
    # per-cell Otsu); region bboxes map 1:1 between frames (asserted in preprocess()).
    if result.recognition_page_images is not None:
        raw_imgs = result.recognition_page_images
    else:
        raw_imgs = result.page_images
        extra_warnings.append(
            "Surya+Kiri(VLM): recognition images unavailable — falling back to "
            "photometrically-preprocessed pages; accuracy may be reduced."
        )

    pages: list[SuryaPageResult] = []
    for idx, page in enumerate(base.pages):
        img = raw_imgs[idx]
        h, w = img.shape[:2]
        new_tables: list[dict] = []

        for t_idx, table in enumerate(page.tables, start=1):
            cells = table.get("cells", [])
            bbox = table.get("bbox") or table.get("image_bbox")
            khmer_cells = [c for c in cells if _khmer_ratio(_cell_text(c)) >= _KHMER_HEAVY_MIN_RATIO]
            if not khmer_cells or not bbox or len(bbox) < 4:
                new_tables.append(table)  # nothing to upgrade → pure Surya table
                continue

            x0, y0 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x1, y1 = min(w, int(bbox[2])), min(h, int(bbox[3]))
            if x1 <= x0 or y1 <= y0:
                new_tables.append(table)
                continue
            crop = img[y0:y1, x0:x1]

            try:
                tr_cells = tbl_pred([Image.fromarray(crop)])[0].cells
            except Exception as exc:
                extra_warnings.append(
                    f"Surya+Kiri(VLM) page {page.page_index + 1}: cell geometry failed for "
                    f"table {t_idx} ({exc}) — Khmer re-read skipped (Surya text kept)."
                )
                new_tables.append(table)
                continue

            # GATE: the two grids must agree exactly on shape, otherwise (row, col)
            # indices are not comparable and any crop mapping would be a guess.
            tr_rows = max((c.row_id for c in tr_cells), default=-1) + 1
            tr_cols = max((c.col_id for c in tr_cells), default=-1) + 1
            vlm_rows, vlm_cols = len(table.get("rows", [])), len(table.get("cols", []))
            if (tr_rows, tr_cols) != (vlm_rows, vlm_cols):
                extra_warnings.append(
                    f"Surya+Kiri(VLM) page {page.page_index + 1} table {t_idx}: grid mismatch "
                    f"(VLM {vlm_rows}x{vlm_cols} vs TableRec {tr_rows}x{tr_cols}) — Khmer "
                    f"re-read skipped (Surya text kept)."
                )
                new_tables.append(table)
                continue

            unit: dict[tuple[int, int], list[float]] = {}
            for c in tr_cells:
                xs = [p[0] for p in c.polygon]
                ys = [p[1] for p in c.polygon]
                unit[(c.row_id, c.col_id)] = [min(xs), min(ys), max(xs), max(ys)]

            # Build the batch: one crop per Khmer-heavy cell; a colspan anchor's
            # crop is the union of its spanned TableRec units.
            pending: list[dict] = []
            crops: list[np.ndarray] = []
            for cell in khmer_cells:
                r, c0 = cell["row_id"], cell["col_id"]
                boxes = [unit.get((r, c0 + k)) for k in range(cell.get("col_span", 1))]
                boxes = [b for b in boxes if b is not None]
                if not boxes:
                    continue
                cx0 = max(0, int(min(b[0] for b in boxes)) - _CROP_PAD_PX)
                cy0 = max(0, int(min(b[1] for b in boxes)) - _CROP_PAD_PX)
                cx1 = min(crop.shape[1], int(max(b[2] for b in boxes)) + _CROP_PAD_PX)
                cy1 = min(crop.shape[0], int(max(b[3] for b in boxes)) + _CROP_PAD_PX)
                if cx1 - cx0 < 3 or cy1 - cy0 < 3:
                    continue
                pending.append(cell)
                crops.append(crop[cy0:cy1, cx0:cx1])

            if not crops:
                new_tables.append(table)
                continue

            texts_confs = recognize_cells_conf(crops, warning_sink=kiri_warnings)
            replacements: dict[int, dict] = {}
            for cell, (text, conf) in zip(pending, texts_confs):
                if conf >= _KIRI_REPLACE_MIN_CONF and text.strip():
                    replacements[id(cell)] = {
                        **cell,
                        "text_lines": [{"text": text, "bbox": []}],
                        "confidence": conf,
                    }
            if not replacements:
                new_tables.append(table)
                continue
            # Copy-on-write: the base result is never mutated.
            new_tables.append({
                **table,
                "cells": [replacements.get(id(c), c) for c in cells],
            })

        clear_device_cache()
        pages.append(SuryaPageResult(
            page_index=page.page_index,
            text_blocks=page.text_blocks,
            tables=new_tables,
            ocr_text=page.ocr_text,
        ))

    extra_warnings.extend(dict.fromkeys(kiri_warnings))
    return SuryaResult(
        source_name=base.source_name,
        pages=pages,
        warnings=base.warnings + extra_warnings,
    )
