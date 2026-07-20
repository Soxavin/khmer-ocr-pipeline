"""Surya + Kiri + Otsu hybrid OCR engine.

Combines Surya's layout detection and a table-structure model with per-cell Otsu
binarization and Kiri CTC recognition. Registered as ``OCR_ENGINE=surya_kiri``.

Architecture
------------
1. ``run_surya(skip_tables=True)`` for the page-level text layer only — Table
   regions are dropped before recognition, so Surya's expensive table-HTML VLM
   never runs (this engine rebuilds tables from raw-image structure instead).
2. A dedicated layout pass on the RAW page, then a structure model per table
   region, selected via ``KHMER_KIRI_STRUCTURE``:
   - ``tablerec`` (default) — pure TableRec. Its simple path emits row×col
     INTERSECTIONS with no spanning info, so column-spanning cells are
     fragmented and their text split at internal boundaries ("14-06-26" →
     "14" | "6-26") — a KNOWN, accepted limitation: it never corrupts data
     cells, and analysts can repair the few affected header cells in the UI.
   - ``merged`` (opt-in, §2.40) — TableRec grid + SLANet cell boxes as span
     PROPOSALS confirmed by pixel evidence (`_has_vertical_separator`). Passed
     the eval gate (zero regressions on 8 GT pages) but produced a data-cell
     false merge in production UI use (ID column merged into the product-name
     column at a rendering the probes didn't cover), so it was demoted from
     default the same day: the stroke-vs-broken-gridline margin (0.31 vs 0.51)
     is too narrow to trust across renderings. Data integrity > header
     cosmetics.
   - ``slanet`` — structure wholly from ``slanet_structure.predict_cells``.
     Real spans, but its grid measured worse on data regions (lost a column on
     ARDB p1) — kept for comparison only.
3. Per cell: Otsu threshold → Kiri CTC decode → strip trailing dots.
4. ``_build_table_from_grid`` to produce standard pipeline ``Table`` dicts
   (+ optional ``row_span``/``col_span`` metadata on spanning cells).
"""
from __future__ import annotations

import os
import warnings
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..models import PreprocessResult, SuryaResult, SuryaPageResult
from ..utils.memory import clear_device_cache
from .surya import run_surya, get_manager, _get_predictors, _build_table_from_grid
from .layout_detect import detect_table_boxes, detector_enabled
from .slanet_structure import predict_cells
from .table_stitch import merge_table_regions
from .kiri_recognizer import recognize_cells_conf, reset_kiri_failure

# Cells below this per-cell recognizer confidence trigger a page-level warning.
_LOW_CONF_THRESHOLD = 0.80


def _kiri_structure() -> str:
    """Structure-model source: 'tablerec' (default — pure TableRec; splits
    column-spanning headers but NEVER merges data cells), 'merged' (opt-in —
    TableRec grid + SLANet span merging with pixel confirmation; passed the
    §2.40 eval gate but produced a data-cell false merge in production UI use,
    so the default was reverted the same day — see §2.40 postscript), or
    'slanet' (structure wholly from SLANet — worse on data grids, comparison
    only)."""
    return os.environ.get("KHMER_KIRI_STRUCTURE", "tablerec")


# Coverage thresholds for span merging, measured on ARDB p1: genuine spans
# cover their constituent TableRec units 92–105% in x, while a drifted
# neighbour box only reaches ~33% — 0.60 separates them cleanly. y stays at
# 0.50 (row alignment is solid in both models, and a loose y would let header
# boxes grab data rows).
_SPAN_X_OVERLAP_MIN = 0.60
_SPAN_Y_OVERLAP_MIN = 0.50


# Vertical-separator detection between two unit cells. Metric: the longest
# CONTIGUOUS strong-gradient run along y in any single x-column of the scan
# band, as a fraction of the band height. Measured on ARDB pages: genuine
# merged cells (text strokes only) max out at ~0.31, while real boundaries
# score 0.51 (p3's text-broken ល.រ/item rule) to 1.0 (clean fill gaps) —
# 0.45 splits the populations. The scan spans center-of-A → center-of-B
# because the painted boundary need not coincide with TableRec's geometric
# column edge (measured 7px off).
_SEP_EDGE_LUMA_DELTA = 30.0
_SEP_MIN_RUN_COVERAGE = 0.45


def _longest_run_fraction(mask_col: np.ndarray) -> float:
    """Longest contiguous run of True in a 1-D bool array / its length."""
    best = cur = 0
    for v in mask_col:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best / len(mask_col) if len(mask_col) else 0.0


def _table_regions(layout_pred, img: np.ndarray) -> list[tuple]:
    """Table-region bboxes for a page: the fine-tuned layout detector when
    KHMER_LAYOUT_WEIGHTS is set, else Surya's own layout pass.

    surya_kiri runs its own layout pass (it never sees run_surya's table boxes), so
    the detector must be wired in HERE as well as in surya.py — otherwise
    KHMER_LAYOUT_WEIGHTS silently no-ops on this engine (§2.43). An empty detection
    falls back to Surya's boxes rather than dropping the table.
    """
    layout = layout_pred([Image.fromarray(img)])[0]
    surya_boxes = [tuple(float(v) for v in b.bbox)
                   for b in layout.bboxes if b.label == "Table"]
    if not detector_enabled():
        return surya_boxes
    detected = detect_table_boxes(img)
    if not detected:
        warnings.warn("Fine-tuned layout detector found no table; keeping Surya's layout boxes.")
        return surya_boxes
    return [tuple(float(v) for v in b) for b in detected]


def _has_vertical_separator(crop: np.ndarray, unit_a: list[float], unit_b: list[float]) -> bool:
    """True when a vertical rule or fill-gap separates two horizontally adjacent
    unit-cell bboxes in `crop` — i.e. the cells are visually distinct, so a
    proposed span merge across them is bogus. A separator is an x-column whose
    luminance gradient is strong over a near-full-height CONTIGUOUS run; text
    strokes crossing a genuine merged cell are short runs."""
    y0 = int(max(unit_a[1], unit_b[1]))
    y1 = int(min(unit_a[3], unit_b[3]))
    x0 = max(0, int((unit_a[0] + unit_a[2]) / 2))
    x1 = min(crop.shape[1], int((unit_b[0] + unit_b[2]) / 2))
    if y1 - y0 < 8 or x1 - x0 < 2:
        # Nothing to measure → treat as separated: a merge may only proceed on
        # POSITIVE pixel evidence of openness (a false merge eats a data cell;
        # a missed merge just keeps today's split-text behaviour).
        return True
    band = crop[y0:y1, x0:x1].astype(float).mean(axis=2)  # luminance
    gx = np.abs(np.diff(band, axis=1)) > _SEP_EDGE_LUMA_DELTA
    return any(
        _longest_run_fraction(gx[:, i]) >= _SEP_MIN_RUN_COVERAGE
        for i in range(gx.shape[1])
    )


def _merge_spans(records: list[tuple], span_boxes: list[list[float]],
                 crop: np.ndarray | None = None) -> list[tuple]:
    """Merge TableRec unit-cell records covered by one SLANet cell box into ONE
    record (union bbox, anchored at the min col, col_span = count of units).

    `span_boxes` are ALL SLANet cell bboxes, not only its logical spans: SLANet
    under-reports col_span when its own grid loses a column (measured on ARDB
    p1 — the 2nd date header came back logical-unit cs=1 while its PHYSICAL box
    covered two TableRec columns at 100%/92%). The physical box is the
    trustworthy signal; a span "exists" exactly when one SLANet box
    substantially covers ≥2 TableRec units.

    `records` are (row_id, col_id, bbox, row_span, col_span) with crop-relative
    bboxes. Only SAME-ROW, CONSECUTIVE-COLUMN merges are produced — SLANet's
    multi-row block spans over sparse data regions consumed real data cells
    when honored (measured on ARDB p1, col-4 digit rows 22→16). A unit is
    covered when ≥60% of its width and ≥50% of its height lie inside the box,
    consumed by at most one box (first wins); boxes covering fewer than two
    free units are ignored (the normal unit-cell case). Pure geometry — no
    text heuristics (PROJECT_LOG §2.30) and TableRec's trusted unit grid stays
    the base (§2.37): SLANet only says which units belong together.
    """
    consumed: set[int] = set()
    merged: list[tuple] = []
    for sb in span_boxes:
        sx0, sy0, sx1, sy1 = (float(v) for v in sb[:4])
        covered = []
        for i, (_r, _c, b, _rs, _cs) in enumerate(records):
            if i in consumed:
                continue
            bw, bh = b[2] - b[0], b[3] - b[1]
            if bw <= 0 or bh <= 0:
                continue
            ox = max(0.0, min(sx1, b[2]) - max(sx0, b[0]))
            oy = max(0.0, min(sy1, b[3]) - max(sy0, b[1]))
            if ox / bw >= _SPAN_X_OVERLAP_MIN and oy / bh >= _SPAN_Y_OVERLAP_MIN:
                covered.append(i)
        if len(covered) < 2:
            continue
        rows = {records[i][0] for i in covered}
        if len(rows) != 1:
            continue  # block span → data-eating risk; ignore
        covered.sort(key=lambda i: records[i][1])
        cols = [records[i][1] for i in covered]
        if cols != list(range(cols[0], cols[0] + len(cols))):
            continue  # non-consecutive columns → not a real horizontal span
        if crop is not None and any(
            _has_vertical_separator(crop, records[a][2], records[b][2])
            for a, b in zip(covered, covered[1:])
        ):
            continue  # a rule/gap divides the units → the "span" is grid drift
        union = [
            min(records[i][2][0] for i in covered),
            min(records[i][2][1] for i in covered),
            max(records[i][2][2] for i in covered),
            max(records[i][2][3] for i in covered),
        ]
        merged.append((rows.pop(), cols[0], union, 1, len(cols)))
        consumed.update(covered)
    return [rec for i, rec in enumerate(records) if i not in consumed] + merged


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

    # 2. Lazy-init predictors on Surya's shared inference manager. TableRec is
    #    only instantiated when it is the selected structure source — the slanet
    #    path must not pay for (or depend on) it.
    structure = _kiri_structure()
    manager = get_manager()
    layout_pred, _ = _get_predictors()
    _tbl_pred = None
    if structure != "slanet":
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
        table_bboxes = _table_regions(layout_pred, img)
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
                # Normalize both structure sources to (row_id, col_id, bbox,
                # row_span, col_span) with crop-relative bboxes. TableRec's simple
                # path has no span info (its schema: cells are row×col
                # intersections), so its spans are always 1×1; SLANet emits real
                # spans, giving a spanning cell ONE full-width crop.
                if structure == "slanet":
                    records = [
                        (c["row_id"], c["col_id"], c["bbox"],
                         c.get("row_span", 1), c.get("col_span", 1))
                        for c in predict_cells(crop)
                    ]
                else:
                    records = []
                    for cell in _tbl_pred([Image.fromarray(crop)])[0].cells:
                        xs = [p[0] for p in cell.polygon]
                        ys = [p[1] for p in cell.polygon]
                        records.append((cell.row_id, cell.col_id,
                                        [min(xs), min(ys), max(xs), max(ys)], 1, 1))
                    if structure == "merged":
                        # SLANet is only a span detector here; if it breaks, the
                        # TableRec table still stands — warn and keep it unmerged.
                        # ALL its cell boxes are candidates (not just logical
                        # spans): see _merge_spans on why physical boxes are the
                        # trustworthy span signal.
                        try:
                            span_boxes = [c["bbox"] for c in predict_cells(crop)]
                            records = _merge_spans(records, span_boxes, crop=crop)
                        except Exception as exc:
                            extra_warnings.append(
                                f"Surya+Kiri page {page.page_index + 1}: span detection "
                                f"failed for region {m_idx} ({exc}) — table kept without "
                                f"span merging."
                            )
            except Exception as exc:
                # Never drop a table silently: the analyst would see "no table" on
                # a page that visibly has one. Surface it via the warnings channel.
                extra_warnings.append(
                    f"Surya+Kiri page {page.page_index + 1}: table structure prediction "
                    f"failed for region {m_idx} ({exc}) — table omitted; try the Surya "
                    f"engine for this page."
                )
                continue

            if not records:
                extra_warnings.append(
                    f"Surya+Kiri page {page.page_index + 1}: table structure prediction "
                    f"returned no cells for region {m_idx} — table omitted; try the Surya "
                    f"engine for this page."
                )
                continue

            # Build a (row, col) → text grid from the structure cells, plus a
            # parallel confidence grid.
            # Pass 1: compute axis-aligned crops, applying the size guard directly
            # so tiny cells get "" without ever reaching the batched recognizer.
            grid: dict[tuple[int, int], str] = {}
            conf_grid: dict[tuple[int, int], float] = {}
            # Per-cell geometry in PAGE space (crop-relative coords + the table
            # crop's origin). Kept so an analyst's correction can be paired with the
            # exact pixels Kiri read — the HITL capture loop's training pair. Without
            # it _build_table_from_grid would leave every cell's bbox empty.
            bbox_grid: dict[tuple[int, int], list[float]] = {}
            span_map: dict[tuple[int, int], tuple[int, int]] = {}
            pending_keys: list[tuple[int, int]] = []
            pending_crops: list[np.ndarray] = []
            for row_id, col_id, bbox, row_span, col_span in records:
                cx0, cy0, cx1, cy1 = (int(v) for v in bbox)
                cx0, cy0 = max(0, cx0), max(0, cy0)
                cx1, cy1 = min(crop.shape[1], cx1), min(crop.shape[0], cy1)

                key = (row_id, col_id)
                # x0/y0 is the table crop's origin in `img` (the frame the cells were
                # cropped from), so this box indexes the page image directly.
                bbox_grid[key] = [float(cx0 + x0), float(cy0 + y0),
                                  float(cx1 + x0), float(cy1 + y0)]
                if row_span > 1 or col_span > 1:
                    span_map[key] = (row_span, col_span)
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
                key = (tcell["row_id"], tcell["col_id"])
                conf = conf_grid.get(key, 1.0)
                tcell["confidence"] = conf
                # Same key as confidence: a mismatch here would pair one cell's
                # text with another cell's pixels in the HITL capture.
                tcell["bbox"] = bbox_grid.get(key, [])
                if conf < _LOW_CONF_THRESHOLD:
                    page_low_conf_count += 1
                # Span metadata (slanet path only): lets exports/UI know the cell
                # covers multiple grid positions; unit cells stay shape-unchanged.
                if key in span_map:
                    rs, cs = span_map[key]
                    if rs > 1:
                        tcell["row_span"] = rs
                    if cs > 1:
                        tcell["col_span"] = cs
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
