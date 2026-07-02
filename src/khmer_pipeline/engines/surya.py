from __future__ import annotations
import html as _html_mod
import os
import time
import warnings
from html.parser import HTMLParser
from typing import Callable, Optional
from PIL import Image
from ..models import PreprocessResult, SuryaResult, SuryaPageResult, Cell, Table, TextBlock
from ..utils.device import configure_runtime
from ..model_config import CONFIDENCE_LOW
from .table_stitch import merge_table_regions, merge_table_rowbands

_BBOX_MATCH_TOLERANCE = 20.0  # max summed |Δ| across all 4 coords (layout vs OCR pass)

# Merge fragmented Table layout regions before recognition (see table_stitch.py).
# DEFAULT OFF: the benchmark showed merging all fragments into one master box fixes
# detection (page 2: 8 regions → 1) but the VLM then degrades badly on the large
# dense crop (Content_Recall 0.76 → 0.16) → net regression. Kept behind the flag for
# experiments (e.g. a future row-band variant). Set KHMER_STITCH_TABLES=1 to enable.
_STITCH_TABLES = False


def _stitch_enabled() -> bool:
    env = os.environ.get("KHMER_STITCH_TABLES")
    return env != "0" if env is not None else _STITCH_TABLES


def _stitch_mode() -> str:
    # "rowband" = full-width row strips (default); "master" = one box (regressed, see 2.12)
    return os.environ.get("KHMER_STITCH_MODE", "rowband")


def _log(msg: str) -> None:
    print(f"[Surya] {msg}", flush=True)

_manager = None
_layout_pred = None
_rec_pred = None


def models_loaded() -> bool:
    """Return True if the Surya layout/recognition predictors are already initialized."""
    return _manager is not None


def preload_models() -> None:
    """Eagerly initialize the Surya inference manager and predictors, if not already loaded."""
    _get_predictors()


def _get_predictors():
    global _manager, _layout_pred, _rec_pred
    if _manager is None:
        configure_runtime()
        _log("Initializing SuryaInferenceManager...")
        t0 = time.perf_counter()
        from surya.inference import SuryaInferenceManager
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        _manager = SuryaInferenceManager()
        _layout_pred = LayoutPredictor(manager=_manager)
        _rec_pred = RecognitionPredictor(manager=_manager)
        _log(f"Manager ready in {time.perf_counter()-t0:.1f}s")
    return _layout_pred, _rec_pred


def run_surya(
    result: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> SuryaResult:
    """Run Surya layout detection, OCR, and table recognition over every page image
    in `result`. `on_page(idx, total)` is called before each page if given. Returns
    a `SuryaResult` with per-page text blocks/tables and any warnings raised during
    processing."""
    layout_pred, rec_pred = _get_predictors()
    pil_images = [Image.fromarray(img) for img in result.page_images]
    total = len(pil_images)
    pages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for idx, pil_img in enumerate(pil_images):
            if on_page is not None:
                on_page(idx, total)
            pages.append(_process_page(idx, pil_img, layout_pred, rec_pred))
        collected_warnings = [str(w.message) for w in caught]
    return SuryaResult(source_name=result.source_name, pages=pages, warnings=collected_warnings)


class _TagStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        """Collect a chunk of text data encountered while parsing HTML."""
        self._parts.append(data)

    def get_text(self) -> str:
        """Return all collected text data joined into a single whitespace-normalized string."""
        return " ".join(p.strip() for p in self._parts if p.strip())


def _html_to_text(html_str: str) -> str:
    if not html_str:
        return ""
    stripper = _TagStripper()
    stripper.feed(_html_mod.unescape(html_str))
    return stripper.get_text()


class _TableHTMLParser(HTMLParser):
    # Parse <table><tr><td/th> HTML into a row/col grid.

    def __init__(self):
        super().__init__()
        self._rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._current_colspan = 1

    def handle_starttag(self, tag: str, attrs) -> None:
        """Start tracking a new row (`tr`) or cell (`td`/`th`), reading `colspan` if present."""
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []
            self._current_colspan = 1
            for name, value in attrs:
                if name == "colspan" and value and value.isdigit():
                    self._current_colspan = int(value)

    def handle_endtag(self, tag: str) -> None:
        """Close the current cell or row, appending padding cells for any `colspan`."""
        if tag in ("td", "th") and self._current_cell is not None:
            text = " ".join(self._current_cell).strip()
            if self._current_row is not None:
                self._current_row.append(text)
                # Pad spanned columns so col_id indices stay aligned.
                for _ in range(self._current_colspan - 1):
                    self._current_row.append("")
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self._rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        """Append text data to the currently open table cell, if any."""
        if self._current_cell is not None:
            self._current_cell.append(data)

    def grid(self) -> dict[tuple[int, int], str]:
        """Return the parsed table as a `(row, col) -> cell text` mapping."""
        return {
            (r, c): text
            for r, row in enumerate(self._rows)
            for c, text in enumerate(row)
        }


def _parse_html_table(html: str) -> dict[tuple[int, int], str]:
    parser = _TableHTMLParser()
    parser.feed(_html_mod.unescape(html))
    return parser.grid()


def _find_matching_html(layout_bbox: list[float], table_html_map: dict[tuple, str]) -> str:
    # Layout and recognition are separate passes; bboxes for the same region differ
    # slightly. Match by closest bbox within tolerance rather than exact key.
    if not table_html_map:
        return ""
    lx0, ly0, lx1, ly1 = layout_bbox
    best_html = ""
    min_diff = float("inf")
    for key, html in table_html_map.items():
        kx0, ky0, kx1, ky1 = key
        diff = abs(lx0 - kx0) + abs(ly0 - ky0) + abs(lx1 - kx1) + abs(ly1 - ky1)
        if diff < min_diff:
            min_diff = diff
            best_html = html
    return best_html if min_diff < _BBOX_MATCH_TOLERANCE else ""


def _build_table_from_grid(grid: dict[tuple[int, int], str], html: str,
                           region_bbox: list[float]) -> Table:
    # Single source of truth: the VLM's HTML grid. Text is already in its
    # correct cell, so no index-join against a second (geometric) grid.
    if not grid:
        flat = _html_to_text(html)
        cell: Cell = {"row_id": 0, "col_id": 0, "cell_id": 0, "bbox": [], "polygon": [],
                      "text_lines": [{"text": flat, "bbox": []}] if flat else []}
        return {"rows": [{"row_id": 0}], "cols": [{"col_id": 0}],
                "cells": [cell], "image_bbox": list(region_bbox)}
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    cells: list[Cell] = []
    for cid, ((r, c), text) in enumerate(sorted(grid.items())):
        cells.append({"row_id": r, "col_id": c, "cell_id": cid,
                      "bbox": [], "polygon": [],
                      "text_lines": [{"text": text, "bbox": []}] if text else []})
    return {"rows": [{"row_id": i} for i in range(n_rows)],
            "cols": [{"col_id": j} for j in range(n_cols)],
            "cells": cells, "image_bbox": list(region_bbox)}


def _process_page(
    page_index: int,
    pil_img: Image.Image,
    layout_pred,
    rec_pred,
) -> SuryaPageResult:
    try:
        _log(f"Page {page_index}: layout detection...")
        t0 = time.perf_counter()
        layout_result = layout_pred([pil_img])[0]
        _log(f"Page {page_index}: layout done in {time.perf_counter()-t0:.1f}s → {len(layout_result.bboxes)} regions")

        if layout_result.error:
            warnings.warn(f"Layout failed on page {page_index + 1}; returning empty result.")
            return SuryaPageResult(page_index=page_index, text_blocks=[], tables=[], ocr_text="")

        # De-fragment tables: merge adjacent Table regions into master boxes so
        # recognition OCRs each whole table at once (not column-wise fragments).
        if _stitch_enabled():
            table_lboxes = [b for b in layout_result.bboxes if b.label == "Table"]
            if len(table_lboxes) > 1:
                others = [b for b in layout_result.bboxes if b.label != "Table"]
                _boxes = [tuple(float(v) for v in b.bbox) for b in table_lboxes]
                merged = (merge_table_regions(_boxes) if _stitch_mode() == "master"
                          else merge_table_rowbands(_boxes))
                if len(merged) < len(table_lboxes):
                    template = table_lboxes[0]
                    base_pos = min((getattr(b, "position", 0) or 0) for b in table_lboxes)
                    new_tables = []
                    for k, (x0, y0, x1, y1) in enumerate(merged):
                        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                        new_tables.append(template.model_copy(update={
                            "bbox": [x0, y0, x1, y1], "polygon": poly, "position": base_pos + k,
                        }))
                    layout_result.bboxes = others + new_tables
                    _log(f"Page {page_index}: stitched {len(table_lboxes)} table region(s) → {len(new_tables)}")

        text_blocks: list[TextBlock] = []
        # Maps rounded table bbox → VLM-generated HTML (contains <table><tr><td> structure).
        # Populated from OCR blocks labelled "Table"; used later to fill cell text.
        table_html_map: dict[tuple, str] = {}

        try:
            _log(f"Page {page_index}: OCR recognition...")
            t0 = time.perf_counter()
            page_ocr = rec_pred([pil_img], layout_results=[layout_result])[0]
            _log(f"Page {page_index}: OCR done in {time.perf_counter()-t0:.1f}s → {len(page_ocr.blocks)} blocks")
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index + 1}: {e}")
            page_ocr = None

        if page_ocr is not None:
            for block in page_ocr.blocks:
                if block.skipped or block.error:
                    continue

                if block.label == "Table":
                    # Store the VLM's HTML output keyed by rounded bbox.
                    # Do NOT include in text_blocks — table content belongs in cells.
                    if block.html:
                        key = tuple(round(v) for v in block.bbox)
                        table_html_map[key] = block.html
                    continue

                text = _html_to_text(block.html)
                if not text:
                    continue
                text_blocks.append({
                    "text": text,
                    "bbox": list(block.bbox),
                    "polygon": block.polygon,
                    "confidence": block.confidence or 0.0,
                    "label": block.label,
                    "region_label": block.label,
                    "reading_order": block.reading_order,
                })

        text_blocks.sort(key=lambda b: b.get("reading_order", 0))
        ocr_text = "\n\n".join(b["text"] for b in text_blocks if b.get("text"))

        table_bboxes = [b for b in layout_result.bboxes if b.label == "Table"]
        tables: list[Table] = []
        for b in table_bboxes:
            table_html = _find_matching_html(b.bbox, table_html_map)
            if not table_html:
                warnings.warn(
                    f"Page {page_index + 1}: no OCR HTML for table {len(tables) + 1}; cells will be empty."
                )
                tbl: Table = {"rows": [], "cols": [], "cells": [], "image_bbox": list(b.bbox)}
            else:
                grid = _parse_html_table(table_html)
                if not grid:
                    warnings.warn(
                        f"Page {page_index + 1}: VLM produced no <table> structure for table "
                        f"{len(tables) + 1}; using flat text in first cell."
                    )
                tbl = _build_table_from_grid(grid, table_html, b.bbox)
                _log(f"Page {page_index}: built table with {len(tbl['cells'])} cells from HTML")
            tbl["bbox"] = list(b.bbox)
            tables.append(tbl)

        low_conf_blocks = sum(
            1 for b in text_blocks if (b.get("confidence") or 0.0) < CONFIDENCE_LOW
        )
        if low_conf_blocks:
            warnings.warn(
                f"Page {page_index + 1}: {low_conf_blocks} text block(s) have low OCR "
                f"confidence (<{CONFIDENCE_LOW})."
            )

        return SuryaPageResult(
            page_index=page_index,
            text_blocks=text_blocks,
            tables=tables,
            ocr_text=ocr_text,
        )

    except Exception as e:
        warnings.warn(f"Critical failure processing page {page_index + 1}: {e}")
        return SuryaPageResult(
            page_index=page_index,
            text_blocks=[],
            tables=[],
            ocr_text="",
        )
