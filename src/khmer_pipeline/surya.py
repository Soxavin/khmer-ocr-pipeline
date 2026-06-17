from __future__ import annotations
import html as _html_mod
import time
import warnings
from html.parser import HTMLParser
from typing import Any, Callable, Optional
from PIL import Image
from .models import PreprocessResult, SuryaResult, SuryaPageResult
from .model_config import CONFIDENCE_LOW

_BBOX_MATCH_TOLERANCE = 20.0  # max summed |Δ| across all 4 coords (layout vs OCR pass)


def _log(msg: str) -> None:
    print(f"[Surya] {msg}", flush=True)

_manager = None
_layout_pred = None
_rec_pred = None


def models_loaded() -> bool:
    return _manager is not None


def preload_models() -> None:
    _get_predictors()


def _get_predictors():
    global _manager, _layout_pred, _rec_pred
    if _manager is None:
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
        self._parts.append(data)

    def get_text(self) -> str:
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
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []
            self._current_colspan = 1
            for name, value in attrs:
                if name == "colspan" and value and value.isdigit():
                    self._current_colspan = int(value)

    def handle_endtag(self, tag: str) -> None:
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
        if self._current_cell is not None:
            self._current_cell.append(data)

    def grid(self) -> dict[tuple[int, int], str]:
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
                           region_bbox: list[float]) -> dict[str, Any]:
    # Single source of truth: the VLM's HTML grid. Text is already in its
    # correct cell, so no index-join against a second (geometric) grid.
    if not grid:
        flat = _html_to_text(html)
        cell = {"row_id": 0, "col_id": 0, "cell_id": 0, "bbox": [], "polygon": [],
                "text_lines": [{"text": flat, "bbox": []}] if flat else []}
        return {"rows": [{"row_id": 0}], "cols": [{"col_id": 0}],
                "cells": [cell], "image_bbox": list(region_bbox)}
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    cells = []
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
            warnings.warn(f"Layout failed on page {page_index}; returning empty result.")
            return SuryaPageResult(page_index=page_index, text_blocks=[], tables=[], ocr_text="")

        text_blocks: list[dict] = []
        # Maps rounded table bbox → VLM-generated HTML (contains <table><tr><td> structure).
        # Populated from OCR blocks labelled "Table"; used later to fill cell text.
        table_html_map: dict[tuple, str] = {}

        try:
            _log(f"Page {page_index}: OCR recognition...")
            t0 = time.perf_counter()
            page_ocr = rec_pred([pil_img], layout_results=[layout_result])[0]
            _log(f"Page {page_index}: OCR done in {time.perf_counter()-t0:.1f}s → {len(page_ocr.blocks)} blocks")
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index}: {e}")
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
        tables: list[dict] = []
        for b in table_bboxes:
            table_html = _find_matching_html(b.bbox, table_html_map)
            if not table_html:
                warnings.warn(
                    f"Page {page_index}: no OCR HTML for table {len(tables)}; cells will be empty."
                )
                tbl = {"rows": [], "cols": [], "cells": [], "image_bbox": list(b.bbox)}
            else:
                grid = _parse_html_table(table_html)
                if not grid:
                    warnings.warn(
                        f"Page {page_index}: VLM produced no <table> structure for table "
                        f"{len(tables)}; using flat text in first cell."
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
                f"Page {page_index}: {low_conf_blocks} text block(s) have low OCR "
                f"confidence (<{CONFIDENCE_LOW})."
            )

        return SuryaPageResult(
            page_index=page_index,
            text_blocks=text_blocks,
            tables=tables,
            ocr_text=ocr_text,
        )

    except Exception as e:
        warnings.warn(f"Critical failure processing page {page_index}: {e}")
        return SuryaPageResult(
            page_index=page_index,
            text_blocks=[],
            tables=[],
            ocr_text="",
        )
