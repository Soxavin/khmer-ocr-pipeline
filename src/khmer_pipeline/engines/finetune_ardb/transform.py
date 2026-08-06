"""Shared region -> SuryaPageResult transform for the ARDB fine-tune engines.

Both models emit one JSON array of {"box_2d": [y1,x1,y2,x2] (0-1000 grid),
"label", "text"} per page (the `_UNIFIED_INSTRUCTION` contract, see
`datagen/build_ardb_unified_sft.py`) — structurally the same shape
`gemini_engine.py` already turns into a SuryaPageResult, differing only in the
bbox encoding (normalized 0-1000, not pixels) and the Table region's text
format (HTML for Gemma, markdown for Qwen). Reuses `surya.py`'s HTML table
parser unchanged; the markdown path is this module's own `markdown_table_to_grid`.
"""
from __future__ import annotations

from typing import Literal

from ...models import SuryaPageResult, Table, TextBlock
from ..surya import _parse_html_table_with_spans, _build_table_from_grid
from .parsing import markdown_table_to_grid

_TABLE_LABEL = "Table"
_PICTURE_LABEL = "Picture"

TableTextFormat = Literal["html", "markdown"]


def _rescale_box_2d(box_2d, page_w: int, page_h: int) -> list[float]:
    """[y1, x1, y2, x2] on a 0-1000 grid -> [x1, y1, x2, y2] in pixels.
    Malformed input degrades to [] (bbox is not load-bearing for scoring — the
    grid/text is — so never crash on a bad box)."""
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return []
    try:
        y1, x1, y2, x2 = (float(v) for v in box_2d)
    except (TypeError, ValueError):
        return []
    return [x1 / 1000 * page_w, y1 / 1000 * page_h, x2 / 1000 * page_w, y2 / 1000 * page_h]


def _table_grid(text: str, table_text_format: TableTextFormat) -> tuple[dict, dict]:
    """Returns (grid, spans) — spans is always empty for markdown (no colspan/
    rowspan concept in a plain pipe table); HTML reuses the production parser,
    spans included."""
    if table_text_format == "html":
        return _parse_html_table_with_spans(text)
    return markdown_table_to_grid(text), {}


def regions_to_page(
    regions: list[dict],
    page_index: int,
    table_text_format: TableTextFormat,
    page_w: int,
    page_h: int,
) -> SuryaPageResult:
    """Split a fine-tune's region list into Tables + TextBlocks, same shape
    `gemini_engine._elements_to_page` produces — so the result flows through
    the existing review UI unchanged."""
    text_blocks: list[TextBlock] = []
    tables: list[Table] = []
    pooled: list[str] = []
    order = 0
    for region in regions:
        if not isinstance(region, dict):
            continue
        label = region.get("label") or ""
        text = region.get("text") or ""
        bbox = _rescale_box_2d(region.get("box_2d"), page_w, page_h)
        if label == _TABLE_LABEL:
            grid, spans = _table_grid(text, table_text_format)
            table = _build_table_from_grid(grid, text, bbox)
            if spans:
                for cell in table["cells"]:
                    span = spans.get((cell["row_id"], cell["col_id"]))
                    if span and span > 1:
                        cell["col_span"] = span
            tables.append(table)
            pooled.extend(
                t["text"] for c in table["cells"] for t in (c.get("text_lines") or [])
                if t.get("text")
            )
        elif label == _PICTURE_LABEL:
            continue  # pictures carry no transcribable text
        elif text:
            text_blocks.append({"text": text, "bbox": bbox, "label": label,
                                "reading_order": order})
            order += 1
            pooled.append(text)
    return SuryaPageResult(page_index=page_index, text_blocks=text_blocks,
                           tables=tables, ocr_text="\n".join(pooled))
