"""Shared visual helpers: one unified confidence palette (used by BOTH the image overlay
and the cell tinting — closing the old app.py inconsistency where the overlay and the cell
view used different colors), the SVG overlay builder for `ui.interactive_image`, and a
read-only tinted confidence table.
"""
from __future__ import annotations

import html
from typing import Any

from khmer_pipeline.model_config import (
    CONFIDENCE_LOW, CONFIDENCE_MID, CELL_CONF_LOW, CELL_CONF_MID,
)

# One palette everywhere. High = green, medium = amber, low = red.
CONF_HIGH = "#16a34a"
CONF_MID = "#f59e0b"
CONF_LOW = "#dc2626"

# Region-type overlay colors (ported from app.py `_LABEL_COLORS`).
LABEL_COLORS = {
    "Text": "#4A90D9",
    "Table": "#E74C3C",
    "TableOfContents": "#E67E22",
    "Picture": "#27AE60",
    "Figure": "#27AE60",
    "Caption": "#8E44AD",
}
_DEFAULT_LABEL_COLOR = "#95A5A6"


def conf_color(v: float | None, low: float, mid: float) -> str | None:
    """Bucket a confidence into the unified palette; None → no color (no data)."""
    if v is None:
        return None
    if v < low:
        return CONF_LOW
    if v < mid:
        return CONF_MID
    return CONF_HIGH


def _rect(bbox: list[float], stroke: str, sid: str = "") -> str:
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    ident = f' data-id="{sid}"' if sid else ""
    return (
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
        f'fill="none" stroke="{stroke}" stroke-width="2"{ident} />'
    )


def overlay_svg(text_blocks: list[dict], table_blocks: list[dict], mode: str) -> str:
    """Inner SVG for `ui.interactive_image` (image-pixel coordinates). `mode` is
    'Region type' (color by layout label) or 'Confidence' (color by OCR confidence)."""
    parts: list[str] = []
    if mode == "Confidence":
        for b in text_blocks:
            bbox = b.get("bbox")
            if bbox and len(bbox) >= 4:
                color = conf_color(b.get("confidence") or 0.0, CONFIDENCE_LOW, CONFIDENCE_MID)
                parts.append(_rect(bbox, color or _DEFAULT_LABEL_COLOR))
    else:
        for b in text_blocks:
            bbox = b.get("bbox")
            if bbox and len(bbox) >= 4:
                parts.append(_rect(bbox, LABEL_COLORS.get(b.get("label", ""), _DEFAULT_LABEL_COLOR)))
        for t in table_blocks:
            bbox = t.get("bbox")
            if bbox and len(bbox) >= 4:
                parts.append(_rect(bbox, LABEL_COLORS["Table"]))
    return "".join(parts)


def highlight_rect(bbox: list[float]) -> str:
    """A bright rect marking the currently linked cell on `ui.interactive_image`."""
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    return (
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
        f'fill="#2563eb" fill-opacity="0.18" stroke="#2563eb" stroke-width="4" />'
    )


def table_bbox_index(surya_tables: list[dict], page_blocks: list[dict]) -> dict[str, list[float]]:
    """Map table_id → its region bbox in page-image pixels, aligning Surya's raw tables
    with the export blocks (which carry table_id). Table-level, because the pipeline does
    not expose per-cell geometry (Surya leaves cell bbox/polygon empty and the hybrid
    engine discards SLANet cell boxes in `_build_table`). Only valid unstitched, where
    page tables map 1:1 to this page's Surya tables."""
    index: dict[str, list[float]] = {}
    for t_idx, t in enumerate(surya_tables):
        if t_idx >= len(page_blocks):
            break
        bbox = t.get("bbox")
        if bbox and len(bbox) >= 4:
            index[page_blocks[t_idx]["table_id"]] = bbox
    return index


def conf_view_html(grid: list[list[str]], conf_grid: list[list[Any]]) -> str:
    """Read-only HTML table tinting each cell by the unified confidence palette. Ported
    from app.py's `_conf_css` styled dataframe."""
    rows = []
    for r, row in enumerate(grid):
        cells = []
        for c, text in enumerate(row):
            v = conf_grid[r][c] if r < len(conf_grid) and c < len(conf_grid[r]) else None
            if v is not None and v < CELL_CONF_LOW:
                bg = "background:#fde2e2;"
            elif v is not None and v < CELL_CONF_MID:
                bg = "background:#fff3cd;"
            else:
                bg = ""
            cells.append(f'<td style="border:1px solid #ccc;padding:2px 6px;{bg}">{html.escape(text)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    body = "".join(rows)
    return f'<table style="border-collapse:collapse;font-size:0.85rem">{body}</table>'
