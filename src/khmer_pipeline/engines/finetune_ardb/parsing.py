"""Shared parsing for the ARDB fine-tune engines (Gemma, Qwen): both models were
fine-tuned on the same unified instruction/output contract (see
`datagen/build_ardb_unified_sft.py`'s `_UNIFIED_INSTRUCTION`) — one JSON array of
`{"box_2d": [y1, x1, y2, x2] (0-1000 grid), "label": ..., "text": ...}` per page
region. What differs between the two models is only the Table region's `text`
format (Gemma: HTML, matching `surya.py`'s existing parser; Qwen: markdown pipe
tables), so only the markdown grid parser is new here — JSON repair is shared.
"""
from __future__ import annotations

import json
import re


def _try_repair_json(text: str) -> str | None:
    """Recovers two failure modes confirmed in real Colab generations (see
    eval/qwen_finetune_runs.md, mirrored from scripts/eval_finetune_real.py so
    parse-failure counts stay comparable between eval scoring and this live
    engine): missing only the outer list's closing ']' (a stray end-of-turn
    token follows), and missing BOTH '[' and ']' (a bare comma-separated
    sequence of '{...}' objects with no wrapping list at all). Finds the last
    complete object (last '}'), discards anything after it, then adds whichever
    bracket is missing. Text cut off mid-field (no trailing '}' at all) is left
    alone rather than guessing at missing content."""
    stripped = text.strip()
    last_brace = stripped.rfind("}")
    if last_brace == -1:
        return None
    core = stripped[: last_brace + 1]
    if not core.startswith("["):
        core = "[" + core
    if not core.endswith("]"):
        core = core + "]"
    if core == stripped:
        return None
    return core


def parse_regions(text: str) -> list[dict] | None:
    """Decode a fine-tune's raw output into its region list, repairing common
    truncation failures first. Returns None when nothing recoverable parses —
    callers fail soft (empty page + warning), never crash the run."""
    try:
        regions = json.loads(text)
        return regions if isinstance(regions, list) else None
    except json.JSONDecodeError:
        pass
    repaired = _try_repair_json(text)
    if repaired is None:
        return None
    try:
        regions = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return regions if isinstance(regions, list) else None


# Matches one markdown table row: leading/trailing pipes optional, cells split on '|'.
_ROW_RE = re.compile(r"^\s*\|?(.+?)\|?\s*$")
# A separator row (|---|---|, with optional alignment colons) — never real data.
_SEP_CELL_RE = re.compile(r"^:?-+:?$")


def markdown_table_to_grid(text: str) -> dict[tuple[int, int], str]:
    """Parse a markdown pipe table into the same `dict[(row, col)] -> text` grid
    shape `surya._build_table_from_grid` already consumes for the HTML path —
    so both table-text formats reach the identical `Table` builder. Ragged rows
    (fewer cells than the widest row) are padded with "" rather than dropped or
    raising; a non-table input returns an empty grid."""
    grid: dict[tuple[int, int], str] = {}
    out_row = 0
    max_cols = 0
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m or "|" not in line:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if all(_SEP_CELL_RE.match(c) for c in cells):
            continue  # the |---|---| separator row: structure only, no data
        for col, cell in enumerate(cells):
            grid[(out_row, col)] = cell
        max_cols = max(max_cols, len(cells))
        out_row += 1
    # Pad ragged rows so every row has the same column count (missing = "").
    for row in range(out_row):
        for col in range(max_cols):
            grid.setdefault((row, col), "")
    return grid
