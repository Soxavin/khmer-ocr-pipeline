"""Pure bulk-edit helpers (no NiceGUI imports) — currently find/replace across table cells,
for fixing a systematic OCR error (e.g. a mis-recognized glyph) in one action.
"""
from __future__ import annotations


def replace_in_grid(grid: list[list[str]], find: str, replace: str) -> tuple[list[list[str]], int]:
    """Return (new_grid, occurrences_replaced). No-op (count 0) when `find` is empty."""
    if not find:
        return grid, 0
    count = 0
    new: list[list[str]] = []
    for row in grid:
        new_row = []
        for cell in row:
            n = cell.count(find)
            if n:
                count += n
                cell = cell.replace(find, replace)
            new_row.append(cell)
        new.append(new_row)
    return new, count


def replace_across(tables_list: list[tuple[str, list[list[str]]]], find: str, replace: str
                   ) -> tuple[dict[str, list[list[str]]], int]:
    """Apply find/replace to every (table_id, grid). Returns the changed grids keyed by
    table_id (only those that actually changed) and the total occurrences replaced."""
    changed: dict[str, list[list[str]]] = {}
    total = 0
    for tid, grid in tables_list:
        new, c = replace_in_grid(grid, find, replace)
        if c:
            changed[tid] = new
            total += c
    return changed, total
