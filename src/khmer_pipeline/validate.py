"""Post-OCR failure-mode classification for extracted table cells.

Inspects the cells of each corrected page and attaches machine-readable `flags`
(the taxonomy below) to any cell that fails a check. Purely annotative: it
mutates cells in place and returns per-page summary warnings for the caller's
warnings channel — it never blocks export or rewrites cell text.

Taxonomy (stable — deliberate report material):
  low_conf            per-cell confidence below the recognizer's low bucket
  sequence_illegal    Khmer Unicode sequence violation (OCR failure signature)
  digit_mixed         Khmer and ASCII digits mixed in one cell
  numeric_unparseable non-empty cell in a numeric column that isn't a number
  numeric_mismatch    total-row value != the column sum of the body rows it covers
  structure_ragged    cell's row length differs from the table's majority
"""
from __future__ import annotations

import re
from collections import Counter

from .models import CorrectedPageResult
# Private char-class helpers from the normalizer (same package): base consonant /
# combining mark / subscript-coeng classification, reused so sequence checks and
# the normalizer agree on what a Khmer base/mark is.
from .utils.khmer_normalize import _COENG, _is_base, _is_mark

# Fraction of a column's non-empty body cells that must parse as numbers for the
# column to count as numeric.
_NUMERIC_COL_THRESHOLD = 0.70
# Rounding slack when comparing a total-row value to the summed body rows.
_NUMERIC_MISMATCH_TOLERANCE = 0.5
# Keep in sync with engines/surya_kiri_engine._LOW_CONF_THRESHOLD — duplicated
# here (rather than imported) to avoid pulling in that engine's heavy import chain.
_LOW_CONF_FLAG_THRESHOLD = 0.80
# Tables with fewer rows than this are too small to have a meaningful "majority"
# row length, so the ragged check is skipped for them.
_MIN_ROWS_FOR_RAGGED = 3

# Khmer digits U+17E0–U+17E9.
_KHMER_DIGIT_LO, _KHMER_DIGIT_HI = 0x17E0, 0x17E9

# Total/subtotal row label strings, harvested programmatically from the
# user-verified ground-truth JSONs in eval/datasets/real/*_ground_truth.json.
# The verified GT tables carry NO first-column total-row labels (the budget
# table uses hierarchical numbering like "I. …"/"១. …" for its subtotals and the
# only "សរុប" token is a *column header*, not a row label), so there is nothing
# usable to key on. Left empty by design → numeric_mismatch stays dormant rather
# than firing on guessed keywords. See the task report / final notes.
_TOTAL_ROW_LABELS: tuple[str, ...] = ()

_DECIMAL_RE = re.compile(r"\d+(\.\d+)?")

_VERIFY_HINT = {
    "low_conf": "review low-confidence text",
    "sequence_illegal": "check Khmer spelling",
    "digit_mixed": "check mixed digits",
    "numeric_unparseable": "check number formatting",
    "numeric_mismatch": "verify totals",
    "structure_ragged": "check row structure",
}


def validate_pages(pages: list[CorrectedPageResult]) -> list[str]:
    """Classify failure modes across every table cell of `pages`, attaching a
    `flags` list to each cell that fails a check (mutated in place).

    Returns human-readable per-page summary warnings (one per flag type seen on
    a page) for the caller to append to its warnings channel."""
    warnings: list[str] = []
    for page in pages:
        page_counts: Counter[str] = Counter()
        for table in page.tables:
            for flag in _validate_table(table):
                page_counts[flag] += 1
        for flag in _VERIFY_HINT:  # stable, taxonomy order
            n = page_counts.get(flag, 0)
            if n:
                warnings.append(
                    f"page {page.page_index + 1}: {n} cell(s) flagged {flag} "
                    f"— {_VERIFY_HINT[flag]}"
                )
    return warnings


def _cell_text(cell: dict) -> str:
    """Join a cell's text_lines the same way export._json_cell does."""
    return " ".join(
        t["text"] for t in (cell.get("text_lines") or []) if t.get("text")
    ).strip()


def _add_flag(cell: dict, flag: str) -> None:
    """Append `flag` to the cell's flags list (creating it on first use), never
    duplicating a flag on one cell."""
    flags = cell.setdefault("flags", [])
    if flag not in flags:
        flags.append(flag)


def _has_khmer_digit(text: str) -> bool:
    return any(_KHMER_DIGIT_LO <= ord(c) <= _KHMER_DIGIT_HI for c in text)


def _has_ascii_digit(text: str) -> bool:
    return any(c.isascii() and c.isdigit() for c in text)


def _parse_number(text: str) -> tuple[float, bool] | None:
    """Parse a cell's text as a number, returning (value, is_percent) or None.

    Strips whitespace, converts Khmer digits to Arabic, removes thousands
    separators (`,`), treats `(123)` and a leading `-` as negative, allows one
    decimal point and a trailing `%`."""
    # Imported lazily to avoid a circular import (export imports validate_pages).
    from .export import _KHMER_TO_ARABIC
    t = "".join(text.split())
    if not t:
        return None
    t = "".join(_KHMER_TO_ARABIC.get(c, c) for c in t)
    is_percent = t.endswith("%")
    if is_percent:
        t = t[:-1]
    negative = False
    if t.startswith("(") and t.endswith(")"):
        negative = True
        t = t[1:-1]
    if t.startswith("-"):
        negative = True
        t = t[1:]
    t = t.replace(",", "")
    if not _DECIMAL_RE.fullmatch(t):
        return None
    value = float(t)
    return (-value if negative else value, is_percent)


def _sequence_illegal(text: str) -> bool:
    """Conservative Khmer Unicode sequence check — flags OCR failure signatures:
    (a) coeng at end / followed by a non-base, (b) two identical consecutive
    marks, (c) a mark with no base to attach to in its cluster."""
    n = len(text)
    for i, ch in enumerate(text):
        if ch == _COENG:
            if i == n - 1 or not _is_base(text[i + 1]):  # (a)
                return True
        elif _is_mark(ch):
            if i == 0:  # (c) mark with no preceding base at all
                return True
            prev = text[i - 1]
            if ch == prev:  # (b) identical consecutive marks
                return True
            if not (_is_base(prev) or _is_mark(prev) or prev == _COENG):  # (c)
                return True
    return False


def _rows_by_id(cells: list[dict]) -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = {}
    for c in cells:
        rows.setdefault(c.get("row_id", 0), []).append(c)
    return rows


def _row_label(row_cells: list[dict]) -> str:
    """First non-empty cell text of a row (used to spot total/subtotal rows)."""
    for c in sorted(row_cells, key=lambda c: c.get("col_id") or 0):
        text = _cell_text(c)
        if text:
            return text
    return ""


def _is_total_label(label: str) -> bool:
    return any(kw in label for kw in _TOTAL_ROW_LABELS)


def _validate_table(table: dict) -> list[str]:
    """Run every check over one table, mutating flagged cells in place and
    returning the flat list of flags applied (for summary counting)."""
    cells = table.get("cells", [])
    if not cells:
        return []
    applied: list[str] = []

    def flag(cell: dict, name: str) -> None:
        _add_flag(cell, name)
        applied.append(name)

    # --- per-cell content checks (all rows) ---
    for cell in cells:
        text = _cell_text(cell)
        # Blank cells are normally intentional structure, not OCR errors — skip
        # them entirely (low_conf is only for non-empty text, per the taxonomy).
        if not text:
            continue
        conf = cell.get("confidence")
        if conf is not None and conf < _LOW_CONF_FLAG_THRESHOLD:
            flag(cell, "low_conf")
        if _sequence_illegal(text):
            flag(cell, "sequence_illegal")
        if _has_khmer_digit(text) and _has_ascii_digit(text):
            flag(cell, "digit_mixed")

    rows = _rows_by_id(cells)
    row_ids = sorted(rows)

    # --- structure_ragged ---
    if len(row_ids) >= _MIN_ROWS_FOR_RAGGED:
        lengths = [len(rows[r]) for r in row_ids]
        majority = Counter(lengths).most_common(1)[0][0]
        for r in row_ids:
            if len(rows[r]) != majority:
                for cell in rows[r]:
                    flag(cell, "structure_ragged")

    # --- numeric checks ---
    # Header = first row; total rows = keyword-matched labels; body = the rest.
    header_id = row_ids[0]
    total_ids = {r for r in row_ids if _is_total_label(_row_label(rows[r]))}
    body_ids = [r for r in row_ids if r != header_id and r not in total_ids]

    col_ids = sorted({c.get("col_id") or 0 for c in cells})
    cell_at = {(c.get("row_id", 0), c.get("col_id") or 0): c for c in cells}

    for col in col_ids:
        body_values = []
        parseable = 0
        for r in body_ids:
            c = cell_at.get((r, col))
            if c is None:
                continue
            text = _cell_text(c)
            if not text:
                continue
            parsed = _parse_number(text)
            body_values.append((r, c, parsed))
            if parsed is not None:
                parseable += 1
        nonempty = len(body_values)
        if nonempty == 0 or parseable / nonempty < _NUMERIC_COL_THRESHOLD:
            continue  # not a numeric column

        # numeric_unparseable: non-empty body cells that failed to parse.
        for _r, c, parsed in body_values:
            if parsed is None:
                flag(c, "numeric_unparseable")

        # numeric_mismatch: each total row vs the sum of the contiguous body rows
        # above it (since the previous total row). Dormant while _TOTAL_ROW_LABELS
        # is empty, but fully implemented.
        span_sum = 0.0
        span_clean = True
        span_has_body = False
        for r in row_ids:
            if r == header_id:
                continue
            if r in total_ids:
                total_cell = cell_at.get((r, col))
                if total_cell is not None and span_has_body and span_clean:
                    total_parsed = _parse_number(_cell_text(total_cell))
                    if (total_parsed is not None
                            and abs(total_parsed[0] - span_sum) > _NUMERIC_MISMATCH_TOLERANCE):
                        flag(total_cell, "numeric_mismatch")
                span_sum, span_clean, span_has_body = 0.0, True, False
                continue
            c = cell_at.get((r, col))
            if c is None:
                continue
            text = _cell_text(c)
            if not text:
                continue
            span_has_body = True
            parsed = _parse_number(text)
            if parsed is None:
                span_clean = False
            elif not parsed[1]:  # percent cells are excluded from sums
                span_sum += parsed[0]

    return applied
