"""Build a DRAFT document-level ground truth for the 15.06.26 market-price
bulletin by cross-referencing the VERIFIED 09.06.26 document GT.

Both PDFs are the SAME daily template (one continuous 9-column table across
3 pages, same items/order, only prices/dates differ) and BOTH have a broken
ToUnicode CMap so extracted Khmer text is garbled -- but garbled text is
STABLE across the two docs (same item -> same garbled string), so it is used
here only for row-alignment, never as the final GT text. Digits/prices/
percentages/dates extract correctly in both.

Strategy:
  1. Extract the 9-column logical table from both PDFs via fitz
     page.find_tables() on all 3 pages. Each physical row's non-empty cells,
     read left to right, collapse directly onto the 9 logical columns (see
     _collapse_row) -- this is robust to the differing physical column counts
     between page 1 (33 cols, 2-row header) and pages 2-3 (18 cols).
  2. Build an ordered alignment key per data row: (section_index, rownum,
     garbled_name). Assert 09's key sequence == 15's key sequence.
  3. Clone 09's verified document_gt.json; for each aligned data row, copy
     09's col0/col1/col2 (No./Khmer name/unit) verbatim and 15's cols 3-8
     (the 6 numeric cells) from 15's extraction. Update the two date-header
     strings. Track blank/value mismatches into needs_review_rows.

Usage:
    uv run python scripts/build_15_gt.py

Writes eval/datasets/real/<15-stem>_document_gt.json. DRAFT for human review
-- do not treat as verified without eyeballing against the rendered PNGs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

_REAL_DIR = Path("eval/datasets/real")
_SAMPLE_DIR = Path("sample_data")
_STEM_09 = "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-09.06.26"
_STEM_15 = "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-15.06.26"
_N_COLS = 9

_KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
_ROWNUM_RE = re.compile(rf"^[{_KHMER_DIGITS}]+$")


def _is_rownum(s: str) -> bool:
    return bool(s) and bool(_ROWNUM_RE.match(s))


def _is_section_divider(s: str) -> bool:
    return "/-" in s


def _is_header(s: str) -> bool:
    return "ល.រ" in s


def _collapse_row(raw_row: list) -> list[str]:
    """Non-empty physical cells, left to right, stripped."""
    out = []
    for v in raw_row:
        if v is None:
            continue
        v = v.strip()
        if v:
            out.append(v)
    return out


def _classify_and_expand(cells: list[str]) -> tuple[str, list[str] | str]:
    """Classify a collapsed row and map onto the 9 logical columns where applicable.

    Returns (kind, payload):
      kind == "header"  -> payload is the raw cells (ignored downstream)
      kind == "section" -> payload is the divider text
      kind == "data"     -> payload is a 9-element list [no, name, unit, c3..c8]
                             with numeric cells right-padded into their slot
                             (blank wholesale rows only have 6 cells).
      kind == "note"     -> payload is the raw text (single-cell trailing notes)
      kind == "unknown"  -> payload is raw cells
    """
    if not cells:
        return "unknown", cells
    if _is_header(cells[0]):
        return "header", cells
    if len(cells) == 1:
        if _is_section_divider(cells[0]):
            return "section", cells[0]
        return "note", cells[0]
    if _is_rownum(cells[0]):
        no, name, unit = cells[0], cells[1], cells[2]
        nums = cells[3:]
        row = [no, name, unit, "", "", "", "", "", ""]
        if len(nums) == 6:
            row[3:9] = nums
        elif len(nums) == 3:
            # blank-wholesale row: only retail_prev, retail_cur, pct_retail present
            row[4], row[6], row[8] = nums
        else:
            raise ValueError(f"Unexpected numeric cell count {len(nums)} in row {cells}")
        return "data", row
    return "unknown", cells


def _extract_page_rows(page) -> list[tuple]:
    tabs = page.find_tables()
    rows: list[tuple] = []
    for t in sorted(tabs.tables, key=lambda tb: tb.bbox[1]):
        for raw_row in t.extract():
            cells = _collapse_row(raw_row)
            kind, payload = _classify_and_expand(cells)
            if kind == "unknown":
                # header continuation row (sub-column labels) or blank -- skip
                continue
            rows.append((kind, payload))
    return rows


def extract_document_rows(pdf_path: Path) -> list[tuple]:
    """Returns an ordered list of (kind, payload) tuples across all pages,
    concatenating multiple physical tables on a page top-to-bottom, and
    concatenating pages in order."""
    doc = fitz.open(str(pdf_path))
    all_rows: list[tuple] = []
    with doc:
        for page in doc:
            all_rows.extend(_extract_page_rows(page))
    return all_rows


def extract_rows_per_page(pdf_path: Path) -> list[list[tuple]]:
    """Same as extract_document_rows but keeps pages separate."""
    doc = fitz.open(str(pdf_path))
    with doc:
        return [_extract_page_rows(page) for page in doc]


def alignment_key(rows: list[tuple]) -> list[tuple]:
    """Ordered (section_index, rownum, garbled_name) for data rows only."""
    key = []
    section_idx = -1
    for kind, payload in rows:
        if kind == "section":
            section_idx += 1
        elif kind == "data":
            key.append((section_idx, payload[0], payload[1]))
    return key


def extract_dates(pdf_path: Path) -> tuple[str, str]:
    """Pull the two dd-mm-yy date strings from page 1 text (digits extract
    correctly even though Khmer letters are garbled)."""
    doc = fitz.open(str(pdf_path))
    with doc:
        text = doc[0].get_text()
    dates = re.findall(r"\b\d{2}-\d{2}-\d{2}\b", text)
    seen = []
    for d in dates:
        if d not in seen:
            seen.append(d)
    if len(seen) < 2:
        raise ValueError(f"Expected 2 distinct dates in {pdf_path.name}, found {seen}")
    return seen[0], seen[1]


def main() -> int:
    pdf_09 = _SAMPLE_DIR / f"{_STEM_09}.pdf"
    pdf_15 = _SAMPLE_DIR / f"{_STEM_15}.pdf"
    gt_09_path = _REAL_DIR / f"{_STEM_09}_document_gt.json"

    print("=== Extracting tables ===")
    rows_09 = extract_document_rows(pdf_09)
    rows_15 = extract_document_rows(pdf_15)
    n_data_09 = sum(1 for k, _ in rows_09 if k == "data")
    n_data_15 = sum(1 for k, _ in rows_15 if k == "data")
    print(f"09.06.26: {len(rows_09)} classified rows ({n_data_09} data rows)")
    print(f"15.06.26: {len(rows_15)} classified rows ({n_data_15} data rows)")

    print("\n=== Alignment check ===")
    key_09 = alignment_key(rows_09)
    key_15 = alignment_key(rows_15)
    aligned = key_09 == key_15
    print(f"09 data-row count: {len(key_09)}  15 data-row count: {len(key_15)}")
    if aligned:
        print("PASS -- (section_index, rownum, garbled_name) sequences are IDENTICAL.")
    else:
        print("FAIL -- sequences diverge. Diffing...")
        max_len = max(len(key_09), len(key_15))
        first_diff = None
        for i in range(max_len):
            a = key_09[i] if i < len(key_09) else None
            b = key_15[i] if i < len(key_15) else None
            if a != b:
                first_diff = i
                break
        print(f"  first divergence at index {first_diff}")
        lo = max(0, (first_diff or 0) - 2)
        hi = min(max_len, (first_diff or 0) + 5)
        for i in range(lo, hi):
            a = key_09[i] if i < len(key_09) else "<missing>"
            b = key_15[i] if i < len(key_15) else "<missing>"
            marker = "  <-- DIFF" if a != b else ""
            print(f"  [{i}] 09={a}  15={b}{marker}")
        print("\nSTOPPING per task instructions -- alignment must pass before building GT.")
        return 1

    print("\n=== Dates ===")
    prev_09, cur_09 = extract_dates(pdf_09)
    prev_15, cur_15 = extract_dates(pdf_15)
    print(f"09.06.26 doc dates: prev={prev_09} cur={cur_09}")
    print(f"15.06.26 doc dates: prev={prev_15} cur={cur_15}")

    print("\n=== Building 15's document_gt.json from 09's verified structure ===")
    gt_09 = json.loads(gt_09_path.read_text(encoding="utf-8"))
    grid_09 = gt_09["tables"][0]["data"]

    # Build the ordered list of 15's data rows (9-col already) aligned 1:1
    # with rows_09's data rows in the same order.
    data_rows_15 = [payload for kind, payload in rows_15 if kind == "data"]
    assert len(data_rows_15) == n_data_09, "data row count mismatch after alignment pass"

    new_grid: list[list[str]] = []
    needs_review: list[dict] = []
    data_cursor = 0
    for row_idx, row_09 in enumerate(grid_09):
        is_header = _is_header(row_09[0])
        is_section = _is_section_divider(row_09[0]) if not is_header else False
        if is_header or is_section:
            new_grid.append(list(row_09))
            continue
        # data row: keep 09's no/name/unit, take 15's numeric cells
        row_15 = data_rows_15[data_cursor]
        data_cursor += 1
        new_row = [row_09[0], row_09[1], row_09[2]] + row_15[3:9]
        # sanity: rownum should match (already guaranteed by alignment check)
        assert row_09[0] == row_15[0], f"rownum mismatch at grid row {row_idx}"
        for col_idx in range(3, 9):
            v09 = row_09[col_idx]
            v15 = row_15[col_idx]
            blank09 = v09 == ""
            blank15 = v15 == ""
            if blank09 != blank15:
                needs_review.append({
                    "row_index": row_idx,
                    "col_index": col_idx,
                    "09_value": v09,
                    "15_value": v15,
                    "reason": "blank/value pattern differs between 09 and 15",
                })
        new_grid.append(new_row)

    assert data_cursor == len(data_rows_15), "not all 15 data rows were consumed"

    # Update date headers: header row is new_grid[0]
    header = new_grid[0]
    date_map = {
        f"{prev_09} បោះដុំ": f"{prev_15} បោះដុំ",
        f"{prev_09} លក់រាយ": f"{prev_15} លក់រាយ",
        f"{cur_09} បោះដុំ": f"{cur_15} បោះដុំ",
        f"{cur_09} លក់រាយ": f"{cur_15} លក់រាយ",
    }
    for i, cell in enumerate(header):
        if cell in date_map:
            header[i] = date_map[cell]

    out = {
        "source_stem": _STEM_15,
        "n_pages": 3,
        "mode_cols": _N_COLS,
        "tables": [{"data": new_grid}],
        "needs_review_rows": needs_review,
        "notes": gt_09.get("notes", []) + [
            "DRAFT: built from 09.06.26 verified GT structure via scripts/build_15_gt.py; "
            "numeric cells (cols 3-8) sourced from 15.06.26's own PDF text layer. "
            "Khmer item names (col1) and units (col2) inherited verbatim from 09 -- "
            "notes above may need updating for 15's actual date/content.",
        ],
    }
    out_path = _REAL_DIR / f"{_STEM_15}_document_gt.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # ==== VALIDATION REPORT ====
    print("\n=== Row/col counts ===")
    print(f"15 grid: {len(new_grid)} rows x {len(new_grid[0])} cols")
    print(f"09 grid: {len(grid_09)} rows x {len(grid_09[0])} cols")

    print("\n=== Sample side-by-side (6 data rows) ===")
    sample_idx = [i for i, r in enumerate(grid_09) if _is_rownum(r[0])][:6]
    for i in sample_idx:
        r09 = grid_09[i]
        r15 = new_grid[i]
        print(f"row {i}: {r09[1]!r}")
        print(f"   09 nums: {r09[3:9]}")
        print(f"   15 nums: {r15[3:9]}")

    print(f"\n=== needs_review_rows: {len(needs_review)} cells ===")
    for entry in needs_review:
        print(f"  {entry}")

    print("\n=== PNG dimensions ===")
    from PIL import Image
    for p in sorted(_REAL_DIR.glob(f"{_STEM_15}_p*.png")):
        with Image.open(p) as im:
            print(f"  {p.name}: {im.size}")

    # ==== Per-page GT (step 5) ====
    print("\n=== Per-page GT ===")
    write_per_page_gt(pdf_15, new_grid)

    return 0


def write_per_page_gt(pdf_15: Path, new_grid: list[list[str]]) -> None:
    """Slice the final (name/unit from 09, numbers from 15) merged grid into
    per-page tables, using each PDF page's own data-row boundaries (rownum +
    section index) to find the split points -- mirrors how 09's per-page GT
    files are split. Header row is repeated on every page that starts with
    one in the source PDF (matches 09's p1/p2 pattern); the whole grid is a
    single logical table so a page's slice is just a contiguous run of rows
    from new_grid.
    """
    pages_15 = extract_rows_per_page(pdf_15)

    # Map each data row's (section_index, rownum) -> index in new_grid, so we
    # can find, for each PDF page, which contiguous span of new_grid rows it covers.
    grid_lookup: dict[tuple[int, str], int] = {}
    section_idx = -1
    for i, row in enumerate(new_grid):
        if _is_header(row[0]):
            continue
        if _is_section_divider(row[0]):
            section_idx += 1
            continue
        grid_lookup[(section_idx, row[0])] = i

    section_idx = -1  # carried across pages -- a page may continue a section
    # started on a previous page, with no divider row of its own
    for page_num, page_rows in enumerate(pages_15, start=1):
        first_grid_i = None
        last_grid_i = None
        saw_header = False
        for kind, payload in page_rows:
            if kind == "header":
                saw_header = True
            elif kind == "section":
                section_idx += 1
            elif kind == "data":
                gi = grid_lookup[(section_idx, payload[0])]
                if first_grid_i is None:
                    first_grid_i = gi
                last_grid_i = gi

        if first_grid_i is None:
            print(f"  page {page_num}: no data rows found -- SKIPPING per-page GT (ambiguous)")
            continue

        # include any section-divider row(s) immediately preceding the first
        # data row on this page (it visually belongs to this page)
        span_start = first_grid_i
        while span_start > 0 and _is_section_divider(new_grid[span_start - 1][0]):
            span_start -= 1
        page_grid = [list(new_grid[i]) for i in range(span_start, last_grid_i + 1)]
        # 09's verified per-page GT convention: header row is present on p1
        # (native to the PDF's own text layer there) AND repeated on p2 as a
        # readability aid (even though p2's own PDF table doesn't re-embed
        # it) -- but NOT repeated on p3. Match that exactly: page 1 already
        # carries a native header row (saw_header True there); force it onto
        # page 2 as well; never add it on page 3+.
        if saw_header or page_num == 2:
            page_grid.insert(0, list(new_grid[0]))

        gt = {
            "font_family": "real",
            "template": _STEM_15,
            "document_type": "real",
            "paragraphs": [],
            "tables": [{"data": page_grid}],
            "footer": "",
        }
        out_path = _REAL_DIR / f"{_STEM_15}_p{page_num}_ground_truth.json"
        out_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  page {page_num}: wrote {out_path.name} ({len(page_grid)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
