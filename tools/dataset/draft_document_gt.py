"""Draft a document-level ground truth for a multi-page real doc.

The real ARDB price docs are ONE continuous 9-column table split across page
images, with embedded section-divider rows (e.g. "ខ/-តម្មៃបន្លែ"). Per-page GT
already exists, but some pages have a clean tables[0].data grid while others
(p3) left the table content in `paragraphs`. This stitches them into a single
document-level grid so multi-page stitching can be scored.

    uv run python scripts/draft_document_gt.py [template_stem]

Writes eval/datasets/real/<stem>_document_gt.json = {"tables": [{"data": grid}]}
plus "needs_review_rows" (rows whose column count != the table's mode — verify
these by hand) and "notes" (non-table lines). Born-digital content is already
there; this only restructures it. ALWAYS eyeball the output before scoring.
"""
from __future__ import annotations
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REAL_DIR = Path("eval/datasets/real")
_DEFAULT_STEM = "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-09.06.26"
_N_COLS = 9
_ITEM_NAME_RE = re.compile(r"^([០-៩]+)\s+(.+)$")  # leading Khmer-numeral item no. + name


def _page_files(stem: str) -> list[Path]:
    files = glob.glob(str(_REAL_DIR / f"{stem}_p*_ground_truth.json"))
    return sorted(Path(f) for f in files)


def _parse_paragraph(para: str):
    # returns ("row", cells) | ("section", text) | ("note", text)
    fields = [f.strip() for f in para.split("\n")]
    nonempty = [f for f in fields if f]
    if len(nonempty) <= 1:
        text = nonempty[0] if nonempty else ""
        return ("section", text) if "/-" in text else ("note", text)
    head = nonempty[0]
    m = _ITEM_NAME_RE.match(head)
    if m:
        nonempty = [m.group(1), m.group(2)] + nonempty[1:]
    return ("row", nonempty)


def _section_row(text: str) -> list[str]:
    return [text] + [""] * (_N_COLS - 1)


def _is_header(row: list[str]) -> bool:
    return bool(row) and "ល.រ" in row[0]


def _rows_from_page(gt: dict) -> tuple[list[list[str]], list[str]]:
    tables = gt.get("tables") or []
    if tables and tables[0].get("data"):
        return [list(r) for r in tables[0]["data"]], []
    rows: list[list[str]] = []
    notes: list[str] = []
    for para in gt.get("paragraphs", []):
        kind, payload = _parse_paragraph(para)
        if kind == "row":
            rows.append(payload)
        elif kind == "section":
            rows.append(_section_row(payload))
        else:
            if payload:
                notes.append(payload)
    return rows, notes


def main() -> int:
    stem = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_STEM
    pages = _page_files(stem)
    if not pages:
        print(f"No per-page GT found for stem: {stem}")
        return 1

    merged: list[list[str]] = []
    notes: list[str] = []
    for i, pf in enumerate(pages):
        gt = json.loads(pf.read_text(encoding="utf-8"))
        rows, page_notes = _rows_from_page(gt)
        notes.extend(page_notes)
        for r in rows:
            if _is_header(r):
                # keep only the first header; drop repeats on later pages
                if any(_is_header(m) for m in merged):
                    continue
            merged.append(r)
        print(f"  {pf.name.split('-')[-1].replace('_ground_truth.json','')}: +{len(rows)} rows")

    mode_cols = Counter(len(r) for r in merged).most_common(1)[0][0] if merged else _N_COLS
    needs_review = [i for i, r in enumerate(merged) if len(r) != mode_cols]

    out = {
        "source_stem": stem,
        "n_pages": len(pages),
        "mode_cols": mode_cols,
        "tables": [{"data": merged}],
        "needs_review_rows": needs_review,
        "notes": notes,
    }
    out_path = _REAL_DIR / f"{stem}_document_gt.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {out_path}")
    print(f"  total rows: {len(merged)}  (mode columns: {mode_cols})")
    print(f"  rows needing review (col count != {mode_cols}): {len(needs_review)} -> {needs_review}")
    print(f"  non-table notes captured: {len(notes)}")
    if needs_review:
        print("\n  --- review these rows (likely the sparse grain section) ---")
        for i in needs_review[:40]:
            print(f"   row {i} ({len(merged[i])} cols): {merged[i]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
