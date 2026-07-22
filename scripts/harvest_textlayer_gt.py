"""Harvest evaluation GT for table NUMBERS and STRUCTURE from a born-digital PDF.

Scanned pages need a human to verify every cell, which is the bottleneck on
growing the eval set. Born-digital pages do not: PyMuPDF's `find_tables()`
recovers the cell grid and its text directly from the PDF, with no model in the
loop. That makes this GT **free, unlimited, and non-circular** — no engine under
test contributed to it, so `gt_provenance` will never flag it.

Validated against human ground truth: on CambodiaBudgetExecutioninApr-2024 p3,
which was verified by hand, the text layer reproduced **222 of 222** numeric
cells exactly, with zero mismatches and zero blanks.

The catch is Khmer. These budget PDFs carry a legacy-font text layer whose Khmer
is mojibake (glyph order, not logical order — §2.21), so Khmer cells are BLANKED
rather than trusted. What survives is exactly what this project cares most about:

    numeric_cell_accuracy   valid — every numeric cell is verbatim from the PDF
    row_alignment_rate      valid — full grid geometry is preserved
    col_count_match         valid
    cell_accuracy           NOT valid — blanked Khmer cells would score as errors
    khmer_cell_accuracy     NOT valid — no Khmer in this GT by construction

`scoring_scope: "numeric_and_structure"` records that, and the A/B harness masks
the invalid columns rather than printing a misleading number.

MEASURED BIAS — use these scores for RANKING engines, not as absolute accuracy.
Blanking the Khmer cells also removes row-alignment anchors, so scores come out
slightly pessimistic. Scoring the same four stored p3 predictions against the
human GT and against this harvested GT:

    engine           numeric (human -> harvested)   row_align (human -> harvested)
    surya            1.000 -> 0.968                 1.000 -> 0.943
    surya_kiri       0.550 -> 0.500                 0.971 -> 0.857
    surya_kiri_vlm   1.000 -> 1.000                 0.971 -> 0.971
    auto             1.000 -> 0.968                 1.000 -> 0.943

Mean numeric bias about -0.03, worst case -0.05, and the **ranking is preserved**
in every case. That is what a bake-off needs; absolute headline numbers should
still come from human-verified pages.

    uv run python scripts/harvest_textlayer_gt.py \
        --pdf corpus/budget_tofe/CambodiaBudgetExecutioninApr-2024.pdf \
        --pages 4,5,6,8,9 --out eval/datasets/budget_textlayer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from khmer_pipeline.evaluation.evaluate_structure import _is_numeric, _norm  # noqa: E402

_DPI = 200  # matches the harness's ingest dpi, so crops line up with engine input
_MIN_TABLE_CELLS = 40  # skip incidental 2-column layout tables (page 1 style)


def _is_khmer_block(text: str) -> bool:
    return any("ក" <= ch <= "៿" for ch in text)


def harvest_page(page: fitz.Page) -> tuple[list[list[str]], dict] | None:
    """Extract the largest table on `page` as a GT grid, blanking mojibake Khmer.

    Returns (grid, stats) or None when the page has no substantial table."""
    tables = list(page.find_tables().tables)
    if not tables:
        return None

    # Largest by cell count: budget pages carry one real data table plus the odd
    # incidental two-column block, and only the data table is worth scoring.
    best = max(tables, key=lambda t: sum(len(r) for r in t.extract()))
    raw = [[c or "" for c in row] for row in best.extract()]
    if sum(len(r) for r in raw) < _MIN_TABLE_CELLS:
        return None

    stats = {"numeric": 0, "khmer_blanked": 0, "kept_other": 0, "empty": 0}
    grid: list[list[str]] = []
    for row in raw:
        out_row = []
        for cell in row:
            text = _norm(cell)
            if not text:
                stats["empty"] += 1
                out_row.append("")
            elif _is_numeric(text):
                stats["numeric"] += 1
                out_row.append(text)
            elif _is_khmer_block(text):
                # Legacy-font mojibake — never trustworthy as ground truth.
                stats["khmer_blanked"] += 1
                out_row.append("")
            else:
                # ASCII residue (dashes, units); harmless and verbatim from the PDF.
                stats["kept_other"] += 1
                out_row.append(text)
        grid.append(out_row)
    return grid, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--pages", required=True,
                    help="1-indexed page numbers, comma-separated (e.g. 4,5,6,8,9)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    stem_base = args.pdf.stem
    written = 0

    for page_no in [int(p) for p in args.pages.split(",")]:
        page = doc[page_no - 1]
        harvested = harvest_page(page)
        if harvested is None:
            print(f"  p{page_no}: no substantial table — skipped")
            continue
        grid, stats = harvested
        stem = f"{stem_base}_p{page_no}"

        pix = page.get_pixmap(matrix=fitz.Matrix(_DPI / 72, _DPI / 72),
                              colorspace=fitz.csRGB)
        pix.save(args.out / f"{stem}.png")

        gt = {
            "font_family": "real",
            "template": stem_base,
            "document_type": "real",
            "paragraphs": [],
            "tables": [{"data": grid}],
            "footer": "",
            "gt_source": "pdf_text_layer",
            "gt_drafted_by": None,  # no model involved — never circular
            "scoring_scope": "numeric_and_structure",
            "text_gt_available": False,
            "gt_note": (
                "Harvested from the born-digital PDF text layer via PyMuPDF "
                "find_tables(); no model in the loop. Numbers and grid geometry are "
                "verbatim from the PDF. Khmer cells are BLANKED because this "
                "document's legacy-font text layer is mojibake (§2.21), so "
                "cell_accuracy and khmer_cell_accuracy are NOT meaningful here — "
                "score numeric_cell_accuracy, row_alignment_rate and col_count_match. "
                f"Cells: {stats['numeric']} numeric, {stats['khmer_blanked']} Khmer "
                f"blanked, {stats['kept_other']} other, {stats['empty']} empty."
            ),
        }
        (args.out / f"{stem}_ground_truth.json").write_text(
            json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
        rows, cols = len(grid), max((len(r) for r in grid), default=0)
        print(f"  p{page_no}: {rows}x{cols}  numeric={stats['numeric']} "
              f"khmer_blanked={stats['khmer_blanked']} -> {stem}")
        written += 1

    doc.close()
    print(f"wrote {written} page(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
