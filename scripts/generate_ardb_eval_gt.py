"""Generate evaluation GT for the ARDB daily series by template mapping.

Derives the text-layer→GT column mapping from the existing hand-verified pages
(one template per page number), then replays it across every corpus daily to
produce `*_ground_truth.json` + page PNGs for the rest of the series.

Numbers come from each date's own text layer; Khmer labels are carried from the
verified template (byte-identical across dates — measured). Drafts are STAGED to
eval/datasets/real_draft/ and never written into the scored set directly.

    uv run python scripts/generate_ardb_eval_gt.py [--out eval/datasets/real_draft] [--limit N]

Honest scope: these pages add NUMERIC samples only. Layout and Khmer labels are
identical across dates, so they are not independent samples of table structure or
Khmer recognition — each page records `gt_source` so the report can say so.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fitz

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from khmer_pipeline.datagen.harvest_eval_gt import (  # noqa: E402
    apply_mapping, derive_column_mapping, numeric_fidelity,
)

_REAL = _REPO / "eval/datasets/real"
_CORPUS = _REPO / "corpus/ardb_daily"
_DPI = 200
_GT_SOURCE = "textlayer_template_mapped"
# A mapped numeric column should round-trip almost perfectly against the trusted
# page (measured 26/27). Below this the mapping is untrustworthy for that page.
_MIN_NUMERIC_FIDELITY = 0.90


def _raw_grid(page: fitz.Page) -> list[list[str]] | None:
    tabs = page.find_tables().tables
    if not tabs:
        return None
    return [[(c or "").strip() for c in row] for row in tabs[0].extract()]


def _templates() -> dict[int, tuple[list[list[str]], list[list[str]], dict]]:
    """page_number → (verified GT grid, its raw grid, column mapping).

    Built from the hand-verified pages already in eval/datasets/real."""
    out: dict[int, tuple] = {}
    pdfs = {p.stem: p for p in _CORPUS.glob("*.pdf")}
    for gt_path in sorted(_REAL.glob("*_ground_truth.json")):
        m = re.match(r"^(.*)_p(\d+)_ground_truth$", gt_path.stem)
        if not m or m.group(1) not in pdfs:
            continue
        page_no = int(m.group(2))
        if page_no in out:
            continue  # first verified page for this page-number wins
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        grids = [t["data"] for t in gt.get("tables", []) if t.get("data")]
        if not grids:
            continue
        with fitz.open(str(pdfs[m.group(1)])) as doc:
            if page_no - 1 >= len(doc):
                continue
            raw = _raw_grid(doc[page_no - 1])
        if not raw:
            continue
        gt_grid = grids[0]
        mapping = derive_column_mapping(gt_grid, raw)
        fid = numeric_fidelity(gt_grid, raw, mapping)
        # Sanity: replaying the mapping on its own page must return that page's GT.
        # Rows that do not align 1:1 with the raw extract (measured: ARDB p1 has a
        # 24-row GT against a 25-row extract) cannot be template-mapped — skip that
        # page number rather than emit misaligned GT.
        try:
            if apply_mapping(gt_grid, raw, mapping) != gt_grid:
                print(f"  [skip] p{page_no}: mapping does not round-trip its own GT")
                continue
        except ValueError as e:
            print(f"  [skip] p{page_no}: {e}")
            continue
        out[page_no] = (gt_grid, raw, mapping, fid)
        good = sum(1 for v in fid.values() if v >= _MIN_NUMERIC_FIDELITY)
        print(f"  template p{page_no}: {len(gt_grid)}x{len(gt_grid[0])} "
              f"from {gt_path.name[:28]}… ({good} high-fidelity cols)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Template-map ARDB evaluation GT.")
    ap.add_argument("--out", type=Path, default=_REPO / "eval/datasets/real_draft")
    ap.add_argument("--limit", type=int, default=None, help="cap number of PDFs")
    ap.add_argument("--dpi", type=int, default=_DPI)
    args = ap.parse_args()

    print("Deriving templates from verified pages:")
    templates = _templates()
    if not templates:
        sys.exit("no usable templates — need at least one verified *_ground_truth.json")

    args.out.mkdir(parents=True, exist_ok=True)
    verified_stems = {re.sub(r"_p\d+_ground_truth$", "", p.stem)
                      for p in _REAL.glob("*_ground_truth.json")}

    pdfs = sorted(_CORPUS.glob("*.pdf"))[: args.limit]
    stats: dict[str, int] = defaultdict(int)
    for pdf in pdfs:
        if pdf.stem in verified_stems:
            stats["skipped_already_verified"] += 1
            continue
        with fitz.open(str(pdf)) as doc:
            for page_no, tpl in templates.items():
                if page_no - 1 >= len(doc):
                    continue
                gt_grid, tpl_raw, mapping, _fid = tpl
                page = doc[page_no - 1]
                raw = _raw_grid(page)
                if raw is None:
                    stats["no_table"] += 1
                    continue
                # The method rests on the template being rigid; anything with a
                # different shape is a different form and must not be mapped.
                if len(raw) != len(tpl_raw) or len(raw[0]) != len(tpl_raw[0]):
                    stats["shape_mismatch"] += 1
                    continue
                try:
                    grid = apply_mapping(gt_grid, raw, mapping)
                except ValueError:
                    stats["map_refused"] += 1
                    continue

                base = f"{pdf.stem}_p{page_no}"
                pix = page.get_pixmap(matrix=fitz.Matrix(args.dpi / 72, args.dpi / 72))
                pix.save(str(args.out / f"{base}.png"))
                (args.out / f"{base}_ground_truth.json").write_text(
                    json.dumps({
                        "tables": [{"data": grid}],
                        "paragraphs": [],
                        "gt_source": _GT_SOURCE,
                        "gt_template": f"p{page_no}",
                        "gt_note": ("Numbers from this date's text layer; Khmer labels carried "
                                    "from the verified template (identical across dates). "
                                    "Adds numeric samples only — layout/labels are not "
                                    "independent samples."),
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                stats["written"] += 1

    print("\n" + " | ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print(f"→ {args.out}  (STAGED — spot-check before promoting into eval/datasets/real/)")


if __name__ == "__main__":
    main()
