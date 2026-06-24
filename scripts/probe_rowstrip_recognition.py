"""Probe whether Surya can column-split a single full-width table ROW strip.

The hybrid engine's per-cell recognition fails on tiny isolated crops (PROJECT_LOG
2.15). This probe tests the row-strip alternative: crop each SLANet row as a
full-width strip and ask Surya to recognise it, two ways, to decide the column
mechanism for the rowband hybrid mode:

  (a) label="Table"  -> does block.html come back as a multi-<td> row we can feed
      to _parse_html_table?  (Primary mechanism)
  (b) label="Text"   -> plain line; does the block expose any per-word/char sub-
      structure with x-coords we could split on?  (Fallback mechanism)

Usage:
    uv run python scripts/probe_rowstrip_recognition.py [page_png] [n_strips]

Defaults to real page 2 (the dense table) and 3 strips.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

from khmer_pipeline.ingest import ingest
from khmer_pipeline.models import PreprocessResult
from khmer_pipeline.surya import run_surya, _get_predictors, _html_to_text, _parse_html_table
from khmer_pipeline.table_stitch import merge_table_regions
from khmer_pipeline.slanet_structure import predict_cells

_DEFAULT_PAGE = ("eval/datasets/real/"
                 "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-09.06.26_p2.png")
_Y_PAD = 8


def _raw_pre(img_path: Path) -> PreprocessResult:
    ing = ingest(img_path.read_bytes(), img_path.name, dpi=200)
    return PreprocessResult(source_name=ing.source_name, page_images=ing.page_images,
                            dpi=ing.dpi, page_count=ing.page_count)


def _row_strips(cells, crop_w, crop_h):
    by_row: dict[int, list] = {}
    for c in cells:
        by_row.setdefault(c["row_id"], []).append(c)
    strips = []
    for row_id in sorted(by_row):
        ys = [v for c in by_row[row_id] for v in (c["bbox"][1], c["bbox"][3])]
        y0 = max(0, int(min(ys)) - _Y_PAD)
        y1 = min(crop_h, int(max(ys)) + _Y_PAD)
        strips.append({"row_id": row_id, "bbox": [0, y0, crop_w, y1],
                       "n_cells": len(by_row[row_id])})
    return strips


def _recognise(rec_pred, strip_rgb, label):
    from surya.layout.schema import LayoutResult, LayoutBox
    h, w = strip_rgb.shape[:2]
    box = LayoutBox(polygon=[[0, 0], [w, 0], [w, h], [0, h]], label=label,
                    raw_label=label, position=0, count=0, confidence=1.0)
    layout = LayoutResult(bboxes=[box], image_bbox=[0.0, 0.0, float(w), float(h)], error=False)
    return rec_pred([Image.fromarray(strip_rgb)], layout_results=[layout], full_page=False)[0]


def main() -> int:
    page_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(_DEFAULT_PAGE)
    n_strips = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if not page_path.exists():
        print("page not found:", page_path)
        return 1

    pre = _raw_pre(page_path)
    base = run_surya(pre)
    page = base.pages[0]
    boxes = [tuple(float(v) for v in t["bbox"]) for t in page.tables if t.get("bbox")]
    print("detected table regions:", len(boxes))
    if not boxes:
        print("no tables detected; nothing to probe")
        return 1

    img = pre.page_images[0]
    h, w = img.shape[:2]
    merged = merge_table_regions(boxes)
    mb = max(merged, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x0, y0, x1, y1 = (max(0, int(mb[0])), max(0, int(mb[1])), min(w, int(mb[2])), min(h, int(mb[3])))
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    print(f"merged table crop: {cw}x{ch} at ({x0},{y0})")

    cells = predict_cells(crop)
    n_rows = max((c["row_id"] for c in cells), default=-1) + 1
    n_cols = max((c["col_id"] for c in cells), default=-1) + 1
    print(f"SLANet grid: {n_rows} rows x {n_cols} cols, {len(cells)} cells")

    strips = _row_strips(cells, cw, ch)
    print(f"row strips: {len(strips)}; probing {min(n_strips, len(strips))}")

    _, rec_pred = _get_predictors()
    picks = strips[: max(1, n_strips)]
    for s in picks:
        sx0, sy0, sx1, sy1 = s["bbox"]
        strip_rgb = crop[sy0:sy1, sx0:sx1]
        print("\n" + "=" * 70)
        print(f"row {s['row_id']}  strip {sx1-sx0}x{sy1-sy0}  slanet_cells={s['n_cells']}")

        t0 = time.perf_counter()
        tbl_ocr = _recognise(rec_pred, strip_rgb, "Table")
        dt = time.perf_counter() - t0
        for b in tbl_ocr.blocks:
            print(f"  [Table] block attrs: {[a for a in dir(b) if not a.startswith('_')]}")
            print(f"  [Table] html: {b.html!r}")
            grid = _parse_html_table(b.html or "")
            cols = (max((c for _, c in grid), default=-1) + 1) if grid else 0
            print(f"  [Table] parsed grid cols={cols} cells={dict(grid)}")
            break
        print(f"  [Table] {len(tbl_ocr.blocks)} block(s), {dt:.1f}s")

        txt_ocr = _recognise(rec_pred, strip_rgb, "Text")
        for b in txt_ocr.blocks:
            print(f"  [Text]  text: {_html_to_text(b.html)!r}")
            extra = {a: getattr(b, a) for a in dir(b)
                     if not a.startswith("_") and a not in ("html",) and not callable(getattr(b, a))}
            print(f"  [Text]  block fields: {extra}")
            break
        print(f"  [Text]  {len(txt_ocr.blocks)} block(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
