"""Gate-first probe: does an alternative layout detector see the dense GDDE table
as ONE region where Surya fragments it into ~8 (PROJECT_LOG §2.12)?

Detection only — no recognition, no pipeline wiring, no end-to-end scoring. Renders
one page via the pipeline's own ingest code, runs each available layout detector,
collects its "table"-labelled boxes, and reports whether the single largest box
already covers ~all of the detector's combined table area (i.e. one clean region).

Detectors probed:
  (a) Surya layout (the pipeline's own LayoutPredictor) — baseline, expected ~8 boxes.
  (b) DocLayout-YOLO via rapid_layout (model_type="doclayout_docstructbench"), ONNX.
  (c) PP-DocLayout via rapid_layout (model_type="pp_doc_layoutv3"), ONNX.

Phase 1 finding: `rapid_layout` (the RapidAI ONNX layout package, same family as the
project's `rapid_table` dep) resolved cleanly via `uv add` alongside surya-ocr/rapid-table
with no PyTorch/PaddlePaddle conflicts, and it bundles ONNX ports of both DocLayout-YOLO
and PP-DocLayout — so no isolated `--no-project` sub-call was needed for this probe.

Usage:
    uv run python scripts/probe_layout_detectors.py
    uv run python scripts/probe_layout_detectors.py --pdf sample_data/foo.pdf --page 1
"""
from __future__ import annotations
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from khmer_pipeline.ingest import ingest
from khmer_pipeline.engines.surya import _get_predictors

# Real GDDE market-price PDF; page index 1 (page 2) is the dense table Surya
# fragments into ~8 regions (docs/PROJECT_LOG.md §2.12).
_DEFAULT_PDF = ("sample_data/"
                 "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-09.06.26.pdf")
_DEFAULT_PAGE = 1
_DEFAULT_DPI = 200

# A single table box is judged to "cover" the whole table region if its area is at
# least this fraction of the union-bbox area of all that detector's table boxes.
_COVERAGE_THRESHOLD = 0.9
# Cross-detector sanity IoU is purely informational; no pass/fail gate on it.
_BOX_COLOR = (255, 0, 0)
_BOX_WIDTH = 4

_Box = tuple[float, float, float, float]


def _union_bbox(boxes: list[_Box]) -> _Box:
    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]
    ys1 = [b[3] for b in boxes]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _area(b: _Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a: _Box, b: _Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _covers_as_one(boxes: list[_Box]) -> tuple[bool, float]:
    if not boxes:
        return False, 0.0
    if len(boxes) == 1:
        return True, 1.0
    union = _union_bbox(boxes)
    largest = max(boxes, key=_area)
    ratio = _area(largest) / _area(union) if _area(union) > 0 else 0.0
    return ratio >= _COVERAGE_THRESHOLD, ratio


def _draw_overlay(img: np.ndarray, boxes: list[_Box], out_path: Path) -> None:
    pil_img = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    for b in boxes:
        draw.rectangle(list(b), outline=_BOX_COLOR, width=_BOX_WIDTH)
    pil_img.save(out_path)


def _run_surya(img: np.ndarray) -> list[_Box]:
    layout_pred, _ = _get_predictors()
    pil_img = Image.fromarray(img)
    t0 = time.perf_counter()
    layout_result = layout_pred([pil_img])[0]
    dt = time.perf_counter() - t0
    boxes = [tuple(float(v) for v in b.bbox) for b in layout_result.bboxes if b.label == "Table"]
    print(f"  [surya] layout in {dt:.1f}s, {len(layout_result.bboxes)} total regions")
    return boxes


def _run_rapid_layout(img: np.ndarray, model_type: str, label: str) -> list[_Box] | None:
    try:
        from rapid_layout import RapidLayout
    except ImportError:
        print(f"  [{label}] rapid_layout not installed; skipping")
        return None
    try:
        engine = RapidLayout(model_type=model_type)
    except Exception as e:
        print(f"  [{label}] failed to load model {model_type!r}: {e}")
        return None
    t0 = time.perf_counter()
    res = engine(img)
    dt = time.perf_counter() - t0
    names = res.class_names if res.class_names is not None else []
    boxes = res.boxes if res.boxes is not None else []
    table_boxes = [tuple(float(v) for v in b) for b, c in zip(boxes, names) if c.lower() == "table"]
    print(f"  [{label}] inference in {dt:.1f}s, {len(names)} total regions, classes seen: {sorted(set(names))}")
    return table_boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default=_DEFAULT_PDF, help="Path to the PDF to probe.")
    parser.add_argument("--page", type=int, default=_DEFAULT_PAGE, help="0-indexed page number.")
    parser.add_argument("--dpi", type=int, default=_DEFAULT_DPI, help="Render DPI (pipeline default 200).")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 1

    ing = ingest(pdf_path.read_bytes(), pdf_path.name, dpi=args.dpi)
    if args.page >= ing.page_count:
        print(f"Page {args.page} out of range; PDF has {ing.page_count} page(s)")
        return 1
    img = ing.page_images[args.page]
    h, w = img.shape[:2]
    print(f"Rendered {pdf_path.name} page {args.page} at {args.dpi} DPI -> {w}x{h}")

    run_dir = Path("eval/runs") / f"{datetime.now():%Y%m%d_%H%M%S}_layout_probe"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing overlays to {run_dir}")

    detectors: list[tuple[str, list[_Box] | None]] = []

    print("\nrunning surya...")
    detectors.append(("surya", _run_surya(img)))

    print("\nrunning doclayout-yolo (rapid_layout)...")
    detectors.append(("doclayout_yolo", _run_rapid_layout(img, "doclayout_docstructbench", "doclayout_yolo")))

    print("\nrunning pp-doclayout (rapid_layout)...")
    detectors.append(("pp_doclayout", _run_rapid_layout(img, "pp_doc_layoutv3", "pp_doclayout")))

    rows = []
    surya_union: _Box | None = None
    for name, boxes in detectors:
        if boxes is None:
            rows.append((name, None, None, "not available (see log above)"))
            continue
        if not boxes:
            rows.append((name, 0, None, "no table regions detected"))
            continue
        n = len(boxes)
        covers, ratio = _covers_as_one(boxes)
        union = _union_bbox(boxes)
        if name == "surya":
            surya_union = union
        note = f"largest/union area ratio={ratio:.2f}"
        rows.append((name, n, covers, note))
        _draw_overlay(img, boxes, run_dir / f"{name}.png")

    # Cross-detector sanity check: are detectors looking at the same area as Surya?
    print("\ncross-detector sanity (IoU of union-of-table-boxes vs Surya's union):")
    if surya_union is not None:
        for name, boxes in detectors:
            if name == "surya" or not boxes:
                continue
            union = _union_bbox(boxes)
            iou = _iou(surya_union, union)
            print(f"  {name}: IoU={iou:.2f}")
    else:
        print("  surya detected no table regions; skipping")

    print("\n" + "=" * 78)
    print(f"{'detector':<16} {'n_table_regions':<16} {'covers_table_as_one':<22} notes")
    print("-" * 78)
    for name, n, covers, note in rows:
        n_str = "n/a" if n is None else str(n)
        covers_str = "n/a" if covers is None else str(covers)
        print(f"{name:<16} {n_str:<16} {covers_str:<22} {note}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
