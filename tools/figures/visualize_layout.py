"""Per-page overlay of Surya vs DocLayout-YOLO table boxes for manual verification.

Companion to scripts/probe_layout_detectors.py's finding (docs/PROJECT_LOG.md
§2.12): on the 3-page real GDDE market-price doc, Surya FRAGMENTS the dense
table into many `Table` boxes, while DocLayout-YOLO (via rapid_layout, model
doclayout_docstructbench) gives ONE box but CLIPS OFF the two leftmost columns
(Khmer item-name + unit), treating them as non-table text.

This script draws both detectors' table boxes on every page of the document so
a human can see the fragmentation and the left-column clipping at a glance.
Detection only — no recognition, no pipeline wiring.

Usage:
    uv run python scripts/visualize_layout.py
    uv run python scripts/visualize_layout.py --stem <other_stem>
"""
from __future__ import annotations
import argparse
import glob
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from khmer_pipeline.ingest import ingest
from eval_document import _REAL_DIR, _DEFAULT_STEM
from probe_layout_detectors import _run_surya, _run_rapid_layout, _Box

_DPI = 200

_SURYA_COLOR = (255, 0, 0)
_DOCLAYOUT_COLOR = (0, 200, 0)
_BOX_WIDTH = 6
_LEGEND_COLOR = (0, 0, 0)
_LEGEND_BG = (255, 255, 255)
_LEGEND_MARGIN = 10
_LEGEND_LINE_HEIGHT = 14


def _load_pages(stem: str) -> list:
    pngs = sorted(glob.glob(str(_REAL_DIR / f"{stem}_p*.png")))
    images = []
    for p in pngs:
        images.extend(ingest(Path(p).read_bytes(), Path(p).name, dpi=_DPI).page_images)
    return images


def _draw_legend(draw: ImageDraw.ImageDraw, surya_n: int, doclayout_n: int) -> None:
    lines = [
        "red = Surya, green = DocLayout-YOLO",
        f"Surya: {surya_n} box{'es' if surya_n != 1 else ''} | "
        f"DocLayout-YOLO: {doclayout_n} box{'es' if doclayout_n != 1 else ''}",
    ]
    text_w = max(draw.textlength(line) for line in lines)
    text_h = _LEGEND_LINE_HEIGHT * len(lines)
    draw.rectangle(
        [0, 0, text_w + 2 * _LEGEND_MARGIN, text_h + 2 * _LEGEND_MARGIN],
        fill=_LEGEND_BG,
    )
    for i, line in enumerate(lines):
        draw.text(
            (_LEGEND_MARGIN, _LEGEND_MARGIN + i * _LEGEND_LINE_HEIGHT),
            line,
            fill=_LEGEND_COLOR,
        )


def _draw_both_overlay(
    img,
    surya_boxes: list[_Box],
    doclayout_boxes: list[_Box],
    out_path: Path,
) -> None:
    pil_img = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    for b in surya_boxes:
        draw.rectangle(list(b), outline=_SURYA_COLOR, width=_BOX_WIDTH)
    for b in doclayout_boxes:
        draw.rectangle(list(b), outline=_DOCLAYOUT_COLOR, width=_BOX_WIDTH)
    _draw_legend(draw, len(surya_boxes), len(doclayout_boxes))
    pil_img.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default=_DEFAULT_STEM, help="Document stem under eval/datasets/real.")
    args = parser.parse_args()

    pages = _load_pages(args.stem)
    if not pages:
        print(f"No page PNGs found for stem: {args.stem}")
        return 1
    print(f"Loaded {len(pages)} page(s) for stem: {args.stem}")

    run_dir = Path("eval/runs") / f"{datetime.now():%Y%m%d_%H%M%S}_layout_viz"
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for idx, img in enumerate(pages):
        print(f"\npage {idx}:")
        surya_boxes = _run_surya(img)
        doclayout_boxes = _run_rapid_layout(img, "doclayout_docstructbench", "doclayout_yolo") or []

        out_path = run_dir / f"page{idx}_both.png"
        _draw_both_overlay(img, surya_boxes, doclayout_boxes, out_path)
        saved_paths.append(out_path)

        print(f"  Surya: {len(surya_boxes)} box(es) | DocLayout-YOLO: {len(doclayout_boxes)} box(es)")
        print(f"  saved -> {out_path}")

    print(f"\nrun dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
