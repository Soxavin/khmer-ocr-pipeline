"""Cell-by-cell review sheet for OCR-drafted GT on a scanned page.

A scan has no text layer, so its GT can only start from the OCR's own output —
which carries an anchoring risk: a wrong cell looks plausible precisely because
the model produced it. This sheet therefore shows EVERY cell's crop beside the
drafted text (not just low-confidence ones), so the human reads pixels and
corrects the draft rather than ratifying it.

    uv run python scripts/review_scan_gt.py --dir eval/datasets/moc_gas_draft

Writes <dir>/review_sheet.html (self-contained) and a corrections template the
verified values can be typed into.
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OCR_ENGINE", "auto")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from khmer_pipeline.ingest import ingest  # noqa: E402
from khmer_pipeline.preprocess import preprocess  # noqa: E402
from khmer_pipeline.engines.engine_registry import get_ocr_engine  # noqa: E402

_PAD = 2  # px of context around each cell crop


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cell-level review sheet for scanned-page GT.")
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()

    pngs = [p for p in glob.glob(str(args.dir / "*.png"))]
    if not pngs:
        sys.exit(f"no page PNG in {args.dir}")
    png = Path(pngs[0])

    pre = preprocess(ingest(png.read_bytes(), png.name))
    result = get_ocr_engine("auto")(pre)
    frame = (pre.recognition_page_images or pre.page_images)[0]
    tables = result.pages[0].tables
    if not tables:
        sys.exit("no table detected — cannot build a review sheet")

    cells = sorted(tables[0]["cells"], key=lambda c: (c["row_id"], c["col_id"]))
    n_cols = max(c["col_id"] for c in cells) + 1
    h, w = frame.shape[:2]

    tiles, missing = [], 0
    for c in cells:
        box = c.get("bbox")
        text = " ".join(t["text"] for t in (c.get("text_lines") or []) if t.get("text")).strip()
        if not box or len(box) != 4:
            missing += 1
            crop_html = '<div style="height:38px;background:#f3f3f3;border:1px dashed #bbb"></div>'
        else:
            x0, y0, x1, y1 = (int(v) for v in box)
            x0, y0 = max(0, x0 - _PAD), max(0, y0 - _PAD)
            x1, y1 = min(w, x1 + _PAD), min(h, y1 + _PAD)
            if x1 - x0 < 2 or y1 - y0 < 2:
                missing += 1
                crop_html = '<div style="height:38px;background:#f3f3f3"></div>'
            else:
                crop_html = (f'<img src="data:image/png;base64,{_b64(Image.fromarray(frame[y0:y1, x0:x1]))}" '
                             'style="display:block;max-width:100%;border:1px solid #eee">')
        conf = c.get("confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        tiles.append(
            '<figure style="margin:0;padding:6px;border:1px solid #ddd;border-radius:6px;background:#fff">'
            f'{crop_html}'
            f'<figcaption style="font:12px/1.4 monospace;margin-top:4px">'
            f'<b>{html.escape(text or "∅")}</b>'
            f'<span style="color:#888"> r{c["row_id"]}c{c["col_id"]} conf={conf_s}</span>'
            "</figcaption></figure>")

    out = args.dir / "review_sheet.html"
    out.write_text(
        '<!doctype html><meta charset="utf-8"><title>Scanned-page GT review</title>'
        '<body style="font-family:system-ui;background:#fafafa;margin:24px">'
        f"<h1 style='font-size:18px'>{html.escape(png.name[:48])} — verify EVERY cell</h1>"
        "<p style='color:#a00;font:13px/1.6 system-ui;max-width:70ch'><b>This draft is the OCR's own "
        "output, not ground truth.</b> A wrong cell looks plausible because the model wrote it — read "
        "the image, not the label. Correct anything that differs; only then is it GT.</p>"
        f'<div style="display:grid;grid-template-columns:repeat({n_cols},1fr);gap:8px">'
        + "".join(tiles) + "</div></body>", encoding="utf-8")
    print(f"{len(cells)} cells ({missing} without usable geometry) → {out}")


if __name__ == "__main__":
    main()
