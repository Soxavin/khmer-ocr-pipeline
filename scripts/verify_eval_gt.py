"""Visual verification for template-mapped evaluation GT.

Crops each drafted cell from the rendered page and shows it beside the drafted
value, so a human confirms the GT against PIXELS rather than against the process
that produced it. The mapping is derived from one trusted page and replayed, so
the failure mode to hunt is a column slip — which shows up instantly as a crop
whose digits differ from its label.

    uv run python scripts/verify_eval_gt.py [--dir eval/datasets/real_draft] [--pages 3]

Writes <dir>/verify_sheet.html (self-contained). Nothing is promoted into the
scored set by this script — promotion is a separate, deliberate step.
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import random
import sys
from pathlib import Path

import fitz
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
_CORPUS = _REPO / "corpus/ardb_daily"
_DPI = 200
# Columns worth eyeballing: the numeric ones carry this method's only per-date
# content (Khmer labels are carried verbatim from the verified template).
_NUMERIC_COLS = (3, 4, 5, 6, 7, 8)


def _tile(img: Image.Image, drafted: str, r: int, c: int) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        '<figure style="margin:0;padding:6px;border:1px solid #ddd;border-radius:6px;background:#fff">'
        f'<img src="data:image/png;base64,{b64}" style="display:block;max-width:100%;border:1px solid #eee">'
        f'<figcaption style="font:12px/1.4 monospace;margin-top:4px">'
        f'<b style="color:#070">{html.escape(drafted or "∅")}</b>'
        f'<span style="color:#888"> r{r} c{c}</span></figcaption></figure>'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Visually verify drafted evaluation GT.")
    ap.add_argument("--dir", type=Path, default=_REPO / "eval/datasets/real_draft")
    ap.add_argument("--pages", type=int, default=3, help="how many drafted pages to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    drafts = sorted(glob.glob(str(args.dir / "*_ground_truth.json")))
    if not drafts:
        sys.exit(f"no drafts in {args.dir}")
    rng = random.Random(args.seed)
    sample = rng.sample(drafts, min(args.pages, len(drafts)))

    sections = []
    for gt_path in sample:
        stem = Path(gt_path).name.replace("_ground_truth.json", "")
        doc_stem, page_no = stem.rsplit("_p", 1)
        pdfs = list(_CORPUS.glob(f"{doc_stem}.pdf"))
        if not pdfs:
            continue
        grid = json.loads(Path(gt_path).read_text(encoding="utf-8"))["tables"][0]["data"]
        with fitz.open(str(pdfs[0])) as doc:
            page = doc[int(page_no) - 1]
            tabs = page.find_tables().tables
            if not tabs:
                continue
            # Crop whole ROW bands via the table's own row bboxes — robust, unlike
            # indexing the flat cell list (its layout is not a simple row-major grid).
            row_boxes = [r.bbox for r in tabs[0].rows]
            pix = page.get_pixmap(matrix=fitz.Matrix(_DPI / 72, _DPI / 72))
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        scale = _DPI / 72
        tiles = []
        for r in range(min(6, len(grid), len(row_boxes))):  # first rows suffice to spot a slip
            x0, y0, x1, y1 = row_boxes[r]
            box = (int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = img.crop(box)
            drafted = " | ".join(grid[r][cc] for cc in _NUMERIC_COLS)
            tiles.append(_tile(crop, drafted, r, -1))
        sections.append(
            f'<h2 style="font:600 14px system-ui">{html.escape(stem[-24:])}</h2>'
            '<div style="display:grid;grid-template-columns:1fr;gap:8px;margin-bottom:24px">'
            + "".join(tiles) + "</div>")

    out = args.dir / "verify_sheet.html"
    out.write_text(
        '<!doctype html><meta charset="utf-8"><title>Eval GT verification</title>'
        '<body style="font-family:system-ui;background:#fafafa;margin:24px">'
        "<h1 style='font-size:18px'>Drafted evaluation GT — verify against pixels</h1>"
        "<p style='color:#555;font:13px/1.6 system-ui;max-width:70ch'>Each strip is one table row "
        "as rendered; the green text is the DRAFTED numeric values for that row (cols 3-8, in order). "
        "Read the digits in the image and confirm they match, in order. A column slip shows up as "
        "values that are all present but shifted.</p>"
        + "".join(sections) + "</body>", encoding="utf-8")
    print(f"sampled {len(sections)} page(s) → {out}")


if __name__ == "__main__":
    main()
