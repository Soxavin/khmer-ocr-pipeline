"""Verify HITL correction crops, and demo the capture loop.

Two modes, deliberately separate:

  --inspect <dir>   Render a contact sheet from an EXISTING corrections store.
                    Read-only. This is the mode the retrain runbook uses before
                    training on accumulated corrections.

  (default)         DEMO the capture plumbing on a GT page. It synthesizes fake
                    edits by appending a marker to the model's OWN output, so the
                    resulting "corrections" carry the model's errors verbatim and
                    are NOT valid training data — they exist only to prove the
                    capture path works. The demo therefore REFUSES to write into a
                    directory that already holds a corrections.jsonl, so synthetic
                    records can never contaminate a real training corpus.

The contact sheet is a HARD GATE, not a nicety: an off-by-origin bbox produces
plausible-looking-but-shifted crops that would silently poison a fine-tune. Each
crop is labelled `prediction → correction` so the sheet proves the crop is the
RIGHT cell (not merely a well-formed one), and rows are grouped by layout path so
a drift affecting only one detector is obvious by comparison.

Images are base64-embedded — one file, no server, no external assets.

    uv run python scripts/verify_corrections.py [--out corrections_demo]
    KHMER_LAYOUT_DETECTOR=rapid uv run python scripts/verify_corrections.py  # PP path
"""

from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OCR_ENGINE", "surya_kiri")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from khmer_pipeline.ingest import ingest  # noqa: E402
from khmer_pipeline.preprocess import preprocess  # noqa: E402
from khmer_pipeline.engines.engine_registry import get_ocr_engine  # noqa: E402
from khmer_pipeline.corrections import capture_corrections  # noqa: E402

_MAX_TILES = 20


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _contact_sheet(records: list[dict], out_dir: Path, layout_path: str) -> str:
    """One labelled tile per captured crop, as a self-contained HTML fragment."""
    tiles = []
    for rec in records[:_MAX_TILES]:
        img = out_dir / rec["image"]
        if not img.exists():
            continue
        prov = rec["provenance"]
        tiles.append(
            '<figure style="margin:0;padding:8px;border:1px solid #ddd;border-radius:6px;'
            'background:#fff">'
            f'<img src="data:image/png;base64,{_b64(img)}" '
            'style="display:block;max-width:100%;image-rendering:pixelated;'
            'border:1px solid #eee">'
            f'<figcaption style="font:12px/1.5 monospace;margin-top:6px">'
            f'<span style="color:#b00">{html.escape(prov["prediction"] or "∅")}</span>'
            f' <span style="color:#888">→</span> '
            f'<span style="color:#070">{html.escape(rec["text"] or "∅")}</span><br>'
            f'<span style="color:#888">r{prov["row"]} c{prov["col"]} '
            f'conf={prov["confidence"]}</span>'
            "</figcaption></figure>"
        )
    return (
        f'<h2 style="font:600 15px system-ui">Layout path: {html.escape(layout_path)} '
        f'<span style="font-weight:400;color:#666">({len(tiles)} crops shown)</span></h2>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));'
        f'gap:10px;margin-bottom:28px">{"".join(tiles)}</div>'
    )


def _write_sheet(records: list[dict], out_dir: Path, layout_path: str, note: str) -> Path:
    sheet = _contact_sheet(records, out_dir, layout_path)
    html_path = out_dir / "contact_sheet.html"
    html_path.write_text(
        '<!doctype html><meta charset="utf-8"><title>HITL crop verification</title>'
        '<body style="font-family:system-ui;background:#fafafa;margin:24px">'
        "<h1 style='font-size:18px'>HITL captured-crop verification</h1>"
        f"<p style='color:#555;font:13px/1.6 system-ui;max-width:60ch'>{note}</p>"
        f"{sheet}</body>", encoding="utf-8")
    return html_path


def _check_geometry(records: list[dict], dir_: Path) -> list[str]:
    """Every crop's pixel size must equal its recorded bbox. A mismatch means the
    coordinate mapping drifted — the crop is not the cell the label describes, so
    the pair would teach the recognizer a wrong association. Caught here, before
    the data ever reaches training."""
    problems = []
    for rec in records:
        box = (rec.get("provenance") or {}).get("bbox")
        img_path = dir_ / rec.get("image", "")
        if not box or len(box) != 4 or not img_path.exists():
            problems.append(f"{rec.get('image')}: missing crop or bbox")
            continue
        expected = (int(box[2]) - int(box[0]), int(box[3]) - int(box[1]))
        with Image.open(img_path) as im:
            if im.size != expected:
                problems.append(f"{rec['image']}: crop {im.size} != bbox {expected}")
    return problems


def _inspect(dir_: Path) -> None:
    """Render a contact sheet from an existing corrections store. Read-only."""
    jsonl = dir_ / "corrections.jsonl"
    if not jsonl.exists():
        sys.exit(f"no corrections.jsonl in {dir_}")
    records = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(records)} accumulated corrections in {dir_}")
    problems = _check_geometry(records, dir_)
    if problems:
        print(f"GEOMETRY MISMATCH in {len(problems)} record(s) — do not train on this store:")
        for p in problems[:10]:
            print(f"  {p}")
        sys.exit(1)
    print("geometry OK: every crop matches its recorded bbox")
    path = _write_sheet(
        records, dir_, "as captured",
        "Each tile is a real analyst correction that would become a training pair. "
        "Confirm the glyphs are centred and complete, and that the green text is the "
        "TRUE value for the pictured cell — a wrong label here teaches the model the error.")
    print(f"contact sheet → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="HITL crop verification / capture demo.")
    ap.add_argument("--inspect", type=Path, default=None,
                    help="render a contact sheet from an existing corrections store (read-only)")
    ap.add_argument("--out", type=Path, default=_REPO / "corrections_demo",
                    help="demo output dir (must NOT be a real corrections store)")
    ap.add_argument("--page", default=None, help="GT page PNG (default: an ARDB page)")
    args = ap.parse_args()

    if args.inspect:
        _inspect(args.inspect if args.inspect.is_absolute() else _REPO / args.inspect)
        return

    # Demo mode writes synthetic, deliberately-wrong labels. Never let them land in
    # a store that build_trainset.py might later read.
    if (args.out / "corrections.jsonl").exists():
        sys.exit(
            f"{args.out} already holds corrections.jsonl — refusing to append synthetic "
            f"demo records to a real training corpus.\n"
            f"Use:  --inspect {args.out}   to visualise what is already captured.")

    png = args.page or sorted(
        f for f in glob.glob(str(_REPO / "eval/datasets/real/*p2.png")) if "ecution" not in f
    )[0]
    layout_path = "PP-DocLayout (rapid)" if os.environ.get("KHMER_LAYOUT_DETECTOR") else "Surya layout"
    print(f"page={Path(png).name}  layout={layout_path}")

    pre = preprocess(ingest(Path(png).read_bytes(), Path(png).name))
    result = get_ocr_engine("surya_kiri")(pre)
    tables = result.pages[0].tables
    if not tables:
        sys.exit("no tables detected — cannot demo the loop")

    # Synthesize a realistic analyst correction: take the model's grid and edit a
    # handful of non-empty cells, imitating the measured ៛-glyph error class.
    grid: dict[tuple[int, int], str] = {}
    for cell in tables[0]["cells"]:
        lines = cell.get("text_lines") or []
        grid[(cell["row_id"], cell["col_id"])] = (
            " ".join(t["text"] for t in lines if t.get("text")).strip())
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    edited = [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]

    edits = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if edits >= _MAX_TILES:
                break
            if edited[r][c].strip():
                edited[r][c] = edited[r][c] + "✎"  # a real (non-cosmetic) change
                edits += 1
    print(f"synthesized {edits} analyst corrections")

    records = capture_corrections(
        tables=tables, edited_grids={"0": edited},
        page_images=pre.recognition_page_images or pre.page_images,
        source_name=Path(png).name, out_dir=args.out,
    )
    print(f"captured {len(records)} training pairs → {args.out}")
    if not records:
        sys.exit("no pairs captured — check bbox persistence")

    html_path = _write_sheet(
        records, args.out, layout_path,
        "DEMO DATA — the 'corrections' here are the model's own output plus a marker, so they "
        "reproduce its errors and are NOT valid training data. This sheet only proves the capture "
        "path works: check that glyphs are centred and complete (a systematic shift means the "
        "bbox origin math is wrong).")
    print(f"contact sheet → {html_path}")


if __name__ == "__main__":
    main()
