"""Extract born-digital PDF pages into eval/datasets/real/ ground-truth pairs.

CLI:
    python -m khmer_pipeline.datagen.harvest_ground_truth <pdf> [--output-dir eval/datasets/real] [--dpi 200]

NOTE: harvested GT paragraphs must be hand-verified and tables hand-filled
with {"data": [[...]]} entries before use in benchmarks.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import fitz

_DEFAULT_OUTPUT_DIR = Path("eval/datasets/real")
_DEFAULT_DPI = 200


def harvest(pdf_path: Path, output_dir: Path, dpi: int = _DEFAULT_DPI) -> list[Path]:
    """Render each page to PNG + emit _ground_truth.json. Returns list of written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    doc = fitz.open(str(pdf_path))
    written: list[Path] = []

    with doc:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            base = f"{stem}_p{page_num}"

            # Render page to PNG
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            png_path = output_dir / f"{base}.png"
            pix.save(str(png_path))
            written.append(png_path)

            # Extract paragraphs from text blocks
            blocks = page.get_text("blocks")
            paragraphs: list[str] = []
            for block in blocks:
                # blocks: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type 0 = text block
                if len(block) >= 6 and block[6] == 0:
                    raw = block[4]
                    cleaned = unicodedata.normalize("NFC", raw).strip()
                    if cleaned:
                        paragraphs.append(cleaned)

            gt = {
                "font_family": "real",
                "template": stem,
                "document_type": "real",
                "paragraphs": paragraphs,
                "tables": [],  # intentionally empty — hand-fill with {"data": [[...]]}
                "footer": "",
            }

            gt_path = output_dir / f"{base}_ground_truth.json"
            gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2))
            written.append(gt_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest born-digital PDF pages into eval ground-truth pairs."
    )
    parser.add_argument("pdf", type=Path, help="Born-digital PDF to process")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--dpi", type=int, default=_DEFAULT_DPI,
                        help=f"Render DPI (default: {_DEFAULT_DPI})")
    args = parser.parse_args()

    written = harvest(args.pdf, args.output_dir, args.dpi)

    print(f"Harvested {len(written)} files into {args.output_dir}/")
    for p in written:
        print(f"  {p}")

    print()
    print("IMPORTANT: Before using these files in benchmarks you must:")
    print("  (a) Verify paragraph text is correct (OCR-free text layer may be imperfect)")
    print('  (b) Hand-fill "tables" in each _ground_truth.json with {"data": [[cell, ...]]} entries')
    print("  (c) For scanned pages, label the ground truth entirely by hand")


if __name__ == "__main__":
    main()
