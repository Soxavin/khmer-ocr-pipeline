"""Diagnose real PDFs: text layer presence, Unicode vs legacy Khmer encoding, scanned/image-only.

CLI:
    python -m khmer_pipeline.datagen.inspect_pdf <path-to-pdf-or-dir> [--output inspect_report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz

# Classification thresholds
_MIN_TEXT_CHARS = 100          # below this = "no substantial text layer"
_UNICODE_KHMER_RATIO = 0.5    # khmer_block / alpha_chars >= this → born_digital_unicode
_LEGACY_KHMER_RATIO = 0.15    # khmer_block / alpha_chars <= this → possibly legacy-encoded


def _khmer_char_count(text: str) -> int:
    # Count chars in Unicode Khmer block U+1780–U+17FF
    return sum(1 for ch in text if "ក" <= ch <= "៿")


def _latin_char_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha() and ch.isascii())


def inspect_pdf(path: Path) -> list[dict]:
    """Return one dict per PDF found at path (file) or under path (dir)."""
    pdfs: list[Path] = []
    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
    elif path.suffix.lower() == ".pdf":
        pdfs = [path]
    else:
        raise ValueError(f"Expected a PDF file or directory, got: {path}")

    results: list[dict] = []
    for pdf_path in pdfs:
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            results.append({
                "filename": pdf_path.name,
                "error": str(e),
                "classification": "error",
            })
            continue

        with doc:
            page_count = len(doc)
            total_text = ""
            total_images = False
            max_w = 0
            max_h = 0

            for page in doc:
                text = page.get_text("text")
                total_text += text

                img_list = page.get_images(full=True)
                if img_list:
                    total_images = True
                    for img_info in img_list:
                        xref = img_info[0]
                        try:
                            img_data = doc.extract_image(xref)
                            w = img_data.get("width", 0)
                            h = img_data.get("height", 0)
                            if w * h > max_w * max_h:
                                max_w, max_h = w, h
                        except Exception:
                            pass

        text_chars = len(total_text)
        khmer_block_chars = _khmer_char_count(total_text)
        latin_chars = _latin_char_count(total_text)
        alpha_chars = khmer_block_chars + latin_chars
        khmer_ratio = khmer_block_chars / max(1, alpha_chars)

        substantial = text_chars >= _MIN_TEXT_CHARS

        if substantial and khmer_ratio >= _UNICODE_KHMER_RATIO:
            classification = "born_digital_unicode"
        elif substantial and khmer_ratio <= _LEGACY_KHMER_RATIO and latin_chars > khmer_block_chars:
            # legacy Limon/ABC encode Khmer glyphs as Latin code points → high latin ratio
            classification = "likely_legacy_encoded"
        elif not substantial and total_images:
            classification = "scanned_image_only"
        else:
            classification = "mixed_or_unknown"

        results.append({
            "filename": pdf_path.name,
            "page_count": page_count,
            "text_chars": text_chars,
            "khmer_block_chars": khmer_block_chars,
            "latin_chars": latin_chars,
            "khmer_ratio": round(khmer_ratio, 4),
            "has_images": total_images,
            "max_image_dims": [max_w, max_h],
            "classification": classification,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PDFs for encoding type and text layer.")
    parser.add_argument("path", type=Path, help="PDF file or directory of PDFs")
    parser.add_argument("--output", type=Path, default=Path("inspect_report.json"),
                        help="JSON output path (default: inspect_report.json)")
    args = parser.parse_args()

    report = inspect_pdf(args.path)

    # Print readable table to stdout
    header = f"{'Filename':<35} {'Pages':>5} {'TextChars':>10} {'KhmerRatio':>11} {'HasImg':>6} {'Classification'}"
    print(header)
    print("-" * len(header))
    for r in report:
        if "error" in r:
            print(f"{r['filename']:<35} ERROR: {r['error']}")
            continue
        print(
            f"{r['filename']:<35} {r['page_count']:>5} {r['text_chars']:>10} "
            f"{r['khmer_ratio']:>11.4f} {str(r['has_images']):>6}  {r['classification']}"
        )

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nFull report written to: {args.output}")


if __name__ == "__main__":
    main()
