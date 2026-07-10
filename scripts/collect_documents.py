"""Week-1 dataset collection helper: batch-download PDFs + classify them for the dataset factory.

Usage:
    # Download from a URL list (one URL per line, # comments allowed), then report:
    uv run python scripts/collect_documents.py corpus/ --urls urls.txt

    # Just classify an existing folder of PDFs:
    uv run python scripts/collect_documents.py corpus/

Classification comes from khmer_pipeline.datagen.inspect_pdf; this script adds the
dataset-product routing (which PDFs can feed which of the three W2 products) and
progress toward the collection target (>=40 docs, >=100 pages).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from khmer_pipeline.datagen.inspect_pdf import inspect_pdf, khmer_layer_suspect

_TARGET_DOCS = 40
_TARGET_PAGES = 100
_DOWNLOAD_TIMEOUT_S = 60

# classification -> which dataset products the doc can feed (plan §2: one factory, three products)
_ROUTING = {
    "born_digital_unicode": "layout + recognition pairs + VLM SFT (text layer is free GT)",
    "likely_legacy_encoded": "layout + numeric-only harvest (find_tables; Khmer needs manual pass)",
    "scanned_image_only": "layout only (no text layer; recognition GT would be manual)",
    "mixed_or_unknown": "layout only until inspected manually",
    "error": "unusable — file failed to open",
}


def download_urls(url_file: Path, dest_dir: Path) -> list[Path]:
    """Download every URL in url_file into dest_dir; skips files that already exist."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    urls = [
        line.strip()
        for line in url_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for url in urls:
        name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name or "download.pdf"
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        dest = dest_dir / name
        if dest.exists():
            print(f"skip (exists): {name}")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                print(f"WARNING: not a PDF, skipped: {url}")
                continue
            dest.write_bytes(data)
            downloaded.append(dest)
            print(f"downloaded: {name} ({len(data) // 1024} KB)")
        except Exception as e:
            print(f"WARNING: failed {url}: {e}")
    return downloaded


def report(folder: Path, output: Path) -> None:
    """Classify all PDFs under folder (recursive), print routing summary + progress, write JSON."""
    pdfs = sorted(folder.rglob("*.pdf")) if folder.is_dir() else [folder]
    if not pdfs:
        print(f"No PDFs found under {folder}")
        return

    results: list[dict] = []
    for pdf in pdfs:
        (r,) = inspect_pdf(pdf)
        r["relpath"] = str(pdf.relative_to(folder)) if folder.is_dir() else pdf.name
        if r["classification"] == "born_digital_unicode":
            r["khmer_layer_suspect"] = khmer_layer_suspect(pdf)
        results.append(r)
    suspects = [r["relpath"] for r in results if r.get("khmer_layer_suspect")]

    by_class: dict[str, list[dict]] = {}
    for r in results:
        by_class.setdefault(r["classification"], []).append(r)

    total_pages = sum(r.get("page_count", 0) for r in results)

    print(f"\n{'Classification':<24} {'Docs':>4} {'Pages':>5}  Feeds")
    print("-" * 96)
    for cls, docs in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        pages = sum(d.get("page_count", 0) for d in docs)
        print(f"{cls:<24} {len(docs):>4} {pages:>5}  {_ROUTING.get(cls, '?')}")

    if suspects:
        print("\nWARNING: Khmer text layer looks LEGACY/mojibake in (numbers usable via "
              "find_tables, Khmer is NOT free recognition GT — §2.21/§2.37):")
        for name in suspects:
            print(f"  - {name}")

    print("-" * 96)
    doc_ok = "OK" if len(results) >= _TARGET_DOCS else f"need {_TARGET_DOCS - len(results)} more"
    page_ok = "OK" if total_pages >= _TARGET_PAGES else f"need {_TARGET_PAGES - total_pages} more"
    print(f"TOTAL: {len(results)} docs ({doc_ok}), {total_pages} pages ({page_ok})  "
          f"[target >= {_TARGET_DOCS} docs / >= {_TARGET_PAGES} pages]")

    output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nFull per-file report: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect + classify PDFs for the dataset factory.")
    parser.add_argument("folder", type=Path, help="Corpus folder (PDFs live/land here)")
    parser.add_argument("--urls", type=Path, default=None,
                        help="Optional text file of PDF URLs to download first (one per line)")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON report path (default: <folder>/collection_report.json)")
    args = parser.parse_args()

    if args.urls:
        download_urls(args.urls, args.folder)
    report(args.folder, args.output or args.folder / "collection_report.json")


if __name__ == "__main__":
    main()
