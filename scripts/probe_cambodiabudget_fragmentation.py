"""Ground-truth-free cross-layout fragmentation probe (E3).

PROJECT_LOG §2.26 established, on the market-price bulletin template
(09.06.26 / 15.06.26), that a raw dense table page gives Surya ~8 "Table"
layout regions, while preprocessing (always-on `_crop_margins` +
`_cap_resolution` downscale-to-<=2048px long edge) collapses it to ~1
region. That evidence (E1/E2) is n=2 on the *same* document template.

This script (E3) tests whether the mechanism generalizes to a
STRUCTURALLY DIFFERENT layout: CambodiaBudgetExecutioninApr-2024.pdf, a
budget-execution report with different dense-table shapes. This is a
pure layout/fragmentation probe — Tables_Found comes from Surya's layout
model on rendered pixels, which is font-independent, so no ground truth
or text-layer scoring is needed or attempted.

For each target page: ingest at 200 DPI (pipeline default), then run
Surya layout detection on (a) the raw page image and (b) the
preprocessed image (default all-on PreprocessConfig). Count "Table"
labelled layout regions in each condition and report raw long-edge
pixel dims (the mechanism predicts fragmentation when raw long edge
> 2048px, collapsing toward 1 region after the downscale).

Usage:
    OCR_ENGINE=surya uv run python scripts/probe_cambodiabudget_fragmentation.py
"""
from __future__ import annotations
import time
from pathlib import Path

from PIL import Image

from khmer_pipeline.ingest import ingest
from khmer_pipeline.preprocess import preprocess, PreprocessConfig
from khmer_pipeline.engines.surya import _get_predictors
from khmer_pipeline.utils.memory import clear_device_cache

_PDF = Path("sample_data/CambodiaBudgetExecutioninApr-2024.pdf")
_DPI = 200
# 1-indexed pages flagged by the user as dense tables (page 6 noted as less dense).
_TARGET_PAGES_1INDEXED = [3, 4, 5, 6, 8, 9]
_CAP_RESOLUTION_MAX_DIM = 2048  # mirrors preprocess._CAP_RESOLUTION_MAX_DIM


def _count_table_regions(img) -> int:
    layout_pred, _ = _get_predictors()
    pil_img = Image.fromarray(img)
    t0 = time.perf_counter()
    layout_result = layout_pred([pil_img])[0]
    dt = time.perf_counter() - t0
    n_tables = sum(1 for b in layout_result.bboxes if b.label == "Table")
    print(f"    layout in {dt:.1f}s -> {len(layout_result.bboxes)} total regions, "
          f"{n_tables} Table region(s)")
    return n_tables


def main() -> int:
    if not _PDF.exists():
        print(f"PDF not found: {_PDF}")
        return 1

    print(f"Ingesting {_PDF.name} at {_DPI} DPI...")
    ing = ingest(_PDF.read_bytes(), _PDF.name, dpi=_DPI)
    print(f"  {ing.page_count} page(s) total")

    rows: list[tuple[int, int, int, int, int]] = []
    # (1-indexed page, raw_tables, preproc_tables, raw_long_edge, raw_short_edge)

    for page_1idx in _TARGET_PAGES_1INDEXED:
        page_idx = page_1idx - 1
        if page_idx >= ing.page_count:
            print(f"Page {page_1idx} out of range (doc has {ing.page_count} pages); skipping")
            continue

        print(f"\n=== Page {page_1idx} (0-indexed {page_idx}) ===")
        raw_img = ing.page_images[page_idx]
        h, w = raw_img.shape[:2]
        long_edge, short_edge = max(h, w), min(h, w)
        print(f"  raw dims: {w}x{h} (long edge {long_edge}px)")

        print("  [raw] running layout...")
        raw_tables = _count_table_regions(raw_img)
        clear_device_cache()

        # Preprocess only this single page (wrap in a minimal IngestResult-like
        # object via the real preprocess() call restricted to one image) using
        # the pipeline's own default all-on PreprocessConfig.
        single_page_ingest = ing.__class__(
            source_name=ing.source_name,
            page_images=[raw_img],
            dpi=ing.dpi,
            page_count=1,
        )
        pre = preprocess(single_page_ingest, PreprocessConfig())
        pre_img = pre.page_images[0]
        ph, pw = pre_img.shape[:2]
        print(f"  preprocessed dims: {pw}x{ph} (long edge {max(ph, pw)}px)")

        print("  [preprocessed] running layout...")
        pre_tables = _count_table_regions(pre_img)
        clear_device_cache()

        rows.append((page_1idx, raw_tables, pre_tables, long_edge, short_edge))

    print("\n" + "=" * 78)
    print(f"{'page':<6} {'RAW Table-regions':<20} {'PREPROC Table-regions':<24} {'raw long edge (px)':<20}")
    print("-" * 78)
    for page_1idx, raw_t, pre_t, long_edge, short_edge in rows:
        flag = " (>2048)" if long_edge > _CAP_RESOLUTION_MAX_DIM else " (<=2048)"
        print(f"{page_1idx:<6} {raw_t:<20} {pre_t:<24} {long_edge}{flag}")
    print("=" * 78)

    print("\nNote: Surya layout inference is non-deterministic run-to-run; this is a "
          "single pass per condition, matching E1/E2's reporting convention (binary "
          "fragmentation signal treated as the robust readout, not exact counts).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
