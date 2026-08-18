"""Document-level evaluation of multi-page table stitching.

Runs a whole real doc (all page images) through the active OCR engine, stitches
the per-page tables into document tables, then reports:
  - stitch sanity checks (logical-table count, source pages, row totals, no
    duplicated header rows), which need no ground truth, and
  - scored metrics (Cell_Accuracy / Recall / Table_CER) of the stitched table vs
    the verified document GT (<stem>_document_gt.json), reusing evaluate_table.

    OCR_ENGINE=surya  uv run python scripts/eval_document.py [stem]
    OCR_ENGINE=hybrid KHMER_HYBRID_MODE=rowband uv run python scripts/eval_document.py [stem]

Add --preprocess to run the same preprocess() the product applies (matches
app.py/pipeline.py); default is raw images (see PROJECT_LOG §2.25).

When --preprocess is set, one (and only one) leave-one-out flag may also be
given to flip a single PreprocessConfig field off while leaving the other four
on — used for the E1 component-isolation ablation (PROJECT_LOG §2.25 confound):
  --no-deskew        deskew=False
  --no-sharpen        sharpen=False
  --no-normalise      normalise=False (CLAHE contrast)
  --no-remove-stamps  remove_stamps=False
  --no-table-bg       normalise_table_backgrounds=False (HSV cell desaturation)
These flags are no-ops without --preprocess.

NOTE: scored numbers are only meaningful once you have verified the drafted GT
(see scripts/draft_document_gt.py). Until then treat them as provisional.
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

from khmer_pipeline.ingest import ingest
from khmer_pipeline.models import IngestResult, PreprocessResult
from khmer_pipeline.preprocess import preprocess, PreprocessConfig
from khmer_pipeline.engines.engine_registry import ACTIVE_OCR_ENGINE
from khmer_pipeline.engines.table_merge_pages import merge_document_tables, _rows, _row_sig
from khmer_pipeline.evaluation.evaluate_structure import evaluate_table, pred_table_grid

_REAL_DIR = Path("eval/datasets/real")
_DEFAULT_STEM = "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-09.06.26"

# CLI flag -> PreprocessConfig field, for the E1 leave-one-out ablation.
_ABLATION_FLAGS = {
    "--no-deskew": "deskew",
    "--no-sharpen": "sharpen",
    "--no-normalise": "normalise",
    "--no-remove-stamps": "remove_stamps",
    "--no-table-bg": "normalise_table_backgrounds",
}


def _build_preprocess_config(argv: list[str]) -> PreprocessConfig:
    """Build a PreprocessConfig with all flags on except any --no-<flag> in argv."""
    overrides = {field: False for flag, field in _ABLATION_FLAGS.items() if flag in argv}
    return PreprocessConfig(**overrides)


def _load_pages(stem: str, do_preprocess: bool = False, config: PreprocessConfig | None = None) -> PreprocessResult:
    # Default (raw) matches the historical §2.24 A/B. --preprocess runs the same
    # preprocess() the product (app.py/pipeline.py) applies, so eval reflects
    # production conditions (see PROJECT_LOG §2.25).
    pngs = sorted(glob.glob(str(_REAL_DIR / f"{stem}_p*.png")))
    images = []
    for p in pngs:
        images.extend(ingest(Path(p).read_bytes(), Path(p).name, dpi=200).page_images)
    if do_preprocess:
        ing = IngestResult(source_name=stem, page_images=images, dpi=200, page_count=len(images))
        return preprocess(ing, config or PreprocessConfig())
    return PreprocessResult(source_name=stem, page_images=images, dpi=200, page_count=len(images))


def _duplicate_header_rows(table: dict) -> int:
    rows = _rows(table)
    if not rows:
        return 0
    header = _row_sig(rows[0])
    return sum(1 for r in rows[1:] if _row_sig(r) == header)


def main() -> int:
    do_preprocess = "--preprocess" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    stem = positional[0] if positional else _DEFAULT_STEM
    engine = getattr(ACTIVE_OCR_ENGINE, "__name__", "ocr")
    config = _build_preprocess_config(sys.argv[1:]) if do_preprocess else None
    pre = _load_pages(stem, do_preprocess=do_preprocess, config=config)
    if not pre.page_images:
        print(f"No page PNGs for stem: {stem}")
        return 1
    ablation_note = ""
    if do_preprocess:
        off = [field for flag, field in _ABLATION_FLAGS.items() if flag in sys.argv]
        ablation_note = f"  ablation_off={off or 'none (all-on)'}"
    print(f"engine={engine}  pages={pre.page_count}  preprocess={'on' if do_preprocess else 'off'}{ablation_note}")

    result = ACTIVE_OCR_ENGINE(pre)
    print("\n--- per-page Table region counts (fragmentation signal) ---")
    for p in result.pages:
        print(f"    page{p.page_index + 1}: Tables_Found={len(p.tables)}")
    per_page_tables = sum(len(p.tables) for p in result.pages)
    per_page_rows = sum(len(pred_table_grid(t)) for p in result.pages for t in p.tables)

    merged = merge_document_tables(result.pages)
    merged_rows = sum(len(pred_table_grid(t)) for t in merged)
    dups = sum(_duplicate_header_rows(t) for t in merged)

    print("\n--- stitch sanity checks ---")
    print(f"  per-page tables: {per_page_tables}  ->  logical tables after stitch: {len(merged)}")
    for i, t in enumerate(merged):
        g = pred_table_grid(t)
        cols = max((len(r) for r in g), default=0)
        print(f"    table{i+1}: {len(g)} rows x {cols} cols, source_pages={t.get('source_pages')}")
    print(f"  rows: per-page total {per_page_rows} -> stitched {merged_rows} "
          f"(diff {per_page_rows - merged_rows} = deduped headers/empties)")
    print(f"  duplicated header rows remaining inside stitched tables: {dups}")

    gt_path = _REAL_DIR / f"{stem}_document_gt.json"
    if not gt_path.exists():
        print(f"\n(no {gt_path.name} yet — run draft_document_gt.py + verify for scored metrics)")
        return 0
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    needs_review = gt.get("needs_review_rows", [])
    gt_grid = gt["tables"][0]["data"]
    m = evaluate_table(merged, gt_grid)
    print("\n--- scored vs document GT" + (" (PROVISIONAL — GT has unverified rows)" if needs_review else "") + " ---")
    print(f"  GT {m['gt_rows']}x{m['gt_cols']}  pred {m['pred_rows']}x{m['pred_cols']}")
    print(f"  Cell_Accuracy={m['cell_accuracy']:.3f}  Cell_Content_Recall={m['cell_content_recall']:.3f}  "
          f"Table_CER={m['table_cer']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
