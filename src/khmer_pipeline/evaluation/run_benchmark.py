from __future__ import annotations
import csv
import importlib.metadata
import json
import platform
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path

from ..ingest import ingest
from ..models import PreprocessResult
from ..preprocess import preprocess, PreprocessConfig
from ..engines.engine_registry import ACTIVE_OCR_ENGINE, ACTIVE_CORRECTION_ENGINE
from .evaluate_structure import gt_table_grid, evaluate_table, evaluate_text, evaluate_document, pool_gt_text, pool_pred_text
from ..utils.memory import clear_device_cache
from .analyze_benchmark import summarize

# Anchor eval/ to the repo root (…/src/khmer_pipeline/evaluation/run_benchmark.py
# → parents[3]) so runs land in the repo's eval/ regardless of the CWD the
# benchmark is launched from.
_EVAL_ROOT = Path(__file__).resolve().parents[3] / "eval"
_DATASETS_ROOT = _EVAL_ROOT / "datasets"
_RUNS_ROOT = _EVAL_ROOT / "runs"
_DEFAULT_DATASETS = [_DATASETS_ROOT / "synthetic_tables", _DATASETS_ROOT / "synthetic_documents"]

_CSV_FIELDS = [
    "Engine", "Corrected",
    "Dataset", "Image_File", "Font", "Template",
    "Tables_Expected", "Tables_Found",
    "GT_Rows", "GT_Cols", "Pred_Rows", "Pred_Cols",
    "Cell_Accuracy", "Cell_Content_Recall", "Table_CER",
    "Numeric_Cell_Accuracy", "Numeric_Khmer_Digit_Slips", "Empty_Cell_Precision",
    "Text_CER", "Document_CER", "Paragraph_Recall", "Paragraph_Leak",
    "Error",
]


def _fmt(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _engine_name() -> str:
    return getattr(ACTIVE_OCR_ENGINE, "__name__", "ocr")


def _default_run_dir(engine: str, now) -> Path:
    # now: datetime, injected for tests
    return _RUNS_ROOT / f"{now:%Y%m%d_%H%M%S}_{engine}"


def _done_keys(csv_path: Path) -> set[tuple[str, str]]:
    # set of (Dataset, Image_File) already present; empty set if file missing
    if not csv_path.exists():
        return set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {(r["Dataset"], r["Image_File"]) for r in reader}


def _read_all_rows(csv_path: Path) -> list[dict]:
    """Read every data row from results.csv (empty list if missing). Used after
    the loop to recompute aggregates/summary over the FULL run, incl. --resume."""
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _raw_preprocess_result(ingest_result) -> PreprocessResult:
    # wrap IngestResult as PreprocessResult with no transforms applied
    return PreprocessResult(
        source_name=ingest_result.source_name,
        page_images=ingest_result.page_images,
        dpi=ingest_result.dpi,
        page_count=ingest_result.page_count,
    )


def _git_commit() -> tuple[str, bool]:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        sha = sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        dirty = bool(dirty_result.stdout.strip()) if dirty_result.returncode == 0 else False
        return sha, dirty
    except Exception:
        return ("unknown", False)


def _tool_versions() -> dict:
    versions: dict = {}
    try:
        versions["surya_ocr"] = importlib.metadata.version("surya-ocr")
    except Exception:
        versions["surya_ocr"] = "unknown"
    try:
        versions["python"] = platform.python_version()
    except Exception:
        versions["python"] = "unknown"
    return versions


def _write_manifest(
    run_dir: Path,
    engine: str,
    with_correction: bool,
    dataset_counts: list[tuple[str, Path, int]],
    aggregates: dict,
    with_preprocess: bool = False,
) -> None:
    try:
        sha, dirty = _git_commit()
        versions = _tool_versions()
        total_images = sum(c for _, _, c in dataset_counts)
        manifest = {
            "run_id": run_dir.name,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "engine": engine,
            "correction": with_correction,
            "preprocessing": "full PreprocessConfig" if with_preprocess else "none (raw render)",
            "git_commit": sha,
            "git_dirty": dirty,
            "versions": versions,
            "datasets": [
                {"name": name, "path": str(path), "images": count}
                for name, path, count in dataset_counts
            ],
            "image_count": total_images,
            "aggregates": aggregates,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[WARN] Could not write manifest.json: {exc}")


def run_benchmark(
    data_dirs: list[Path],
    run_dir: Path | None = None,
    with_correction: bool = False,
    resume: bool = False,
    use_qwen: bool = False,
    with_preprocess: bool = False,
) -> None:
    engine = _engine_name()
    if run_dir is None:
        run_dir = _default_run_dir(engine, datetime.now())

    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"

    done = _done_keys(results_csv) if resume and results_csv.exists() else set()
    append_mode = resume and results_csv.exists() and bool(done)

    file_mode = "a" if append_mode else "w"

    processed = 0
    ok_count = 0

    # per-dataset image counts: name -> (path, count)
    ds_counts: dict[str, tuple[Path, int]] = {}

    with results_csv.open(file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not append_mode:
            writer.writeheader()
            f.flush()

        for data_dir in data_dirs:
            dataset_name = data_dir.name
            if dataset_name not in ds_counts:
                ds_counts[dataset_name] = (data_dir, 0)

            gt_files = sorted(data_dir.glob("*_ground_truth.json"))
            if not gt_files:
                print(f"No ground truth files found in {data_dir}")
                continue

            for gt_path in gt_files:
                img_path = gt_path.with_name(gt_path.name.replace("_ground_truth.json", ".png"))
                if not img_path.exists():
                    print(f"[SKIP] No image found for {gt_path.name}")
                    continue

                if (dataset_name, img_path.name) in done:
                    print(f"[RESUME SKIP] {img_path.name}")
                    continue

                # Defaults so a malformed GT file yields an Error row (with blank
                # font/template) instead of aborting the whole run — the GT parse
                # now happens inside the per-image try below.
                font = ""
                template = ""
                tables_expected = 1

                try:
                    gt = json.loads(gt_path.read_text(encoding="utf-8"))
                    font = gt.get("font_family", "")
                    template = gt.get("template", "")
                    # tables_expected: docs schema has a list; isolated has one table
                    tables_expected = len(gt["tables"]) if "tables" in gt else 1

                    img_bytes = img_path.read_bytes()
                    ingest_result = ingest(img_bytes, img_path.name, dpi=200)
                    pre = (preprocess(ingest_result, PreprocessConfig())
                           if with_preprocess else _raw_preprocess_result(ingest_result))
                    ocr_result = ACTIVE_OCR_ENGINE(pre)

                    if with_correction:
                        corrected_result = ACTIVE_CORRECTION_ENGINE(ocr_result, skip_qwen=not use_qwen)
                        # tables are unchanged by correction; use corrected page text for text metrics
                        pred_tables = [t for page in ocr_result.pages for t in page.tables]
                        ocr_text = "\n".join(p.corrected_text for p in corrected_result.pages)
                    else:
                        pred_tables = [t for page in ocr_result.pages for t in page.tables]
                        ocr_text = "\n".join(p.ocr_text for p in ocr_result.pages)

                    table_metrics = evaluate_table(pred_tables, gt_table_grid(gt))
                    text_metrics = evaluate_text(ocr_text, pred_tables, gt)
                    doc_metrics = evaluate_document(ocr_text, pred_tables, gt)

                    row = {
                        "Engine": engine,
                        "Corrected": with_correction,
                        "Dataset": dataset_name,
                        "Image_File": img_path.name,
                        "Font": font,
                        "Template": template,
                        "Tables_Expected": tables_expected,
                        "Tables_Found": table_metrics["tables_found"],
                        "GT_Rows": table_metrics["gt_rows"],
                        "GT_Cols": table_metrics["gt_cols"],
                        "Pred_Rows": table_metrics["pred_rows"],
                        "Pred_Cols": table_metrics["pred_cols"],
                        "Cell_Accuracy": _fmt(table_metrics["cell_accuracy"]),
                        "Cell_Content_Recall": _fmt(table_metrics["cell_content_recall"]),
                        "Table_CER": _fmt(table_metrics["table_cer"]),
                        "Numeric_Cell_Accuracy": _fmt(table_metrics["numeric_cell_accuracy"]),
                        "Numeric_Khmer_Digit_Slips": _fmt(table_metrics["numeric_cells_khmer_digit_slips"]),
                        "Empty_Cell_Precision": _fmt(table_metrics.get("empty_cell_precision")),
                        "Text_CER": _fmt(text_metrics["text_cer"]),
                        "Document_CER": _fmt(doc_metrics["document_cer"]),
                        "Paragraph_Recall": _fmt(text_metrics["paragraph_recall"]),
                        "Paragraph_Leak": _fmt(text_metrics["paragraph_leak"]),
                        "Error": "",
                    }
                    print(f"[OK] {img_path.name}  cell_acc={row['Cell_Accuracy']}")
                    ok_count += 1

                    try:
                        dump = (
                            f"=== {img_path.name} ===\n"
                            f"Document_CER: {row['Document_CER']}\n"
                            f"\n--- GROUND TRUTH (pooled) ---\n"
                            f"{pool_gt_text(gt)}\n"
                            f"\n--- OCR PREDICTION (pooled) ---\n"
                            f"{pool_pred_text(ocr_text, pred_tables)}\n"
                        )
                        (pred_dir / f"{img_path.stem}.txt").write_text(dump, encoding="utf-8")
                    except Exception as exc:
                        print(f"[WARN] Could not write prediction dump for {img_path.name}: {exc}")

                except Exception as exc:
                    print(f"[ERROR] {img_path.name}: {exc}")
                    row = {
                        "Engine": engine,
                        "Corrected": with_correction,
                        "Dataset": dataset_name,
                        "Image_File": img_path.name,
                        "Font": font,
                        "Template": template,
                        "Tables_Expected": tables_expected,
                        "Tables_Found": "",
                        "GT_Rows": "", "GT_Cols": "", "Pred_Rows": "", "Pred_Cols": "",
                        "Cell_Accuracy": "", "Cell_Content_Recall": "", "Table_CER": "",
                        "Numeric_Cell_Accuracy": "", "Numeric_Khmer_Digit_Slips": "", "Empty_Cell_Precision": "",
                        "Text_CER": "", "Document_CER": "", "Paragraph_Recall": "", "Paragraph_Leak": "",
                        "Error": str(exc),
                    }

                writer.writerow(row)
                f.flush()
                processed += 1
                # increment per-dataset count
                ds_counts[dataset_name] = (ds_counts[dataset_name][0], ds_counts[dataset_name][1] + 1)
                clear_device_cache()

    if processed == 0 and not done:
        print("No images processed.")
        return

    # Recompute aggregates, per-dataset counts, and the summary from the FULL
    # results.csv (every row, not just this session's), so --resume writes correct
    # manifest.json / summary.txt instead of numbers for only the resumed images.
    full_rows = _read_all_rows(results_csv)

    def _avg_metric(csv_key: str) -> float:
        vals = [float(r[csv_key]) for r in full_rows if r.get(csv_key)]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    aggregates = {
        "avg_cell_accuracy": _avg_metric("Cell_Accuracy"),
        "avg_cell_content_recall": _avg_metric("Cell_Content_Recall"),
        "avg_table_cer": _avg_metric("Table_CER"),
        "avg_numeric_cell_accuracy": _avg_metric("Numeric_Cell_Accuracy"),
        "avg_empty_cell_precision": _avg_metric("Empty_Cell_Precision"),
        "avg_text_cer": _avg_metric("Text_CER"),
        "avg_document_cer": _avg_metric("Document_CER"),
    }

    full_counts: dict[str, int] = {}
    for r in full_rows:
        ds = r.get("Dataset", "")
        full_counts[ds] = full_counts.get(ds, 0) + 1
    dataset_counts = [
        (name, path, full_counts.get(name, count))
        for name, (path, count) in ds_counts.items()
    ]

    _write_manifest(run_dir, engine, with_correction, dataset_counts, aggregates, with_preprocess)

    # write summary.txt from analyze_benchmark.summarize (over the full CSV)
    summary_text = summarize(full_rows)
    (run_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    print(
        f"\nBenchmark complete. Processed {processed} images "
        f"({ok_count} ok). "
        f"Avg Cell_Accuracy: {aggregates['avg_cell_accuracy']:.3f} | "
        f"Avg Table_CER: {aggregates['avg_table_cer']:.3f}"
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR benchmark on synthetic data.")
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=_DEFAULT_DATASETS,
        type=Path,
    )
    parser.add_argument("--run-dir", default=None, type=Path)
    parser.add_argument("--with-correction", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--qwen", action="store_true", default=False, dest="use_qwen",
                        help="With --with-correction, also run the slow Qwen LLM pass "
                             "(default: deterministic normalizer only).")
    parser.add_argument("--preprocess", action="store_true", default=False, dest="with_preprocess",
                        help="Apply the full preprocessing stack (deskew/denoise/contrast/etc.) "
                             "instead of the raw render (default: raw, to isolate OCR quality).")
    args = parser.parse_args()

    run_benchmark(
        args.data_dir,
        run_dir=args.run_dir,
        with_correction=args.with_correction,
        resume=args.resume,
        use_qwen=args.use_qwen,
        with_preprocess=args.with_preprocess,
    )
