from __future__ import annotations
import csv
import importlib.metadata
import json
import platform
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .ingest import ingest
from .models import PreprocessResult
from .engine_registry import ACTIVE_OCR_ENGINE, ACTIVE_CORRECTION_ENGINE
from .evaluate_structure import gt_table_grid, evaluate_table, evaluate_text
from .memory import clear_device_cache
from .analyze_benchmark import summarize

_EVAL_ROOT = Path("eval")
_DATASETS_ROOT = _EVAL_ROOT / "datasets"
_RUNS_ROOT = _EVAL_ROOT / "runs"
_DEFAULT_DATASETS = [_DATASETS_ROOT / "synthetic_tables", _DATASETS_ROOT / "synthetic_documents"]

_CSV_FIELDS = [
    "Engine", "Corrected",
    "Dataset", "Image_File", "Font", "Template",
    "Tables_Expected", "Tables_Found",
    "GT_Rows", "GT_Cols", "Pred_Rows", "Pred_Cols",
    "Cell_Accuracy", "Cell_Content_Recall", "Table_CER",
    "Text_CER", "Paragraph_Recall", "Paragraph_Leak",
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
            "preprocessing": "none (raw render)",
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
) -> None:
    engine = _engine_name()
    if run_dir is None:
        run_dir = _default_run_dir(engine, datetime.now())

    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"

    done = _done_keys(results_csv) if resume and results_csv.exists() else set()
    append_mode = resume and results_csv.exists() and bool(done)

    file_mode = "a" if append_mode else "w"

    processed = 0
    ok_count = 0

    # per-dataset image counts: name -> (path, count)
    ds_counts: dict[str, tuple[Path, int]] = {}
    # metric accumulator for aggregates
    metric_vals: dict[str, list[float]] = {
        "cell_accuracy": [],
        "cell_content_recall": [],
        "table_cer": [],
        "text_cer": [],
    }

    all_rows: list[dict] = []

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

                gt = json.loads(gt_path.read_text(encoding="utf-8"))
                font = gt.get("font_family", "")
                template = gt.get("template", "")

                # tables_expected: docs schema has a list; isolated has one table
                if "tables" in gt:
                    tables_expected = len(gt["tables"])
                else:
                    tables_expected = 1

                try:
                    img_bytes = img_path.read_bytes()
                    ingest_result = ingest(img_bytes, img_path.name, dpi=200)
                    pre = _raw_preprocess_result(ingest_result)
                    ocr_result = ACTIVE_OCR_ENGINE(pre)

                    if with_correction:
                        corrected_result = ACTIVE_CORRECTION_ENGINE(ocr_result)
                        # tables are unchanged by correction; use corrected page text for text metrics
                        pred_tables = [t for page in ocr_result.pages for t in page.tables]
                        ocr_text = "\n".join(p.corrected_text for p in corrected_result.pages)
                    else:
                        pred_tables = [t for page in ocr_result.pages for t in page.tables]
                        ocr_text = "\n".join(p.ocr_text for p in ocr_result.pages)

                    table_metrics = evaluate_table(pred_tables, gt_table_grid(gt))
                    text_metrics = evaluate_text(ocr_text, pred_tables, gt)

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
                        "Text_CER": _fmt(text_metrics["text_cer"]),
                        "Paragraph_Recall": _fmt(text_metrics["paragraph_recall"]),
                        "Paragraph_Leak": _fmt(text_metrics["paragraph_leak"]),
                        "Error": "",
                    }
                    print(f"[OK] {img_path.name}  cell_acc={row['Cell_Accuracy']}")
                    ok_count += 1

                    # accumulate metrics for aggregates (skip blank)
                    for csv_key, acc_key in [
                        ("Cell_Accuracy", "cell_accuracy"),
                        ("Cell_Content_Recall", "cell_content_recall"),
                        ("Table_CER", "table_cer"),
                        ("Text_CER", "text_cer"),
                    ]:
                        if row[csv_key]:
                            metric_vals[acc_key].append(float(row[csv_key]))

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
                        "Text_CER": "", "Paragraph_Recall": "", "Paragraph_Leak": "",
                        "Error": str(exc),
                    }

                writer.writerow(row)
                f.flush()
                processed += 1
                all_rows.append(row)
                # increment per-dataset count
                ds_counts[dataset_name] = (ds_counts[dataset_name][0], ds_counts[dataset_name][1] + 1)
                clear_device_cache()

    if processed == 0 and not done:
        print("No images processed.")
        return

    def _avg_metric(key: str) -> float:
        vals = metric_vals[key]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    aggregates = {
        "avg_cell_accuracy": _avg_metric("cell_accuracy"),
        "avg_cell_content_recall": _avg_metric("cell_content_recall"),
        "avg_table_cer": _avg_metric("table_cer"),
        "avg_text_cer": _avg_metric("text_cer"),
    }

    dataset_counts = [
        (name, path, count) for name, (path, count) in ds_counts.items()
    ]

    _write_manifest(run_dir, engine, with_correction, dataset_counts, aggregates)

    # write summary.txt from analyze_benchmark.summarize
    summary_text = summarize(all_rows)
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
    args = parser.parse_args()

    run_benchmark(
        args.data_dir,
        run_dir=args.run_dir,
        with_correction=args.with_correction,
        resume=args.resume,
    )
