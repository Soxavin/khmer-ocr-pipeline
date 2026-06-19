from __future__ import annotations
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

from .ingest import ingest
from .models import PreprocessResult
from .engine_registry import ACTIVE_OCR_ENGINE, ACTIVE_CORRECTION_ENGINE
from .evaluate_structure import gt_table_grid, evaluate_table, evaluate_text
from .memory import clear_device_cache

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


def _default_output(engine: str, now) -> Path:
    # now: datetime, injected for tests
    return Path(f"benchmark_results_{engine}_{now:%Y%m%d_%H%M%S}.csv")


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


def run_benchmark(
    data_dirs: list[Path],
    output_csv: Path | None = None,
    with_correction: bool = False,
    resume: bool = False,
) -> None:
    engine = _engine_name()
    if output_csv is None:
        output_csv = _default_output(engine, datetime.now())

    done = _done_keys(output_csv) if resume and output_csv.exists() else set()
    append_mode = resume and output_csv.exists() and bool(done)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if append_mode else "w"

    processed = 0
    ok_count = 0
    acc_vals: list[float] = []

    with output_csv.open(file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not append_mode:
            writer.writeheader()
            f.flush()

        for data_dir in data_dirs:
            dataset_name = data_dir.name
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
                    if row["Cell_Accuracy"]:
                        acc_vals.append(float(row["Cell_Accuracy"]))
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
                clear_device_cache()

    if processed == 0 and not done:
        print("No images processed.")
        return

    avg_acc = sum(acc_vals) / len(acc_vals) if acc_vals else 0.0
    print(
        f"\nBenchmark complete. Processed {processed} images "
        f"({ok_count} ok). Avg Cell_Accuracy: {avg_acc:.3f}"
    )
    print(f"Results written to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR benchmark on synthetic data.")
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=[Path("./synthetic_data"), Path("./synthetic_documents")],
        type=Path,
    )
    parser.add_argument("--output-csv", default=None, type=Path)
    parser.add_argument("--with-correction", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    run_benchmark(
        args.data_dir,
        args.output_csv,
        with_correction=args.with_correction,
        resume=args.resume,
    )
