from __future__ import annotations
import csv
import json
import os
import sys
import argparse
from pathlib import Path

from .ingest import ingest
from .preprocess import preprocess, PreprocessConfig
from .engine_registry import ACTIVE_OCR_ENGINE, ACTIVE_CORRECTION_ENGINE
from .evaluate_judge import evaluate_ocr_quality
from .memory import clear_device_cache


def run_benchmark(data_dir: Path, output_csv: Path) -> None:
    gt_files = sorted(data_dir.glob("*_ground_truth.json"))
    if not gt_files:
        print(f"No ground truth files found in {data_dir}")
        return

    results = []
    for gt_path in gt_files:
        img_path = gt_path.with_name(gt_path.name.replace("_ground_truth.json", ".png"))
        if not img_path.exists():
            print(f"[SKIP] No image found for {gt_path.name}")
            continue

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        font = gt.get("font_family", "")
        template = gt.get("template", "")

        try:
            img_bytes = img_path.read_bytes()
            ingest_result = ingest(img_bytes, img_path.name, dpi=200)
            preprocess_result = preprocess(
                ingest_result,
                PreprocessConfig(
                    remove_stamps=False,  # blue synthetic headers get erased by stamp removal
                    sharpen=True,
                    normalise=True,
                    deskew=True,
                    normalise_table_backgrounds=True,
                ),
            )
            ocr_result = ACTIVE_OCR_ENGINE(preprocess_result)
            corrected_result = ACTIVE_CORRECTION_ENGINE(ocr_result)
            combined_text = "\n".join(p.corrected_text for p in corrected_result.pages)
            scores = evaluate_ocr_quality(str(img_path), combined_text)
        except Exception as exc:
            print(f"[ERROR] {img_path.name}: {exc}")
            scores = {
                "overall_score": 0,
                "estimated_cer_percent": 100,
                "hallucinated_words": [],
                "omitted_words": [],
                "reasoning": f"Pipeline error: {exc}",
            }

        results.append({
            "Image_File": img_path.name,
            "Font": font,
            "Template": template,
            "Overall_Score": scores["overall_score"],
            "Estimated_CER": scores["estimated_cer_percent"],
            "Hallucinations_Count": len(scores["hallucinated_words"]),
            "Omissions_Count": len(scores["omitted_words"]),
            "Reasoning": scores["reasoning"],
        })
        print(f"[OK] {img_path.name}  score={scores['overall_score']}")
        clear_device_cache()

    if not results:
        print("No images processed.")
        return

    fieldnames = [
        "Image_File", "Font", "Template", "Overall_Score",
        "Estimated_CER", "Hallucinations_Count", "Omissions_Count", "Reasoning",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    avg_score = sum(r["Overall_Score"] for r in results) / len(results)
    print(f"\nBenchmark complete. Processed {len(results)} images. Average Score: {avg_score:.1f}%")
    print(f"Results written to: {output_csv}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. The LLM judge requires it.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run automated OCR benchmark on synthetic data.")
    parser.add_argument("--data-dir", default="./synthetic_data", type=Path)
    parser.add_argument("--output-csv", default="./benchmark_results.csv", type=Path)
    args = parser.parse_args()

    run_benchmark(args.data_dir, args.output_csv)
