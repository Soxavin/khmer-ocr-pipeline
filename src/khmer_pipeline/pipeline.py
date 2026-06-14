"""
Batch processing entrypoint for the Khmer OCR pipeline.
Processes a single PDF or image file without the Streamlit UI.

Usage:
    uv run python -m khmer_pipeline.pipeline input.pdf output/
    uv run python -m khmer_pipeline.pipeline input.pdf output/ --dpi 300 --no-qwen
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .ingest import ingest
from .preprocess import preprocess, PreprocessConfig
from .surya import run_surya
from .postprocess import postprocess
from .export import export


def run(
    source_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    remove_stamps: bool = True,
    sharpen: bool = True,
    normalise: bool = True,
    skip_qwen: bool = False,
    convert_numerals: bool = False,
) -> None:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {source_path.name}...")
    data = source_path.read_bytes()

    ingest_result = ingest(data, source_path.name, dpi=dpi)
    print(f"  Ingested {ingest_result.page_count} page(s)")

    config = PreprocessConfig(remove_stamps=remove_stamps, sharpen=sharpen, normalise=normalise)
    preprocess_result = preprocess(ingest_result, config)
    print(f"  Preprocessing complete")

    surya_result = run_surya(preprocess_result)
    if surya_result.warnings:
        for w in surya_result.warnings:
            print(f"  WARNING: {w}")
    print(f"  OCR complete — {sum(len(p.text_blocks) for p in surya_result.pages)} text blocks")

    postprocess_result = postprocess(surya_result, skip_qwen=skip_qwen)
    print(f"  Post-processing complete")

    export_result = export(postprocess_result, convert_numerals=convert_numerals)

    json_path = output_dir / f"{source_path.stem}_extracted.json"
    json_path.write_text(
        json.dumps(export_result.document_json, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  JSON written to {json_path}")

    for table_id, csv_string in export_result.tables_csv:
        if csv_string.strip().strip("﻿"):
            csv_path = output_dir / f"{table_id}.csv"
            csv_path.write_text(csv_string, encoding="utf-8-sig")
            print(f"  CSV written to {csv_path}")

    print(f"Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khmer OCR Pipeline — batch processor")
    parser.add_argument("input", help="Path to PDF or image file")
    parser.add_argument("output", help="Output directory for CSV and JSON files")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-stamps", action="store_false", dest="remove_stamps")
    parser.add_argument("--no-sharpen", action="store_false", dest="sharpen")
    parser.add_argument("--no-normalise", action="store_false", dest="normalise")
    parser.add_argument("--no-qwen", action="store_true", dest="skip_qwen")
    parser.add_argument("--convert-numerals", action="store_true", dest="convert_numerals")
    args = parser.parse_args()
    run(
        args.input, args.output,
        dpi=args.dpi,
        remove_stamps=args.remove_stamps,
        sharpen=args.sharpen,
        normalise=args.normalise,
        skip_qwen=args.skip_qwen,
        convert_numerals=args.convert_numerals,
    )
