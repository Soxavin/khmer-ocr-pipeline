"""Package an image+instruction+text SFT dataset (pairs.jsonl per split) as a
self-contained HF upload folder, mirroring package_layout_dataset.py's parquet pattern.
Schema-flexible: `page`/`date` columns are included only if present in the source rows,
so this works for both the table-transcription and layout-detection SFT datasets.

Usage:
    uv run python scripts/package_sft_dataset.py eval/datasets/ardb_table_sft_v1 \
        --out eval/datasets/ardb_table_sft_v1_hf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_SPLITS = ("train", "validation", "test")
_CORE_FIELDS = ("instruction", "text", "doc_id", "source")
_OPTIONAL_FIELDS = ("page", "date")  # only table-transcription rows carry these


def package(dataset_dir: Path, hf_dir: Path) -> dict[str, int]:
    from datasets import Dataset, Features, Image as HFImage, Value

    data_dir = hf_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    features: "Features | None" = None
    for split in _SPLITS:
        split_dir = dataset_dir / split
        pairs_path = split_dir / "pairs.jsonl"
        if not pairs_path.is_file():
            continue
        lines = pairs_path.read_text().splitlines()
        if not lines:
            continue
        if features is None:
            first = json.loads(lines[0])
            schema = {"image": HFImage(), **{f: Value("string") for f in _CORE_FIELDS}}
            if "page" in first:
                schema["page"] = Value("int64")
            if "date" in first:
                schema["date"] = Value("string")
            features = Features(schema)

        rows = []
        for line in lines:
            r = json.loads(line)
            img_path = split_dir / r["image"]
            row = {
                # embed the image bytes so the parquet is self-contained (path-only rows
                # would upload without the images)
                "image": {"path": r["image"], "bytes": img_path.read_bytes()},
                **{f: r[f] for f in _CORE_FIELDS},
            }
            for f in _OPTIONAL_FIELDS:
                if f in features:
                    row[f] = r.get(f, "")
            rows.append(row)
        ds = Dataset.from_list(rows, features=features)
        ds.to_parquet(data_dir / f"{split}-00000-of-00001.parquet")
        counts[split] = len(rows)
        print(f"parquet: {split} ({len(rows)} rows)")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Package SFT pairs as an HF upload folder.")
    parser.add_argument("dataset_dir", type=Path,
                        help="Folder holding {train,validation,test}/pairs.jsonl + images")
    parser.add_argument("--out", type=Path, required=True, help="HF upload folder for data/*.parquet")
    args = parser.parse_args()
    package(args.dataset_dir, args.out)


if __name__ == "__main__":
    main()
