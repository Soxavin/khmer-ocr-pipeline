"""Reassign an already-packaged ARDB layout COCO dataset's train/valid/test split
membership using corrected, date-and-era-aware clustering -- WITHOUT touching any box
annotation content.

Why this exists: ardb-layout-coco-v3 (and ardb-sft-v3, which inherits its splits from
it) got their split assignment from the original pre-Roboflow pseudo-labeling pass,
which used plain assign_splits() (date-blind -- confirmed to let adjacent-day ARDB
bulletins, which repeat most of their content, straddle train/valid/test). v4 fixed
that with date-clustering alone, but a later audit found the two structural templates
(retail_only / wholesale_retail) still ended up skewed across splits (~57% one era in
test vs ~19% in train) purely by chance, since date-clustering has no era concept. v5
adds era stratification on top of date-clustering. In all cases, box annotations come
from a separate, human-corrected Roboflow export and are untouched by any of this --
only which split each document's pages land in changes. This script corrects exactly
that, reusing every box/label/image byte from the source dataset unchanged.

Usage:
    uv run python scripts/reassign_dataset_splits.py eval/datasets/ardb_layout_coco_v4_hf \
        --corpus corpus/ardb_daily --out eval/datasets/ardb_layout_coco_v5 \
        --hf-out-dir eval/datasets/ardb_layout_coco_v5_hf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from khmer_pipeline.datagen.build_ardb_template_sft import classify_era
from khmer_pipeline.datagen.pseudo_label_layout import (
    CLASS_NAMES,
    PageBoxes,
    _pdf_date,
    assign_splits_by_date_cluster_stratified,
    write_coco,
)

_SPLITS = ("train", "valid", "test")
_HF_SPLIT_NAME = {"train": "train", "valid": "validation", "test": "test"}


def reassign(hf_dir: Path, corpus_dir: Path, out_dir: Path, hf_out_dir: Path,
            seed: int = 0) -> dict[str, int]:
    from datasets import Dataset, Features, Image as HFImage, Sequence, Value, load_dataset

    ds = load_dataset(str(hf_dir / "data"))
    all_rows = []
    for split in ds:
        all_rows.extend(ds[split])

    pdf_by_name = {p.name: p for p in corpus_dir.rglob("*.pdf")}
    doc_source: dict[str, str] = {}
    for row in all_rows:
        doc_source.setdefault(row["doc_id"], row["source"])

    doc_dates = {}
    doc_eras = {}
    for doc_id, source in doc_source.items():
        pdf_path = pdf_by_name.get(Path(source).name)
        if pdf_path is None:
            raise ValueError(
                f"Source PDF not found for doc_id={doc_id!r} (source={source!r}) -- "
                "era stratification needs every doc's PDF to classify its template era."
            )
        doc_dates[doc_id] = _pdf_date(pdf_path)
        doc_eras[doc_id] = classify_era(pdf_path)
    new_split_by_doc = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=seed)

    rows_by_split: dict[str, list[dict]] = {s: [] for s in _SPLITS}
    for row in all_rows:
        rows_by_split[new_split_by_doc[row["doc_id"]]].append(row)

    cat_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    features = Features({
        "image": HFImage(),
        "image_id": Value("int64"),
        "file_name": Value("string"),
        "doc_id": Value("string"),
        "source": Value("string"),
        "width": Value("int64"),
        "height": Value("int64"),
        "objects": {
            "id": Sequence(Value("int64")),
            "bbox": Sequence(Sequence(Value("float32"), length=4)),
            "category_id": Sequence(Value("int64")),
            "category": Sequence(Value("string")),
            "area": Sequence(Value("float32")),
            "iscrowd": Sequence(Value("int64")),
            "score": Sequence(Value("float32")),
        },
    })

    counts: dict[str, int] = {}
    data_dir = hf_out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in _SPLITS:
        rows = rows_by_split[split]
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        pages: list[PageBoxes] = []
        hf_rows = []
        ann_id = 0  # unique across the whole split, matching write_coco's own convention --
                    # not reset per page, so it can't collide with an earlier page's IDs
        for image_id, row in enumerate(rows):
            img = row["image"]
            img.save(split_out / row["file_name"])
            boxes = [(cat, tuple(bbox), score) for cat, bbox, score in
                    zip(row["objects"]["category"], row["objects"]["bbox"], row["objects"]["score"])]
            pages.append(PageBoxes(row["file_name"], row["width"], row["height"], boxes))

            ids, bboxes, category_ids, category, areas, iscrowds, scores = [], [], [], [], [], [], []
            for cat, bbox, area, iscrowd, score in zip(
                    row["objects"]["category"], row["objects"]["bbox"], row["objects"]["area"],
                    row["objects"]["iscrowd"], row["objects"]["score"]):
                ids.append(ann_id)
                ann_id += 1
                bboxes.append(bbox)
                category_ids.append(cat_id[cat])
                category.append(cat)
                areas.append(area)
                iscrowds.append(iscrowd)
                scores.append(score)
            hf_rows.append({
                "image": {"path": row["file_name"], "bytes": (split_out / row["file_name"]).read_bytes()},
                "image_id": image_id,
                "file_name": row["file_name"],
                "doc_id": row["doc_id"],
                "source": row["source"],
                "width": row["width"],
                "height": row["height"],
                "objects": {
                    "id": ids, "bbox": bboxes, "category_id": category_ids,
                    "category": category, "area": areas,
                    "iscrowd": iscrowds, "score": scores,
                },
            })

        write_coco(pages, split_out / "_annotations.coco.json")
        hf_split = _HF_SPLIT_NAME[split]
        hf_ds = Dataset.from_list(hf_rows, features=features)
        hf_ds.to_parquet(data_dir / f"{hf_split}-00000-of-00001.parquet")
        counts[split] = len(rows)
        print(f"{split}: {len(rows)} pages, "
             f"{sum(len(p.boxes) for p in pages)} boxes -> {split_out}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reassign a packaged layout COCO dataset's splits (date-aware), boxes untouched.")
    parser.add_argument("src_hf_dir", type=Path, help="Existing packaged HF folder (has data/*.parquet)")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus folder holding the source PDFs")
    parser.add_argument("--out", type=Path, required=True, help="Output COCO dataset folder")
    parser.add_argument("--hf-out-dir", type=Path, required=True, help="Output HF upload folder")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    reassign(args.src_hf_dir, args.corpus, args.out, args.hf_out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
