"""Package a corrected Roboflow YOLOv8 export as a COCO dataset, optionally source-filtered.

Converts YOLO txt labels back to COCO JSON per split (mentor's training format), keeping
Roboflow's split assignment. --sources filters documents by their corpus subfolder using the
pseudo-labeler's manifest.json (doc_id prefixes survive Roboflow's filename mangling).

Usage:
    uv run python scripts/package_layout_dataset.py eval/datasets/layout_v1_corrected \
        --manifest eval/datasets/layout_v1/manifest.json \
        --out eval/datasets/ardb_layout_coco_v1 --sources ardb_daily
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from khmer_pipeline.datagen.pseudo_label_layout import CLASS_NAMES, PageBoxes, write_coco

_SPLITS = ("train", "valid", "test")
_DOC_ID_RE = re.compile(r"^(doc_\d+)_p\d+")


# HF-viewer split naming: parquet files under data/ named <split>-XXXXX-of-XXXXX.parquet
_HF_SPLIT_NAME = {"train": "train", "valid": "validation", "test": "test"}


def _write_parquet(pages_by_split: dict[str, list[PageBoxes]], out_dir: Path,
                   hf_dir: Path, doc_source: dict[str, str]) -> None:
    """Write the HF upload folder (hanuman-100k-style: just data/*.parquet + README):
    one tabular row per page with the embedded image bytes, provenance columns, and an
    objects struct carrying all COCO fields (bbox [x,y,w,h], category_id/category in
    CLASS_NAMES order, area, iscrowd, score) for the Data Viewer."""
    from datasets import Dataset, Features, Image as HFImage, Sequence, Value

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
    data_dir = hf_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, pages in pages_by_split.items():
        rows = []
        ann_id = 0
        for image_id, page in enumerate(pages):
            doc_match = _DOC_ID_RE.match(page.image_name)
            doc_id = doc_match.group(1) if doc_match else ""
            ids, bboxes, category_ids, category, areas, scores = [], [], [], [], [], []
            for cls, (x0, y0, x1, y1), conf in page.boxes:
                w, h = x1 - x0, y1 - y0
                ids.append(ann_id)
                bboxes.append([x0, y0, w, h])
                category_ids.append(cat_id[cls])
                category.append(cls)
                areas.append(w * h)
                scores.append(conf)
                ann_id += 1
            img_path = out_dir / split / page.image_name
            rows.append({
                # embed the JPG bytes so the parquet is self-contained (path-only rows
                # would upload without the images)
                "image": {"path": page.image_name, "bytes": img_path.read_bytes()},
                "image_id": image_id,
                "file_name": page.image_name,
                "doc_id": doc_id,
                "source": doc_source.get(doc_id, ""),
                "width": page.width,
                "height": page.height,
                "objects": {
                    "id": ids, "bbox": bboxes, "category_id": category_ids,
                    "category": category, "area": areas,
                    "iscrowd": [0] * len(ids), "score": scores,
                },
            })
        ds = Dataset.from_list(rows, features=features)
        hf_split = _HF_SPLIT_NAME[split]
        ds.to_parquet(data_dir / f"{hf_split}-00000-of-00001.parquet")
        print(f"parquet: {hf_split} ({len(rows)} rows)")


def _class_names(data_yaml: Path) -> list[str]:
    """Parse the class-name list out of a Roboflow data.yaml (names: ['a', 'b', ...])."""
    m = re.search(r"names:\s*\[(.*?)\]", data_yaml.read_text())
    if not m:
        raise ValueError(f"No names: [...] list found in {data_yaml}")
    return [n.strip().strip("'\"") for n in m.group(1).split(",")]


def _yolo_to_boxes(label_file: Path, names: list[str],
                   width: int, height: int) -> list[tuple[str, tuple[float, float, float, float], float]]:
    """Convert one YOLO label file (class cx cy w h, normalized) to (class, xyxy, conf) tuples."""
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
        x0, y0 = (cx - w / 2) * width, (cy - h / 2) * height
        x1, y1 = (cx + w / 2) * width, (cy + h / 2) * height
        boxes.append((names[cls_id], (x0, y0, x1, y1), 1.0))  # human-corrected → conf 1.0
    return boxes


def package(export_dir: Path, out_dir: Path, manifest_path: Path | None,
            sources: list[str] | None, hf_dir: Path | None = None) -> dict[str, int]:
    """Convert the YOLOv8 export to per-split COCO folders under out_dir; returns page counts."""
    names = _class_names(export_dir / "data.yaml")
    keep_doc_ids: set[str] | None = None
    doc_source: dict[str, str] = {}
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        doc_source = {d["doc_id"]: d["source"] for d in manifest["documents"]}
        if sources:
            keep_doc_ids = {i for i, src in doc_source.items() if src.split("/")[0] in sources}
    elif sources:
        raise ValueError("--sources requires --manifest to map doc_ids to corpus subfolders")

    counts: dict[str, int] = {}
    pages_by_split: dict[str, list[PageBoxes]] = {}
    for split in _SPLITS:
        images_dir = export_dir / split / "images"
        labels_dir = export_dir / split / "labels"
        if not images_dir.is_dir():
            continue
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        pages: list[PageBoxes] = []
        for img_path in sorted(images_dir.iterdir()):
            doc_match = _DOC_ID_RE.match(img_path.name)
            if keep_doc_ids is not None and (
                    doc_match is None or doc_match.group(1) not in keep_doc_ids):
                continue
            with Image.open(img_path) as im:
                width, height = im.size
            label_file = labels_dir / (img_path.stem + ".txt")
            boxes = _yolo_to_boxes(label_file, names, width, height) if label_file.exists() else []
            shutil.copy2(img_path, split_out / img_path.name)
            pages.append(PageBoxes(img_path.name, width, height, boxes))
        write_coco(pages, split_out / "_annotations.coco.json")
        # remove any stale imagefolder metadata: parquet is the sole viewer/loader source
        (split_out / "metadata.jsonl").unlink(missing_ok=True)
        counts[split] = len(pages)
        pages_by_split[split] = pages
        print(f"{split}: {len(pages)} pages, {sum(len(p.boxes) for p in pages)} boxes")
    _write_parquet(pages_by_split, out_dir, hf_dir or out_dir.parent / (out_dir.name + "_hf"),
                   doc_source)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboflow YOLOv8 export → filtered COCO dataset.")
    parser.add_argument("export_dir", type=Path, help="Roboflow YOLOv8 export folder (has data.yaml)")
    parser.add_argument("--out", type=Path, required=True, help="Output COCO dataset folder")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="pseudo_label_layout manifest.json (needed for --sources)")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Keep only docs from these corpus subfolders (e.g. ardb_daily)")
    parser.add_argument("--hf-dir", type=Path, default=None,
                        help="HF upload folder for data/*.parquet (default: <out>_hf)")
    args = parser.parse_args()
    package(args.export_dir, args.out, args.manifest, args.sources, hf_dir=args.hf_dir)


if __name__ == "__main__":
    main()
