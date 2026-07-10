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

from khmer_pipeline.datagen.pseudo_label_layout import PageBoxes, write_coco

_SPLITS = ("train", "valid", "test")
_DOC_ID_RE = re.compile(r"^(doc_\d+)_p\d+")


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
            sources: list[str] | None) -> dict[str, int]:
    """Convert the YOLOv8 export to per-split COCO folders under out_dir; returns page counts."""
    names = _class_names(export_dir / "data.yaml")
    keep_doc_ids: set[str] | None = None
    if sources:
        if manifest_path is None:
            raise ValueError("--sources requires --manifest to map doc_ids to corpus subfolders")
        manifest = json.loads(manifest_path.read_text())
        keep_doc_ids = {d["doc_id"] for d in manifest["documents"]
                        if d["source"].split("/")[0] in sources}

    counts: dict[str, int] = {}
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
        counts[split] = len(pages)
        print(f"{split}: {len(pages)} pages, {sum(len(p.boxes) for p in pages)} boxes")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboflow YOLOv8 export → filtered COCO dataset.")
    parser.add_argument("export_dir", type=Path, help="Roboflow YOLOv8 export folder (has data.yaml)")
    parser.add_argument("--out", type=Path, required=True, help="Output COCO dataset folder")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="pseudo_label_layout manifest.json (needed for --sources)")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Keep only docs from these corpus subfolders (e.g. ardb_daily)")
    args = parser.parse_args()
    package(args.export_dir, args.out, args.manifest, args.sources)


if __name__ == "__main__":
    main()
