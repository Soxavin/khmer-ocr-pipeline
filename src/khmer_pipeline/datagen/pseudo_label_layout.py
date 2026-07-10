"""Surya layout pseudo-labeler: corpus PDFs → COCO detection dataset for Roboflow/YOLO.

Renders each page via ingest(), runs Surya layout detection, maps Surya labels to a
minimal class set, and writes Roboflow-style split folders (train/valid/test, each
with page PNGs + _annotations.coco.json). Split is BY DOCUMENT to avoid
near-duplicate page leakage.

CLI:
    python -m khmer_pipeline.datagen.pseudo_label_layout corpus/ --out eval/datasets/layout_v1 \
        [--min-conf 0.5] [--seed 0] [--dpi 200]
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

CLASS_NAMES = ["Table", "Text", "Section-Header", "Page-Furniture", "Picture"]

# Surya layout label → our minimal class set. Fewer classes = more labels per class,
# which matters at ~100 pages. Unknown labels fold to Text (with a warning at runtime).
_LABEL_MAP = {
    "Table": "Table",
    "Text": "Text",
    "ListItem": "Text",
    "Caption": "Text",
    "Footnote": "Text",
    "Formula": "Text",
    "Form": "Text",
    "Handwriting": "Text",
    "TableOfContents": "Text",
    "Code": "Text",
    "SectionHeader": "Section-Header",
    "Title": "Section-Header",
    "PageHeader": "Page-Furniture",
    "PageFooter": "Page-Furniture",
    "Picture": "Picture",
    "Figure": "Picture",
}

_SPLIT_FRACTIONS = {"valid": 0.1, "test": 0.1}  # remainder → train


@dataclass
class PageBoxes:
    """One page's pseudo-labels: image file name, pixel dims, and (class, xyxy bbox, conf) boxes."""

    image_name: str
    width: int
    height: int
    boxes: list[tuple[str, tuple[float, float, float, float], float]]


def map_surya_label(label: str) -> str:
    """Map a Surya layout label to one of CLASS_NAMES; unknown labels fold to Text."""
    mapped = _LABEL_MAP.get(label)
    if mapped is None:
        warnings.warn(f"Unknown Surya layout label {label!r}; folding to Text.")
        return "Text"
    return mapped


def assign_splits(doc_names: list[str], seed: int = 0) -> dict[str, str]:
    """Deterministically assign each document to train/valid/test (order-independent).

    Splits by DOCUMENT so near-duplicate pages of one doc never straddle splits.
    Small corpora degrade gracefully: test/valid counts round down, so with very
    few docs everything lands in train."""
    ordered = sorted(set(doc_names))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_valid = int(n * _SPLIT_FRACTIONS["valid"])
    n_test = int(n * _SPLIT_FRACTIONS["test"])
    splits: dict[str, str] = {}
    for i, name in enumerate(ordered):
        if i < n_valid:
            splits[name] = "valid"
        elif i < n_valid + n_test:
            splits[name] = "test"
        else:
            splits[name] = "train"
    return splits


def write_coco(pages: list[PageBoxes], out_path: Path) -> None:
    """Write pages' boxes as a COCO detection JSON (bbox in [x, y, w, h])."""
    categories = [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
    cat_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    images = []
    annotations = []
    ann_id = 0
    for img_id, page in enumerate(pages):
        images.append({
            "id": img_id,
            "file_name": page.image_name,
            "width": page.width,
            "height": page.height,
        })
        for cls, (x0, y0, x1, y1), conf in page.boxes:
            w, h = x1 - x0, y1 - y0
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id[cls],
                "bbox": [x0, y0, w, h],
                "area": w * h,
                "iscrowd": 0,
                "score": conf,  # pseudo-label confidence; ignored by trainers, useful for QA
            })
            ann_id += 1
    out_path.write_text(json.dumps(
        {"images": images, "annotations": annotations, "categories": categories},
        ensure_ascii=False,
    ))


def pseudo_label_corpus(corpus_dir: Path, out_dir: Path, min_conf: float = 0.5,
                        seed: int = 0, dpi: int = 200) -> dict:
    """Run the full pseudo-labeling pass; returns the manifest dict (also written to disk)."""
    from ..ingest import ingest
    from ..engines.surya import _get_predictors

    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found under {corpus_dir}")
    splits = assign_splits([p.name for p in pdfs], seed=seed)
    layout_pred, _ = _get_predictors()

    pages_by_split: dict[str, list[PageBoxes]] = {"train": [], "valid": [], "test": []}
    manifest: dict = {"seed": seed, "min_conf": min_conf, "dpi": dpi, "documents": []}

    for doc_idx, pdf in enumerate(pdfs):
        split = splits[pdf.name]
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        result = ingest(pdf.read_bytes(), pdf.name, dpi=dpi)
        doc_id = f"doc_{doc_idx:03d}"
        page_names = []
        for page_index, page_image in enumerate(result.page_images):
            pil_img = Image.fromarray(page_image)
            image_name = f"{doc_id}_p{page_index}.png"
            pil_img.save(split_dir / image_name)
            layout_result = layout_pred([pil_img])[0]
            boxes = []
            if not layout_result.error:
                for b in layout_result.bboxes:
                    conf = float(getattr(b, "confidence", None) or 1.0)
                    if conf < min_conf:
                        continue
                    x0, y0, x1, y1 = (float(v) for v in b.bbox)
                    boxes.append((map_surya_label(b.label), (x0, y0, x1, y1), conf))
            else:
                warnings.warn(f"{pdf.name} p{page_index}: layout failed; page kept with 0 boxes.")
            pages_by_split[split].append(
                PageBoxes(image_name, pil_img.width, pil_img.height, boxes))
            page_names.append(image_name)
        manifest["documents"].append(
            {"doc_id": doc_id, "source": str(pdf.relative_to(corpus_dir)),
             "split": split, "pages": page_names})
        print(f"[{doc_idx + 1}/{len(pdfs)}] {pdf.name} → {split} ({len(page_names)} pages)")

    for split, pages in pages_by_split.items():
        if pages:
            write_coco(pages, out_dir / split / "_annotations.coco.json")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    counts = {s: len(p) for s, p in pages_by_split.items()}
    print(f"Done: {counts} pages → {out_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Surya layout pseudo-labeler → COCO dataset.")
    parser.add_argument("corpus", type=Path, help="Folder of source PDFs (scanned recursively)")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="Drop layout boxes below this confidence (default 0.5)")
    parser.add_argument("--seed", type=int, default=0, help="Split-assignment seed (default 0)")
    parser.add_argument("--dpi", type=int, default=200, help="Page render DPI (default 200)")
    args = parser.parse_args()
    pseudo_label_corpus(args.corpus, args.out, min_conf=args.min_conf,
                        seed=args.seed, dpi=args.dpi)


if __name__ == "__main__":
    main()
