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
from datetime import date
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
# 1 day, not wider: ARDB bulletins repeat most content day-to-day, so any window merges
# genuinely adjacent-day near-duplicates -- but on this corpus's densely-dated June-July
# 2026 stretch, a 3-day window chains transitively into one ~20-doc supercluster (all
# landing in train), starving valid/test of that whole sub-era. 1 day still closes every
# adjacent-day leak found in the corpus while keeping clusters small enough for valid/test
# to retain real representation.
_DATE_CLUSTER_WINDOW_DAYS = 1


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


def _cluster_by_date(doc_dates: dict[str, date | None], window_days: int) -> list[list[str]]:
    """Group document names whose dates fall within window_days of a neighbor into one
    cluster (transitive: A-B close and B-C close puts all three in one cluster). A
    document with no parseable date becomes its own singleton cluster -- undated docs
    are never merged, since we have no evidence they're near-duplicates of anything."""
    dated = sorted(((d, name) for name, d in doc_dates.items() if d is not None))
    undated = [name for name, d in doc_dates.items() if d is None]

    clusters: list[list[str]] = [[name] for name in undated]
    cluster_by_date: list[tuple[date, list[str]]] = []
    for d, name in dated:
        if cluster_by_date and (d - cluster_by_date[-1][0]).days <= window_days:
            cluster_by_date[-1] = (d, cluster_by_date[-1][1] + [name])
        else:
            cluster_by_date.append((d, [name]))
    clusters.extend(members for _, members in cluster_by_date)
    return clusters


def assign_splits_by_date_cluster(doc_dates: dict[str, date | None], seed: int = 0,
                                  window_days: int = _DATE_CLUSTER_WINDOW_DAYS) -> dict[str, str]:
    """Like assign_splits, but groups documents dated within window_days of each other
    into one cluster before splitting, so no cluster straddles train/valid/test.

    ARDB bulletins repeat most of their content day-to-day (same items/layout, prices
    usually unchanged or barely moved) -- assign_splits' plain per-document shuffle can
    place two near-identical adjacent-day bulletins in different splits, letting a
    validation/test row leak content the model already saw in training. Reuses
    assign_splits as the underlying split-assignment primitive, applied to cluster
    representatives, then expands each cluster's assignment to all its members --
    doesn't change assign_splits' own behavior or its other callers.
    """
    clusters = _cluster_by_date(doc_dates, window_days)
    representatives = [cluster[0] for cluster in clusters]
    rep_splits = assign_splits(representatives, seed=seed)
    splits: dict[str, str] = {}
    for cluster in clusters:
        split = rep_splits[cluster[0]]
        for name in cluster:
            splits[name] = split
    return splits


def assign_splits_by_date_cluster_stratified(
    doc_dates: dict[str, date | None],
    doc_eras: dict[str, str],
    seed: int = 0,
    window_days: int = _DATE_CLUSTER_WINDOW_DAYS,
) -> dict[str, str]:
    """Like assign_splits_by_date_cluster, but stratifies by doc_eras so every era with
    enough clusters gets its own valid/test representation instead of leaving it to
    chance -- a corpus-wide shuffle can (and did, on this corpus) place an entire
    minority era's clusters disproportionately in one split by luck of the seed.

    Clusters first (same adjacent-day leakage protection as assign_splits_by_date_cluster),
    then splits each era's clusters independently via assign_splits, then merges. A
    cluster whose members disagree on era raises ValueError -- era is a slow-moving axis
    (template changes happen roughly yearly) vs. daily bulletins, so a mixed-era cluster
    means date parsing or era classification is wrong upstream and should be surfaced,
    not silently resolved.

    Small-era floor: an era with only 1 cluster stays entirely in train (nothing to
    spare). An era with >=2 clusters gets at least 1 forced into valid if assign_splits'
    rounding would otherwise give it none. An era with >=3 clusters additionally gets at
    least 1 forced into test under the same condition. This means _SPLIT_FRACTIONS isn't
    strictly honored for very small eras -- a deliberate trade favoring guaranteed era
    coverage in valid/test over exact proportionality.
    """
    clusters = _cluster_by_date(doc_dates, window_days)

    clusters_by_era: dict[str, list[list[str]]] = {}
    for cluster in clusters:
        eras_in_cluster = {doc_eras[name] for name in cluster}
        if len(eras_in_cluster) != 1:
            raise ValueError(f"Cluster {cluster} spans multiple eras: {eras_in_cluster}")
        clusters_by_era.setdefault(eras_in_cluster.pop(), []).append(cluster)

    splits: dict[str, str] = {}
    for era_clusters in clusters_by_era.values():
        era_clusters = sorted(era_clusters, key=lambda c: c[0])
        representatives = [cluster[0] for cluster in era_clusters]
        rep_splits = assign_splits(representatives, seed=seed)

        if len(era_clusters) >= 2 and "valid" not in rep_splits.values():
            rep_splits[representatives[0]] = "valid"
        if len(era_clusters) >= 3 and "test" not in rep_splits.values():
            fallback = next(r for r in representatives if rep_splits[r] == "train")
            rep_splits[fallback] = "test"

        for cluster in era_clusters:
            split = rep_splits[cluster[0]]
            for name in cluster:
                splits[name] = split
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


def _pdf_date(pdf_path: Path) -> date | None:
    """Best-effort document date for split clustering: try the filename convention
    first (cheap, no PDF parsing), then the table's own header dates on page 0 via
    find_tables(), then extract_row_blocks() as a second fallback -- find_tables() is
    confirmed to fragment several documents into multiple table objects (same issue
    build_document works around), which can push the header date row out of the first
    2 rows of the first fragment. Returns None (never merged into a cluster) only if
    none of the three sources parse."""
    import fitz

    # Lazy import: build_ardb_template_sft imports assign_splits from this module, so a
    # module-level import here would be circular.
    from .build_ardb_template_sft import (
        _extract_dates,
        extract_row_blocks,
        parse_filename_date,
        parse_header_date,
    )

    parsed = parse_filename_date(pdf_path.name)
    if parsed is not None:
        day, month, year = parsed
        return date(2000 + year, month, day)
    try:
        with fitz.open(str(pdf_path)) as doc:
            page = doc[0]
            tables = page.find_tables().tables
            dates = _extract_dates(tables[0].extract()[:2]) if tables else []
            parsed = parse_header_date(dates)
            if parsed is None:
                parsed = parse_header_date(_extract_dates(extract_row_blocks(page)[:6]))
    except Exception:
        return None
    if parsed is None:
        return None
    day, month, year = parsed
    return date(2000 + year, month, day)


def pseudo_label_corpus(corpus_dir: Path, out_dir: Path, min_conf: float = 0.5,
                        seed: int = 0, dpi: int = 200) -> dict:
    """Run the full pseudo-labeling pass; returns the manifest dict (also written to disk)."""
    from ..ingest import ingest
    from ..engines.surya import _get_predictors

    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found under {corpus_dir}")
    doc_dates = {p.name: _pdf_date(p) for p in pdfs}
    splits = assign_splits_by_date_cluster(doc_dates, seed=seed)
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
