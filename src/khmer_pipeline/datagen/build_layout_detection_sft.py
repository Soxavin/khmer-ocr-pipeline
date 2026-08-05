"""Build ARDB page-layout detection SFT pairs from the existing ardb-layout-coco dataset.

Converts each page's human-corrected COCO boxes (Table/Text/Section-Header/Page-Furniture/
Picture) into Gemma's native detection output format — a JSON list of
{"box_2d": [y1, x1, y2, x2], "label": category}, coordinates normalized to a 0-1000 grid
regardless of the image's actual pixel size (this is Gemma's own documented convention, not
an invented one). Paired with khmer_pipeline.datagen.build_ardb_template_sft's table-
transcription pairs, this lets one Gemma fine-tune learn both page layout and table
character recognition from the same corpus.

**2026-07-27 reservation**: matches build_ardb_template_sft.py's document scope exactly —
the frozen 09.06.26/15.06.26 eval documents (also present in ardb-layout-coco, which has
looser eval-integrity needs than this) are excluded here too, so this model's training
never includes them, consistent with the reservation in that module's docstring.

CLI:
    python -m khmer_pipeline.datagen.build_layout_detection_sft \
        eval/datasets/ardb_layout_coco_v1_hf --out eval/datasets/ardb_layout_detection_sft_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harvest_table_gt import _DEFAULT_EXCLUDE_STEMS

_INSTRUCTION = (
    "Detect the layout regions in this page. Output a JSON list of objects, each "
    '{"box_2d": [y1, x1, y2, x2], "label": category}, with coordinates normalized to a '
    "0-1000 grid regardless of the image's actual size. Categories: Table, Text, "
    "Section-Header, Page-Furniture, Picture."
)


def coco_box_to_gemma(bbox_xywh: list[float], width: int, height: int) -> list[int]:
    """COCO [x, y, w, h] pixels -> Gemma's [y1, x1, y2, x2] normalized to 0-1000."""
    x, y, w, h = bbox_xywh
    return [
        round(y / height * 1000), round(x / width * 1000),
        round((y + h) / height * 1000), round((x + w) / width * 1000),
    ]


def convert_row(row: dict) -> str:
    """Build one page's Gemma detection target JSON from its COCO objects struct."""
    boxes = [
        {"box_2d": coco_box_to_gemma(bbox, row["width"], row["height"]), "label": cat}
        for cat, bbox in zip(row["objects"]["category"], row["objects"]["bbox"])
    ]
    return json.dumps(boxes, ensure_ascii=False)


def build(coco_hf_dir: Path, out_dir: Path,
         exclude_stems: list[str] | None = None) -> dict[str, int]:
    from datasets import load_dataset

    if exclude_stems is None:
        exclude_stems = _DEFAULT_EXCLUDE_STEMS
    ds = load_dataset(str(coco_hf_dir / "data"))
    counts: dict[str, int] = {}
    for split in ds:
        split_dir = out_dir / split
        rows: list[str] = []
        n = 0
        for row in ds[split]:
            source = Path(row["source"]).name
            if any(s in source for s in exclude_stems):
                continue
            split_dir.mkdir(parents=True, exist_ok=True)
            name = row["file_name"]
            row["image"].save(split_dir / name)
            rows.append(json.dumps({
                "image": name, "instruction": _INSTRUCTION, "text": convert_row(row),
                "doc_id": row["doc_id"], "source": source,
            }, ensure_ascii=False))
            n += 1
        if rows:
            (split_dir / "pairs.jsonl").write_text("\n".join(rows) + "\n")
        counts[split] = n
        print(f"{split}: {n} pages")
    print(f"Done: {counts} -> {out_dir}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ardb-layout-coco boxes to Gemma detection SFT pairs.")
    parser.add_argument("coco_hf_dir", type=Path, help="Packaged ardb-layout-coco HF folder")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    parser.add_argument("--exclude-stems", nargs="+", default=_DEFAULT_EXCLUDE_STEMS,
                        help="Skip docs whose source filename contains any of these")
    args = parser.parse_args()
    build(args.coco_hf_dir, args.out, exclude_stems=args.exclude_stems)


if __name__ == "__main__":
    main()
