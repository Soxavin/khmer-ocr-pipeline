"""Build ARDB unified page-understanding SFT pairs: one page image -> one JSON list of
{"box_2d", "label", "text"} objects, in a single forward pass.

Supersedes the v1 split (build_ardb_template_sft.py's cropped-region transcription +
build_layout_detection_sft.py's bbox-only layout detection) per mentor feedback: those two
tasks never taught the model to connect "this region" to "this text" from one image, since
transcription rows were cropped using ground-truth boxes rather than the model's own
detections, and layout rows carried no text at all. This module reuses both pipelines'
already-verified logic rather than re-deriving it — box_2d values come straight from
ardb-layout-coco-v2's raw per-page COCO objects (same as build_layout_detection_sft.py's
convert_row), and text values come from build_ardb_template_sft.py's existing template
substitution (Table, Section-Header) and static constants (Page-Furniture, Text). These two
pipelines never needed to touch each other before; this module is the first place they're
combined, zipped by (page, label).

Splits and doc_ids are taken directly from the already-packaged ardb-layout-coco-v2 HF
dataset (not recomputed) so this dataset's held-out documents stay identical to the v1
layout config's — recomputing a fresh split here risked leaking a document that v1 already
committed to train into this dataset's validation set, or vice versa.

CLI:
    python -m khmer_pipeline.datagen.build_ardb_unified_sft corpus/ardb_daily \
        --coco-hf-dir eval/datasets/ardb_layout_coco_v1_hf --out eval/datasets/ardb_unified_sft_v1

**2026-07-27 reservation** (inherited from build_ardb_template_sft.py): every corpus/ardb_daily/
document this module trains on (all except the frozen 09.06.26/15.06.26 stems) must NOT later
be promoted into eval/datasets/real/ by khmer_pipeline.datagen.harvest_eval_gt /
scripts/generate_ardb_eval_gt.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz

from .build_ardb_template_sft import (
    _LETTERHEAD_TEXT,
    _PRICE_COLS_BY_ERA,
    _RETAIL_ONLY_GT_STEM,
    _extract_dates,
    build_page,
    build_title_text,
    classify_era,
    extract_row_blocks,
    load_templates,
    load_title_template,
    parse_filename_date,
    parse_header_date,
)
from .build_layout_detection_sft import coco_box_to_gemma
from .harvest_table_gt import _DEFAULT_EXCLUDE_STEMS, grid_to_html

_FILE_NAME_PAGE_RE = re.compile(r"_p(\d+)_")

# Source note + legend footer for the COCO `Text` category (page-3 only). Static boilerplate,
# character-for-character identical between the two frozen eval documents' verified GT
# (09.06.26 and 15.06.26 page-3 `paragraphs`) -- already human-verified, not re-typed from a
# fresh visual read. Same treatment as _LETTERHEAD_TEXT.
_FOOTER_TEXT = (
    "តម្មៃយកពីប្រភព៖\n"
    "១ \nផ្សារមួយចំនួនក្នុងរាជធានីភ្នំពេញ\n"
    "តួេលខពណ៍លឿងសម្រាប់តម្មៃកសិផលប្រែប្រួល"
)

# Generic, not "this ARDB bulletin page" -- the model has no way to know what "ARDB" means,
# and naming the specific institution in the prompt would make the instruction depend on
# domain knowledge the model doesn't have rather than on what's visually in front of it.
_UNIFIED_INSTRUCTION = (
    "Extract every layout region on this page. Output a JSON list of "
    'objects, each {"box_2d": [y1, x1, y2, x2], "label": category, "text": ...} (text is '
    "the region's transcribed content, or an empty string if it has none, e.g. a photo), "
    "with box_2d normalized to a 0-1000 grid. Categories: Table, Text, Section-Header, "
    "Page-Furniture, Picture."
)


def load_page_regions(coco_hf_dir: Path) -> dict[tuple[str, int], dict]:
    """Group ardb-layout-coco-v2's raw per-page COCO objects by (source, page index).

    Returns dict[(source, page_idx)] -> {"split", "doc_id", "image", "regions"}, where
    "regions" is a list of (label, box_2d) already normalized via coco_box_to_gemma -- the
    same conversion build_layout_detection_sft.py's convert_row applies, just kept as
    structured tuples here instead of immediately serialized, so text can be attached to
    each entry before this module builds its own JSON.
    """
    from datasets import load_dataset

    ds = load_dataset(str(coco_hf_dir / "data"))
    pages: dict[tuple[str, int], dict] = {}
    for split in ds:
        for row in ds[split]:
            m = _FILE_NAME_PAGE_RE.search(row["file_name"])
            if not m:
                continue
            page_idx = int(m.group(1))
            source = Path(row["source"]).name
            regions = [
                (cat, coco_box_to_gemma(bbox, row["width"], row["height"]))
                for cat, bbox in zip(row["objects"]["category"], row["objects"]["bbox"])
            ]
            pages[(source, page_idx)] = {
                "split": split, "doc_id": row["doc_id"], "image": row["image"], "regions": regions,
            }
    return pages


def build_region_texts(regions: list[tuple[str, list[int]]], table_text: str | None,
                       title_text: str | None) -> list[dict] | None:
    """Zip each raw (label, box_2d) with its transcribed text and sort by reading order.

    Quarantines (returns None) the whole page -- never a silently-wrong single field --
    when a Table region has no aligned table_text, a Section-Header region has no resolved
    title_text, or the Page-Furniture count doesn't match _LETTERHEAD_TEXT's line count
    (always expected to be 2; a mismatch means the geometric top/bottom zip can't be
    trusted).
    """
    furniture = sorted([r for r in regions if r[0] == "Page-Furniture"], key=lambda r: r[1][0])
    letterhead_lines = _LETTERHEAD_TEXT.split("\n")
    if furniture and len(furniture) != len(letterhead_lines):
        return None
    furniture_text = {tuple(box): line for (_, box), line in zip(furniture, letterhead_lines)}

    out = []
    for label, box in regions:
        if label == "Table":
            if table_text is None:
                return None
            text = table_text
        elif label == "Section-Header":
            if title_text is None:
                return None
            text = title_text
        elif label == "Page-Furniture":
            text = furniture_text[tuple(box)]
        elif label == "Text":
            text = _FOOTER_TEXT
        else:  # Picture, or any future/unexpected category -- never crash on the unknown
            text = ""
        out.append({"box_2d": box, "label": label, "text": text})
    out.sort(key=lambda r: r["box_2d"][0])
    return out


def build_document(pdf_path: Path, source: str, templates: list[list[list[str]]],
                   title_template: str, pages: dict[int, dict], doc_id: str, hf_split: str,
                   out_dir: Path,
                   price_cols: tuple[int, ...] = (3, 4, 5, 6, 7, 8)) -> dict[str, int]:
    counts = {"pages_ok": 0, "pages_quarantined": 0}
    split_dir = out_dir / hf_split
    rows: list[str] = []
    page0_dates: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_idx in sorted(pages):
            if page_idx >= len(templates) or page_idx >= doc.page_count:
                counts["pages_quarantined"] += 1
                continue
            page = doc[page_idx]
            tables = page.find_tables().tables
            if not tables:
                counts["pages_quarantined"] += 1
                continue
            raw_grid = tables[0].extract()
            final_grid = build_page(templates[page_idx], raw_grid, has_header=(page_idx == 0),
                                    price_cols=price_cols)
            if final_grid is None:
                # Same find_tables()-fragmentation issue confirmed for the older retail_only
                # era in build_ardb_template_sft.py — retry with the row-per-text-block
                # extractor before quarantining the page.
                row_blocks = extract_row_blocks(page)
                template = templates[page_idx]
                body_len = len(template) - 1 if page_idx == 0 else len(template)
                header_rows = max(0, len(row_blocks) - body_len)
                final_grid = build_page(template, row_blocks, has_header=(page_idx == 0),
                                        header_rows=header_rows, price_cols=price_cols)
            table_text = (grid_to_html(final_grid, has_header=(page_idx == 0))
                         if final_grid is not None else None)
            if page_idx == 0 and final_grid is not None:
                page0_dates = _extract_dates([final_grid[0]])

            regions = pages[page_idx]["regions"]
            title_text = None
            if any(label == "Section-Header" for label, _ in regions):
                date_parts = parse_header_date(page0_dates) or parse_filename_date(source)
                if date_parts is not None:
                    title_text = build_title_text(title_template, date_parts[0], date_parts[1])

            final_regions = build_region_texts(regions, table_text, title_text)
            if final_regions is None:
                counts["pages_quarantined"] += 1
                continue

            split_dir.mkdir(parents=True, exist_ok=True)
            name = f"{doc_id}_p{page_idx}.png"
            pages[page_idx]["image"].save(split_dir / name)
            date = (next(iter(_extract_dates([final_grid[0]])), "")
                   if page_idx == 0 and final_grid is not None else "")
            rows.append(json.dumps({
                "image": name, "instruction": _UNIFIED_INSTRUCTION,
                "text": json.dumps(final_regions, ensure_ascii=False),
                "doc_id": doc_id, "source": source, "page": page_idx, "date": date,
            }, ensure_ascii=False))
            counts["pages_ok"] += 1

    if rows:
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "pairs.jsonl").open("a") as f:
            f.write("\n".join(rows) + "\n")
    return counts


def build_corpus(coco_hf_dir: Path, corpus_dir: Path, gt_dir: Path, out_dir: Path,
                 exclude_stems: list[str] | None = None) -> dict[str, int]:
    if exclude_stems is None:
        exclude_stems = _DEFAULT_EXCLUDE_STEMS
    # Era-keyed, mirroring build_ardb_template_sft.py's build_corpus: each document is
    # classified once (via classify_era) and built against its own era's template/title/
    # price-columns, so pre-May-2024 retail_only docs don't get misaligned against the
    # wholesale_retail template.
    templates_by_era = {
        "wholesale_retail": load_templates(gt_dir),
        "retail_only": load_templates(gt_dir, stem=_RETAIL_ONLY_GT_STEM),
    }
    title_template_by_era = {
        "wholesale_retail": load_title_template(gt_dir),
        "retail_only": load_title_template(gt_dir, stem=_RETAIL_ONLY_GT_STEM),
    }
    page_data = load_page_regions(coco_hf_dir)

    by_source: dict[str, dict[int, dict]] = {}
    for (source, page_idx), data in page_data.items():
        by_source.setdefault(source, {})[page_idx] = data

    pdf_by_name = {p.name: p for p in corpus_dir.rglob("*.pdf")}
    totals = {"pages_ok": 0, "pages_quarantined": 0, "skipped_docs": 0}
    for doc_idx, source in enumerate(sorted(by_source)):
        if any(s in source for s in exclude_stems):
            totals["skipped_docs"] += 1
            continue
        pdf_path = pdf_by_name.get(source)
        if pdf_path is None:
            totals["skipped_docs"] += 1
            continue
        pages = by_source[source]
        first_page = pages[min(pages)]
        era = classify_era(pdf_path)
        counts = build_document(pdf_path, source, templates_by_era[era],
                                title_template_by_era[era], pages,
                                first_page["doc_id"], first_page["split"], out_dir,
                                price_cols=_PRICE_COLS_BY_ERA[era])
        for k in totals:
            if k in counts:
                totals[k] += counts[k]
        print(f"[{doc_idx + 1}/{len(by_source)}] {source} -> {first_page['split']} "
              f"({counts['pages_ok']} ok, {counts['pages_quarantined']} quarantined)")
    print(f"Done: {totals} -> {out_dir}")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ARDB unified (bbox + text, one forward pass) SFT pairs.")
    parser.add_argument("corpus", type=Path, help="Folder of ARDB PDFs (scanned recursively)")
    parser.add_argument("--coco-hf-dir", type=Path, required=True,
                        help="Packaged ardb-layout-coco HF folder (boxes + splits + images)")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    parser.add_argument("--gt-dir", type=Path, default=Path("eval/datasets/real"),
                        help="Folder holding the verified 09.06.26 GT used as the template")
    parser.add_argument("--exclude-stems", nargs="+", default=_DEFAULT_EXCLUDE_STEMS,
                        help="Skip docs whose filename contains any of these (eval-GT guard)")
    args = parser.parse_args()
    build_corpus(args.coco_hf_dir, args.corpus, args.gt_dir, args.out,
                exclude_stems=args.exclude_stems)


if __name__ == "__main__":
    main()
