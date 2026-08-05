"""Build ARDB daily-bulletin table-transcription SFT pairs by template substitution.

ARDB daily bulletins share one fixed layout across every issue (same commodities, same
column structure, same section headers) — only the two header dates and the numeric
price/percent cells change per document. Rather than trusting PyMuPDF's raw per-document
table extraction as SFT ground truth (it over-splits merged cells into phantom columns,
and Khmer text occasionally scrambles even in clean born-digital PDFs — see
harvest_table_gt.py), this module verifies the clean structure ONCE from
eval/datasets/real/*_09.06.26_p{1,2,3}_ground_truth.json (already mentor-verified) and,
for every other document, substitutes only the dates and numbers that actually vary. The
frozen eval documents (09.06.26, 15.06.26) are never used as training examples themselves
— only their label *structure* is reused as a template.

CLI:
    python -m khmer_pipeline.datagen.build_ardb_template_sft corpus/ardb_daily \
        --out eval/datasets/ardb_table_sft_v1

**2026-07-27 reservation**: every corpus/ardb_daily/ document this module trains on
(all except the frozen 09.06.26/15.06.26 stems) must NOT later be promoted into
eval/datasets/real/ by khmer_pipeline.datagen.harvest_eval_gt /
scripts/generate_ardb_eval_gt.py — that would let a track's eval score benefit from
pages this model already trained on. Check that reservation still holds before
either script's scope changes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz

from .harvest_table_gt import (
    _DEFAULT_EXCLUDE_STEMS,
    _crop,
    _render_page,
    grid_to_html,
    is_numeric_text,
    passes_numeric_qa,
)
from .pseudo_label_layout import _pdf_date, assign_splits_by_date_cluster_stratified

_DPI = 200
_TEMPLATE_GT_STEM = "តារាងតម្លៃទំនិញតាមទីផ្សារមួយចំនួននៅរាជធានីភ្នំពេញ-ប្រចាំថ្ងៃ-09.06.26"
# Era-keyed templates (2026-07-30): the multi-year probe found only 2 structural table
# layouts across 2022-2026, not 3 — Era A2 (~May-Dec 2024) turned out to share Era B's
# 9-col wholesale+retail structure, just with a different title. "wholesale_retail" is
# the default/majority template (existing 2026 corpus + A2/B); "retail_only" is the
# older 6-col Era A layout (~2022-May 2024).
# Anchored on the verified Feb 2024 GT, NOT the also-verified Aug 2022 GT: both share the
# same ~73-item commodity list/order, but the PDF's own page break falls at a different
# item across issues (probably font/line-wrap variance between PDF-generation instances,
# not a content difference) — Aug 2022 breaks page 1 after item 27, Feb 2024 after item
# 21. Checked page-1 boundaries across all 16 Era A probe/anchor docs: only the two March
# 2022 docs share Aug 2022's boundary; the other 14 (incl. every 2023-2024 doc) match Feb
# 2024's. Since build_page aligns per-page against a fixed row count, the template must
# use the majority boundary or most documents misalign on page count alone, not content.
_RETAIL_ONLY_GT_STEM = "តម្លៃទំនិញមួយចំនួន_នៅរាជធានីភ្នំពេញថ្ងៃទី_០៦_០៧_កុម្ភៈ_២០២៤"
_WHOLESALE_MARKER = "បោះដុំ"  # present in every wholesale+retail-era document, absent from Era A
_TEMPLATE_GT_PAGES = ("p1", "p2", "p3")  # GT page N -> PDF page index N-1
_PRICE_COLS = (3, 4, 5, 6, 7, 8)  # wholesale1, retail1, wholesale2, retail2, %wholesale, %retail
_RETAIL_ONLY_PRICE_COLS = (3, 4, 5)  # Era A's narrower 6-col layout: retail1, retail2, %
_PRICE_COLS_BY_ERA = {"wholesale_retail": _PRICE_COLS, "retail_only": _RETAIL_ONLY_PRICE_COLS}
_DATE_RE = re.compile(r"\d{2}-\d{2}-\d{2}")
# Recurring PDF rendering quirk (not a find_tables()-specific bug — present in the raw
# text itself): on certain rows, the row-number and item-name render without a newline
# between them (e.g. "១៣ មាន់ស្រែ (សាច់)" as one line instead of two), shifting that row's
# cell count. Confirmed on multiple unrelated rows across both the Feb 2024 and Jan 2023
# (probe) documents, so it's a recurring layout fluke, not a one-off typo.
_MERGED_ROW_NUM_RE = re.compile(r"^([០-៩0-9]+)\s+(\S.*)$")
_HF_SPLIT_NAME = {"train": "train", "valid": "validation", "test": "test"}
_FILE_NAME_PAGE_RE = re.compile(r"_p(\d+)_")
_DEFAULT_COCO_HF_DIR = Path("eval/datasets/ardb_layout_coco_v1_hf")

# Month names verified by the user against the rendered page image (2026-07-27) — the
# PDF text layer scrambles these the same way it scrambles table labels, and only June
# (month 6) is covered by the existing eval/datasets/real GT, so the other 6 months
# genuinely needed human verification rather than being derivable from any existing source.
_MONTH_NAMES = {
    1: "មករា", 2: "កុម្ភៈ", 3: "មីនា", 4: "មេសា", 5: "ឧសភា", 6: "មិថុនា", 7: "កក្កដា",
    8: "សីហា", 9: "កញ្ញា", 10: "តុលា", 11: "វិច្ឆិកា", 12: "ធ្នូ",
}
# 8-12 verified against real ARDB document text: សីហា/ធ្នូ extract cleanly from the
# 06.08.25/01.12.25 probe docs; កញ្ញា/តុលា extract cleanly from Era A 2022-2023 probe
# docs; វិច្ឆិកា confirmed by the user (2026-07-30) after extraction from the 10.11.25
# probe doc came out glyph-scrambled (see the Era A/A2 glyph-confusion reference memo).
# Verified by the user against the rendered letterhead crop (2026-07-27) — static across
# every document, not covered by the existing eval GT (which only covers title + table).
_LETTERHEAD_TEXT = "ធនាគារ ARDB\nដើម្បីកសិករនិងអភិវឌ្ឍន៍សេដ្ឋកិច្ចសង្គម"
_ARABIC_TO_KHMER_DIGITS = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")
_FILENAME_DATE_RE = re.compile(r"-(\d{2})\.(\d{2})\.(\d{2})\.pdf$")
_DAY_RE = re.compile(r"(ថ្ងៃទី)([០-៩]+)")
_MONTH_WORD_RE = re.compile(r"(ខែ)(\S+)")


def load_coco_table_boxes(coco_hf_dir: Path) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    """Map (source PDF basename, page index) -> human-verified Table bbox [x,y,w,h] in
    pixels at the layout dataset's 200 DPI render, same DPI this module renders at.
    Human-corrected boxes are tighter/more accurate than PyMuPDF's own table.bbox, which
    can be inflated by spurious rows (e.g. a footer caption swept into the table)."""
    from datasets import load_dataset

    ds = load_dataset(str(coco_hf_dir / "data"))
    boxes: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    for split in ds:
        for row in ds[split]:
            m = _FILE_NAME_PAGE_RE.search(row["file_name"])
            if not m:
                continue
            page_idx = int(m.group(1))
            source = Path(row["source"]).name
            for cat, bbox in zip(row["objects"]["category"], row["objects"]["bbox"]):
                if cat == "Table":
                    boxes[(source, page_idx)] = tuple(bbox)
                    break
    return boxes


def load_coco_header_furniture_boxes(coco_hf_dir: Path) -> tuple[
        dict[str, tuple[float, float, float, float]], dict[str, tuple[float, float, float, float]]]:
    """Map source PDF basename -> page-0 bbox for the title (Section-Header, single box)
    and the letterhead (Page-Furniture, unioned from its 2 boxes — bank name + tagline are
    separate regions but always transcribed as one crop). Only page 0 carries these."""
    from datasets import load_dataset

    ds = load_dataset(str(coco_hf_dir / "data"))
    headers: dict[str, tuple[float, float, float, float]] = {}
    furniture_parts: dict[str, list[tuple[float, float, float, float]]] = {}
    for split in ds:
        for row in ds[split]:
            m = _FILE_NAME_PAGE_RE.search(row["file_name"])
            if not m or int(m.group(1)) != 0:
                continue
            source = Path(row["source"]).name
            for cat, bbox in zip(row["objects"]["category"], row["objects"]["bbox"]):
                if cat == "Section-Header" and source not in headers:
                    headers[source] = tuple(bbox)
                elif cat == "Page-Furniture":
                    furniture_parts.setdefault(source, []).append(tuple(bbox))
    furniture = {src: _union_bbox(boxes) for src, boxes in furniture_parts.items()}
    return headers, furniture


def _union_bbox(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def load_title_template(gt_dir: Path, stem: str = _TEMPLATE_GT_STEM) -> str:
    """Build the title template (day/month replaced with {day}/{month} placeholders) from
    the same verified GT the table templates come from — mechanical string substitution on
    already-extracted, already-verified text, not a new transcription."""
    path = gt_dir / f"{stem}_p1_ground_truth.json"
    title = json.loads(path.read_text())["paragraphs"][0].strip()
    title = _DAY_RE.sub(r"\1{day}", title, count=1)
    title = _MONTH_WORD_RE.sub(r"\1{month}", title, count=1)
    return title


def _classify_era_text(page0_text: str) -> str:
    """"wholesale_retail" (majority template: 2026 corpus + Era A2/B) if the page-0 text
    contains the wholesale marker, else "retail_only" (older Era A layout, ~2022-May 2024)
    — the same content-based signal already verified reliable for era classification
    across the multi-year probe set (more robust than filename conventions, which vary).
    Whitespace is stripped before matching: confirmed the marker itself can render with a
    stray space inserted mid-word (e.g. "ប ោះដុំ" instead of "បោះដុំ") depending on the
    PDF's own text-flow quirks, unrelated to the separate glyph-scramble phenomenon."""
    collapsed = re.sub(r"\s+", "", page0_text)
    return "wholesale_retail" if _WHOLESALE_MARKER in collapsed else "retail_only"


def classify_era(pdf_path: Path) -> str:
    """See `_classify_era_text` — this just supplies the page-0 text from the PDF."""
    with fitz.open(str(pdf_path)) as doc:
        return _classify_era_text(doc[0].get_text("text"))


def parse_filename_date(name: str) -> tuple[int, int, int] | None:
    """ARDB filenames end in -DD.MM.YY.pdf; returns (day, month, year) or None."""
    m = _FILENAME_DATE_RE.search(name)
    if not m:
        return None
    day, month, year = (int(g) for g in m.groups())
    return day, month, year


def parse_header_date(dates: list[str]) -> tuple[int, int, int] | None:
    """Use the table's own second (later) header date for the title's day/month —
    confirmed against the 09.06.26 GT (title reads '...ថ្ងៃទី៩ ខែមិថុនា...', matching
    the second header date 09-06-26, not the first/earlier 08-06-26). Unlike
    `parse_filename_date`, this works regardless of filename convention, since every
    era's table header embeds these dates in the same dd-mm-yy form."""
    if len(dates) < 2:
        return None
    day, month, year = (int(x) for x in dates[1].split("-"))
    return day, month, year


def build_title_text(template: str, day: int, month: int) -> str | None:
    """Fill the title template with this document's own day/month. Returns None if the
    month isn't one of the user-verified names — quarantine rather than guess."""
    month_name = _MONTH_NAMES.get(month)
    if month_name is None:
        return None
    khmer_day = str(day).translate(_ARABIC_TO_KHMER_DIGITS)
    return template.format(day=khmer_day, month=month_name)


def _crop_pixels(img, bbox_xywh: tuple[float, float, float, float], pad: int = 6):
    """Crop by a bbox already in the rendered image's own pixel space (COCO boxes,
    unlike table.bbox, are not in PDF-point space, so no dpi/72 scaling here)."""
    x, y, w, h = bbox_xywh
    x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
    x1, y1 = min(img.width, int(x + w) + pad), min(img.height, int(y + h) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img.crop((x0, y0, x1, y1))


def _normalize_two_row_header(rows: list[list[str]]) -> list[list[str]]:
    """Some era GT stores the header as 2 physical rows (row-num/name/unit spanning both,
    dates on row 0, retail/wholesale sub-labels on row 1) rather than one pre-merged row
    like the Era B/2026 GT convention (e.g. "08-06-26 បោះដុំ"). Merge them into one row so
    `build_page`'s single-header-row logic works the same regardless of which era's GT
    a template was loaded from."""
    if len(rows) < 2 or rows[1][0]:
        return rows  # already single-row-header convention (row 1 is real content)
    if not any(rows[1][1:]):
        return rows
    merged = [f"{a} {b}".strip() if b else a for a, b in zip(rows[0], rows[1])]
    return [merged] + rows[2:]


def load_templates(gt_dir: Path, stem: str = _TEMPLATE_GT_STEM) -> list[list[list[str]]]:
    """Load the 3 canonical page templates (each a list of rows) from the verified GT.
    Never used as a training example itself — only its row structure."""
    templates = []
    for page in _TEMPLATE_GT_PAGES:
        path = gt_dir / f"{stem}_{page}_ground_truth.json"
        data = json.loads(path.read_text())["tables"][0]["data"]
        templates.append(_normalize_two_row_header(data))
    return templates


def _is_section_row(row: list[str]) -> bool:
    return bool(row[0]) and not any(row[1:])


def _extract_dates(raw_rows: list[list[str | None]]) -> list[str]:
    """Pull the two dd-mm-yy header dates out of raw header rows, in first-seen order."""
    dates: list[str] = []
    for row in raw_rows:
        for cell in row:
            if not cell:
                continue
            for m in _DATE_RE.findall(cell):
                if m not in dates:
                    dates.append(m)
    return dates


def extract_row_blocks(page: fitz.Page) -> list[list[str]]:
    """Row-per-text-block extraction via get_text('blocks'), as a fallback for documents
    whose layout makes find_tables() fragment or drop columns — confirmed on several Era A
    probe PDFs, where find_tables() detects the item-name column as one table and the
    date/price header as separate near-empty 2-row tables, losing the price data entirely.
    Each real body row renders as one contiguous newline-joined block here (row-number,
    name, unit, both prices, % — all present, correctly spelled, no glyph scramble), sorted
    top-to-bottom. Deliberately scans the whole page rather than restricting to
    find_tables()'s own bbox — that bbox is exactly what's untrustworthy here (confirmed to
    clip off a page's very first data row in one case) — and relies on the caller's
    tail-alignment (leading `header_rows` trim on page 0, trailing length-trim on
    continuation pages, both already in `build_page`) to drop title/footer noise instead."""
    blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1], 1), b[0]))
    rows: list[list[str]] = []
    for b in blocks:
        lines = [ln.strip() for ln in b[4].split("\n")]
        while lines and lines[-1] == "":
            lines.pop()
        if lines:
            rows.append(_split_merged_row_number(lines))
    return rows


def _split_merged_row_number(lines: list[str]) -> list[str]:
    """If the row-number and item-name rendered without a newline between them (the
    recurring quirk this module works around), split that first line back into 2 cells.
    A no-op for every normal row, where the first line is just the row number alone."""
    if not lines:
        return lines
    m = _MERGED_ROW_NUM_RE.match(lines[0])
    if not m:
        return lines
    return [m.group(1), m.group(2)] + lines[1:]


def substitute_header(template_row: list[str], new_dates: list[str]) -> list[str] | None:
    """Swap the template header's two embedded dates for this document's own dates."""
    old_dates = _extract_dates([template_row])
    if len(old_dates) != 2 or len(new_dates) != 2:
        return None
    return [
        cell.replace(old_dates[0], new_dates[0]).replace(old_dates[1], new_dates[1])
        if cell else cell
        for cell in template_row
    ]


def substitute_row(template_row: list[str], raw_row: list[str | None],
                   price_cols: tuple[int, ...] = _PRICE_COLS) -> list[str] | None:
    """Fill a template data row's price/percent slots with this document's own numbers.
    Returns None (caller quarantines the page) if the row-number doesn't match the
    template's, or the extracted value count/QA doesn't match the template's populated
    slots — this is the safety net for the "structure never changes" assumption."""
    raw_values = [v.strip() for v in raw_row if v not in (None, "") and v.strip()]
    numeric = [v for v in raw_values if is_numeric_text(v)]
    if not numeric or numeric[0] != template_row[0]:
        return None
    values = numeric[1:]
    if not all(passes_numeric_qa(v) for v in values):
        return None
    populated = [c for c in price_cols if template_row[c]]
    if len(values) != len(populated):
        return None
    row = list(template_row)
    for col, val in zip(populated, values):
        row[col] = val
    return row


def build_page(template: list[list[str]], raw_grid: list[list[str | None]],
               has_header: bool, header_rows: int = 2,
               price_cols: tuple[int, ...] = _PRICE_COLS) -> list[list[str]] | None:
    """Align a document's raw table grid to one page template and substitute in this
    document's own dates/numbers. Returns None if alignment fails anywhere (quarantine).
    `header_rows` lets callers using block-extracted grids (variable amounts of header
    noise ahead of the real body rows) tell this function where the body actually starts,
    instead of assuming the find_tables()-shaped "always first 2 rows" convention."""
    if has_header:
        dates = _extract_dates(raw_grid[:header_rows])
        header = substitute_header(template[0], dates)
        if header is None:
            return None
        body_template, body_raw = template[1:], raw_grid[header_rows:]
        out = [header]
    else:
        body_template, body_raw = template, raw_grid
        out = []

    if len(body_raw) < len(body_template):
        return None  # missing rows -> can't align
    body_raw = body_raw[: len(body_template)]  # drop trailing footer/caption rows

    for t_row, r_row in zip(body_template, body_raw):
        if _is_section_row(t_row):
            out.append(list(t_row))
            continue
        filled = substitute_row(t_row, r_row, price_cols)
        if filled is None:
            return None
        out.append(filled)
    return out


_INSTRUCTION = (
    "Transcribe this ARDB commodity price table as a markdown table, exactly as shown. "
    "Every ARDB bulletin uses the same commodities, units, and column layout — only the "
    "two header dates and the numeric price/percent values change. Preserve the Khmer "
    "text, all numbers, and empty cells exactly."
)
_TITLE_INSTRUCTION = (
    "Transcribe the title line of this ARDB bulletin page, exactly as shown. Every "
    "bulletin uses the same title text — only the day and month change."
)
_LETTERHEAD_INSTRUCTION = (
    "Transcribe the bank letterhead (name and tagline) shown in this image, exactly as shown."
)


def build_document(pdf_path: Path, templates: list[list[list[str]]],
                   out_dir: Path, doc_id: str, hf_split: str,
                   coco_boxes: dict[tuple[str, int], tuple[float, float, float, float]] | None = None,
                   title_template: str | None = None,
                   header_boxes: dict[str, tuple[float, float, float, float]] | None = None,
                   furniture_boxes: dict[str, tuple[float, float, float, float]] | None = None,
                   price_cols: tuple[int, ...] = _PRICE_COLS,
                   ) -> dict[str, int]:
    coco_boxes = coco_boxes or {}
    header_boxes = header_boxes or {}
    furniture_boxes = furniture_boxes or {}
    counts = {"pages_ok": 0, "pages_quarantined": 0, "coco_crop": 0, "fallback_crop": 0,
              "title_ok": 0, "title_skipped": 0, "letterhead_ok": 0, "letterhead_skipped": 0}
    split_dir = out_dir / hf_split
    unaligned_dir = out_dir / "unaligned"
    rows: list[str] = []
    page0_img = None
    page0_dates: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_idx, template in enumerate(templates):
            if page_idx >= doc.page_count:
                counts["pages_quarantined"] += 1
                continue
            page = doc[page_idx]
            tables = page.find_tables().tables
            if not tables:
                counts["pages_quarantined"] += 1
                continue
            table = tables[0]
            table_bbox = _union_bbox([t.bbox for t in tables])
            raw_grid = table.extract()
            final_grid = build_page(template, raw_grid, has_header=(page_idx == 0),
                                    price_cols=price_cols)
            if final_grid is None:
                # find_tables() confirmed unreliable on several older-era layouts (it can
                # fragment one logical table into a name/unit table plus separate
                # near-empty price-header tables, losing the price data entirely) — retry
                # with the row-per-text-block extractor before quarantining the page.
                row_blocks = extract_row_blocks(page)
                body_len = len(template) - 1 if page_idx == 0 else len(template)
                header_rows = max(0, len(row_blocks) - body_len)
                final_grid = build_page(template, row_blocks, has_header=(page_idx == 0),
                                        header_rows=header_rows, price_cols=price_cols)
            page_img = _render_page(page, _DPI)
            if page_idx == 0:
                page0_img = page_img
                if final_grid is not None:
                    page0_dates = _extract_dates([final_grid[0]])
            coco_bbox = coco_boxes.get((pdf_path.name, page_idx))
            if coco_bbox is not None:
                crop = _crop_pixels(page_img, coco_bbox, pad=6)
                counts["coco_crop"] += 1
            else:
                # Falls back to the table bbox (unioned across every find_tables()
                # fragment, so a fragmented detection still yields the full crop region),
                # which can be inflated by spurious rows — only expected when a document
                # isn't in the layout dataset at all.
                crop = _crop(page_img, table_bbox, _DPI, pad=6)
                counts["fallback_crop"] += 1
            if final_grid is None or crop is None:
                counts["pages_quarantined"] += 1
                if crop is not None:
                    unaligned_dir.mkdir(parents=True, exist_ok=True)
                    crop.save(unaligned_dir / f"{doc_id}_p{page_idx}_t0.png")
                continue
            split_dir.mkdir(parents=True, exist_ok=True)
            name = f"{doc_id}_p{page_idx}_t0.png"
            crop.save(split_dir / name)
            date = next(iter(_extract_dates([final_grid[0]])), "") if page_idx == 0 else ""
            rows.append(json.dumps({
                "image": name, "instruction": _INSTRUCTION,
                "text": grid_to_html(final_grid, has_header=(page_idx == 0)),
                "doc_id": doc_id, "source": pdf_path.name, "page": page_idx, "date": date,
            }, ensure_ascii=False))
            counts["pages_ok"] += 1

    # Title (Section-Header) and letterhead (Page-Furniture) rows — page 0 only.
    if page0_img is not None and title_template is not None:
        date_parts = parse_header_date(page0_dates) or parse_filename_date(pdf_path.name)
        header_bbox = header_boxes.get(pdf_path.name)
        title_text = (build_title_text(title_template, date_parts[0], date_parts[1])
                     if date_parts is not None else None)
        if title_text is not None and header_bbox is not None:
            crop = _crop_pixels(page0_img, header_bbox, pad=8)
        else:
            crop = None
        if title_text is not None and crop is not None:
            name = f"{doc_id}_p0_title.png"
            crop.save(split_dir / name)
            rows.append(json.dumps({
                "image": name, "instruction": _TITLE_INSTRUCTION, "text": title_text,
                "doc_id": doc_id, "source": pdf_path.name, "page": 0, "date": "",
            }, ensure_ascii=False))
            counts["title_ok"] += 1
        else:
            counts["title_skipped"] += 1

    furniture_bbox = furniture_boxes.get(pdf_path.name)
    if page0_img is not None and furniture_bbox is not None:
        crop = _crop_pixels(page0_img, furniture_bbox, pad=8)
        if crop is not None:
            name = f"{doc_id}_p0_letterhead.png"
            crop.save(split_dir / name)
            rows.append(json.dumps({
                "image": name, "instruction": _LETTERHEAD_INSTRUCTION, "text": _LETTERHEAD_TEXT,
                "doc_id": doc_id, "source": pdf_path.name, "page": 0, "date": "",
            }, ensure_ascii=False))
            counts["letterhead_ok"] += 1
        else:
            counts["letterhead_skipped"] += 1
    else:
        counts["letterhead_skipped"] += 1

    if rows:
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "pairs.jsonl").open("a") as f:
            f.write("\n".join(rows) + "\n")
    return counts


def build_corpus(corpus_dir: Path, out_dir: Path, gt_dir: Path,
                 exclude_stems: list[str] | None = None, seed: int = 0,
                 coco_hf_dir: Path | None = None) -> dict[str, int]:
    if exclude_stems is None:
        exclude_stems = _DEFAULT_EXCLUDE_STEMS
    # Era-keyed: each document is classified once (via classify_era) and built against its
    # own era's template/title/price-columns. "wholesale_retail" stays the default so the
    # already-working 2026 corpus is unaffected by this change.
    templates_by_era = {
        "wholesale_retail": load_templates(gt_dir),
        "retail_only": load_templates(gt_dir, stem=_RETAIL_ONLY_GT_STEM),
    }
    title_template_by_era = {
        "wholesale_retail": load_title_template(gt_dir),
        "retail_only": load_title_template(gt_dir, stem=_RETAIL_ONLY_GT_STEM),
    }
    has_coco = bool(coco_hf_dir and coco_hf_dir.is_dir())
    coco_boxes = load_coco_table_boxes(coco_hf_dir) if has_coco else {}
    header_boxes, furniture_boxes = (
        load_coco_header_furniture_boxes(coco_hf_dir) if has_coco else ({}, {}))
    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found under {corpus_dir}")
    # Date-clustered (adjacent-day bulletins never straddle train/valid/test) AND
    # era-stratified (each of the two structural templates gets its own proportional
    # valid/test representation -- date-clustering alone left this to chance, and an
    # audit found the two eras skewed ~57%/~19% between test and train by luck of seed).
    doc_dates = {p.name: _pdf_date(p) for p in pdfs}
    doc_eras = {p.name: classify_era(p) for p in pdfs}
    splits = assign_splits_by_date_cluster_stratified(doc_dates, doc_eras, seed=seed)

    totals = {"pages_ok": 0, "pages_quarantined": 0, "skipped_docs": 0,
              "coco_crop": 0, "fallback_crop": 0,
              "title_ok": 0, "title_skipped": 0, "letterhead_ok": 0, "letterhead_skipped": 0}
    for doc_idx, pdf in enumerate(pdfs):
        if any(s in pdf.stem for s in exclude_stems):
            totals["skipped_docs"] += 1
            continue
        hf_split = _HF_SPLIT_NAME[splits[pdf.name]]
        doc_id = f"doc_{doc_idx:03d}"
        era = doc_eras[pdf.name]
        counts = build_document(pdf, templates_by_era[era], out_dir, doc_id, hf_split,
                                coco_boxes, title_template_by_era[era], header_boxes,
                                furniture_boxes, price_cols=_PRICE_COLS_BY_ERA[era])
        for k in totals:
            if k in counts:
                totals[k] += counts[k]
        print(f"[{doc_idx + 1}/{len(pdfs)}] {pdf.name} -> {hf_split} ({era}) "
              f"({counts['pages_ok']} ok, {counts['pages_quarantined']} quarantined, "
              f"title={counts['title_ok']}, letterhead={counts['letterhead_ok']})")
    print(f"Done: {totals} -> {out_dir}")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Template-fill ARDB table SFT pairs from born-digital PDFs.")
    parser.add_argument("corpus", type=Path, help="Folder of ARDB PDFs (scanned recursively)")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    parser.add_argument("--gt-dir", type=Path, default=Path("eval/datasets/real"),
                        help="Folder holding the verified 09.06.26 GT used as the template")
    parser.add_argument("--coco-hf-dir", type=Path, default=_DEFAULT_COCO_HF_DIR,
                        help="Packaged ardb-layout-coco HF folder, for human-verified crop "
                             "boxes (falls back to PyMuPDF's own bbox if a doc isn't found)")
    parser.add_argument("--seed", type=int, default=0, help="Split seed (match other tracks)")
    parser.add_argument("--exclude-stems", nargs="+", default=_DEFAULT_EXCLUDE_STEMS,
                        help="Skip docs whose filename contains any of these (eval-GT guard)")
    args = parser.parse_args()
    build_corpus(args.corpus, args.out, args.gt_dir,
                exclude_stems=args.exclude_stems, seed=args.seed, coco_hf_dir=args.coco_hf_dir)


if __name__ == "__main__":
    main()
