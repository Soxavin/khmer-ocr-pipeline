"""Harvest recognition + VLM-SFT training pairs from born-digital PDF text layers.

Per corpus PDF: PyMuPDF find_tables() gives cell grids + rects for free (§2.37 method);
each cell rect becomes a recognition pair (crop PNG + text) and each table becomes an SFT
pair (table crop + markdown grid). Docs with a mojibake Khmer layer contribute numeric
cells only; split is by document, consistent with the layout dataset (same seed).

CLI:
    python -m khmer_pipeline.datagen.harvest_table_gt corpus/ --out eval/datasets/table_gt_v1
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import fitz
from PIL import Image

from .inspect_pdf import khmer_layer_suspect
from .pseudo_label_layout import assign_splits
from ..postprocess import _is_malformed_number

_DPI = 200
_MIN_TEXT_CHARS = 100        # matches inspect_pdf's substantial-text threshold
_CELL_PAD_PX = 2             # small margin around cell crops
_MIN_CROP_PX = 8             # skip degenerate rects
_MAX_EMPTY_RATIO = 0.25      # empty-cell negatives kept per page, as share of kept cells
# The frozen eval-GT documents must never enter training data.
_DEFAULT_EXCLUDE_STEMS = ["09.06.26", "15.06.26", "CambodiaBudget"]

# \d matches Khmer digits ០-៩ too; allows grouping commas, decimals, percent, sign.
_NUMERIC_TEXT_RE = re.compile(r"^[+-]?[\d,]*\d(?:[.,]\d+)?%?$")

# Khmer dependent vowels/signs/coeng (U+17B6–U+17D3) can never START a token, and coeng
# (U+17D2) can never END one — scrambled legacy extraction produces exactly these patterns
# (§2.21: text layers can hold Khmer-block codepoints in glyph order, not logical order).
_BAD_TOKEN_START_RE = re.compile(r"(?:^|\s)[ា-៓]")
_TRAILING_COENG_RE = re.compile(r"្(?:\s|$)")

_HF_SPLIT_NAME = {"train": "train", "valid": "validation", "test": "test"}


def khmer_order_valid(text: str) -> bool:
    """True if the text contains no detectably-invalid Khmer ordering (marks at token
    starts, dangling coeng). Scrambled cells fail; note valid-looking scrambles pass."""
    return not (_BAD_TOKEN_START_RE.search(text) or _TRAILING_COENG_RE.search(text))


def is_trusted_cell_text(text: str) -> bool:
    """True if the text-layer value is trustworthy as recognition GT. Visually verified
    (2026-07-10): numbers, empties, Latin, and ៛-unit strings extract correctly; Khmer
    WORDS are pervasively glyph-scrambled (even when the ordering looks valid) → any
    other Khmer-bearing cell must go to the unverified pool for human review."""
    if not text or is_numeric_text(text):
        return True
    if not any("ក" <= ch <= "៿" for ch in text):
        return True  # Latin/dates/punctuation only
    return text.startswith("៛") and khmer_order_valid(text)


def prune_empty_columns(grid: list[list[str | None]]) -> list[list[str]]:
    """Pad ragged rows and drop columns that are empty in every row (find_tables
    over-splits merged-cell tables into many phantom columns)."""
    width = max(len(row) for row in grid)
    padded = [[(c or "").strip() for c in row] + [""] * (width - len(row)) for row in grid]
    keep = [i for i in range(width) if any(row[i] for row in padded)]
    return [[row[i] for i in keep] for row in padded]


def is_numeric_text(text: str) -> bool:
    """True if the cell text is a bare number/percent (Arabic or Khmer digits)."""
    return bool(_NUMERIC_TEXT_RE.match(text.strip()))


def passes_numeric_qa(text: str) -> bool:
    """True unless the numeric text matches a Stage-4 malformed pattern (bad comma
    grouping, suspicious integer percent) — those would poison training GT."""
    return not _is_malformed_number(text)


def cap_empty_cells(cells: list[dict], max_empty_ratio: float = _MAX_EMPTY_RATIO) -> list[dict]:
    """Keep all non-empty cells but cap empty ones so they are at most max_empty_ratio
    of the kept list (empty crops are useful negatives, but only in small doses)."""
    n_non_empty = sum(1 for c in cells if c["text"])
    allowed = int(n_non_empty * max_empty_ratio / (1 - max_empty_ratio))
    if sum(1 for c in cells if not c["text"]) <= allowed:
        return cells
    kept: list[dict] = []
    empties_kept = 0
    for c in cells:
        if c["text"]:
            kept.append(c)
        elif empties_kept < allowed:
            kept.append(c)
            empties_kept += 1
    return kept


def grid_to_markdown(grid: list[list[str | None]]) -> str:
    """Serialize a table grid as a pipe markdown table (first row = header); cells are
    padded to the widest row, None becomes empty, and literal pipes are escaped."""
    width = max(len(row) for row in grid)

    def fmt(row: list[str | None]) -> str:
        # newlines inside a cell would break the pipe-table row structure
        cells = [re.sub(r"\s+", " ", (c or "")).replace("|", "\\|").strip() for c in row]
        cells += [""] * (width - len(cells))
        return "| " + " | ".join(cells) + " |"

    lines = [fmt(grid[0]), "| " + " | ".join(["---"] * width) + " |"]
    lines += [fmt(row) for row in grid[1:]]
    # fmt pads with single spaces around empty cells → "|  |" like common emitters
    return "\n".join(line.replace("|  |", "|  |") for line in lines)


_KHMER_DIGIT_FOLD = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
_MIN_FINGERPRINT_NUMERICS = 2  # rows with fewer numbers are too ambiguous to pair


def _row_fingerprint(row: list[str | None]) -> tuple[str, ...] | None:
    nums = [c.strip().translate(_KHMER_DIGIT_FOLD).replace(",", "")
            for c in row if c and is_numeric_text(c.strip())]
    return tuple(nums) if len(nums) >= _MIN_FINGERPRINT_NUMERICS else None


def _khmer_word_cells(row: list[str | None]) -> list[str]:
    return [c.strip() for c in row
            if c and c.strip() and not is_numeric_text(c.strip())
            and any("ក" <= ch <= "៿" for ch in c)]


def build_row_lexicon(gt_grid: list[list[str | None]],
                      tl_grid: list[list[str | None]]) -> dict[str, str]:
    """Map scrambled text-layer Khmer cells to verified-GT spellings by pairing rows on
    their numeric fingerprint (numbers extract correctly in both sources). Rows whose
    fingerprint is not unique on both sides, or whose Khmer cell counts differ, are
    skipped; keys that would map to conflicting values are dropped."""
    def unique_by_fp(grid):
        by_fp: dict[tuple, list] = {}
        for row in grid:
            fp = _row_fingerprint(row)
            if fp is not None:
                by_fp.setdefault(fp, []).append(row)
        return {fp: rows[0] for fp, rows in by_fp.items() if len(rows) == 1}

    gt_rows, tl_rows = unique_by_fp(gt_grid), unique_by_fp(tl_grid)
    lexicon: dict[str, str] = {}
    conflicted: set[str] = set()
    for fp, gt_row in gt_rows.items():
        tl_row = tl_rows.get(fp)
        if tl_row is None:
            continue
        gt_cells, tl_cells = _khmer_word_cells(gt_row), _khmer_word_cells(tl_row)
        if len(gt_cells) != len(tl_cells):
            continue
        for tl_text, gt_text in zip(tl_cells, gt_cells):
            key = re.sub(r"\s+", "", tl_text)
            if key in lexicon and lexicon[key] != gt_text:
                conflicted.add(key)
            lexicon[key] = gt_text
    for key in conflicted:
        del lexicon[key]
    return lexicon


def build_khmer_lexicon(corpus_dir: Path, gt_dir: Path) -> dict[str, str]:
    """Build the scrambled→verified lexicon from every *_ground_truth.json in gt_dir whose
    source PDF exists under corpus_dir (page number parsed from the _pN suffix)."""
    lexicon: dict[str, str] = {}
    conflicted: set[str] = set()
    pdfs_by_stem = {p.stem: p for p in corpus_dir.rglob("*.pdf")}
    for gt_path in sorted(gt_dir.glob("*_ground_truth.json")):
        m = re.match(r"^(.*)_p(\d+)_ground_truth$", gt_path.stem)
        if not m or m.group(1) not in pdfs_by_stem:
            continue
        gt = json.loads(gt_path.read_text())
        gt_grids = [t["data"] for t in gt.get("tables", []) if t.get("data")]
        if not gt_grids:
            continue
        with fitz.open(str(pdfs_by_stem[m.group(1)])) as doc:
            page = doc[int(m.group(2)) - 1]
            tl_grids = [t.extract() for t in page.find_tables().tables]
        for gt_grid in gt_grids:
            for tl_grid in tl_grids:
                for key, value in build_row_lexicon(gt_grid, tl_grid).items():
                    if key in lexicon and lexicon[key] != value:
                        conflicted.add(key)
                    else:
                        lexicon[key] = value
    for key in conflicted:
        del lexicon[key]
    return lexicon


def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _crop(img: Image.Image, rect, dpi: int, pad: int = _CELL_PAD_PX) -> Image.Image | None:
    scale = dpi / 72
    x0 = max(0, int(rect[0] * scale) - pad)
    y0 = max(0, int(rect[1] * scale) - pad)
    x1 = min(img.width, int(rect[2] * scale) + pad)
    y1 = min(img.height, int(rect[3] * scale) + pad)
    if x1 - x0 < _MIN_CROP_PX or y1 - y0 < _MIN_CROP_PX:
        return None
    return img.crop((x0, y0, x1, y1))


def harvest_corpus(corpus_dir: Path, out_dir: Path, dpi: int = _DPI,
                   exclude_stems: list[str] | None = None, seed: int = 0,
                   max_empty_ratio: float = _MAX_EMPTY_RATIO,
                   gt_dir: Path | None = None) -> dict[str, int]:
    """Harvest recognition and SFT pairs from every text-layer PDF under corpus_dir into
    out_dir/{recognition,sft}/{train,validation,test}/; returns a counts summary.
    If gt_dir holds verified *_ground_truth.json files, a scrambled→verified Khmer
    lexicon is built from them and applied to otherwise-untrusted cells."""
    if exclude_stems is None:
        exclude_stems = _DEFAULT_EXCLUDE_STEMS
    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found under {corpus_dir}")
    lexicon: dict[str, str] = {}
    if gt_dir is not None and gt_dir.is_dir():
        lexicon = build_khmer_lexicon(corpus_dir, gt_dir)
        print(f"Khmer lexicon from verified GT: {len(lexicon)} entries")
    # Split over the FULL doc list (before exclusion) so assignment matches the layout
    # dataset built from the same corpus with the same seed.
    splits = assign_splits([p.name for p in pdfs], seed=seed)

    counts = {"recognition_pairs": 0, "sft_pairs": 0, "dropped_qa": 0,
              "unverified_khmer": 0, "gt_corrected": 0,
              "skipped_docs": 0, "suspect_docs": 0}
    writers: dict[Path, list[str]] = {}

    for doc_idx, pdf in enumerate(pdfs):
        if any(s in pdf.stem for s in exclude_stems):
            counts["skipped_docs"] += 1
            continue
        hf_split = _HF_SPLIT_NAME[splits[pdf.name]]
        doc_id = f"doc_{doc_idx:03d}"
        suspect = khmer_layer_suspect(pdf)
        counts["suspect_docs"] += int(suspect)

        with fitz.open(str(pdf)) as doc:
            if sum(len(p.get_text("text")) for p in doc) < _MIN_TEXT_CHARS:
                counts["skipped_docs"] += 1
                continue
            for page_idx, page in enumerate(doc):
                tables = page.find_tables().tables
                if not tables:
                    continue
                page_img = _render_page(page, dpi)
                for t_idx, table in enumerate(tables):
                    grid = [[unicodedata.normalize("NFC", c).strip() if c else ""
                             for c in row] for row in table.extract()]

                    # --- product (b): SFT pair (skip for suspect-Khmer docs: the
                    # markdown would contain mojibake labels)
                    if not suspect and any(any(row) for row in grid):
                        crop = _crop(page_img, table.bbox, dpi, pad=6)
                        if crop is not None:
                            pruned = [[lexicon.get(re.sub(r"\s+", "", c), c) for c in row]
                                      for row in prune_empty_columns(grid)]
                            invalid_cells = sum(1 for row in pruned for c in row
                                                if c and not khmer_order_valid(c))
                            sft_dir = out_dir / "sft" / hf_split
                            sft_dir.mkdir(parents=True, exist_ok=True)
                            name = f"{doc_id}_p{page_idx}_t{t_idx}.png"
                            crop.save(sft_dir / name)
                            writers.setdefault(sft_dir / "pairs.jsonl", []).append(json.dumps(
                                {"image": name, "markdown": grid_to_markdown(pruned),
                                 "doc_id": doc_id, "source": pdf.name, "page": page_idx,
                                 "n_invalid_khmer_cells": invalid_cells},
                                ensure_ascii=False))
                            counts["sft_pairs"] += 1

                    # --- product (a): recognition pairs, cell by cell
                    cells: list[dict] = []
                    flat_rects = [cell for row in table.rows for cell in row.cells]
                    flat_texts = [c for row in grid for c in row]
                    for cell_idx, (rect, text) in enumerate(zip(flat_rects, flat_texts)):
                        if rect is None:
                            continue
                        numeric = is_numeric_text(text)
                        if suspect and text and not numeric:
                            continue  # mojibake Khmer — unusable as GT
                        if numeric and not passes_numeric_qa(text):
                            counts["dropped_qa"] += 1
                            continue
                        cells.append({"rect": rect, "text": text, "idx": cell_idx,
                                      "is_numeric": numeric})
                    for cell in cap_empty_cells(cells, max_empty_ratio):
                        crop = _crop(page_img, cell["rect"], dpi)
                        if crop is None:
                            continue
                        # scrambled Khmer would be poisonous GT: correct it via the
                        # verified-GT lexicon when possible, else quarantine for review
                        text = cell["text"]
                        corrected = False
                        if not is_trusted_cell_text(text):
                            fixed = lexicon.get(re.sub(r"\s+", "", text))
                            if fixed is not None:
                                text, corrected = fixed, True
                        trusted = corrected or is_trusted_cell_text(text)
                        subdir = hf_split if trusted else f"{hf_split}_unverified"
                        rec_dir = out_dir / "recognition" / subdir
                        rec_dir.mkdir(parents=True, exist_ok=True)
                        name = f"{doc_id}_p{page_idx}_t{t_idx}_c{cell['idx']}.png"
                        crop.save(rec_dir / name)
                        writers.setdefault(rec_dir / "pairs.jsonl", []).append(json.dumps(
                            {"image": name, "text": text, "doc_id": doc_id,
                             "source": pdf.name, "page": page_idx,
                             "is_empty": not text, "is_numeric": cell["is_numeric"],
                             "gt_corrected": corrected},
                            ensure_ascii=False))
                        counts["recognition_pairs" if trusted else "unverified_khmer"] += 1
                        counts["gt_corrected"] += int(corrected)
        print(f"[{doc_idx + 1}/{len(pdfs)}] {pdf.name} → {hf_split}"
              f"{' (numeric-only: suspect Khmer layer)' if suspect else ''}")

    for path, lines in writers.items():
        path.write_text("\n".join(lines) + "\n")
    print(f"Done: {counts} → {out_dir}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest recognition/SFT pairs from PDF text layers.")
    parser.add_argument("corpus", type=Path, help="Folder of source PDFs (scanned recursively)")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    parser.add_argument("--dpi", type=int, default=_DPI, help=f"Render DPI (default {_DPI})")
    parser.add_argument("--seed", type=int, default=0,
                        help="Split seed; keep 0 to match the layout dataset")
    parser.add_argument("--exclude-stems", nargs="+", default=_DEFAULT_EXCLUDE_STEMS,
                        help="Skip docs whose filename contains any of these (eval-GT guard)")
    parser.add_argument("--gt-dir", type=Path, default=Path("eval/datasets/real"),
                        help="Verified *_ground_truth.json folder for the Khmer lexicon")
    args = parser.parse_args()
    harvest_corpus(args.corpus, args.out, dpi=args.dpi,
                   exclude_stems=args.exclude_stems, seed=args.seed, gt_dir=args.gt_dir)


if __name__ == "__main__":
    main()
