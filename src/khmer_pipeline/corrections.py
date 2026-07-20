"""Human-in-the-loop capture: turn analyst table edits into training pairs.

When an analyst fixes a cell the model got wrong, that fix is the scarcest thing
we have — a verified label for a real long-tail failure (misread ៛, slipped
digit). This module diffs the model's output against the analyst's corrected grid
and writes each genuine fix as a ``(cell crop PNG, corrected text)`` pair in the
JSONL shape ``experiments/kiri_finetune/build_trainset.py`` already consumes, so
captured corrections feed the §2.39 fine-tune path with no reformatting.

Curation is the point, not an afterthought: training on every keystroke would
teach the recognizer an analyst's formatting habits instead of fixing its
character errors. Three rules, in order:

1. **Gold-standard** — callers pass only analyst-VERIFIED tables. The model never
   learns from its own unverified output.
2. **Cosmetic edits are dropped** — differences that vanish under normalization
   (whitespace, Unicode form, invisible joiners) are not recognition errors.
3. **Flags are recorded, not filtered** — `validate.py`'s taxonomy travels with
   each record so a future retrain can select error classes (e.g. only
   ``sequence_illegal``) rather than re-deriving them from raw text.

Crops come from the frame the recognizer actually read, so a training crop looks
exactly like an inference crop.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from .utils.khmer_normalize import normalize_khmer

# Zero-width / invisible format characters folded ONLY when deciding whether an
# edit is cosmetic. khmer_normalize deliberately preserves ZWNJ/ZWJ because they
# affect Khmer shaping — that is correct for text handling, but two strings
# differing only by a joiner are visually identical and are NOT a character
# recognition error, so they must not become a training pair. The stored text
# always remains the analyst's exact string.
_INVISIBLE_RE = re.compile(r"[​‌‍⁠﻿­]")

# Cells smaller than this (px, either side) carry no legible glyph — the same
# guard surya_kiri applies before recognition.
_MIN_CROP_PX = 3

_JSONL_NAME = "corrections.jsonl"
_CROP_DIR = "crops"


def _cell_text(cell: dict) -> str:
    lines = cell.get("text_lines") or []
    return " ".join(t["text"] for t in lines if t.get("text")).strip()


def _training_equivalent(a: str, b: str) -> bool:
    """True when two strings differ only cosmetically — i.e. correcting one to the
    other teaches the recognizer nothing about reading characters.

    Applies the project normalizer (NFC, format-char strip, whitespace) and
    additionally folds invisible joiners, which the normalizer intentionally keeps.
    """
    return _INVISIBLE_RE.sub("", normalize_khmer(a)) == \
           _INVISIBLE_RE.sub("", normalize_khmer(b))


def _valid_bbox(bbox, img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Clamp a page-space bbox to the image; None if absent or degenerate.

    surya's VLM cells carry no per-cell geometry, so a missing box is expected and
    must never crash a save — those cells are simply not capturable."""
    if not bbox or len(bbox) != 4:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < _MIN_CROP_PX or y1 - y0 < _MIN_CROP_PX:
        return None
    return x0, y0, x1, y1


def _edited_value(grid: Sequence[Sequence[str]], row: int, col: int) -> Optional[str]:
    if row >= len(grid):
        return None
    line = grid[row]
    return line[col] if col < len(line) else None


def capture_corrections(
    tables: list[dict],
    edited_grids: dict[str, Sequence[Sequence[str]]],
    page_images: Sequence[np.ndarray],
    source_name: str,
    out_dir: Path | str,
    engine: str = "surya_kiri",
    page_index: int = 0,
) -> list[dict]:
    """Write verified analyst corrections as training pairs; return the records.

    *tables* are the model's tables for one page (cells carrying page-space
    ``bbox``), *edited_grids* maps a table's index (as a string key, matching the
    webapp's table ids) to the analyst's corrected 2-D grid, and *page_images* is
    the frame the recognizer read. Cells whose text is unchanged, only
    cosmetically changed, or lacking usable geometry are skipped. Appends to
    ``<out_dir>/corrections.jsonl`` and writes one crop PNG per pair."""
    if not edited_grids:
        return []

    out_dir = Path(out_dir)
    records: list[dict] = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    img = page_images[page_index] if page_index < len(page_images) else None
    if img is None:
        return []

    for t_idx, table in enumerate(tables):
        grid = edited_grids.get(str(t_idx))
        if grid is None:
            continue
        for cell in table.get("cells", []):
            row, col = cell.get("row_id", 0), cell.get("col_id", 0)
            corrected = _edited_value(grid, row, col)
            if corrected is None:
                continue
            predicted = _cell_text(cell)
            if _training_equivalent(predicted, corrected):
                continue  # unchanged or cosmetic — not a recognition error

            box = _valid_bbox(cell.get("bbox"), img)
            if box is None:
                continue  # no usable geometry (e.g. a surya VLM cell)
            x0, y0, x1, y1 = box

            crops = out_dir / _CROP_DIR
            crops.mkdir(parents=True, exist_ok=True)
            stem = f"{Path(source_name).stem}_p{page_index}_t{t_idx}_r{row}_c{col}"
            rel = f"{_CROP_DIR}/{stem}.png"
            Image.fromarray(img[y0:y1, x0:x1]).save(out_dir / rel)

            records.append({
                # Top-level keys match build_trainset.py's schema exactly.
                "image": rel,
                "text": corrected,
                "origin": "correction",
                # Everything else nested, so retrain-time filtering (e.g. by flag)
                # needs no data-cleaning pass.
                "provenance": {
                    "prediction": predicted,
                    "flags": list(cell.get("flags") or []),
                    "confidence": cell.get("confidence"),
                    "source": source_name,
                    "page": page_index,
                    "table": t_idx,
                    "row": row,
                    "col": col,
                    "bbox": [x0, y0, x1, y1],
                    "engine": engine,
                    "timestamp": stamp,
                },
            })

    if records:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Append: corrections accumulate across sessions into one growing corpus.
        with (out_dir / _JSONL_NAME).open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records
