from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TypedDict
import numpy as np


# ---------------------------------------------------------------------------
# Shared TypedDicts for the cell/table/text_block payloads that flow through
# surya.py (producer) -> postprocess.py -> export.py (consumer) and are also
# serialized to/from JSON. Typing only: these are plain dicts at runtime
# (annotations erased), not a schema migration — see docs/CODE_AUDIT.md §2.
# ---------------------------------------------------------------------------

class TextLine(TypedDict, total=False):
    """One line of OCR'd text inside a table cell (`Cell.text_lines`) or, in
    the flat single-cell fallback, the whole cell's text."""
    text: str
    bbox: list[float]


class Cell(TypedDict, total=False):
    """One table cell as produced by `surya._build_table_from_grid` /
    `surya._process_page` and consumed by `export.py`'s CSV/JSON builders."""
    row_id: int
    col_id: int
    cell_id: int
    bbox: list[float]
    polygon: list[list[float]]
    text_lines: list[TextLine]
    confidence: float  # per-cell recognizer confidence (0..1), set by surya_kiri; absent for other engines
    row_span: int      # >1 when the cell spans multiple rows; set by span-aware structure (surya_kiri slanet path), else absent
    col_span: int      # >1 when the cell spans multiple columns; same provenance as row_span


class RowSpec(TypedDict):
    row_id: int


class ColSpec(TypedDict):
    col_id: int


class Table(TypedDict, total=False):
    """A detected table region as produced by `surya._build_table_from_grid`
    and consumed by `export.py` (CSV/XLSX/JSON) and `postprocess.py` (passed
    through unchanged)."""
    rows: list[RowSpec]
    cols: list[ColSpec]
    cells: list[Cell]
    bbox: list[float]           # layout region bbox (set by surya._process_page)
    image_bbox: list[float]     # bbox at table-build time (may differ from `bbox`)
    was_repaired: bool          # set by export._validate_and_repair_table, optional
    source_pages: list[int]     # set by table_merge_pages.merge_document_tables, optional


class TextBlock(TypedDict, total=False):
    """One non-table layout region's OCR output, as produced by
    `surya._process_page` and consumed by `postprocess._correct_page`."""
    text: str
    bbox: list[float]
    polygon: list[list[float]]
    confidence: float
    label: str
    region_label: str
    reading_order: int


@dataclass
class IngestResult:
    source_name: str
    page_images: list[np.ndarray]       # RGB uint8, shape (H, W, 3), one per page
    dpi: int                            # 0 means native image resolution (image inputs)
    page_count: int


@dataclass
class PreprocessResult:
    source_name: str
    page_images: list[np.ndarray]       # RGB uint8, cleaned
    dpi: int                            # preserved from IngestResult
    page_count: int
    # Geometric-only preprocessed pages (crop + deskew per config, NO photometric
    # normalization), RGB uint8, aligned 1:1 with page_images. Used by recognizers
    # that binarize per-cell (surya_kiri), which need deskew but are hurt by
    # photometric changes. None when unavailable (older callers / benchmark
    # harness) — engines fall back to page_images.
    recognition_page_images: list[np.ndarray] | None = None


@dataclass
class SuryaPageResult:
    page_index: int                     # 0-indexed
    text_blocks: list[TextBlock]        # Surya layout detection output
    tables: list[Table]                 # Surya table recognition output
    ocr_text: str                       # raw OCR string from Surya, never modified


@dataclass
class SuryaResult:
    source_name: str
    pages: list[SuryaPageResult]
    warnings: list[str] = field(default_factory=list)


@dataclass
class CorrectedPageResult:
    page_index: int
    text_blocks: list[TextBlock]
    tables: list[Table]
    raw_ocr_text: str                   # copied from SuryaPageResult.ocr_text, unchanged
    corrected_text: str                 # after rule-based + optional Qwen2.5-VL pass
    correction_diff: str                # difflib.ndiff output between raw and corrected
    qwen_used: bool                     # True if Qwen fallback fired for this page


@dataclass
class PostprocessResult:
    source_name: str
    pages: list[CorrectedPageResult]
    # Stage-4 issues (e.g. malformed-number flags) — displayed alongside
    # SuryaResult.warnings in app.py / printed by pipeline.py.
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExportResult:
    source_name: str
    document_json: dict[str, Any]
    # table_id convention: {source_stem}_page{n}_table{m}, 1-indexed
    # e.g. ardb_sample_page1_table1 → file ardb_sample_page1_table1.csv
    tables_csv: list[tuple[str, str]]   # (table_id, csv_string)
