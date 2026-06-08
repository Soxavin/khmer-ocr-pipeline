from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class IngestResult:
    source_name: str
    page_images: list[np.ndarray]   # RGB uint8, shape (H, W, 3), one per page
    dpi: int                        # 0 means native image resolution (image inputs)
    page_count: int


@dataclass
class PreprocessResult:
    source_name: str
    page_images: list[np.ndarray]   # RGB uint8, cleaned


@dataclass
class SuryaPageResult:
    page_index: int                 # 0-indexed
    text_blocks: list[dict]         # Surya layout detection output
    tables: list[dict]              # Surya table recognition output
    ocr_text: str                   # raw OCR string from Surya, never modified


@dataclass
class SuryaResult:
    source_name: str
    pages: list[SuryaPageResult]


@dataclass
class CorrectedPageResult:
    page_index: int
    text_blocks: list[dict]
    tables: list[dict]
    raw_ocr_text: str               # copied from SuryaPageResult.ocr_text, unchanged
    corrected_text: str             # after rule-based + optional Qwen2.5-VL pass
    qwen_used: bool                 # True if Qwen fallback fired for this page


@dataclass
class PostprocessResult:
    source_name: str
    pages: list[CorrectedPageResult]


@dataclass
class ExportResult:
    document_json: dict
    # table_id convention: {source_stem}_page{n}_table{m}, 1-indexed
    # e.g. ardb_sample_page1_table1 → file ardb_sample_page1_table1.csv
    tables_csv: list[tuple[str, str]]   # (table_id, csv_string)
