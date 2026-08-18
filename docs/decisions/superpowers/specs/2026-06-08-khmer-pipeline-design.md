# Khmer Document Extraction Pipeline — Design Spec

**Date:** 2026-06-08
**Status:** Approved
**Scope:** Full pipeline architecture + Stage 1 implementation

---

## Context

A 5-stage in-memory pipeline to extract structured data from Khmer financial and economic PDFs (primary test case: ARDB daily price tables). Output is one CSV per detected table and one JSON per document. Built as a working prototype for data analysts at the Department of Digital Economy, Ministry of Economy and Finance, Cambodia.

**Machine:** M4 Pro MacBook, 24GB RAM
**Package manager:** uv
**UI:** Streamlit

---

## Architecture Decision

**Typed dataclass pipeline (Option B).** Each stage consumes one typed dataclass and produces the next. No intermediate files — all data stays in memory. This makes the contract between stages explicit, avoids "what keys does this dict have?" debugging, and lets any stage be tested or inspected in isolation.

---

## Project Structure

```
khmer-ocr-pipeline/
├── pyproject.toml
├── uv.lock
├── src/
│   └── khmer_pipeline/
│       ├── __init__.py
│       ├── models.py        ← all typed dataclasses
│       ├── ingest.py        ← Stage 1: PDF/image → IngestResult
│       ├── preprocess.py    ← Stage 2: stamp removal → PreprocessResult
│       ├── surya.py         ← Stage 3: Surya 2 → SuryaResult
│       ├── postprocess.py   ← Stage 4: Khmer correction → PostprocessResult
│       ├── export.py        ← Stage 5: CSV + JSON → ExportResult
│       └── pipeline.py      ← orchestrator
├── app.py                   ← Streamlit UI
├── tests/
│   └── test_ingest.py
└── sample_data/
    └── ardb_sample.pdf
```

---

## Data Contracts (models.py)

```python
@dataclass
class IngestResult:
    source_name: str
    page_images: list[np.ndarray]   # RGB uint8, one per page
    dpi: int
    page_count: int

@dataclass
class PreprocessResult:
    source_name: str
    page_images: list[np.ndarray]

@dataclass
class SuryaPageResult:
    page_index: int
    text_blocks: list[dict]         # Surya layout output
    tables: list[dict]              # Surya table recognition output
    ocr_text: str                   # raw OCR string, untouched

@dataclass
class SuryaResult:
    source_name: str
    pages: list[SuryaPageResult]

@dataclass
class CorrectedPageResult:
    page_index: int
    text_blocks: list[dict]
    tables: list[dict]
    raw_ocr_text: str               # Surya output, never modified
    corrected_text: str             # after rule-based + optional Qwen pass
    qwen_used: bool                 # whether Qwen fallback fired

@dataclass
class PostprocessResult:
    source_name: str
    pages: list[CorrectedPageResult]   # distinct type from SuryaPageResult

@dataclass
class ExportResult:
    document_json: dict
    tables_csv: list[tuple[str, str]]  # (table_id, csv_string)
    # table_id convention: {source_stem}_page{n}_table{m}
    # e.g. ardb_sample_page1_table1, ardb_sample_page2_table1
    # source_stem = source_name with extension stripped, n and m are 1-indexed
```

**Naming invariant:** `ocr_text` always means raw Surya output. `corrected_text` always means post-processed output. These never appear on the same object — enforced by using separate dataclasses (`SuryaPageResult` vs `CorrectedPageResult`).

---

## Stage Designs

### Stage 1 — Ingest (ingest.py)

**Goal:** Accept a PDF or image file and return a list of RGB numpy arrays, one per page.

- **PDF renderer:** `pymupdf` (fitz) — fastest on M-series, no external binary deps
- **DPI:** 200 default (sufficient for digital PDFs); configurable, recommend 300 for scanned docs
- **Image input:** PNG/JPG/TIFF wrapped into a 1-element list to match the same `IngestResult` contract
- **Page guard:** raise `ValueError` if page count > 50 (prototype safety limit)
- **Output format:** RGB uint8 numpy arrays (what Surya and Qwen both expect natively)

### Stage 2 — Preprocessing (preprocess.py) [stub for now]

- Colored stamp removal via HSV masking
- Gaussian blur for noise/stain handling
- Contrast normalisation

### Stage 3 — Surya 2 (surya.py) [stub for now]

- Single call per page for layout detection + OCR + table recognition
- Maps Surya output dict → `SuryaPageResult`

### Stage 4 — Post-processing (postprocess.py) [stub for now]

- Rule-based Khmer correction first (common character substitutions, spacing rules)
- Error detection triggers Qwen2.5-VL-7B fallback
- Populates `raw_ocr_text` (copied from Surya), `corrected_text`, and `qwen_used`

### Stage 5 — Export (export.py) [stub for now]

- Tables → CSV, one file per detected table, Khmer headers preserved as-is
- CSV filename = `{table_id}.csv` where `table_id` follows `{source_stem}_page{n}_table{m}` (1-indexed, e.g. `ardb_sample_page1_table1.csv`)
- **All CSV files must be written with explicit `encoding="utf-8-sig"`** — the BOM variant ensures Khmer text opens correctly in Excel and common analyst tools without manual encoding selection
- Full document → JSON preserving document structure
- English column header normalisation is a stretch goal, not in scope

---

## Streamlit UI (app.py)

Three zones:
1. **Upload** — `st.file_uploader` accepting PDF, PNG, JPG, TIFF
2. **Progress** — `st.status` block, one entry per pipeline stage
3. **Results** — `st.dataframe` preview per extracted table; download buttons for each CSV and combined JSON

**Stage 1 UI behaviour:** after upload, show page thumbnails of rendered images to confirm extraction before any OCR runs.

---

## Document Types

| Type | DPI | Notes |
|------|-----|-------|
| Digital PDF (primary) | 200 | ARDB price tables, budget execution reports |
| Scanned document (future) | 300 | Same Khmer content, preprocessing more critical |

Both types share the same pipeline. DPI is the only tunable parameter between them.

---

## Out of Scope (prototype)

- Model training or fine-tuning
- Multi-language support beyond Khmer + numbers
- Batch processing of multiple documents
- Authentication or multi-user access
- English column header normalisation (stretch goal only)
