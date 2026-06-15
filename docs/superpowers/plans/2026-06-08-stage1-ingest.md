# Stage 1 — Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the uv project, define all typed dataclass contracts in `models.py`, and implement `ingest.py` — the stage that converts a PDF or image upload into a list of RGB numpy arrays (`IngestResult`). All other stages are stubbed. The Streamlit app shows page thumbnails after upload to confirm Stage 1 works end-to-end.

**Architecture:** Typed dataclass pipeline (Option B). Each stage consumes one dataclass and produces the next. All processing is in-memory — no intermediate files written to disk. `ingest.py` uses `pymupdf` (fitz) to render PDF pages at configurable DPI; image inputs are wrapped into a 1-element list to satisfy the same `IngestResult` contract.

**Tech Stack:** Python 3.11, uv, pymupdf (fitz), numpy, pillow, streamlit, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Create | Project metadata, dependencies, src layout config |
| `.gitignore` | Create | Exclude venv, caches, superpowers session data |
| `src/khmer_pipeline/__init__.py` | Create | Package marker (empty) |
| `src/khmer_pipeline/models.py` | Create | All typed dataclasses — single source of truth for stage contracts |
| `src/khmer_pipeline/ingest.py` | Create | PDF/image → `IngestResult` |
| `src/khmer_pipeline/preprocess.py` | Create | Stub — raises `NotImplementedError` |
| `src/khmer_pipeline/surya.py` | Create | Stub — raises `NotImplementedError` |
| `src/khmer_pipeline/postprocess.py` | Create | Stub — raises `NotImplementedError` |
| `src/khmer_pipeline/export.py` | Create | Stub — raises `NotImplementedError` |
| `src/khmer_pipeline/pipeline.py` | Create | Stub orchestrator — raises `NotImplementedError` |
| `tests/__init__.py` | Create | Test package marker (empty) |
| `tests/test_ingest.py` | Create | TDD tests for `ingest.py` |
| `app.py` | Create | Streamlit UI — upload + Stage 1 thumbnail preview |
| `sample_data/` | Create dir | Place test PDFs here (not committed) |

---

## Task 1: Git init + uv project setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.python-version`

- [ ] **Step 1: Initialise git and uv project**

Run from `/Users/vin/Internship/khmer-ocr-pipeline`:

```bash
git init
uv init --lib --name khmer-pipeline --python 3.11 --no-readme
```

`uv init --lib` creates `src/khmer_pipeline/__init__.py`, `pyproject.toml` with hatchling, and `.python-version`. Delete the stub it generates if present:

```bash
rm -f src/khmer_pipeline/py.typed 2>/dev/null; true
```

- [ ] **Step 2: Replace pyproject.toml with the project config**

Overwrite `pyproject.toml` with:

```toml
[project]
name = "khmer-pipeline"
version = "0.1.0"
description = "Khmer document extraction pipeline"
requires-python = ">=3.11"
dependencies = [
    "pymupdf>=1.24",
    "numpy>=1.26",
    "pillow>=10.0",
    "streamlit>=1.35",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/khmer_pipeline"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync --extra dev
```

Expected: uv resolves and installs all packages, creates `uv.lock`. No errors.

- [ ] **Step 4: Create .gitignore**

Create `.gitignore`:

```
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
dist/
*.egg-info/
.superpowers/
.DS_Store
```

- [ ] **Step 5: Create directory structure**

```bash
mkdir -p tests sample_data
touch tests/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .python-version src/ tests/
git commit -m "chore: initialise uv project with src layout"
```

---

## Task 2: models.py — all stage dataclasses

**Files:**
- Create: `src/khmer_pipeline/models.py`

No tests for this task — pure dataclasses with no logic to test.

- [ ] **Step 1: Write models.py**

Create `src/khmer_pipeline/models.py`:

```python
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
```

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from khmer_pipeline.models import IngestResult, ExportResult; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/khmer_pipeline/models.py
git commit -m "feat: add typed dataclass contracts for all pipeline stages"
```

---

## Task 3: TDD — ingest.py

**Files:**
- Create: `tests/test_ingest.py`
- Create: `src/khmer_pipeline/ingest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest.py`:

```python
from __future__ import annotations
import io
import numpy as np
import pytest
import fitz
from PIL import Image

from khmer_pipeline.models import IngestResult
from khmer_pipeline.ingest import ingest, MAX_PAGES


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_pdf(n_pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), "Test page", fontsize=12)
    return doc.tobytes()


def _make_png(width: int = 100, height: int = 80) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_pdf_returns_ingest_result():
    result = ingest(_make_pdf(), "test.pdf")
    assert isinstance(result, IngestResult)


def test_pdf_page_count_single():
    result = ingest(_make_pdf(1), "test.pdf")
    assert result.page_count == 1
    assert len(result.page_images) == 1


def test_pdf_page_count_multi():
    result = ingest(_make_pdf(3), "three.pdf")
    assert result.page_count == 3
    assert len(result.page_images) == 3


def test_pdf_images_are_rgb_uint8():
    result = ingest(_make_pdf(), "test.pdf")
    arr = result.page_images[0]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3


def test_pdf_dpi_stored():
    result = ingest(_make_pdf(), "test.pdf", dpi=300)
    assert result.dpi == 300


def test_pdf_default_dpi_is_200():
    result = ingest(_make_pdf(), "test.pdf")
    assert result.dpi == 200


def test_pdf_page_limit_raises():
    data = _make_pdf(MAX_PAGES + 1)
    with pytest.raises(ValueError, match="limit is"):
        ingest(data, "big.pdf")


def test_pdf_at_exact_limit_passes():
    data = _make_pdf(MAX_PAGES)
    result = ingest(data, "edge.pdf")
    assert result.page_count == MAX_PAGES


def test_image_png_wraps_to_single_page():
    result = ingest(_make_png(100, 80), "scan.png")
    assert result.page_count == 1
    assert len(result.page_images) == 1


def test_image_dimensions_preserved():
    result = ingest(_make_png(100, 80), "scan.png")
    arr = result.page_images[0]
    assert arr.shape == (80, 100, 3)


def test_image_dpi_is_zero():
    result = ingest(_make_png(), "scan.png")
    assert result.dpi == 0


def test_source_name_stored():
    result = ingest(_make_pdf(), "ardb_sample.pdf")
    assert result.source_name == "ardb_sample.pdf"


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        ingest(b"fake", "document.docx")
```

- [ ] **Step 2: Run tests — confirm they all fail**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: all tests fail with `ModuleNotFoundError` or `ImportError` because `ingest.py` doesn't exist yet.

- [ ] **Step 3: Implement ingest.py**

Create `src/khmer_pipeline/ingest.py`:

```python
from __future__ import annotations
import io
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

from .models import IngestResult

MAX_PAGES = 50
DEFAULT_DPI = 200

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def ingest(source: bytes, source_name: str, dpi: int = DEFAULT_DPI) -> IngestResult:
    suffix = Path(source_name).suffix.lower()
    if suffix == ".pdf":
        return _ingest_pdf(source, source_name, dpi)
    if suffix in _IMAGE_SUFFIXES:
        return _ingest_image(source, source_name)
    raise ValueError(f"Unsupported file type: {suffix!r}. Expected PDF or image.")


def _ingest_pdf(data: bytes, source_name: str, dpi: int) -> IngestResult:
    doc = fitz.open(stream=data, filetype="pdf")
    page_count = len(doc)
    if page_count > MAX_PAGES:
        raise ValueError(
            f"Document has {page_count} pages; limit is {MAX_PAGES} for this prototype."
        )
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images: list[np.ndarray] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        images.append(arr.copy())
    return IngestResult(
        source_name=source_name,
        page_images=images,
        dpi=dpi,
        page_count=page_count,
    )


def _ingest_image(data: bytes, source_name: str) -> IngestResult:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return IngestResult(
        source_name=source_name,
        page_images=[arr],
        dpi=0,
        page_count=1,
    )
```

- [ ] **Step 4: Run tests — confirm they all pass**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: all 13 tests PASS, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add src/khmer_pipeline/ingest.py tests/test_ingest.py
git commit -m "feat: implement ingest stage — PDF/image to IngestResult (TDD)"
```

---

## Task 4: Stub remaining stage files

**Files:**
- Create: `src/khmer_pipeline/preprocess.py`
- Create: `src/khmer_pipeline/surya.py`
- Create: `src/khmer_pipeline/postprocess.py`
- Create: `src/khmer_pipeline/export.py`
- Create: `src/khmer_pipeline/pipeline.py`

- [ ] **Step 1: Create preprocess.py stub**

Create `src/khmer_pipeline/preprocess.py`:

```python
from __future__ import annotations
from .models import IngestResult, PreprocessResult


def preprocess(result: IngestResult) -> PreprocessResult:
    raise NotImplementedError("Stage 2 (preprocessing) not yet implemented.")
```

- [ ] **Step 2: Create surya.py stub**

Create `src/khmer_pipeline/surya.py`:

```python
from __future__ import annotations
from .models import PreprocessResult, SuryaResult


def run_surya(result: PreprocessResult) -> SuryaResult:
    raise NotImplementedError("Stage 3 (Surya 2 OCR) not yet implemented.")
```

- [ ] **Step 3: Create postprocess.py stub**

Create `src/khmer_pipeline/postprocess.py`:

```python
from __future__ import annotations
from .models import SuryaResult, PostprocessResult


def postprocess(result: SuryaResult) -> PostprocessResult:
    raise NotImplementedError("Stage 4 (post-processing) not yet implemented.")
```

- [ ] **Step 4: Create export.py stub**

Create `src/khmer_pipeline/export.py`:

```python
from __future__ import annotations
from .models import PostprocessResult, ExportResult


def export(result: PostprocessResult) -> ExportResult:
    raise NotImplementedError("Stage 5 (export) not yet implemented.")
```

- [ ] **Step 5: Create pipeline.py stub**

Create `src/khmer_pipeline/pipeline.py`:

```python
from __future__ import annotations
from .ingest import ingest
from .preprocess import preprocess
from .surya import run_surya
from .postprocess import postprocess
from .export import export
from .models import ExportResult


def run(source: bytes, source_name: str, dpi: int = 200) -> ExportResult:
    ingest_result = ingest(source, source_name, dpi=dpi)
    preprocess_result = preprocess(ingest_result)
    surya_result = run_surya(preprocess_result)
    postprocess_result = postprocess(surya_result)
    return export(postprocess_result)
```

- [ ] **Step 6: Verify all imports resolve**

```bash
uv run python -c "
from khmer_pipeline.pipeline import run
from khmer_pipeline.preprocess import preprocess
from khmer_pipeline.surya import run_surya
from khmer_pipeline.postprocess import postprocess
from khmer_pipeline.export import export
print('All stubs import OK')
"
```

Expected output: `All stubs import OK`

- [ ] **Step 7: Commit**

```bash
git add src/khmer_pipeline/
git commit -m "feat: add NotImplementedError stubs for stages 2-5 and pipeline orchestrator"
```

---

## Task 5: Streamlit app — Stage 1 UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py**

Create `app.py` in the project root:

```python
from __future__ import annotations
import streamlit as st
from khmer_pipeline.ingest import ingest

st.set_page_config(page_title="Khmer Document Extraction", layout="wide")
st.title("Khmer Document Extraction Pipeline")
st.caption("Stage 1 — PDF / image ingestion")

uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

if uploaded is not None:
    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Stage 1: Converting pages to images...")
        try:
            result = ingest(uploaded.read(), uploaded.name)
            status.update(
                label=f"Stage 1 complete — {result.page_count} page(s) extracted",
                state="complete",
            )
        except ValueError as e:
            status.update(label="Stage 1 failed", state="error")
            st.error(str(e))
            st.stop()

    st.subheader(f"Extracted {result.page_count} page(s) from `{result.source_name}`")

    cols_per_row = 3
    cols = st.columns(cols_per_row)
    for i, img_arr in enumerate(result.page_images):
        with cols[i % cols_per_row]:
            st.image(img_arr, caption=f"Page {i + 1}", width="stretch")
```

- [ ] **Step 2: Run the app and test manually**

```bash
uv run streamlit run app.py
```

Expected: browser opens at `http://localhost:8501`. Upload the ARDB sample PDF. You should see:
- Status bar shows "Stage 1 complete — 3 page(s) extracted"
- Three page thumbnails rendered side-by-side
- Uploading a PNG shows 1 page thumbnail
- Uploading a `.docx` (rename any file) shows the error message

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit Stage 1 UI — upload and page thumbnail preview"
```

---

## Self-Review

**Spec coverage:**
- ✅ uv project setup with src layout
- ✅ `models.py` with all six dataclasses including `CorrectedPageResult`/`PostprocessResult` distinction
- ✅ `table_id` naming convention documented in models.py comment
- ✅ `ingest.py` using pymupdf, DPI configurable (200 default), image input wrapped to 1-element list
- ✅ Page limit guard (>50 raises ValueError)
- ✅ Stages 2–5 stubbed
- ✅ Streamlit UI with upload + thumbnails for Stage 1
- ✅ TDD with 13 tests covering all specified behaviours
- ✅ `.gitignore` excludes `.superpowers/`

**Placeholder scan:** None found. All steps contain exact code and exact commands.

**Type consistency check:**
- `IngestResult` defined in Task 2, consumed in Task 3 (`ingest.py`) and Task 4 (`preprocess.py` stub) ✅
- `preprocess(result: IngestResult)` in Task 4 matches `ingest()` return type ✅
- `run_surya(result: PreprocessResult)` matches `preprocess()` return type ✅
- `pipeline.py` chain matches all stage signatures ✅
- `CorrectedPageResult` (not `SuryaPageResult`) used in `PostprocessResult.pages` ✅
