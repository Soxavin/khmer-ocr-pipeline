# Stage 3: Surya 2 OCR Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `surya.py` — takes `PreprocessResult`, runs Surya 2 layout detection + OCR + table recognition per page, returns `SuryaResult`.

**Architecture:** Lazy-initialized module-level predictor singletons (`_get_predictors()`), one `_process_page()` call per page, numpy RGB arrays converted to PIL Images, table recognition runs on cropped table regions only when layout detects a table bbox labelled `"Table"` or `"TableOfContents"`. All Surya output serialized to plain dicts. Unit tests mock `_get_predictors` — no real model runs.

**Tech Stack:** surya-ocr, PIL (already available via pillow), Python 3.11, uv, pytest + unittest.mock

---

### Task 1: Add surya-ocr dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add surya-ocr to dependencies**

Edit `pyproject.toml` `dependencies` list to include `"surya-ocr>=0.7"`:

```toml
dependencies = [
    "pymupdf>=1.24",
    "numpy>=1.26",
    "pillow>=10.0",
    "streamlit>=1.35",
    "opencv-python-headless>=4.8",
    "surya-ocr>=0.7",
]
```

- [ ] **Step 2: Sync dependencies**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv sync --extra dev
```

Expected: surya-ocr and its dependencies (transformers, torch, etc.) install. First install may take several minutes.

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from surya.inference import SuryaInferenceManager; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add surya-ocr dependency"
```

---

### Task 2: Write failing contract tests

**Files:**
- Create: `tests/test_surya.py`

- [ ] **Step 1: Create test file**

Create `tests/test_surya.py` with the full content below:

```python
from __future__ import annotations
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from khmer_pipeline.models import PreprocessResult, SuryaResult, SuryaPageResult
from khmer_pipeline.surya import run_surya


def _make_preprocess_result(n_pages: int = 2) -> PreprocessResult:
    row = np.arange(100, dtype=np.uint8).reshape(1, 100)
    channel = np.tile(row, (100, 1))
    img = np.stack([channel, channel, channel], axis=2)
    return PreprocessResult(
        source_name="ardb.pdf",
        page_images=[img.copy() for _ in range(n_pages)],
        dpi=200,
        page_count=n_pages,
    )


def _make_text_block_mock(reading_order: int = 0) -> MagicMock:
    b = MagicMock()
    b.label = "Text"
    b.html = f"<p>ខ្មែរ {reading_order}</p>"
    b.bbox = [10.0, 10.0, 200.0, 50.0]
    b.polygon = [[10.0, 10.0], [200.0, 10.0], [200.0, 50.0], [10.0, 50.0]]
    b.reading_order = reading_order
    b.confidence = 0.95
    b.skipped = False
    b.error = False
    return b


def _make_predictors(with_table: bool = False):
    """Returns (layout_pred, rec_pred, table_pred) mocks."""
    text_bbox = MagicMock()
    text_bbox.label = "Text"
    text_bbox.bbox = [10.0, 10.0, 200.0, 50.0]

    layout_bboxes = [text_bbox]
    if with_table:
        table_bbox = MagicMock()
        table_bbox.label = "Table"
        table_bbox.bbox = [10.0, 60.0, 200.0, 150.0]
        layout_bboxes.append(table_bbox)

    layout_result = MagicMock()
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    ocr_result = MagicMock()
    ocr_result.blocks = [_make_text_block_mock(0)]
    rec_pred = MagicMock(return_value=[ocr_result])

    if with_table:
        table_result = MagicMock()
        table_result.rows = []
        table_result.cols = []
        table_result.cells = []
        table_result.html = "<table><tr><td>ខ្មែរ</td></tr></table>"
        table_result.error = False
        table_result.mode = "full"
        table_result.image_bbox = [0.0, 0.0, 190.0, 90.0]
        table_pred = MagicMock(return_value=[table_result])
    else:
        table_pred = MagicMock(return_value=[])

    return layout_pred, rec_pred, table_pred


# --- Contract tests ---

def test_run_surya_returns_surya_result():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r, SuryaResult)


def test_run_surya_preserves_source_name():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert r.source_name == "ardb.pdf"


def test_run_surya_page_count_matches():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=3))
    assert len(r.pages) == 3


def test_run_surya_pages_are_surya_page_result():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    for page in r.pages:
        assert isinstance(page, SuryaPageResult)


def test_run_surya_page_index_is_zero_based():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=2))
    assert r.pages[0].page_index == 0
    assert r.pages[1].page_index == 1


def test_run_surya_text_blocks_is_list_of_dicts():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r.pages[0].text_blocks, list)
    assert all(isinstance(b, dict) for b in r.pages[0].text_blocks)


def test_run_surya_block_has_required_keys():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    block = r.pages[0].text_blocks[0]
    for key in ("label", "html", "bbox", "polygon", "reading_order", "confidence", "skipped", "error"):
        assert key in block, f"Missing key: {key}"


def test_run_surya_ocr_text_is_str():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r.pages[0].ocr_text, str)


def test_run_surya_ocr_text_contains_block_html():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert "<p>ខ្មែរ" in r.pages[0].ocr_text


def test_run_surya_no_tables_gives_empty_list():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors(with_table=False)):
        r = run_surya(_make_preprocess_result())
    assert r.pages[0].tables == []


def test_run_surya_with_table_gives_non_empty_list():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors(with_table=True)):
        r = run_surya(_make_preprocess_result())
    assert len(r.pages[0].tables) == 1


def test_run_surya_table_dict_has_required_keys():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors(with_table=True)):
        r = run_surya(_make_preprocess_result())
    table = r.pages[0].tables[0]
    for key in ("rows", "cols", "cells", "html", "error", "mode", "image_bbox"):
        assert key in table, f"Missing table key: {key}"
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run pytest tests/test_surya.py -v
```

Expected: All 12 tests FAIL with `NotImplementedError: Stage 3 (Surya 2 OCR) not yet implemented.`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_surya.py
git commit -m "test: add failing contract tests for Stage 3 Surya OCR integration"
```

---

### Task 3: Implement run_surya

**Files:**
- Modify: `src/khmer_pipeline/surya.py`

- [ ] **Step 1: Replace stub with full implementation**

Replace the entire contents of `src/khmer_pipeline/surya.py`:

```python
from __future__ import annotations
from typing import Any
import numpy as np
from PIL import Image
from .models import PreprocessResult, SuryaResult, SuryaPageResult

_layout_pred = None
_rec_pred = None
_table_pred = None


def _get_predictors():
    global _layout_pred, _rec_pred, _table_pred
    if _layout_pred is None:
        from surya.inference import SuryaInferenceManager
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.table_rec import TableRecPredictor
        manager = SuryaInferenceManager()
        _layout_pred = LayoutPredictor(manager)
        _rec_pred = RecognitionPredictor(manager)
        _table_pred = TableRecPredictor(manager)
    return _layout_pred, _rec_pred, _table_pred


def run_surya(result: PreprocessResult) -> SuryaResult:
    layout_pred, rec_pred, table_pred = _get_predictors()
    pil_images = [Image.fromarray(img) for img in result.page_images]
    pages = [
        _process_page(idx, pil_img, layout_pred, rec_pred, table_pred)
        for idx, pil_img in enumerate(pil_images)
    ]
    return SuryaResult(source_name=result.source_name, pages=pages)


def _process_page(
    page_index: int,
    pil_img: Image.Image,
    layout_pred,
    rec_pred,
    table_pred,
) -> SuryaPageResult:
    layout_result = layout_pred([pil_img])[0]
    ocr_result = rec_pred([pil_img], [layout_result])[0]

    text_blocks = [_serialize_block(b) for b in ocr_result.blocks]
    ocr_text = _build_ocr_text(ocr_result.blocks)

    table_bboxes = [b for b in layout_result.bboxes if b.label in ("Table", "TableOfContents")]
    if table_bboxes:
        crops = [pil_img.crop(tuple(map(int, b.bbox))) for b in table_bboxes]
        table_results = table_pred(crops, mode="full")
        tables = [_serialize_table(t) for t in table_results]
    else:
        tables = []

    return SuryaPageResult(
        page_index=page_index,
        text_blocks=text_blocks,
        tables=tables,
        ocr_text=ocr_text,
    )


def _serialize_block(b) -> dict[str, Any]:
    return {
        "label": b.label,
        "html": b.html,
        "bbox": b.bbox,
        "polygon": b.polygon,
        "reading_order": b.reading_order,
        "confidence": b.confidence,
        "skipped": b.skipped,
        "error": b.error,
    }


def _serialize_table(t) -> dict[str, Any]:
    return {
        "rows": [r.model_dump() for r in t.rows],
        "cols": [c.model_dump() for c in t.cols],
        "cells": [cell.model_dump() for cell in t.cells],
        "html": t.html,
        "error": t.error,
        "mode": t.mode,
        "image_bbox": t.image_bbox,
    }


def _build_ocr_text(blocks) -> str:
    active = [b for b in blocks if not b.skipped and not b.error]
    ordered = sorted(active, key=lambda b: b.reading_order)
    return "\n\n".join(b.html for b in ordered)
```

- [ ] **Step 2: Run tests to verify they all pass**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run pytest tests/test_surya.py -v
```

Expected: All 12 tests PASS.

- [ ] **Step 3: Run full suite to verify no regressions**

```bash
uv run pytest -v
```

Expected: All 36 tests PASS (24 existing + 12 new).

- [ ] **Step 4: Commit**

```bash
git add src/khmer_pipeline/surya.py
git commit -m "feat: implement Stage 3 Surya 2 OCR integration"
```

---

### Task 4: Update app.py for Stage 3 visualization

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace app.py with Stage 3 version**

Replace the entire contents of `app.py`. Note: `_draw_layout` must be defined **before** it is called in the `if uploaded:` block.

```python
from __future__ import annotations
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from khmer_pipeline.ingest import ingest
from khmer_pipeline.preprocess import preprocess
from khmer_pipeline.surya import run_surya

_LABEL_COLORS = {
    "Text": "#4A90D9",
    "Table": "#E74C3C",
    "TableOfContents": "#E67E22",
    "Picture": "#27AE60",
    "Figure": "#27AE60",
    "Caption": "#8E44AD",
}


def _draw_layout(img_array: np.ndarray, blocks: list[dict]) -> np.ndarray:
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    for block in blocks:
        color = _LABEL_COLORS.get(block["label"], "#95A5A6")
        x0, y0, x1, y1 = [int(v) for v in block["bbox"]]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    return np.array(img)


st.set_page_config(page_title="Khmer Document Extraction", layout="wide")
st.title("Khmer Document Extraction Pipeline")
st.caption("Stage 1 — Ingest  |  Stage 2 — Preprocess  |  Stage 3 — Surya OCR")

uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

if uploaded is not None:
    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Stage 1: Converting pages to images...")
        try:
            ingest_result = ingest(uploaded.read(), uploaded.name)
        except ValueError as e:
            status.update(label="Stage 1 failed", state="error")
            st.error(str(e))
            st.stop()

        st.write("Stage 2: Cleaning pages...")
        preprocess_result = preprocess(ingest_result)

        st.write("Stage 3: Running Surya OCR (this may take a moment)...")
        surya_result = run_surya(preprocess_result)

        status.update(
            label=f"Stages 1–3 complete — {ingest_result.page_count} page(s) processed",
            state="complete",
        )

    st.subheader(f"{ingest_result.page_count} page(s) from `{ingest_result.source_name}`")

    for i, (orig, proc, surya_page) in enumerate(
        zip(ingest_result.page_images, preprocess_result.page_images, surya_result.pages)
    ):
        st.caption(
            f"Page {i + 1} — {len(surya_page.text_blocks)} block(s), {len(surya_page.tables)} table(s)"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(orig, caption="Original", use_container_width=True)
        with col2:
            st.image(proc, caption="Preprocessed", use_container_width=True)
        with col3:
            st.image(
                _draw_layout(proc, surya_page.text_blocks),
                caption="Layout detection",
                use_container_width=True,
            )

        if surya_page.ocr_text:
            with st.expander(f"OCR text — page {i + 1}"):
                st.markdown(surya_page.ocr_text, unsafe_allow_html=True)

        if surya_page.tables:
            with st.expander(f"Tables — page {i + 1} ({len(surya_page.tables)} detected)"):
                for j, tbl in enumerate(surya_page.tables):
                    st.write(f"Table {j + 1}: {len(tbl['rows'])} rows × {len(tbl['cols'])} cols")
                    if tbl.get("html"):
                        st.markdown(tbl["html"], unsafe_allow_html=True)
```

- [ ] **Step 2: Run full test suite to confirm no regressions**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run pytest -v
```

Expected: All 36 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: update app.py to show Stage 3 Surya layout detection overlay"
```
