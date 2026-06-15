# Stage 2: Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `preprocess.py` — takes an `IngestResult`, applies configurable image cleaning steps (stamp removal, sharpening, contrast normalisation), and returns a `PreprocessResult`.

**Architecture:** A `PreprocessConfig` dataclass holds boolean flags for each step; all default to `True`. The main `preprocess()` function iterates pages and calls `_preprocess_image()`, which converts RGB→BGR, applies enabled steps, and converts back to RGB. Each step is a private function operating on a BGR numpy array. OpenCV is used for all image operations.

**Tech Stack:** Python 3.11+, opencv-python-headless>=4.8, numpy>=1.26, existing `IngestResult`/`PreprocessResult` dataclasses from `models.py`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `opencv-python-headless>=4.8` dependency |
| `src/khmer_pipeline/preprocess.py` | Modify | `PreprocessConfig`, `preprocess()`, `_remove_stamps()`, `_sharpen()`, `_normalise()` |
| `tests/test_preprocess.py` | Create | All preprocessing tests |
| `app.py` | Modify | Add Stage 2 call, show before/after thumbnails side by side |

---

### Task 1: Add opencv-python-headless dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependency to pyproject.toml**

Edit `pyproject.toml`, add `"opencv-python-headless>=4.8"` to the `dependencies` list:

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
    "opencv-python-headless>=4.8",
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

- [ ] **Step 2: Sync dependencies**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv sync --extra dev
```

Expected: `Resolved ... packages` with no errors. OpenCV wheel downloads and installs.

- [ ] **Step 3: Verify import**

```bash
uv run python -c "import cv2; print(cv2.__version__)"
```

Expected: prints a version string like `4.x.x`, no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add opencv-python-headless dependency"
```

---

### Task 2: PreprocessConfig + skeleton + base contract tests

**Files:**
- Modify: `src/khmer_pipeline/preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_preprocess.py`:

```python
from __future__ import annotations
import numpy as np
import pytest
from khmer_pipeline.models import IngestResult, PreprocessResult
from khmer_pipeline.preprocess import PreprocessConfig, preprocess


def _make_ingest_result(n_pages: int = 1, h: int = 100, w: int = 100) -> IngestResult:
    """Creates an IngestResult with solid-colour pages (non-flat gradient for reliable tests)."""
    row = np.arange(w, dtype=np.uint8).reshape(1, w)
    channel = np.tile(row, (h, 1))
    img = np.stack([channel, channel, channel], axis=2)
    return IngestResult(
        source_name="test.pdf",
        page_images=[img.copy() for _ in range(n_pages)],
        dpi=200,
        page_count=n_pages,
    )


def test_preprocess_returns_preprocess_result():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert isinstance(r, PreprocessResult)


def test_preprocess_preserves_source_name():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert r.source_name == "test.pdf"


def test_preprocess_preserves_dpi():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert r.dpi == 200


def test_preprocess_preserves_page_count():
    r = preprocess(_make_ingest_result(3), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert r.page_count == 3
    assert len(r.page_images) == 3


def test_preprocess_image_shape_unchanged():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert r.page_images[0].shape == (100, 100, 3)


def test_preprocess_images_are_rgb_uint8():
    r = preprocess(_make_ingest_result(), PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    arr = r.page_images[0]
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3


def test_preprocess_all_flags_false_is_passthrough():
    ingest_r = _make_ingest_result()
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=False, normalise=False))
    assert np.array_equal(r.page_images[0], original)


def test_preprocess_default_config_does_not_raise():
    # Smoke test: default config (all True) must not raise an exception
    preprocess(_make_ingest_result())
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run pytest tests/test_preprocess.py -v
```

Expected: All 8 tests FAIL with `NotImplementedError` (the current stub raises it immediately).

- [ ] **Step 3: Implement skeleton**

Replace all of `src/khmer_pipeline/preprocess.py`:

```python
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from .models import IngestResult, PreprocessResult


@dataclass
class PreprocessConfig:
    remove_stamps: bool = True
    sharpen: bool = True
    normalise: bool = True


def preprocess(result: IngestResult, config: PreprocessConfig | None = None) -> PreprocessResult:
    if config is None:
        config = PreprocessConfig()
    processed = [_preprocess_image(img, config) for img in result.page_images]
    return PreprocessResult(
        source_name=result.source_name,
        page_images=processed,
        dpi=result.dpi,
        page_count=result.page_count,
    )


def _preprocess_image(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if cfg.remove_stamps:
        bgr = _remove_stamps(bgr)
    if cfg.sharpen:
        bgr = _sharpen(bgr)
    if cfg.normalise:
        bgr = _normalise(bgr)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _remove_stamps(bgr: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def _sharpen(bgr: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def _normalise(bgr: np.ndarray) -> np.ndarray:
    raise NotImplementedError
```

- [ ] **Step 4: Run tests — all 8 should pass except the smoke test**

```bash
uv run pytest tests/test_preprocess.py -v
```

Expected: 7 tests PASS (the 7 that use `remove_stamps=False, sharpen=False, normalise=False`). `test_preprocess_default_config_does_not_raise` FAILS with `NotImplementedError` — that is expected at this stage; it will pass after Tasks 3–5.

- [ ] **Step 5: Commit**

```bash
git add src/khmer_pipeline/preprocess.py tests/test_preprocess.py
git commit -m "feat: add PreprocessConfig and preprocess() skeleton with contract tests"
```

---

### Task 3: Implement stamp removal

**Files:**
- Modify: `src/khmer_pipeline/preprocess.py` — implement `_remove_stamps()`
- Modify: `tests/test_preprocess.py` — add stamp removal test

Red HSV wraps around 0°/180° in OpenCV, so two ranges are needed:
- Lower red: H=[0,10], S=[100,255], V=[100,255]
- Upper red: H=[160,180], S=[100,255], V=[100,255]
- Blue: H=[100,130], S=[100,255], V=[100,255]

The mask is dilated to cover stamp edges before inpainting.

- [ ] **Step 1: Add stamp removal test to tests/test_preprocess.py**

Add this helper and test at the bottom of `tests/test_preprocess.py`:

```python
def _make_red_blob_image() -> IngestResult:
    """100x100 white image with a 20x20 red square in the centre."""
    img = np.full((100, 100, 3), 240, dtype=np.uint8)  # off-white background
    img[40:60, 40:60] = [255, 0, 0]  # solid red blob in RGB
    return IngestResult(
        source_name="stamp_test.pdf",
        page_images=[img],
        dpi=200,
        page_count=1,
    )


def test_stamp_removal_changes_red_region():
    ingest_r = _make_red_blob_image()
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=True, sharpen=False, normalise=False))
    # The output image must differ from the original — the red blob was inpainted
    assert not np.array_equal(r.page_images[0], original)
```

- [ ] **Step 2: Run the new test to confirm it fails**

```bash
uv run pytest tests/test_preprocess.py::test_stamp_removal_changes_red_region -v
```

Expected: FAIL with `NotImplementedError` from `_remove_stamps`.

- [ ] **Step 3: Implement _remove_stamps in preprocess.py**

Replace the `_remove_stamps` stub:

```python
def _remove_stamps(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around H=0/180 in OpenCV HSV
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))

    combined = cv2.bitwise_or(mask_red, mask_blue)

    # Dilate to cover stamp edges and bleed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(combined, kernel, iterations=2)

    if cv2.countNonZero(dilated) == 0:
        return bgr  # nothing to inpaint

    return cv2.inpaint(bgr, dilated, 5, cv2.INPAINT_TELEA)
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/test_preprocess.py -v
```

Expected: All tests that were passing before still pass. `test_stamp_removal_changes_red_region` now PASSES. `test_preprocess_default_config_does_not_raise` still FAILS (sharpen and normalise still unimplemented) — that is expected.

- [ ] **Step 5: Commit**

```bash
git add src/khmer_pipeline/preprocess.py tests/test_preprocess.py
git commit -m "feat: implement stamp removal via HSV thresholding and TELEA inpainting"
```

---

### Task 4: Implement sharpening

**Files:**
- Modify: `src/khmer_pipeline/preprocess.py` — implement `_sharpen()`
- Modify: `tests/test_preprocess.py` — add sharpening test

Uses a 3×3 unsharp-mask kernel: `[[0,-1,0],[-1,5,-1],[0,-1,0]]`. Applied with `cv2.filter2D(..., ddepth=-1, ...)` which preserves uint8 dtype and clips values automatically.

Note: a flat (uniform-colour) image is unchanged by this kernel. The test fixture uses a gradient image (values 0–99) so sharpening produces measurable changes.

- [ ] **Step 1: Add sharpening test to tests/test_preprocess.py**

Append to `tests/test_preprocess.py`:

```python
def test_sharpen_changes_pixels():
    ingest_r = _make_ingest_result()  # gradient image, not flat
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=True, normalise=False))
    assert not np.array_equal(r.page_images[0], original)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_preprocess.py::test_sharpen_changes_pixels -v
```

Expected: FAIL with `NotImplementedError` from `_sharpen`.

- [ ] **Step 3: Implement _sharpen in preprocess.py**

Replace the `_sharpen` stub:

```python
def _sharpen(bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(bgr, ddepth=-1, kernel=kernel)
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/test_preprocess.py -v
```

Expected: All previous passing tests still pass. `test_sharpen_changes_pixels` now PASSES. `test_preprocess_default_config_does_not_raise` still FAILS (normalise still unimplemented).

- [ ] **Step 5: Commit**

```bash
git add src/khmer_pipeline/preprocess.py tests/test_preprocess.py
git commit -m "feat: implement sharpening via unsharp-mask filter2D kernel"
```

---

### Task 5: Implement contrast normalisation (CLAHE)

**Files:**
- Modify: `src/khmer_pipeline/preprocess.py` — implement `_normalise()`
- Modify: `tests/test_preprocess.py` — add normalisation test

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the **L channel in LAB colour space only**. This enhances local contrast without shifting hue or saturation — critical for Khmer text visibility on stained paper.

- [ ] **Step 1: Add normalisation test to tests/test_preprocess.py**

Append to `tests/test_preprocess.py`:

```python
def test_normalise_changes_pixels():
    ingest_r = _make_ingest_result()  # gradient image gives CLAHE something to work with
    original = ingest_r.page_images[0].copy()
    r = preprocess(ingest_r, PreprocessConfig(remove_stamps=False, sharpen=False, normalise=True))
    assert not np.array_equal(r.page_images[0], original)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_preprocess.py::test_normalise_changes_pixels -v
```

Expected: FAIL with `NotImplementedError` from `_normalise`.

- [ ] **Step 3: Implement _normalise in preprocess.py**

Replace the `_normalise` stub:

```python
def _normalise(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/test_preprocess.py -v
```

Expected: **All 11 tests PASS**, including `test_preprocess_default_config_does_not_raise`.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
uv run pytest -v
```

Expected: All 24 tests (13 ingest + 11 preprocess) PASS.

- [ ] **Step 6: Commit**

```bash
git add src/khmer_pipeline/preprocess.py tests/test_preprocess.py
git commit -m "feat: implement CLAHE normalisation on LAB L-channel"
```

---

### Task 6: Update app.py — before/after comparison UI

**Files:**
- Modify: `app.py`

Show Stage 1 (original) and Stage 2 (preprocessed) thumbnails side by side per page so the cleaning effect is visible. Each page gets a two-column layout: "Original" left, "Preprocessed" right.

- [ ] **Step 1: Replace app.py**

```python
from __future__ import annotations
import streamlit as st
from khmer_pipeline.ingest import ingest
from khmer_pipeline.preprocess import preprocess

st.set_page_config(page_title="Khmer Document Extraction", layout="wide")
st.title("Khmer Document Extraction Pipeline")
st.caption("Stage 1 — Ingest  |  Stage 2 — Preprocess")

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

        status.update(
            label=f"Stages 1–2 complete — {ingest_result.page_count} page(s) processed",
            state="complete",
        )

    st.subheader(f"{ingest_result.page_count} page(s) from `{ingest_result.source_name}`")

    for i, (orig, proc) in enumerate(
        zip(ingest_result.page_images, preprocess_result.page_images)
    ):
        st.caption(f"Page {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig, caption="Original", width="stretch")
        with col2:
            st.image(proc, caption="Preprocessed", width="stretch")
```

- [ ] **Step 2: Verify full test suite still passes**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run pytest -v
```

Expected: All 24 tests PASS. (app.py is not under test — it's verified manually.)

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: update app.py to show Stage 2 before/after comparison"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| Stamp removal via HSV thresholding | Task 3 |
| Red and blue stamp detection | Task 3 — both ranges covered |
| Inpaint underneath stamp | Task 3 — INPAINT_TELEA |
| Sharpening for text edges | Task 4 |
| Contrast normalisation for stains | Task 5 — CLAHE on L channel |
| Each step independently toggleable | Task 2 — PreprocessConfig flags |
| Skippable per document | Task 2 — `if cfg.X:` guards |
| TDD: tests before implementation | All tasks follow red→green order |
| Add opencv-python-headless (headless) | Task 1 |
| Before/after in Streamlit UI | Task 6 |

**Type consistency check:**
- `PreprocessConfig` defined once in Task 2, imported in tests and app.py — consistent throughout.
- `_remove_stamps`, `_sharpen`, `_normalise` all take `np.ndarray` (BGR) and return `np.ndarray` (BGR) — consistent.
- `preprocess()` signature `(IngestResult, PreprocessConfig | None) -> PreprocessResult` — consistent across all tasks.

**No placeholders:** All steps contain complete code. No TBDs.
