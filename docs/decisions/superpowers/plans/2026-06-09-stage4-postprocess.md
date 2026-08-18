# Stage 4: Post-Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `NotImplementedError` stub in `postprocess.py` with a two-layer Khmer OCR correction pipeline: rule-based substitutions first, then a lazy-loaded Qwen2.5-VL-7B fallback triggered by foreign-script or missing-Khmer-numeral detection.

**Architecture:** Each `SuryaPageResult` passes through `_apply_rules` → `_detect_errors` → optionally `_qwen_correct`. The `raw_ocr_text` field is set once from `SuryaPageResult.ocr_text` and never modified. Module-level singletons hold the Qwen model so it loads only once across the entire document. All 13 tests mock `_get_qwen` — no real model runs during pytest.

**Tech Stack:** Python 3.11, mlx-lm>=0.21 (`mlx-community/Qwen2.5-7B-Instruct-4bit`), difflib (stdlib), dataclasses (stdlib), uv for dependency management.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `src/khmer_pipeline/models.py` | Modify | Add `correction_diff: str` field to `CorrectedPageResult` |
| `pyproject.toml` | Modify | Add `mlx-lm>=0.21` to dependencies |
| `tests/test_postprocess.py` | Create | 13 contract tests, all mocked |
| `src/khmer_pipeline/postprocess.py` | Modify | Full implementation replacing stub |
| `app.py` | Modify | Stage 4 UI: badge + corrected text + diff expander |

---

## Task 1: Add `correction_diff` field to `CorrectedPageResult`

**Files:**
- Modify: `src/khmer_pipeline/models.py:38-45`

Context: `CorrectedPageResult` currently has 6 fields. Add `correction_diff: str` between `corrected_text` and `qwen_used`. This must happen before tests or postprocess.py because both reference this field.

- [ ] **Step 1: Edit models.py**

Open `src/khmer_pipeline/models.py`. The current `CorrectedPageResult` dataclass (lines 38–45) looks like this:

```python
@dataclass
class CorrectedPageResult:
    page_index: int
    text_blocks: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    raw_ocr_text: str                   # copied from SuryaPageResult.ocr_text, unchanged
    corrected_text: str                 # after rule-based + optional Qwen2.5-VL pass
    qwen_used: bool                     # True if Qwen fallback fired for this page
```

Replace it with:

```python
@dataclass
class CorrectedPageResult:
    page_index: int
    text_blocks: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    raw_ocr_text: str                   # copied from SuryaPageResult.ocr_text, unchanged
    corrected_text: str                 # after rule-based + optional Qwen2.5-VL pass
    correction_diff: str                # difflib.ndiff output between raw and corrected
    qwen_used: bool                     # True if Qwen fallback fired for this page
```

- [ ] **Step 2: Verify the import works**

Run:
```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run python -c "from khmer_pipeline.models import CorrectedPageResult; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Run existing tests to confirm nothing broke**

Run:
```bash
uv run pytest -q --tb=short
```

Expected: all existing tests still pass (currently 37: 13 ingest + 11 preprocess + 13 surya including the phantom cell test).

- [ ] **Step 4: Commit**

```bash
git add src/khmer_pipeline/models.py
git commit -m "feat: add correction_diff field to CorrectedPageResult"
```

---

## Task 2: Add mlx-lm dependency

**Files:**
- Modify: `pyproject.toml:6-14`

- [ ] **Step 1: Edit pyproject.toml**

In the `dependencies` list, add `mlx-lm>=0.21` after the `bleach` entry:

```toml
dependencies = [
    "pymupdf>=1.24",
    "numpy>=1.26",
    "pillow>=10.0",
    "streamlit>=1.35",
    "opencv-python-headless>=4.8",
    "surya-ocr>=0.7",
    "bleach>=6.0",
    "mlx-lm>=0.21",
]
```

- [ ] **Step 2: Sync the environment**

Run:
```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv sync --extra dev
```

Expected: uv downloads and installs mlx-lm and its dependencies (mlx, transformers, etc). This may take a minute.

- [ ] **Step 3: Verify mlx-lm is importable**

Run:
```bash
uv run python -c "from mlx_lm import load, generate; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add mlx-lm>=0.21 for Qwen2.5-VL fallback"
```

---

## Task 3: Write all tests (failing — TDD)

**Files:**
- Create: `tests/test_postprocess.py`

All tests mock `_get_qwen` so no real model loads. The mock returns `(MagicMock(), MagicMock())`. Tests call `_detect_errors` and `_apply_rules` directly where relevant, and call `postprocess()` / `_correct_page()` for integration tests.

- [ ] **Step 1: Create tests/test_postprocess.py**

```python
from __future__ import annotations
from unittest.mock import MagicMock, patch
import khmer_pipeline.postprocess as pp
from khmer_pipeline.models import PreprocessResult, SuryaResult, SuryaPageResult, PostprocessResult, CorrectedPageResult
import numpy as np


def _make_surya_result(ocr_text: str = "ខ្មែរ") -> SuryaResult:
    page = SuryaPageResult(
        page_index=0,
        text_blocks=[],
        tables=[],
        ocr_text=ocr_text,
    )
    return SuryaResult(source_name="test.pdf", pages=[page])


def _mock_qwen():
    """Returns a patch context that makes _get_qwen return dummy model/tokenizer."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    return patch("khmer_pipeline.postprocess._get_qwen", return_value=(mock_model, mock_tokenizer))


# --- Contract tests ---

def test_postprocess_returns_postprocess_result():
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result())
    assert isinstance(r, PostprocessResult)


def test_raw_ocr_text_never_modified():
    original = "ខ្មែរ original"
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result(ocr_text=original))
    assert r.pages[0].raw_ocr_text == original


def test_rules_apply_correctly():
    saved = dict(pp.RULE_BASED_CORRECTIONS)
    pp.RULE_BASED_CORRECTIONS["WRONG"] = "RIGHT"
    try:
        result = pp._apply_rules("some WRONG text")
        assert result == "some RIGHT text"
    finally:
        pp.RULE_BASED_CORRECTIONS.clear()
        pp.RULE_BASED_CORRECTIONS.update(saved)


# --- _detect_errors: foreign script checks ---

def test_foreign_script_sinhala_triggers():
    # U+0D9A = ක (Sinhala letter)
    assert pp._detect_errors("normal text ක more") is True


def test_foreign_script_lao_triggers():
    # U+0E81 = Lao letter
    assert pp._detect_errors("text ກ here") is True


def test_latin_does_not_trigger():
    assert pp._detect_errors("CP ARDB 03-06-26 0.00%") is False


# --- _detect_errors: Khmer numeral check ---

def test_khmer_numeral_check_triggers():
    # 6 Arabic numerals, 0 Khmer numerals → should trigger
    text = "ទំនិញ 1 2 3 4 5 6 នៅ"
    assert pp._detect_errors(text) is True


def test_khmer_numeral_check_does_not_trigger_below_threshold():
    # 3 Arabic numerals (≤5), 0 Khmer numerals → should NOT trigger
    text = "ទំនិញ 1 2 3 នៅ"
    assert pp._detect_errors(text) is False


# --- qwen_used flag ---

def test_qwen_used_false_when_no_errors():
    # Pure Khmer text with no foreign scripts, no Arabic numeral excess
    clean = "ចំណូល ៣ ៤ ៥"  # uses Khmer numerals, no foreign script
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result(ocr_text=clean))
    assert r.pages[0].qwen_used is False


def test_qwen_used_true_when_errors():
    # Sinhala character forces Qwen path
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate", return_value="corrected") as _:
        mock_get.return_value = (MagicMock(), MagicMock())
        r = pp.postprocess(_make_surya_result(ocr_text="លකtext"))
    assert r.pages[0].qwen_used is True


def test_qwen_failure_falls_back_gracefully():
    # When generate raises, corrected_text should equal rule-based output, no crash
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate", side_effect=RuntimeError("GPU OOM")):
        mock_get.return_value = (MagicMock(), MagicMock())
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = pp.postprocess(_make_surya_result(ocr_text="លකtext"))
        assert len(w) == 1
        assert "Qwen correction failed" in str(w[0].message)
    # corrected_text equals rule-applied input (Qwen failed, returned input unchanged)
    assert r.pages[0].corrected_text == pp._apply_rules("លකtext")
    assert r.pages[0].qwen_used is True  # Qwen was attempted


def test_correction_diff_populated():
    # After any correction, correction_diff must be a string (may be empty if no change)
    with _mock_qwen():
        r = pp.postprocess(_make_surya_result(ocr_text="ខ្មែរ"))
    assert isinstance(r.pages[0].correction_diff, str)


def test_qwen_not_called_when_no_errors():
    # generate should never be called when _detect_errors returns False
    clean = "ចំណូល ៣ ៤ ៥"
    with patch("khmer_pipeline.postprocess._get_qwen") as mock_get, \
         patch("khmer_pipeline.postprocess.generate") as mock_gen:
        mock_get.return_value = (MagicMock(), MagicMock())
        pp.postprocess(_make_surya_result(ocr_text=clean))
    mock_gen.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they all fail**

Run:
```bash
uv run pytest tests/test_postprocess.py -v --tb=short
```

Expected: all 13 tests FAIL (ImportError or AttributeError on `pp._detect_errors`, `pp._apply_rules`, etc. — the stub raises NotImplementedError). If any test passes unexpectedly, investigate before proceeding.

---

## Task 4: Implement postprocess.py

**Files:**
- Modify: `src/khmer_pipeline/postprocess.py`

Replace the entire file content.

- [ ] **Step 1: Write the full implementation**

Replace `src/khmer_pipeline/postprocess.py` with:

```python
from __future__ import annotations
import difflib
import warnings
from typing import Any
from .models import SuryaResult, SuryaPageResult, PostprocessResult, CorrectedPageResult

# ---------------------------------------------------------------------------
# Rule table — deliberately empty. Add targeted pairs only after review.
# ---------------------------------------------------------------------------
RULE_BASED_CORRECTIONS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Qwen2.5-VL module-level singletons (lazy-loaded on first use)
# ---------------------------------------------------------------------------
_qwen_model = None
_qwen_tokenizer = None


def _get_qwen():
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        from mlx_lm import load
        _qwen_model, _qwen_tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    return _qwen_model, _qwen_tokenizer


# ---------------------------------------------------------------------------
# Correction layers
# ---------------------------------------------------------------------------

def _apply_rules(text: str) -> str:
    for wrong, correct in RULE_BASED_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text


def _detect_errors(text: str) -> bool:
    # Check A: foreign script characters
    for ch in text:
        cp = ord(ch)
        if (0x0D80 <= cp <= 0x0DFF or   # Sinhala
                0x0E80 <= cp <= 0x0EFF or   # Lao
                0x0E00 <= cp <= 0x0E7F or   # Thai
                0x1000 <= cp <= 0x109F or   # Myanmar
                0x0600 <= cp <= 0x06FF or   # Arabic
                0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
                0x3040 <= cp <= 0x30FF):    # Hiragana/Katakana
            return True

    # Check B: Arabic numerals present but no Khmer numerals
    arabic_count = sum(1 for ch in text if '0' <= ch <= '9')
    khmer_count = sum(1 for ch in text if '០' <= ch <= '៩')
    if arabic_count > 5 and khmer_count == 0:
        return True

    return False


def _qwen_correct(text: str) -> str:
    prompt = (
        "You are correcting Khmer OCR errors in Khmer financial document text.\n"
        "The document is a daily market price table from Cambodia.\n"
        "Fix misread characters, wrong scripts, and missing diacritics.\n"
        "Return only the corrected Khmer text with no explanation.\n\n"
        "Example 1:\n"
        "Wrong: \"មេន្គារំ ជពរជស\"\n"
        "Correct: \"ធនាគារ ARDB\"\n\n"
        "Example 2:\n"
        "Wrong: \"ពា សាច់ជ្រូករស់\"\n"
        "Correct: \"៣ សាច់ជ្រូករស់\"\n\n"
        "Example 3:\n"
        "Wrong: \"មាន្រ​ ​ ​ ​ ​ ​ ​ ​\"\n"
        "Correct: \"មាន់\"\n\n"
        f"Now correct this text:\n"
        f"Wrong: \"{text}\"\n"
        "Correct:"
    )
    try:
        from mlx_lm import generate
        model, tokenizer = _get_qwen()
        return generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
    except Exception as e:
        warnings.warn(f"Qwen correction failed: {e}")
        return text


def _build_diff(raw: str, corrected: str) -> str:
    diff = difflib.ndiff(raw.splitlines(), corrected.splitlines())
    return "\n".join(diff)


def _correct_page(page: SuryaPageResult) -> CorrectedPageResult:
    raw = page.ocr_text
    after_rules = _apply_rules(raw)
    if _detect_errors(after_rules):
        corrected = _qwen_correct(after_rules)
        qwen_used = True
    else:
        corrected = after_rules
        qwen_used = False
    diff = _build_diff(raw, corrected)
    return CorrectedPageResult(
        page_index=page.page_index,
        text_blocks=page.text_blocks,
        tables=page.tables,
        raw_ocr_text=raw,
        corrected_text=corrected,
        correction_diff=diff,
        qwen_used=qwen_used,
    )


def postprocess(result: SuryaResult) -> PostprocessResult:
    return PostprocessResult(
        source_name=result.source_name,
        pages=[_correct_page(page) for page in result.pages],
    )
```

- [ ] **Step 2: Run the full test suite**

Run:
```bash
uv run pytest -q --tb=short
```

Expected: all tests pass. Count should be 37 existing + 13 new = 50 passed.

If `test_qwen_used_true_when_errors` or `test_qwen_failure_falls_back_gracefully` fail, it is likely because `generate` is imported inside `_qwen_correct` (using a local import). Those two tests patch `khmer_pipeline.postprocess.generate` — the local import in `_qwen_correct` means the patch target must be on the module where `generate` is first bound. If the import is inside the function, patch `mlx_lm.generate` instead, or hoist the import to module level guarded by `try/except ImportError`. See the fix note below.

**Fix if `generate` patch fails:** Change the `_qwen_correct` function to import `generate` at module top-level:

At top of file, after the stdlib imports, add:
```python
try:
    from mlx_lm import generate
except ImportError:
    generate = None  # type: ignore[assignment]
```

Then in `_qwen_correct`, remove the `from mlx_lm import generate` local import and use the module-level `generate` directly:
```python
def _qwen_correct(text: str) -> str:
    ...
    try:
        model, tokenizer = _get_qwen()
        return generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
    except Exception as e:
        warnings.warn(f"Qwen correction failed: {e}")
        return text
```

This makes `khmer_pipeline.postprocess.generate` a real patchable name.

- [ ] **Step 3: Commit**

```bash
git add src/khmer_pipeline/postprocess.py tests/test_postprocess.py
git commit -m "feat: implement Stage 4 post-processing with rule-based + Qwen fallback"
```

---

## Task 5: Update app.py for Stage 4

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add postprocess import**

At the top of `app.py`, add the postprocess import after the surya import:

```python
from khmer_pipeline.surya import run_surya, models_loaded, preload_models
from khmer_pipeline.postprocess import postprocess
```

- [ ] **Step 2: Add Stage 4 call in the pipeline block**

In the `with st.status(...)` block, after `surya_result = run_surya(preprocess_result, on_page=_on_page)`, add:

```python
st.write("Running post-processing (rule-based + Qwen fallback if needed)...")
postprocess_result = postprocess(surya_result)
```

- [ ] **Step 3: Update the status label**

Change the `status.update(...)` call to:

```python
status.update(
    label=f"Stages 1–4 complete — {ingest_result.page_count} page(s) from {ingest_result.source_name}",
    state="complete",
)
```

- [ ] **Step 4: Update the caption**

Change the `st.caption(...)` line to include Stage 4:

```python
st.caption(
    "Stage 1 — Ingest  |  Stage 2 — Preprocess  |  Stage 3 — Surya OCR  |  Stage 4 — Post-process"
)
```

- [ ] **Step 5: Update the page loop**

The current loop iterates over `zip(ingest_result.page_images, preprocess_result.page_images, surya_result.pages)`. Extend it to include `postprocess_result.pages`:

```python
for i, (orig, proc, surya_page, post_page) in enumerate(
    zip(ingest_result.page_images, preprocess_result.page_images, surya_result.pages, postprocess_result.pages)
):
```

- [ ] **Step 6: Add Stage 4 expander per page**

After the existing `if surya_page.tables:` block (the last block in the per-page loop), add:

```python
        with st.expander(f"Post-processing — page {i + 1}"):
            if post_page.qwen_used:
                st.markdown("**⚡ Qwen correction applied**")
            else:
                st.markdown("**✓ rule-based only**")
            if post_page.corrected_text:
                st.markdown(_safe_html(post_page.corrected_text), unsafe_allow_html=True)
            if post_page.correction_diff:
                st.code(post_page.correction_diff, language="diff")
```

- [ ] **Step 7: Run the test suite to make sure app changes don't break imports**

Run:
```bash
uv run pytest -q --tb=short
```

Expected: 50 passed (all tests still pass — app.py is not tested directly).

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat: add Stage 4 post-processing UI to Streamlit app"
```

---

## Task 6: Smoke test in Streamlit

This task cannot be automated — it requires manual verification.

- [ ] **Step 1: Start the app**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run streamlit run app.py
```

- [ ] **Step 2: Upload the ARDB sample PDF**

Upload the test document. Verify the following in order:

1. Pipeline status shows "Running post-processing..." message
2. Status banner reads "Stages 1–4 complete — N page(s) from ardb_sample.pdf"
3. Each page has a "Post-processing" expander
4. Expander shows either "✓ rule-based only" or "⚡ Qwen correction applied" badge
5. Corrected text renders as HTML
6. Diff block is present (may be empty if no changes)
7. No Python errors in the terminal running Streamlit

- [ ] **Step 3: Verify Qwen does NOT fire on clean English/Khmer text**

Upload a document with only clean Khmer and Latin text (no foreign scripts). All pages should show "✓ rule-based only".

---

## Self-Review Checklist

Post-writing spec check (performed by plan author):

**Spec coverage:**
- ✅ `correction_diff` field added to models → Task 1
- ✅ `mlx-lm>=0.21` dependency → Task 2
- ✅ `RULE_BASED_CORRECTIONS` empty dict → Task 4 Step 1
- ✅ `_apply_rules` → Task 4 Step 1
- ✅ `_detect_errors` (both checks) → Task 4 Step 1
- ✅ Qwen singletons → Task 4 Step 1
- ✅ `_get_qwen` lazy loader → Task 4 Step 1
- ✅ `_qwen_correct` with prompt + error handling → Task 4 Step 1
- ✅ `_build_diff` → Task 4 Step 1
- ✅ `_correct_page` exact logic → Task 4 Step 1
- ✅ `postprocess` function → Task 4 Step 1
- ✅ All 13 tests → Task 3 Step 1
- ✅ app.py badge + corrected text + diff + status label → Task 5

**Placeholder scan:** None found.

**Type consistency:**
- `_get_qwen()` returns `(model, tokenizer)` — used in `_qwen_correct` ✅
- `_correct_page` returns `CorrectedPageResult` — used in `postprocess` list comprehension ✅
- `correction_diff: str` defined in Task 1, populated in `_correct_page` in Task 4 ✅
- `postprocess_result.pages` iteration in app.py uses `post_page.qwen_used`, `.corrected_text`, `.correction_diff` — all defined in `CorrectedPageResult` ✅
