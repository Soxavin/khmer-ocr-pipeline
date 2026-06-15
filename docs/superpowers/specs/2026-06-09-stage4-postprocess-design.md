# Stage 4 Post-Processing Design

## Overview

Stage 4 takes a `SuryaResult` (raw Surya OCR output) and returns a `PostprocessResult` (corrected text, per page). Each page passes through two correction layers in sequence: rule-based substitutions first, then a Qwen2.5-VL-7B fallback if error detection fires. The `raw_ocr_text` field is set exactly once from `SuryaPageResult.ocr_text` and is never modified.

**Pipeline position:** `SuryaResult → postprocess() → PostprocessResult`

---

## Models Change

Add `correction_diff: str` to `CorrectedPageResult` in `models.py`. This is the only change to the models file.

```python
@dataclass
class CorrectedPageResult:
    page_index: int
    text_blocks: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    raw_ocr_text: str         # copied from SuryaPageResult.ocr_text, NEVER modified
    corrected_text: str       # output after all correction layers
    correction_diff: str      # difflib.ndiff output between raw and corrected
    qwen_used: bool           # True if Qwen fallback fired
```

---

## Dependency

Add `mlx-lm>=0.21` to `pyproject.toml` dependencies. The mlx-lm API used:

```python
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
corrected = generate(model, tokenizer, prompt=prompt_str, max_tokens=512, verbose=False)
```

`load()` returns `(model, tokenizer)`. `generate()` returns a `str`.

---

## Components

### `RULE_BASED_CORRECTIONS: dict[str, str]`

Empty dict. Structure is kept for future targeted substitutions. No broad character substitutions will be added without context-aware regex review.

### `_apply_rules(text: str) -> str`

Iterates over `RULE_BASED_CORRECTIONS` and applies `str.replace(wrong, correct)` for each pair. Returns text unchanged when dict is empty.

### `_detect_errors(text: str) -> bool`

Returns `True` if either check fires (OR logic):

**Check A — Foreign script detection:**
Returns `True` if any character falls in:
- Sinhala: U+0D80–U+0DFF
- Lao: U+0E80–U+0EFF
- Thai: U+0E00–U+0E7F
- Myanmar: U+1000–U+109F
- Arabic: U+0600–U+06FF
- CJK Unified Ideographs: U+4E00–U+9FFF
- Hiragana/Katakana: U+3040–U+30FF

Must NOT trigger on: Khmer (U+1780–U+17FF), Latin (U+0000–U+007F), Arabic numerals 0–9, common punctuation. Strings like `"CP ARDB 03-06-26 0.00%"` return `False`.

**Check B — Missing Khmer numerals:**
Count Arabic numerals (0–9) in text. Count Khmer numerals (U+17E0–U+17E9, i.e. ០–៩). If Arabic count > 5 AND Khmer count == 0, return `True`. Financial documents use Khmer numerals for row numbering — absence when Arabic numerals are present indicates misreading.

### `_qwen_model`, `_qwen_tokenizer` (module-level globals)

Both initialised to `None`. Lazy-loaded on first Qwen call.

### `_get_qwen() -> tuple[Any, Any]`

Lazy loader. On first call, imports and calls `load("mlx-community/Qwen2.5-7B-Instruct-4bit")`, stores in globals, returns `(model, tokenizer)`.

### `_qwen_correct(text: str) -> str`

Builds a 3-example few-shot prompt instructing correction of Khmer OCR errors in financial document context. Calls `generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)`. Wrapped in `try/except`: on any exception, calls `warnings.warn(f"Qwen correction failed: {e}")` and returns the input text unchanged.

**Prompt template:**
```
You are correcting Khmer OCR errors in Khmer financial document text.
The document is a daily market price table from Cambodia.
Fix misread characters, wrong scripts, and missing diacritics.
Return only the corrected Khmer text with no explanation.

Example 1:
Wrong: "មេន្គារំ ជពរជស"
Correct: "ធនាគារ ARDB"

Example 2:
Wrong: "ពា សាច់ជ្រូករស់"
Correct: "៣ សាច់ជ្រូករស់"

Example 3:
Wrong: "មាន្រ​ ​ ​ ​ ​ ​ ​ ​"
Correct: "មាន់"

Now correct this text:
Wrong: "{text}"
Correct:
```

### `_build_diff(raw: str, corrected: str) -> str`

```python
import difflib
diff = difflib.ndiff(raw.splitlines(), corrected.splitlines())
return "\n".join(diff)
```

### `_correct_page(page: SuryaPageResult) -> CorrectedPageResult`

```python
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
```

### `postprocess(result: SuryaResult) -> PostprocessResult`

```python
def postprocess(result: SuryaResult) -> PostprocessResult:
    return PostprocessResult(
        source_name=result.source_name,
        pages=[_correct_page(page) for page in result.pages],
    )
```

---

## Error Handling

- `_qwen_correct` catches all exceptions, warns, returns input unchanged. Pipeline never crashes due to Qwen failure.
- Rule corrections are pure string operations — no failure path.
- `_detect_errors` is pure character iteration — no failure path.

---

## Testing

All tests in `tests/test_postprocess.py`. Mock `_get_qwen` — no real model runs during pytest.

Required test cases (13 total):

| Test | What it verifies |
|------|-----------------|
| `test_postprocess_returns_postprocess_result` | Correct return type |
| `test_raw_ocr_text_never_modified` | `raw_ocr_text` equals original `ocr_text` |
| `test_rules_apply_correctly` | Temporary rule in test applies via `_apply_rules` |
| `test_foreign_script_sinhala_triggers` | Sinhala char → `_detect_errors` True |
| `test_foreign_script_lao_triggers` | Lao char → `_detect_errors` True |
| `test_latin_does_not_trigger` | `"CP ARDB 03-06-26 0.00%"` → `_detect_errors` False |
| `test_khmer_numeral_check_triggers` | 6 Arabic + 0 Khmer numerals → True |
| `test_khmer_numeral_check_does_not_trigger_below_threshold` | 3 Arabic + 0 Khmer numerals → False |
| `test_qwen_used_false_when_no_errors` | Clean Khmer text → `qwen_used = False` |
| `test_qwen_used_true_when_errors` | Sinhala char → `qwen_used = True` |
| `test_qwen_failure_falls_back_gracefully` | Qwen raises Exception → `corrected_text` = rule output, no crash |
| `test_correction_diff_populated` | `correction_diff` is non-empty string after correction |
| `test_qwen_not_called_when_no_errors` | Qwen never called when `_detect_errors` False |

---

## app.py Changes

Import `postprocess` from `khmer_pipeline.postprocess`. Call it after `run_surya`. For each page, add a new expander section with:

1. Badge: `"✓ rule-based only"` (green) if `qwen_used = False`, `"⚡ Qwen correction applied"` (yellow) if `True`
2. Corrected text rendered as markdown
3. Diff rendered with `st.code(page.correction_diff, language="diff")`

Update caption to include `"Stage 4 — Post-process"`. Update `status.update` label to `"Stages 1–4 complete"`.

---

## Commit Sequence

1. `models.py` — add `correction_diff` field
2. `pyproject.toml` — add `mlx-lm>=0.21`, run `uv sync --extra dev`
3. `tests/test_postprocess.py` — all 13 tests (failing)
4. `src/khmer_pipeline/postprocess.py` — full implementation (tests pass)
5. `app.py` — Stage 4 UI integration
