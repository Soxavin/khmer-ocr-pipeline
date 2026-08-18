# Per-Region OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-page OCR pass in `_process_page()` with per-region OCR — crop each detected layout region, run RecognitionPredictor on the crop, adjust coordinates back to full-page space, then reassemble in reading order.

**Architecture:** Only `surya.py` changes. `SuryaPageResult` contract is unchanged. Two new private helpers added: `_serialize_text_line()` and `_adjust_coordinates()`. `_build_ocr_text()` removed (inlined). Three new tests; existing 13 surya tests updated to reflect the new call pattern.

**Tech Stack:** surya-ocr==0.17.1, Python 3.11, uv, pytest

---

## API Corrections (spec vs. reality — verified via Context7)

The original spec contained three API errors. The implementation below corrects all three:

| Spec said | Actual Surya 0.17.1 API | Correction |
|-----------|-------------------------|------------|
| `rec_pred([crop])` | `rec_pred([crop])` raises `AssertionError` — requires `bboxes` or `det_predictor` | Use `rec_pred([crop], bboxes=[[[0, 0, crop_w, crop_h]]])[0]` |
| `region_ocr.blocks` | `OCRResult` has no `blocks` attribute | Use `region_ocr.text_lines` |
| `b["html"]` in ocr_text | `TextLine` has `text: str`, not `html` | Use `b["text"]` |

---

## Files

- **Modify:** `src/khmer_pipeline/surya.py` — replace `_process_page()`, add `_serialize_text_line()`, add `_adjust_coordinates()`, remove `_build_ocr_text()`
- **Modify:** `tests/test_surya.py` — update `_make_predictors()` mock comment, add 3 new tests

---

## Task 1: Write 3 new failing tests

**File:** `tests/test_surya.py`

No changes to existing tests or helpers yet — just append three new test functions.

- [ ] **Step 1: Append the three new tests to `tests/test_surya.py`**

Add at the end of the file:

```python
def test_small_region_skipped():
    """Layout bbox smaller than 50×20 pixels produces no text blocks."""
    tiny_bbox = _make_layout_bbox_mock("Text")
    tiny_bbox.bbox = [10.0, 10.0, 40.0, 25.0]   # 30×15 — below both thresholds

    layout_result = MagicMock()
    layout_result.bboxes = [tiny_bbox]
    layout_pred = MagicMock(return_value=[layout_result])

    ocr_result = MagicMock()
    ocr_result.text_lines = [_make_text_line_mock(0)]
    rec_pred = MagicMock(return_value=[ocr_result])

    table_pred = MagicMock(return_value=[])

    with patch("khmer_pipeline.surya._get_predictors",
               return_value=(layout_pred, rec_pred, table_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    assert r.pages[0].text_blocks == []
    rec_pred.assert_not_called()


def test_region_label_in_text_blocks():
    """Every text block must have a 'region_label' key."""
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=1))
    for block in r.pages[0].text_blocks:
        assert "region_label" in block, f"Block missing region_label: {block}"


def test_ocr_text_has_no_region_labels():
    """ocr_text must be plain text — layout label names must not appear as prefixes."""
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=1))
    ocr_text = r.pages[0].ocr_text
    for label in ("Text:", "Table:", "Title:", "Figure:", "Caption:", "Picture:"):
        assert label not in ocr_text, f"ocr_text contains label prefix '{label}'"
```

- [ ] **Step 2: Run just the new tests to verify they fail**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run pytest tests/test_surya.py::test_small_region_skipped tests/test_surya.py::test_region_label_in_text_blocks tests/test_surya.py::test_ocr_text_has_no_region_labels -v --tb=short
```

Expected: 3 failures. `test_region_label_in_text_blocks` and `test_ocr_text_has_no_region_labels` fail with `AssertionError` (key missing / text mismatch). `test_small_region_skipped` fails because rec_pred is called when it shouldn't be.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_surya.py
git commit -m "test: add 3 failing tests for per-region OCR (small-region skip, region_label, plain ocr_text)"
```

---

## Task 2: Implement per-region OCR in `surya.py`

**File:** `src/khmer_pipeline/surya.py`

**Step-by-step changes:**

- [ ] **Step 1: Add `_serialize_text_line()` helper**

Add after `_serialize_layout_box()` (around line 108):

```python
def _serialize_text_line(line) -> dict[str, Any]:
    return {
        "text": line.text,
        "bbox": list(line.bbox),
        "polygon": [list(p) for p in line.polygon],
        "confidence": line.confidence,
    }
```

- [ ] **Step 2: Add `_adjust_coordinates()` helper**

Add immediately after `_serialize_text_line()`:

```python
def _adjust_coordinates(block_dict: dict, offset_x: float, offset_y: float) -> dict:
    if block_dict.get("bbox"):
        b = block_dict["bbox"]
        block_dict["bbox"] = [b[0] + offset_x, b[1] + offset_y, b[2] + offset_x, b[3] + offset_y]
    if block_dict.get("polygon"):
        block_dict["polygon"] = [
            [p[0] + offset_x, p[1] + offset_y]
            for p in block_dict["polygon"]
        ]
    return block_dict
```

- [ ] **Step 3: Replace `_process_page()` with the per-region implementation**

Replace the entire `_process_page()` function (lines 50–105 in the current file) with:

```python
def _process_page(
    page_index: int,
    pil_img: Image.Image,
    layout_pred,
    rec_pred,
    table_pred,
) -> SuryaPageResult:
    layout_result = layout_pred([pil_img])[0]

    # Per-region OCR: crop each non-Table layout region, run rec_pred on the crop
    text_blocks: list[dict] = []
    for layout_bbox in layout_result.bboxes:
        if layout_bbox.label == "Table":
            continue
        x0, y0, x1, y1 = layout_bbox.bbox
        if (x1 - x0) < 50 or (y1 - y0) < 20:
            continue
        crop = pil_img.crop((int(x0), int(y0), int(x1), int(y1)))
        crop_w, crop_h = crop.size
        try:
            region_ocr = rec_pred([crop], bboxes=[[[0, 0, crop_w, crop_h]]])[0]
            for line in region_ocr.text_lines:
                block = _serialize_text_line(line)
                block = _adjust_coordinates(block, x0, y0)
                block["label"] = layout_bbox.label
                block["region_label"] = layout_bbox.label
                block["reading_order"] = layout_bbox.position
                text_blocks.append(block)
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index}: {e}")

    # Sort blocks: primary by reading_order (if set), fallback top-to-bottom left-to-right
    def _sort_key(block: dict) -> tuple:
        ro = block.get("reading_order") or 0
        bbox = block.get("bbox") or [0, 0, 0, 0]
        if ro > 0:
            return (0, ro, 0.0, 0.0)
        return (1, 0, bbox[1], bbox[0])

    sorted_blocks = sorted(text_blocks, key=_sort_key)

    # Plain text — no region labels embedded
    ocr_text = "\n\n".join(b["text"] for b in sorted_blocks if b.get("text"))

    # Table recognition (unchanged)
    table_bboxes = [b for b in layout_result.bboxes if b.label == "Table"]
    if table_bboxes:
        crops = [pil_img.crop(tuple(map(int, b.bbox))) for b in table_bboxes]
        table_results = table_pred(crops)
        for t, crop in zip(table_results, crops):
            if t.cells:
                try:
                    cell_bboxes = [list(map(int, c.bbox)) for c in t.cells]
                    cell_ocr = rec_pred([crop], bboxes=[cell_bboxes])[0]
                    for cell, line in zip(t.cells, cell_ocr.text_lines):
                        cell.text_lines = [{"text": line.text, "bbox": line.bbox}]
                except Exception as e:
                    warnings.warn(f"Cell OCR failed: {e}")
        tables = []
        for t in table_results:
            tbl = _serialize_table(t)
            tbl["cells"] = _filter_phantom_cells(tbl["cells"], tbl["image_bbox"])
            tables.append(tbl)
    else:
        tables = []

    return SuryaPageResult(
        page_index=page_index,
        text_blocks=sorted_blocks,
        tables=tables,
        ocr_text=ocr_text,
    )
```

- [ ] **Step 4: Remove `_build_ocr_text()`**

Delete the `_build_ocr_text()` function (it was `_build_ocr_text` at the bottom of the file). It's now inlined and no longer needed.

- [ ] **Step 5: Run the 3 new tests — expect them to pass**

```bash
uv run pytest tests/test_surya.py::test_small_region_skipped tests/test_surya.py::test_region_label_in_text_blocks tests/test_surya.py::test_ocr_text_has_no_region_labels -v --tb=short
```

Expected: 3 passed.

- [ ] **Step 6: Run all 16 surya tests**

```bash
uv run pytest tests/test_surya.py -v --tb=short
```

**Known mock compatibility notes for the existing 13 tests:**

- `test_run_surya_block_has_required_keys` checks `("label", "bbox", "polygon", "reading_order")`. The new blocks set all four from the layout bbox. ✓
- `test_run_surya_ocr_text_contains_khmer` — mock TextLine has `text = "ខ្មែរ 0"`, which becomes `ocr_text`. ✓
- `test_table_cells_get_ocr_text` checks `rec_pred.call_count == 1` — layout has only a Table bbox, so no per-region text calls; rec_pred called once for cell OCR only. ✓

If any of the 13 existing tests fail, diagnose before proceeding. The most likely failure is `_make_predictors()` — the mock `rec_pred = MagicMock(return_value=[ocr_result])` is called once per non-table layout bbox (one "Text" bbox in the default setup), so `return_value` works correctly for repeated calls.

- [ ] **Step 7: Run the full test suite**

```bash
uv run pytest -q --tb=short
```

Expected: 67 passed (64 existing + 3 new).

- [ ] **Step 8: Commit**

```bash
git add src/khmer_pipeline/surya.py
git commit -m "feat: per-region OCR in _process_page() — crop each layout region before rec_pred"
```

---

## Verification

After both commits:

```bash
uv run pytest -q --tb=short   # 67 passed
```

Manual check (optional):
- Upload a Khmer PDF in `uv run streamlit run app.py`
- Layout overlay should look the same (text_blocks still have `label` and full-page `bbox`)
- OCR text expander should show plain Khmer text without any "Text:" / "Table:" prefixes
