# Batch Per-Region OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current per-region loop in `_process_page()` — which calls `rec_pred([crop], bboxes=[[[0, 0, crop_w, crop_h]]])` once per non-Table layout region — with a single batched `rec_pred(crops, bboxes=[...])` call covering all regions on the page.

**Architecture:** Only `_process_page()` in `src/khmer_pipeline/surya.py` changes. Two passes over `layout_result.bboxes`: first collect crops + offsets + layout boxes for all valid non-Table regions, then make one `rec_pred` call with all crops and zip the per-image results back to their regions for serialization. Sorting, `ocr_text` assembly, and table handling are unchanged.

**Tech Stack:** surya-ocr==0.17.1, Python 3.11, uv, pytest

---

## Background: Surya batched `rec_pred` call shape

`RecognitionPredictor.__call__(images, bboxes=...)` accepts a list of images and a parallel list of per-image bbox-lists, returning a list of `OCRResult` (one per input image, same order). For N crops:

```python
rec_pred(crops, bboxes=[[[0, 0, w0, h0]], [[0, 0, w1, h1]], ..., [[0, 0, wN, hN]]])
# -> [OCRResult_0, OCRResult_1, ..., OCRResult_N]
```

Each crop gets a single bbox covering the whole crop (same as today, just batched into one call).

---

## Known tradeoff (document, do not "fix")

Today, if `rec_pred` raises for one region's crop, only that region's text is lost (the `try/except` is per-region, inside the loop). After batching, a single `rec_pred` call covers the whole page's text regions — if it raises, **all** text regions on that page lose OCR text for that call (table cell OCR is a separate, unaffected call). This is accepted: the existing 50×20px filter already eliminates the degenerate-bbox failure mode that originally caused per-region exceptions, so a batch-level failure would likely indicate a page-wide problem (OOM, model error) that would affect every region anyway.

---

## Files

- **Modify:** `src/khmer_pipeline/surya.py` — `_process_page()` only
- **Modify:** `tests/test_surya.py` — add 1 new test

---

## Task 1: Write 1 failing test for batched OCR

**File:** `tests/test_surya.py`

- [ ] **Step 1: Append the new test to `tests/test_surya.py`**

Add at the end of the file:

```python
def test_per_region_ocr_batched_in_single_call():
    """Multiple non-Table regions are OCR'd in a single batched rec_pred call."""
    bbox1 = _make_layout_bbox_mock("Text")
    bbox1.bbox = [10.0, 10.0, 200.0, 50.0]
    bbox1.position = 1

    bbox2 = _make_layout_bbox_mock("Text")
    bbox2.bbox = [10.0, 100.0, 200.0, 150.0]
    bbox2.position = 2

    layout_result = MagicMock()
    layout_result.bboxes = [bbox1, bbox2]
    layout_pred = MagicMock(return_value=[layout_result])

    line1 = _make_text_line_mock(0)
    line1.text = "ខ្មែរ first"
    ocr_result_1 = MagicMock()
    ocr_result_1.text_lines = [line1]

    line2 = _make_text_line_mock(1)
    line2.text = "ខ្មែរ second"
    ocr_result_2 = MagicMock()
    ocr_result_2.text_lines = [line2]

    rec_pred = MagicMock(return_value=[ocr_result_1, ocr_result_2])
    table_pred = MagicMock(return_value=[])

    with patch("khmer_pipeline.surya._get_predictors",
               return_value=(layout_pred, rec_pred, table_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    # One batched call for all text regions on the page
    assert rec_pred.call_count == 1

    # Both regions' crops were passed in a single call
    call_args = rec_pred.call_args
    images_arg = call_args[0][0]
    bboxes_arg = call_args[1]["bboxes"]
    assert len(images_arg) == 2
    assert len(bboxes_arg) == 2

    # Both regions' text made it into the result
    texts = [b["text"] for b in r.pages[0].text_blocks]
    assert "ខ្មែរ first" in texts
    assert "ខ្មែរ second" in texts
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run pytest tests/test_surya.py::test_per_region_ocr_batched_in_single_call -v --tb=short
```

Expected: FAIL — under the current implementation, `rec_pred` is called once per region, so `rec_pred.call_count == 2`, not `1`.

- [ ] **Step 3: Run the full surya test file to confirm no other breakage**

```bash
uv run pytest tests/test_surya.py -v --tb=short
```

Expected: 17 passed, 1 failed (18 total).

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/test_surya.py
git commit -m "test: add failing test for batched per-region OCR"
```

---

## Task 2: Batch the per-region OCR calls in `_process_page()`

**File:** `src/khmer_pipeline/surya.py`

- [ ] **Step 1: Replace the per-region loop**

The current code (lines 59–80) is:

```python
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
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index}: {e}")
            continue
        for line in region_ocr.text_lines:
            block = _serialize_text_line(line)
            _adjust_coordinates(block, x0, y0)
            block["label"] = layout_bbox.label
            block["region_label"] = layout_bbox.label
            block["reading_order"] = layout_bbox.position
            text_blocks.append(block)
```

Replace it with:

```python
    # Collect crops for all non-Table, non-degenerate layout regions
    crops: list[Image.Image] = []
    regions: list = []
    offsets: list[tuple[float, float]] = []
    for layout_bbox in layout_result.bboxes:
        if layout_bbox.label == "Table":
            continue
        x0, y0, x1, y1 = layout_bbox.bbox
        if (x1 - x0) < 50 or (y1 - y0) < 20:
            continue
        crops.append(pil_img.crop((int(x0), int(y0), int(x1), int(y1))))
        regions.append(layout_bbox)
        offsets.append((x0, y0))

    # Batch all region crops into a single rec_pred call
    text_blocks: list[dict] = []
    if crops:
        bboxes_per_crop = [[[0, 0, c.size[0], c.size[1]]] for c in crops]
        try:
            region_ocr_results = rec_pred(crops, bboxes=bboxes_per_crop)
        except Exception as e:
            warnings.warn(f"Text OCR failed on page {page_index}: {e}")
            region_ocr_results = []
        for region_ocr, layout_bbox, (x0, y0) in zip(region_ocr_results, regions, offsets):
            for line in region_ocr.text_lines:
                block = _serialize_text_line(line)
                _adjust_coordinates(block, x0, y0)
                block["label"] = layout_bbox.label
                block["region_label"] = layout_bbox.label
                block["reading_order"] = layout_bbox.position
                text_blocks.append(block)
```

Everything below this point in `_process_page()` (the `_sort_key` function, `sorted_blocks`, `ocr_text`, table handling, and the `return SuryaPageResult(...)`) is unchanged — leave it exactly as-is.

- [ ] **Step 2: Run the new test — expect it to pass**

```bash
uv run pytest tests/test_surya.py::test_per_region_ocr_batched_in_single_call -v --tb=short
```

Expected: 1 passed.

- [ ] **Step 3: Run the full surya test file**

```bash
uv run pytest tests/test_surya.py -v --tb=short
```

Expected: 18 passed.

**Mock compatibility note:** `_make_predictors()` sets `rec_pred = MagicMock(return_value=[ocr_result])` and the default layout has exactly one non-Table "Text" bbox, so `crops` has 1 element and `rec_pred(crops, bboxes=...)` still returns the same 1-element list as before — all 17 existing tests continue to pass unchanged. `test_table_cells_get_ocr_text` and `test_small_region_skipped` are unaffected since `crops` is empty in both (Table-only layout, or region filtered out by size).

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest -q --tb=short
```

Expected: 68 passed (67 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add src/khmer_pipeline/surya.py
git commit -m "feat: batch per-region OCR into a single rec_pred call per page"
```

---

## Verification

```bash
uv run pytest -q --tb=short   # 68 passed
```

Manual check (optional):
- Upload a multi-region Khmer PDF in `uv run streamlit run app.py`
- OCR text and layout overlay should be unchanged from before — only the call pattern changed (1 batched `rec_pred` call per page for text regions, instead of one call per region)
