# Stage 6 — UI Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `app.py` genuinely usable for data analysts by adding session state caching, sidebar controls, table selection, manual text correction, and per-stage error handling.

**Architecture:** All changes are in `app.py` (plus one backend parameter on `postprocess.py`). No new pipeline logic — pure UI wiring. The full pipeline runs only once per settings combination; Streamlit interactions (download clicks, expander toggles) use cached results from `st.session_state`.

**Tech Stack:** Streamlit, Python 3.11, uv. Existing pipeline: `ingest`, `preprocess` (with `PreprocessConfig`), `run_surya`, `postprocess`, `export`.

---

## Files

- **Modify:** `src/khmer_pipeline/postprocess.py` — add `skip_qwen: bool = False` parameter
- **Modify:** `app.py` — all UI changes (session state, sidebar, layout, selection, editing, errors)

No new files. No changes to `ingest.py`, `preprocess.py` (except calling it with `PreprocessConfig`), `surya.py`, `export.py`, or `models.py`.

---

## Task 1: Add `skip_qwen` parameter to `postprocess.py`

**Files:**
- Modify: `src/khmer_pipeline/postprocess.py:101-126`

This is the only backend change. Adding `skip_qwen=True` bypasses `_detect_errors` and `_qwen_correct`, using the rule-based output directly as `corrected_text`.

- [ ] **Step 1: Edit `_correct_page` to accept and apply `skip_qwen`**

Replace `_correct_page` (lines 101–119) with:

```python
def _correct_page(page: SuryaPageResult, skip_qwen: bool = False) -> CorrectedPageResult:
    raw = page.ocr_text
    after_rules = _apply_rules(raw)
    if not skip_qwen and _detect_errors(after_rules):
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

- [ ] **Step 2: Edit `postprocess` to accept and forward `skip_qwen`**

Replace the `postprocess` function (lines 122–126) with:

```python
def postprocess(result: SuryaResult, skip_qwen: bool = False) -> PostprocessResult:
    return PostprocessResult(
        source_name=result.source_name,
        pages=[_correct_page(page, skip_qwen=skip_qwen) for page in result.pages],
    )
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd /Users/vin/Internship/khmer-ocr-pipeline
uv run pytest -q --tb=short
```

Expected: `64 passed`

- [ ] **Step 4: Commit**

```bash
git add src/khmer_pipeline/postprocess.py
git commit -m "feat: add skip_qwen parameter to postprocess()"
```

---

## Task 2: Session state caching

**Files:**
- Modify: `app.py`

Currently the full pipeline reruns on every Streamlit interaction. This task adds a `settings_key` check so the pipeline runs only when the uploaded file or any setting changes.

The settings key is computed from all inputs that affect pipeline output. Any change to it triggers a full re-run.

- [ ] **Step 1: Replace the pipeline execution block in `app.py`**

Replace the entire `if uploaded is not None:` block (lines 53–153) with this structure. Keep all existing imports at the top unchanged. This task only adds the cache skeleton — sidebar variables (`dpi`, `page_selection`, etc.) are hardcoded to defaults for now and will be wired in Task 3.

```python
if uploaded is not None:
    # Settings key — hardcoded defaults until Task 3 adds the sidebar
    dpi = 200
    page_selection = "All pages"
    page_num = 1
    page_start = 1
    page_end = 1
    remove_stamps = True
    sharpen = True
    normalise = True
    tables_only = False
    enable_qwen = True

    page_sel_part = "all"
    settings_key = f"{uploaded.name}_{dpi}_{page_sel_part}_{remove_stamps}_{sharpen}_{normalise}_{enable_qwen}"

    if st.session_state.get("last_key") != settings_key:
        with st.status("Running pipeline...", expanded=True) as status:
            st.write("Converting pages to images...")
            try:
                ingest_result = ingest(uploaded.read(), uploaded.name, dpi=dpi)
            except ValueError as e:
                status.update(label="Stage 1 failed", state="error")
                st.error(str(e))
                st.stop()

            # Page selection (all pages for now — Task 3 wires the sidebar)
            selected_indices = list(range(ingest_result.page_count))
            filtered_ingest = IngestResult(
                source_name=ingest_result.source_name,
                page_images=[ingest_result.page_images[i] for i in selected_indices],
                dpi=ingest_result.dpi,
                page_count=len(selected_indices),
            )

            st.write("Cleaning pages...")
            from khmer_pipeline.preprocess import PreprocessConfig
            config = PreprocessConfig(remove_stamps=remove_stamps, sharpen=sharpen, normalise=normalise)
            preprocess_result = preprocess(filtered_ingest, config)

            if not models_loaded():
                st.write("Loading Surya models — first run takes about a minute...")
            preload_models()

            def _on_page(idx: int, total: int) -> None:
                st.write(f"Page {idx + 1} / {total}: running OCR...")

            surya_result = run_surya(preprocess_result, on_page=_on_page)

            st.write("Running post-processing...")
            postprocess_result = postprocess(surya_result, skip_qwen=not enable_qwen)

            st.write("Exporting structured output...")
            export_result = export(postprocess_result)

            st.session_state["ingest_result"] = ingest_result
            st.session_state["filtered_ingest"] = filtered_ingest
            st.session_state["preprocess_result"] = preprocess_result
            st.session_state["surya_result"] = surya_result
            st.session_state["postprocess_result"] = postprocess_result
            st.session_state["export_result"] = export_result
            st.session_state["last_key"] = settings_key

            status.update(
                label=f"Stages 1–5 complete — {filtered_ingest.page_count} page(s) from {ingest_result.source_name}",
                state="complete",
            )
    else:
        ingest_result = st.session_state["ingest_result"]
        filtered_ingest = st.session_state["filtered_ingest"]
        preprocess_result = st.session_state["preprocess_result"]
        surya_result = st.session_state["surya_result"]
        postprocess_result = st.session_state["postprocess_result"]
        export_result = st.session_state["export_result"]

    st.subheader(f"{filtered_ingest.page_count} page(s) from `{ingest_result.source_name}`")

    for i, (orig, proc, surya_page, post_page) in enumerate(
        zip(filtered_ingest.page_images, preprocess_result.page_images, surya_result.pages, postprocess_result.pages)
    ):
        st.caption(
            f"Page {surya_page.page_index + 1} — {len(surya_page.text_blocks)} block(s), {len(surya_page.tables)} table(s)"
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

        if not tables_only and surya_page.ocr_text:
            with st.expander(f"OCR text — page {surya_page.page_index + 1}"):
                st.markdown(_safe_html(surya_page.ocr_text), unsafe_allow_html=True)

        if surya_page.tables:
            with st.expander(f"Tables — page {surya_page.page_index + 1} ({len(surya_page.tables)} detected)"):
                for j, tbl in enumerate(surya_page.tables):
                    st.write(f"Table {j + 1}: {len(tbl['rows'])} rows × {len(tbl['cols'])} cols")
                    cells = tbl["cells"]
                    if cells:
                        max_row = max(c["row_id"] for c in cells) + 1
                        max_col = max((c.get("col_id") or 0) for c in cells) + 1
                        grid = [[""] * max_col for _ in range(max_row)]
                        for c in cells:
                            r = c["row_id"]
                            col = c.get("col_id") or 0
                            if c.get("text_lines"):
                                text = " ".join(
                                    t["text"] for t in c["text_lines"] if t.get("text")
                                ).strip()
                                if 0 <= r < max_row and 0 <= col < max_col:
                                    grid[r][col] = text
                        st.dataframe(grid)

        with st.expander(f"Post-processing — page {surya_page.page_index + 1}"):
            if post_page.qwen_used:
                st.markdown("**⚡ Qwen correction applied**")
            else:
                st.markdown("**✓ rule-based only**")
            if post_page.corrected_text:
                st.write(post_page.corrected_text)
            if post_page.correction_diff:
                st.code(post_page.correction_diff, language="diff")

    st.subheader("Downloads")
    st.download_button(
        label="⬇ Download document JSON",
        data=json.dumps(export_result.document_json, ensure_ascii=False, indent=2),
        file_name=f"{Path(uploaded.name).stem}_extracted.json",
        mime="application/json",
    )
    for table_id, csv_string in export_result.tables_csv:
        st.download_button(
            label=f"⬇ Download {table_id}.csv",
            data=csv_string.encode("utf-8-sig"),
            file_name=f"{table_id}.csv",
            mime="text/csv",
        )
```

Also update the imports at the top of `app.py`. Replace:

```python
from khmer_pipeline.preprocess import preprocess
```

With:

```python
from khmer_pipeline.models import IngestResult
from khmer_pipeline.preprocess import preprocess, PreprocessConfig
```

And in the pipeline block, replace the inline `from khmer_pipeline.models import IngestResult as _IngestResult` line and `_IngestResult(` with just `IngestResult(` — the import is now at the top.

- [ ] **Step 2: Run tests and verify app loads**

```bash
uv run pytest -q --tb=short
```

Expected: `64 passed`

```bash
uv run streamlit run app.py
```

Upload ARDB sample PDF. Verify:
- Pipeline runs once
- Clicking download buttons does NOT reprint the status bar
- Re-uploading the same file does NOT rerun (status bar not shown again)
- Re-uploading a different file DOES rerun

- [ ] **Step 3: Commit**

```bash
git add app.py src/khmer_pipeline/postprocess.py
git commit -m "feat: add session state caching — pipeline runs once per settings key"
```

---

## Task 3: Sidebar controls

**Files:**
- Modify: `app.py`

Add `with st.sidebar:` block above the `uploaded = st.file_uploader(...)` line. Replace the hardcoded defaults from Task 2 with sidebar widget values. Update the `settings_key` computation to use the actual sidebar values.

- [ ] **Step 1: Add the sidebar block**

Insert the following block between the `st.caption(...)` line and the `uploaded = st.file_uploader(...)` line:

```python
with st.sidebar:
    st.header("Document settings")
    dpi = st.select_slider("Scan quality (DPI)", options=[150, 200, 300], value=200)
    st.caption("200 for digital PDFs, 300 for scanned documents.")

    page_selection = st.radio("Pages to process", ["All pages", "Single page", "Page range"])
    if page_selection == "Single page":
        page_num = st.number_input("Page number", min_value=1, value=1, step=1)
    elif page_selection == "Page range":
        page_start = st.number_input("From page", min_value=1, value=1, step=1)
        page_end = st.number_input("To page", min_value=1, value=5, step=1)

    st.header("Preprocessing")
    remove_stamps = st.checkbox("Remove colored stamps", value=True)
    sharpen = st.checkbox("Sharpen text", value=True)
    normalise = st.checkbox("Enhance contrast", value=True)

    st.header("Extraction")
    extraction_mode = st.radio(
        "Extraction mode",
        ["Full extraction (text + tables)", "Tables only"],
    )
    tables_only = extraction_mode == "Tables only"

    st.header("Post-processing")
    enable_qwen = st.checkbox("Enable Qwen correction", value=True)
```

- [ ] **Step 2: Remove the hardcoded defaults block**

In the `if uploaded is not None:` block, remove these lines that were added as temporary defaults in Task 2:

```python
    # Settings key — hardcoded defaults until Task 3 adds the sidebar
    dpi = 200
    page_selection = "All pages"
    page_num = 1
    page_start = 1
    page_end = 1
    remove_stamps = True
    sharpen = True
    normalise = True
    tables_only = False
    enable_qwen = True
```

- [ ] **Step 3: Update the `settings_key` computation**

Replace the two-line `page_sel_part` / `settings_key` block with:

```python
    if page_selection == "Single page":
        page_sel_part = f"page_{page_num}"
    elif page_selection == "Page range":
        page_sel_part = f"range_{page_start}_{page_end}"
    else:
        page_sel_part = "all"
    settings_key = f"{uploaded.name}_{dpi}_{page_sel_part}_{remove_stamps}_{sharpen}_{normalise}_{enable_qwen}"
```

- [ ] **Step 4: Update page selection logic**

Replace the `selected_indices = list(range(ingest_result.page_count))` line with the full selector:

```python
            total_pages = ingest_result.page_count
            if page_selection == "Single page":
                idx = min(int(page_num) - 1, total_pages - 1)
                selected_indices = [max(0, idx)]
            elif page_selection == "Page range":
                start = max(0, int(page_start) - 1)
                end = min(int(page_end), total_pages)
                selected_indices = list(range(start, max(start + 1, end)))
            else:
                selected_indices = list(range(total_pages))
```

- [ ] **Step 5: Run tests and verify sidebar**

```bash
uv run pytest -q --tb=short
```

Expected: `64 passed`

```bash
uv run streamlit run app.py
```

Verify manually:
- Sidebar appears with all controls
- Changing DPI reruns the pipeline (new status bar appears)
- Changing page selection to "Single page 2" processes only page 2
- Unchecking "Sharpen text" reruns with different preprocessing
- Unchecking "Enable Qwen correction" reruns (qwen_used badge shows ✓ rule-based only)
- "Tables only" mode: OCR text expanders are hidden; table grids and download buttons still show

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add sidebar controls — DPI, page selection, preprocessing toggles, extraction mode, Qwen toggle"
```

---

## Task 4: Layout overlay toggle + table selection checkboxes

**Files:**
- Modify: `app.py`

Two display improvements: (1) a checkbox to hide the third "Layout detection" column; (2) per-table checkboxes that control which tables appear as download buttons.

- [ ] **Step 1: Add layout overlay toggle**

Add this line immediately after `st.subheader(f"{filtered_ingest.page_count} page(s)...")` and before the `for i, ...` loop:

```python
    show_layout = st.checkbox("Show layout overlay", value=True)
```

- [ ] **Step 2: Make the three-column block conditional**

Replace the existing three-column image block:

```python
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
```

With:

```python
        if show_layout:
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
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig, caption="Original", use_container_width=True)
            with col2:
                st.image(proc, caption="Preprocessed", use_container_width=True)
```

- [ ] **Step 3: Replace CSV download loop with selection checkboxes**

Replace the CSV download loop at the bottom of `app.py`:

```python
    for table_id, csv_string in export_result.tables_csv:
        st.download_button(
            label=f"⬇ Download {table_id}.csv",
            data=csv_string.encode("utf-8-sig"),
            file_name=f"{table_id}.csv",
            mime="text/csv",
        )
```

With:

```python
    for table_id, csv_string in export_result.tables_csv:
        if st.checkbox(f"Include {table_id} in export", value=True, key=f"export_{table_id}"):
            st.download_button(
                label=f"⬇ Download {table_id}.csv",
                data=csv_string.encode("utf-8-sig"),
                file_name=f"{table_id}.csv",
                mime="text/csv",
            )
```

- [ ] **Step 4: Run tests and verify**

```bash
uv run pytest -q --tb=short
```

Expected: `64 passed`

```bash
uv run streamlit run app.py
```

Verify manually with ARDB PDF:
- Unchecking "Show layout overlay" collapses to 2 columns — no layout image
- Re-checking restores 3 columns
- Table selection checkboxes appear before each download button
- Unchecking a table checkbox hides its download button
- Unchecking does NOT retrigger pipeline (no status bar reappears)

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add layout overlay toggle and table selection checkboxes"
```

---

## Task 5: Manual text correction + edited JSON export

**Files:**
- Modify: `app.py`

Add an editable text area per page. When the analyst edits text and clicks download, the exported JSON uses their edited version instead of the pipeline output.

- [ ] **Step 1: Add edit expander per page**

After the `with st.expander(f"Post-processing — page ...")` block (and before the next iteration), add:

```python
        with st.expander(f"Edit corrected text — page {surya_page.page_index + 1}"):
            edited = st.text_area(
                "Corrected text (editable)",
                value=st.session_state.get(f"edited_text_{i}", post_page.corrected_text),
                height=200,
                key=f"edit_{i}",
            )
            if edited != post_page.corrected_text:
                st.session_state[f"edited_text_{i}"] = edited
```

Note: `i` is the loop index (0-based within the displayed pages), used as the session state key. `surya_page.page_index` is the original document page index (used only for display labels).

- [ ] **Step 2: Build patched JSON for download**

Replace the existing JSON download button:

```python
    st.download_button(
        label="⬇ Download document JSON",
        data=json.dumps(export_result.document_json, ensure_ascii=False, indent=2),
        file_name=f"{Path(uploaded.name).stem}_extracted.json",
        mime="application/json",
    )
```

With:

```python
    # Build export JSON, substituting any analyst-edited text
    doc_json = dict(export_result.document_json)
    patched_pages = []
    for i, page_data in enumerate(doc_json.get("pages", [])):
        edited_text = st.session_state.get(f"edited_text_{i}")
        if edited_text is not None:
            page_data = dict(page_data)
            page_data["corrected_text"] = edited_text
        patched_pages.append(page_data)
    doc_json["pages"] = patched_pages

    st.download_button(
        label="⬇ Download document JSON",
        data=json.dumps(doc_json, ensure_ascii=False, indent=2),
        file_name=f"{Path(uploaded.name).stem}_extracted.json",
        mime="application/json",
    )
```

- [ ] **Step 3: Run tests and verify**

```bash
uv run pytest -q --tb=short
```

Expected: `64 passed`

```bash
uv run streamlit run app.py
```

Verify manually with ARDB PDF:
- "Edit corrected text — page 1" expander opens and shows the pipeline corrected text
- Editing the text area and downloading JSON shows the edited text in `pages[0].corrected_text`
- Leaving the text area unchanged produces the original pipeline text in the JSON
- Editing text does NOT retrigger the pipeline

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add manual text correction and edited JSON export"
```

---

## Task 6: Per-stage error handling

**Files:**
- Modify: `app.py`

Wrap each of the 5 pipeline stages in `try/except`. On failure, show the error message and a Retry button that clears the session state cache so the analyst can change settings and rerun.

- [ ] **Step 1: Wrap Stage 1 (ingest)**

The ingest call already has a `try/except ValueError`. Extend it to a general `Exception` and add the Retry button:

```python
            try:
                ingest_result = ingest(uploaded.read(), uploaded.name, dpi=dpi)
            except Exception as e:
                status.update(label="Stage 1 failed", state="error")
                st.error(f"Stage 1 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
```

- [ ] **Step 2: Wrap Stage 2 (preprocess)**

```python
            try:
                config = PreprocessConfig(remove_stamps=remove_stamps, sharpen=sharpen, normalise=normalise)
                preprocess_result = preprocess(filtered_ingest, config)
            except Exception as e:
                status.update(label="Stage 2 failed", state="error")
                st.error(f"Stage 2 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
```

- [ ] **Step 3: Wrap Stage 3 (Surya OCR)**

```python
            try:
                if not models_loaded():
                    st.write("Loading Surya models — first run takes about a minute...")
                preload_models()

                def _on_page(idx: int, total: int) -> None:
                    st.write(f"Page {idx + 1} / {total}: running OCR...")

                surya_result = run_surya(preprocess_result, on_page=_on_page)
            except Exception as e:
                status.update(label="Stage 3 failed", state="error")
                st.error(f"Stage 3 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
```

- [ ] **Step 4: Wrap Stage 4 (postprocess)**

```python
            try:
                st.write("Running post-processing...")
                postprocess_result = postprocess(surya_result, skip_qwen=not enable_qwen)
            except Exception as e:
                status.update(label="Stage 4 failed", state="error")
                st.error(f"Stage 4 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
```

- [ ] **Step 5: Wrap Stage 5 (export)**

```python
            try:
                st.write("Exporting structured output...")
                export_result = export(postprocess_result)
            except Exception as e:
                status.update(label="Stage 5 failed", state="error")
                st.error(f"Stage 5 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
```

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -q --tb=short
```

Expected: `64 passed` (no change — no new tests for UI stage)

- [ ] **Step 7: Final manual verification**

```bash
uv run streamlit run app.py
```

Upload ARDB sample PDF. Run through the full checklist:

1. Pipeline runs once. Clicking download buttons does not show the status bar again.
2. Changing DPI in sidebar reruns the pipeline.
3. "Single page" → page 2: only page 2 processed and displayed.
4. "Page range" → 1–3: three pages processed and displayed.
5. Unchecking all preprocessing options runs without stamps removal/sharpening/contrast.
6. "Tables only" mode: OCR text expanders are hidden. Table grids and download buttons still show.
7. Unchecking "Enable Qwen correction": post-processing badge shows ✓ rule-based only.
8. "Show layout overlay" toggle: unchecking collapses to 2 columns; rechecking restores 3.
9. Table selection checkbox: unchecking a table hides its download button without rerunning pipeline.
10. Manual text edit: edit page 1 text, download JSON, confirm `pages[0].corrected_text` matches edit.
11. Error handling: manually break a stage (e.g., temporary bad import), confirm "Stage N failed" message and Retry button appear.

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat: add per-stage error handling with Retry button"
```

---

## Final state summary

After all 6 tasks, the commit log should have:

```
feat: add per-stage error handling with Retry button
feat: add manual text correction and edited JSON export
feat: add layout overlay toggle and table selection checkboxes
feat: add sidebar controls — DPI, page selection, preprocessing toggles, extraction mode, Qwen toggle
feat: add session state caching — pipeline runs once per settings key
feat: add skip_qwen parameter to postprocess()
```

Test count is still 64 (no new automated tests for UI stage — verified manually).
