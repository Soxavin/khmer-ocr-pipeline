from __future__ import annotations
import datetime
import io
import json
import time
import zipfile
import bleach
import fitz
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
from khmer_pipeline.ingest import ingest
from khmer_pipeline.models import IngestResult
from khmer_pipeline.preprocess import preprocess, PreprocessConfig
from khmer_pipeline.surya import preload_models
from khmer_pipeline.postprocess import qwen_loaded
from khmer_pipeline.engine_registry import ACTIVE_OCR_ENGINE, ACTIVE_CORRECTION_ENGINE
from khmer_pipeline.export import export, grid_to_csv, tables_to_xlsx
from khmer_pipeline.model_config import CONFIDENCE_LOW, CONFIDENCE_MID, ANOMALY_THRESHOLD
from khmer_pipeline.memory import clear_device_cache  # NEW: Memory management import
from khmer_pipeline.backend_status import llama_server_running

# Effective-page threshold (pages x DPI/200) above which a soft "large job" warning
# shows. Stress test (10 pages @ 300 DPI, effective 15) showed NO memory distress
# (~7 min, ~2 GB peak RSS, +384 MB swap) — memory is per-page bounded. This is a
# heads-up for very large jobs, not a measured ceiling. See docs/OPERATIONS.md.
_MEMORY_WARN_PAGES = 15


@st.cache_resource(show_spinner="Loading Surya OCR models — first run takes ~30s...")
def _preload_surya() -> bool:
    preload_models()
    return True


_SAFE_TAGS = [
    "p", "br", "b", "i", "em", "strong", "span",
    "table", "thead", "tbody", "tr", "td", "th",
    "math",
]


def _safe_html(html: str) -> str:
    return bleach.clean(html, tags=_SAFE_TAGS, attributes={}, strip=True)


def _clear_edit_state() -> None:
    for key in list(st.session_state.keys()):
        if (
            key.startswith("edited_text_")
            or key.startswith("edit_")
            or key.startswith("edited_table_")
            or key.startswith("editor_")
        ):
            del st.session_state[key]


def _reset_table(tid: str) -> None:
    st.session_state.pop(f"edited_table_{tid}", None)
    st.session_state.pop(f"editor_{tid}", None)


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


def _draw_layout_confidence(img_array: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """Draw bounding boxes coloured by OCR confidence score.
    Green = high (>=CONFIDENCE_MID), yellow = medium (>=CONFIDENCE_LOW), red = low."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    for block in blocks:
        conf = block.get("confidence") or 0.0
        if conf >= CONFIDENCE_MID:
            color = "#27AE60"   # green
        elif conf >= CONFIDENCE_LOW:
            color = "#F39C12"   # yellow
        else:
            color = "#E74C3C"   # red
        bbox = block.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    return np.array(img)


st.set_page_config(page_title="Khmer Document Extraction", layout="wide")
_preload_surya()
st.title("Khmer Document Extraction Pipeline")
st.caption("Upload a financial document → review the extracted tables → download Excel/CSV.")

with st.sidebar:
    uploaded = st.file_uploader(
        "Upload a PDF or image file",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
    )

    st.divider()
    stitch_pages = st.checkbox(
        "Stitch multi-page tables",
        value=True,
        help="Join a table that continues across pages into one table, so a multi-page "
             "report exports as one CSV per table instead of one per page. A column-count "
             "change starts a new table. On by default — turn off to keep per-page tables.",
    )
    convert_numerals = st.checkbox(
        "Convert Khmer numerals",
        value=False,
        help="Converts ០១២... to 012... in exported CSV/Excel files. "
             "Useful for loading into Excel or databases. "
             "Does not affect the JSON export.",
    )

    with st.expander("⚙️ Advanced Engine Settings", expanded=False):
        st.header("Document settings")
        dpi = st.select_slider("Scan quality (DPI)", options=[150, 200, 300], value=200)
        st.caption("200 for digital PDFs, 300 for scanned documents.")

        page_selection = st.radio("Pages to process", ["All pages", "Single page", "Page range"])
        if page_selection == "Single page":
            page_num = st.number_input("Page number", min_value=1, value=1, step=1)
        elif page_selection == "Page range":
            page_start = st.number_input("From page", min_value=1, value=1, step=1)
            page_end = st.number_input("To page", min_value=1, value=5, step=1)
            if page_end < page_start:
                st.warning("'To page' is less than 'From page' — only the first page will be processed.")

        st.header("Preprocessing")
        remove_stamps = st.checkbox("Remove colored stamps", value=True)
        sharpen = st.checkbox("Sharpen text", value=True)
        normalise = st.checkbox("Enhance contrast", value=True)
        deskew = st.checkbox("Deskew (straighten rotated scans)", value=True)
        normalise_table_backgrounds = st.checkbox(
            "Normalise colored table backgrounds",
            value=True,
            help="Flattens shaded or colored table cell backgrounds (e.g. header "
                 "row fills) toward white before OCR. Improves table-grid detection "
                 "on documents with colored cell shading.",
        )

        st.header("Extraction")
        extraction_mode = st.radio(
            "Extraction mode",
            ["Full extraction (text + tables)", "Tables only"],
        )
        tables_only = extraction_mode == "Tables only"

        st.header("Post-processing")
        enable_qwen = st.checkbox(
            "Enable Qwen correction (experimental, slow)",
            value=False,
            help="One-time ~4GB model download, then a slow per-run load on a 24GB "
                 "Mac. Off by default — the deterministic Khmer normalizer always "
                 "runs and is usually sufficient.",
        )
        anomaly_threshold = st.slider(
            "Anomaly threshold for Qwen correction",
            min_value=0.0,
            max_value=1.0,
            value=ANOMALY_THRESHOLD,
            step=0.01,
            help="Proportion of non-Khmer/non-Latin script characters in a text "
                 "block that triggers Qwen correction. Lower = more aggressive "
                 "(more blocks sent to Qwen). Only applies when Qwen correction is enabled.",
        )

        st.header("Export")
        repair_tables = st.checkbox(
            "Auto-repair inconsistent table grids",
            value=False,
            help="If a detected table has rows with different numbers of cells, "
                 "pad short rows with empty cells so the CSV/JSON grid is rectangular. "
                 "Repaired tables are flagged with 'was_repaired' in the JSON and a "
                 "warning in the UI. Off by default — review the raw grid first.",
        )

        st.header("Layout overlay")
        show_layout = st.checkbox("Show layout overlay", value=True)
        overlay_mode = st.radio(
            "Layout overlay mode",
            ["Region type", "Confidence"],
            horizontal=True,
        ) if show_layout else None

    st.divider()
    if llama_server_running():
        st.caption("🟢 OCR backend ready")
    else:
        st.caption("⚪ OCR backend idle — starts automatically on the first extraction")

if uploaded is None:
    st.markdown("### Upload a document to get started")
    st.markdown(
        "Supported formats: PDF, PNG, JPG, TIFF  \n"
        "Upload a file using the uploader in the sidebar, configure your options there, "
        "then click **Run Extraction**."
    )
    st.button("▶ Run Extraction", type="primary", disabled=True)
else:
    if page_selection == "Single page":
        page_sel_part = f"page_{page_num}"
    elif page_selection == "Page range":
        page_sel_part = f"range_{page_start}_{page_end}"
    else:
        page_sel_part = "all"
    
    settings_key = f"{uploaded.name}_{dpi}_{page_sel_part}_{remove_stamps}_{sharpen}_{normalise}_{enable_qwen}_{convert_numerals}_{repair_tables}_{stitch_pages}_{anomaly_threshold}_{deskew}_{normalise_table_backgrounds}"

    # Reset run state when a different file is uploaded
    if uploaded.name != st.session_state.get("last_uploaded_name"):
        st.session_state["run_triggered"] = False
        st.session_state["last_uploaded_name"] = uploaded.name
        st.session_state.pop("last_key", None)
        st.session_state.pop("current_page_idx", None)  # NEW: Reset pagination index
        st.session_state.pop("stage_times", None)
        _clear_edit_state()

    file_size_kb = round(len(uploaded.getvalue()) / 1024, 1)
    st.markdown(f"**File:** {uploaded.name}  \n**Size:** {file_size_kb} KB")
    doc_page_count = 1
    if Path(uploaded.name).suffix.lower() == ".pdf":
        try:
            with fitz.open(stream=uploaded.getvalue(), filetype="pdf") as doc:
                doc_page_count = len(doc)
                st.markdown(f"**Pages:** {doc_page_count}")
        except Exception:
            doc_page_count = 0
            st.markdown("**Pages:** (could not read page count)")
    else:
        st.markdown("**Pages:** 1 (image file)")

    # Memory soft guard — estimate pages to be processed, scaled by DPI vs the 200 baseline.
    if page_selection == "Single page":
        _est_pages = 1
    elif page_selection == "Page range":
        _est_pages = max(1, int(page_end) - int(page_start) + 1)
    else:
        _est_pages = doc_page_count
    if _est_pages * (dpi / 200.0) > _MEMORY_WARN_PAGES:
        st.warning(
            "Large document detected — processing may take several minutes. "
            "If needed, process in smaller page ranges (pages run sequentially)."
        )

    if page_selection == "Single page":
        page_info = f"Single page — page {page_num}"
    elif page_selection == "Page range":
        page_info = f"Page range — {page_start} to {page_end}"
    else:
        page_info = "All pages"

    preprocessing_steps = []
    if deskew:
        preprocessing_steps.append("Deskew")
    if remove_stamps:
        preprocessing_steps.append("Stamp removal")
    if sharpen:
        preprocessing_steps.append("Sharpen")
    if normalise:
        preprocessing_steps.append("Contrast enhancement")
    if normalise_table_backgrounds:
        preprocessing_steps.append("Background normalisation")
    preprocessing_info = ", ".join(preprocessing_steps) if preprocessing_steps else "None"

    with st.expander("Current settings", expanded=False):
        st.markdown(
            f"- **Scan quality:** {dpi} DPI\n"
            f"- **Pages:** {page_info}\n"
            f"- **Preprocessing:** {preprocessing_info}\n"
            f"- **Extraction mode:** {extraction_mode}\n"
            f"- **Qwen correction:** {'On' if enable_qwen else 'Off'}\n"
            f"- **Anomaly threshold:** {anomaly_threshold:.2f}\n"
            f"- **Numeral conversion:** {'On' if convert_numerals else 'Off'}\n"
            f"- **Table auto-repair:** {'On' if repair_tables else 'Off'}\n"
            f"- **Stitch tables across pages:** {'On' if stitch_pages else 'Off'}"
        )

    run_triggered = st.session_state.get("run_triggered", False)
    last_key = st.session_state.get("last_key")
    if run_triggered and last_key is not None and last_key != settings_key:
        st.info("Settings changed. Click **Run Extraction** to reprocess.")

    run_clicked = st.button("▶ Run Extraction", type="primary")

    if run_clicked:
        st.session_state["run_triggered"] = True
        _clear_edit_state()

    if run_clicked and last_key != settings_key:
        stage_times: dict[str, float] = {}
        with st.status("Running pipeline...", expanded=True) as status:
            st.write("Reading the document…")
            _t0 = time.perf_counter()
            try:
                uploaded.seek(0)
                ingest_result = ingest(uploaded.read(), uploaded.name, dpi=dpi)
            except Exception as e:
                status.update(label="Stage 1 failed", state="error")
                st.error(f"Stage 1 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
            stage_times["Stage 1 — Ingest"] = time.perf_counter() - _t0

            total_pages = ingest_result.page_count
            if page_selection == "Single page":
                if total_pages == 0:
                    st.warning("Document has no pages.")
                    st.stop()
                idx = max(0, min(int(page_num) - 1, total_pages - 1))
                selected_indices = [idx]
            elif page_selection == "Page range":
                if total_pages == 0:
                    st.warning("Document has no pages.")
                    st.stop()
                start = max(0, int(page_start) - 1)
                if start >= total_pages:
                    st.warning(
                        f"Start page {page_start} exceeds document length ({total_pages} page(s))."
                    )
                    st.stop()
                end = min(int(page_end), total_pages)
                selected_indices = list(range(start, max(start + 1, end)))
            else:
                selected_indices = list(range(total_pages))
            filtered_ingest = IngestResult(
                source_name=ingest_result.source_name,
                page_images=[ingest_result.page_images[i] for i in selected_indices],
                dpi=ingest_result.dpi,
                page_count=len(selected_indices),
            )
            st.session_state["ingest_result"] = ingest_result
            st.session_state["filtered_ingest"] = filtered_ingest
            clear_device_cache()  # NEW: Free memory after ingest

            st.write("Cleaning the pages…")
            _t0 = time.perf_counter()
            try:
                config = PreprocessConfig(remove_stamps=remove_stamps, sharpen=sharpen, normalise=normalise, deskew=deskew, normalise_table_backgrounds=normalise_table_backgrounds)
                preprocess_result = preprocess(filtered_ingest, config)
                st.session_state["preprocess_result"] = preprocess_result
                clear_device_cache()  # NEW: Free memory after preprocessing
            except Exception as e:
                status.update(label="Stage 2 failed", state="error")
                st.error(f"Stage 2 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
            stage_times["Stage 2 — Preprocess"] = time.perf_counter() - _t0

            st.write("Finding text & tables…")
            _t0 = time.perf_counter()
            try:
                ocr_progress = st.progress(0, text="Starting OCR...")

                def _on_page(idx: int, total: int) -> None:
                    ocr_progress.progress((idx + 1) / total, text=f"OCR: Page {idx + 1} of {total}")

                surya_result = ACTIVE_OCR_ENGINE(preprocess_result, on_page=_on_page)
                ocr_progress.progress(1.0, text="OCR Complete!")
                if surya_result.warnings:
                    st.warning(
                        f"Stage 3: {len(surya_result.warnings)} issue(s) — "
                        + "; ".join(surya_result.warnings[:3])
                    )
                st.session_state["surya_result"] = surya_result
                clear_device_cache()  # NEW: Free PyTorch MPS memory after Surya
            except Exception as e:
                status.update(label="Stage 3 failed", state="error")
                st.error(f"Stage 3 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
            stage_times["Stage 3 — OCR"] = time.perf_counter() - _t0

            st.write("Tidying the text…")
            if enable_qwen and not qwen_loaded():
                st.write("Loading Qwen model — first run downloads ~4GB, may take several minutes...")
            _t0 = time.perf_counter()
            try:
                postprocess_result = ACTIVE_CORRECTION_ENGINE(
                    surya_result,
                    skip_qwen=not enable_qwen,
                    anomaly_threshold=anomaly_threshold,
                )
                st.session_state["postprocess_result"] = postprocess_result
                clear_device_cache()  # NEW: Free MLX memory after Qwen fallback
            except Exception as e:
                status.update(label="Stage 4 failed", state="error")
                st.error(f"Stage 4 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
            stage_times["Stage 4 — Post-process"] = time.perf_counter() - _t0

            st.write("Preparing your files…")
            _t0 = time.perf_counter()
            try:
                export_result = export(postprocess_result, convert_numerals=convert_numerals, repair_tables=repair_tables, stitch_pages=stitch_pages)
                st.session_state["export_result"] = export_result
                clear_device_cache()  # NEW: Final memory cleanup
            except Exception as e:
                status.update(label="Stage 5 failed", state="error")
                st.error(f"Stage 5 failed: {str(e)}")
                st.button("Retry", on_click=lambda: st.session_state.clear())
                st.stop()
            stage_times["Stage 5 — Export"] = time.perf_counter() - _t0

            st.session_state["stage_times"] = stage_times
            st.session_state["last_key"] = settings_key

            status.update(
                label=f"Stages 1–5 complete — {filtered_ingest.page_count} page(s) from {ingest_result.source_name}",
                state="complete",
            )
            
    if not (st.session_state.get("run_triggered") and "export_result" in st.session_state):
        st.stop()

    ingest_result = st.session_state["ingest_result"]
    filtered_ingest = st.session_state["filtered_ingest"]
    preprocess_result = st.session_state["preprocess_result"]
    surya_result = st.session_state["surya_result"]
    postprocess_result = st.session_state["postprocess_result"]
    export_result = st.session_state["export_result"]

    st.subheader(f"{filtered_ingest.page_count} page(s) from `{ingest_result.source_name}`")

    # Results overview — document-level summary for at-a-glance review / demo
    total_tables = sum(len(p.tables) for p in surya_result.pages)
    n_warnings = len(surya_result.warnings)
    ov1, ov2, ov3 = st.columns(3)
    ov1.metric("Pages", filtered_ingest.page_count)
    ov2.metric("Tables detected", total_tables)
    ov3.metric("Warnings", n_warnings)
    if total_tables == 0:
        st.error(
            "No tables detected. Try Advanced → enable Preprocessing, raise Scan "
            "quality (DPI), or confirm the document contains tables."
        )
    elif n_warnings:
        st.warning(f"Extraction complete with {n_warnings} warning(s) — review the panel below.")
    else:
        st.success("Extraction complete — review the pages below.")

    # Persistent warnings panel (the in-run st.warning vanishes after pagination reruns)
    if surya_result.warnings:
        with st.expander(f"⚠️ Pipeline warnings ({n_warnings})", expanded=False):
            for w in surya_result.warnings:
                st.markdown(f"- {w}")

    # ==========================================
    # PAGINATED UI STARTS HERE
    # ==========================================
    
    pages_data = list(zip(
        filtered_ingest.page_images, 
        preprocess_result.page_images, 
        surya_result.pages, 
        postprocess_result.pages
    ))
    total_pages = len(pages_data)

    if "current_page_idx" not in st.session_state:
        st.session_state.current_page_idx = 0
        
    st.session_state.current_page_idx = max(0, min(st.session_state.current_page_idx, total_pages - 1))
    current_idx = st.session_state.current_page_idx

    st.markdown(f"### 📄 Reviewing Page {current_idx + 1} of {total_pages}")
    col_select, col_nav = st.columns([2, 3])
    
    with col_select:
        page_options = [f"Page {i+1}" for i in range(total_pages)]
        selected_option = st.selectbox(
            "Jump to Page",
            options=page_options,
            index=current_idx,
            label_visibility="collapsed"
        )
        new_idx = page_options.index(selected_option)
        if new_idx != current_idx:
            st.session_state.current_page_idx = new_idx
            st.rerun()

    with col_nav:
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous", disabled=(current_idx == 0), width="stretch"):
                st.session_state.current_page_idx -= 1
                st.rerun()
        with col_next:
            if st.button("Next ➡️", disabled=(current_idx == total_pages - 1), width="stretch"):
                st.session_state.current_page_idx += 1
                st.rerun()

    st.divider()

    orig, proc, surya_page, post_page = pages_data[current_idx]
    i = current_idx  # Match original session state keys

    # Per-page quality banner — at-a-glance trust signal before the detail tabs
    low_conf = sum(1 for b in surya_page.text_blocks if (b.get("confidence") or 0.0) < CONFIDENCE_LOW)
    n_page_tables = len(surya_page.tables)
    qcols = st.columns(4)
    qcols[0].metric("Text blocks", len(surya_page.text_blocks))
    qcols[1].metric("Tables", n_page_tables)
    qcols[2].metric("Low-confidence", low_conf)
    qcols[3].metric("Qwen", "Yes" if post_page.qwen_used else "No")
    if n_page_tables > 1:
        st.caption("⚠ Multiple table regions detected — a single table may have been fragmented; verify the grids below.")
    if low_conf:
        st.caption("⚠ Some blocks have low OCR confidence — switch the overlay to 'Confidence' (Advanced settings) to locate them.")

    # Build (table_id, grid) pairs for the FINAL export tables (document-level,
    # built once — not per-page). Raw cell text, not numeral-converted: when
    # stitch_pages is on, document_json["document_tables"] holds the stitched
    # tables; otherwise each page's "tables" block holds the per-page tables.
    _export_table_blocks = export_result.document_json.get("document_tables")
    if _export_table_blocks is None:
        _export_table_blocks = [
            table
            for page_data in export_result.document_json.get("pages", [])
            for table in page_data.get("tables", [])
        ]
    _export_tables = []
    for _block in _export_table_blocks:
        _cells = _block.get("cells", [])
        _max_row = max((c.get("row", 0) for c in _cells), default=0) + (1 if _cells else 0)
        _max_col = max((c.get("col", 0) for c in _cells), default=0) + (1 if _cells else 0)
        _grid = [["" for _ in range(_max_col)] for _ in range(_max_row)]
        for c in _cells:
            r, col = c.get("row", 0), c.get("col", 0)
            if 0 <= r < _max_row and 0 <= col < _max_col:
                _grid[r][col] = c.get("text", "")
        _export_tables.append((_block["table_id"], _grid))

    # ==========================================
    # SIDE-BY-SIDE REVIEW: page image (left) + editable tables (right)
    # ==========================================
    left, right = st.columns([0.4, 0.6])

    with left:
        if show_layout:
            if overlay_mode == "Confidence":
                overlay_img = _draw_layout_confidence(proc, surya_page.text_blocks)
                overlay_caption = "Confidence (🟢 high 🟡 medium 🔴 low)"
            else:
                table_blocks = [{"label": "Table", "bbox": t["bbox"]} for t in surya_page.tables]
                overlay_img = _draw_layout(proc, surya_page.text_blocks + table_blocks)
                overlay_caption = "Layout detection"
            st.image(overlay_img, caption=overlay_caption, width="stretch")
        else:
            st.image(proc, caption="Preprocessed", width="stretch")
        with st.expander("Original image"):
            st.image(orig, caption="Original", width="stretch")

    with right:
        if _export_tables:
            st.subheader("✏️ Review & edit tables")
            for table_id, grid in _export_tables:
                if not grid:
                    continue
                st.markdown(f"**{table_id}**")
                # All rows stay editable (incl. the real header row); columns get
                # neutral "Col N" labels — we don't fold the first row into headers.
                n_cols = max((len(r) for r in grid), default=0)
                col_labels = [f"Col {c + 1}" for c in range(n_cols)]
                edited_df = st.data_editor(
                    pd.DataFrame(grid, columns=col_labels),
                    num_rows="dynamic",
                    width="stretch",
                    key=f"editor_{table_id}",
                )
                edited_grid = edited_df.fillna("").astype(str).values.tolist()
                if edited_grid != grid:
                    st.session_state[f"edited_table_{table_id}"] = edited_grid
                else:
                    st.session_state.pop(f"edited_table_{table_id}", None)
                st.button(
                    "↺ Reset table to original",
                    key=f"reset_{table_id}",
                    on_click=_reset_table,
                    args=(table_id,),
                )
        else:
            st.caption("No tables detected on this page.")

    final_tables = [
        (table_id, st.session_state.get(f"edited_table_{table_id}", original_grid))
        for table_id, original_grid in _export_tables
    ]

    # ==========================================
    # TEXT & PROCESSING DETAILS (secondary)
    # ==========================================
    with st.expander("Text & processing details", expanded=False):
        if "stage_times" in st.session_state:
            cols = st.columns(len(st.session_state["stage_times"]))
            for col, (name, secs) in zip(cols, st.session_state["stage_times"].items()):
                col.metric(name, f"{secs:.1f}s")

        st.markdown("**OCR text**")
        if tables_only:
            st.caption("Text output is hidden in 'Tables only' extraction mode.")
        elif surya_page.ocr_text:
            st.markdown(_safe_html(surya_page.ocr_text), unsafe_allow_html=True)
        else:
            st.caption("No OCR text on this page.")

        st.markdown("**Corrected text**")
        if post_page.qwen_used:
            st.markdown("**⚡ Qwen correction applied**")
        else:
            st.markdown("**✓ rule-based only**")
        if post_page.corrected_text:
            st.write(post_page.corrected_text)
        if post_page.correction_diff:
            st.code(post_page.correction_diff, language="diff")

        st.markdown("**Edit corrected text**")
        edited = st.text_area(
            "Corrected text (editable)",
            value=st.session_state.get(f"edited_text_{i}", post_page.corrected_text),
            height=200,
            key=f"edit_{i}",
        )
        if edited != post_page.corrected_text:
            st.session_state[f"edited_text_{i}"] = edited
        else:
            st.session_state.pop(f"edited_text_{i}", None)

    # ==========================================
    # DOWNLOADS SECTION
    # ==========================================
    st.divider()
    st.subheader("Downloads")

    doc_json = dict(export_result.document_json)
    patched_pages = []
    for idx, page_data in enumerate(doc_json.get("pages", [])):
        edited_text = st.session_state.get(f"edited_text_{idx}")
        if edited_text is not None:
            page_data = dict(page_data)
            page_data["corrected_text"] = edited_text
        patched_pages.append(page_data)
    doc_json["pages"] = patched_pages

    # Patch the exported table block's cells from the edited grids so the JSON
    # matches the CSVs — document_tables when stitch is on, else pages[].tables.
    _final_grids_by_id = {table_id: grid for table_id, grid in final_tables}

    def _patch_table_block(block):
        grid = _final_grids_by_id.get(block["table_id"])
        if grid is None:
            return block
        patched = dict(block)
        patched["cells"] = [
            {"row": r, "col": c, "text": grid[r][c]}
            for r in range(len(grid))
            for c in range(len(grid[r]))
        ]
        return patched

    if doc_json.get("document_tables") is not None:
        doc_json["document_tables"] = [_patch_table_block(b) for b in doc_json["document_tables"]]
    else:
        doc_json["pages"] = [
            {**page_data, "tables": [_patch_table_block(t) for t in page_data.get("tables", [])]}
            for page_data in doc_json["pages"]
        ]

    st.download_button(
        label="⬇ Download document JSON",
        data=json.dumps(doc_json, ensure_ascii=False, indent=2),
        file_name=f"{Path(uploaded.name).stem}_extracted.json",
        mime="application/json",
        width="stretch"
    )

    _stage_times = st.session_state.get("stage_times", {})
    _timing_lines = "\n".join(
        f"  {name:<28}: {secs:.1f}s" for name, secs in _stage_times.items()
    )
    _divider = "=" * 72
    _txt_header = (
        f"{_divider}\n"
        f"KHMER DOCUMENT EXTRACTION REPORT\n"
        f"{_divider}\n"
        f"Source        : {uploaded.name}\n"
        f"Extracted     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Pages         : {filtered_ingest.page_count}\n"
        f"DPI           : {dpi}\n"
        f"Preprocessing : {preprocessing_info}\n"
        f"Mode          : {extraction_mode}\n"
        f"Qwen          : {'Enabled' if enable_qwen else 'Disabled'} (threshold: {anomaly_threshold:.2f})\n"
        + (f"{'-' * 72}\n{_timing_lines}\n" if _timing_lines else "")
        + _divider
    )
    _page_sections = [
        f"--- Page {idx + 1} of {len(patched_pages)} ---\n\n{p.get('corrected_text', '')}"
        for idx, p in enumerate(patched_pages)
        if p.get("corrected_text")
    ]
    all_text = _txt_header + "\n\n" + "\n\n".join(_page_sections)
    st.download_button(
        label="⬇ Download extracted text (.txt)",
        data=all_text.encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_extracted.txt",
        mime="text/plain",
        width="stretch",
    )

    _stem = Path(uploaded.name).stem
    _nonempty_tables = [
        (table_id, grid)
        for table_id, grid in final_tables
        if any(cell.strip() for row in grid for cell in row)
    ]

    if _nonempty_tables:
        st.download_button(
            label="⬇ Download as Excel (.xlsx)",
            data=tables_to_xlsx(final_tables, convert_numerals),
            file_name=f"{_stem}_extracted.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    # One-click bundle: JSON + .txt report + Excel + every non-empty table CSV
    _zip_buf = io.BytesIO()
    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
        _zf.writestr(
            f"{_stem}_extracted.json",
            json.dumps(doc_json, ensure_ascii=False, indent=2),
        )
        _zf.writestr(f"{_stem}_extracted.txt", all_text)
        if _nonempty_tables:
            _zf.writestr(f"{_stem}_extracted.xlsx", tables_to_xlsx(final_tables, convert_numerals))
        for table_id, grid in _nonempty_tables:
            csv_string = grid_to_csv(grid, convert_numerals)
            _zf.writestr(f"{table_id}.csv", csv_string.encode("utf-8-sig"))
    st.download_button(
        label="⬇ Download everything (.zip)",
        data=_zip_buf.getvalue(),
        file_name=f"{_stem}_extracted.zip",
        mime="application/zip",
        width="stretch",
    )

    skipped_tables = len(final_tables) - len(_nonempty_tables)
    if _nonempty_tables:
        for table_id, grid in _nonempty_tables:
            if st.checkbox(f"Include {table_id} in export", value=True, key=f"export_{table_id}"):
                st.download_button(
                    label=f"⬇ Download {table_id}.csv",
                    data=grid_to_csv(grid, convert_numerals).encode("utf-8-sig"),
                    file_name=f"{table_id}.csv",
                    mime="text/csv",
                    width="stretch"
                )
    if skipped_tables:
        st.caption(
            f"{skipped_tables} table(s) had no extractable content and were excluded from downloads."
        )
