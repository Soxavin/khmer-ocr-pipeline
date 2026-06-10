from __future__ import annotations
import json
import bleach
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
from khmer_pipeline.ingest import ingest
from khmer_pipeline.models import IngestResult
from khmer_pipeline.preprocess import preprocess, PreprocessConfig
from khmer_pipeline.surya import run_surya, models_loaded, preload_models
from khmer_pipeline.postprocess import postprocess
from khmer_pipeline.export import export

_SAFE_TAGS = [
    "p", "br", "b", "i", "em", "strong", "span",
    "table", "thead", "tbody", "tr", "td", "th",
    "math",
]


def _safe_html(html: str) -> str:
    return bleach.clean(html, tags=_SAFE_TAGS, attributes={}, strip=True)

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
st.caption("Stage 1 — Ingest  |  Stage 2 — Preprocess  |  Stage 3 — Surya OCR  |  Stage 4 — Post-process  |  Stage 5 — Export")

uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

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
                uploaded.seek(0)
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
