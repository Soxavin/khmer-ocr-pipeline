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
