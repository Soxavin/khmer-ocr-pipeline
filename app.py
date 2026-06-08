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
            st.image(orig, caption="Original", use_container_width=True)
        with col2:
            st.image(proc, caption="Preprocessed", use_container_width=True)
