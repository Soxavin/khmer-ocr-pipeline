from __future__ import annotations
import streamlit as st
from khmer_pipeline.ingest import ingest

st.set_page_config(page_title="Khmer Document Extraction", layout="wide")
st.title("Khmer Document Extraction Pipeline")
st.caption("Stage 1 — PDF / image ingestion")

uploaded = st.file_uploader(
    "Upload a PDF or image file",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

if uploaded is not None:
    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Stage 1: Converting pages to images...")
        try:
            result = ingest(uploaded.read(), uploaded.name)
            status.update(
                label=f"Stage 1 complete — {result.page_count} page(s) extracted",
                state="complete",
            )
        except ValueError as e:
            status.update(label="Stage 1 failed", state="error")
            st.error(str(e))
            st.stop()

    st.subheader(f"Extracted {result.page_count} page(s) from `{result.source_name}`")

    cols_per_row = 3
    cols = st.columns(cols_per_row)
    for i, img_arr in enumerate(result.page_images):
        with cols[i % cols_per_row]:
            st.image(img_arr, caption=f"Page {i + 1}", use_container_width=True)
