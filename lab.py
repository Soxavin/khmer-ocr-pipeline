"""Researcher lab for the Khmer OCR pipeline.

Opposite of app.py (which hides the ML): this tool lets a developer follow
the full pipeline, test inputs, and compare OCR engines side-by-side.

Two tabs:
  - Compare engines: run multiple engines on the same input, view overlays and
    extracted tables, and score against eval-doc ground truth when available.
  - Inspect stages: trace one engine through Ingest → Preprocess → Layout →
    Structure (SLANet) → Recognition for a single page.

Run with:  uv run streamlit run lab.py
"""
from __future__ import annotations

import contextlib
import glob
import json
import os
import re
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from khmer_pipeline.evaluation.evaluate_structure import evaluate_table, pred_table_grid
from khmer_pipeline.ingest import ingest
from khmer_pipeline.engines.layout_detect import detect_table_boxes
from khmer_pipeline.utils.memory import clear_device_cache
from khmer_pipeline.models import IngestResult, PreprocessResult, SuryaResult
from khmer_pipeline.preprocess import PreprocessConfig, preprocess
from khmer_pipeline.engines.slanet_structure import predict_cells
from khmer_pipeline.engines.surya import preload_models, run_surya
from khmer_pipeline.engines.hybrid_engine import run_hybrid
from khmer_pipeline.engines.surya_kiri_engine import run_surya_kiri
from khmer_pipeline.engines.table_merge_pages import merge_document_tables

# lab.py calls run_surya / run_hybrid directly rather than using
# engine_registry.ACTIVE_OCR_ENGINE because the registry exposes ONE active
# engine; a comparison tool must invoke several engines per run — this is the
# justified research-tool exception. The registry rule targets production
# orchestrators (app.py / pipeline.py).

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
_EVAL_REAL_DIR = Path("eval/datasets/real")
_INGEST_DPI = 200
_PAGE_PICKER_LABEL = "Page (1-based)"

# Label → overlay colour (mirrors app.py _LABEL_COLORS)
_LABEL_COLORS: dict[str, str] = {
    "Text": "#4A90D9",
    "Table": "#E74C3C",
    "TableOfContents": "#E67E22",
    "Picture": "#27AE60",
    "Figure": "#27AE60",
    "Caption": "#8E44AD",
}
# Distinct colour for DocLayout-YOLO boxes (so they're visually separate from
# Surya-detected regions on the same overlay)
_DOCLAYOUT_COLOR = "#F1C40F"  # yellow
_SLANET_CELL_COLOR = "#9B59B6"  # purple

# Engine names — the canonical strings used as keys throughout the UI
_ENGINE_SURYA = "Surya"
_ENGINE_SURYA_KIRI = "Surya + Kiri"
_ENGINE_HYBRID_ROW = "Hybrid (rowband)"
_ENGINE_HYBRID_DOC = "Hybrid + DocLayout-YOLO"
_ALL_ENGINES = [_ENGINE_SURYA, _ENGINE_SURYA_KIRI, _ENGINE_HYBRID_ROW, _ENGINE_HYBRID_DOC]

_METRICS_COLUMNS = ["Engine", "Pred dims", "Cell_Accuracy", "Cell_Content_Recall", "Table_CER"]

# What each engine actually IS — the model stack behind the label, so "hybrid"
# isn't a black box. Keyed by engine name; each entry lists the model used for
# each of the three OCR sub-jobs (layout → structure → recognition).
_ENGINE_INFO: dict[str, dict[str, str]] = {
    _ENGINE_SURYA: {
        "Layout": "Surya layout model — detects text/table regions",
        "Structure": "Surya — the recognition VLM emits the table's <td> HTML itself",
        "Recognition": "Surya recognition VLM (surya-ocr, llama.cpp Metal backend)",
    },
    _ENGINE_SURYA_KIRI: {
        "Layout": "Surya layout on the raw page — table regions (fragments merged via table_stitch)",
        "Structure": "Surya TableRecPredictor — per-cell polygons + row/col ids",
        "Recognition": "KiriOCR (vendored, CTC 'fast' path) reads each cell, with per-cell Otsu binarization",
    },
    _ENGINE_HYBRID_ROW: {
        "Layout": "Surya layout + geometric merge of fragmented table boxes (table_stitch)",
        "Structure": "SLANet-plus (rapid_table, ONNX) — table grid + per-cell coordinates",
        "Recognition": "Surya VLM reads each SLANet row as one full-width strip (rowband)",
    },
    _ENGINE_HYBRID_DOC: {
        "Layout": "DocLayout-YOLO (rapid_layout, ONNX) — one table box, no geometric merge",
        "Structure": "SLANet-plus (rapid_table, ONNX) — table grid + per-cell coordinates",
        "Recognition": "Surya VLM reads each SLANet row as one full-width strip (rowband)",
    },
}


def _engine_caption(engine_name: str) -> str:
    # Compact "Layout / Structure / Recognition" stack for st.caption/st.markdown.
    info = _ENGINE_INFO[engine_name]
    return "  \n".join(f"**{stage}:** {desc}" for stage, desc in info.items())


# ---------------------------------------------------------------------------
# Model preload (cached for the process lifetime, like app.py)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Surya OCR models — first run takes ~30s...")
def _preload_surya() -> bool:
    preload_models()
    return True


# ---------------------------------------------------------------------------
# Overlay helpers (local replicas — do NOT import from app.py)
# ---------------------------------------------------------------------------
def _draw_layout(img_array: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """Draw coloured bounding boxes keyed by block['label']."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    for block in blocks:
        color = _LABEL_COLORS.get(block["label"], "#95A5A6")
        x0, y0, x1, y1 = [int(v) for v in block["bbox"]]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    return np.array(img)


def _table_bbox(table: dict) -> Optional[list]:
    # Hybrid-rowband tables (built by _build_table_from_grid) carry only
    # "image_bbox"; Surya tables also set "bbox". Accept either.
    return table.get("bbox") or table.get("image_bbox")


def _draw_boxes(img_array: np.ndarray, boxes: list, color: str, width: int = 2) -> np.ndarray:
    """Draw a list of [x0, y0, x1, y1] boxes in a single colour."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x0, y0, x1, y1 = [int(v) for v in b]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return np.array(img)


def _overlay_for_engine(
    page_img: np.ndarray,
    page_result,
    engine_name: str,
) -> np.ndarray:
    """Build detection overlay for one engine's page result.

    For DocLayout engine also overlays detect_table_boxes() boxes in a
    distinct colour so the difference vs Surya layout is visible.
    """
    text_blocks = list(page_result.text_blocks)
    table_blocks = [
        {"label": "Table", "bbox": bb}
        for t in page_result.tables
        if (bb := _table_bbox(t)) is not None
    ]
    overlay = _draw_layout(page_img, text_blocks + table_blocks)
    if engine_name == _ENGINE_HYBRID_DOC:
        dl_boxes = detect_table_boxes(page_img)
        overlay = _draw_boxes(overlay, dl_boxes, _DOCLAYOUT_COLOR, width=3)
    return overlay


# ---------------------------------------------------------------------------
# Engine env-var context manager
# ---------------------------------------------------------------------------
# NOTE: this mutates process-global os.environ; two simultaneous lab sessions in
# one process would race on these vars. Fine for this single-user local tool.
@contextlib.contextmanager
def _engine_env(hybrid_mode: Optional[str], layout_detector: Optional[str]) -> Generator:
    """Snapshot, set, then restore KHMER_HYBRID_MODE and KHMER_LAYOUT_DETECTOR.

    Ensures parallel / sequential engine runs in the same process don't bleed
    env state into each other (both helpers read os.environ per call).
    """
    _saved_mode = os.environ.get("KHMER_HYBRID_MODE")
    _saved_detector = os.environ.get("KHMER_LAYOUT_DETECTOR")
    try:
        if hybrid_mode is not None:
            os.environ["KHMER_HYBRID_MODE"] = hybrid_mode
        else:
            os.environ.pop("KHMER_HYBRID_MODE", None)
        if layout_detector is not None:
            os.environ["KHMER_LAYOUT_DETECTOR"] = layout_detector
        else:
            os.environ.pop("KHMER_LAYOUT_DETECTOR", None)
        yield
    finally:
        if _saved_mode is not None:
            os.environ["KHMER_HYBRID_MODE"] = _saved_mode
        else:
            os.environ.pop("KHMER_HYBRID_MODE", None)
        if _saved_detector is not None:
            os.environ["KHMER_LAYOUT_DETECTOR"] = _saved_detector
        else:
            os.environ.pop("KHMER_LAYOUT_DETECTOR", None)


def _run_engine(
    engine_name: str,
    pre: PreprocessResult,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> SuryaResult:
    """Dispatch to the correct engine with the right env vars."""
    if engine_name == _ENGINE_SURYA:
        with _engine_env(None, None):
            return run_surya(pre, on_page=on_page)
    elif engine_name == _ENGINE_SURYA_KIRI:
        with _engine_env(None, None):
            return run_surya_kiri(pre, on_page=on_page)
    elif engine_name == _ENGINE_HYBRID_ROW:
        with _engine_env("rowband", None):
            return run_hybrid(pre, on_page=on_page)
    elif engine_name == _ENGINE_HYBRID_DOC:
        with _engine_env("rowband", "doclayout"):
            return run_hybrid(pre, on_page=on_page)
    else:
        raise ValueError(f"Unknown engine: {engine_name!r}")


# ---------------------------------------------------------------------------
# Eval-doc stem discovery
# ---------------------------------------------------------------------------
def _discover_eval_stems() -> list[str]:
    """Return unique document stems from eval/datasets/real/*_p*.png."""
    if not _EVAL_REAL_DIR.exists():
        return []
    pngs = glob.glob(str(_EVAL_REAL_DIR / "*_p*.png"))
    stems: list[str] = []
    seen: set[str] = set()
    for p in sorted(pngs):
        name = Path(p).stem  # e.g. "stem_p1"
        # Strip trailing _p<digits>
        m = re.match(r"^(.+)_p\d+$", name)
        stem = m.group(1) if m else name
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


# ---------------------------------------------------------------------------
# Input loading helper
# ---------------------------------------------------------------------------
def _load_input(
    source: str,
    upload_bytes: Optional[bytes],
    upload_name: Optional[str],
    eval_stem: Optional[str],
    single_page: bool,
    page_num: int,
    config: PreprocessConfig,
) -> tuple[list[np.ndarray], PreprocessResult]:
    """Load + preprocess the chosen input; return (original_images, PreprocessResult).

    original_images are the raw page images (pre-preprocess), used for the
    Ingest stage overlay in the Inspect tab.
    """
    if source == "Upload":
        assert upload_bytes is not None and upload_name is not None
        ingest_result = ingest(upload_bytes, upload_name, dpi=_INGEST_DPI)
    else:
        # Eval doc: load page PNGs exactly as eval_document._load_pages does
        assert eval_stem is not None
        pngs = sorted(glob.glob(str(_EVAL_REAL_DIR / f"{eval_stem}_p*.png")))
        images: list[np.ndarray] = []
        for p in pngs:
            images.extend(
                ingest(Path(p).read_bytes(), Path(p).name, dpi=_INGEST_DPI).page_images
            )
        ingest_result = IngestResult(
            source_name=eval_stem,
            page_images=images,
            dpi=_INGEST_DPI,
            page_count=len(images),
        )

    total = ingest_result.page_count
    if single_page:
        idx = max(0, min(page_num - 1, total - 1))
        selected = [ingest_result.page_images[idx]]
    else:
        selected = list(ingest_result.page_images)

    filtered = IngestResult(
        source_name=ingest_result.source_name,
        page_images=selected,
        dpi=ingest_result.dpi,
        page_count=len(selected),
    )
    pre = preprocess(filtered, config)
    return selected, pre


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _run_cache_key(
    source_id: str,
    single_page: bool,
    page_num: int,
    config: PreprocessConfig,
    engine_name: str,
) -> str:
    scope = f"p{page_num}" if single_page else "all"
    cfg = f"{config.remove_stamps}{config.sharpen}{config.normalise}{config.deskew}{config.normalise_table_backgrounds}"
    return f"{source_id}|{scope}|{cfg}|{engine_name}"


def _get_cached_run(key: str) -> Optional[SuryaResult]:
    return st.session_state.get("lab_runs", {}).get(key)


def _store_run(key: str, result: SuryaResult) -> None:
    if "lab_runs" not in st.session_state:
        st.session_state["lab_runs"] = {}
    st.session_state["lab_runs"][key] = result


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Khmer OCR Lab", layout="wide")
_preload_surya()
st.title("Khmer OCR Research Lab")
st.caption(
    "Developer tool — compare engines, trace pipeline stages, inspect raw model output. "
    "For the analyst review tool use `app.py`."
)

# ---------------------------------------------------------------------------
# Sidebar — shared input panel
# ---------------------------------------------------------------------------
eval_stems = _discover_eval_stems()

with st.sidebar:
    st.header("Input")

    if eval_stems:
        source_radio = st.radio("Source", ["Upload", "Eval dataset doc"], horizontal=True)
    else:
        source_radio = "Upload"
        st.caption("No eval docs found — using file uploader only.")

    uploaded = None
    eval_stem_sel = None

    if source_radio == "Upload":
        uploaded = st.file_uploader(
            "Upload a PDF or image",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
        )
        source_id = f"{uploaded.name}_{len(uploaded.getvalue())}" if uploaded else ""
    else:
        eval_stem_sel = st.selectbox("Eval document", eval_stems)
        source_id = eval_stem_sel or ""

    st.divider()
    st.header("Page scope")
    scope_radio = st.radio(
        "Scope",
        ["Single page", "All pages"],
        index=0,  # Single page default — hybrid is ~3 min/page
        help="Single page is the default because Hybrid engines take ~3 min/page.",
    )
    single_page = scope_radio == "Single page"
    page_num = 1
    if single_page:
        page_num = st.number_input(_PAGE_PICKER_LABEL, min_value=1, value=1, step=1)

    st.divider()
    with st.expander("Preprocessing", expanded=False):
        remove_stamps = st.checkbox("Remove colored stamps", value=True)
        sharpen = st.checkbox("Sharpen text", value=True)
        normalise = st.checkbox("Enhance contrast", value=True)
        deskew = st.checkbox("Deskew", value=True)
        normalise_table_backgrounds = st.checkbox("Normalise table backgrounds", value=True)

config = PreprocessConfig(
    remove_stamps=remove_stamps,
    sharpen=sharpen,
    normalise=normalise,
    deskew=deskew,
    normalise_table_backgrounds=normalise_table_backgrounds,
)

# Guard: nothing loaded yet
input_ready = (source_radio == "Upload" and uploaded is not None) or (
    source_radio == "Eval dataset doc" and eval_stem_sel is not None
)

if not input_ready:
    st.info("Choose an input in the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_compare, tab_inspect = st.tabs(["Compare engines", "Inspect stages"])

# ===========================================================================
# TAB 1 — Compare engines
# ===========================================================================
with tab_compare:
    selected_engines = st.multiselect(
        "Engines to compare",
        _ALL_ENGINES,
        default=_ALL_ENGINES,
    )

    with st.expander("ℹ️ What do these engines mean? (layout → structure → recognition)"):
        for _eng in _ALL_ENGINES:
            st.markdown(f"**{_eng}**")
            st.caption(_engine_caption(_eng))

    run_comparison = st.button("Run comparison", type="primary")

    if run_comparison and selected_engines:
        # Load input once (shared across engines)
        with st.status("Loading & preprocessing input...", expanded=True) as load_status:
            st.write("Ingesting and preprocessing...")
            try:
                orig_imgs, pre = _load_input(
                    source=source_radio,
                    upload_bytes=uploaded.getvalue() if uploaded else None,
                    upload_name=uploaded.name if uploaded else None,
                    eval_stem=eval_stem_sel,
                    single_page=single_page,
                    page_num=page_num,
                    config=config,
                )
                # Store for the Inspect tab to reuse
                st.session_state["lab_orig_imgs"] = orig_imgs
                st.session_state["lab_pre"] = pre
                st.session_state["lab_source_id"] = source_id
                st.session_state["lab_single_page"] = single_page
                st.session_state["lab_page_num"] = page_num
                st.session_state["lab_config"] = config
                load_status.update(label="Preprocessing done.", state="complete")
            except Exception as exc:
                load_status.update(label="Preprocessing failed", state="error")
                st.error(f"Failed to load input: {exc}")
                st.stop()

        for eng in selected_engines:
            key = _run_cache_key(source_id, single_page, page_num, config, eng)
            if _get_cached_run(key) is not None:
                st.caption(f"{eng}: using cached result.")
                continue
            with st.status(f"Running {eng}...", expanded=True) as eng_status:
                prog = st.progress(0, text=f"{eng}: starting...")

                def _make_on_page(engine_label: str, prog_widget):
                    def _on_page(idx: int, total: int) -> None:
                        prog_widget.progress(
                            (idx + 1) / total,
                            text=f"{engine_label}: page {idx + 1}/{total}",
                        )
                    return _on_page

                try:
                    result = _run_engine(eng, pre, on_page=_make_on_page(eng, prog))
                    prog.progress(1.0, text=f"{eng}: done.")
                    _store_run(key, result)
                    clear_device_cache()
                    eng_status.update(label=f"{eng} complete.", state="complete")
                except Exception as exc:
                    eng_status.update(label=f"{eng} failed", state="error")
                    st.error(f"{eng} failed: {exc}")

    # ------------------------------------------------------------------
    # Results rendering
    # ------------------------------------------------------------------
    results: dict[str, SuryaResult] = {}
    for eng in selected_engines:
        key = _run_cache_key(source_id, single_page, page_num, config, eng)
        cached = _get_cached_run(key)
        if cached is not None:
            results[eng] = cached

    if not results:
        st.caption("Select engines and click 'Run comparison' to see results.")
    else:
        # Retrieve stored originals/pre (may be from a prior run if not re-run)
        orig_imgs = st.session_state.get("lab_orig_imgs")
        pre = st.session_state.get("lab_pre")

        # Page picker for multi-page results
        any_result = next(iter(results.values()))
        n_pages = len(any_result.pages)
        view_page_idx = 0
        if n_pages > 1:
            view_page_idx = (
                st.number_input("View page", min_value=1, max_value=n_pages, value=1, step=1) - 1
            )

        # Side-by-side columns per engine
        cols = st.columns(len(results))
        for col, (eng, result) in zip(cols, results.items()):
            with col:
                st.subheader(eng)
                st.caption(_engine_caption(eng))
                page = result.pages[view_page_idx]
                page_img = (
                    pre.page_images[view_page_idx]
                    if pre and view_page_idx < len(pre.page_images)
                    else None
                )

                # --- Detection overlay ---
                if page_img is not None:
                    overlay = _overlay_for_engine(page_img, page, eng)
                    caption = f"{eng} — layout overlay"
                    if eng == _ENGINE_HYBRID_DOC:
                        caption += f" (yellow = DocLayout-YOLO)"
                    st.image(overlay, caption=caption, width="stretch")

                # --- Counts ---
                n_tables = len(page.tables)
                n_text = len(page.text_blocks)
                st.caption(f"Tables: {n_tables}  |  Text blocks: {n_text}")

                # --- Extracted table grids ---
                if page.tables:
                    for t_idx, table in enumerate(page.tables):
                        grid = pred_table_grid(table)
                        if grid:
                            n_rows = len(grid)
                            n_cols = max(len(r) for r in grid)
                            st.caption(f"Table {t_idx + 1}: {n_rows}×{n_cols}")
                            st.dataframe(
                                pd.DataFrame(grid),
                                width="stretch",
                            )
                else:
                    st.caption("No tables detected on this page.")

        # ------------------------------------------------------------------
        # Metrics section (eval docs, all-pages scope, GT available)
        # ------------------------------------------------------------------
        st.divider()
        st.subheader("Metrics vs ground truth")

        # GT is scored for eval docs: document-level (All pages → _document_gt.json,
        # stitched) or per-page (Single page → _p<N>_ground_truth.json, that page's
        # tables directly). Uploads have no GT.
        gt_grid = None
        gt_path = None
        doc_level = False
        if source_radio == "Eval dataset doc" and eval_stem_sel:
            if single_page:
                gt_path = _EVAL_REAL_DIR / f"{eval_stem_sel}_p{page_num}_ground_truth.json"
            else:
                gt_path = _EVAL_REAL_DIR / f"{eval_stem_sel}_document_gt.json"
                doc_level = True
            if gt_path.exists():
                gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
                gt_tables = gt_data.get("tables")
                gt_grid = gt_tables[0].get("data") if gt_tables else None

        if gt_grid is None:
            st.caption(
                "No ground truth for this input — visual comparison only. "
                "(Scored for eval docs when a matching GT file exists: "
                "`_document_gt.json` for All pages, `_p<N>_ground_truth.json` for a single page.)"
            )
        else:
            metrics_rows = []
            for eng, result in results.items():
                if doc_level:
                    pred_tables = merge_document_tables(result.pages)
                else:
                    # single page: score the viewed page's tables directly (no stitching)
                    pred_tables = result.pages[view_page_idx].tables
                m = evaluate_table(pred_tables, gt_grid)
                metrics_rows.append(
                    {
                        "Engine": eng,
                        "Pred dims": f"{m['pred_rows']}×{m['pred_cols']}",
                        "Cell_Accuracy": round(m["cell_accuracy"], 3),
                        "Cell_Content_Recall": round(m["cell_content_recall"], 3),
                        "Table_CER": round(m["table_cer"], 3),
                    }
                )
            gt_dims = f"{len(gt_grid)}×{max(len(r) for r in gt_grid)}"
            st.caption(f"GT dims: {gt_dims}  |  source: `{gt_path.name}`")
            st.dataframe(pd.DataFrame(metrics_rows, columns=_METRICS_COLUMNS), width="stretch")


# ===========================================================================
# TAB 2 — Inspect stages
# ===========================================================================
with tab_inspect:
    st.header("Inspect pipeline stages")

    eng_sel = st.selectbox("Engine", _ALL_ENGINES, index=0)
    st.caption(_engine_caption(eng_sel))

    # Check if we already have a cached run for this configuration
    insp_key = _run_cache_key(source_id, single_page, page_num, config, eng_sel)
    insp_result = _get_cached_run(insp_key)

    # Also recover stored preprocessing artifacts if available
    _stored_pre: Optional[PreprocessResult] = st.session_state.get("lab_pre")
    _stored_orig: Optional[list[np.ndarray]] = st.session_state.get("lab_orig_imgs")

    # Config match check: stored artifacts may be from a different source/config
    _stored_matches = (
        st.session_state.get("lab_source_id") == source_id
        and st.session_state.get("lab_single_page") == single_page
        and st.session_state.get("lab_page_num") == page_num
        and st.session_state.get("lab_config") == config
    )
    if not _stored_matches:
        _stored_pre = None
        _stored_orig = None

    if insp_result is None or _stored_pre is None:
        run_inspect = st.button(f"Run stage inspection ({eng_sel})", type="primary")
        if run_inspect:
            with st.status("Running stage inspection...", expanded=True) as insp_status:
                st.write("Ingesting and preprocessing...")
                try:
                    orig_imgs_i, pre_i = _load_input(
                        source=source_radio,
                        upload_bytes=uploaded.getvalue() if uploaded else None,
                        upload_name=uploaded.name if uploaded else None,
                        eval_stem=eval_stem_sel,
                        single_page=single_page,
                        page_num=page_num,
                        config=config,
                    )
                    st.session_state["lab_orig_imgs"] = orig_imgs_i
                    st.session_state["lab_pre"] = pre_i
                    st.session_state["lab_source_id"] = source_id
                    st.session_state["lab_single_page"] = single_page
                    st.session_state["lab_page_num"] = page_num
                    st.session_state["lab_config"] = config
                    _stored_pre = pre_i
                    _stored_orig = orig_imgs_i
                except Exception as exc:
                    insp_status.update(label="Load failed", state="error")
                    st.error(f"Failed to load input: {exc}")
                    st.stop()

                st.write(f"Running {eng_sel}...")
                prog_i = st.progress(0, text="Starting OCR...")

                def _on_page_i(idx: int, total: int) -> None:
                    prog_i.progress((idx + 1) / total, text=f"Page {idx + 1}/{total}")

                try:
                    insp_result = _run_engine(eng_sel, pre_i, on_page=_on_page_i)
                    prog_i.progress(1.0, text="Done.")
                    _store_run(insp_key, insp_result)
                    clear_device_cache()
                    insp_status.update(label="Done.", state="complete")
                except Exception as exc:
                    insp_status.update(label=f"{eng_sel} failed", state="error")
                    st.error(f"{eng_sel} failed: {exc}")
                    st.stop()

    if insp_result is None or _stored_pre is None:
        st.caption("Click the button above to run the inspection.")
        st.stop()

    # Page selector for multi-page results
    n_insp_pages = len(insp_result.pages)
    insp_page_idx = 0
    if n_insp_pages > 1:
        insp_page_idx = (
            st.number_input(
                "Inspect page", min_value=1, max_value=n_insp_pages, value=1, step=1
            )
            - 1
        )

    orig_page = (
        _stored_orig[insp_page_idx]
        if _stored_orig and insp_page_idx < len(_stored_orig)
        else None
    )
    pre_page = (
        _stored_pre.page_images[insp_page_idx]
        if insp_page_idx < len(_stored_pre.page_images)
        else None
    )
    insp_page_result = insp_result.pages[insp_page_idx]

    # -----------------------------------------------------------------------
    # Stage 1 — Ingest
    # -----------------------------------------------------------------------
    st.subheader("Stage 1 — Ingest")
    st.caption("Raw page image as loaded from the source (pre-preprocessing).")
    if orig_page is not None:
        st.image(orig_page, caption="Original page", width="stretch")
    else:
        st.caption("Original image not available.")

    # -----------------------------------------------------------------------
    # Stage 2 — Preprocess
    # -----------------------------------------------------------------------
    st.subheader("Stage 2 — Preprocess")
    if orig_page is not None and pre_page is not None:
        col_orig, col_pre = st.columns(2)
        with col_orig:
            st.image(orig_page, caption="Before preprocessing", width="stretch")
        with col_pre:
            st.image(pre_page, caption="After preprocessing", width="stretch")
    elif pre_page is not None:
        st.image(pre_page, caption="Preprocessed page", width="stretch")
    else:
        st.caption("Preprocessed image not available.")

    # -----------------------------------------------------------------------
    # Stage 3 — Layout
    # -----------------------------------------------------------------------
    st.subheader("Stage 3 — Layout")
    st.caption("Detected regions: text blocks (coloured by label) + tables (red).")
    if pre_page is not None:
        layout_overlay = _overlay_for_engine(pre_page, insp_page_result, eng_sel)
        caption_layout = "Layout overlay"
        if eng_sel == _ENGINE_HYBRID_DOC:
            caption_layout += " — yellow boxes = DocLayout-YOLO table regions"
        st.image(layout_overlay, caption=caption_layout, width="stretch")
        n_tbl = len(insp_page_result.tables)
        n_txt = len(insp_page_result.text_blocks)
        st.caption(f"Tables: {n_tbl}  |  Text blocks: {n_txt}")
    else:
        st.caption("Preprocessed image not available for layout overlay.")

    # -----------------------------------------------------------------------
    # Stage 4 — Structure (SLANet) — hybrid engines only
    # -----------------------------------------------------------------------
    st.subheader("Stage 4 — Structure (SLANet)")
    if eng_sel == _ENGINE_SURYA:
        st.caption(
            "SLANet table-structure analysis applies only to the hybrid engines. "
            "Surya detects and reads table regions directly without a separate "
            "structure step."
        )
    elif eng_sel == _ENGINE_SURYA_KIRI:
        st.caption(
            "Surya + Kiri uses Surya's TableRecPredictor for cell structure "
            "(not SLANet); each detected cell is then read by KiriOCR."
        )
    else:
        if not insp_page_result.tables:
            st.caption("No table regions detected on this page.")
        elif pre_page is None:
            st.caption("Preprocessed image not available for SLANet crop.")
        else:
            for t_idx, table in enumerate(insp_page_result.tables):
                bbox = _table_bbox(table)
                if not bbox or len(bbox) < 4:
                    st.caption(f"Table {t_idx + 1}: no bbox available.")
                    continue
                x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                h, w = pre_page.shape[:2]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 <= x0 or y1 <= y0:
                    st.caption(f"Table {t_idx + 1}: degenerate bbox, skipping.")
                    continue
                crop = pre_page[y0:y1, x0:x1]
                cells = predict_cells(crop)
                # Draw SLANet cell bboxes on the crop
                crop_vis = _draw_boxes(crop, [c["bbox"] for c in cells], _SLANET_CELL_COLOR)
                if cells:
                    n_rows_sla = max(c["row_id"] for c in cells) + 1
                    n_cols_sla = max(c["col_id"] for c in cells) + 1
                    dims = f"{n_rows_sla}×{n_cols_sla}"
                else:
                    dims = "no cells"
                st.caption(f"Table {t_idx + 1} — SLANet: {dims} (purple = cell bboxes)")
                st.image(crop_vis, caption=f"Table {t_idx + 1} crop", width="stretch")

    # -----------------------------------------------------------------------
    # Stage 5 — Recognition
    # -----------------------------------------------------------------------
    st.subheader("Stage 5 — Recognition")
    st.caption("Extracted table grids and raw OCR text from this page.")
    if insp_page_result.tables:
        for t_idx, table in enumerate(insp_page_result.tables):
            grid = pred_table_grid(table)
            if grid:
                n_rows_r = len(grid)
                n_cols_r = max(len(r) for r in grid)
                st.caption(f"Table {t_idx + 1}: {n_rows_r}×{n_cols_r}")
                st.dataframe(pd.DataFrame(grid), width="stretch")
            else:
                st.caption(f"Table {t_idx + 1}: empty grid.")
    else:
        st.caption("No tables detected.")

    if insp_page_result.ocr_text:
        st.markdown("**OCR text (raw)**")
        st.write(insp_page_result.ocr_text)
    else:
        st.caption("No OCR text on this page.")
