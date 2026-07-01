# CONTEXT.md — Khmer OCR Pipeline

Khmer-language document OCR pipeline for GDDE financial documents
(e.g. ARDB forms). Two entry points: a Streamlit UI (`app.py`) for
interactive use, and a CLI batch processor
(`src/khmer_pipeline/pipeline.py`).

> For the *why* behind major design decisions (the table-cell redesign, the
> evaluation framework, benchmark results), see `docs/PROJECT_LOG.md`.

## Tech stack
- Python >=3.11, managed with `uv` (pyproject.toml + uv.lock)
- OpenCV (`opencv-python-headless`) — image preprocessing
- PyMuPDF (`fitz`) — PDF ingestion
- `surya-ocr` (pinned `>=0.20.0,<0.21`) — layout detection, OCR, table recognition. Device is auto-selected by `device.py` (`configure_runtime()` sets `TORCH_DEVICE` → CUDA on NVIDIA, MPS on Apple Silicon, else CPU). On Apple Silicon, `setup-metal-macos.sh` opts into the faster built-in llamacpp Metal backend (`SURYA_INFERENCE_BACKEND=llamacpp`), which `device.py` respects. `mlx-lm` is a Mac-only (marker-gated) dependency; see `Dockerfile` for the Linux/GPU lane.
- `mlx-lm` + `transformers` (pinned, see pyproject.toml) — Qwen2.5-7B-Instruct-4bit text correction
- Streamlit >=1.35 — UI
- pytest — tests

## Pipeline architecture

Data flows through dataclasses in `src/khmer_pipeline/models.py`:

```
IngestResult -> PreprocessResult -> SuryaResult -> PostprocessResult -> ExportResult
```

| Stage | Module | Entry point | Notes |
|---|---|---|---|
| 1. Ingest | `ingest.py` | `ingest(bytes, name, dpi) -> IngestResult` | PDF/image -> page images (numpy arrays). `MAX_PAGES = 50`. |
| 2. Preprocess | `preprocess.py` | `preprocess(IngestResult, PreprocessConfig) -> PreprocessResult` | OpenCV cleanup: deskew, stamp removal, sharpen, contrast, table-background normalisation. All steps are `PreprocessConfig` flags (default on). |
| 3. Surya OCR | `surya.py` | `run_surya(PreprocessResult, on_page=callback) -> SuryaResult` | Layout detection + OCR + table recognition via lazily-loaded Surya model singletons. Issues (low confidence, phantom cells, OCR/table failures) collected in `SuryaResult.warnings`. |
| 4. Postprocess | `postprocess.py` | `postprocess(SuryaResult, skip_qwen, anomaly_threshold) -> PostprocessResult` | Rule-based Khmer text correction; falls back to Qwen2.5-VL when the anomaly score (fraction of non-Khmer/non-Latin chars) exceeds `anomaly_threshold`. |
| 5. Export | `export.py` | `export(PostprocessResult, convert_numerals, repair_tables, stitch_pages) -> ExportResult` | Produces document JSON + per-table CSV/Excel. Optional Khmer->Arabic numeral conversion, table-grid repair (pads short rows), and `stitch_pages` to join a table that continues across pages into one. |

Model checkpoints and tunable thresholds (Surya checkpoints, Qwen model
path, `ANOMALY_THRESHOLD`, `CONFIDENCE_LOW`/`CONFIDENCE_MID`) all live in
`model_config.py` — change them there, not inline in stage modules.

`_process_page` in `surya.py` wraps its entire body in a try/except: any
unexpected failure (layout/OCR/our own code) is caught, logs a
"Critical failure processing page N" warning, and returns an empty
`SuryaPageResult` so one bad page doesn't crash a multi-page run.

## Engine Swappability (Strategy Pattern)

`src/khmer_pipeline/protocols.py` defines two structural interfaces:
`OCREngine` (Stage 3: `(PreprocessResult, on_page=...) -> SuryaResult`) and
`CorrectionEngine` (Stage 4: `(SuryaResult, skip_qwen=..., anomaly_threshold=...)
-> PostprocessResult`). `src/khmer_pipeline/engine_registry.py` is the single
source of truth for which implementation is active: it maps the `OCR_ENGINE` env var
(`surya` (default) / `tesseract` / `hybrid`) to the OCR engine and binds the correction
engine, exposing them as `ACTIVE_OCR_ENGINE` / `ACTIVE_CORRECTION_ENGINE`. (`hybrid` =
SLANet table grid + Surya row-strip recognition, for dense fragmented tables.)
The `hybrid` engine has two further env knobs: `KHMER_HYBRID_MODE` (`rowband` (default)
/ `cell`) and `KHMER_LAYOUT_DETECTOR` (`surya` (default) / `doclayout`) which chooses the
table-region source — Surya layout + geometric merge, or DocLayout-YOLO via `rapid_layout`.
`doclayout` is opt-in and lost the end-to-end A/B (it clips the leftmost label columns); see
`docs/PROJECT_LOG.md` §2.23–2.24.

**Rule:** orchestrators (`pipeline.py`, `app.py`) must only import execution
functions (`ACTIVE_OCR_ENGINE`, `ACTIVE_CORRECTION_ENGINE`) from
`engine_registry.py`, never directly from `surya.py`/`postprocess.py`.
State-checking helpers (`models_loaded`, `preload_models`, `qwen_loaded`) are
exempt and still imported directly from the stage modules — they're not part
of the swappable execution path.

To add a new engine: write a wrapper function matching the relevant Protocol's
`__call__` signature exactly (including the `skip_qwen` parameter name for
`CorrectionEngine`), then reassign `ACTIVE_OCR_ENGINE`/`ACTIVE_CORRECTION_ENGINE`
in `engine_registry.py` — that one-line change is the only thing orchestrators
need to swap models.

## Memory management (`memory.py`)

`src/khmer_pipeline/memory.py` provides `clear_device_cache()` —
`gc.collect()` + `torch.cuda.empty_cache()` (Linux/NVIDIA) + `mx.clear_cache()`
(MLX/Qwen), each best-effort/wrapped in try/except. On Apple Silicon, Surya 0.20+
delegates to a C++ `llama-server` process that manages its own VRAM, so
`torch.mps.empty_cache()` is not called there.
Called after every stage in both `pipeline.py` and `app.py`, and also
after any page in `postprocess()` where `qwen_used` is true. Exists to
avoid OOM on 24GB unified-memory Macs during multi-stage ML inference —
call it after any new heavy model invocation you add.

## UI (`app.py`)

Single-file Streamlit app. Flow: sidebar config (Primary settings + a
collapsed Advanced expander) -> file upload -> "Run Extraction" button ->
runs all 5 stages (with `clear_device_cache()` after each, results cached
into `st.session_state` incrementally per stage) -> paginated
**side-by-side review** (page image + editable tables) -> downloads
(patched JSON + per-table CSV / Excel / zip).

Results are cached in `st.session_state` keyed by a `settings_key`
string so re-renders don't re-run the pipeline. Once results exist, the
UI shows **one page at a time** via `st.session_state.current_page_idx`:
a "Jump to page" selectbox plus Previous/Next buttons (`st.rerun()` on
change), clamped to `[0, total_pages - 1]`. `current_page_idx` is reset
(`st.session_state.pop(...)`) whenever a new file is uploaded. Per-page
widget keys (`edit_{i}`, `edited_text_{i}`) are unchanged, so edits made
on a page persist when navigating away and back.

All `st.image`/`st.button`/`st.download_button` calls use `width="stretch"`
(installed Streamlit is 1.58, which supports `width=` on all of these) —
**do not use the deprecated `use_container_width=True`**.

## CLI (`pipeline.py`)

```bash
uv run python -m khmer_pipeline.pipeline input.pdf output/ [--dpi 200] [--no-deskew] [--no-qwen] ...
```
Same 5 stages, writes `<name>_extracted.json` + per-table CSVs to
`output/`, prints `WARNING:`-prefixed lines for anything in
`SuryaResult.warnings`. Calls `clear_device_cache()` after preprocess,
Surya, and postprocess (same as `app.py`).

## Benchmark runner (`run_benchmark.py`)

```bash
uv run python -m khmer_pipeline.run_benchmark [--data-dir eval/datasets/synthetic_tables eval/datasets/synthetic_documents] [--run-dir eval/runs/my_run]
```
Scans `--data-dir` for `*_ground_truth.json` files, runs the full pipeline
on each paired `.png` (raw render — no preprocessing), and writes one run
folder under `eval/runs/<YYYYMMDD_HHMMSS>_<engine>/` containing
`results.csv`, `manifest.json` (provenance + aggregates), and `summary.txt`
(captured analyze output). Calls `clear_device_cache()` after each image.
No API key required. See `eval/README.md` for full CLI options and metric
definitions.

## Where to look for X

- **Add/tune a preprocessing step** -> `preprocess.py` (`PreprocessConfig`
  field + `_preprocess_image` step order) + sidebar checkbox in `app.py` +
  CLI flag in `pipeline.py`. Follow the existing pattern (see `deskew`,
  `normalise_table_backgrounds`).
- **Change OCR/layout/table model** -> `model_config.py` (checkpoints) +
  `surya.py` (call sites).
- **Change correction rules / Qwen behavior** -> `postprocess.py`.
- **Change export format / CSV/JSON shape** -> `export.py` +
  `models.py` (`ExportResult`).
- **UI changes** -> `app.py`.
- **Tests** mirror `src/khmer_pipeline/` 1:1 in `tests/` (e.g.
  `preprocess.py` <-> `tests/test_preprocess.py`).

## Further history

`docs/superpowers/plans/` and `docs/superpowers/specs/` contain the
design docs and implementation plans for each stage (stages 1-6 +
batch region OCR) — useful for "why was this built this way" context.

See `CLAUDE.md` for coding conventions.
