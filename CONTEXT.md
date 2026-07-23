# CONTEXT.md — Khmer OCR Pipeline

Khmer-language document OCR pipeline for GDDE financial documents
(e.g. ARDB forms). Entry points: a **React review workspace** (`frontend/`,
primary — served at `http://localhost:8600/app` by `uv run python -m
webapp.main`, which also serves the **NiceGUI fallback UI** at `/`), an older
Streamlit UI (`app.py`, legacy), and a CLI batch processor
(`src/khmer_pipeline/pipeline.py`).

> For the *why* behind major design decisions (the table-cell redesign, the
> evaluation framework, benchmark results), see `docs/PROJECT_LOG.md`.

## Tech stack
- Python >=3.11, managed with `uv` (pyproject.toml + uv.lock)
- OpenCV (`opencv-python-headless`) — image preprocessing
- PyMuPDF (`fitz`) — PDF ingestion
- `surya-ocr` (pinned `>=0.20.0,<0.21`) — layout detection, OCR, table recognition. Device is auto-selected by `utils/device.py` (`configure_runtime()` sets `TORCH_DEVICE` → CUDA on NVIDIA, MPS on Apple Silicon, else CPU). On Apple Silicon, `setup-metal-macos.sh` opts into the faster built-in llamacpp Metal backend (`SURYA_INFERENCE_BACKEND=llamacpp`), which `utils/device.py` respects. `mlx-lm` is a Mac-only (marker-gated) dependency; see `Dockerfile` for the Linux/GPU lane.
- `mlx-lm` + `transformers` (pinned, see pyproject.toml) — Qwen2.5-7B-Instruct-4bit text correction
- React 19 + Vite + TypeScript + Tailwind 4 + AG Grid community + TanStack Query (`frontend/`) — primary review workspace, served at `/app` from the built `frontend/dist` (build: `cd frontend && npm run build`, or `./dev.sh build`; dev: `./dev.sh` starts backend + Vite HMR at :5173/app/, proxying `/api` to :8600)
- NiceGUI (pinned `>=2.0,<3.0`) — fallback review UI (`webapp/main.py`) AND the FastAPI host for the REST layer (`webapp/api.py` registers routes on `nicegui.app`, a FastAPI subclass — one process, models load once); Streamlit >=1.35 — legacy UI (`app.py`)
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
| 3. Surya OCR | `engines/surya.py` | `run_surya(PreprocessResult, on_page=callback) -> SuryaResult` | Layout detection + OCR + table recognition via lazily-loaded Surya model singletons. Issues (low confidence, phantom cells, OCR/table failures) collected in `SuryaResult.warnings`. |
| 4. Postprocess | `postprocess.py` | `postprocess(SuryaResult, skip_qwen, anomaly_threshold) -> PostprocessResult` | Rule-based Khmer text correction; falls back to Qwen2.5-VL when the anomaly score (fraction of non-Khmer/non-Latin chars) exceeds `anomaly_threshold`. Table cells are copy-on-write normalized + passed through the GDDE-domain cell rules (riel-prefix repair, percent digit fold — see `_apply_cell_rules`); foreign-script characters are scrubbed from cells AND page text (output is Khmer/English only, warned per removal); pipe-only gridline noise in cells is emptied; malformed-number cells (dot-drop/digit-duplication patterns) are flagged (confidence cap + `PostprocessResult.warnings`), never rewritten. |
| 5. Export | `export.py` | `export(PostprocessResult, convert_numerals, repair_tables, stitch_pages) -> ExportResult` | Produces document JSON + per-table CSV/Excel. Optional Khmer->Arabic numeral conversion, table-grid repair (pads short rows), and `stitch_pages` to join a table that continues across pages into one. |

Model checkpoints and tunable thresholds (Surya checkpoints, Qwen model
path, `ANOMALY_THRESHOLD`, `CONFIDENCE_LOW`/`CONFIDENCE_MID`) all live in
`model_config.py` — change them there, not inline in stage modules.

`_process_page` in `engines/surya.py` wraps its entire body in a try/except: any
unexpected failure (layout/OCR/our own code) is caught, logs a
"Critical failure processing page N" warning, and returns an empty
`SuryaPageResult` so one bad page doesn't crash a multi-page run.

## Engine Swappability (Strategy Pattern)

`src/khmer_pipeline/engines/protocols.py` defines two structural interfaces:
`OCREngine` (Stage 3: `(PreprocessResult, on_page=...) -> SuryaResult`) and
`CorrectionEngine` (Stage 4: `(SuryaResult, skip_qwen=..., anomaly_threshold=...)
-> PostprocessResult`). `src/khmer_pipeline/engines/engine_registry.py` is the single
source of truth for which implementation is active: it maps the `OCR_ENGINE` env var
(`surya` (default) / `surya_kiri` / `surya_kiri_vlm` / `tesseract` / `hybrid`) to the OCR
engine and binds the
correction engine, exposing them as `ACTIVE_OCR_ENGINE` / `ACTIVE_CORRECTION_ENGINE`.
(`surya_kiri` = Surya layout + TableRec structure + vendored Kiri CTC per-cell recognition;
`surya_kiri_vlm` = plain Surya (VLM) + gated Kiri re-read of Khmer-heavy cells; `hybrid` =
SLANet table grid + Surya row-strip recognition, for dense fragmented tables.)
The `hybrid` engine has two further env knobs: `KHMER_HYBRID_MODE` (`rowband` (default)
/ `cell`) and `KHMER_LAYOUT_DETECTOR` (`surya` (default) / `doclayout`) which chooses the
table-region source — Surya layout + geometric merge, or DocLayout-YOLO via `rapid_layout`.
`doclayout` is opt-in and lost the end-to-end A/B (it clips the leftmost label columns); see
`docs/PROJECT_LOG.md` §2.23–2.24.
The `surya_kiri` engine has one env knob: `KHMER_KIRI_STRUCTURE` (`tablerec` (default) /
`merged` / `slanet`) selecting its table-structure source. `tablerec` splits column-spanning
headers mid-text ("14-06-26" → "14"|"6-26") — a known, accepted limitation: it never merges
data cells. `merged` (opt-in) fixes the spans via SLANet box proposals + a pixel-evidence
separator check; it passed the eval gate but produced a data-cell false merge in production
UI use, so it is NOT the default (data integrity > header cosmetics). `slanet` measured
worse on data grids — comparison only. See `docs/PROJECT_LOG.md` §2.40.
A fourth engine, `surya_kiri_vlm` (§2.41), runs plain Surya IN FULL (VLM structure+text —
spans correct) and re-reads only Khmer-heavy cells with Kiri, gated on exact VLM↔TableRec
grid-shape agreement. Its floor is plain Surya (every fallback keeps Surya's text); where
Surya's structure holds it beats both other engines (ARDB p3: CellAcc 0.98+, NumAcc 1.0).
Slowest engine (the table VLM runs).

**Rule:** orchestrators (`pipeline.py`, `app.py`) must only import execution
functions (`ACTIVE_OCR_ENGINE`, `ACTIVE_CORRECTION_ENGINE`) from
`engines/engine_registry.py`, never directly from `engines/surya.py`/`postprocess.py`.
State-checking helpers (`models_loaded`, `preload_models`, `qwen_loaded`) are
exempt and still imported directly from the stage modules — they're not part
of the swappable execution path.

To add a new engine: write a wrapper function matching the relevant Protocol's
`__call__` signature exactly (including the `skip_qwen` parameter name for
`CorrectionEngine`), then reassign `ACTIVE_OCR_ENGINE`/`ACTIVE_CORRECTION_ENGINE`
in `engines/engine_registry.py` — that one-line change is the only thing orchestrators
need to swap models.

## Memory management (`utils/memory.py`)

`src/khmer_pipeline/utils/memory.py` provides `clear_device_cache()` —
`gc.collect()` + `torch.cuda.empty_cache()` (Linux/NVIDIA) + `mx.clear_cache()`
(MLX/Qwen), each best-effort/wrapped in try/except. On Apple Silicon, Surya 0.20+
delegates to a C++ `llama-server` process that manages its own VRAM, so
`torch.mps.empty_cache()` is not called there.
Called after every stage in both `pipeline.py` and `app.py`, and also
after any page in `postprocess()` where `qwen_used` is true. Exists to
avoid OOM on 24GB unified-memory Macs during multi-stage ML inference —
call it after any new heavy model invocation you add.

## UI (React workspace at `/app` — primary)

**Stitching is an EXPORT choice here, not an extraction one (§2.43):** the React
UI always runs with `stitch_pages=False` so per-page tables (and therefore
page↔image linking) survive; joining continuation tables happens at export via
`webapp/tables.py::stitch_grids` on the *edited* grids, chosen with `?combine=`
(default true) and surfaced in the Export menu. The pipeline's own `stitch_pages`
is unchanged for the CLI/NiceGUI. Don't reintroduce a review-time stitch toggle.

Three-zone analyst workspace (`frontend/src/`): queue rail (`components/queue/`),
zoom/pan page viewer with confidence/region overlays and two-way table↔image
linking (`components/viewer/PageViewer.tsx`), and AG Grid table editing with
undo/redo, row context menu, diff view, ✓ verify, and per-table CSV
(`components/review/`). One morphing primary action (Upload → Run → Export);
Issues (N) triage panel jumps to low-confidence cells (`n`/`p`); Ctrl-F
find/replace; settings drawer; `?` shortcuts overlay. The confidence percentages
these surfaces show are the recognizer's own self-report, NOT an accuracy measure,
and text blocks and table cells derive them from different models — see
`docs/GLOSSARY.md` §5 "Confidence ≠ accuracy" before treating one as evidence. Server state lives in
`webapp/registry.py` (process-global `Document` dict + a global `run_lock` —
one GPU, concurrent runs get 409), so the React app is **refresh-safe**:
reloading the tab keeps queue/results/edits; ■ Stop is the cancel path (page-
granular via `Progress.cancel_requested`). REST layer: `webapp/api.py` — thin
handlers over `runner`/`tables`/`downloads`/`edits`, JSON errors via a custom
`ApiError` (NiceGUI intercepts `HTTPException` with HTML pages), RFC 5987
Content-Disposition for Khmer filenames. Tested in `tests/test_webapp_api.py`
(FastAPI TestClient — no context manager, NiceGUI lifespan needs `ui.run`).
Khmer rendering: bundled Noto Sans Khmer (`frontend/src/assets/fonts/`, OFL),
`.khmer-content` line-height 1.9, adjustable size (A−/A+, localStorage).

## UI (`webapp/main.py`, NiceGUI — fallback)

Modular NiceGUI app; presentation layer only — it imports and calls the same
pipeline functions as `app.py` (nothing pipeline-side changed). Run with
`uv run python -m webapp.main` (port 8600). Modules: `settings.py` (`Settings`
dataclass + `settings_key` re-run guard + page-range logic), `state.py`
(`Document` per uploaded file + `AppState` session holding shared `Settings` +
a document list — enables **batch upload**; plus a thread-safe `Progress`),
`runner.py` (drives the 5 stages via **`nicegui.run.io_bound`** — a thread pool,
so the multi-GB Surya/Qwen models stay loaded in-process; `clear_device_cache()`
between stages; **not** `run.cpu_bound`, which would fork+reload them),
`tables.py`/`downloads.py`/`edits.py` (pure, unit-tested ports of app.py's
two-scope table build, JSON edit-patch, exports, and bulk find/replace),
`components.py` (unified confidence palette used by BOTH the image overlay and
cell tinting, SVG overlay builder, `table_bbox_index`), `main.py` (the page).

Flow: sidebar config -> multi-file upload -> "Run Extraction" / "Run all" ->
5 stages off the event loop with a live `ui.timer` progress bar -> per-document
paginated **side-by-side review** (`ui.interactive_image` with an SVG overlay
left, editable `ui.aggrid` right) -> downloads via `ui.download` (built lazily
on click, so no re-run/caching needed). **Table↔image linking:** clicking a grid
cell highlights its whole table region on the image and vice-versa. NOTE — the
pipeline exposes **no per-cell geometry** (plain Surya leaves cell bbox empty;
the hybrid engine discards SLANet cell boxes in `_build_table`), so linking is
table-level by necessity, not per-cell. UI-only; verified by running the app +
the `tests/test_webapp_*.py` unit tests (no browser tests).

## UI (`app.py`, Streamlit — legacy)

Single-file Streamlit app. Flow: sidebar config (Primary settings + a
collapsed Advanced expander) -> file upload -> "Run Extraction" button ->
runs all 5 stages (with `clear_device_cache()` after each, results cached
into `st.session_state` incrementally per stage) -> paginated
**side-by-side review** (page image + editable tables) -> downloads
(patched JSON + per-table CSV / Excel / zip). When per-cell confidence
exists (surya_kiri), each table also gets a collapsed read-only
**"🔍 Confidence view"** (cells tinted by the `CELL_CONF_LOW`/`CELL_CONF_MID`
buckets from `model_config.py`); tables always render without it too —
never gate display on optional data.

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

## Benchmark runner (`evaluation/run_benchmark.py`)

```bash
uv run python -m khmer_pipeline.evaluation.run_benchmark [--data-dir eval/datasets/synthetic_tables eval/datasets/synthetic_documents] [--run-dir eval/runs/my_run]
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
  `engines/surya.py` (call sites).
- **Change correction rules / Qwen behavior** -> `postprocess.py`.
- **Change export format / CSV/JSON shape** -> `export.py` +
  `models.py` (`ExportResult`).
- **UI changes** -> `frontend/` (React, primary; iterate with `./dev.sh` for
  hot reload, `./dev.sh build` when :8600/app must serve the new bundle)
  + `webapp/api.py` if the server must expose new data (TDD in
  `tests/test_webapp_api.py`); `webapp/main.py` (NiceGUI fallback) or `app.py`
  (Streamlit legacy) only when keeping them in sync matters.
- **Tests** mirror `src/khmer_pipeline/` 1:1 in `tests/` (e.g.
  `preprocess.py` <-> `tests/test_preprocess.py`).

## Further history

`docs/superpowers/plans/` and `docs/superpowers/specs/` contain the
design docs and implementation plans for each stage (stages 1-6 +
batch region OCR) — useful for "why was this built this way" context.

See `CLAUDE.md` for coding conventions.
