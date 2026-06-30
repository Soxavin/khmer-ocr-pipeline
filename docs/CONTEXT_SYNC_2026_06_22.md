# Context Sync Report — Khmer OCR Pipeline (GDDE Internship)
*Generated 2026-06-22. Scope: everything since Phase 0 of the Final Internship Sprint.*

### 1. Executive Summary
Since Phase 0 kicked off, we completed the **Reality Check (Phase 0)**, the **multi-table evaluation work and first real-document benchmark (Phase 1 core)**, the **Tesseract baseline engine (Phase 2 code)**, and a full **Phase 3 productionization pass** (UI overhaul, llama-server lifecycle, reproducibility freeze, memory guard). We **mostly stayed on the Opus roadmap but made two deliberate, evidence-driven pivots**: (a) we *added* an unplanned **Stage-4 redesign** — replacing an ineffective 7B-LLM correction step with a deterministic Khmer Unicode normalizer after data showed the LLM did nothing useful; and (b) we **reframed the thesis centerpiece** from "OCR character accuracy" to "**table-structure fragmentation is the real bottleneck**," which the real-document data forced. The infrastructure/eval layer is now mature and arguably ahead of the data (only one real document is labeled), so the strategic emphasis has shifted toward results, productionization, and writing rather than more machinery.

### 2. Phase 0 Deliverables (Reality Check)
**Committed in `8742dc6`.**

- **`inspect_pdf.py`** — Classifies each PDF into `born_digital_unicode` / `likely_legacy_encoded` / `scanned_image_only` / `mixed_or_unknown`, using the Khmer Unicode block ratio (U+1780–U+17FF) with thresholds `_MIN_TEXT_CHARS=100`, `_UNICODE_KHMER_RATIO=0.5`, `_LEGACY_KHMER_RATIO=0.15`; also reports text-layer presence and raster DPI. Emits a report + JSON (`inspect_report.json`, gitignored).
  - **Run on the real GDDE document. Result: NO legacy Limon/ABC encoding was detected.** The doc classified as **born-digital Unicode** — *but with a critical caveat*: its embedded text layer is **garbled due to a broken ToUnicode CMap**. So the "free" text layer is unusable, and **OCR on rendered pixels is genuinely necessary** (text extraction is not a shortcut). This was the key Phase 0 decision-gate outcome: no transcoding sub-task needed, but the text layer can't be trusted as ground truth.

- **`harvest_ground_truth.py`** — Renders born-digital PDF pages → page PNGs + auto-drafts `*_ground_truth.json` stubs (paragraphs from the text layer, NFC-normalized; tables stubbed for manual fill) into `eval/datasets/real/`. It ran, but because the real doc's text layer is garbled (above), the harvested paragraphs needed **substantial manual correction** rather than being usable as-is.

- **Real Dataset Status** — `eval/datasets/real/` currently holds **one real document**: a **GDDE daily market-price report dated 09.06.26**, **3 pages** (`p1`, `p2`, `p3`), each as a PNG + `*_ground_truth.json`. State: **manually labeled** — the user hand-corrected the paragraph text; the 9-column table grids were drafted from the corrected text for p1 (24 rows) and p2 (28 rows), with a couple of meat-row price-placement cells flagged to verify against the PNG. **These real files are gitignored** (sensitive financial inputs). This is the project's only real labeled doc so far.

### 3. Codebase & Architecture Changes
**New modules**
- `src/khmer_pipeline/inspect_pdf.py`, `harvest_ground_truth.py` (Phase 0).
- `src/khmer_pipeline/tesseract_engine.py` — `run_tesseract` baseline engine (commit `d2fe5ae`). Re-packs Tesseract's parallel-list `image_to_data` output into the **exact same 7-key `text_blocks` shape Surya emits** (`text, bbox, polygon, confidence, label, region_label, reading_order`), so the eval harness scores it unchanged. `tables=[]` (classic OCR has no structure). Lazy `pytesseract` import with a brew-hint `ImportError`; 21 offline tests (monkeypatched).
- `src/khmer_pipeline/khmer_normalize.py` — deterministic Khmer Unicode normalizer (commit `d1bb7e2`): NFC → strip noise format chars (ZWSP/BOM/soft-hyphen; ZWNJ/ZWJ preserved) → collapse duplicate combining marks → whitespace tidy (**Tier A, on by default**); plus an **opt-in** canonical cluster reorder (**Tier B**, `reorder=False` default).
- `src/khmer_pipeline/backend_status.py` — `llama_server_running()` / `llama_server_pids()` via `pgrep` (commit `f578629`).
- `src/khmer_pipeline/fonts.py` — builds base64 `@font-face` CSS from vendored TTFs (commit `f099d77`).
- `src/khmer_pipeline/visualize_benchmark.py` — benchmark figures generator (commit `62cf1b1`, delegated then reviewed/verified).
- `stop-metal-macos.sh`, `docs/OPERATIONS.md`, `fonts/` (5 OFL TTFs + OFL license texts + `MANIFEST.txt` with sha256).

**Major refactors / behavior changes**
- **`evaluate_structure.py`** — `evaluate_table` now **combines ALL detected tables** (`combined = [row for t in pred_tables for row in pred_table_grid(t)]`) instead of `pred_tables[0]` only, so row-wise table fragmentation is scored fairly (commit `b9f5c26`); the `gt_grid is None` branch reports `tables_found: len(pred_tables)`. (Known limitation: handles row-fragmentation, **not** column-wise fragmentation.)
- **`postprocess.py`** — `_apply_rules` now calls `normalize_khmer()`; **Qwen demoted to opt-in** (`postprocess`/`_correct_page` default `skip_qwen=True`). Qwen logic (anomaly-score routing, batched single-call-per-page, graceful fallback) preserved but off by default.
- **`engine_registry.py`** — `OCR_ENGINE` env switch (`surya` default, unknown → `run_surya`).
- **`surya.py`** — all six user-facing warnings switched to **1-based** page/table numbers (was "page 0").
- **`pipeline.py`** — CLI `--no-qwen` replaced by opt-in `--qwen`.
- **`run_benchmark.py`** — added `--qwen` flag to optionally measure the LLM path; per-run output folders (`eval/runs/<ts>_<engine>/` with `results.csv` + `manifest.json` + `summary.txt` + `predictions/`).
- **Both synthetic generators** — switched from live `fonts.googleapis.com` `<link>` to **offline base64 `@font-face`** via `fonts.py` (deterministic, no network).
- **`app.py`** (commit `1ac1078`) — major UI overhaul: per-page review as **`st.tabs`** (Images/Text/Tables/Corrected/Edit), **persistent warnings panel** (survives pagination), results-overview metrics, **"Download everything (.zip)"**, table grids with first-row-header toggle + **de-duplicated column names**, sidebar **OCR-backend status indicator**, **memory soft-guard** warning (`_MEMORY_WARN_PAGES=20`, provisional), and Qwen checkbox now off-by-default.

**Bug fixes / edge cases**
- `analyze_benchmark.py` ZeroDivision when all `Tables_Expected="0"` on real docs (commit `0b9962d`).
- **pyarrow "Duplicate column names" crash** in the Streamlit table view on the real table (its header row has repeated/blank cells) → fixed with `_unique_headers()` (blanks → `col{n}`, dups → ` (2)`).
- "Page 0" in the UI warnings panel → 1-based fix in `surya.py`/`tesseract_engine.py`.
- Declared **`pandas` dependency** (used by `app.py` but previously undeclared), pinned `>=2.0,<4.0` and locked to **3.0.3** (commit `c14d37f`) — see §5.

### 4. Evaluation & Benchmark Results
**Real-document benchmark** (`eval/runs/20260622_114939_run_surya`, engine `run_surya`, raw render):

| Page | Tables_Found | Document_CER | Note |
|---|---|---|---|
| p1 | 1 | 0.30 | clean single table |
| p2 | **8** | **0.70** | one table fragmented into 8 regions |
| p3 | 1 | 0.22 | clean single table |

**Headline finding: the bottleneck is layout/table-structure detection, not character recognition.** On the saved OCR-vs-GT prediction dumps, character recognition is strong (~90%+ of product names and **all** numeric values correct; only minor slips like `ត្រកួន→ត្រកូន`, riel sign `៛→រ`). But on dense page 2, Surya's layout model **fragmented one table into 8 regions**, serializing content column-wise (all names, then all numbers, then all percentages), destroying row↔value associations. Because CER is order-sensitive, this *reordering* — not bad OCR — drives Document_CER to 0.70. Real-data aggregate metrics are correspondingly harsh: **Cell_Accuracy ≈ 0.05, Text_CER ≈ 0.95, Paragraph_Leak very high (~748)**, all symptomatic of fragmentation rather than recognition failure.

**Stage-4 A/B (deterministic normalizer), variance-free, 33 images.** Two live OCR runs (`20260622_154407` raw vs `20260622_155048` deterministic) showed table metrics drifting ~6 points purely from **OCR run-to-run non-determinism** — so the normalizer was measured on **fixed** OCR output (the saved prediction dumps) instead:

| dataset | n | raw | Tier A | + reorder |
|---|---|---|---|---|
| synthetic_tables | 15 | 0.1650 | 0.1650 | 0.1644 |
| synthetic_documents | 15 | 0.4498 | **0.4353** | 0.4353 |
| real | 3 | 0.5030 | 0.5030 | 0.5031 |
| ALL | 33 | 0.3252 | **0.3186** | 0.3183 |

→ **Tier A is a real, safe win** (synthetic_documents CER −3.2% relative, neutral elsewhere, never hurts). **Reorder is below the noise floor** (Surya already emits canonical Khmer) → kept off behind a flag.

**Synthetic baseline (reference, PROJECT_LOG §3):** Noto Sans Khmer decisively best (Cell_Acc 0.82, Text_CER 0.044); Battambang/Hanuman usable; Moul/Fasthand poor (decorative typefaces). On synthetics, `Tables_Found == Expected == 1` and `Paragraph_Leak == 0` — i.e., fragmentation is a **real-document-only** phenomenon, the core synthetic-vs-real gap.

**Tesseract baseline: NOT yet benchmarked.** The engine is committed and unit-tested, but **no `run_tesseract` benchmark run exists** — the Phase 2 "run both engines + `analyze_benchmark <surya_run> <tesseract_run>`" comparison is still outstanding. (Caveat when run: Tesseract emits no table structure → text-only comparison; it inserts spaces between Khmer clusters, inflating its CER — a real property, fair to report.)

**No preprocessing A/B yet** — the OpenCV preprocessing stack remains **untested on real scans** (we have no scanned doc; the one real doc is born-digital).

### 5. Blockers, Bugs, & Pivots
- **Pivot 1 — Stage-4 redesign (unplanned).** The existing Stage-4 used **Qwen2.5-7B-Instruct (MLX, ~4GB)**, a *general* LLM never trained for Khmer OCR, while the "deterministic" layer was a **no-op** (empty rule dict, NFC only). Slow (multi-minute load) and useless. Pivoted to a deterministic normalizer (Tier A) and made Qwen opt-in. Honest thesis takeaway: *a general LLM did not help; deterministic Unicode normalization does, modestly.*
- **Pivot 2 — thesis framing.** Real data forced reframing from "character accuracy" to "**table-structure fragmentation is the bottleneck**." Order-sensitive Document_CER over-penalizes column-wise fragmentation; the metrics that matter for financial tables are structural (`Cell_Accuracy`, `Tables_Found vs Expected`).
- **Upstream/structural issue: table fragmentation.** Surya's layout model splits one dense table into many regions; our multi-table aggregation fixes *row-wise* fragmentation in scoring but **column-wise fragmentation (the page-2 case) is unsolved** — flagged as the highest-value future engineering target (table-region reconstruction).
- **Garbled text layer (broken ToUnicode CMap)** on the born-digital PDF — abandoned using the embedded text layer as a shortcut/ground-truth; OCR on pixels is mandatory.
- **OCR non-determinism** confounds two-run comparisons → adopted fixed-output (prediction-dump) A/B methodology for correction evaluation.
- **pandas pin churn (resolved).** Initially over-pinned `<3.0`, which force-downgraded the already-working transitively-installed 3.0.3 → relaxed to `>=2.0,<4.0`, locked at 3.0.3.
- **Anti-overengineering checkpoint.** Explicitly assessed "are we overengineering?" and agreed the eval/infra is ahead of the data; deliberately declined Docker (breaks Metal/MLX on macOS — future-work note only) and declined an auto-kill-on-exit footgun (explicit stop script instead).

### 6. Current State & Immediate Next Steps
**Repo state: clean working tree, all work committed.** Recent history (newest first):
- `c14d37f` chore(deps): relax pandas pin to `>=2.0,<4.0`
- `1ac1078` feat(ui): tabbed review, persistent warnings, zip export, backend status, memory guard
- `62cf1b1` feat(eval): benchmark figures generator (visualize_benchmark)
- `b39bdbb` docs: PROJECT_LOG §2.11
- `f099d77` feat(repro): vendor OFL Khmer fonts, offline generation
- `f578629` feat(ops): llama-server teardown + status helper, OPERATIONS guide
- `d1bb7e2` feat(stage4): deterministic Khmer normalizer; Qwen opt-in
- `d2fe5ae` feat(eval): Tesseract OCR baseline engine

**Test suite: 294 passing** (`uv run pytest -q`); `python3 -m py_compile app.py src/khmer_pipeline/*.py` clean. Stack: Surya 0.20 (llamacpp Metal backend, resident `llama-server`, `KEEP_ALIVE=true`), MLX Qwen (opt-in), 24 GB M4 Pro. Conventions in `CLAUDE.md`; decision log in `docs/PROJECT_LOG.md` (through §2.11); roadmap in `docs/FINAL_SPRINT_PLAN.md`; ops in `docs/OPERATIONS.md`.

**Immediate next tasks (priority order):**
1. **Run the Tesseract vs Surya benchmark** (missing Phase 2 measurement): `OCR_ENGINE=tesseract` run on synthetic + real, then `analyze_benchmark <surya_run> <tesseract_run>`, and feed both runs to `visualize_benchmark` (the `engine_comparison` chart will then render). **Prerequisite:** system `tesseract` + `khm` traineddata (`brew install tesseract tesseract-lang`) — not yet confirmed present.
2. **Memory stress test (parked, user-gated):** awaiting a large multi-page **scanned** PDF in `sample_data/`; then measure the real page limit and set `_MEMORY_WARN_PAGES` (currently provisional 20) + document in `OPERATIONS.md`.
3. **Thesis report** `docs/REPORT.md` — assemble from PROJECT_LOG + manifests + the `visualize_benchmark` figures.
4. **(Stretch) Column-fragmentation table reconstruction** — highest-value real-world engineering target; would touch `surya.py`/eval.

**Security constraints still in force:** `sample_data/`, `.streamlit/`, `eval/datasets/`, `eval/runs/`, `benchmark_results*.csv`, `inspect_report.json` are gitignored (real financial inputs never committed); `fonts/` IS tracked (OFL). Always commit with explicit `git add <path>` (never `git add -A`); commit footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

> **Methodology caveat for any reader:** the live two-run aggregate numbers (`154407` raw / `155048` corrected) are **confounded by OCR non-determinism** — cite the **fixed-output A/B table** in §4 for the normalizer's true effect, not the run-to-run deltas.
