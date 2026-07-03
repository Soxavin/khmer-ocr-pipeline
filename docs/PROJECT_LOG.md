# Project Engineering Log — Khmer OCR Pipeline

A curated record of the significant problems, root causes, design decisions, and
results during development. Intended as a reference for documentation and report
writing — it captures *why* the system looks the way it does, not an exhaustive
commit history. Newest milestones are toward the bottom of each section.

---

## 1. Overview

**Goal.** Extract structured data from Khmer-language financial/economic documents
(ARDB-style price tables, budget execution reports) into one CSV per table and one
JSON per document, for analysts at GDDE. A working prototype — no model
training.

**Pipeline.** Five in-memory stages, typed dataclasses between them:

```
IngestResult → PreprocessResult → SuryaResult → PostprocessResult → ExportResult
```

1. **Ingest** — PDF/image → page images.
2. **Preprocess** — OpenCV cleanup (deskew, stamp removal, sharpen, contrast, table-background normalisation).
3. **OCR** — Surya: layout detection + text recognition + table structure.
4. **Postprocess** — rule-based Khmer correction, with a Qwen LLM fallback for anomalous text.
5. **Export** — CSV (one per table, UTF-8 BOM) + document JSON.

**Stack.** Python 3.11 (managed with `uv`), Surya OCR `0.20.x` (llamacpp Metal
backend on Apple Silicon), Qwen2.5-7B-Instruct (MLX) for correction, OpenCV,
PyMuPDF, Streamlit UI + CLI batch runner.

**Hardware constraint.** Single 24 GB unified-memory M4 Pro Mac running PyTorch
(Surya) and MLX (Qwen) in the same process — memory pressure is a real design
factor (`clear_device_cache()` is called after each heavy stage).

---

## 2. Decision Records

Each entry: **Problem → Investigation → Decision → Outcome.**

### 2.1 Surya 0.17 → 0.20 migration

- **Problem.** The OCR engine was upgraded to Surya 0.20 ("Surya 2"), a ground-up
  rewrite with a different API; the old per-region call pattern no longer existed.
- **Investigation.** 0.20 introduces a shared `SuryaInferenceManager` across all
  predictors and a *block mode*: `rec_pred([img], layout_results=[lr])` makes one
  OCR call per layout region and returns HTML per block. On Apple Silicon it
  auto-selects a llamacpp Metal backend that runs a local `llama-server`.
- **Decision.** Rewrite `surya.py` for the new API; extract text from each block's
  HTML. Tune the backend for sequential page processing.
- **Outcome.** Working migration. Two backend settings mattered enormously:
  `SURYA_INFERENCE_KEEP_ALIVE=true` (the default `false` unloads the model from
  VRAM after every call → 15–30 s reload penalty *per call*) and
  `SURYA_INFERENCE_PARALLEL=1` (we process pages sequentially; the default 8
  reserved ~98 k context tokens of VRAM for nothing). See `setup-metal-macos.sh`.

### 2.2 Table cell text — "every cell shows the whole table"

- **Problem.** In the Streamlit table view, every cell contained the *entire*
  table's text, repeated.
- **Investigation.** In block mode, a Table region returns a single
  `BlockOCRResult` whose `.bbox` is the whole table and whose `.html` is a full
  `<table>…</table>`. The code was discarding that structure and mapping the one
  big block to every cell.
- **Three approaches that failed:** (1) per-cell OCR — 225 sequential calls to
  `llama-server`, ~19-minute hang; (2) a cell-count cap — skipped large tables
  entirely, leaving cells empty; (3) bbox-overlap mapping — the table block's bbox
  covers every cell, so all cells received the full concatenated text.
- **Decision.** The VLM's `block.html` *already* contains the correct
  `<table><tr><td>` grid. Parse it (stdlib `html.parser`) into a
  `(row, col) → text` map and fill cells by index — **zero** extra model calls.
- **Outcome.** Correct per-cell text; table text no longer leaked into the page
  body text; the UI became responsive again (no blocking call loop).

### 2.3 Robustness review (external "Qwen" review)

- **Problem.** The HTML-parsing fix had three latent gaps.
- **Decision/Outcome.** Three guards added: **colspan padding** (a
  `<th colspan="3">` now pads the row so column indices stay aligned);
  **flat-text fallback** (if the VLM emits `<p>` text instead of a `<table>`, fall
  back to flat text in the first cell with a warning); **bbox tolerance matching**
  (layout and recognition are separate passes that return slightly different
  float bboxes — match the closest within a 20-px tolerance instead of exact key).

### 2.4 Table cells still misclassified → **VLM HTML as single source of truth**

- **Problem.** Even after 2.2–2.3, real documents still placed text in the wrong
  cells (and some cells came out empty).
- **Investigation.** The pipeline was building each table from **two
  independently-derived grids** and joining them by index: Surya's geometric
  `table_pred` (one cell per detected row × column intersection, with its own
  row/column counts) versus the VLM's `<table>` HTML (its own row/column counts).
  When the two disagreed by even one row or column — a title row, a wrapped line,
  a different column count — every subsequent cell shifted.
- **Decision.** Stop joining two grids. Build table cells **directly from the VLM
  HTML** (text is in its correct cell by construction) and **remove `table_pred`
  entirely** — tables are already detected from the layout pass, and no downstream
  consumer used the geometric cell coordinates. Deleted `_serialize_table`,
  `_filter_phantom_cells`, and the index-join helper.
- **Outcome.** Misclassification from the join eliminated; code simplified; the
  `TableRecPredictor`'s VRAM was freed on the 24 GB machine.

### 2.5 Evaluation — paid LLM judge → free deterministic metrics

- **Problem.** "Is the OCR good enough?" had no measurement. An initial benchmark
  used a GPT-4o vision "judge" to score each image — paid, non-deterministic, and
  the wrong tool when exact ground truth exists.
- **Investigation.** The synthetic datasets ship exact ground truth (table grids +
  paragraph text). With ground truth, accuracy can be measured **deterministically
  and for free**; an LLM judge is only justified where no ground truth exists
  (real documents).
- **Decision.** Replace the judge with `evaluate_structure.py` (stdlib only):
  real **CER** (Levenshtein), **table cell accuracy** (positional) and
  **content recall** (order-insensitive), plus **layout signals** (paragraph
  recall, and *paragraph leak* — body text wrongly captured inside a table).
  Reference-free judges for real documents (a local Qwen2.5-VL judge; dual-OCR
  consensus) were considered and **deferred** — Qwen-VL's Khmer judging is itself
  suspect, and a second OCR engine has weak Khmer support.
- **Outcome.** Free, reproducible, exact metrics. `evaluate_judge.py` remains as a
  standalone tool but is no longer on the benchmark path.

### 2.6 Test-environment hardening

- **Problem.** Before trusting any number, the environment had to be fair and
  crash-resistant.
- **Investigation/findings.** (a) 14 of 15 isolated-table images were generated
  *before* a margin fix and had tables touching the image edge — Surya's layout
  model won't classify an edge-to-edge table as a table, so those would fail
  *detection*, not OCR. (b) The full-page document set had only one font. (c) The
  image generators waited for `networkidle` but never verified the *intended*
  Google Font actually rendered — a silent fallback-font risk. (d) Preprocessing
  (`_crop_margins`, deskew, etc.) confounds the OCR-quality signal on pristine
  synthetic inputs.
- **Decisions.** Regenerate both datasets full-sweep (5 fonts × 3 templates each);
  add a **font-load guarantee** (`document.fonts.check()` → hard error, never a
  fallback render); **raw-render bypass** — feed the pristine PNG straight to OCR
  (no preprocessing) to isolate the model's true capability; **crash-safe
  incremental CSV** with `--resume`; **engine-tagged, auto-named output**
  (`benchmark_results_<engine>_<timestamp>.csv` + `Engine` column) so models can
  be compared across runs; a `.gitignore` for sensitive inputs (`sample_data/`,
  `.streamlit/`) and generated outputs.
- **Outcome.** A fair, complete, reproducible harness ready for swapping in other
  OCR models via the engine registry.

### 2.8 Evaluation artifact organization

- **Problem.** Datasets (`synthetic_data/`, `synthetic_documents/`) sat at the repo root, auto-named CSVs scattered there too (`benchmark_results_<engine>_<ts>.csv`), and `analyze`'s positional arg glob could silently mix rows from two different runs. No record existed of *what* a result file covered, *by* which code version, or on *what* dataset.
- **Decision.** Consolidate under a single `eval/` home: datasets move to `eval/datasets/{synthetic_tables,synthetic_documents}/`; each benchmark run gets `eval/runs/<YYYYMMDD_HHMMSS>_<engine>/` containing `results.csv` + `manifest.json` (run_id, engine, correction, git commit + dirty flag, surya/python versions, per-dataset image counts, aggregate metrics) + `summary.txt` (captured analyze output). `analyze_benchmark` defaults to the latest run dir when called with no args. A committed `eval/README.md` documents layout, CLI, manifest schema, metric definitions, and compare-run workflow.
- **Outcome.** Every run is self-describing and citable by `run_id`. The old glob-contamination footgun is gone (one run = one folder). Generators default into the new paths; `.gitignore` swaps old patterns for `eval/datasets/` + `eval/runs/`.

### 2.7 Metric robustness — row-aligned cell accuracy

- **Problem.** The first real benchmark showed `Cell_Accuracy` averaging 0.266,
  which looked like Surya failing.
- **Investigation.** Wherever `Cell_Accuracy ≈ 0`, `Content_Recall` was high on the
  same row, and `Pred_Rows = GT_Rows + 1`: the rendered tables have a merged title
  row the VLM captures but the ground truth (for documents) omits, so a single
  unstripped row shifted everything by one — the *content* was correct, only the
  *positions* were off.
- **Decision.** Align rows between the predicted and ground-truth grids
  (`difflib.SequenceMatcher` over normalized row signatures) before positional
  scoring, so an inserted/dropped row no longer cascades.
- **Outcome.** Average `Cell_Accuracy` rose **0.266 → 0.643**, and the per-font
  ranking by accuracy now agrees with the independent content/CER metrics.

### 2.9 Tesseract baseline engine

- **Problem.** A thesis needs a recognised, off-the-shelf comparison point for the
  Surya-based pipeline. Tesseract (`khm` traineddata) is the standard Khmer OCR
  baseline.
- **Decision.** Add `run_tesseract` (`tesseract_engine.py`) behind the existing
  engine registry, switchable via the `OCR_ENGINE` env var (`surya` default,
  unknown → `run_surya`). It re-packs Tesseract's parallel-list `image_to_data`
  output into the **same 7-key `text_blocks` shape Surya emits**, so the eval
  harness scores it unchanged. `pytesseract` is lazily imported (clear brew-hint
  `ImportError`) and pinned `>=0.3,<0.4`.
- **Caveats (fair to report).** Tesseract yields **no table structure**
  (`tables=[]`), so the Surya-vs-Tesseract comparison is **text-only** — table
  metrics are not applicable to it. It also tends to **insert spaces between Khmer
  clusters**, which inflates its CER; this is a real property of the engine, not a
  measurement artifact, and is reported as-is.

### 2.10 Stage 4 redesign — Qwen demoted to opt-in, deterministic Khmer normalizer

- **Problem.** Stage 4 was both slow and useless: it loaded Qwen2.5-**7B-Instruct**
  (~4GB, slow per-run load on the 24GB Mac) — a *general* LLM never trained for
  Khmer OCR — yet the deterministic layer was a **no-op** (`RULE_BASED_CORRECTIONS`
  empty → only NFC). Qwen fired only on blocks with ≥15% *foreign-script* chars
  (rare on clean Khmer), but `enable_qwen` defaulted **on**, so every fresh run
  risked the load for no benefit.
- **Decision.** (1) **Qwen → opt-in**: `postprocess`/`_correct_page` now default
  `skip_qwen=True`; UI checkbox defaults off (relabelled "experimental, slow");
  CLI `--no-qwen` replaced by `--qwen`; `run_benchmark` gained `--qwen`. The
  deterministic layer always runs. (2) New **`khmer_normalize.py`** — a 100%-local
  deterministic normalizer: NFC + strip noise format chars (ZWSP/BOM/soft-hyphen;
  ZWNJ/ZWJ preserved) + collapse duplicate combining marks + whitespace tidy
  (**Tier A**), plus an opt-in canonical cluster reorder (**Tier B**).
- **Validation (variance-free A/B on fixed OCR output, 33 images).** Comparing
  `CER(GT, raw)` vs `CER(GT, normalize)` on the saved prediction dumps (so OCR
  run-to-run variance can't confound it — table metrics drift ~6pts between two
  live OCR runs, confirming the need for fixed-output comparison):

  | dataset | n | raw | Tier A | + reorder |
  |---|---|---|---|---|
  | synthetic_tables | 15 | 0.1650 | 0.1650 | 0.1644 |
  | synthetic_documents | 15 | 0.4498 | **0.4353** | 0.4353 |
  | real | 3 | 0.5030 | 0.5030 | 0.5031 |
  | ALL | 33 | 0.3252 | **0.3186** | 0.3183 |

- **Outcome.** **Tier A ships on by default** — a real, safe win (synthetic_documents
  CER −3.2% relative, neutral elsewhere, never hurts). **Tier B reorder is below the
  noise floor** (helps tables 0.0006, ties docs, +0.0001 on real → fails the
  pre-agreed "reduces-or-ties on both" gate) because Surya already emits canonical
  Khmer; it is kept **behind a default-off `reorder=` flag**, validated-neutral and
  reserved for legacy/scanned docs with mis-ordered Khmer. Honest thesis takeaway:
  a general LLM did not help; deterministic Unicode normalization does, modestly.

### 2.11 Productionization polish (single-user desktop)

- **llama-server lifecycle.** Surya keeps a resident `llama-server` (Metal,
  `KEEP_ALIVE=true`); a crash/unclean exit can orphan it (leaked unified memory +
  port). Added `stop-metal-macos.sh` (graceful then forced kill, reports PIDs) and a
  `backend_status.py` helper (`llama_server_running()` via `pgrep`) surfaced as a
  sidebar 🟢/⚪ indicator. **No** auto-kill-on-exit in CLI/benchmark — a blanket kill
  would also stop a server a concurrently-running app is using; explicit teardown only.
- **Memory guard.** Added a soft `st.warning` in `app.py` when a job exceeds
  `_MEMORY_WARN_PAGES` (scaled by DPI). The definitive limit is **measured** via a
  stress test on a large scanned PDF (method + result in `docs/OPERATIONS.md`);
  the constant is provisional until that run.
- **Reproducibility freeze.** Synthetic generators previously pulled fonts live from
  `fonts.googleapis.com` (non-deterministic, network-dependent). Vendored the 5 OFL
  Khmer fonts under `fonts/` (+ `MANIFEST.txt` with sha256 + OFL-1.1 license texts)
  and switched both generators to embed them as base64 `@font-face` via a shared
  `fonts.py` helper — datasets now regenerate **byte-for-byte offline**. Verified one
  doc + one table render correctly with no network. Fonts are OFL-1.1 → redistributable.
- **Docker — declined (future work).** Deliberately not containerized: macOS containers
  can't reach Metal and MLX doesn't run on Linux, so a container would drop to CPU.
  Reconsider only for a Linux/CUDA multi-user server pivot. (See `docs/OPERATIONS.md`.)

### 2.12 Table de-fragmentation — geometric stitcher (Path A): a useful negative result

> **⚠ Caveat (see §2.25):** the fragmentation below was measured on **raw** (un-preprocessed) images. With
> the pipeline's `preprocess()` — which the product always runs — Surya's page-2 layout collapses from 8
> boxes to **1**, and plain Surya becomes the best engine. This whole arc addresses a problem preprocessing
> largely solves.

- **Problem.** On dense real pages Surya's *layout* model shatters one table into many
  regions (real GDDE page 2 → a 2 row-band × 4 col-group grid of **8 Table boxes**);
  recognition then OCRs each fragment separately and serializes content column-wise,
  destroying row↔value links.
- **Approach (Path A).** New `table_stitch.merge_table_regions` (transitive 2-D adjacency
  clustering: connected components via inflated-intersection, union each) merges fragments
  into master boxes **before** `rec_pred`, hooked into `surya._process_page` behind
  `_STITCH_TABLES` / `KHMER_STITCH_TABLES`. Verified it merges page 2's 8 regions → 1.
- **A/B result (raw render, 33 imgs, stitch OFF vs ON).** Isolated to page 2 (the only
  fragmented page; p1/p3 and all synthetics were byte-identical no-ops, confirming the
  delta is the stitcher, not engine drift):

  | page 2 | Tables_Found | Cell_Acc | Content_Recall | Document_CER |
  |---|---|---|---|---|
  | stitch OFF | 8 | 0.024 | **0.758** | 0.670 |
  | stitch ON | 1 | 0.016 | **0.156** | 0.893 |

- **Finding (the value).** Stitching **fixes detection** (8→1) but the VLM then **degrades
  badly on the large dense merged crop** — Content_Recall collapses 0.76→0.16. Fragmented,
  the VLM reads each narrow column-strip and recovers ~76% of cell text (just mis-structured);
  given the whole dense table at once it recovers ~16% (almost certainly internal downscaling
  losing small Khmer glyphs/digits). **The bottleneck is not only detection — it is VLM
  recognition on large dense crops.**
- **Decision.** Gate failed (no Cell_Accuracy gain; Recall/CER regressed) → `_STITCH_TABLES`
  shipped **default OFF**; code + flag retained. This negative result **decomposes the problem**
  and motivates the next experiment: merge fragments into **full-width row-band strips** (short
  crops that preserve whole rows without overwhelming the VLM), or escalate to Hybrid B
  (SLANet structure + Surya cell recognition). Runs: `*_surya_stitchOFF` / `*_surya_stitchON`.

### 2.13 Row-band stitch variant — best stitcher, still not decisive

- **Idea.** Instead of one giant master box (§2.12), merge fragments into **full-width
  row-band strips** (`merge_table_rowbands`: cluster by Y-band, X ignored) — short crops
  that keep whole rows intact at a VLM-readable scale. Real page 2: 8 regions → **2 strips**.
- **A/B on the fragmented page (real p2), all three variants:**

  | variant | Tables_Found | Cell_Acc | Content_Recall | Document_CER |
  |---|---|---|---|---|
  | OFF (fragmented) | 8 | 0.024 | **0.758** | 0.670 |
  | master (one box, §2.12) | 1 | 0.016 | 0.156 | 0.893 |
  | **row-band (2 strips)** | 2 | **0.036** | 0.348 | 0.788 |

- **Finding.** Row-band **beats master on every metric** (confirms "smaller crops help the
  VLM") and **lifts the structural metric** Cell_Accuracy 0.024→0.036 (+50% rel) — but still
  **loses Content_Recall** (0.758→0.348): the VLM reads wide strips less completely than
  narrow column-fragments. So there is a real **crop-size ↔ VLM-recognition tradeoff**, and
  **no geometric stitch variant is decisive**. The root limit is **VLM table recognition on
  wide dense Khmer tables**, not just layout fragmentation.
- **Why post-OCR cell reassembly won't rescue it cheaply:** the VLM-HTML cells carry **no
  per-cell bbox** (`"bbox": []`), so we can't geometrically re-place fragmented cells into a
  global grid without a structure model that emits cell coordinates.
- **Decision.** Keep stitching **default OFF** (both modes retained behind
  `KHMER_STITCH_TABLES` / `KHMER_STITCH_MODE`). Row-band is the documented best-effort
  geometric fix. **Escalate to Hybrid B** — a structure model (e.g. SLANet) for the cell grid
  **with coordinates** + Surya recognition on small cell/region crops (small crops = high
  recall, like fragments, *plus* correct structure). Runs: `rb_*_OFF` / `rb_*_ROWBAND`.

### 2.14 Hybrid B structure prototype — SLANet go/no-go = **GO**

- **Goal.** Before any integration, verify a structure model produces a *unified* grid with
  *cell coordinates* on the dense Khmer table (the thing Surya's layout fragments and whose
  VLM-HTML cells lack bboxes).
- **Setup.** `rapid_table` 3.0.2 (SLANETPLUS, **7.4 MB ONNX**, onnxruntime CPU — no Paddle),
  installed **ephemerally** (`uv pip install`, not in pyproject). Ran on real page-2 table
  region with `use_ocr=False` (structure only).
- **Result (strong GO).** One coherent grid — **no fragmentation** — **27 rows × 9 cols vs
  GT 28×9** (off by one), **188 cells each with quad coordinates** (`cell_bboxes`) + logical
  spans (`logic_points`, incl. merged-header spans like `[0,4,3,3]`), cells tiling the full
  region; ~0.07 s inference. Visual overlay confirmed cells map onto the real №/name/unit/
  price/% columns.
- **Why this matters.** It supplies exactly what no stitch variant could: correct structure
  **with per-cell coordinates**. Hybrid B can crop each cell box → OCR with **Surya** (small
  crops = high recall, like the fragments) → place text by `logic_points` → emit our standard
  `cells[]` table dict. Decouples structure (SLANet) from Khmer recognition (Surya).
- **Next.** Build Hybrid B: `uv add` rapid_table (pinned) + new engine wrapper +
  per-cell Surya OCR; A/B vs Surya baseline on the eval harness.

### 2.15 Hybrid B (SLANet + per-cell Surya) — built, but per-cell recognition fails

- **Built** `slanet_structure.py` (SLANet wrapper) + `hybrid_engine.py` (`run_hybrid`,
  `OCR_ENGINE=hybrid`): reuse Surya for page text + table *detection*, take the master box of
  the fragmented Table regions, run SLANet for the grid + cell coords, then OCR **each cell**
  via Surya block-mode recognition (one `LayoutBox` per cell). 6 offline tests; shape verified.
- **A/B on real (raw render):** structure is fixed but recognition collapses.

  | page | Surya Acc/Recall/DocCER | Hybrid Acc/Recall/DocCER |
  |---|---|---|
  | p1 | 0.134 / 0.529 / 0.618 | 0.125 / **0.110** / 0.712 |
  | p2 (fragmented) | 0.024 / **0.758** / 0.670 | 0.028 (Found 8→**1**) / **0.037** / 0.754 |
  | p3 (no table) | – / – / 0.220 | – / – / **1.894** |

- **Finding (negative, but clear).** SLANet's structure works (p2 `Tables_Found` 8→1, grid
  ~27×9) and `Cell_Accuracy` is ~flat, but **`Content_Recall` collapses** (p2 0.758→0.037)
  and it's **~258 s/page (~4.3 min)**. Two causes: (1) Surya's recognizer is a **VLM built for
  text lines/blocks**, and on **tiny single-cell crops it hallucinates** (emits foreign scripts —
  Arabic/Burmese/Sinhala — on small/ambiguous inputs); (2) SLANet sometimes **over-merges** cells
  (a cell spanning 5 rows swallows a whole column, e.g. `"360 350 350"`). Net: **worse than the
  Surya baseline and far slower** → decision gate failed.
- **Decision.** `hybrid` stays **registered but not default** (opt-in `OCR_ENGINE=hybrid`) as a
  documented experiment. Root insight stands (from 2.13): **the limit is recognition on small,
  isolated Khmer table cells**, not structure. SLANet *solved* structure; pairing it with
  per-cell VLM OCR doesn't work. Candidate next refinement: **row-strip recognition** — OCR each
  full-width row as one text line (what the VLM is good at; ~27 calls not 188) and split into
  columns by SLANet's column x-boundaries. Runs: `hy_*_surya` / `hy_*_hybrid`.

### 2.16 Preprocessing A/B on degraded input — modest, consistent, non-harmful

- **Setup.** The OpenCV preprocessing stack (deskew/stamp/sharpen/contrast/table-bg) had never
  been tested on degraded input (`REPORT.md §6`). No real scan exists, so a **proxy**: synthetically
  degrade the GT'd born-digital 09.06.26 render (`generate_degraded.py`: rotation 2.5° > deskew
  threshold, blur, seeded noise, contrast cut) and A/B with the new `run_benchmark --preprocess`
  flag against the **existing** ground truth.
- **Result (Document_CER, lower = better):**

  | page | clean (ceiling) | degraded, preprocess OFF | degraded, preprocess ON |
  |---|---|---|---|
  | p1 | 0.618 | 0.714 | **0.691** |
  | p2 | 0.670 | 0.685 | **0.653** |
  | p3 | 0.220 | 0.847 | **0.833** |
  | **avg** | **0.503** | **0.749** | **0.726** |

- **Finding.** Degradation clearly hurts OCR (0.503 → 0.749). Preprocessing recovers a **small but
  consistent** slice — **ON beats OFF on all three pages** (avg −3% relative) — but does **not**
  restore toward the clean ceiling. So the stack is a **modest, directionally-robust, non-harmful**
  improvement on scan-like input (worth keeping on for scans), not a silver bullet. Consistency
  across all pages mitigates the OCR non-determinism concern.
- **Caveats.** Synthetic degradation **≠ real scan artifacts** — this is a controlled proxy, not
  field evidence. `Text_CER` (~0.95) is fragmentation-dominated and uninformative here; `Document_CER`
  is the signal. **Real-scan A/B remains future work.** Runs: `prep_*_clean` / `_degOFF` / `_degON`.

### 2.17 Row-strip recognition — the fragmentation arc's first win

- **Idea (the open lead from 2.15).** Keep SLANet for structure, but recognise each row as **one
  full-width strip** instead of per-cell — a strip is a natural line, which is what Surya's VLM is
  built for, and it's ~27 calls/page not 188. New `KHMER_HYBRID_MODE` (`hybrid_engine.py`):
  `"rowband"` (now default) vs `"cell"` (2.15, kept for comparison).
- **Phase-0 probe** (`scripts/probe_rowstrip_recognition.py`). Key finding: a strip sent with
  `label="Table"` makes Surya emit a one-row `<table><tr><td>…` we can parse with the existing
  pure-Surya `_parse_html_table` — **Surya does the column splitting itself** (9 `<td>` = SLANet's
  9 cols), no x-boundary math needed. Clean data rows read correct Khmer; the "Burmese
  hallucination" first seen was an artifact of probing the messy multi-line **header** band. ~40% of
  isolated strips come back **blank** (recall loss, not corruption).
- **Strip geometry** (`_row_bands`): x is **always** the full crop width (a short/missing cell must
  not narrow the strip and rob the VLM of column context); y gets `_ROW_STRIP_Y_PAD_PX`=8 padding
  (keep ascenders/descenders + grid lines the VLM uses to emit `<td>`s).
- **A/B on real (raw render), the fragmented p2 is the point:**

  | page | Surya Acc/Recall/TblCER | **rowband** Acc/Recall/TblCER | cell Acc/Recall/TblCER |
  |---|---|---|---|
  | p1 (clean table) | 0.134 / 0.529 / 0.274 | 0.231 / 0.390 / 0.455 | 0.120 / 0.105 / 0.931 |
  | p2 (fragmented) | 0.024 (Found **8**/1) / 0.758 / 0.657 | **0.393** (Found **1**/1) / 0.525 / **0.424** | 0.024 / 0.025 / 1.339 |
  | p3 (no table) | DocCER 0.220 | DocCER 0.526 | DocCER 1.974 |

- **Finding (positive, qualified GO).** On the fragmented table, **rowband is the first method in
  the whole arc to fix detection (8→1) AND recover row↔value accuracy** — `Cell_Accuracy`
  0.024→**0.393** (~16× over both Surya and cell) and `Table_CER` 0.657→**0.424** — by giving the
  VLM a natural full-width line and letting it column-split. It **strictly dominates cell mode** on
  every metric (cell's recall stays collapsed at 0.025, confirming 2.15) and is faster (~3.3 min/page
  vs cell's ~4.3). The trade is **recall** (0.758→0.525, the blank strips) and it still **hurts
  pages without a real table** (p3 phantom-table region inflates DocCER 0.220→0.526).
- **Decision.** Default `KHMER_HYBRID_MODE=rowband`; `cell` kept opt-in for comparison. `hybrid`
  stays opt-in vs Surya for **production** (the recall trade + phantom-table behaviour on non-table
  pages aren't fixed yet) — but for **table-heavy** GDDE docs rowband is the recommended engine and
  **closes the fragmentation arc**: structure is solvable (SLANet) *and* recognition of dense tables
  is now usable (rowband), where geometric stitching (2.12–2.13) and per-cell (2.15) both failed.
  Next leads if pursued: recover blank rows (retry blanks with extra context) and suppress
  hybrid processing on no-table pages. Runs: `*_ab_surya` / `_ab_hybrid_rowband` / `_ab_hybrid_cell`.

### 2.18 Row-strip recall fix — blank-strip retry (the recall half of 2.17's trade)

- **Two leads from 2.17:** (a) ~40% of strips returned **blank** (recall 0.758→0.525); (b) the
  hybrid **hurts no-table pages** (p3 DocCER 0.220→0.526, a phantom table region).
- **Phase-0 probe** (on the known-blank p2 rows 15/20): re-running the *same* pad-8 strip does **not**
  recover them (blanks are deterministic, not OCR jitter); pad-30 doesn't either; **pad-60 recovers
  both** as a single, correctly-columned row. So the fix is a **second recognition pass over only the
  blank bands with a much taller crop** (`_ROW_STRIP_RETRY_Y_PAD_PX=60`), keeping the row with the
  most non-empty cells (`_best_row`, in case the taller crop grabs a neighbour sliver).
- **A/B after the fix (real, raw render):**

  | page | Surya Acc/Recall/TblCER/DocCER | rowband 2.17 | **rowband + retry (2.18)** |
  |---|---|---|---|
  | p1 | 0.134 / 0.529 / 0.274 / 0.618 | 0.231 / 0.390 / 0.455 / 0.707 | 0.222 / **0.500** / **0.363** / **0.662** |
  | p2 | 0.024 / 0.758 / 0.657 / 0.670 | 0.393 / 0.525 / 0.424 / 0.686 | **0.425** / **0.623** / **0.288** / **0.612** |
  | p3 (no table) | DocCER 0.220 | DocCER 0.526 | DocCER 0.583 |

- **Finding.** The retry recovers a real slice of recall — p2 0.525→**0.623** (closing ~⅓ of the gap
  to Surya's 0.758) and p1 0.390→**0.500** — while **accuracy and CER also improve** (p2 Acc
  0.393→0.425, Table_CER 0.424→0.288, DocCER 0.686→0.612). **Rowband now beats pure Surya on every
  p2 metric, DocCER included.** The residual recall gap is genuinely-illegible rows (a recogniser
  limit, not a strip-sizing one).
- **Phantom suppression — dropped, with evidence.** Probing p3's phantom region: SLANet returns a
  *full* 26×9 / 123-cell grid (not degenerate), and after the retry the phantom **fills like a real
  table** (0.85 of rows ≥2 cells, ~5.4 cells/row, median 6 cols) vs p2's real (1.0, 8.8, 9). There is
  **no structural or fill-rate threshold that suppresses the phantom without risking real sparse
  tables**, and we have only one no-table page to tune against — so adding a heuristic would overfit.
  Left as a characterised limitation; the right fix is upstream table-**detection** gating or more
  labelled no-table pages. p3 stays slightly worse (0.583) because the retry fills more phantom rows.
- **Decision.** Blank-retry shipped (default on in `rowband`). `hybrid` remains opt-in vs Surya for
  production **only** because of the no-table-page behaviour; on table pages rowband is now clearly
  best. Run: `*_recallfix_rowband`.
- **Correction (added §2.19).** The "no-table page" / "phantom" framing above was **wrong**: p3 is a
  *real* continuation table whose content the GT had mislabelled as `paragraphs` (`tables: []`), so
  `evaluate_table` had no grid to score and the page looked table-less. The p3 DocCER gap was rowband
  re-formatting a *real* table, not inventing a phantom. GT fixed in §2.19; the no-table-page concern
  is therefore overstated (we still lack a true no-table page in the set).

### 2.19 Multi-page table stitching — one report → one table

- **Why.** The real ARDB price reports are **one continuous 9-col table split across page images**
  (with embedded section-divider rows); the per-page engines emitted a table per page, forcing the
  analyst to re-stitch in Excel. Added `table_merge_pages.py` (`merge_document_tables`): join
  consecutive tables that share a column count (±1), drop the repeated header at each page break, and
  start a new logical table when columns change. Wired as `stitch_pages` into Stage-5 `export.py`
  (default **on** in `app.py`/`pipeline.py`; per-table CSV + a `document_tables` block in the JSON).
- **GT integrity.** `scripts/draft_document_gt.py` restructures the existing per-page GT (incl. p3's
  mislabelled paragraphs) into a document-level grid (`*_document_gt.json`) for human verification —
  fixing the §2.18 issue. Eval: `scripts/eval_document.py` (whole doc → stitch → sanity checks +
  `evaluate_table` vs the document GT).
- **Result (09.06.26, 3 pages, `eval_document.py`), GT verified (75×9):**

  | engine | per-page → logical tables | pred shape | Cell_Acc | Recall | Table_CER | dup hdrs |
  |---|---|---|---|---|---|---|
  | **hybrid rowband** | 3 → **1** (pages [0,1,2]) | 101×10 | 0.139 | 0.576 | **0.337** | 0 |
  | surya | 10 → 3 (p2's 8 frags stay 4-col) | 146×10 | **0.170** | **0.722** | 0.348 | 0 |

- **Finding (two parts).** (1) **Stitching works with the hybrid rowband engine** — consistent 9-col
  pages → all 3 collapse into one table, headers de-duplicated; **Surya can't join** (per-page
  fragmentation → inconsistent column counts), so stitching and the structure-aware engine go
  together. (2) **At the *whole-document* level hybrid does not beat Surya** — which does *not*
  contradict §2.18: that win was specific to the dense fragmented **p2**, whereas the doc GT is
  dominated by the cleaner p1/p3 where Surya is already strong, so the average swings back. Honest
  read: **hybrid is the engine for dense tables and the only one that enables clean stitching; Surya
  stays strong on mixed/clean content.**
- **Spurious 10th column — found + fixed (rowband), but metric-neutral.** Diagnostic: Surya's
  row-strip HTML sometimes emits an extra **trailing empty `<td>`**, so rowband tables on p2/p3 became
  10-col (col 9 empty in every row); p1 was clean. Fix: clamp the rowband grid to **SLANet's column
  count** in `_ocr_rowbands(..., n_cols)` — principled, not a content heuristic (content-based
  trimming would wrongly collapse a sparsely-OCR'd page). After the fix the stitched table is **9×**
  (matches GT): `Cell_Acc 0.139→0.145, Recall 0.576→0.561, Table_CER 0.337→0.350` — i.e. **within
  OCR run-to-run noise**. So it's an **output-cleanliness win** (no junk column in the analyst CSV),
  *not* a scored-accuracy win: the row-aligned scorer was already treating the empty column as
  empty-vs-empty.
- **Row over-production — diagnosed + the safe slice fixed.** Dumping the 101-row merge showed the
  ~26 extra rows are: **~15 fully-blank rows** (SLANet over-segments into empty bands — the p1
  meat/poultry page is worst, 37 rows / 12 blank), **~8 near-duplicate rows** (SLANet splits one
  visual row into two bands, OCR'd twice with minor diffs), and **~6 hallucinated rows** (rowband
  recognition failing on divider/header/merged regions). Fixed the clean, safe slice: **drop
  fully-empty rows** in `_combine` (also better analyst output — no blank CSV rows). Result: rows
  **101→85**, `Cell_Accuracy 0.145→0.181`, `Recall 0.561→0.590`, `Table_CER 0.350→0.331` — a real
  lift (hybrid now even edges Surya's doc-level Acc 0.170 while being the only stitching-capable
  engine). The residual gap (85 vs 75) is near-dup splits + hallucinations — **OCR-quality noise, not
  chased further** (fuzzy de-dup would risk dropping real rows; over-tuning one doc isn't worth it
  per the project's breadth-over-depth focus). Honest takeaway: rowband stitching yields a **usable,
  review-ready draft** (the project's stated workflow — analysts review/correct), not a perfect
  extraction. GT-free stitch structure checks all pass. Modules: `table_merge_pages.py`,
  `scripts/draft_document_gt.py`, `scripts/eval_document.py`.

### 2.20 Hybrid on a genuine no-table page — safe (resolves the §2.18 worry)

- **Why.** §2.18/§6 feared the hybrid fabricates a table on text-only pages, but that was tested on a
  *mislabelled* page (p3 is really a table, §2.19). Re-tested on a **genuine text page** —
  `CambodiaBudgetExecutioninApr-2024.pdf` p2 (1,527-char born-digital text layer as GT),
  `scripts/eval_notable_page.py`.
- **Result.** Both engines **identical**: `Tables_Found=0` (no phantom), `table_cells=0`,
  `Document_CER=0.312`. Hybrid reuses Surya for text + table *detection* and only rebuilds tables
  **when Surya detects them**; with zero detected, hybrid's output *is* Surya's
  (`run_hybrid`: `if not boxes: pages.append(page)`).
- **Finding.** **Hybrid is safe on real text pages** — no phantom, no garbling. The earlier p3
  "regression" was entirely the GT mislabel, not the engine. Residual phantom risk reduces to Surya's
  *detection* false-positive rate (zero here). So the reason `hybrid` stays opt-in vs Surya is no
  longer safety — it's **speed** (~3.3 min/page vs ~74 s) and Surya being competitive except on dense
  fragmented tables. Module: `scripts/eval_notable_page.py`.

### 2.21 Off-the-shelf recognizer A/B — Surya wins; an open VLM does not

- **Why.** Before deciding whether to *fine-tune* a recognizer (mentor idea #1), establish how well
  off-the-shelf engines *recognize* Khmer and **where Surya fails** — don't fine-tune blind.
- **Metric (recognition-only, new).** Per-page **recognition CER** on *single-source* pooled text:
  `evaluate_recognition` / `pool_gt_recognition_text` (`evaluate_structure.py`). It is
  **placement-agnostic** — pools all recognized text on each side and compares characters, scoring
  *reading*, not *layout*. Deliberately distinct from the §2.18 `evaluate_table` ruler (row-aligned,
  structure-aware); the two answer different questions, which is why the hybrid row below reads the way
  it does. Single-source pooling (table grid if present, else paragraphs+footer) avoids the
  paragraph/table double-count baked into `pool_gt_text`.
- **Eval set.** 3 ARDB `09.06.26` table pages + 1 genuine text page (CambodiaBudget p2). Local engines
  swap via `OCR_ENGINE`; an external model is scored from a predictions JSON
  (`scripts/eval_recognizers.py --predictions`, same metric). 4-way table via
  `scripts/compare_recognizers.py`.
- **Results (recognition CER, lower = better):**

  | Page | Surya | Tesseract-khm | Qwen2.5-VL-7B (4-bit MLX) | Hybrid (rowband) |
  |---|---|---|---|---|
  | ARDB p1 (table) | **0.369** | 0.710 | 2.363 | 0.414 |
  | ARDB p2 (dense table) | 0.667 | 0.797 | 1.978 | **0.288** |
  | ARDB p3 (table) | **0.220** | 0.733 | 2.748 | 0.547 |
  | CambodiaBudget (text) | **0.009** | 0.065 | 1.993 | **0.009** |
  | **mean** | **0.316** | 0.576 | 2.271 | 0.315 |

- **Findings.**
  - **Surya wins the baseline** (mean 0.316); **Tesseract-khm is far behind on tables** (0.71–0.80),
    competitive only on prose.
  - **Hybrid ties Surya overall (0.315) but is a *targeted* tool:** it nearly halves the error on the
    **dense fragmented p2 (0.667 → 0.288)** while *hurting* the cleaner p1/p3 (rowband re-segmentation
    adds noise where Surya already reads well). Consistent with §2.17–2.18 — hybrid is for the
    dense-fragmentation case, not a universal default. (Note the contrast with the §2.18 *structure*
    ruler: here we measure characters read, not cell placement.)
  - **An off-the-shelf VLM did NOT beat Surya.** Qwen2.5-VL-7B (4-bit, local MLX) scored CER **> 1 on
    every page** — i.e. it *failed to produce usable output*, not "2.3× worse recognition." CER > 1
    means the output is both wrong **and** longer than the truth (garble + repetition bloat).
- **Qwen failure detail (decoding fragility).** The 4-bit model collapsed into repetition loops and
  needed deliberate decoding tuning even to reach the above: a "use Markdown tables" prompt → empty-grid
  loop; plain-text prompt → word-repeat loop; `repetition_penalty=1.3` was the sweet spot (broke the
  prose loop; dense tables still loop on near-identical numbers); 1.4 made it worse (broke the prose
  page too). So the result is **bounded to the 4-bit MLX build** (8-bit untested by choice) and says
  "this off-the-shelf *local* VLM is not turnkey for dense Khmer tables," not "Qwen2.5-VL can't do
  Khmer." Run isolated from the project env (`uv run --no-project --with mlx-vlm`) because mlx-vlm needs
  `transformers>=5.1` but Surya pins `<5.0`.
- **Data-quality finding (legacy Khmer fonts).** The CambodiaBudget PDF's born-digital text layer uses
  a **legacy Khmer font** (glyphs mapped onto Latin/extended codepoints: `ƒ Ǝ ſ ȥ`) — PyMuPDF returns
  those raw codepoints, so it is **unusable as GT** (the page renders as Khmer but extracts as
  mojibake). GT was rebuilt by OCR-draft + manual correction. **This retroactively voids §2.20's
  `Document_CER = 0.312`** (scored against that corrupt text) — treat that number as meaningless; the
  §2.20 `Tables_Found = 0` phantom-safety conclusion is GT-independent and still stands.
- **Models flagged as likely silent failures for Khmer (recorded for rigor, not individually tested).**
  GOT-OCR2.0, Florence-2, PaddleOCR/MinerU, Donut/Nougat — English/CJK-biased encoders/tokenizers that
  mangle the Khmer script (stacked subscripts/coeng).
- **Axis note.** This A/B is the **recognition** axis (reading text). The separate **layout/structure**
  axis (DocLayout-YOLO, PP-Structure, more Paddle vs Surya-layout + SLANet) targets the *fragmentation*
  problem and is the next thread. Modules: `scripts/eval_recognizers.py`, `scripts/mlx_recognizer.py`,
  `scripts/colab_recognizer.ipynb`, `scripts/compare_recognizers.py`.

### 2.22 Analyst UI overhaul — "hide the ML, show the data" (the deliverable)

- **Why.** The pipeline produced good output, but `app.py` read like an ML control panel. The actual
  deliverable is a tool non-technical GDDE analysts can use to review and correct extractions, so the
  Streamlit UI was reworked around that.
- **Editable tables (the core).** Read-only `st.dataframe` → `st.data_editor` on the **final export
  tables** (the stitched document-level tables when stitching is on — *what-you-edit-is-what-you-
  download*). ALL rows editable (including the real Khmer header row), neutral "Col N" column labels,
  in-cell edits + add/delete rows, and a per-table "↺ Reset to original" button. Edits flow into the
  CSV / Excel / JSON / zip downloads.
- **Excel export.** New `tables_to_xlsx` (openpyxl; one worksheet per table, sanitized sheet names) +
  `grid_to_csv` refactored out of `_table_to_csv` — both in `export.py`, TDD (~360 tests). Government
  analysts live in Excel, so `.xlsx` is a first-class deliverable.
- **Layout.** Sidebar split into **Primary** (stitch, numerals) vs a collapsed **⚙️ Advanced Engine
  Settings** (DPI, preprocessing, overlay, etc.); **side-by-side review** (page image left, editable
  tables right); OCR text / correction diff / stage timings demoted to a details expander.
- **Guardrails.** >15-page "large document" warning; a prominent error (not a green "success") when 0
  tables are detected; plain-language progress labels; backend-status caption reworded (the resident
  `llama-server` spawns lazily on the first run — not an error before then).
- **Design notes.** Editing the stitched (document-level) table means on multi-page docs the right-hand
  editor spans pages while the left image paginates (cross-reference by flipping pages; 1:1 for
  single-page docs). Engine selection stays env-only (`OCR_ENGINE`), deliberately not surfaced in the UI.
- Modules: `app.py`, `export.py` (`grid_to_csv`, `tables_to_xlsx`), `tests/test_export.py`. Merged to
  `main` (`15ebee5`).

### 2.23 Layout-detector A/B (Thread B) — gate-first probe = **GO** for DocLayout-YOLO

> **⚠ Caveat (see §2.25):** this probe (and §2.24) ran on **raw** images. With preprocessing, Surya no
> longer fragments the table, so the problem DocLayout-YOLO "fixed" is mostly moot under production
> conditions — and preprocessed Surya beats both hybrid variants.

- **Why.** The central finding (§2.12) is that the bottleneck is table **structure/fragmentation**, not
  recognition (the recognition axis closed in §2.21 — nothing turnkey beats Surya). Surya's *layout*
  model fragments one dense table into multiple `Table` regions. Our structure model SLANet (`rapid_table`)
  *is already* PaddleOCR's table model, so the genuinely open lever is the **layout/region detector** that
  produces the table box. Question: does an alternative layout detector see the dense table as **one**
  region where Surya fragments it?
- **Gate-first probe** (`scripts/probe_layout_detectors.py`, standalone — no `src/` changes, no engine
  wire-in, no end-to-end re-score yet). On the known fragmented page (real ARDB market-price PDF, p2,
  §2.12), counts table regions per detector + a `covers_table_as_one` coverage ratio (largest box /
  union of all table boxes) + saves visual overlays to `eval/runs/<ts>_layout_probe/`.
- **Dependency win.** `rapid_layout` (RapidAI, same ONNX family as our `rapid_table`) resolved cleanly
  (`uv add "rapid-layout>=1.2.1,<2.0"`, zero torch/surya/transformers churn, **no PaddlePaddle**) and
  bundles ONNX ports of *both* candidates: `doclayout_docstructbench` (= DocLayout-YOLO, the
  `juliozhao/DocLayout-YOLO-DocStructBench` weights) and `pp_doc_layoutv2/v3` (PP-DocLayout). No isolated
  `--no-project` PyTorch path needed.
- **Result (decisive):**

  | detector | n_table_regions | covers_table_as_one | notes |
  |---|---|---|---|
  | surya | **8** | False | largest/union area ratio 0.27 |
  | **doclayout_yolo** | **1** | **True** | ratio 1.00, no tuning; IoU vs Surya union 0.59 |
  | pp_doclayout | 0 | n/a | below default conf 0.5 (table scored 0.34); at conf 0.1 v3→1 box but v2→4 (threshold-sensitive, inconclusive) |

  Overlays confirm visually: Surya carves the table into column-group boxes (labels excluded);
  DocLayout-YOLO wraps the whole data table in one box.
- **Decision = GO** for DocLayout-YOLO (via `rapid_layout`). Next (separate plan): wire it in as a layout
  source at the `surya.py` seam (~L211-228, where the existing stitcher rewrites `layout_result.bboxes`
  before recognition) or as a new `OCR_ENGINE`, then re-score end-to-end with the existing
  `evaluate_table` metrics (Cell_Accuracy / Recall / Table_CER) on the document GT. PP-DocLayout dropped
  (threshold-sensitive, not decisive).
- Modules: `scripts/probe_layout_detectors.py` (new), `scripts/README.md`, `pyproject.toml` +
  `uv.lock` (added `rapid-layout`). ~360 tests still pass; nothing in `src/` changed.
- **(Superseded by §2.24:** the gate GO held only for *detection*; end-to-end it lost — see below.)

### 2.24 Layout-detector wire-in + end-to-end A/B — **NO-GO** (detection win ≠ extraction win)

- **Why.** §2.23's gate proved DocLayout-YOLO *detects* the table as 1 clean box. But detection is not the
  deliverable — better final tables are. This is the decisive end-to-end test.
- **Wire-in (kept, opt-in).** New `src/khmer_pipeline/layout_detect.py` (`detect_table_boxes`, isolated
  `rapid_layout` wrapper mirroring `slanet_structure.py`); `hybrid_engine.py` gains a
  `KHMER_LAYOUT_DETECTOR` env knob (`surya` (default) / `doclayout`) that swaps the table-region source —
  `doclayout` feeds DocLayout-YOLO's box straight to the *unchanged* SLANet + row-strip pipeline (no
  `merge_table_regions`). Default `surya` preserves prior behavior exactly. TDD: **370 tests pass**.
- **A/B (3-way, verified 75×9 document GT, `scripts/eval_document.py`):**

  | engine | pred dims | Cell_Accuracy | Cell_Content_Recall | Table_CER |
  |---|---|---|---|---|
  | surya | 145×10 | 0.170 | **0.722** | 0.348 |
  | **hybrid (surya-layout, rowband)** — current best | 84×9 | **0.181** | 0.566 | **0.341** |
  | hybrid (doclayout) | 118×**8** | 0.080 | 0.542 | 0.560 |

- **Result = NO-GO.** DocLayout-YOLO end-to-end is **less than half** the Cell_Accuracy of the current
  hybrid (0.080 vs 0.181) and worse Table_CER (0.560 vs 0.341), and yields **8 columns, not 9**.
- **Root cause (visually confirmed).** DocLayout-YOLO's `table` box covers only the **numeric grid** — it
  *clips off the two leftmost columns* (Khmer item-name + unit), classing them as plain text. So its tidy
  "1 box, coverage 1.00" gate result masked a semantic amputation: the most matchable column (item names)
  is dropped → wrong column count, low accuracy. Surya's *fragmented* boxes, run through `merge_table_regions`,
  actually preserve the full 9-column table better. (Verify with `scripts/visualize_layout.py`, which
  overlays both detectors' boxes per page; or flip `KHMER_LAYOUT_DETECTOR=doclayout` in the app.)
- **Lesson (for REPORT).** Echoes §2.12: a better table *bounding box* does not help if what it encloses is
  wrong. Detection-only metrics (box count, coverage) can be actively misleading without an end-to-end
  score. **Current hybrid (Surya-layout + rowband) remains the best engine.** Not chased: padding the
  DocLayout box leftward to recover the label columns (breadth over depth — the gap is large and the box
  semantics are the detector's, not a tuning artifact).
- **Decision.** Keep the wire-in opt-in + this negative result on record (reproducible). Thread B closed;
  next priority = **Thread A** (Khmer recognizer fine-tuning).
- Modules: `src/khmer_pipeline/layout_detect.py` (new), `hybrid_engine.py`, `tests/test_layout_detect.py`
  (new), `tests/test_hybrid_engine.py`, `scripts/visualize_layout.py` (new, verification overlays).
- **(Superseded by §2.25:** measured on **raw** images; with preprocessing Surya wins and the ranking flips.)

### 2.25 The preprocessing confound — re-scored A/B flips the ranking (Surya wins)

- **Why (methodology gap).** The product (`app.py`, `pipeline.py`) always runs `preprocess()` before OCR,
  but the eval harness did **not** — `scripts/eval_document.py:_load_pages` fed Surya **raw** `ingest()`
  images (as did the layout probe and `visualize_layout.py`). So the whole fragmentation arc was scored in
  a regime the real system never runs in. Surfaced by a `lab.py` smoke-test (the lab preprocesses).
- **The measurement.** On the dense page 2, Surya's layout gives **8 Table boxes raw but 1 clean box after
  preprocessing** (contrast + table-background flattening). Fragmentation is largely a *raw-image artifact*.
- **Re-scored A/B** (`eval_document.py --preprocess`, added this session; verified 75×9 doc GT):

  | engine | RAW (§2.24) Acc / Rec / CER | **PREPROCESSED** Acc / Rec / CER | pred dims raw → pre |
  |---|---|---|---|
  | **surya** | 0.170 / 0.722 / 0.348 | **0.259 / 0.623 / 0.249** 🏆 | 145×10 → **75×9 (= GT)** |
  | hybrid (rowband) | 0.181 / 0.566 / 0.341 | 0.145 / 0.569 / 0.258 | 84×9 → 82×9 |
  | hybrid + doclayout | 0.080 / 0.542 / 0.560 | 0.135 / 0.561 / 0.279 | 118×8 → 79×9 |

- **Result — the ranking flips.** Raw, hybrid narrowly "won" (0.181 vs 0.170). **Preprocessed, plain Surya
  wins decisively** (Cell_Accuracy 0.259 vs 0.145/0.135) and lands the **exact GT dimensions 75×9** (raw it
  over-produced 145×10). The hybrid gets *worse* with preprocessing, not better.
- **Revised conclusion.** The "structure/fragmentation is the bottleneck" thesis (§2.12) was largely an
  artifact of off-pipeline evaluation. **Under production conditions Surya handles the structure well**; the
  hybrid engine (SLANet + rowband) and DocLayout-YOLO — the whole fragmentation-mitigation effort — are
  **unnecessary and underperform**. The remaining gap is *recognition* (Recall ~0.62, CER ~0.25), which
  realigns with §2.21 (recognition is the open axis → Thread A). **Reassuring corollary:** `app.py` has
  always defaulted to **Surya + preprocessing** — i.e. the winning config — so the *deliverable* was correct
  all along; only the R&D *narrative* was skewed.
- **Eval hygiene going forward.** Run `eval_document.py --preprocess` to match production (recommended in
  `eval/README.md`). Raw stays the default flag-off for now so §2.24's numbers remain reproducible; flipping
  the default to preprocess is a deferred follow-up.
- **Not chased (breadth over depth).** Re-running the full fragmentation arc (§2.12–2.20) under
  preprocessing — only the current A/B was re-scored. The hybrid/DocLayout code stays in-tree, opt-in, as a
  documented negative result.
- Modules: `scripts/eval_document.py` (`--preprocess`), `lab.py` (per-page GT scoring), plus this log +
  memory. No `src/` engine change (the product already does the right thing).

### 2.26 Preprocessing ablation (E1) — the fragmentation fix is RESOLUTION normalization, not the OpenCV flags

- **Why.** §2.25 established that preprocessing collapses the dense page-2 layout **8→1** boxes, but not
  *which* step. Working hypothesis (from the design intent of `normalise_table_backgrounds`): stripping
  colored-cell cues is what stops the layout model fragmenting. Tested by component isolation.
- **Method.** Added a per-flag ablation to `scripts/eval_document.py`
  (`--no-deskew` / `--no-sharpen` / `--no-normalise` / `--no-remove-stamps` / `--no-table-bg`,
  leave-one-out on top of `--preprocess`) plus per-page `Tables_Found` printing. Scored on the verified
  75×9 09.06.26 document GT, `OCR_ENGINE=surya`. Committed `d7a9beb`.
- **Result — leave-one-out (page-2 `Tables_Found = 1` in EVERY preprocessed config):**

  | config | p2 boxes | Cell_Acc | Recall | Table_CER | pred dims |
  |---|---|---|---|---|---|
  | raw (no preprocess) | **8** | 0.170 | 0.722 | 0.348 | 145×10 |
  | all-on | 1 | 0.179 | 0.700 | 0.155 | 67×11 |
  | −deskew | 1 | 0.600 | 0.623 | 0.230 | 74×9 |
  | −sharpen | 1 | 0.243 | 0.783 | 0.074 | 75×11 |
  | −normalise (CLAHE) | 1 | 0.265 | 0.750 | 0.123 | 75×9 |
  | −remove_stamps | 1 | 0.206 | 0.755 | 0.142 | 75×9 |
  | −table_bg | 1 | 0.227 | 0.691 | 0.179 | 75×9 |

- **Decisive probe — all 5 flags OFF (crop+resize only):** page-2 `Tables_Found = 1`, Cell_Acc 0.225,
  dims **75×9**. With every tunable flag disabled, fragmentation stays fixed.
- **Conclusion.** No single `PreprocessConfig` flag is *necessary*, and disabling all five still prevents
  fragmentation. The cause is the two **always-on, ungated** steps in `preprocess.py` — `_crop_margins`
  and `_cap_resolution` (downscale long edge ≤ 2048 px, `_CAP_RESOLUTION_MAX_DIM`) — i.e.
  **geometric / resolution normalization**, not deskew / contrast / stamps / color. The §2.25 color-cue
  hypothesis (`normalise_table_backgrounds`) is **falsified**: removing the only color-stripping step
  changes nothing. The mechanism (a too-large dense table makes Surya's layout model tile & fragment it;
  downscaling merges it into one region). **⚠ Corrected by §2.28 (E3):** originally framed as
  document-agnostic / expected to generalize, but a structurally different dense table (CambodiaBudget)
  does **not** fragment at any resolution — so the effect is **layout-specific, not universal**.
- **⚠ Variance caveat.** Surya is non-deterministic: all-on scored Cell_Acc **0.179 / 67×11** here vs
  §2.25's **0.259 / 75×9** (same config). The **binary 8→1 fragmentation signal is robust and reproduced**;
  the accuracy point-estimates are **noisy** and must be reported with repeats, not as single numbers.
  (`−deskew`'s 0.600 is a single-run outlier — a hint that some steps may *hurt* on clean born-digital
  docs — needs repeats before trusting.)
- **E2 — multi-doc validation (2nd document, 15.06.26, same template / different day).** The 8→1 collapse
  **reproduces**: page-2 `Tables_Found` = **8 raw → 1 preprocessed**, identical to 09.06.26. Preprocessing
  also sharply improves content on this 2nd doc — Table_CER **0.360 → 0.091**, Recall **0.736 → 0.783**,
  pred dims **144×11 → 75×10** (≈ GT 75×9). The small Cell_Acc dip (0.187→0.170) is a spurious 10th column
  shifting cells, not a content regression. **n=2 generalization of the resolution mechanism confirmed** —
  across *instances of this template*. **Cross-*layout* generalization is now tested in §2.28 (E3) and is
  NEGATIVE** — the effect does not extend to a structurally different dense-table layout.
- **15.06.26 GT provenance.** Its ground truth was built by transferring 09.06.26's hand-verified Khmer
  item-names + table structure and injecting 15.06.26's own numeric cells (prices/percentages/dates
  extracted from its text layer). This is valid because the two PDFs share the *same* broken ToUnicode
  CMap: the garbled Khmer is unusable as text but **stable** (same item → same garbled string), so it
  aligns rows reliably, while digits extract correctly in both. All 71 data rows were verified to align
  1:1 (section index + row number + garbled name) before transfer; document GT = 75×9. (The one-off
  builder script is not kept in-tree.)
- **Open follow-ups.** Confirm resize-vs-crop is the lever + find the resolution threshold (sweep the
  2048 px cap); variance repeats on raw / all-on / all-off.
- Modules: `scripts/eval_document.py` (`--no-*` ablation flags + per-page `Tables_Found`).

### 2.27 Recall-failure taxonomy — the residual gap is RECOGNITION, not layout → fine-tuning is justified

- **Why.** Under production (Surya + preprocessing) the doc reaches ~correct dims but `Cell_Content_Recall`
  ≈ 0.62–0.78 — 20–38% of GT content unrecovered. *Which* failure mode? This decides the fine-tuning fork:
  recognition misses → fine-tuning can help; segmentation misses → it won't.
- **Method.** `scripts/recall_taxonomy.py` reruns the production pipeline on 09.06.26, aligns the stitched
  predicted grid to the 75×9 GT (the same `evaluate_structure` difflib alignment), and classifies every
  unrecovered GT cell. Row correspondence cross-checked with an independent fuzzy item-name match to rule
  out an alignment artifact (difflib collapsed to one `replace` block because pred had 11 cols vs GT 9).
  Single run (`Cell_Content_Recall = 0.759`; 139 missed of 576 non-empty). Surya variance applies.
- **Taxonomy:** WRONG-TEXT 64.0%, CELL-BLANK 31.7%, MERGED 4.3%, ROW-DROPPED / SPLIT 0. →
  **recognition-attributable 95.7%, segmentation 4.3%** (the 6 merged rows are all in the grains section).
- **Where misses cluster.** Unit column `ឯកតា` = **51%** of misses; item names 25%; the four numeric price
  columns are barely affected (0.7–2.9% each). Root cause of the unit misses: the **Riel glyph `៛` is
  systematically misrecognized** (`៛/គ.ក` → `#គ.ក` 22×, `វ/គ.ក` 12×, `អ/គ.ក` 9×) — one narrow, concentrated
  confusion. Item-name misses are Khmer subscript-consonant substitutions (e.g. `គោ`→`តោ`). By section,
  grains is worst (50.6% miss + all 6 merged rows).
- **Conclusion.** The residual gap is **recognition (glyph-level) on correctly-segmented cells**, not
  layout. **Fine-tuning is the justified lever** (echoes §2.21: no turnkey model beats Surya). Layout /
  stitch work would touch only the ~4% segmentation slice.
- **Cheap near-term win (not yet done).** The unit column is near-constant (`៛/គ.ក` / `៛/គ្រាប់` / `៛/ផ្លែ`)
  and drives 51% of misses through one glyph, so a **deterministic post-processing rule** normalizing the
  misrecognized Riel prefix (`#` / `វ` / `អ` `/…` → `៛/…`) could recover a large share of recall for
  near-zero cost — worth trying before the 4–6 week fine-tune. (Extends `postprocess.py` / `khmer_normalize.py`.)
- Modules: `scripts/recall_taxonomy.py` (new).

### 2.28 Cross-layout fragmentation probe (E3) — the defrag effect is LAYOUT-SPECIFIC, not universal (corrects §2.26)

- **Why.** E1/E2 established the raw→~8, preprocessed→1 collapse and its resolution lever, but only on the
  market-price *bulletin* template (09/15 = same layout, different dates). Does it generalize to a
  structurally different dense table? GT-free test — fragmentation = `Tables_Found` from Surya's layout
  output on pixels, so no ground truth is needed (font-independent).
- **Method.** `scripts/probe_cambodiabudget_fragmentation.py`: on `CambodiaBudgetExecutioninApr-2024.pdf`
  dense-table pages (3,4,5,6,8,9), count Surya "Table" layout regions on RAW vs PREPROCESSED images
  (default all-on config), DPI 200, `OCR_ENGINE=surya`, cache cleared between passes. Variance re-check on
  page 3 (2 passes, identical).
- **Result — NO fragmentation on any page, either condition:**

  | page(s) | raw Table-regions | preprocessed | raw long edge |
  |---|---|---|---|
  | 3 / 4 / 5 / 6 | 1 | 1 | 4400 px |
  | 8 / 9 | 1 | 1 | 4151 px |

- **Correction to §2.26.** These pages have raw long edges **4151–4400 px — far above** the 2048 px
  `_cap_resolution` threshold — yet **do not fragment raw**. So high resolution is NOT *sufficient* to cause
  fragmentation, and §2.26's "large raw dims → tile → downscale merges → document-agnostic, expected to
  generalize" was **too strong**. The defrag effect is not a universal dense-table fix — on this layout
  there is nothing to fix.
- **Refined mechanism (hypothesis).** Fragmentation is **layout-specific**: the bulletin is a mosaic of
  many small, individually shaded/colored price cells packed edge-to-edge (plausible sub-structure for
  Surya's layout tiler to split along at high resolution); the budget-execution table is one bounded
  uniform grid with no cell-level color fill (nothing to fragment along), regardless of pixel count. So the
  trigger is a **visual-density / cell-structure pattern interacting with resolution**, not raw pixel count
  alone. (Downscaling still resolves it *on the bulletin*, per E1 — it just isn't a general fix.)
- **Thesis consequence.** Scope the claim to *"preprocessing resolves the fragmentation of the dense
  colored-cell market-bulletin layout"* (validated across 09/15), **not** *"preprocessing fixes dense-table
  fragmentation in general."*
- **Open.** Separate visual-structure vs resolution on the bulletin (color-flatten-without-downscale vs
  downscale-keeping-color); resolution-threshold sweep.
- Modules: `scripts/probe_cambodiabudget_fragmentation.py` (new).

---

## 3. Results Snapshot

First trustworthy benchmark — engine `run_surya`, 30 images (5 fonts × 3 templates
× 2 datasets), **raw render, no preprocessing, free deterministic metrics**,
after the row-alignment fix (§2.7).

**Structural health (all 30 images):** `Tables_Found == Tables_Expected == 1`
(table detection never failed) and `Paragraph_Leak == 0` (no body text leaked into
tables — the §2.4 redesign holds).

**Per-font** (Cell_Accuracy and Content_Recall higher = better; CER lower = better):

| Font | Cell_Acc | Content_Recall | Table_CER | Text_CER |
|---|---|---|---|---|
| **Noto Sans Khmer** | **0.82** | **0.94** | **0.043** | **0.044** |
| Battambang | 0.74 | 0.85 | 0.171 | 0.14 |
| Hanuman | 0.65 | 0.87 | 0.106 | 0.19 |
| Moul | 0.52 | 0.62 | 0.252 | 0.42 |
| Fasthand | 0.48 | 0.68 | 0.203 | 0.30 |

**Headline.** **Noto Sans Khmer is decisively the best-supported font** (near-
perfect on isolated tables: 1.00 / 0.96 / 0.96 cell accuracy). Battambang and
Hanuman are usable; **Moul and Fasthand are poor** — both are decorative/display
typefaces, an expected limitation for OCR rather than a pipeline defect.

**Known residual limitation.** In ~2–3 of 30 images (e.g. `table_1_Hanuman`,
`doc_0_Fasthand`) the model emits a **spurious extra column** (`Pred_Cols = 5` vs
`GT_Cols = 4`), which shifts cells horizontally; row alignment does not correct
column drift. Rare; logged rather than chased. A column-alignment counterpart to
§2.7 is the natural future fix if it proves common.

*(Numbers from `eval/runs/<ts>_run_surya/results.csv`; regenerate with
`uv run python -m khmer_pipeline.run_benchmark` then `uv run python -m khmer_pipeline.analyze_benchmark`.)*

### Real-document results (first real GDDE doc, 2026-06-22)

A real born-digital GDDE daily market-price PDF (3 pages, dense Khmer price tables),
hand-labelled as ground truth (paragraphs). Run `eval/runs/20260622_114939_run_surya`.

| Page | Tables_Found | Document_CER | Note |
|---|---|---|---|
| p1 | 1 | 0.30 | clean single table |
| p2 | **8** | **0.70** | one table fragmented into 8 regions |
| p3 | 1 | 0.22 | clean single table |

**Key finding — the bottleneck is layout/table-structure, not character recognition.**
Inspecting the saved OCR-vs-GT prediction dumps: the model's *character* recognition
is strong (~90%+ of product names and **all** numeric values correct, only minor
slips like `ត្រកួន→ត្រកូន`, riel sign `៛→រ`). But on the dense page 2, Surya's layout
model **fragmented one table into 8 regions**, which serialized the content
column-wise (all names, then all numbers, then all percentages) and destroyed the
row↔value associations. Because CER is order-sensitive, this *reordering* — not bad
OCR — is what drives `Document_CER` to 0.70 (vs 0.22–0.30 on the cleanly-detected
pages). Two minor noise artifacts on page 2: a hallucinated Kannada line and a
repeated column header.

**Implications.** (1) Raw Khmer OCR quality on real born-digital docs is better than a
flat CER suggests. (2) For financial tables the metric that matters is *structural*
(`Cell_Accuracy` — does item N map to price N?), and `Tables_Found vs Expected` is a
useful **fragmentation** signal. (3) Reducing table-region fragmentation on dense
tables is the highest-value engineering target for real-world use. (4) The
born-digital PDF's own embedded text layer is garbled (broken ToUnicode CMap), so OCR
on rendered pixels is genuinely necessary — text extraction is not a shortcut.

### Surya vs Tesseract baseline (text-only, 2026-06-23)

Recognised-baseline comparison. Both engines run on the **same images, raw render, no
preprocessing**. Runs: `eval/runs/20260622_154407_run_surya` (raw) vs
`eval/runs/20260623_100406_run_tesseract` (Tesseract 5.5.2, `khm` traineddata).

**Per-engine aggregate (33 images each):**

| Engine | Cell_Acc | Table_CER | Text_CER | Document_CER |
|---|---|---|---|---|
| `run_surya` | **0.589** | **0.180** | **0.335** | **0.325** |
| `run_tesseract` | 0.000 | 0.970 | 0.367 | 0.443 |

**Per-dataset Document_CER (lower = better):**

| Dataset | Surya | Tesseract |
|---|---|---|
| synthetic_tables | **0.165** | 0.656 |
| synthetic_documents | 0.450 | **0.161** |
| real | **0.503** | 0.792 |

**Scope / caveats (this is a *text-only* comparison — read with care):**
1. **Tesseract produces no table structure** (`tables=[]` → `Cell_Accuracy = 0.000`
   on every dataset). For the financial-table use case this is disqualifying on its
   own, independent of CER — it is a flat text reader, not a layout/structure engine.
2. **Tesseract inserts spaces between Khmer clusters and garbles dense numeric
   columns.** On the real doc its prediction reads the title and product names
   reasonably but turns the price columns into spaced gibberish
   (e.g. `@យ 2១ 2១ 2១ …`), inflating its CER — a real property of the engine,
   reported as-is.
3. **The one place Tesseract "wins" (synthetic_documents, 0.161 vs 0.450) is a
   metric artifact, not superiority.** `Document_CER` pools all text into one
   order-sensitive string; a linear top-to-bottom reader (Tesseract) aligns with the
   GT pooling order, whereas Surya's *structured* output (paragraphs in `ocr_text` +
   cells in `tables`) pools in a different order even when the content is correct.
   The same order-sensitivity inflates Surya's **real** `Text_CER` (0.946 vs
   Tesseract 0.731) because fragmentation reorders paragraph text. The structural
   metric `Cell_Accuracy`, not pooled CER, is what reflects real-world usefulness.

**Conclusion.** Surya is the correct engine for structured Khmer financial documents
(it is the only one that yields cell-level table structure, and it wins on the
pooled metric overall: Document_CER 0.325 vs 0.443). Tesseract is a legitimate,
recognised *flat-text* baseline but is not structure-aware; the text-CER comparison
is genuinely mixed and confounded by reading-order effects and its Khmer
cluster-spacing, which we report honestly rather than cherry-picking. Figure:
`engine_comparison.png` (regenerate via `visualize_benchmark <surya_run> <tesseract_run>`).

---

## 4. Lessons / Principles

- **Prefer a single source of truth over joining independently-derived
  structures.** The table-cell bug persisted as long as two separately-inferred
  grids were index-joined; it vanished once one source (the VLM HTML) owned both
  text and structure (§2.4).
- **Use deterministic metrics over an LLM judge wherever ground truth exists** —
  free, exact, reproducible; reserve a (preferably local) judge for the genuinely
  reference-free case (§2.5).
- **Isolate the component under test.** Feeding raw renders straight to OCR
  removed preprocessing as a confound and made font-to-font comparison meaningful
  (§2.6).
- **Fail loud on silent-failure risks.** A font that doesn't load now aborts
  generation instead of silently producing a fallback-font image that would
  corrupt the comparison (§2.6).
- **Make long runs crash-safe and provenance-tagged.** Incremental writes plus an
  `Engine` column mean a mid-run crash loses nothing and every result is
  attributable to the model that produced it (§2.6).
- **Distrust a single headline metric; cross-check.** `Cell_Accuracy` looked
  catastrophic until a second metric (`Content_Recall`) on the same rows revealed
  it was a row-alignment artifact, not OCR failure (§2.7).
