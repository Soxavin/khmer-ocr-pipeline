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

### 2.29 Recognizer track kickoff — CRNN training exercise + off-the-shelf Khmer-OCR survey (2026-07-06)

Thread A (recognition) opened on two fronts. Full write-ups live under `experiments/khmer_crnn/`
(`FINDINGS.md`, `FINETUNING_PLAN.md`, `HANDOFF_TASKS.md`); summary here.

- **CRNN training exercise.** Adapted a mentor-provided (CUDA-oriented) starter script into a portable,
  rigor-added trainer (`experiments/khmer_crnn/train.py`) that trains a ResNet+BiRNN+CTC recognizer **from
  scratch** on `seanghay/khmer-hanuman-100k` (single font) — purpose: **learn the training loop + benchmark
  epoch time on the M4 (MPS)**. Adaptations: portable device (`utils/device.detect_device`), MPS **CTC
  runs on CPU fallback** (`aten::_ctc_loss` unimplemented on MPS), leakage-safe split + train-only vocab,
  validation CER, seeding/checkpoints, warmup-aware timing, and a **CTC-feasibility check** that surfaced
  the real dataset shape (labels up to 139 chars, images ~1068px wide) → widened input 256→1024px + label
  filter.
- **Benchmark + convergence.** ~**121 s/epoch** (ResNet34) / ~**76 s** (ResNet18, ~1.6× faster); GRU≈LSTM
  for speed (CNN + CPU-CTC bound); no thermal throttling. Trains cleanly: CTC blank-collapse breakout at
  epoch 3–4 → **~3.4% CER** (short-label curriculum) and **~3.7% CER** (full sentence-length task). Confirms
  the pipeline is sound; single-font Hanuman won't read the real GDDE docs (by design).
- **Off-the-shelf survey (via `seanghay/awesome-khmer-language`).** Empirically tested two Khmer OCRs on the
  **real** page `09.06.26_p2`:
  - **seanghay/KhmerOCR** — Khmer-**script-only** output vocab (98 chars; no Arabic digits/punctuation).
    **Dropped all six Arabic-numeral price/percentage columns** → architecturally unusable for our tables.
  - **mrrtmob/kiri-ocr** — bilingual EN+Khmer, **Apache-2.0**, transformer CTC+attention (vocab 967 covers
    Arabic digits + `%.,-/()` + Latin + Khmer). Off-the-shelf it duplicated digits — **traced to the decoder**:
    `decode_method="accurate"/"beam"` doubles digits, but **`decode_method="fast"` (pure CTC) reads the real
    page's Khmer, row numbers, `៛` units, and all Arabic prices correctly at ~99% confidence**. Only the small
    %-cells fail, and that's a *detector* mis-crop (recognizer reads `-2.86%` perfectly when cleanly cropped).
- **Direction.** The near-term win is a **Surya-detect + Kiri-recognize(fast) hybrid** (Surya's table
  structure + Kiri's mixed-script recognition), evaluated vs Surya-alone via the `evaluation/` harness — a
  local, MEF-safe (Apache-2.0), **no-fine-tune** path to better recognition. Fine-tuning (Kiri ships
  `training.py`, or our own CRNN) stays as an optional later quality lever. Spec: `HANDOFF_TASKS.md` Task #4.
- Modules: `experiments/khmer_crnn/{train,metrics,plot_metrics}.py` + `{README,FINDINGS,FINETUNING_PLAN,HANDOFF_TASKS}.md` (new);
  `pyproject.toml` (new `experiments` optional dep group: torchvision/datasets/psutil, pinned). Training run
  artifacts (`experiments/khmer_crnn/runs/`) are gitignored.

---

### 2.30 `surya_kiri` engine shipped + honest head-to-head — a modest, situational win (2026-07-06)

Productionised the Surya-detect + Kiri-recognize(fast) + per-cell-Otsu hybrid as a selectable engine
`OCR_ENGINE=surya_kiri` (`engines/surya_kiri_engine.py`, `engines/kiri_recognizer.py`, vendored recognizer
under `engines/kiri_vendor/`). Kiri is **vendored, not depended-on**: only the CTC (`fast`) path + a
weights loader, so there is **no `onnxruntime-gpu`** (no macOS-ARM wheels) and no network dep beyond the HF
weight download. Equivalence-tested against the upstream git-main package: **12/12 byte-identical** cell reads.

- **Vendoring gotcha (the hard part).** The HF checkpoint's `config.json` is **stale** (describes an older
  dim-256/4-layer variant) and uses a non-`CFG` schema, so trusting it silently mis-sizes the model and
  `load_state_dict(strict=False)` leaves whole modules random → garbage OCR. The architecture must be
  **inferred from the weights** and copied **verbatim**: `SiLU` (not GELU), conv strides `(1,1),(2,2),(2,2),(2,1)`,
  6 encoder layers, **6 attention heads** (`dim//64`, not the config's "8"), the exact 2-D positional encoding,
  and gray-128 padding. The loader now infers all of this and hard-fails if any CTC-path key is missing.
- **Step 0 — raw vs preprocessed (resolved: raw).** Preprocessing (CLAHE/desaturation) helps Surya's
  structure but **degrades Kiri recognition even after Otsu** (p2 CellAcc 0.790 raw → 0.675 preprocessed).
  Because preprocessing also deskews/crops, preprocessed-space bboxes don't map onto raw pixels, so the engine
  runs its **whole** table pipeline (layout → TableRec → crop) on the raw page. Threaded via a new optional
  `PreprocessResult.raw_page_images` (populated by `preprocess()`; falls back to `page_images`). Verified: the
  production path `ingest → preprocess → surya_kiri` reproduces the raw score (p2 = 0.790).
- **Honest head-to-head (both engines, production path, all 6 real pages — corrects the §2.29 direction).**

  | engine | Cell_Accuracy | Recall | Table_CER |
  |---|---|---|---|
  | `surya` | 0.511 | **0.759** | 0.097 |
  | `surya_kiri` | **0.580** | 0.755 | **0.086** |

  `surya_kiri` wins Cell_Accuracy (+0.07) and CER and ties Recall — but the earlier **"beats Surya on ALL
  metrics"** claim (based on a stale single-page Surya baseline of 0.259, pre-§2.25) **does NOT hold**. A
  fair, fresh full-page comparison shows Surya-alone is strong (0.511) and actually **edges the hybrid on the
  cleanest data page p2 (0.844 vs 0.790)**. The hybrid's real advantage is **robustness on structurally harder
  pages** (p3: 0.75 vs 0.51, where Surya mis-counts rows) and lower CER. Verdict: a **modest, situational**
  improvement worth shipping as an option — not a landslide.
- **Known limitation (p1 header pages, ~0.20 CellAcc / 0.75 Recall).** Diagnosed precisely: Surya's
  `TableRecPredictor` splits the **two-physical-line column header** (date line + `បោះដុំ/លក់រាយ` line) into
  **two** rows, while the GT merges them into **one** logical header row (pred 25×9 vs GT 24×9). Everything
  from the category-title row onward aligns; recall is unaffected. Left as a documented limitation rather than
  a header-merge heuristic (which would risk overfitting these 2 pages and regressing the matched p2).
- Modules: `engines/surya_kiri_engine.py`, `engines/kiri_recognizer.py`, `engines/kiri_vendor/{model,loader}.py`,
  `tests/test_{surya_kiri_engine,kiri_recognizer}.py` (new); `engines/engine_registry.py` (register),
  `engines/surya.py` (`get_manager()`), `models.py` + `preprocess.py` (`raw_page_images`). ~394 tests green.

---

### 2.31 `surya_kiri` productionised — UI-selectable, ~2.4× faster, skew-robust, confidence-aware (2026-07-08)

Took the validated `surya_kiri` engine from "works in a script" to a first-class, integrated pipeline engine.
Every change is data-driven; several *rejected* options are recorded because the measurement is the finding.

- **UI integration.** `get_ocr_engine(name)` helper on the registry + a sidebar **"OCR engine"** picker in
  `app.py` (Surya default / Surya+Kiri opt-in, wired into `settings_key`); same engine added to the `lab.py`
  comparison tool. No `OCR_ENGINE` env var needed. CLI/eval keep the env-var default via `ACTIVE_OCR_ENGINE`.
- **Speed: ~42s → ~17.5s/page (~2.4×), output byte-identical.** (1) `run_surya(skip_tables=True)` drops Table
  regions before recognition so Surya's expensive table-HTML VLM never runs (base OCR pass 32s → 1.2s) — the
  hybrid rebuilds tables itself anyway. (2) Batched Kiri recognition (`recognize_cells`, `_BATCH_SIZE=64`)
  replaces 240 per-cell forwards + temp-PNG round-trips.
- **REJECTED — Kiri on MPS.** Measured ~1s gain (17.5→16.5s, within noise) → reverted. Finding: after batching,
  **Surya's models (2 layout passes + TableRecPredictor), not Kiri, are the floor.** MPS also risks GPU-memory
  contention + output drift. **REJECTED — eliminate the 2nd layout pass** (~3s): a core-path refactor of shared
  Surya code for a small gain + a text-from-raw tradeoff; not worth it.
- **Geometric-only preprocessing (the skew fix).** The engine previously recognised from *fully-raw* pixels
  (photometric steps hurt per-cell Otsu) — but that also skipped **deskew**, leaving it catastrophically
  fragile: a **4° tilt dropped it 0.79 → 0.03** (silent garbage; TableRec collapses). A 4-way experiment
  (straight/skewed × raw/geometric) settled it: recognise from a **geometric-only** image (crop + deskew, NO
  photometric) via new `_geometric_preprocess` + `PreprocessResult.recognition_page_images` (renamed from
  `raw_page_images`). Result: skew recovers **0.03 → 0.58**, AND the clean-eval mean *rose* **0.580 → 0.586**
  (p3 structure fixed to 24×9). This also makes the app's `deskew` toggle — previously a silent no-op for the
  hybrid — actually work, bringing it to parity with Surya-alone on geometric robustness.
- **Per-cell confidence.** `recognize_cells_conf` returns `(text, conf)` (mean max-softmax over non-blank CTC
  timesteps — the value we were discarding); every table cell now carries `cell["confidence"]`, and a per-page
  warning flags cells below 80% ("verify those cells"). Confidence lives on the cells → a visual heatmap is a
  clean UI-only add later.
- **REJECTED — `៛`/digit normalization toggle.** Blanket digit conversion already exists (`convert_numerals`,
  export.py); the only surgical add (fix mixed-script `០.00%` slips while preserving the *legitimately-Khmer*
  row-index column `២៣`) is niche + overlapping. The real fix for the `៛`-glyph systematic error is
  **fine-tuning Kiri**, not a postprocess band-aid.
- Commits `1849e0f`, `8270dcf`, `75f0258`, `1371938`, `9ba748a`, `1ccce04`; 409 tests green. NEXT (optional):
  fine-tune Kiri on the `៛` glyph; visual confidence heatmap (data already on the cells).

### 2.32 Hardening pass from the architecture audit — fail-loud, metric-neutral (2026-07-08)

Implemented the Phase 1 + Phase 2 fixes from the architecture/code-quality audit
(`docs/` audit plan, 2026-07-08). Every change is additive or fail-loud and
**metric-neutral**: the full unit suite went 409 → 447 green and
`OCR_ENGINE=surya_kiri scripts/eval_document.py --preprocess` reproduces the §2.31
baseline exactly (Cell_Accuracy 0.173 / Recall 0.762 / Table_CER 0.086 on the
local real doc; the eval bypasses postprocess, so A6 cannot move it). Phase 3
(A4 wide-cell splitting, A2's Surya-table fallback, B6 stitch heuristics, B8 stamp
mask) is deliberately out of scope — each is an A/B-gated experiment.

- **Correctness / fail-loud.**
  - **A1** — the app's `settings_key` and new-file reset now key on Streamlit's
    per-upload `file_id` (fallback: session-cached content hash), so re-uploading a
    *modified* file with the same name can no longer serve stale results.
  - **A5** — pinned the Kiri HF download to `revision=3a3819874…` (model + vocab,
    same snapshot) so an upstream re-push can't silently swap the weights.
  - **A2 / A7 / A3** — silent table drops (`surya_kiri`), one recognition-HTML block
    claimed by two tables (`surya`), and Kiri recognizer failures now all surface
    through `SuryaResult.warnings` instead of vanishing. Kiri failures route through
    a per-run `warning_sink`; the load-failure latch is reset once per run
    (`reset_kiri_failure()`), so a transient first-run blip no longer disables Kiri
    for the whole Streamlit process.
  - **B2** — an unknown `OCR_ENGINE`/`get_ocr_engine` name now raises `ValueError`
    (was a silent fallback to Surya — a typo'd benchmark tested the wrong engine).
  - **B3** — `--resume` benchmarks recompute aggregates/summary from the FULL
    `results.csv`; a malformed GT file yields an Error row instead of aborting.
  - **B7** — multi-frame TIFFs are fully ingested (was frame-0 only).

- **Invariant decisions (Phase 2).**
  - **A6 ends the tables aliasing.** Stage 4 (`postprocess`) now normalizes table
    cell text (NFC / ZWSP-BOM strip / dup-diacritic collapse) copy-on-write: new
    table + cell dicts, so `SuryaPageResult.tables` stay byte-identical. `export`'s
    in-place table repair therefore touches only the `PostprocessResult`; the
    `was_repaired` badge reaches the UI via the JSON (app.py never read it through
    `SuryaResult`). `run_benchmark.py`'s "tables unchanged by correction" assumption
    still holds by construction (it reads `ocr_result.pages`, not the corrected copy).
    This is the cheapest accuracy-hygiene win: the CSV/JSON cells finally get the
    same normalization the page text already had.
  - **B5 gates the second preprocessing pass.** New INTERNAL
    `PreprocessConfig.with_recognition_images` (default True; **exempt** from the
    4-point sidebar/CLI pattern by design) — orchestrators set it False for every
    engine except `surya_kiri`, halving preprocessing work on the default path. The
    `recognition_page_images is None` fallback in `surya_kiri` now **warns** (it is a
    measured 0.79→0.675 loss, §2.30), and `preprocess()` asserts each recognition
    frame shares its page frame's H×W (geometric steps must precede photometric ones,
    or bboxes desynchronize).
  - **B4 page-selective ingest.** `ingest(page_indices=…)` renders only the requested
    PDF pages (`doc.load_page(i)`); the `MAX_PAGES` cap applies to rendered pages, so
    a long PDF is fine when few pages are selected. `app.py` computes the selection
    *before* ingest and no longer keeps the full-document `IngestResult` in
    session_state. Page-index semantics are preserved (0-based within the selection).

- **Polish (D).** BOM literal → `_CSV_BOM` constant; `traceback` added to the
  critical-failure console log (not the analyst warning); `datetime.utcnow()` →
  `datetime.now(timezone.utc)`; `_anomaly_score` divides by non-whitespace count;
  `playwright`/`openai` moved to an `eval-extras` optional group; `eval/` paths
  anchored to the repo root; provenance block (engine/version/settings) added to the
  exported JSON (C6); per-cell `confidence` carried into the exported JSON (C3); docs
  drift fixed (CONTEXT engine list, eval/README preprocessing field).

### 2.33 Kiri-era numeric/failure measurement — the "fusion" premises fail their own data (2026-07-08)

- **Why.** A proposed `surya_kiri_fusion` engine rested on three premises: Kiri is weak on
  Arabic numerals, Kiri drifts columns, and a per-cell Surya second opinion helps. Two are
  already refuted architecturally (§2.15 per-cell Surya failed; Kiri emits no bbox so it *cannot*
  drift — all structure is Surya's, §2.30). This entry replaces the numeric premise with
  measurement: **where do `surya_kiri`'s errors actually live, how accurate are numeric cells,
  and is per-cell confidence calibrated?**
- **Method.** `scripts/recall_taxonomy.py` (OCR_ENGINE=surya_kiri, `--preprocess`) on the two
  verified 75×9 document GTs (09.06.26 + 15.06.26; `needs_review_rows==[]` for both, so neither is
  provisional). Added a new value-accuracy metric `Numeric_Cell_Accuracy` (+ `Numeric_Khmer_Digit_Slips`)
  to `evaluation/evaluate_structure.py`, threaded through `run_benchmark.py`/`analyze_benchmark.py`
  (TDD, +19 tests), and a **per-cell confidence-calibration hook** to `recall_taxonomy.py`.
- **Alignment caveat (important, honest).** Both docs predict **76×9 vs GT 75×9** — the known p1
  two-physical-line header split (§2.30) adds one leading pred row. `recall_taxonomy.py`'s difflib
  row-pairing collapses this into one giant `replace` block and mislabels **all** misses as `SPLIT`
  (its printed mode table + a naïve calibration are therefore *artifacts*). The document's numeric
  row-index column proves a **clean constant +1 offset** (`GT[i] ↔ pred[i+1]`, 71/75 anchor hits),
  so the multiset recall, by-column and by-section distributions (all alignment-independent) are
  trustworthy, and the corrected 1:1 taxonomy/calibration below use the detected offset.
- **Failure taxonomy (offset-corrected, non-empty GT cells, pooled 1152 cells over both docs).**
  **RECOGNITION-attributable: 100%** — `WRONG-TEXT 24.0%`, `CELL-BLANK 0%`, `ROW-DROPPED/MERGED/SPLIT 0`.
  Rows are correctly segmented and 1:1; every miss is a legible cell read as different text (confirms
  §2.27's Surya-era conclusion: the residual gap is glyph-level recognition, not layout).
- **Where misses cluster (by column, 09.06 / 15.06 multiset-miss share).** Unit `ឯកតា`
  **52.6% / 51.4%** — every unit cell wrong. Then retail-%chg **17.5% / 15.7%**, wholesale-%chg
  **7.3% / 8.6%**, `08-06 retail` **3.6% / 5.0%**; the other three numeric price columns are near-perfect
  (0.7–2.1%). By section, grains (`គ`) is worst (46.8%) — matches §2.27 exactly.
- **The dominant error is ONE non-numeric glyph, at HIGH confidence.** The Riel sign `៛`: pooled **142**
  confusions, overwhelmingly `៛/គ.ក → អគ.ក` (58× + 57×), plus `→ #គ.ក` (5×+5×), `អគៈក`/`អគ:ក`. This alone
  is ~52% of all misses, and it is emitted at **0.94–0.99 confidence** — so a confidence gate never flags it.
  This reproduces §2.27's Surya-era `៛` finding on a *different* recognizer → the `៛` glyph is hard for
  both models, not a Kiri-specific numeric weakness. Item-name misses (5–6%) are subscript-consonant
  substitutions (`សាច់ជ្រូក→សាច់ជ្រក`, dropped `ូ`).
- **Numeric-cell accuracy — the premise-killer.**

  | doc | numeric GT cells | value-correct (folded) | Numeric_Cell_Accuracy | Khmer-digit slips |
  |---|---|---|---|---|
  | 09.06.26 | 422 | 402 | **0.953** | 100 |
  | 15.06.26 | 422 | 399 | **0.946** | 99 |
  | **pooled** | **844** | **801** | **0.949** | **199** |

  Numbers are read **94.9% correct by value**. Of the 199 "Khmer-digit slips", ~71/doc are the
  **legitimately-Khmer row-index column** (GT is `១,២,៣…`; correct AND flagged); the only true Arabic→Khmer
  slip is the leading `0` in zero-change cells (`0.00% → ០.00%`, ~25/doc) which folds back to the right value.
  Genuine value errors are rare: digit-duplication (`8.33%→8333%`, `-13.33%→-13333%`) appeared **3× in 09.06,
  0× in 15.06**, plus a few `%`-cell mis-crops (`7,000→7,000ក`, `2,500→2;500`).
- **Confidence calibration (offset-corrected, non-empty GT, strict match).**

  | conf bucket | 09.06 cells / match-frac | 15.06 cells / match-frac |
  |---|---|---|
  | `<0.50` | 1 / 0.000 | 1 / 0.000 |
  | `0.50–0.80` | 19 / 0.368 | 14 / 0.357 |
  | `0.80–0.95` | 239 / 0.665 | 234 / 0.671 |
  | `≥0.95` | 317 / 0.861 | 327 / 0.838 |
  | **< 0.80 (warns)** | **20 / 0.350** | **15 / 0.333** |
  | **≥ 0.80 (no warn)** | **556 / 0.777** | **561 / 0.768** |

  Monotonic → confidence **is** calibrated; the `_LOW_CONF_THRESHOLD = 0.80` edge is reasonable (below-0.80
  cells are ~2.3× likelier wrong). But the top bucket is still only ~85% correct **because the systematic `៛`
  misread is confident** — the threshold cannot catch the single biggest error class.
- **Conclusion.** (a) **The "Kiri numeric weakness" premise is false**: numeric cells are **94.9% value-correct**;
  the numeral-blindness that motivated fusion does not exist in production. The real error is the **non-numeric
  `៛` unit glyph** (~52% of misses) plus mixed-script `0→០` cosmetics — neither is what a Surya numeric second
  opinion would fix, and both models miss `៛` identically. (b) **Step 2 rule-based corrections** should target the
  measured, deterministic patterns: normalize the Riel prefix `អ/គ.ក`, `#/គ.ក`, `អគៈក`, `អគ:ក` → `៛/គ.ក` (and
  `អគ្រាប់→៛/គ្រាប់`, `#ផ្លែ→៛/ផ្លែ`) on the near-constant unit column, and fold leading `០→0` in `%`-pattern cells;
  **never** auto-rewrite the digits themselves — instead cap confidence + warn on the digit-duplication /
  malformed-number pattern (`\d,\d{4}`, `\d+%` with 4+ fractional digits) so it routes to analyst review.
  (c) The **0.80 threshold is calibrated** as a general error-likelihood signal but is **blind to the confident `៛`
  error**, so routing/verification must pair it with the deterministic `៛` rule (or Kiri fine-tuning, §2.29/Step 3),
  not rely on confidence alone. **Net: build the deterministic corrections + fine-tune; do NOT build the fusion engine.**
- Modules: `evaluation/evaluate_structure.py` (`Numeric_Cell_Accuracy`, `_is_numeric`/`_fold_numeric`/`_has_khmer_digit`),
  `evaluation/run_benchmark.py` + `analyze_benchmark.py` (CSV col + `avg_numeric_cell_accuracy` + summary col),
  `scripts/recall_taxonomy.py` (confidence-calibration hook, offset-robust alignment, conf-grid dump);
  `tests/test_{evaluate_structure,run_benchmark}.py` (+19). Measurement-only: no engine/pipeline behavior changed.

---

### 2.34 GDDE-domain cell rules + malformed-number flag + per-cell confidence view (2026-07-09)

Implements §2.33's conclusion (deterministic corrections, no fusion) plus the analyst-facing
confidence view. Design constraint from the user: the bulletin docs are the TEST SET, not the target
scope — rules must be provably unable to alter other document types.

- **Domain rules (`postprocess._apply_cell_rules`, applied to table cells in Stage 4).** Kept
  deliberately separate from the script-level normalizer (`khmer_normalize.py` untouched). Two rules,
  both full-cell pattern matches on corrupt forms that are not plausible Khmer text: (1) riel-prefix
  repair — `^[អ#វ]/?(គ.ក|គ្រាប់|ផ្លែ)$` (and the `ៈ`/`:` dot-misread variant) → `៛/<unit>`;
  (2) percent Khmer-digit fold — percent-shaped cells containing Khmer digits get digits folded to
  Arabic (`០.00% → 0.00%`). Khmer row-index cells (no `%`) pass through untouched.
- **Malformed-number FLAG, never a rewrite.** Digit-duplication artifacts (`\d,\d{4}` comma
  violations; `^[+-]?\d{4,}%$` implausible integer percents) get their confidence capped to 0.4
  (< CONFIDENCE_LOW → red in the UI) + a warning naming page/table/row/col. Financial digits are
  never auto-corrected. Carried by a minimal Stage-4 warnings channel: `PostprocessResult.warnings`
  (new field), shown in app.py's warnings expander and printed by pipeline.py (first slice of the
  audit's B1).
- **Generalization gate (single-inference dual-scoring — score the same OCR output with rules
  monkeypatched off vs on, so Surya run-variance can't confound it):**
  - **Part A (identity):** all **30/30 synthetic images METRIC-IDENTICAL**, 0 stage-4 warnings —
    the rules never fire outside the bulletin domain. Anti-overfit contract holds.
  - **Part B (lift, surya_kiri + full preprocess vs §2.33's raw-OCR baseline):**

    | doc | Cell_Acc | Recall | Table_CER | Numeric_Acc |
    |---|---|---|---|---|
    | 09.06.26 before → after | 0.173 → **0.904** | 0.762 → **0.932** | 0.086 → **0.037** | 0.178 → **0.953** |
    | 15.06.26 before → after | 0.159 → **0.904** | 0.757 → **0.922** | 0.081 → **0.036** | 0.142 → **0.945** |

    The Recall lift (+0.17) matches §2.33's taxonomy arithmetic (riel ≈52% + percent slips ≈18% of
    misses). **Honest read of the Cell_Acc jump:** 0.17→0.90 is NOT pure recognition gain — with the
    unit column fixed, whole rows now match GT exactly, so difflib's row alignment snaps into place
    and the §2.33 +1-header alignment artifact dissolves; the corrected numbers converge on the
    offset-corrected §2.33 values (NumAcc 0.953/0.945 ≈ §2.33's 0.953/0.946), which cross-validates
    both measurements. Malformed flag fired on exactly the 2 pattern-matching digit-duplication cells
    in 09.06 (`8333%`, `-13333%`; §2.33's third case doesn't match the conservative patterns — accepted)
    and 0 false positives on 15.06.
- **Per-cell confidence view (app.py).** Each exported table with any per-cell confidence gets a
  collapsed read-only "🔍 Confidence view": cells tinted red (< `CELL_CONF_LOW` 0.80) / amber
  (0.80–0.95 `CELL_CONF_MID`) per the §2.33 calibration; legend states the ៛ caveat (systematic glyph
  errors can be high-confidence — tinting flags likely errors, untinted is not a guarantee). Tables
  always render without it (never gate display on optional data); the editable grid stays the single
  export source. Malformed-flagged cells surface red here automatically. Image-space heatmap deferred
  (needs cell polygons retained through `_build_table_from_grid` — engine change).
- **UI clarity rider:** engine-picker caption now states that surya_kiri reads cells from an internal
  deskew-only image (§2.31), so users must not hand-disable photometric preprocessing for it.
- Modules: `postprocess.py` (rules, flag, warnings sink), `models.py` (`PostprocessResult.warnings`),
  `model_config.py` (`CELL_CONF_LOW/MID`), `app.py` (confidence view, combined warnings, captions),
  `pipeline.py` (Stage-4 WARNING lines), `CONTEXT.md`; `tests/test_postprocess.py` (+9; 479 total).
- **NEXT:** smart preprocessing suggestions on upload (queued, separate plan); Kiri fine-tune (§2.29,
  now with §2.33/§2.34 defining the training emphasis: ៛, subscripts, digit-duplication).

---

### 2.35 Three user-observed defects closed — pipe noise, dot-drop percents, foreign-script garbage (2026-07-09)

User reported three residual defects; each was verified against the §2.33 taxonomy dumps (no new
runs) before fixing, and the fixes are deterministic + benchmark-gated like §2.34.

- **Observation 1 — empty-cell noise.** 19/99 (09.06) and 13/99 (15.06) empty GT cells carried junk,
  **pipe-dominated** (`|` 15×/10× — Kiri reading the cell's border line). Invisible to Recall (which
  only scores non-empty GT cells). Fix: `_strip_cell_noise` empties a cell whose text is only
  gridline chars **and contains a `|`** (conservative — a bare `-`/`—` may be a legit "no data"
  marker elsewhere, so it survives). New eval metric **`Empty_Cell_Precision`** (fraction of empty GT
  cells left empty, `None` when GT has no empties) makes this visible to the harness henceforth.
- **Observation 2 — dot-dropped percents.** `-4.76%→-476%`, `2.94%→294%` (15.06): plausible-looking
  wrong values the §2.34 flag (`\d{4,}%`) missed. Fix: widened `_MALFORMED_PERCENT_RE` to
  `^[+-]?\d{2,}%$` — any ≥2-digit integer percent (these docs' %-values all carry decimals, so an
  integer form is a likely dot-drop); `5%` survives. Still a FLAG (confidence cap + warning), digits
  never rewritten.
- **Observation 3 — foreign-script garbage in the UI.** 0 in surya_kiri *cells* (Kiri's vocab is
  Khmer+Latin) — it comes from **Surya's narrative text** (§2.15 hallucinations). The existing
  `_is_foreign_script` detector only *routed to Qwen* (off by default) → did nothing. Fix:
  `_strip_foreign_scripts` deterministically removes Sinhala/Lao/Thai/Myanmar/Arabic/CJK/Kana from
  BOTH cells and page text (product constraint: Khmer/English only), one aggregated warning per
  page/table. Benefits both engines.
- **Generalization gate — decomposed (the naive "30/30 identical" bar conflates a global rule with
  domain rules, so attribute carefully):**
  - Adding domain rules + noise-strip + malformed-flag alone = **byte-identical on all 30 synthetic
    images** (proven: with the scrub isolated, the domain-config Table_CERs equal the pre-change
    baseline exactly). Anti-overfit contract holds — the riel/percent rules never fire off-domain.
  - The **global foreign-scrub** legitimately fires on 2 decorative-font synthetic images (Hanuman,
    Moul — the §2.33-worst fonts where Surya hallucinates foreign scripts), moving only Table_CER by
    ±0.04. That is correct universal garbage-removal (synthetic GT is clean Khmer, so only genuine
    hallucinations are removed), not overfitting.
  - **Real docs (surya_kiri, full preprocess) — before→after (§2.34 baseline → §2.35):**

    | doc | Cell_Acc | Recall | Table_CER | Empty_Cell_Prec |
    |---|---|---|---|---|
    | 09.06.26 | 0.173 → **0.926** | 0.762 → **0.932** | 0.086 → **0.030** | 0.586 → **0.889** |
    | 15.06.26 | 0.159 → **0.919** | 0.757 → **0.922** | 0.081 → **0.031** | 0.657 → **0.899** |

    Empty-cell precision +0.30/+0.24 (pipe fix); Table_CER now below §2.34; **all four dot-drop cells
    flagged** (09.06 `8333%`/`-13333%`, 15.06 `-476%`/`294%` — the exact cells the user reported);
    foreign scrub fired on p3 of both docs (6 chars, Surya narrative). Cell_Acc edged above §2.34 as
    the pipe cleanup let more cells match.
- **Not done (deliberate):** an engine-level ink-ratio guard for empty cells — a thin legit glyph
  ("1") has ink density near a border line, so the threshold risk outweighs the gain; empty/gridline
  negatives go to the Kiri fine-tune corpus instead (joining ៛, subscripts, dot-drops as §2.33/§2.35
  training emphases).
- Modules: `postprocess.py` (`_strip_cell_noise`, `_strip_foreign_scripts`, widened percent flag,
  cell + narrative wiring), `evaluation/evaluate_structure.py` (`Empty_Cell_Precision`),
  `run_benchmark.py` + `analyze_benchmark.py` (CSV col + `avg_empty_cell_precision` + summary),
  `CONTEXT.md`; `tests/test_{postprocess,evaluate_structure}.py` (+11; 488 total).

---

### 2.36 surya_kiri worst-case — number-heavy wide table: Surya wins decisively (2026-07-09)

User reported that surya_kiri (vs plain surya) puts content in wrong cells and mangles numbers.
Verified with a head-to-head on a genuinely different layout — `CambodiaBudgetExecutioninApr-2024.pdf`
page 3, a **17-column** born-digital budget-execution table (label + annual + % + cumulative + 12
months; number-dominated, many empty cells, mixed 2- and 5-decimal values). No GT (its text layer is
legacy-corrupt, §2.21) — judged against the page image + structural signals. Both engines via the
production path (ingest → preprocess → engine).

- **Result — plain Surya is near-perfect; surya_kiri is badly broken on this doc.**

  | row (page truth) | Surya | Surya + Kiri |
  |---|---|---|
  | `29,199.60 \| 31.16% \| 9,099.14 \| 1,859.82 \| …` | reproduced exactly | `2219960 \| 311ខេ \| -90991 \| -185922 \| …` (commas/decimals gone, Khmer glyph in `31.16%`, phantom `\|[]` in empty cols) |
  | `263.13 \| 26.84% \| 70.63 \| 27.39622 \| 9.39058 \| …` | reproduced exactly incl. 5-decimals | `26313 \| 2684 \| 7063 \| 2739622 \| 939058 \| …` (**every decimal point dropped → values 100–1000× wrong**) |

  Structural: Surya = clean 8-filled-cols/row (correct); surya_kiri = 10–15 filled/row, ragged —
  content bleeding across columns + phantom cells. Surya's grid is 18 cols consistent; surya_kiri 16
  cols, mangled header.
- **Why.** (1) Structure/recognition split: Surya's VLM reads structure+text jointly with page
  context; surya_kiri makes TableRecPredictor segment a 17-col grid first (far harder than ARDB's 9),
  then Kiri reads each tiny crop blind → segmentation slips scatter content. (2) Kiri is a
  Khmer-optimized recognizer fed pure-number cells → drops decimals, injects Khmer glyphs into numbers.
- **Benchmark blind spot (the honest correction to §2.30).** Our entire eval is the ARDB bulletin =
  Kiri's BEST case (Khmer-heavy, narrow, riel units). This budget table is Kiri's WORST case
  (wide, number-dominated). So §2.30's "surya_kiri modest win" holds **only for ARDB-like docs**; it is
  actively harmful on number-heavy/wide tables. surya_kiri is a **specialist**, not a general upgrade.
  Directly vindicates the user's "don't overfit to ARDB" concern — a second real layout flipped the
  verdict, exactly as eval/README §5's "raw ranking can invert" warning predicts.
- **Actions.** (a) app.py engine picker relabelled ("specialist: Khmer-text-heavy tables") + guidance
  captions steering number-heavy/wide docs to Surya (Surya stays default). (b) This entry. (c) Design
  sketch below.
- **Design sketch — "Surya-structure + Kiri-text" variant (DEFERRED, not built).** The root fault is
  handing structure to TableRecPredictor. Alternative: take the cell GRID from Surya's VLM HTML (its
  placement is reliable — §2.36 shows it perfect here), then replace only the TEXT of cells that are
  *predominantly Khmer* with Kiri's read; leave numeric/Latin cells as Surya read them. Needs: run
  Surya normally (structure+text), classify each cell (Khmer-ratio threshold), re-OCR only Khmer cells
  with Kiri by cropping the cell's Surya bbox. Open risks: Surya's HTML cells carry bboxes? (verify);
  extra Kiri passes cost; a cell-classification threshold to gate. Only pursue after the eval set has
  ≥1 number-heavy doc with GT so the variant is measurable — otherwise we repeat the ARDB overfit.
- **Reinforces:** Kiri fine-tune (§2.29) must include number cells + decimal points; and the eval set
  urgently needs a non-ARDB layout with table GT (this budget doc is the obvious candidate — hand-label
  one dense page).

---

### 2.37 First non-ARDB table GT + variance-aware head-to-head — a clean structure/recognition split (2026-07-09)

Built the eval set's **first non-ARDB table ground truth** to test the §2.36 finding fairly, then ran a
multi-run head-to-head. Result decomposes the two engines into mirror-image strengths and, in passing,
corrects the §2.36 variant plan.

- **The GT (CambodiaBudget p3, 35×16, `eval/datasets/real/`, gitignored → stays local).** Provenance,
  deliberately unbiased: **numbers + grid structure from the PDF text layer via PyMuPDF
  `find_tables()`** (Latin digits are clean even though the Khmer is legacy-font mojibake — confirms
  §2.21, and shows the caveat is "corrupts Khmer GT, not numbers"); **Khmer label column + header +
  title transcribed from the 300-DPI image** (cross-checked vs Surya, then **user-verified**), NOT from
  any OCR engine, so the Khmer column can't bias surya-vs-kiri. Empty May–Dec month columns kept to
  detect phantom cells.
- **Variance-aware scores (production path; Surya ×4, surya_kiri ×2):**

  | engine | Recall (mean / min–max) | Numeric_Cell_Accuracy | pred cols |
  |---|---|---|---|
  | surya | **0.890** / 0.857–0.932 | **bimodal** {0.000, 0.009, 0.982, 1.000} | 14 / 17 / 18 / 19 |
  | surya_kiri | 0.168 (deterministic) | 0.122 (deterministic) | 16 / 16 |

- **Mirror-image diagnosis.**
  - **Surya = excellent recognition, unstable structure.** Content Recall ~0.89 EVERY run (it reads the
    numbers, always), but the VLM emits a different column count each run, so NumAcc is **bimodal**:
    ~1.0 when the grid aligns (18–19 cols → 16 after title-strip), ~0 when it doesn't — the numbers are
    read correctly but land in misaligned columns (Recall stays 0.89, proving displacement not
    misreading). A separate probe saw column counts 14–21 and one run detecting **0 tables** (transient
    total failure). This is §2.28's "large Surya variance," severe on a wide 16-col table.
  - **surya_kiri = perfect stable structure, terrible recognition.** Dead-deterministic 34×16
    (TableRecPredictor nails the column count), but Recall 0.168 / NumAcc 0.122 — Kiri genuinely mangles
    numbers (drops decimals, injects Khmer glyphs, §2.36). Its content is mostly WRONG, not just
    misplaced.
- **User complaint confirmed with numbers.** On number-heavy tables Surya's recognition beats Kiri's
  decisively (Recall 0.89 vs 0.17; aligned NumAcc up to 1.0 vs 0.12) — even Surya's worst run reads the
  numbers; Kiri's best run doesn't. BUT "just use Surya" is not a clean win: its structure variance
  (half the runs misalign) is a real reliability problem our single-doc ARDB eval never exposed.
- **Corrects the §2.36 variant plan.** §2.36 assumed "Surya-structure + Kiri-text" — but here **Surya's
  STRUCTURE is the unstable part**, while TableRecPredictor's (what surya_kiri uses) is rock-solid. The
  genuinely promising combination is closer to the OPPOSITE: **TableRecPredictor structure + Surya
  recognition** — hard, because Surya recognizes whole tables, not per-cell. Cheap interim idea worth
  testing: run Surya N× and take the **modal/best column structure** to damp the variance.
- **Metric lessons (both reinforce earlier notes).** Position-insensitive Recall stayed 0.89 while
  NumAcc hit 0.0 → Recall completely hid misplacement; and on this sparse table Cell_Accuracy is
  inflated by empty-cell agreement (~0.53 for surya_kiri despite 0.12 numeric) — **Recall + NumAcc are
  the honest signals on wide/sparse tables, not Cell_Accuracy.**
- Artifacts: `eval/datasets/real/CambodiaBudgetExecutioninApr-2024_p3.{png,_ground_truth.json}` (local
  only). NEXT (user-directed): re-scope the structure/recognition combination around this finding.

### 2.38 Final-month plan kickoff + continual-learning survey memo (2026-07-09)

**Plan approved & started.** The 4-week final-month strategy (dataset factory → layout fine-tune
Track A / Kiri fine-tune Track B / VLM SFT Track C exploratory → report) is approved; full text lives
outside the repo (planning doc). Confirmed with mentor: **HF dataset repo is PRIVATE-first**;
**VLM fine-tune target is 4B-class on free Colab T4** (8B needs ~24GB — dropped unless paid compute
appears). Week-1 artifacts this entry covers: `scripts/collect_documents.py` (batch-download + corpus
classification vs the ≥40-doc/≥100-page target; adds a `khmer_layer_suspect` flag catching
Khmer-block-codepoint legacy mojibake like CambodiaBudget that `inspect_pdf` alone misses — legacy
embedded fonts OR >3% invalid Khmer ordering at token starts) and the memo below.

**Continual-learning survey memo (mentor directive: "look into only" — nothing here is built).**
What "the pipeline retrains itself monthly" would actually require, and what we deliver instead:

1. **Data versioning & provenance.** Every training example must be traceable to (source PDF, page,
   generation method, human-corrected-or-not). Minimum viable: HF dataset revisions (git-backed) +
   the datacard per version — which the Week-2 factory already produces. Full solution (DVC/lakeFS,
   automated lineage) is infrastructure we do not need at ~100 pages/month.
2. **Drift detection.** Retraining monthly is pointless without knowing whether the document
   distribution moved. Cheap proxy signals we already emit: per-cell confidence distributions,
   Stage-4 warning rates (malformed-number flags, foreign-script scrubs, empty-cell noise), and
   engine disagreement on sampled docs. A monthly dashboard of these ≈ drift monitoring for free;
   a real system would add embedding-space drift tests (not built).
3. **The retrain loop itself.** Pseudo-label new month's PDFs → human correction in Roboflow
   (~2–4 h/100 pages) → retrain (YOLO-s: 1–2 h T4; Kiri CTC: hours) → **gated A/B on the frozen GT
   eval set before swap** (the §2.23/2.24 gate pattern — the eval set is the regression contract).
   This is a *procedure*, not infrastructure, and it is what we deliver: the Week-4
   **monthly-retrain runbook** IS the mentor's "training every month," human-in-the-loop.
4. **What true continual learning would add** (and why we say no for now): automated
   retraining triggers, catastrophic-forgetting mitigation (replay buffers / EWC) when fine-tuning
   on each month's data, canary deployment with automatic rollback, and label-quality QA at scale.
   Each is a project on its own; at GDDE's volume the human-gated monthly runbook gives ~all the
   benefit at ~none of the risk. Revisit only if volume grows 10×+ or labeling moves to a team.

### 2.39 Track B — Kiri CTC fine-tune: GO on all gates (2026-07-13)

**The month's first fine-tuned model, and it clears both gates emphatically.** Pipeline:
`experiments/kiri_finetune/build_trainset.py` (35,324 train / 2,467 val = 16.7k real factory cell
crops referenced in place + 15k `seanghay/khmer-hanuman-100k` + 3k Playwright targeted synthetic
[riel units / decimal percents / long decimals / Khmer digits, 3 vendored fonts, varied bg] + 800
gridline-only empty negatives) → `train_kiri.py` (full-model CTC fine-tune from the pinned
checkpoint; 8 epochs, AdamW 1e-4 cosine, batch 32, ~7 min/epoch **locally on the M4 Pro** with
`PYTORCH_ENABLE_MPS_FALLBACK=1` — no cloud GPU needed).

- **Gate 1 — per-error-class val (real held-out docs + targeted synthetic):** baseline reproduced
  the §2.33 taxonomy exactly (riel acc **0.026**, decimal-percent acc **0.055**, overall CER 0.496)
  → after fine-tune: riel **1.00**, decimal-percent **1.00**, empty **1.00**, overall CER ~0.000.
  Val saturates by epoch 5 — expected for a single-template domain; treated as necessary-not-
  sufficient and NOT the headline.
- **Gate 2 — end-to-end A/B, production path, all 7 verified GT pages** (`gate_ab.py`, stock vs
  `KHMER_KIRI_WEIGHTS=run1`): **every metric improved on every page, zero regressions.** ARDB
  pages: Recall 0.72–0.79 → **0.92–0.98**, worst-page NumAcc 0.09/0.21 → **0.97**, CER →
  **0.005–0.065**. Structure dims identical stock vs tuned (TableRecPredictor untouched — the
  change is recognition-only, as designed).
- **Transfer to the never-seen budget table (p3):** Recall 0.168→0.246, NumAcc 0.122→**0.279**,
  CER 0.353→0.256. Real but partial — fine-tuned surya_kiri is still far below Surya (Recall ~0.89)
  on wide number-heavy docs, so the §2.36 engine guidance stands unchanged. The gain came from
  number/decimal reading transferring; the remaining gap is content-domain, not glyph-level.
- **Consequences:** (a) the ៛-glyph, dot-drop, and pipe-noise defects are now fixed AT THE MODEL —
  the Stage-4 domain rules become redundant-but-harmless on ARDB docs (kept: they're gated and
  still guard other engines/docs); (b) surya_kiri with fine-tuned weights is now the strongest
  engine on ARDB-template docs end-to-end; (c) the recipe is the monthly-retrain runbook's core:
  harvest new month → rebuild trainset → 1h local train → this gate script.
- Reproduce: `build_trainset.py --out trainset` → `train_kiri.py --data trainset --out run1
  --epochs 8` → `gate_ab.py --tag <cfg>` ×2. Artifacts local (gitignored): trainset/, run1/
  (model.safetensors + vocab.json, loadable via `KHMER_KIRI_WEIGHTS=experiments/kiri_finetune/run1`),
  gate_{baseline,finetuned}.json. NEXT: decide default-weights policy for the app (env-var opt-in
  vs bundled), Track A A/B when weights arrive, Track C decision.

### 2.40 surya_kiri column-spans fixed — TableRec grid + SLANet proposals + pixel confirmation (2026-07-13)

**User-reported bug (screenshot, ARDB 15.06.26 p1):** column-spanning cells split mid-text —
merged header "14-06-26" extracted as "14" | "6-26", full-width section header chopped across
columns. **Root cause proven:** surya_kiri's structure comes from Surya's simple-path
`TableRecPredictor`, whose own schema documents that cells are *row×col geometric
intersections with no spanning info* — merged cells are fragmented BY CONSTRUCTION, and Kiri
then recognizes crops that physically contain half the text. (Continues the §2.30 known
limitation and the §2.36/§2.37 structure-source arc.)

- **Failed candidate 1 — structure wholly from SLANet** (`KHMER_KIRI_STRUCTURE=slanet`, kept
  for comparison): SLANet does emit spans and fixed the headers, but its grid lost a column on
  ARDB p1 (25×8 vs true 9) and scattered data digits (col-concentration 22→12/16 rows). §2.37's
  "TableRec structure is the stable part" held.
- **Failed candidate 2 — SLANet logical spans over TableRec grid:** SLANet *under-reports*
  spans when its grid drops a column (the 2nd date header came back `col_span=1` while its
  PHYSICAL box covered two TableRec columns at 100%/92%); conversely honoring its row-span
  blocks ate real data cells (col-4 digit rows 22→16). Logical span flags are not trustworthy.
- **Shipped — `merged` (now default):** TableRec unit grid stays the base; ALL SLANet cell
  boxes act as merge PROPOSALS (physical geometry only); a proposal is accepted only for
  same-row, consecutive-column units with **positive pixel evidence of openness** —
  `_has_vertical_separator` scans center-of-A→center-of-B for an x-column whose strong-gradient
  run is contiguous over ≥45% of the shared height. Measured populations: genuine merged cells
  (text strokes) ≤0.31; real boundaries 0.51 (p3's text-broken ល.រ rule, the one false merge
  the first threshold let through) to 1.00 (clean fill gaps). Unmeasurable band ⇒ treat as
  separated (false merge eats data; missed merge = status-quo split text). Merged cells carry
  optional `row_span`/`col_span` (new optional `Cell` fields in `models.py`).
- **Eval gate (8 GT pages incl. the non-ARDB budget doc): zero regressions, CER improved on 3
  pages** (09.06 p1 0.059→0.057, 09.06 p3 0.022→0.020, 15.06 p1 0.059→0.056 — the intact
  header dates), all other metrics identical; budget p3 byte-identical (no ARDB overfit
  leakage). Probe on the user's page: both dates intact, span metadata correct, digit-column
  concentration identical to pure TableRec. CellAcc did NOT jump on p1: the §2.30 GT convention
  (one logical header row vs two physical) still dominates that metric — expected.
- **Honest limits:** (a) genuine ROW-spans still split per row (deliberately rejected —
  data-eating); (b) the 0.45/0.31/0.51 thresholds are calibrated on ARDB pages — wide margins,
  and the failure asymmetry is safe (missed merge degrades to old behaviour), but new document
  families should re-check via the probe scripts; (c) ornate-header recognition quality is a
  Kiri model issue (§2.39 weights help), independent of spans.
- Modules: `engines/surya_kiri_engine.py` (`_kiri_structure`, `_merge_spans`,
  `_has_vertical_separator`, normalized structure records), `models.py` (optional
  `row_span`/`col_span` on `Cell`), `tests/test_surya_kiri_engine.py` (+16, 577 total),
  `CONTEXT.md` (env knob). Runs: `eval/runs/spanfix_{tablerec,merged2}`. Related: same-day
  rowband col_id placement fix in `hybrid_engine.py` (the OTHER hybrid engine).
- **POSTSCRIPT (same day) — default REVERTED to `tablerec`; `merged` demoted to opt-in.**
  Within hours of the default flip, the user's production UI run (same 15.06.26 p1 doc,
  through the app's preprocessing path) produced a **data-cell false merge the eval+probes
  never showed**: row ២'s ល.រ ID merged into the product-name cell ("២សាច់គោ…"). Diagnosis
  class: the separator's contiguous-run coverage at that row/rendering fell below the 0.45
  threshold — confirming the margin between text strokes (~0.31) and broken gridlines (~0.51)
  is **too narrow to trust across renderings** (probes used raw eval PNGs; the app path
  renders/deskews differently). Decision rule applied: **data integrity > header cosmetics**
  — a split header is a visible, manually-repairable annoyance; a silent ID-into-name merge
  corrupts the export. `merged` remains available via `KHMER_KIRI_STRUCTURE=merged` for
  experimentation. **Lessons:** (1) an eval gate over 8 pages at one rendering is necessary
  but NOT sufficient for a threshold-based pixel heuristic — production-path variation moved
  cases across the boundary; (2) the §2.30 instinct (no pixel/text heuristics for header
  repair) was right for a second time; (3) the durable span fix is upstream — a structure
  model that natively emits spans on our documents (Track A layout fine-tune / future
  TableRec successor), not post-hoc merging.

### 2.41 New engine `surya_kiri_vlm` — Surya-VLM structure+text, Kiri re-reads Khmer cells (2026-07-13)

**The §2.36 "Surya-structure + Kiri-text" variant, finally built** — enabled by §2.39 (fine-tuned
Kiri worth re-reading with) and shaped by §2.40's lesson (discrete gates, not pixel thresholds).
User's insight prompted it: plain Surya's span-correct structure lives in its table VLM (joint
structure+text HTML, emits colspan) — the exact component surya_kiri skips.

- **Design (safety contract: the floor is plain Surya).** Run plain Surya IN FULL (VLM included);
  for tables containing Khmer-heavy cells (`_khmer_ratio ≥ 0.5`), run TableRecPredictor on the
  geometric-only crop; **gate: TableRec grid shape == VLM grid shape (exact integers)**; on pass,
  batch-Kiri ONLY the Khmer-heavy cells (colspan anchors get union crops — VLM colspan now carried
  as optional `col_span` on plain-Surya cells via additive `_parse_html_table_with_spans`), replace
  text only at Kiri conf ≥ 0.5 and non-empty. Every fallback keeps Surya's text; re-read cells
  carry `confidence` (UI shows exactly what Kiri touched). +2px crop pad for Khmer
  ascenders/descenders. Numbers/Latin always stay Surya (§2.36).
- **Eval (8 GT pages, raw-render path):** floor verified — never below plain Surya anywhere;
  budget p3 byte-identical (0.974). **On ARDB p3 pages it is the best engine of all three:**
  CellAcc 0.981/0.986 (surya 0.866/0.870, surya_kiri 0.977/0.981), NumAcc 1.000 (kiri 0.992),
  CER 0.018/0.017 (best). On p1 the gate correctly refused (VLM emitted 25×11 vs true 9 — §2.37's
  VLM column instability, now seen on ARDB too) → identical to plain Surya. On p2 the gate passed
  and Kiri lifted Recall 0.79→0.91, but Surya's own misplacement kept CellAcc low (0.17) — Kiri
  can't fix what Surya misplaces.
- **Honest verdict.** `surya_kiri` (with §2.39 weights) remains the strongest ARDB tool overall
  (wins p1/p2 decisively); `surya_kiri_vlm` is the best choice when Surya's structure holds and is
  never worse than Surya — shipped as a third UI-selectable option, no default change. CAVEAT: the
  eval is the raw-render path; the app preprocesses first, where Surya's structure may behave
  better than these p1/p2 numbers suggest (user reports good Surya structure in the app).
- Modules: `engines/surya_kiri_vlm_engine.py` (new), `engines/surya.py`
  (`_parse_html_table_with_spans` + col_span attach — additive), `engine_registry.py`,
  `webapp/main.py` + `app.py` (picker entries), `tests/test_surya_kiri_vlm_engine.py` (+12) and
  parser/col_span tests in `tests/test_surya.py` (+3); 592 tests. Runs:
  `eval/runs/spanfix_{surya,vlmkiri}`.

### 2.42 React review workspace — new primary UI, analyst-first (2026-07-14/15)

The NiceGUI UI (§ UI arcs) proved the workflow but capped the review UX; user asked for a React
frontend designed "from a data analyst's viewpoint". Full plan (feature audit, residency tiers,
cut list) in `~/.claude/plans/can-you-help-me-proud-quasar.md`; built P0→P5 with two user gates
(P0.5 clickable-mockup sign-off; post-P2 usability check — both passed).

- **Architecture: one process, three surfaces.** `nicegui.app` IS a FastAPI subclass, so a REST
  layer (`webapp/api.py`) rides the existing process — the multi-GB models load once. React
  (Vite+TS+Tailwind+AG Grid+TanStack Query, `frontend/`) is served at `/app` from `frontend/dist`;
  NiceGUI stays at `/` as fallback; `app.py` is legacy. Server state moved to a process-global
  registry (`webapp/registry.py`) → the React app is **refresh-safe** (reload keeps queue/results/
  edits; deliberate semantics change: closing the tab no longer cancels — ■ Stop does, page-
  granularly, and the run task releases the global GPU `run_lock` in a `finally`, so cancel/crash
  can never wedge the registry; second concurrent run → 409).
- **Analyst-first design (the sign-off decisions):** 3-zone workspace (queue | zoom/pan page |
  tables); ONE morphing primary action (Upload → Run → Export); Issues-first triage (resident
  "Issues (N)" badge → worst-first low-conf list, click or `n`/`p` jumps to the exact cell across
  pages); always-on calibrated confidence tints (§2.33 buckets); contextual per-table toolbar
  (undo/redo/diff/reset/+row/csv/✓ verify) only on the focused table; summoned tier: settings
  drawer, Ctrl-F find/replace, `?` overlay. Khmer: bundled Noto Sans Khmer (repo `fonts/`, OFL),
  line-height 1.9 everywhere incl. AG Grid cells, A−/A+ size control.
- **Trust plumbing:** two-way table↔image linking (table-level — pipeline has no per-cell
  geometry); edited-vs-original diff; ✓ verify per table with queue progress ("n/m verified");
  staleness banner (form vs `last_run_settings`); warnings link to their page; last-used
  engine/settings remembered silently (localStorage, no presets UI). Batch: client-side
  sequential "Run all" (409-safe), "Export all" zip (`{stem}/` per doc).
- **Two real-world bugs the live smokes caught** (unit tests alone missed both): (1) Khmer
  filenames crash `Content-Disposition` (headers are latin-1) → RFC 5987 `filename*=UTF-8''…`
  with ASCII fallback, regression-tested; (2) NiceGUI intercepts `HTTPException` with HTML error
  pages → custom `ApiError` + JSON handler for all API errors.
- **Verified:** 625 tests (37 new in `tests/test_webapp_api.py`, TDD), `npm run build` type-clean,
  and live end-to-end on the real ARDB 15.06.26 PDF: upload → run (surya + surya_kiri) → lowconf
  triage (50 issues) → cell edit → diff baseline kept → zip/xlsx/json/txt/csv/export-all → cancel
  mid-OCR → lock released (immediate new run accepted). User usability check: editing trust ✓,
  linking ✓; "more errors than NiceGUI" resolved as always-on tints exposing what the old editable
  grid never colored.
- Deferred to backlog: dark mode (a half-themed UI reads worse than a consistent light one),
  paste-from-Excel, numeric sanity checks, EN/KH label toggle, disk-persisted sessions.

### 2.56 Track A — layout detector: GO on stock Apache-2.0 PP-DocLayout, no training (2026-07-19)

**The directive, reshaped by two constraints.** The mentor asked for a layout detector fine-tuned on
our documents. Two things changed the shape:

1. **Licence.** DocLayout-YOLO (the obvious pick, and what `experiments/layout_yolo/` was built for) is
   an Ultralytics derivative; its exports carry **AGPL-3.0**, whose §13 network clause reaches software
   served over a web UI — which is exactly this deliverable, for a government department. Every other
   dependency we ship is Apache/MIT/BSD, so it would have been the only copyleft component in the tree.
   Switched the default layout model to **`pp_doc_layoutv2`** (PaddlePaddle lineage, **Apache-2.0**),
   already bundled with `rapid_layout` and working off the shelf. `doclayout_docstructbench` stays
   reachable via `KHMER_LAYOUT_MODEL` for measurement only — never shipped. This also removed latent
   AGPL exposure the `hybrid` engine already carried.
2. **No local training.** DocLayout fine-tuning (`imgsz=1024`) froze the 24GB Mac (PyTorch + MLX
   co-resident). Training was moved to a Colab notebook — but see the result below: it wasn't needed.

**A bug the gate caught in my own integration.** The first wiring hooked the detector into
`surya.py::_process_page` and claimed "one integration point serves all three engines." False:
`surya_kiri` runs its OWN layout pass on the geometric-only frame and never sees `run_surya`'s boxes,
so `KHMER_LAYOUT_WEIGHTS` silently no-opped on the ARDB production engine. The gate proved it —
`surya_kiri/layout_on` was byte-identical to `layout_off` on all 7 pages. Fixed via
`surya_kiri_engine._table_regions()`; both hooks now gate on the existing `KHMER_LAYOUT_DETECTOR`
convention.

**Result — stock PP, zero training, `experiments/layout_yolo/gate.py`, 3 runs/config, §2.42 aligner:**

The effect is *engine-dependent*, which is itself the finding:
- **`surya`** (feeds the layout box to its table-HTML VLM): **catastrophic** — a fragmented region
  wrecks the parse (dims `23x9`→`83x5`/`67x5`, numacc collapses). This is §2.24's failure recurring.
- **`surya_kiri`** (only needs a good table *region*; TableRec rebuilds structure): **better on 5/6
  ARDB pages**, mean numeric-cell-accuracy **0.977 → 0.991 (+1.4pp)**, every config stable across 3
  runs. On 15.06 p3 PP recovered a row Surya was dropping (dims 24×9 → **25×9**, numacc → 1.000).

§2.24's "off-the-shelf detectors lose" was measured on ONE consumer (Surya's VLM) and does **not**
generalise: the same detector helps or destroys depending on what consumes the box.

**GO — stock PP-DocLayout as a documented option for the ARDB `surya_kiri` path.** Honest scope: this
is an *accuracy* win (+1.4pp), NOT a stability win — `surya_kiri`'s TableRec structure was already
deterministic; the §2.37 instability was plain `surya`'s VLM layout, a different engine. Small but
consistent, licence-clean, and free. **Fine-tuning is unnecessary for the GO** — deferred as optional
upside (would only help this single ARDB template, on top of an already-strong stock baseline).
Artifacts: `experiments/layout_yolo/gate_pp_stock{,_x3}.json`. Integration is model-agnostic, so the
Colab notebook + ONNX export tooling remain valid if a PP fine-tune is ever wanted.

### 2.63 Stopped state machine + telemetry chips + taxonomy + header progress line (2026-07-20)

- **Native 'stopped' state (TDD, 710 tests)**: `_doc_summary` maps a cancelled
  `run_error` to status `stopped` (neutral slate dot + localized label in the rail,
  `DocStatus` union extended) — never failure-red for a user-requested stop. runAll now
  includes stopped docs as re-runnable. Closed the §2.56 cancel race for real:
  runner no longer clears `cancel_requested` at run start (reset_run's fresh progress
  already guarantees a clean flag; the old clear swallowed cancels landing in the
  reset→start window). Stop button debounce already present (§2.56) — verified.
- **Telemetry relocation**: the scan-check banner ROW is gone (vertical space back to
  the document); the read-only chips now live in the canvas strips next to
  [Single][Grid] — pre-run preview strip, post-run PageViewer strip (new `telemetry`
  slot prop). Same click→open-drawer→scroll+pulse behavior; labels stay the localized
  `tele_on/off` forms (the brief's English-only 'Contrast Mapping: Optimized' style
  would break km — declined). scanNotice state/effects and undoScanCheck removed.
- **Taxonomy**: 'Page cleanup' → 'Preprocessing' (km ការរៀបចំមុនដំណើរការ); 'Output' →
  'Export settings' (km ការកំណត់នាំចេញ) and the section moved LAST (after AI
  correction) as the closing export block; section rhythm unchanged
  (`mt-5 border-t border-line-strong/30 pt-5`), titles `mb-2.5`.
- **Header progress line**: ETA strings + the 56px mini progressbar removed from
  RunControls (stage label + page x/y remain, aria-live kept); new 2px primary line on
  the header's bottom edge driven by `useSmoothProgress` — stage-milestone ranges
  (ingest 2–10, preprocess 10–30, OCR 30–80 riding the real page fraction, tidy 80–95,
  export 95–99), value approaches each ceiling asymptotically (250ms tick + 500ms width
  transition, monotonic, `motion-reduce:transition-none`), completes to 100 and clears
  on finish.

Server restarted (registry cleared). 710 pytest, tsc/build/detector clean; bundle
`index-CVYaiZaA.js`.

### 2.62 Hybrid canvas: single ⇄ grid overview + visual page-range selection (2026-07-20)

The §2.61 preview completed into a full "pick pages by looking" workflow.

- **Backend (TDD, 708 tests)**: `Settings` gained `page_scope: 'list'` + `page_list`
  (1-based) — disjoint grid selections were unrepresentable before. `page_indices`
  sorts/dedupes/clamps to 0-based (all-clamped-away or empty → defensively all pages);
  `settings_key` gets a `list_…` part so a changed selection correctly stales results.
- **State sync (the design's core)**: NO parallel selection state — the grid's checked
  set is DERIVED from `runSettings` (`pagesFromSettings`) and toggles encode BACK to
  the minimal scope (`encodePages`: full→all, one→single, contiguous→range via strict
  numeric sort + `max−min+1===size`, else→list). Drawer reflects instantly; its scope
  select shows "Selected pages ({n})" while list mode drives. Unchecking the last page
  is a no-op (zero-page runs are meaningless).
- **UI**: new `PageGrid.tsx` — responsive 2/3-col thumbnail grid (lazy PNGs), page
  chips, hover ring, primary border when selected; corner checkbox on a blurred chip is
  its own tab stop (`onChange` + propagation stops — Space toggles, never opens);
  card click view-transitions back to single view on that page. `ViewToggle` segmented
  control ([Single][Grid]) in both the pre-run preview strip and PageViewer's strip
  (new optional `view`/`onViewChange` props — renamed on destructure; the pan/zoom
  state already owned `view`). Post-run grid deliberately uses RAW previews: after a
  ranged run processed images exist only for extracted pages, so raw is the only honest
  full-document overview. 5 i18n keys en+km.

Server restarted (registry cleared). 708 pytest, tsc/build/detector clean; bundle
`index-QUqBOG3l.js`. §2.61 polish items re-verified in code (unchanged).

### 2.61 Pre-run page preview + four micro-fixes (2026-07-20)

**Pre-run viewing (workflow fix)**: analysts could not see a document before running,
so choosing a page range was blind. New backend endpoint
`GET /api/documents/{id}/preview/{n}` (api.py) — lazily `ingest`s the upload once,
caches on `doc.ingest_result` (a later run simply replaces it), serves PNG; 422 on
unreadable files, 404 past the end. TDD: 2 new tests (lazy-cache single-ingest
assertion, 422 path) → 706 total. Frontend: the "Ready — press Run" placeholder is
gone; a selected-but-unrun doc shows a preview sheet — h-10 page-nav strip
(‹ page x/N ›, reuses `pageIdx`), scrollable raw page image on canvas, h-10 status
footer carrying working/stopped/failed/`preview_hint` (en+km) lines.

Micro-fixes: engine radio dot `mt-[3px]` (anchors to the title line); scan-check
inactive items' Minus glyph → quiet 1.5px dot (matches the green-check language);
Pages section stacked with block labels (DPI + page scope full-width select — no more
jagged two-column row); tables-header conf legend un-truncated (`shrink-0` chain +
label "…conf — check").

Server restarted (registry cleared); live smoke: uploaded 10-page sample, preview/0
→ 200 image/png 315KB pre-run. pytest 706, tsc/build/detector clean; bundle
`index-wd7fxt55.js`.

### 2.60 SettingsDrawer: the REAL spacing culprit was <legend> (2026-07-20)

Fourth report of "titles choked against the divider" — after §2.59's classes provably
put 20px on each side of the rule. True root cause: browsers render `<legend>` inside a
`<fieldset>` specially — the legend is pulled up ONTO the fieldset's border line,
bypassing the fieldset's own padding-top. With `border-t` group wrappers, every section
title rendered straddling its divider regardless of any padding we wrote; the pt-5
landed BELOW the title instead (title far from its controls). Every prior spacing patch
was fighting UA fieldset/legend layout, not our CSS. Fix: `<fieldset>`→`<section>`,
`<legend>`→`<h3>` (SectionTitle), headings keep the a11y structure; `mb-3` anchor.
Spacing classes from §2.59 unchanged — they now actually apply. Build + tsc + detector
clean; bundle `index-C-DX83DE.js`.

### 2.59 SettingsDrawer semantic grouping fix (2026-07-20)

The spacing was semantically inverted: bottom-borders + big top-padding on sections put
the divider directly ABOVE each next title (title suffocated against the line, far from
its own controls). Restructured: scroll container owns `px-4 pt-5 pb-8`; each fieldset
is a true group wrapper — `mt-5 border-t border-line-strong/30 pt-5` with `first:` zero
— so the divider belongs to the gap BETWEEN groups; legends `mb-2.5` clamp titles to
their controls. Build + tsc + detector clean; bundle `index-C0Ukc5mZ.js`.

### 2.58 Single-box drawer geometry + telemetry bar (2026-07-19)

Root-cause session (plan mode) for the drawer artifact that survived three padding
patches: the drawer was TWO nested boxes — an animated `overflow-hidden` wrapper with
no chrome, and an inner rounded/bordered/shadowed deck. The wrapper clipped the deck's
`shadow-overlay` along the bottom (dirty gray fringe) and let canvas show through the
corner radii. Fix: **the wrapper IS the card** — it carries rounded-xl/border/
bg-surface/shadow only while open (`w-0` state chrome-less), decks deshelled to plain
full-height flex columns; applied to Settings AND Issues drawers.

Also: section legends stepped to `text-[15px]` (they were identical to row titles —
the "typography still matching" report was literal); scan-check ON/OFF pills replaced
by **read-only telemetry badges** ("Sharpen — off (auto)", `tele_on/off/tip` en+km) —
clicking one opens the Settings drawer and scrolls to + ring-pulses that flag's row
(`highlight {k,n}` prop, rowRefs map, reduced-motion instant, 1.6s pulse). One writable
surface per setting again. `pill_*` keys now unused.

Build + tsc + detector clean; bundle `index-DoEuwG4b.js`.

### 2.57 SettingsDrawer structural correction (2026-07-19)

User: padding/headers/bottom edge STILL broken. Removed the parent-selector spacing
(`divide-y` + `[&>*]:…` utilities) in favor of explicit per-section classes — every
`<fieldset>` now carries `border-b border-line px-4 pb-4 pt-6 last:border-b-0 last:pb-8`
and the scroll container is a solid `bg-surface` flex column. Section headers get real
air below each divider (pt-6), the last card can't clip the panel's bottom radius
(pb-8), and toggle-row titles stepped up to `text-sm font-semibold text-ink` (incl. the
AI row) over `mt-1 text-xs text-ink-2` descriptions — full-opacity ink-2 kept over the
requested /80 (12.5px text must hold ≥4.5:1). Build + tsc + detector clean; bundle
`index-IJnqZsfP.js`.

### 2.56 Bug pass: scan-check death, cancel UX, issue-row columns (2026-07-19)

Plan-mode diagnosis then approved fixes (frontend-only scope):

- **Scan-check autodetect death — root cause**: `upload_id = md5(content)[:12]`
  (api.py:124) means a re-uploaded file KEEPS its id; the session-lifetime
  `suggestSeenRef` then silently swallowed the scan check. Fixes: ref pruned on
  document removal; ref upgraded to `Map<id,'queued'|'run'>` so a doc seen WITH results
  that is queued again (new upload generation) speaks again — while merely re-selecting
  a queued doc doesn't re-notify; stale-closure badge filter now reads via
  `runSettingsRef`; duplicate re-add (server `setdefault` dedupe) surfaces a dismissible
  notice (`dup_doc_notice` en+km) via an upload onMutate queue snapshot.
- **Cancel**: backend was sound (lock released in finally; page-granular OCR cancel);
  the jam was presentation. RunControls already had Stopping…; added: cancelled runs no
  longer render failure copy — `wasCancelled` (single string-match point) shows a
  neutral "Extraction stopped — press r or Run" banner + center-pane message
  (`stopped_msg` en+km). Known remaining window: stages 2/4 are single io_bound calls,
  cancel bites at the next checkpoint (documented, backend change deferred).
- **Rail onSelect → `selectDoc`** (was an inline duplicate that skipped the
  `dismissedIssues` reset — §2.55 oversight).
- **Issues rows**: absolute Dismiss badge (+ magic `pr-[4.5rem]`, km-label overflow)
  replaced by true two-column flex — content `flex-1 min-w-0`, right column stacks
  conf% over the hover-revealed Dismiss badge. No overlap in en or km.
- **Settings drawer**: sections `pt-6 pb-3` + `last:pb-6` (container pb dropped);
  legends `mb-3`; scan-check block restyled to the switch-row primitive
  (`bg-rail/20 border-line-strong/30 rounded-md p-2`) and its items sorted into
  PREPROCESS_FLAGS order so each finding sits above the toggle it explains.

Build + tsc + detector clean; bundle `index-klmjNHyr.js` live.

### 2.55 Issue-dismiss goes global + settings geometry + scan-check pills (2026-07-17)

- **Dismiss is App state now**: `dismissedIssues` Set lives in App; the filtered list
  feeds the header Issues chip, drawer count, and n/p stepping identically (chip 12→11
  on dismiss). Drawer keeps only the 150ms exit animation; `onDismiss` dispatches up.
  `setIssueIdx(-1)` on dismissal since indexes renumber; set resets per document.
  Control redesigned: hover-revealed "Dismiss" badge pill (✓ + label, ok-tint hover,
  `active:scale-95`); new `dismiss_badge` key en+km.
- **Settings geometry/type**: deck now opaque `bg-surface` full-height (`h-full
  min-h-0`) — kills the gray bottom edge for good (the artifact was the translucent
  raised tint over canvas); sections breathe `pt-5 pb-2`; section legends explicit
  `text-sm font-semibold text-ink`; switch labels `font-medium`; hints `mt-1 text-xs
  text-ink-2` (kept full ink-2 rather than the requested /80 — contrast floor).
- **Scan-check ribbon**: prose sentence → control strip. Title + one micro-pill per
  suggested adjustment showing live state ("Sharpen · OFF"), click flips that flag for
  the next run and drops its Auto badge; the existing stale-settings notice then offers
  one-click re-run (deliberately NOT auto-rerunning full OCR per pill click). New
  `pill_on/pill_off/pill_toggle_tip` keys en+km; `scan_off_list` now unused.

Build + tsc + detector clean; bundle `index-DHwTLZQt.js` live.

### 2.54 Tightened canvas + dismissible issues + settings form overhaul (2026-07-17)

- **6px spatial rhythm**: workbench `p-3`→`p-1.5`; rail `mr-1.5`; split divider `w-1.5`
  (hairline `w-0.5` centered); drawer wrappers `ml-1.5` when open — more room for the
  page and tables.
- **Issues are triage-interactive**: each row grows a hover ✓ dismiss button (presentation
  state only — Sets for `leaving`/`dismissed`, 150ms `issue-out` slide keyframe matching
  the unmount timer, reduced-motion covered); the header count reflects visible rows
  instantly. Tooltip copy states the semantics honestly: "the cell stays flagged until
  fixed" (new `dismiss_issue` key en+km). n/p stepping still walks the full list — a
  dismissed row is hidden, not renumbered.
- **Settings drawer form pass**: deck root `h-full min-h-0` + scroll area `pb-6` (kills
  the dead gray band at the bottom); every switch row boxed in `bg-rail/20 rounded-md
  p-2 border border-line-strong/30` at a uniform `space-y-1.5` matrix (page cleanup,
  output, AI); engine selector rebuilt as a selection deck — option cards with radio
  dot, selected = `border-primary/60 bg-primary-soft shadow-raised`, unselected quiet
  `bg-rail/20`.

Build + tsc + detector clean; bundle `index-BgUXTBds.js` live.

### 2.53 Macro-refactor: rigid viewport shell + unified card workbench (2026-07-17)

User brief: stop micro-fixing; overhaul the layout shell. Logic untouched (hooks,
handlers, i18n, `useSplit`, grids as black boxes).

- **Rigid grid**: app root `h-screen overflow-hidden bg-canvas`; `<main>` is the padded
  workbench (`p-3 overflow-hidden`, inner wrapper removed); only the page canvas and
  grid bodies scroll. Uniform 12px card seams built explicitly (rail `mr-3`, w-3 split
  divider, drawer `ml-3` when open) instead of `gap-3`, which would leave a phantom gap
  beside the width-0 animated drawer wrappers.
- **Queue rail is now a card**: rounded/bordered/shadowed like the sheets (w-64⇄w-11
  collapse unchanged); new structural header ("Documents" — reuses `group_documents` —
  + count badge left, collapse right); Add-documents is a full-width body row; row hover
  fixed for the bg-surface card.
- **One header spec on every card** (`h-10 bg-rail/30 border-b border-line-strong/50
  px-3 justify-between whitespace-nowrap`): viewer page-nav strip, tables utility header
  (facts cluster `min-w-0 overflow-hidden`, legend `hidden md:flex`, A± `shrink-0` —
  truncation before collision), Settings header (title+subtitle inline-baseline,
  truncating), Issues header.
- **Viewer footer**: h-10 px-3; legend reduced to three dot-chips with tooltips only.

Build + tsc + detector clean; bundle `index-DshBae6u.js` live. User visual gate pending.

### 2.52 Responsive polish: docked viewer footer + strict single-line toolbars (2026-07-17)

Follow-up user brief. (1) The floating bottom pill became a **docked footer bar**
(`h-9 border-t border-line-strong/60 bg-surface`) at the foot of the viewer sheet —
moved out of the pan/zoom canvas entirely; legend compacted to tooltip swatches with
labels only ≥xl; `overflow-x-auto whitespace-nowrap` so it can never stack. (2) Tables
utility header restructured to the spec row (`h-9 bg-rail/30 border-line-strong/50`,
facts left, A± pinned right with `ml-auto shrink-0`). (3) Single-line rule enforced on
all sub-toolbars: viewer top strip and table-card toolbar switched from wrap to
sideways-scroll (`overflow-x-auto whitespace-nowrap`, toolbar `shrink-0`), all strips
now h-9. Build + tsc + detector clean; bundle `index-DnAJa6pA.js`.

### 2.51 Polish: unified viewer pill, anchored tables header, grounded toolbars (2026-07-17)

User polish brief, three fixes: (1) the viewer's floating corner legend merged INTO the
bottom-center control pill as micro-chips after a divider (shown only while the
confidence overlay is active, `lg:` and up) — one floating island instead of two stacked
bars; (2) the Tables panel utility header (facts + legend + A± size controls) anchored
with `bg-rail/40` tint + `border-b border-line-strong/40`; (3) toolbar grounding — the
viewer top strip matches the same `border-line-strong/40` seam, both at `min-h-10`, and
the master header's bottom border stepped up to `border-line-strong/60`. Build + tsc +
detector clean; bundle `index-BFR-Txv1.js` live.

### 2.50 Overdrive: ⌘K palette + sheet-depth workbench + choreography (2026-07-17)

`/impeccable overdrive`; user approved all three directions (palette, choreography, and
their own sheet-depth brief) and supplied new `ui.ts` panel primitives by hand
(`panelMainCls`/`panelFloatingCls`, 150ms ease-out baseline, shadowed primary button).

- **Sheet Depth Framework**: the workspace is now a canvas workbench (`p-3` ground);
  viewer + tables are lifted `panelMainCls` sheets (rounded-xl, border, shadow,
  overflow-hidden); the split divider became the 12px gap between sheets (hairline only
  on hover/drag). Loading pane shares the sheet geometry.
- **Drawers as elevated decks**: Settings/Issues drawers stay mounted inside width-animated
  wrappers (`w-0 ⇄ w-96/w-80`, 200ms expo-out, `motion-reduce:transition-none`,
  `aria-hidden`+`inert` when closed) so the sheets compress smoothly instead of jumping;
  deck chrome = rounded-xl `bg-raised/95 backdrop-blur-md shadow-overlay`. Fallback noted
  by user if flexGrow reflow janks: absolute overlay glide (not needed so far).
- **Engine selector de-cluttered**: big bordered cards → dense quiet radio rows
  (12px dot, `py-1.5`, selected = primary dot + `text-primary-strong`).
- **⌘K Command Palette** (`components/CommandPalette.tsx`): fuzzy subsequence scorer
  (no deps), grouped results (Documents/Pages/Issues/Actions) covering doc switch, page
  jump, issue jump, run, exports (xlsx/json/zip), settings/issues drawers, theme,
  language, engine switch, all 7 preprocess/output flag toggles; combobox/listbox ARIA,
  ↑/↓/Enter/Esc, kbd chips; opened via ⌘K/Ctrl-K, header Search button, listed in help.
  ~20 new i18n keys en+km (km pending native review).
- **Choreography**: fixed `menuCls` dead `animate-in` classes (tailwindcss-animate not
  installed) by baking `scale(0.98)` into `overlay-in`; table cards stagger in on page
  change (`sheet-in`, 25ms steps capped at 6, remount-keyed); morphing primary action
  crossfades its label (`label-fade`). All new classes in the reduced-motion block.

Build + tsc + detector clean; bundle `index-BOzvei8g.js` live. User visual gate pending.

### 2.49 Command-deck pass under rewritten PRODUCT.md (2026-07-17)

The user rewrote PRODUCT.md: personality is now a "high-agency command deck"
(Linear/Raycast), with an AI Implementation Mandate authorizing structural risks,
depth/translucency/kbd styling, and snappy micro-motion. Re-executed the §2.48
four-part critique at that register:

- **Adjustable split-pane** (the structural risk Principle 2 asks for): viewer⇄tables
  seam is now draggable (`useSplit` in App.tsx — ratio 0.2–0.8 in
  `localStorage('workspaceSplit')`, default 0.6, `flexGrow` on `basis-0` sections with
  §2.48's 320/360px floors kept). Double-click resets; divider is `role="separator"`
  with arrow-key nudge; hairline turns primary on hover/drag.
- **Depth & materials**: floating viewer pill → `bg-raised/80 backdrop-blur-md`
  (purposeful glass, now PRODUCT-authorized); Settings/Issues drawers get
  `shadow-overlay` + z-10 so they float above the workspace; header gets `shadow-raised`
  as its own layer; light `--color-canvas` darkened (0.936→0.916 L) so the
  canvas/surface/raised layering reads.
- **Kbd chips**: new `kbdCls` in ui.ts (mono 2xs bordered chip); help dialog key column
  renders real `<kbd>` chips; ⋯-menu Help row shows `?`.
- **Micro-timings**: ui.ts hover trans 100→75ms, overlay 100→90ms, drawer 140→120ms,
  backdrop 100→75ms, engine/DPI hovers 75ms. Ease-out curves + reduced-motion kept;
  switch knob (100ms) and rail collapse (150ms) unchanged — layout moves, not hovers.
- **Authoritative selection**: engine card selected border `primary/50` → full primary.

Build + tsc + detector clean; bundle `index-C4MqRjNE.js` live. User visual gate pending.

### 2.48 Layout hardening: drawer containment, toolbar clamp, snappier motion (2026-07-17)

Screenshots with the §2.47 drawer open exposed two containment regressions: the workspace
sections (`flex-[3]`/`flex-[2] min-w-0`) had no width floor, so the drawer squeezed the
tables panel into visual collapse; and the table card's no-wrap header row let the focused
toolbar overflow, clipping Reset off-screen. Fixes:

- **Containment**: viewer `min-w-[320px]`, tables panel `min-w-[360px]`, `<main>`
  `overflow-x-auto` (very narrow windows scroll instead of crushing panels); drawer keeps
  `w-96 shrink-0` but clamps to `w-80` below 1280px. (Rejected the suggested single
  `min-w-[500px]` — it would starve the sibling panel at 1440px with the drawer open.)
- **Toolbar clamp**: header row + `ml-auto` toolbar get `flex-wrap` (`gap-y-1`,
  `justify-end`) so Reset wraps to a second line instead of clipping.
- **Type rebalance**: engine cards get explicit `text-sm leading-5` labels vs
  `mt-0.5 text-xs leading-4` guidance; card `py-2.5`; same `mt-0.5` on toggle hints.
- **Snappier motion** (durations only — ease-out curves and reduced-motion kept; declined
  the "75ms linear everywhere" ask as off-register): overlay 150→100ms, drawer 200→140ms,
  backdrop 150→100ms, `ui.ts` hover trans 150→100ms, switch knob 150→100ms, rail collapse
  200→150ms. Feedback animations (verify-pop, table-flash, zoom-fly) untouched.
- **Bolder separation**: `--color-line` darkened one step (light 0.918→0.895 L, dark
  0.32→0.35); the three panel seams (rail|viewer|tables|drawers) upgraded to
  `border-line-strong` so the page's architecture reads authoritatively.

Build clean, detector clean, served bundle `index-CWS4Qz7N.js`. User visual gate pending.

### 2.47 Scan check gets a voice + Settings drawer redesign (2026-07-17)

User: auto-suggest "doesn't even pop up and doesn't pick any options"; Settings drawer "ugly and
unpolished". Diagnosis: not a wiring bug — `PreprocessConfig` defaults are all True and the two
heuristics could only suggest turning sharpen/normalise OFF, deltas-only, surfaced solely as tiny
badges inside a drawer nobody opens. The feature had no voice and half a brain.

- **Backend (TDD, 698 tests, +6):** `suggest_preprocess_settings` gains two positive signals reusing
  existing helpers — `skew_deg` (shared `_skew_angle`) and `stamp_ink_ratio` (factored
  `_stamp_ink_mask` out of `_remove_stamps`; new `_SUGGEST_STAMP_INK_RATIO = 0.002`); skew/ink use
  the per-doc MAX (one bad page matters). New backward-compatible `checks` list assesses ALL five
  toggles with stable reason keys (`tilted`/`straight`/`stamps_found`/…) the frontend localizes,
  plus an English `detail` with the measured evidence. `suggested` semantics unchanged.
- **Frontend voice:** a **scan-check notice** appears once per document in the status region
  (primary-soft, ScanSearch icon): "Scan check — turned off Sharpen · Enhance contrast for this
  document." with **Details** (opens Settings) and **Undo** (reverts deltas, clears badges); when
  nothing needed changing it says so and auto-hides after 6s. The Settings button pulses once when
  deltas apply. Localized en+km.
- **Settings drawer redesigned** (calm register, actually designed): real animated **toggle
  switches** (36×20, role="switch") replace every checkbox; **engine as radio cards** with guidance;
  **DPI segmented control**; a permanent **Scan check block** atop Page cleanup listing all five
  assessments (✓/– + localized phrase, measured detail on hover); iconed section titles, header
  subtitle, redundant footer dropped.
- Live-verified on the 09.06.26 bulletin: crisp PDF → sharpen+normalise off, all five checks
  correct (`skew 0.0°`, `ink 0.00%`, `sharpness 1402/500`, `contrast 69/60`).
- Deliberate scope cut: the broader "nothing pops off" feeling gets its own accent pass later if
  the drawer + notice don't settle it — kept out to protect the register.

---

### 2.46 Critique 33/40 + auto-suggest integration review + shaped fixes (2026-07-17)

Dual-agent `/impeccable critique` re-scored the workspace **33/40** (29 → 32 → 33), detector fully
clean for the first time. Reviewed the parallel session's **auto-suggest** work (`/suggest` endpoint,
Auto badges w/ rationale in SettingsDrawer, `validate.py` failure-mode taxonomy in `/lowconf`) —
found it correctly wired end-to-end (692 tests, +52). Then implemented the confirmed shape briefs +
full P1–P3 fix round (`bolder` skipped by choice — the issues were overload/legibility, not blandness):

- **Issue legibility:** the cryptic Latin badges (`sum`/`khmer`/`digits`/…) became **plain localized
  phrases** (en+km, e.g. "Row total doesn't add up") with a severity dot (danger = validator finding,
  warn = low confidence); multiple reasons join with "·".
- **Header de-crowding (~11 → 6 targets):** engine picker moved into the Settings drawer as a
  "Recognition engine" section with its guidance line; language/theme/help collapsed into one ⋯
  overflow menu. Post-results header: Notes · Issues · Re-run · primary · Settings · ⋯.
- **Mechanical fixes:** selection rect + spotlight-adjacent colors now `var(--color-primary)` (was
  hard-coded light blue — wrong salience in dark mode); `text-[10px]` → `text-2xs` (both IssuesDrawer
  and the Auto badge); last non-token border removed from ui.ts; progress bar got `role="progressbar"`
  + `aria-valuenow` and the stage line `aria-live="polite"`; floating viewer pill wraps on narrow
  canvases; triage focus re-centers after autoHeight rows settle; **`v` shortcut verifies/unverifies
  the focused table** (the loop's most-repeated action was mouse-only); remove-button aria-label
  localized.
- Snapshot `.impeccable/critique/2026-07-17T03-16-47Z__frontend-src-app-tsx.md`. Remaining known
  gaps (accepted for now): Notes "view page" regex depends on English server warning strings; server
  rationale/engine names stay English; text-block rects not keyboard stops.

---

### 2.45 UX critiques + overdrive + dark mode + Khmer UI (2026-07-16)

One arc, user-driven: four UX critiques, three approved `/impeccable overdrive` directions, and the
two big deferred items (dark mode, Khmer localization) pulled forward. A control-placement
architecture (documented in the plan) now governs where every button lives, by the scope it acts on.

- **Critique fixes:** grid cells `wrapText/autoHeight` — long Khmer values wrap, rows grow, nothing
  truncates behind a click; queue rail collapses to a 44px strip (localStorage, still a drop target)
  so image+tables get the width; viewer controls moved off the top strip into a floating
  bottom-center pill on the canvas (Fit · 100% · Cleaned⇄Original segmented · overlay select ·
  Loupe), top strip = page nav only; the bland pipeline-warnings banner became a warn-soft **Notes
  (N)** header chip opening a styled popover ("worth a look, not necessarily wrong") with per-page
  jumps.
- **Overdrive (all user-approved):** *zoom-to-evidence* — triage jumps rAF-fly the camera (380ms
  expo-out, 3× cap) to the flagged table's bbox and spotlight it (surroundings dim ~1s; per-cell
  bboxes don't exist, so the TABLE region is the target); *magnifier loupe* — 180px lens at fixed 3×
  natural pixels, pure CSS transform of a second img, hidden while panning; *View Transitions
  morphs* — help dialog and export menu morph from their triggers (`view-transition-name` swaps
  trigger↔surface so names never duplicate; Firefox/reduced-motion fall back to the overlay
  animations).
- **Dark mode:** JS resolves light/dark/system (header Sun/Moon/Monitor cycle, localStorage,
  media-query listener) and stamps the RESOLVED theme on `<html data-theme>`, so `index.css` needs
  exactly one override block. Cell tints/diff became tokens with dark-safe washes; AG Grid params
  switched to `var(--…)` strings so the grid follows; dark primary at L 0.62 keeps white button text
  ≥4.5:1; the scan image stays light (it's evidence) on the recessed dark canvas.
- **Khmer UI localization:** `frontend/src/i18n.tsx` — typed en+km dictionaries (~180 keys),
  `LangProvider`/`useT`, header EN/ខ្មែរ toggle, `[data-lang="km"]` swaps the chrome face to Noto
  Sans Khmer with raised line-heights token-level. All 8 components + App swept; stage matching
  stays on the server's English labels (display-only translation); engine names/pipeline warning
  text remain server-side English (out of scope). **User lifted the "no Khmer from me" rule for UI
  copy** (it was an OCR-GT rule — a mis-read glyph poisons evaluation; UI strings carry no such
  risk); km translations are mine, in administrative register, **pending the user's native review**.
- Build clean per block; impeccable detector 0 findings; live bundle serving. User visual gate next.

---

### 2.44 Visual redesign — the "quiet instrument" pass (2026-07-16)

The user asked to improve the frontend's look (Linear/Vercel craft bar) before Khmer localization and
dark mode. Frontend-only; no behavior, API, or layout-skeleton changes. Their calls: bundle **Inter
Variable** for chrome (`@fontsource-variable/inter`, offline); **layered-zones** light theme.

- **Diagnosis of the old look:** border-soup (every seam a `border-slate-200` line), two competing
  blues (brand `#1565c0` vs Tailwind `#2563eb` in focus/flash/selection), raw palette one-offs
  (`blue-100`, `green-100`, `red-50`, amber), no type system or `tabular-nums`, control metrics drift
  (5 icon sizes, 3 paddings), Bootstrap-era status pills, 5-pill stage stepper.
- **Token layer** (`index.css`): semantic tokens in Tailwind `@theme` — surfaces
  (`surface`/`rail`/`canvas`/`raised` — the zones separate by background step, not borders), ink ramp
  (`ink`/`ink-2`/`ink-3`), lines (`line`/`line-strong`), ONE blue + `primary-soft`, semantic status
  trios (`ok/warn/danger` + `-soft`/`-ink`), shadow scale (raised/overlay/modal). **Dark mode later is
  a variable override, zero component edits.** Body gets `tabular-nums` globally; text-xs/sm re-tuned
  to 12.5/13.5px; thin tokenized scrollbars; shared `overlay-in` motion (150ms expo-out, reduced-motion
  safe).
- **ui.ts v2:** one control height (h-7, small h-6), unified radius/border/focus/transition; new
  `dangerBtnCls`, `chipCls`, `inputCls`, `menuCls`/`menuItemCls` — zero one-off control styling
  remains (closes the critique's consistency finding).
- **Chrome:** 48px header with primary monogram mark + ink title; stage pills → one determinate 2px
  progress bar + stage label + tabular `page n/m · ETA`; queue rail loses the permanent dashed
  dropzone (whole rail is the drop target), cards flatten, status = 6px dot + neutral text, verified
  progress = 2px micro-bar; designed empty state (page-with-table SVG sketch in the product's own
  vocabulary + numbered 3-step sequence).
- **Review surfaces:** unified 40px panel headers; confidence legend merged into the tables header as
  micro swatches; viewer canvas recessed (`canvas` token) with the page floating on an overlay shadow;
  AG Grid re-themed to the tokens (rail headers, hairline borders, primary hover/selection); table
  cards raised with a primary ring when focused; context/export menus on `menuCls`; drawers animate in;
  contrast sweep killed the remaining load-bearing `slate-400`.
- Verified: `npm run build` clean; impeccable detector **0 findings** (was 2 false positives); live
  serve checked on :8600. User visual gate pending.

---

### 2.43 Stitching becomes an export choice — verification-safe by construction (2026-07-16)

Shaped via `/impeccable shape`; the user's challenge ("is stitch mode really a necessity?") overturned the
proposed fix and produced a better one. Worth recording as a design lesson.

- **The bug behind the symptom.** `stitch_pages` was an **extraction** flag (defaulting to True in the
  webapp): it reshaped the pipeline result into `document_tables`, and that one shape then drove BOTH review
  and export. Because a stitched doc-table has no 1:1 mapping to any page's Surya tables, `table_bbox_index`
  came back empty → **page↔image linking silently died**, exactly when tables span pages (the ARDB case, and
  the case an analyst most needs to verify). The UI apologised with a note instead of fixing it.
- **The insight.** Review and export want *opposite* shapes, but at *different moments* — review wants
  per-page (linked to the image being verified against); export wants one table (paste into Excel once,
  header not repeated). They never actually conflict; the flag forced a false choice, and defaulted to the
  un-verifiable side.
- **The fix (smaller than the one first proposed).** Extraction always stays per-page. Joining happens at
  **export**, on the **edited grids**: new pure `webapp/tables.py::stitch_grids(final_tables, stem)` mirrors
  `engines/table_merge_pages.merge_document_tables` at the grid level (same ±1 col tolerance, same NFC+
  whitespace header-repeat signature, same empty-row drop, same `{stem}_tableN` naming). Export endpoints take
  `?combine=` (default true); the React Export menu carries the choice ("Join into one table" / "Keep one table
  per page", remembered silently). The pipeline's own `stitch_pages` is untouched for the CLI.
  **Rejected**: stamping per-row `source_page` provenance through the merge — it would have made the symptom
  navigable while leaving the false choice in place.
- **JSON stays per-page** (the faithful record); CSV/XLSX are the analyst's working artifacts and combine.
- **Verified live on the real 3-page 15.06.26 bulletin:** linking alive on all 3 pages (`linked_regions=1`
  each — previously 0); combined export = 1 CSV / 76 rows vs 3 CSVs per-page; an edit made on **page 3**
  survives into the combined CSV (the risk named in the brief). Note the ARDB continuation pages start with
  data, not a repeated header, so header de-duplication correctly no-ops there — the win is 3 CSVs → 1.
- **Also in this arc:** empty cells are no longer flagged/tinted as low-confidence (blank cells are
  intentional table structure, not OCR errors — this alone cut the triage list on p1 from **50 → 4**);
  replace-all gained a confirm + server-side undo (`/replace/undo` snapshots `edited_tables`); verify-sync
  failures surface instead of silently reverting; Export-all carries an aggregate unverified count; page-image
  confidence boxes are dashed as well as coloured; the four stacked status banners collapsed to one
  prioritised line; "GDDE · review workspace" subtitle dropped.
- 640 tests (10 new: 7 `stitch_grids`, 3 export-combine). Design critique score 29 → 32/40 across the arc.

**Lesson.** *A flag that forces a false choice is a design bug, not a setting.* When two needs look opposed,
check whether they actually collide at the same moment — `stitch_pages` looked like a preference and was
really a bug that silently disabled the tool's verification story. Also: the user asking "is this even
necessary?" was worth more than the implementation I had already justified.

---

### 2.42 Resolution cap raised 2048→2900, after the eval harness was caught inverting the verdict (2026-07-16)

**Trigger.** An outside consult (Gemini) proposed a "resolution A/B": raise `_CAP_RESOLUTION_MAX_DIM`
to 4096 and/or ingest at 300–400 DPI, on the theory that dense-table cells are only 15–25px tall and
are therefore *upsampled* into Kiri's `IMG_H = 48`, feeding the CTC stack interpolated blur.

**The premise is false for the primary corpus — measured, not argued.** All 21,701 GT cell crops
(`table_gt_v1/recognition`, harvested at 200 DPI) have median height **67px**, p10 67 / p90 68, and
**0.0% fall below 48px**. Confirmed on the *production* path (instrumenting the crops handed to
`recognize_cells_conf`): ARDB cells are **64px**, 0.0% below `IMG_H`. ARDB cells are *downscaled*
0.75× into Kiri — extra page resolution is discarded at the resize and cannot help.

Two supporting facts sink the proposal as framed:

- **The cap was inert on ARDB.** The dailies are **720×720pt — square, not A4** → 2000×2000px at 200
  DPI, under the 2048 cap. No downscale occurred. Raising DPI *alone* would render 3000×3000 and
  downscale it straight back to 2048; raising the cap *alone* is a no-op. The suggested "4096px **or**
  300/400 DPI" is wrong in **both** branches.
- **A recognition-path-only cap bypass is impossible.** `preprocess()` asserts
  `full.shape[:2] == geo.shape[:2]` (preprocess.py) — the invariant keeping table bboxes synced to
  text bboxes. The suggested per-path bypass trips it.

**But it is true for large scans.** Budget p3 is **4400×3400** natively → the 2048 cap crushed it
0.465×, leaving median crop height **39px with 97.1% of cells BELOW `IMG_H`** — genuinely upsampling
blur, exactly the hypothesised failure, on exactly one document class. At cap 2900 those crops become
**55px (2.9% below)**.

**The harness inverted the verdict.** The raw A/B *looked* like a regression:

| budget p3 | cap 2048 | cap 2900 (raw) | cap 2900 (aligned) |
|---|---|---|---|
| `numeric_cell_accuracy` | 0.279 | **0.005** ✗ | **0.550** |
| `cell_content_recall` | 0.246 | 0.457 | **0.457** |
| `cell_accuracy` | 0.608 | **0.472** ✗ | **0.721** |
| `table_cer` | 0.256 | 0.164 | **0.164** |

Order-insensitive metrics rose while position-sensitive ones collapsed — the signature of a row
shift, not a read failure. Root cause, two interacting flaws:

1. `_strip_title_row` drops row 0 only if its first cell is non-empty and the rest are empty. GT's
   *clean* title matches and is stripped (35→34 rows); the **same title OCR'd into garbage** (first
   cell empty) does not → pred stays 35. Off by one.
2. `_align_rows` used `difflib` opcodes. On real documents OCR garbles **every** row, so no row
   compares `equal`, the matcher degrades to one big `replace` block, and rows pair **positionally**
   — so the extra row shifted all 34 remaining rows.

cap 2900 was being punished for *correctly* detecting the title row (35×16 = GT's true dims, vs
34×16 at cap 2048).

**Fix.** `_align_rows` is now monotonic Needleman-Wunsch over `_row_similarity` (1 − normalised edit
distance, reusing the `_levenshtein` already in the module), with `_ROW_ALIGN_MIN_SIMILARITY = 0.5`
leaving dissimilar rows unmatched. A better *title heuristic* was rejected: cap 2048's row 0 is a
badly-OCR'd **header** that also looks sparse, so any "strip sparse row 0" rule would strip it and
make things worse. The fix is alignment, not detection. `difflib` dropped (now unused).

The rebuilt harness reproduces the manual title-drop control **exactly** (0.457 / 0.550 / 0.164 /
0.721) with no hack — independent mutual validation.

**GO — cap 2048 → 2900.** Budget p3 `numeric_cell_accuracy` **0.279 → 0.550** (~2×), recall
**0.246 → 0.457** (~1.9×), CER **0.256 → 0.164** (−36% rel.). The 6 ARDB GT pages are **identical**
across both caps (2000px < both) — a control that was unaffected by construction, so the blast radius
is exactly the oversized docs. Runtime ~1m51s for the 8-page gate; no memory pressure on the 24GB box.

**Historical numbers moved** (the harness fix, not the cap): ARDB p1 `cell_accuracy` 0.898 → **0.931**,
`empty_cell_precision` 0.818 → **0.977**. They moved *toward correct* — prior figures under-reported.
Any §2.33–§2.41 metric quoted from the old harness should be re-run before it enters the report.

Artifacts: `experiments/crop_scale/{measure_crop_heights.py,gate_cap.py,heights_*.json,gate_*.json}`.
640 tests pass (3 new: garbled-rows-with-leading-insert, monotonicity, dissimilar-rows-unmatched).

**Lesson.** *A gate that can invert a verdict is more dangerous than no gate.* This is §2.7's lesson
recurring — "Cell_Accuracy looked catastrophic … it was a row-alignment artifact" — and the harness
still shipped the flaw, because the exact-match aligner is only correct when OCR is *clean*, i.e.
precisely when the measurement doesn't matter. It silently punished a change that doubles accuracy.
Also: an outside consultant with no repo access got the *mechanism* right (`IMG_H` starves on
low-res crops) and every *specific* wrong (corpus shape, which knob binds, where it applies) — mine
the reasoning, verify the specifics, and let the measurement rule.

**Open — the real ceiling.** `IMG_H = 48` discards ~30% of the vertical detail already captured on
ARDB (64px → 48px) — for Khmer that is the diacritic band. Page DPI cannot reach it; only a recognizer
with a taller input can (arch change to the conv stem + CTC head; pretrained weights won't transfer).
Logged, not built.

---

### 2.64 Grid decoupling, metric-tiered scan wordings, stale toast, toolbar cleanup (§2.64)

**Problem.** (1) The post-run grid overview rendered every document page, so pages
excluded from a ranged run showed as broken/empty frames; it also used raw previews
and passed document page indices to a viewer that indexes results (a latent
wrong-page bug on ranged runs). (2) Scan-check wordings were binary — "will help"
regardless of how bad the metric actually was. (3) The stale-settings banner stole a
full-width row. (4) The viewer footer clipped instead of wrapping when the split
divider was dragged, the Loupe carried a text label, and pre-run telemetry chips
lingered in post-run review strips. No frontend unit-test runner existed at all.

**Decision.** TDD-first: added **vitest** (`npm run test`) and extracted the pure
logic into `frontend/src/lib/` — `pages.ts` (moved `pagesFromSettings`/`encodePages`
out of App.tsx, new `gridPages(mode, pageCount, lastRun)` with a
`CanvasMode = 'pre-upload' | 'post-analysis'` type) and `scan.ts`
(`scanWordingKey(check, scores)` with severity cut points SEVERE_TILT_DEG=2.5°,
HEAVY_STAMP_RATIO=5%, SEVERE_CONTRAST_STD=40 vs the backend's 60 threshold).
19 tests written red first, then green.

- **Post-analysis grid**: `PageGrid` now takes an explicit `pages: number[]`;
  post-run it renders only the processed pages, as their *processed* renditions,
  with result-index mapping (`processedPages.indexOf(n)`) for both image URLs and
  open-page — fixing the latent ranged-run wrong-page bug. Checkbox selection still
  speaks document-page indices into runSettings.
- **Scan check**: tiered phrasings ('might help' / 'will help' / 'is recommended')
  via new i18n keys (check_tilted_minor/major, check_stamps_minor/major,
  check_contrast_minor/major; en+km, km pending native review). The static AUTO
  badge became a dynamic per-row readout: emerald "Auto: Applied" when the scan
  check acted, slate "Auto: Not applied" when it measured and left the toggle alone.
  SettingsDrawer gains a `scores` prop.
- **Stale toast**: the stale-settings banner left the notice chain; it is now a
  bottom-left floating toast (fixed bottom-4 left-4, max-w-[320px], bg-surface/95
  backdrop-blur, warn-tinted border, toast-in keyframe + reduced-motion). 10 s
  auto-dismiss restarted by further settings edits, explicit ×, inline "Re-run now"
  link, always-mounted aria-live="polite" wrapper. Token colors kept over the
  brief's raw amber-500 (theme consistency).
- **Toolbars**: telemetry chips purged from post-run strips (PageViewer `telemetry`
  prop deleted; chips remain pre-run where they configure the upcoming run); Loupe
  is icon-only (iconBtn + aria-label + loupe_tip tooltip); footer refactored to a
  fluid two-cluster `flex flex-wrap justify-between` rail that wraps gracefully
  under divider drags. The brief's `min-w-[400px]` floor was dropped — it conflicts
  with the viewer panel's established 320 px minimum.

**Outcome.** vitest 19/19, pytest 710, tsc/build clean, detector 0 findings; bundle
`index-Bx8Rx1Mf.js` live without a server restart (frontend-only change).

### 2.65 Delete-all, absolute toggle semantics, passive scan notice (§2.65)

**Problem.** (1) No way to empty the queue short of removing documents one at a
time. (2) The preprocessing panel could show an emerald "Auto: Applied" badge next
to a switch that was OFF — because a suggestion of `sharpen: false` counted as
"applied automation" — so the panel contradicted itself. (3) The pre-run canvas
strip carried read-only telemetry chips ("Sharpen — off (auto)") that spent
permanent layout on information relevant only once, just after upload.

**Investigation.** The "off means off" claim was traced end to end before touching
anything: `Settings.sharpen` -> `PreprocessConfig(sharpen=...)` -> `if cfg.sharpen:`
in `_geometric_preprocess`. The pipeline already honours a false flag exactly; the
defect was purely the badge's wording. One real state bug did surface though: the
suggestion effect merged `{...prev, ...s.suggested}`, so an advisory scan result
arriving after the operator had already flipped a toggle would silently overwrite
their choice. `DELETE /api/documents` and `api.clear()` already existed and were
tested (`test_clear_all`), so no backend work was needed.

**Decision.** TDD via new `frontend/src/lib/settings.ts` (9 tests, red first):

- `autoBadge(on, isAuto)` — OFF always yields a neutral "Off" badge, never an
  automation claim; ON + suggested yields "Auto: Applied"; ON by hand yields no
  badge at all (the switch already says so). This also retires the "Auto: Not
  applied" noise §2.64 put on every row.
- `mergeSuggestion(prev, suggested, touched)` — advisory values fold in only for
  keys the operator has not touched, backed by a `touchedRef` fed by the existing
  override callback and reset per document. Automation advises; the person decides.
- `scanSummary(checks)` — digest powering the new notice.

**Correction (same day, user review).** The first cut of `autoBadge` returned a bare
"Off" for any disabled step, which erased the automation's authorship exactly when it
had acted. Worse, `preprocess.py` only ever emits `sharpen: False` / `normalise:
False` — every suggestion this backend can produce is a turn-OFF — so the emerald
"Auto: Applied" branch was unreachable in practice and the one real case rendered as
an anonymous "Off". Rewritten so the badge answers a single question, "did the scan
check decide this row?", with the text carrying direction: `applied` (emerald, it
switched the step on), `auto-off` (neutral "Auto: Off", it switched the step off —
auditable, and unambiguous that nothing is running), `null` for the operator's own
choices and untouched defaults, where the switch already says everything.

Rail: "Delete all" appears in the Documents header only at >1 document, confirms,
then clears the collection and resets the canvas to the empty dropzone (plus
triage/suggestion state). Strip: telemetry chips deleted; in their place a passive
toast slides in on upload, summarises the scan check, and unmounts after 4 s with
no confirmation click. Its "Review" link keeps the §2.58 jump-to-setting behaviour
alive. Both toasts now share one bottom-left stack instead of overlapping.

**Verification gap found.** `npx tsc --noEmit` — used as the typecheck gate through
this whole arc — is a NO-OP here: `tsconfig.json` is solution-style (`"files": []`
with project references), so it exits 0 without checking anything. It silently
passed a genuinely broken tree (a `ScanSearch` usage with no import). The real gate
is `npx tsc -b`, which the build script already used. Prior "tsc clean" claims in
this arc were vacuous; use `tsc -b` from here on.

**Outcome.** vitest 28/28, pytest 732, `tsc -b` clean, build clean, detector 0
findings; bundle `index-CbTGgIfl.js` live (frontend-only, no restart needed).


### 2.66 Delete guard, concurrency gating, sub-stage telemetry, applied/draft config (§2.66)

**Problem.** (1) Clearing the queue relied on a native `window.confirm`. (2) Launching
a second extraction produced a red banner reading "Another extraction is already
running." — the server's 409 leaking into the UI as though the analyst had erred.
(3) The OCR stage label sat frozen on "Finding text & tables…" for minutes. (4) The
applied-vs-draft settings split existed but was unnamed and re-derived inline, and a
re-run left the previous run's results cached under the old configuration.

**Decision.**

*Phase 1 — test infrastructure.* The frontend had no component-test capability, so
this pass added jsdom + Testing Library. Two obstacles worth recording: vitest does
NOT consume `vite.config.ts` for its own `test` block under rolldown-vite (a separate
`vitest.config.ts` is required), and Node 22 injects a stub global `localStorage`
whose `getItem` is undefined, shadowing jsdom's — `src/test/setup.ts` installs a real
in-memory Storage per test. 5 component tests + 12 logic tests, red first.

*Phase 2.* "Delete all" is now a trash icon opening an anchored confirmation popover
(count named, Escape/outside-click dismiss, `role="dialog"`), replacing the native
confirm. The guard is what the tests pin: the first click can never remove anything.

*Phase 3.* The 409 banner is gone. `isBusy(documents, batchRunning)` derives one
workspace-wide gate from the document list, and `guardedRun` refuses to dispatch
while busy and swallows a server 409 as a benign collision while still propagating
real failures. The gate now disables Run all, the run buttons, the `r` shortcut, the
palette's run command, and every engine/preprocess control in the settings drawer —
run parameters cannot be edited out from under a run that is consuming them.

*Phase 4.* Real sub-stage telemetry, not a socket (the app polls; no socket exists).
`run_surya` gained an optional `on_step` callback firing "layout" / "text" / "tables"
inside each page; `Progress.step` carries it and `/status` exposes it. The runner
passes `on_step` only to engines whose signature accepts it (`inspect.signature`), so
the five other engines keep working untouched — pinned by a test asserting a legacy
two-parameter engine is still called cleanly.

*Phase 5.* The two configurations are now named — `appliedConfiguration` (the frozen
`last_run_settings` snapshot) and `draftConfiguration` (live sidebar state) — and
compared through `configDiffers`, which deep-compares only the keys the applied
snapshot recorded. On re-run the stale result caches (overview/page/lowconf) are
removed rather than left to be invalidated later, and triage state resets, so nothing
from the old configuration survives into the new review.

**Outcome.** vitest 45/45 (5 files), pytest 736, `tsc -b` clean, build clean,
detector 0 findings. Server restarted (backend touched; registry cleared); bundle
`index-BoZU3nNU.js` live.

### 2.67 Sub-stage telemetry fix, `auto` engine exposed, HITL capture wired (§2.67)

**Problem.** (1) The §2.66 sub-stage telemetry did not work and was reported to the
user as working. (2) The validated `auto` router was registered but unreachable from
the UI. (3) `corrections.capture_corrections()` (built + tested, 11 tests) had no
caller in the webapp.

**Investigation — two defects, both from §2.66.** `on_step` was added to `run_surya`
ONLY; `run_surya_kiri`, `run_surya_kiri_vlm` and `run_auto` all take `(result,
on_page)`. Because the runner passes `on_step` only to engines whose signature accepts
it, every Kiri-family engine silently reported nothing. Separately, even on `surya` the
order was wrong: `_step("tables")` fired BEFORE `_step("text")`. The mechanism had been
verified against one engine and the claim generalised without checking which engine was
actually selected.

**Decision.**

*Part 0 — telemetry.* Step order corrected (layout → text → tables, "tables" now fires
after the recognition block). `on_step` threaded through `surya_kiri` (forwarded to its
inner `run_surya`, plus its own per-page table step), `surya_kiri_vlm` (same shape) and
`auto` (forwarded to BOTH routed engines). The durable fix is the **regression guard**
in `tests/test_engine_registry.py`: every engine in `_OCR_ENGINES` must accept `on_step`
or appear in an explicit `_NO_SUBSTEP_ENGINES` set (`tesseract`, `hybrid`). This turns a
silent, engine-dependent runtime failure into a static test failure — it named all three
broken engines immediately.

*Part A — `auto` exposed.* Added to `_ENGINES` as "Automatic" (matching the existing
task-language register — "Standard", "Khmer-text specialist" — rather than an engine
name). `with_recognition_images` now covers `("surya_kiri", "auto")`: `auto` may route to
surya_kiri, which needs the geometric-only frame, and omitting it is a measured accuracy
loss (§2.30) that surya_kiri only *warns* about. `Settings.ocr_engine_key` now defaults
to `"auto"`; the frontend persists the last engine in localStorage, so only fresh
profiles inherit the new default.

*Part B — HITL capture.* Hooked into `api_review` on the `False → True` verification
transition only (gold-standard rule). Two corrections to the handover sketch, found by
reading the code: `tables=` must be `doc.surya_result.pages[n].tables` (export blocks
carry `row`/`col`/`text` and NO geometry, so capture would have silently produced
nothing), and the positional key mapping needs a `table_id → (page_idx, t_idx)` lookup
derived from `document_json` via `tables.page_table_blocks`. Crops come from
`recognition_page_images`. A `captured: set[str]` on `Document` prevents duplicate pairs
when verification is toggled off and on (capture APPENDS). The whole call is wrapped so
a failure can never cost an analyst their save.

*Reviewer safeguards (user).* The `except` reports `repr(e)` so permissions vs index
mismatch vs serialization stay distinguishable; `_locate_table` fails soft (returns
`None`) on stitched or malformed structures; and `verify_corrections.py --inspect` gained
a **geometry gate** asserting each crop's pixel size equals its recorded bbox, exiting
non-zero on drift. Verified both ways against a generated store: clean data passes, a
deliberately corrupted crop is caught.

**Outcome.** pytest 742 (from 736), py_compile clean. Server restarted; `/api/meta` now
serves four engines with `auto` as the default. Frontend untouched — the picker renders
from `/api/meta`.

### 2.68 Engine order, Auto-DPI, settings-badge override count (§2.68)

**Problem.** (1) `auto` was added last in the picker (§2.67) but is the recommended
default — it should lead. (2) Analysts had to guess a DPI; a faint scan at 200 loses
glyph detail. (3) The Settings badge showed a persistent count the user read as "total
settings", not overrides.

**Investigation — item 3 was already diff-based.** The badge already computed
`changedSettings` (fields differing from `/api/meta` defaults) and hid at zero, so the
brief's stated fix was a no-op. The real defect: the diff ran over ALL ~19 dataclass
fields, including ones the drawer never exposes (`show_layout`, `overlay_mode`,
`tables_only`, `stitch_pages`) — a stale or seeded value on any of those inflated the
number with no control the user could touch to clear it. Fixed by scoping the count to
an explicit allowlist of the 11 controls the drawer presents (`countOverrides` in
`frontend/src/lib/settings.ts`, 5 tests), which also collapses page sub-fields so one
page-scope change counts once.

**Decision.**

*Item 1.* Moved `auto` to index 0 of `_ENGINES`; `Settings.ocr_engine_key` default
already `"auto"` (§2.67). Guarded by `test_auto_engine_is_first_in_the_picker`.

*Item 2 — Auto-DPI.* `Settings.dpi` now accepts `"auto"` (the new default) or a
positive int; `_settings_from` validates the union and 400s on anything else (a bad dpi
would otherwise blow up in ingest's `dpi/72`). New `ingest.resolve_auto_dpi(bytes, name)`
inspects each PDF page's embedded-image density (widest image px ÷ page width in inches):
≥250 native DPI, or vector/born-digital, or an image input → 200; below that → 300, so a
faint/low-res scan is upsampled for more pixels per Khmer glyph. **Worst page wins** (one
faint page warrants 300), and unreadable metadata biases to 300 (accuracy over speed).
The runner resolves `"auto"` to a concrete DPI once, from the actual document, before
ingest, and records both the resolved `dpi` and a `dpi_auto` flag in provenance. 6 tests.

*Physical note.* Upsampling a low-native-DPI scan to 300 adds no new information, but
larger glyphs help the recognizer (whose input has an optimal glyph height), and it never
hurts a smooth bicubic upscale — so "faint → 300" is a sound accuracy/speed trade, not
magic resolution recovery. 200 stays the default for clean/high-density work to keep the
24 GB Mac's memory and runtime down.

*Item 3.* `countOverrides(settings, defaults)` over the allowlist replaces the inline
all-fields diff; the badge still hides at zero. The auto-DPI default reads as unchanged.

**Outcome.** pytest 749 (from 742), vitest 50 (from 45), `tsc -b` + build clean, detector
0. Server restarted; `/api/meta` serves `auto` first with `dpi: "auto"` default.

### 2.69 Audit remediation: keyboard a11y, code-split, polish (§2.69)

**Problem.** `/impeccable audit frontend` scored 16/20 (Good) with a contained cluster
of implementation-level gaps: two WCAG 2.1.1 Level-A keyboard violations, a 1.43 MB
single JS bundle, and a few P3 items. Design language and theming already scored 4/4;
nothing aesthetic needed changing.

**Decision — ran all six recommended passes.**

- *P1 keyboard (QueueRail).* The document row was a `<div onClick>` (no role, tabindex,
  or key handler) and the remove button was `hidden … group-hover:block` (mouse-only).
  Row is now `role="button" tabIndex=0` with an Enter/Space handler guarded to fire only
  when focus is on the row itself (not bubbling from the nested remove button), plus a
  stable `aria-label` and `aria-current`. Remove button is always in the tab order,
  revealed by hover OR `focus-visible` (opacity, not display). 2 new component tests.
- *P2 bundle (code-split).* `TablesPanel` — the sole AG-Grid consumer, only rendered
  once a run has results — is now `React.lazy` behind `Suspense`. **Initial JS dropped
  1.43 MB -> 340 KB (104 KB gzip)**; AG-Grid (1.06 MB) split into a chunk that loads on
  first results, never on the empty state or pre-run flow.
- *P2 keyboard (PageViewer).* The page canvas is now focusable with arrow-key pan, +/-
  zoom toward centre (mirroring the wheel math), and 0-to-fit — mouse drag/wheel/loupe
  stay as enhancements over that baseline. `role="group"` + aria-label.
- *P3 polish.* Text "Loading tables..." replaced by a content-shaped `TablesSkeleton`
  (aria-busy), reused as the Suspense fallback — one component covers both the lazy-chunk
  load and the per-page tables fetch.
- *P3 typeset.* The one meaningful 11 px `ink-3` label (command-palette group header)
  lifted to `ink-2` for the small-text contrast floor.
- *P3 colorize.* The PageViewer overlay palette (`PALETTE`/`LABEL_COLORS`) annotated as
  deliberately theme-independent fixed hex drawn over the page photo, matched by hand to
  `--color-conf-*` / `webapp/components.py`.

Responsive stayed 2/4 by design — a desk-bound analyst tool + projected demo; mobile is
explicitly out of scope, so no action (documented in DESIGN.md and the audit).

**Outcome.** vitest 52 (from 50), `tsc -b` + build clean, detector 0. Frontend-only —
no restart; the no-cache index.html serves the split chunks on refresh.

### 2.70 Page Viewer & Table Editor UX refinement (6 items, §2.70)

**Problem.** A design consultation on the Page Viewer + Table Editor surfaced six UX
issues; decisions were locked with the user. Frontend-only.

**Decision.** One confidence vocabulary everywhere — **Check <80% · Skim 80–95% ·
Clean >95%** — extracted to `frontend/src/lib/confidence.ts` (`confBand`,
`bucketCounts`, `bandCells`, `nextInBand`), 8 tests written first.

- *Loupe (1).* `LENS_MAG 3→2`, `LENS 180→200` — 2× reads calm on dense coeng stacks
  while keeping ~6 glyphs of context.
- *Canvas strip (2&3).* Fit/100% replaced by one zoom cluster `[⛶] [−] {pct}% [+] [🎯]`:
  Fit icon-only, a −/+ stepper toward centre with a live % that clicks to reset, and a
  Focus Table button that frames the selected table's bbox. The fly-to-evidence camera
  was extracted to a reusable `flyTo(bbox)` (via a `viewRef` mirror) and now serves both
  the triage `flyToken` effect and the button. Focus Table tooltip: "Focus table" /
  "No table selected" when disabled.
- *Confidence legend (4).* Loose dotted rects → a count cluster joined to the overlay
  selector (`● {n} Check · Skim · Clean`), counts from `text_blocks` via the bucketer.
  `confColor` + the dash rule realigned to the shared bands (a null-confidence block now
  reads Clean, not a false red). Colour is redundant: visible count + band word +
  aria-label.
- *Find + shortcuts modal (5).* Find is page-scoped, so a `[🔍 Find]` button lives in the
  Tables panel header (right utilities), toggling the existing bar via a new `onOpenFind`
  prop; ⌘F still works. The help modal's key column is re-modelled as combos-of-atomic-
  keys rendered as non-wrapping caps with a fixed `w-32` column — fixing the inconsistent
  `' / '` split and the overflow into descriptions.
- *Triage header (6).* The Tables header is now three zones — facts · interactive triage
  chips · grid utilities. The static legend became clickable band chips
  (`🔴 {n} Check` …); a click cycles to the next cell in that band via a new App helper
  `focusGridCell` (mirrors `jumpToIssue`; drives both grid scroll and the page fly), with
  a live `Check 1/3` progress counter. Per-band cursors reset on `[docId, pageIdx]` and
  re-clamp if a band shrinks (out-of-range guard). No "Section Tabs" (would be reinvented
  chrome). Redundant word+count+aria on every chip.

**Outcome.** vitest 60 (from 52), `tsc -b` + build clean, detector 0. Frontend-only —
no restart; the no-cache index.html serves the new bundle on refresh.

### 2.71 Grid thumbnail resilience during active extraction (§2.71)

**Problem.** Toggling Grid → Single → Grid while an extraction is running left pages 3+
as blank cards with the browser's broken-"?" glyph. Root cause is server-side and
concrete: `api_preview_image` rasterizes lazily (`doc.ingest_result = ingest(...)` on
first call). A grid remount fires one request per card at once; during extraction the
runner saturates the process, so the first burst loses the race and those requests
error before the ingest is cached. Not a blob-URL leak — the images are plain HTTP
endpoints, so there is nothing to `revokeObjectURL` and view toggles leak nothing.

**Decision.** Frontend resilience in a self-contained `GridThumb` (in `PageGrid.tsx`)
rather than backend surgery. Each thumbnail: holds an `animate-pulse` skeleton while
loading; on error tries a `fallbackSrc` once (post-analysis grid → raw preview via a new
optional `fallbackUrl` prop), then retries up to 4× with a cache-busting `?retry=n` and
250ms·n backoff — which *actually recovers* the image once the lazy ingest lands; and if
still failing, drops the `<img>` for a calm skeleton so the broken glyph never shows. A
`useEffect([src])` resets the whole lifecycle on src change, so a finished rendition or a
document switch never bleeds a stale thumbnail. Page-number chip hardened
`whitespace-nowrap` (clear of the top-left checkbox island — no clip).

**Outcome.** 4 new `PageGrid.test.tsx` cases (fallback swap, cache-bust retry, skeleton
on exhaustion). vitest 64 (from 60), `tsc -b` + build clean. Detector's 2 `broken-image`
hits are the test's fixture URLs (`/proc/0`), not shipped UI — false positives. Frontend-
only; plain refresh.

### 2.72 Honest stamp removal, Page Text blocks, grid cache (§2.72)

**Problem.** Three findings from one review. (1) Stamp removal was destructive and
dishonestly labelled: `_stamp_ink_mask` thresholded red+blue across the whole page and
`_remove_stamps` dilated it 7×7 ×2 (~14px halo) before inpainting, so blue body text,
red figures and blue rules were erased — **on by default**, while the UI promised it
only "erases signature stamps that sit over the numbers" and the scan check actively
*recommended* it based on a colour ratio that blue text trivially exceeds. (2) Page Text
was one monolithic `<textarea>`, discarding hierarchy. (3) The §2.71 grid fix treated a
symptom: the image endpoints sent **no cache headers**, so every Grid↔Single toggle
refetched every thumbnail — the request burst that broke thumbnails in the first place.

**Measured, not argued** (`scripts/probe_stamp_mask.py`, deterministic — chosen over an
OCR ablation because Surya's run-to-run variance would drown the signal). On the real MoC
gas notification: colour mask 1.66% of the page → **8.61% of pixels destroyed** (5.18×
blow-up), visibly wiping blue paragraphs and blue table figures.

**Decision — gate on SHAPE, and never destroy what we cannot identify.**
`_stamp_regions_mask` keeps a colour component only if it survives a 3px opening, its
bbox clears `_STAMP_MIN_BOX_FRAC` of the page's shorter side in *both* dimensions, its
aspect is within `_STAMP_MAX_ASPECT`, and it is not multi-line text. Nothing qualifies ⇒
image returned unchanged. Two findings shaped it:
- **Fill density cannot be the gate.** A hollow ring seal scores ~0.05 extent — *below* a
  merged text block (~0.1–0.2) — so a density floor rejects real stamps and a ceiling
  misses the text. Density is used only as a fast-accept for solid seals
  (`_STAMP_SOLID_EXTENT`). The actual guard is `_has_text_line_gaps`: multi-line text
  alternates dense rows with blank bands, a ring has ink on its arcs in every row.
- **A 5px opening erased the seal's own ring** (measured: the 333×334 component vanished),
  so the kernel is 3px; and because a seal fragments into ring + inner text + emblem, a
  confirmed stamp's whole bbox is treated as stamp territory (colour filtering stays
  *localized* to a region already proven to be a stamp).

Result on the same page: **8.61% → 2.28%**, stamp fully removed, every blue paragraph and
table figure intact. `suggest_preprocess_settings` now scores the shape-gated mask, so the
UI stops recommending removal on documents that merely contain coloured text.

**Copy (via `/impeccable clarify`).** `hint_stamps` now states the limit ("Coloured text is
left alone; anything inside the stamp outline is erased too"), and `check_stamps_major`
dropped "is recommended" for a consequence, matching its sibling checks. PRODUCT.md
principle 1 — *never reassure falsely* — is the governing rule.

**Page Text Phase 1.** The API already sent `text`/`reading_order`/`region_label` per block
(surya.py:371 → api.py:306); only the TS type under-declared it, so this was frontend-only.
New `PageTextPanel` gives a **Blocks** audit view (one card per region in reading order,
type + confidence chip, colour never the sole signal) and a **Raw** edit view. Blocks are
deliberately **read-only**: they carry raw OCR text while the textarea edits
`corrected_text`, so an editable block list would silently overwrite the correction pass.

**Grid root cause.** `ETag` + `Cache-Control: no-cache` on both image endpoints, 304 on
`If-None-Match` (verified live: 304, 0 bytes — the PNG encode is skipped). `no-cache` +
validator, never `max-age`: a re-run replaces the image at the same URL. Processed pages
are now served from stage 2 rather than after full results, with `processed_pages` in
`/status` mapping result index → document page (a page-scoped run makes those differ).

**Outcome.** 785 backend tests, 74 frontend; `tsc -b`, build, detector clean. `ViewToggle`
generalized to `SegmentedToggle` rather than copy-pasted for the new panel. `dev.sh` gained
`restart` after this work hit the stale-backend trap the reuse behaviour creates.
**Khmer for the changed/new strings is untranslated and flagged in `i18n.tsx`.**

### 2.73 The `auto` router's blind spot, reproduced — and Surya's non-determinism explained (2026-07-22)

**Problem.** The React frontend's OCR output looked far worse than the legacy Streamlit app's.
The pipelines are code-identical per engine (`app.py:419-457` vs `webapp/runner.py:70-113`), so
the only real difference is the default engine: Streamlit's radio has no `index=` and so defaults
to the **first** option, `surya`; the webapp defaults to **`auto`** (`webapp/settings.py:35`).

**The first hypothesis was measured and disproved.** Scored on the two ARDB bulletins,
`auto`/`surya_kiri` beat plain `surya` decisively (0.959 vs 0.579-0.633 cell accuracy) and ran
3-5× faster. But ARDB bulletins are Kiri's specialty — they cannot reproduce the complaint. The
documents that could had only **per-page** GT, which the document-only A/B harness could not score.

**Two harness limits had to be fixed before the real question could even be asked.**

1. **The numeric metric did not measure money.** On the MoC gas bulletin every money cell reads
   `០,៧១១៧ ដុល្លារ` — Khmer digits, **comma** decimal separator, unit-word suffix. `_NUMERIC_RE`
   accepted none of these, so 33 of 56 cells were classed as Khmer *label* text and the "Numeric"
   column reported on the `ល.រ` row-index column instead. `_is_numeric` now decomposes a cell into
   `[currency symbol] [number core] [unit token]`: leading symbols by Unicode category `Sc`,
   one trailing digit-free token ≤ `_UNIT_AFFIX_MAX_CHARS`, accounting parens — generalized rather
   than an enumerated unit list. Comma-decimal is accepted **only** with a locale signal (Khmer
   digits or a unit affix), which preserves the `7,8000` Kiri digit-duplication guard that an
   unrestricted comma reading would silently delete. Effect: moc_gas numeric 9 → **33**, Khmer
   39 → **15**; budget p3 (222/49) and ARDB (422/150) **unchanged** — a no-op where not needed.
   Every GT grid still self-scores 1.000 on all three classes, so the classes stay disjoint.
2. **A metric change used to cost a full GPU re-run.** `compare_engines_ab.py` stored only scores.
   It now stores the **predicted grid** and gained `--rescore`, turning any future metric revision
   into a seconds-long re-score. It also gained per-page GT targets (`mode="page"`, no stitching)
   and `--repeat N` with median±spread reporting.

**Result — the complaint is reproduced, and it is the router's documented blind spot.**
3 runs/engine, medians (`eval/runs/ab_hard/`):

| target | engine | Cell_Acc | Numeric | Khmer | CER | secs |
|---|---|---|---|---|---|---|
| budget_p3 | surya | 0.971 | 1.000 | 0.673 | 0.024 | 50 |
| budget_p3 | **auto** | **0.971** | **1.000** | 0.673 | 0.024 | 75 |
| budget_p3 | surya_kiri_vlm | 0.919 | 1.000 | 0.102 | 0.111 | 55 |
| budget_p3 | surya_kiri | 0.721 | 0.550 | 0.122 | 0.164 | 27 |
| moc_gas_p1 | **surya** | **0.750** | **0.939** | **0.467** | 0.040 | 39 |
| moc_gas_p1 | auto | 0.232 | 0.242 | 0.133 | 0.458 | 35 |
| moc_gas_p1 | surya_kiri_vlm | 0.214 | 0.182 | 0.133 | 0.478 | 38 |
| moc_gas_p1 | surya_kiri | 0.232 | 0.242 | 0.133 | 0.458 | 35 |

- **budget p3: the router works.** `[AutoRouter] fallback surya_kiri->surya | frac=0.539
  cutoff=0.400` — it detects Kiri failing and recovers surya's 0.971/1.000 exactly as §2.57 designed.
- **moc_gas p1: the router fails, and `auto` costs 0.518 cell accuracy against `surya`.**
  `[AutoRouter] kept surya_kiri | frac=0.231 cutoff=0.400` — Kiri reports itself *healthy* while
  producing garbage. Its structure is near-perfect (13×4 vs GT 14×4) and content recall is 0.196:
  a pure **recognition** failure, with digits mis-separated and the unit glyphs corrupted into
  non-words. **This is exactly the ceiling `auto_engine.py:21-23` documented as accepted** — "a
  document where surya_kiri is *confidently wrong* would not trigger the fallback… absent from our
  measured failures." It is no longer absent. The MoC gas bulletin is its first real instance, and
  it is what the frontend was showing.

**Surya's non-determinism is architectural, not a missing seed.** The correlation is exact:

| doc | pages | 3-run spread (Cell_Acc) |
|---|---|---|
| budget_p3 | 1 | 0.000 (bit-identical) |
| moc_gas_p1 | 1 | 0.000 (bit-identical) |
| ardb0 | 3 | 0.630 / 0.721 / 0.643 → **0.091** |
| ardb1 | 3 | 0.520 / 0.633 / 0.578 → **0.113** |

Surya's layout/table_rec generation runs through a spawned **llama.cpp server** at
`temperature=0.0, top_p=0.1` — but with `--parallel 8` slots fed by a `ThreadPoolExecutor`
(`surya/inference/backends/llamacpp.py:147,205`). Concurrent continuous batching makes
floating-point reduction order depend on **how requests happen to co-batch**, which is
non-deterministic even at temperature 0. Single-page documents issue too few requests to vary;
3-page documents do. (A second contributor: `_should_retry` in `openai_client.py` re-issues on
repeat-token detection at `temperature + 0.2×retries`, i.e. **above 0**.) No `torch.manual_seed`
in our code can fix this — the sampling is not in our process.

**Decisions.**
- **The frontend default stays `auto`.** It wins or ties on 2 of 3 measured document classes,
  including the ARDB bulletins that dominate production, where switching to `surya` would cost
  ~0.33 cell accuracy *and* introduce the non-determinism above. Fixing moc_gas by changing the
  default would trade a large regression for a smaller one.
- **Streamlit and the webapp disagreeing on their default is itself a defect** — Streamlit's
  `surya` is an accident of a missing `index=`, not a decision. Reconcile to `auto`.
- **The router's confidence signal is the real bug, and it is now falsifiable.** Kiri's own
  confidence cannot detect Kiri being confidently wrong; a second, independent signal is needed
  (e.g. cross-checking a sample of cells against surya, or scoring unit-token plausibility).
  Deferred — it needs its own measurement, and `moc_gas_p1` is now the regression case for it.
- **Benchmarks on multi-page documents must report medians over ≥3 runs.** Any single-run
  multi-page Surya number in this log carries ±0.09 cell accuracy of noise.

**Not changed.** No engine, default, or UI code was touched — this section is measurement and a
decision to leave the default alone. Harness/metric only: `evaluate_structure.py`,
`compare_engines_ab.py`, tests (12 new; 789 passing in a shared tree).

### 2.74 Frontend audit remediation: state safety, one confirm vocabulary, render calm (§2.74)

**Problem.** A five-axis audit of `frontend/src` (all 23 files read; detector + tests run)
found two silent state-loss defects plus a tier of robustness/perf gaps. Both HIGH bugs were
identity-keyed `useEffect` resets against TanStack Query refetches: structural sharing means
they fire exactly when data changed — and a verify flips one boolean while the analyst holds
unsaved work.

**Decision (ranked, each TDD'd red-first where testable).**
- *H1 draft loss.* `TablesPanel`'s text reset was keyed on the `page` object; verifying any
  table refetched the page and silently overwrote the unsaved Raw-textarea draft. Now keyed
  on `page.corrected_text` itself.
- *H2 undo wipe.* `TableEditor` reset grid+history on every new `table` identity. After
  edit→verify the refetched grid even has new content (the server echoes the saved edit), so
  keying on `table.grid` alone was insufficient — the reset now runs only when the incoming
  grid **differs from the local one** ("confirmation is not new data"); verified/edited sync
  independently.
- *H3 eternal skeleton.* Page-query failure now renders an error + Retry, not a forever-
  skeleton (4-state rule; trust is the product).
- *M tier.* Copy All falls back to `execCommand` + reports failure (clipboard API is absent
  on non-secure LAN origins); batch runs switch docs via `selectDoc` (no page-index bleed)
  and skip docs deleted mid-batch; `v`-verify surfaces errors; one `ConfirmPopover` replaces
  `window.confirm` for clear-all/remove-doc (rendered outside the `role="button"` rows);
  shared `useFocusTrap` makes the ⌘K palette's `aria-modal` hold Tab; `HeaderProgress`
  isolates the 250 ms tick and QueueRail/PageGrid are memoized with stable handlers (a run
  stops repainting the workspace ~7×/s); pre-run single view uses `GridThumb`.
- *LOW sweep.* Localized the last hardcoded alt; keyboard zoom reuses `zoomBy`; rendition +
  DPI segments adopt `SegmentedToggle`; `issues` memoized (global key listener stops
  rebinding per render); context menu clamped to viewport; status poll stops for settled
  docs; 30 s fetch timeout in the API client.

**Not doing (audit-noted, rejected):** grid virtualization (`loading="lazy"` suffices at this
corpus size), typed `RunSettings` (the unknown-bag is load-bearing for server-driven
passthrough), backend stage-string enum (API contract change).

**Outcome.** vitest 80 (from 74; new `TablesPanel.test.tsx` + `TableEditor.test.tsx`),
`tsc -b` + build clean. Detector: 3 hits, all classified false positives (test fixtures ×2;
the `GridThumb` call site pattern-matched as a raw `<img>`). New keys `page_load_failed`,
`retry`, `copy_failed` landed en-only and were translated straight after in `207b1f3`.

**⚠️ Commit-hygiene correction (parallel sessions).** `a16109a` — whose message describes only
UI work — **also carries backend evaluation changes**: `+105` lines in
`src/khmer_pipeline/evaluation/evaluate_structure.py` and `+143` in
`tests/test_evaluate_structure.py`. Those files were already staged in the shared index when
the commit was made, and `git commit` commits the index, not just the freshly-added paths.
The swept-in content is two different authors' work: `_is_khmer_text` / `khmer_cell_accuracy`
(this session's, belonging to the migrated engine investigation) **and**
`_is_unit_token` / `_strip_unit_affixes` (the engine session's in-flight work — `git log -S`
confirms `a16109a` is the only commit introducing them).

*Deliberately NOT reverted.* The engine session still held uncommitted edits in those exact
files in the shared working tree; a revert rewrites the working copy and would likely destroy
them, and `a16109a` was already pushed. Nothing is lost or broken —
`tests/test_evaluate_structure.py` passes at HEAD — so the correct remedy was this record, not
a file operation.

*Consequence for the engine session:* the `evaluate_structure.py` baseline moved forward at
`a16109a`; commit the remaining delta, not the full diff.

*Lesson:* in a shared tree, verify `git diff --cached` (not just `git status`) before
committing, or scope the commit with `git commit -- <paths>`. Verified clean in the other
direction: no other session's commit touches `frontend/`, `webapp/`, or `dev.sh`, and this
session's other nine commits contain no eval/benchmark files.

### 2.75 The router's confidence signal cannot be retuned — the classes overlap (2026-07-22)

§2.73 established that `auto` fails on moc_gas (`frac=0.231`, below the `0.400` cutoff, so it keeps a
confidently-wrong surya_kiri) and deferred the fix as "needs its own measurement". This is that
measurement, and it closes off the cheap option.

**The obvious fix — lower the cutoff — is provably unavailable.** Measuring the same signal
(`frac` of table cells below `CELL_CONF_LOW`) across every page where we know the right answer:

| page | `frac<0.80` | is surya_kiri the right engine? |
|---|---|---|
| ARDB 09.06/15.06 p2 | 0.029 | **yes** |
| ARDB 09.06/15.06 p3 | 0.213 | **yes** |
| ARDB 09.06/15.06 p1 | **0.222** | **yes** |
| **moc_gas p1** | **0.231** | **NO** (costs 0.518 cell accuracy) |
| budget p3 | 0.539 | no — correctly caught today |

The worst page where surya_kiri is *correct* (0.222) and the page where it is *catastrophically
wrong* (0.231) are **0.009 apart**. Any cutoff low enough to catch moc_gas fires on ARDB p1 — and
§2.73 measured what that costs: falling back to surya on ARDB loses ~0.33 cell accuracy *and*
imports Surya's multi-page non-determinism. The two populations are not separable by this signal at
any threshold; the budget-p3 success at 0.539 was a comfortable margin that simply does not
generalize.

**So the design conclusion is stronger than "add a second signal would be nice".** A self-reported
confidence cannot detect a recognizer being confidently wrong — that is a property of the quantity,
not of its calibration. This was documented as an accepted ceiling in `auto_engine.py:21-23` when the
router was built on two document classes; the third class falsified it. A genuinely independent
signal is *required* (cross-checking a cell sample against a second engine, or unit-token
plausibility), and `moc_gas_p1` is the regression case any candidate must pass.

**Method note, and a caution for the report.** The moc_gas GT this rests on was drafted by Gemini
from the page image and verified against pixels. Its **table** GT is sound (one correction applied:
r5c1). Its **prose** GT was deliberately withheld — two independent Gemini runs of the same page
disagreed on 8/8 sampled non-date lines (an inter-ministerial reference number, a $/barrel threshold
90 vs 50, a tax rate 4% vs 45%, the exchange rate, and the office address — two entirely different
Phnom Penh streets, *neither* matching the page). A first run had also silently shifted every date
from 2026 to 2023, rewriting the Buddhist-era year, zodiac year, sak ordinal and CE year in
coordinated fashion with zero uncertainty flagged; naming that failure in the prompt fixed the dates
and nothing else. **LLM-drafted GT was confidently wrong in ways only pixel-level checking caught** —
the concrete argument for the human-in-the-loop discipline, and the reason `text_gt_available: false`
now exists so an unscoreable page reports blank text metrics instead of a fabricated
`Document_CER = 1.0`.

---

### 2.76 Block↔canvas linking, and making "Auto" state its outcome (2026-07-22)

Three UI items; two of them turned out to have a shared shape — *the workspace knew something
it never said out loud*.

**Bidirectional block linking.** Page Text listed the blocks and the canvas drew their boxes,
but nothing connected the two: matching a card to its region on the scan was manual eye-work.
Now hover or click a card and its box gets a primary halo; click a box and its card surfaces.

Three design points worth keeping:

*Identity is the source index into `text_blocks`, never the card's position.* `orderedBlocks`
filters empty layout regions **and** sorts by reading order, so the two genuinely disagree —
a position-keyed link would silently halo the wrong region on any page whose reading order is
not array order. `orderedBlockEntries` carries the source index alongside each block; the test
reverses the array *and* drops a block so a position-keyed implementation fails outright.

*Only the side that did NOT initiate scrolls.* `blockSel` carries `from: 'canvas' | 'text'`
and lives in `App` — the nearest common owner. Without that discriminator the two panes chase
each other's scroll.

*`panTo` centres at UNCHANGED zoom, deliberately not reusing `flyTo`.* The existing fly re-frames
to 3× because triage jumps want the evidence filling the canvas. The block link is a "look here",
not a "zoom in": re-framing the page on every card click would throw away the reading scale the
analyst chose. The halo also renders independently of the overlay mode — the link has to work
with boxes turned off, or it is a feature that quietly stops existing.

Two edge cases outside the brief: a pan ending over a block registered as a click on it (guarded
with a 4 px slop threshold), and page/document switches now clear the link so a stale index cannot
halo an unrelated region.

**The DPI segment's "broken container" was one word.** `SegmentedToggle` was `display: flex`,
which is *block-level*, so the track filled its container. In the viewer footer it is a flex item
and shrinks correctly; in the settings drawer's block container it stretched full-width and read
as a text-input frame. `inline-flex` fixes every segment at once. (`self-start` was briefly added
alongside and removed — as a flex item it would have top-aligned the track inside the viewer's
`items-center` rail, trading one bug for another.)

**"Auto" now states what it chose** — and this needed new backend truth, not new UI. Auto DPI and
the Auto engine router both decide at run time from the document itself, and *neither outcome was
exposed anywhere*. `webapp/effective.py` derives both from what the run already records: the
concrete render DPI from `ingest_result`, and the engine key parsed from the `[AutoRouter]` note
`run_auto` appends to `warnings`.

The rule that shaped it: **`auto` with no router note resolves to `None`, and the badge stays
hidden.** A badge that guessed would assert a routing decision that has not happened yet — worse
than no badge, because the whole point is auditability. Same reason `[AutoRouter]` is parsed rather
than the decision re-derived: the router's own record is the only thing that cannot drift from what
actually ran.

Verified: `npx tsc -b` clean, 86 frontend tests (5 new), 797 backend tests (8 new), build clean.
The six new `km` strings carry user-verified Khmer (provided 2026-07-22) — per the standing rule,
Khmer is never authored here, only mapped in from verified text.

---

### 2.77 Confirm popover, grid badge provenance, and a real Find (2026-07-22)

Three reported UI defects. Two of them were not what they looked like, and one of my own
tests turned out to be worthless — recorded here because the correction is the useful part.

**The grid badge was a disagreement about what a page number means.** Reported as "shows 3
on a 1-page document". My first instinct was to defend document-page numbering as
provenance. The user pushed back: *if they picked 3 pages out of 7, those are pages 1, 2, 3
— that's what they selected.* The deciding evidence settled it against me:
`overview.pages` is `len(preprocess_result.page_images)`, the **result** count
([webapp/api.py:294](webapp/api.py#L294)), so `PageViewer` already renders "Page 1 / 3" for
that run. The grid was the surface deviating from the workspace's own convention, and the
two panes disagreed with each other. Badge now counts position; `alt` text and the
include-checkbox keep the true page identity so assistive tech and run scope stay exact.

Underneath sat a real defect the badge change would have *hidden*: `pagesFromSettings`
never clamped a range's `start` to the document length. Settings persist across uploads, so
a 1-page document carrying a stale `page_start: 3` gave `start=2`, `end=min(5,1)=1`, and
`Math.max(1, end - start)` **forced** a phantom index through. `webapp/settings.py`
`page_indices` had the identical bug, where it meant asking ingest to rasterize a page that
does not exist. Both clamped, both pinned by tests. Worth stating plainly: relative
numbering makes this *invisible*, not fixed — the phantom still reached the run scope.

**The popover's "text overflow" was a string-shape problem.** `remove_confirm` interpolated
the filename mid-sentence, so no amount of `truncate` could bound it without cutting the
sentence. Split into a bounded `subject` slot (`break-all line-clamp-2`) plus a generic
consequence line. The Khmer came from splitting the existing verified string at its own
seam, so nothing was newly authored. The button misalignment was likewise structural, not
spacing: `btnSmCls` is `h-6` and `dangerBtnCls` is `h-7`. (The brief asked for `h-8` /
`btnSmCls`, which are contradictory — `h-8` exists only on `primaryBtnCls`. Both are `h-7`
now, the height the tokens already wanted.)

**Find became a search.** New `lib/search.ts` owns one matching predicate, `cellMatches`,
called by both the counter and the grid's highlight — two implementations of "does this
match" is how a counter and its highlights drift apart. Matches are ordered table → row →
col; stepping wraps; navigation reuses the existing `onFocusCell` triage jump rather than
building a second scroll path. `Enter` now steps instead of firing the document-wide
replace: arming a bulk mutation on Enter-while-typing was defensible when the field was
replace-only, and is the wrong reflex now that it is a live search box.

The highlight is an **inset ring, not a background wash**. Confidence tints and the diff
tint are persistent trust signals that the analyst is searching *in order to check*; a
highlight that painted over them would hide the very thing being looked for. Rings compose
with any background.

**A test of mine was vacuous and is now the opposite.** I justified NFC-normalizing the
search with a Khmer round-trip test — and it passed trivially, because Khmer combining marks
all carry combining class 0, so `NFD === NFC` and canonical normalization never reorders
them. Verified directly rather than trusted. NFC still earns its place on the Latin side, so
it stays; but the Khmer test was replaced with one that **asserts the limitation**: a query
typed in a non-canonical mark order does not match. Stored cell text is normalized by
`utils/khmer_normalize.py` (cluster-aware reordering) which has no TypeScript twin, so this
gap is real and now visible in the suite instead of papered over by a green check.

Verified: 109 frontend tests (24 new), 823 backend (5 new), `tsc -b` and build clean. The
find-bar tests were mutation-checked — breaking the wrap and the cursor reset fails four of
them — after the NFC episode made "it passes" insufficient evidence that a test tests
anything.

---

### 2.78 The export arrow was a View Transition, not a CSS one (2026-07-23)

Three UI refinements. The interesting part is that two of the three briefs described the
wrong mechanism, and following them literally would have made the workspace worse.

**The "arrow artifact" had nothing to do with transitions.** The chevron trigger carried
`viewTransitionName: 'export-menu'` while closed, and the menu panel carried the same name
while open. The View Transitions API morphs the element holding a name in the old snapshot
into the element holding it in the new one — so the browser was being asked to stretch a
~28px chevron button into a 256px panel, and dutifully distorted the arrow on the way. The
morph *was* the artifact. No `transition-*` class was involved, and adding one would not
have touched it. Removed the shared name and the `withViewTransition` wrapper; the panel
already rose via `.overlay-enter` in `menuCls`, so the morph was never carrying the
animation, only damaging the icon.

**"Unify on 150ms ease-out" would have made everything slower.** The brief cited the delete
popover as the standard to copy — but that popover uses `.overlay-enter`, which is **90ms
expo-out**, not 150ms ease-out. The 150ms/ease-out figure is the `trans` token from
`ui.ts`, which governs hover and colour *state* changes on controls: a different concern
from entrance animation. Applying it to entrances meant +67% duration and a softer attack
on every menu — the opposite of "snappy". Raised it, and the user chose to keep the real
90ms profile. **Item 3 therefore required no timing change at all**: every floating surface
already shared `.overlay-enter`, so unifying meant deleting the one outlier (item 2), not
rewriting the vocabulary. The honest outcome was a smaller diff than the brief implied.

The audit did surface one real stray: the **help modal had no `.overlay-enter`**, animating
solely through its view transition. That meant ~250ms (the browser default) where the API
exists, and **no entrance whatsoever** in a browser without it, while every other overlay
animated. Folded onto the shared class.

**The label fix was a key split, not a string edit.** `remove_doc` served four roles — the
row button's `aria-label` (`"Remove document: foo.pdf"`), its `title`, the popover title,
and the popover action label. Only the last needed shortening; editing the shared key would
have degraded the screen-reader name to `"Remove: foo.pdf"`. The sibling case already
modelled the answer (`delete_all_title` / `delete_all_action`), so the single-document case
simply gained the action key it never had. Khmer left unchanged per the brief — it already
fits.

`whitespace-nowrap` went onto the shared `btnCls` / `btnSmCls` / `dangerBtnCls` /
`primaryBtnCls` tokens rather than the one button: these are fixed-height controls, a
wrapped label always breaks out of that height, and fixing it at the token is what stops the
next narrow container from rediscovering it.

Two small scope calls worth recording: `aria-expanded` was added to the export trigger
because the other three menu triggers already had it (it was the outlier), but
`aria-haspopup` was *dropped* after being briefly added — no trigger carries it, and adding
it to one would trade one inconsistency for another. `withViewTransition` survives at
exactly one call site, the grid⇄single canvas switch, which is a genuine single-region
cross-fade rather than a morph between two different elements.

Verified: 110 frontend tests (1 new, 1 updated — the updated one failed correctly against
the shortened label before being adjusted), `tsc -b` and build clean.

---

### 2.79 Kiri was never reading the whole cell — plus the metric that was hiding it (2026-07-23)

§2.75 closed off retuning the router's confidence and called for an independent signal. This
section answers a prior question: **why** is `surya_kiri` catastrophic on moc_gas and excellent on
ARDB? The answer is not "Khmer is hard" — it is a hard architectural limit, and it is partly
fixable. Four commits: `9d58ffb`, `2edfb54`, `2f92364`, `31c3ced`.

**Root cause.** Kiri is a fixed-input line recognizer. `ResizeKeepRatioPadNoCrop` scales a cell
crop to `IMG_H` then calls `crop((0, 0, IMG_W, IMG_H))` — so every pixel past `IMG_W/IMG_H`
(≈13.3:1) was **silently discarded before the model ran**. Measured on real cell crops extracted
from the review sheet:

| document | cells over 13.3:1 | native source | Kiri Khmer acc |
|---|---|---|---|
| ARDB bulletins | ~0% (median AR 2.2) | born-digital vector | **0.920** |
| budget TOFE | 6% | born-digital vector | 0.122 |
| **moc_gas** | **21%** (11/52) | **raster scan, ~124 DPI** | 0.133 |

Direct evidence — Kiri returned a fraction of each long cell while Surya returned all of it:
73ch→42ch (58%), 81ch→49ch (60%), 111ch→row lost entirely. Surya: 100% on every one. **This is
why the failure was *confident*** — Kiri was decoding a truncated image faithfully, so nothing in
its own confidence could reveal the missing text (§2.75's point, now with a mechanism).

**Fix (`31c3ced`).** Over-cap cells are split into fitting chunks, decoded, and rejoined. Cuts
snap to the widest blank gutter near each boundary — Khmer stacks and connects glyphs, so a blind
cut corrupts the character in *both* chunks; solid-ink cells fall back to a hard cut, still better
than discarding. Chunks join the existing batch. A cell's confidence is the **minimum** across its
pieces (the mean would let one clean chunk mask a garbled one). The cap is read from `CFG`, never
hardcoded.

| target | before | after |
|---|---|---|
| ardb0 / ardb1 | 0.959 / 0.920 | **bit-identical** — path correctly never fires |
| budget_p3 | khmer 0.122, cell 0.721, CER 0.164 | khmer **0.163**, cell 0.724, CER **0.157** |
| moc_gas_p1 | CER 0.458 | CER **0.424**; long-cell ratios 0.58/0.60 → **0.89/0.94** |

**The limit, stated plainly.** moc_gas exact-match cell accuracy is **unchanged at 0.232**. The
recovered characters come back *misrecognised*, because that PDF embeds a **1021×1440 raster
(~124 DPI)** which we render at 200 DPI — pure upscaling. Cell crops are ~46px tall interpolated
from ~29px native, and Khmer stacks diacritics above *and* below the baseline. One cell's CER even
worsened (0.47→0.53): the previously-absent tail is now present and wrong. **No DPI setting can
add information the scan does not contain.** Truncation was real and is fixed; on low-DPI scans
the ceiling is the input. That makes native raster DPI a *pre-flight* routing signal — knowable
from the PDF before any inference, unlike the two options §2.75 proposed (both need a second
engine run).

**The metric was hiding results (`2f92364`).** `numeric_cell_accuracy` read **0.000** for surya on
budget p4 while surya had recovered **184/184 GT numeric values** — perfect content recall. Cause:
17 columns emitted instead of 16, shifting every cell. This is the column twin of the §2.42
row-shift problem, and the column axis had no aligner. Every engine drifts ±1 column on these
pages, so a bake-off run before this fix would have ranked engines by column-count luck. Now
`_align_rows` is reused on the column axis (signatures built from already-aligned rows; monotonic,
so genuinely swapped columns still score wrong) and scoring happens at aligned intersections.
Signatures are digit-**folded** — alignment answers "which column is this?", not "is it correct?",
so `១២៣` vs `123` must still align. Effect: **8 of 54** stored records changed, all on the pages
that exposed it (surya numeric 0.000→0.955/0.975/1.000/1.000); **§2.73 is bit-identical**, those
documents reporting `col_alignment_rate 1.000` — they never had column drift, which is why the bug
hid there.

**Eval-set groundwork.** `9d58ffb` added script-independent structure metrics
(`row_alignment_rate`, `col_count_match` — a challenger with no Khmer can still own the best grid,
and every other metric conflates structure with recognition), a GT **circularity guard** (refusing
to score an engine against GT its own model family drafted — our moc_gas GT is Gemini-drafted, and
§2.75 documents exactly how confidently wrong that can be), and corrected `STAGE3_*`, which claimed
surya 0.17.1 with two `vikp/*` checkpoints when we run surya-ocr 0.20.0 (one model,
`datalab-to/surya-ocr-2`). `2edfb54` harvests **free, model-free** numeric+structure GT from
born-digital text layers via `find_tables()` — validated at **222/222 numeric cells** against the
hand-verified budget p3, yielding 5 pages / 711 numeric cells at zero human cost. Khmer is blanked
there (legacy mojibake), so those files declare `scoring_scope: numeric_and_structure` and the
harness prints "—" rather than a misleading number; the measured pessimistic bias from losing
row-alignment anchors (mean −0.03, ranking preserved) is recorded in the harvester docstring.

**Also measured, for the router work ahead.** Cross-engine agreement between `surya` and
`surya_kiri` gives P(error | engines agree) ≤ 0.6% with 99–100% error recall across all three
document classes, while disagreement rate separates them 33% (ARDB, Kiri correct) vs 76% (moc_gas,
Kiri wrong) — where self-confidence managed 0.009. `surya_kiri_vlm` is **disqualified** as a voting
partner: it shares Kiri, so it agrees on wrong answers (P(error|agree) 11.1% on moc_gas).

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
