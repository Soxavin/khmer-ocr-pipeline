# Project Engineering Log — Khmer OCR Pipeline

A curated record of the significant problems, root causes, design decisions, and
results during development. Intended as a reference for documentation and report
writing — it captures *why* the system looks the way it does, not an exhaustive
commit history. Newest milestones are toward the bottom of each section.

---

## 1. Overview

**Goal.** Extract structured data from Khmer-language financial/economic documents
(ARDB-style price tables, budget execution reports) into one CSV per table and one
JSON per document, for analysts at MEF Cambodia. A working prototype — no model
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

- **Problem.** On dense real pages Surya's *layout* model shatters one table into many
  regions (real MEF page 2 → a 2 row-band × 4 col-group grid of **8 Table boxes**);
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

### Real-document results (first real MEF doc, 2026-06-22)

A real born-digital MEF daily market-price PDF (3 pages, dense Khmer price tables),
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
