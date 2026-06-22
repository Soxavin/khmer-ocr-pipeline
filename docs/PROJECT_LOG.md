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
