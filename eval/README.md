# eval/ — Evaluation Artifacts

## 1. Purpose & Layout

All benchmark inputs and outputs live under `eval/`. One run = one folder.

```
eval/
├── README.md                        # this file — committed
├── datasets/                        # gitignored contents
│   ├── synthetic_tables/            # isolated table images (from generate_synthetic_tables)
│   └── synthetic_documents/         # full-page document images (from generate_synthetic_documents)
└── runs/                            # gitignored
    └── <YYYYMMDD_HHMMSS>_<engine>/  # one folder per benchmark run
        ├── results.csv              # per-image rows (fixed schema, see evaluation/run_benchmark.py)
        ├── manifest.json            # provenance record (what / on what / using what / by what)
        └── summary.txt              # captured analyze output for this run
```

`eval/datasets/` and `eval/runs/` are gitignored. This `README.md` is the committed source of truth for how the layout works.

---

## 2. Generate Datasets

Requires Playwright + Chromium (`uv run playwright install chromium` once).

```bash
# Isolated table images (5 fonts × N templates × count each)
uv run python -m khmer_pipeline.datagen.generate_synthetic_tables \
    --output-dir eval/datasets/synthetic_tables \
    --count 3

# Full-page document images
uv run python -m khmer_pipeline.datagen.generate_synthetic_documents \
    --output-dir eval/datasets/synthetic_documents \
    --count 3
```

Both generators default to the paths above (no `--output-dir` needed after migration).
Both include a **font-load guarantee**: `document.fonts.check()` is called after page load and aborts with an error if the intended Google Font did not render — no silent fallback-font images.

---

## 3. Run a Benchmark

Raw render, free, no API key required:

```bash
uv run python -m khmer_pipeline.evaluation.run_benchmark
```

This creates `eval/runs/<YYYYMMDD_HHMMSS>_<engine>/` containing `results.csv`, `manifest.json`, and `summary.txt`.

Options:

| Flag | Description |
|---|---|
| `--data-dir DIR [DIR ...]` | Override dataset directories (default: `eval/datasets/synthetic_tables eval/datasets/synthetic_documents`) |
| `--run-dir PATH` | Use a specific run directory instead of the auto-named one |
| `--with-correction` | Run Stage 4 Qwen correction and use corrected text for `Text_CER` |
| `--resume` | Skip images already present in `results.csv` (safe to re-run after a crash) |
| `--preprocess` | Apply the full `PreprocessConfig` stack (deskew/denoise/contrast/etc.) instead of the default raw render, matching the live pipeline (`app.py`/`pipeline.py`) |

---

## 4. manifest.json Schema

Every run folder contains `manifest.json` answering "what / on what / using what / by what":

| Field | Type | Description |
|---|---|---|
| `run_id` | string | Folder name — unique identifier for this run |
| `timestamp_utc` | string | ISO-8601 UTC timestamp when the run finished writing the manifest |
| `engine` | string | `ACTIVE_OCR_ENGINE.__name__` (e.g. `run_surya`) |
| `correction` | bool | Whether `--with-correction` was passed |
| `preprocessing` | string | `"none (raw render)"` by default (raw PNG fed directly to OCR), or `"full PreprocessConfig"` when the run used `--preprocess` |
| `git_commit` | string | Short git SHA of the repo at run time (`"unknown"` if not a git repo) |
| `git_dirty` | bool | `true` if the working tree had uncommitted changes |
| `versions` | object | `{"surya_ocr": "0.20.x", "python": "3.11.x"}` |
| `datasets` | array | One entry per `--data-dir`: `{"name", "path", "images"}` |
| `image_count` | int | Total images processed across all datasets |
| `aggregates` | object | Blank-skipping averages: `avg_cell_accuracy`, `avg_cell_content_recall`, `avg_table_cer`, `avg_text_cer` |

Example:

```json
{
  "run_id": "20260619_163242_run_surya",
  "timestamp_utc": "2026-06-19T16:32:42Z",
  "engine": "run_surya",
  "correction": false,
  "preprocessing": "none (raw render)",
  "git_commit": "c6d6523",
  "git_dirty": false,
  "versions": {"surya_ocr": "0.20.0", "python": "3.11.9"},
  "datasets": [
    {"name": "synthetic_tables", "path": "eval/datasets/synthetic_tables", "images": 15},
    {"name": "synthetic_documents", "path": "eval/datasets/synthetic_documents", "images": 15}
  ],
  "image_count": 30,
  "aggregates": {
    "avg_cell_accuracy": 0.643,
    "avg_cell_content_recall": 0.780,
    "avg_table_cer": 0.147,
    "avg_text_cer": 0.219
  }
}
```

---

## 5. Metric Definitions

All metrics are computed deterministically from ground truth — no LLM judge, no cost.

| Metric | Direction | Definition |
|---|---|---|
| `Tables_Found` / `Tables_Expected` | ratio → 1.0 | Table detection rate (Surya layout pass). `TabRatio` in analyze output. |
| `Cell_Accuracy` | higher = better | Fraction of cells matching ground truth positionally (row-aligned via `difflib.SequenceMatcher`). Measures *where* content lands. |
| `Cell_Content_Recall` | higher = better | Order-insensitive cell match: was each GT cell value found anywhere in the predicted table (multiset — duplicates count)? Measures *what* content is present. Gap between `Cell_Content_Recall` and `Cell_Accuracy` reveals row shifts vs. garbled text. |
| `Table_CER` | lower = better | Levenshtein character error rate over the full table text (GT vs. predicted, concatenated). |
| `Text_CER` | lower = better | Levenshtein CER over the full page body text (excluding table cells). |
| `Paragraph_Recall` | higher = better | Fraction of GT paragraph lines found in the OCR output. |
| `Paragraph_Leak` | lower = better | Body text wrongly captured inside a table cell — a §2.4 layout-correctness signal. Should be 0 on clean runs. |

### 5.1 How each metric is computed — and why (with formulas)

Source of truth: `src/khmer_pipeline/evaluation/evaluate_structure.py::evaluate_table`. Math notation
below matches that code exactly; formulas render on GitHub and in most Markdown viewers.

**Three fairness steps run first**, so we score OCR quality rather than artefacts:
- **NFC normalisation** (`_norm`) of every GT and predicted cell — Khmer can encode the same visual
  glyph with different codepoint orders; without NFC we'd count identical-looking text as wrong. Also
  collapses runs of whitespace.
- **Title-row strip** (`_strip_title_row`) — drops row 0 when it is a merged-colspan document title
  (first cell filled, rest empty); that band is not tabular data.
- **Monotonic row alignment** (`_align_rows`, `difflib.SequenceMatcher` over row signatures) — pairs
  GT rows to predicted rows around inserted/deleted rows, so one extra/missing row (e.g. a split
  header) does not cascade into "every row below is wrong."

**Notation.** After title-strip the GT grid has `R` rows × `C` cols. `ĝ(r,c)` / `p̂(r,c)` = NFC-normalised
GT / predicted cell strings; `A` = the set of aligned `(i,j)` GT→pred row pairs; `𝟙[·]` = 1 if true else 0.

#### `Cell_Accuracy` — exact positional match

$$\text{Cell\_Accuracy}=\frac{\displaystyle\sum_{(i,j)\in A}\ \sum_{c=1}^{C}\ \mathbb{1}\!\left[\hat g(i,c)=\hat p(j,c)\right]}{R\times C}$$

```text
                 Σ over aligned rows (i,j)  Σ over cols c   1[ ĝ(i,c) == p̂(j,c) ]
Cell_Accuracy = ───────────────────────────────────────────────────────────────────
                                      R × C   (total GT cells)
```

- **How:** denominator is the full GT cell count `R·C`; a cell scores 1 only if the normalised strings
  are **identical**. GT rows with no aligned predicted partner contribute 0 to the numerator, so
  missing/extra rows are penalised. One wrong glyph fails the whole cell.
- **Why:** strictest, most honest *"can an analyst trust this cell without editing it?"* For a financial
  table a cell is usable or it isn't — partial credit would overstate readiness.

#### `Cell_Content_Recall` — multiset content presence

Let `𝒢` = multiset of non-empty GT cell values, `𝒫` = multiset of all predicted cell values, and
`count_X(v)` = occurrences of value `v` in multiset `X`:

$$\text{Cell\_Content\_Recall}=\frac{\displaystyle\sum_{v\,\in\,\text{set}(\mathcal G)}\min\!\big(\text{count}_{\mathcal G}(v),\ \text{count}_{\mathcal P}(v)\big)}{|\mathcal G|}$$

```text
                       Σ over distinct GT values v:  min( count_GT(v), count_pred(v) )
Cell_Content_Recall = ────────────────────────────────────────────────────────────────
                                  |𝒢|   (number of non-empty GT cells)
```

- **How:** whole-table (not per-row) multiset overlap; `|𝒢|` = number of non-empty GT cells; the `min`
  makes duplicated values count only as often as they truly appear. **Position-independent.**
- **Why:** separates *recognition* from *placement*. A value read correctly but dropped one row over
  fails `Cell_Accuracy` yet still scores here — so **`Cell_Content_Recall − Cell_Accuracy` ≈ how much
  error is mis-placement (structure) vs mis-reading (recognition).**

#### `Table_CER` / `Text_CER` — character error rate

Flatten to one normalised string per side (`g` = GT, `p` = pred; table cells row-major for `Table_CER`,
page body text for `Text_CER`):

$$\text{CER}=\frac{\operatorname{Lev}(g,\,p)}{|g|}$$

where `|g|` = GT length in **Unicode codepoints** and `Lev` is Levenshtein edit distance (minimum
single-character insertions + deletions + substitutions):

$$\operatorname{Lev}(i,j)=\begin{cases}\max(i,j)&\min(i,j)=0\\[2pt]\min\begin{cases}\operatorname{Lev}(i-1,j)+1\\ \operatorname{Lev}(i,j-1)+1\\ \operatorname{Lev}(i-1,j-1)+\mathbb{1}[g_i\neq p_j]\end{cases}&\text{otherwise}\end{cases}$$

```text
       Lev(g, p)            Lev = Levenshtein edit distance (ins + del + subst)
CER = ───────────           |g| = length of GT string in Unicode codepoints
         |g|

Lev(i,j) = max(i,j)                          if min(i,j) == 0
         = min( Lev(i-1, j)   + 1,           otherwise  ← delete
                Lev(i,   j-1) + 1,                      ← insert
                Lev(i-1, j-1) + [ g_i != p_j ] )        ← substitute (0 if chars equal)
```

- **How:** codepoint-level; edge cases in `cer()` — 0 if both empty, 1 if GT empty but pred non-empty.
  (Codepoint-level is slightly stricter than grapheme-level for Khmer, where one glyph spans several
  codepoints; fine for relative comparison.)
- **Why:** shows *how close* a wrong cell is — a single-glyph slip is 1 edit in dozens of chars, so CER
  stays tiny even when `Cell_Accuracy` fails. Also the standard OCR metric, comparable across systems.

**Read the trio together — the divergence is the diagnosis:** high `Cell_Content_Recall` + low
`Cell_Accuracy` ⇒ a **structure/placement** fault; low `Table_CER` + low `Cell_Accuracy` ⇒ a
**systematic glyph** fault. No single metric distinguishes these; the three together do.

**Context:** most benchmarks run on raw renders (no preprocessing). This isolates OCR model quality from preprocessing effects. Preprocessing (`deskew`, `normalise_table_backgrounds`, etc.) is applied in the live pipeline (`app.py`, `pipeline.py`).

> **⚠ Raw isolates the model, but does NOT decide which engine to ship.** Engines respond to preprocessing
> very differently, so a raw ranking can invert under production conditions. **For engine selection, run
> `scripts/eval_document.py --preprocess`** to match the live pipeline. In the doc-level A/B this flipped the
> result — raw favoured the hybrid, but *preprocessed* Surya wins decisively and hits the exact GT
> dimensions (see `docs/PROJECT_LOG.md` §2.25). Raw stays the default for now so older numbers stay
> reproducible.

---

## 6. Compare Runs

```bash
# Analyze a specific run directory
uv run python -m khmer_pipeline.evaluation.analyze_benchmark eval/runs/20260619_163242_run_surya/

# Analyze a specific CSV
uv run python -m khmer_pipeline.evaluation.analyze_benchmark eval/runs/20260619_163242_run_surya/results.csv

# Default: latest run under eval/runs/ (no args)
uv run python -m khmer_pipeline.evaluation.analyze_benchmark

# Compare two runs side by side (per-Engine section shows both)
uv run python -m khmer_pipeline.evaluation.analyze_benchmark \
    eval/runs/20260619_163242_run_surya/ \
    eval/runs/20260620_100000_run_other/
```

---

## 7. Figures (visualize_benchmark)

Generate publication-quality charts (matplotlib) from one or more run dirs. Used for
the thesis report. `matplotlib` lives in the **dev** extra (`uv sync` resolves it).

```bash
# One run
uv run python -m khmer_pipeline.evaluation.visualize_benchmark eval/runs/<run>/ --out eval/figures

# Two runs (enables the comparison charts)
uv run python -m khmer_pipeline.evaluation.visualize_benchmark \
    eval/runs/<surya_run>/ eval/runs/<tesseract_run>/ --out eval/figures
```

PNGs (150 DPI) are written to `eval/figures/` (**gitignored** — regenerate as needed).
A chart that has no data for its inputs is **skipped with a printed reason**, never crashes.

| File | Chart | Renders when |
|---|---|---|
| `cer_by_dataset.png` | Grouped bars — mean `Document_CER`, `Text_CER`, `Table_CER` per Dataset | always |
| `accuracy_by_font.png` | Grouped bars — mean `Cell_Accuracy`, `Cell_Content_Recall` per Font | always |
| `table_fragmentation.png` | Paired bars per Dataset — `Tables_Expected` vs `Tables_Found` (the fragmentation signal: real docs show found ≫ expected) | always |
| `engine_comparison.png` | Paired bars per Dataset — `Document_CER` split by **Engine** (e.g. surya vs tesseract) | only if the passed runs contain ≥2 distinct `Engine` values |
| `correction_ab.png` | Paired bars per Dataset — `Document_CER` split by **Corrected** (raw vs corrected) | only if both `Corrected=True` and `Corrected=False` rows are present |

**Notes / caveats**
- **Labels are Latin-only by design.** Only `Dataset`, `Font`, and `Engine` are used as
  axis/legend labels; the Khmer `Template` column is never charted (matplotlib's default
  font renders Khmer as tofu boxes).
- **Combined multi-run charts mix engines.** When you pass two different-engine runs,
  the three always-on charts aggregate across *both* — read them per-engine with care, or
  run the generator on each run dir separately. Only `engine_comparison.png` separates by
  engine. (Empty/`""` metric cells — e.g. `Text_CER` for table-only data — are coerced to
  `None` and skipped, not treated as zero.)

---

## 8. Real Documents

Real MEF PDFs are stored under `eval/datasets/real/` following the same `*_ground_truth.json` schema used by synthetic datasets, so `run_benchmark` / `evaluate_structure` / `analyze_benchmark` work unchanged.

### Convention

```
eval/datasets/real/
├── <stem>_p<N>.png                # page render (born-digital or scan)
└── <stem>_p<N>_ground_truth.json  # ground truth (auto-harvested or hand-labeled)
```

`eval/datasets/` is gitignored — real documents are not committed to the repo.

### Step 1: Diagnose your PDFs

```bash
uv run python -m khmer_pipeline.datagen.inspect_pdf path/to/real_docs/ --output inspect_report.json
```

Each PDF is classified as:

| Classification | Meaning |
|---|---|
| `born_digital_unicode` | Has a real Khmer Unicode text layer — harvest automatically |
| `likely_legacy_encoded` | Text layer exists but uses Latin code points for Khmer glyphs (Limon/ABC legacy fonts). **CER metrics will be invalid** until text is re-encoded or the limitation is documented. This is a headline finding for the thesis. |
| `scanned_image_only` | No text layer; images only — must be hand-labeled |
| `mixed_or_unknown` | Ambiguous — inspect manually |

Thresholds used: `_MIN_TEXT_CHARS=100`, `_UNICODE_KHMER_RATIO=0.5`, `_LEGACY_KHMER_RATIO=0.15`.

### Step 2: Harvest born-digital PDFs

```bash
uv run python -m khmer_pipeline.datagen.harvest_ground_truth path/to/doc.pdf \
    --output-dir eval/datasets/real --dpi 200
```

This renders each page to `<stem>_p<N>.png` and emits `<stem>_p<N>_ground_truth.json` with paragraphs extracted from the text layer (NFC-normalized). `tables` is intentionally left empty (`[]`).

**After harvesting you must:**

1. **Verify paragraphs** — text-layer extraction is unordered and may merge or split lines. Edit each `_ground_truth.json` by hand to match the actual document.
2. **Hand-fill tables** — add `{"data": [[cell, ...], ...]}` entries to the `"tables"` list for every table on the page.
3. **Hand-label scanned pages** — for `scanned_image_only` PDFs, create the `_ground_truth.json` entirely by hand using the same schema.

### Ground-truth JSON schema (full-page documents)

```json
{
  "font_family": "real",
  "template": "<pdf-stem>",
  "document_type": "real",
  "paragraphs": ["paragraph text ...", "..."],
  "tables": [{"data": [["col1", "col2"], ["val1", "val2"]]}],
  "footer": ""
}
```

### Run benchmark on real documents

```bash
uv run python -m khmer_pipeline.evaluation.run_benchmark --data-dir eval/datasets/real
```

---

## 9. Conventions

- **Never edit `results.csv` in place.** If a run is wrong, create a new one.
- **New model = new run folder.** Register it via `engines/engine_registry.py` and run normally.
- **Cite results by `run_id`** (folder name) — it encodes the timestamp and engine so references are unambiguous.
- `eval/datasets/` contents are gitignored — regenerate from the generators above.
- `eval/runs/` are gitignored — reproduce from the same code + datasets.
- This `eval/README.md` is the only committed artifact in `eval/` — it documents the contract.
