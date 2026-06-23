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
        ├── results.csv              # per-image rows (fixed schema, see run_benchmark.py)
        ├── manifest.json            # provenance record (what / on what / using what / by what)
        └── summary.txt              # captured analyze output for this run
```

`eval/datasets/` and `eval/runs/` are gitignored. This `README.md` is the committed source of truth for how the layout works.

---

## 2. Generate Datasets

Requires Playwright + Chromium (`uv run playwright install chromium` once).

```bash
# Isolated table images (5 fonts × N templates × count each)
uv run python -m khmer_pipeline.generate_synthetic_tables \
    --output-dir eval/datasets/synthetic_tables \
    --count 3

# Full-page document images
uv run python -m khmer_pipeline.generate_synthetic_documents \
    --output-dir eval/datasets/synthetic_documents \
    --count 3
```

Both generators default to the paths above (no `--output-dir` needed after migration).
Both include a **font-load guarantee**: `document.fonts.check()` is called after page load and aborts with an error if the intended Google Font did not render — no silent fallback-font images.

---

## 3. Run a Benchmark

Raw render, free, no API key required:

```bash
uv run python -m khmer_pipeline.run_benchmark
```

This creates `eval/runs/<YYYYMMDD_HHMMSS>_<engine>/` containing `results.csv`, `manifest.json`, and `summary.txt`.

Options:

| Flag | Description |
|---|---|
| `--data-dir DIR [DIR ...]` | Override dataset directories (default: `eval/datasets/synthetic_tables eval/datasets/synthetic_documents`) |
| `--run-dir PATH` | Use a specific run directory instead of the auto-named one |
| `--with-correction` | Run Stage 4 Qwen correction and use corrected text for `Text_CER` |
| `--resume` | Skip images already present in `results.csv` (safe to re-run after a crash) |

---

## 4. manifest.json Schema

Every run folder contains `manifest.json` answering "what / on what / using what / by what":

| Field | Type | Description |
|---|---|---|
| `run_id` | string | Folder name — unique identifier for this run |
| `timestamp_utc` | string | ISO-8601 UTC timestamp when the run finished writing the manifest |
| `engine` | string | `ACTIVE_OCR_ENGINE.__name__` (e.g. `run_surya`) |
| `correction` | bool | Whether `--with-correction` was passed |
| `preprocessing` | string | Always `"none (raw render)"` — raw PNG fed directly to OCR |
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
| `Cell_Content_Recall` | higher = better | Order-insensitive cell match: was each GT cell value found anywhere in the predicted row? Measures *what* content is present. Gap between `Cell_Content_Recall` and `Cell_Accuracy` reveals row shifts vs. garbled text. |
| `Table_CER` | lower = better | Levenshtein character error rate over the full table text (GT vs. predicted, concatenated). |
| `Text_CER` | lower = better | Levenshtein CER over the full page body text (excluding table cells). |
| `Paragraph_Recall` | higher = better | Fraction of GT paragraph lines found in the OCR output. |
| `Paragraph_Leak` | lower = better | Body text wrongly captured inside a table cell — a §2.4 layout-correctness signal. Should be 0 on clean runs. |

**Context:** all benchmarks run on raw renders (no preprocessing). This isolates OCR model quality from preprocessing effects. Preprocessing (`deskew`, `normalise_table_backgrounds`, etc.) is only applied in the live pipeline.

---

## 6. Compare Runs

```bash
# Analyze a specific run directory
uv run python -m khmer_pipeline.analyze_benchmark eval/runs/20260619_163242_run_surya/

# Analyze a specific CSV
uv run python -m khmer_pipeline.analyze_benchmark eval/runs/20260619_163242_run_surya/results.csv

# Default: latest run under eval/runs/ (no args)
uv run python -m khmer_pipeline.analyze_benchmark

# Compare two runs side by side (per-Engine section shows both)
uv run python -m khmer_pipeline.analyze_benchmark \
    eval/runs/20260619_163242_run_surya/ \
    eval/runs/20260620_100000_run_other/
```

---

## 7. Figures (visualize_benchmark)

Generate publication-quality charts (matplotlib) from one or more run dirs. Used for
the thesis report. `matplotlib` lives in the **dev** extra (`uv sync` resolves it).

```bash
# One run
uv run python -m khmer_pipeline.visualize_benchmark eval/runs/<run>/ --out eval/figures

# Two runs (enables the comparison charts)
uv run python -m khmer_pipeline.visualize_benchmark \
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
uv run python -m khmer_pipeline.inspect_pdf path/to/real_docs/ --output inspect_report.json
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
uv run python -m khmer_pipeline.harvest_ground_truth path/to/doc.pdf \
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
uv run python -m khmer_pipeline.run_benchmark --data-dir eval/datasets/real
```

---

## 9. Conventions

- **Never edit `results.csv` in place.** If a run is wrong, create a new one.
- **New model = new run folder.** Register it via `engine_registry.py` and run normally.
- **Cite results by `run_id`** (folder name) — it encodes the timestamp and engine so references are unambiguous.
- `eval/datasets/` contents are gitignored — regenerate from the generators above.
- `eval/runs/` are gitignored — reproduce from the same code + datasets.
- This `eval/README.md` is the only committed artifact in `eval/` — it documents the contract.
