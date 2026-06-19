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

## 7. Conventions

- **Never edit `results.csv` in place.** If a run is wrong, create a new one.
- **New model = new run folder.** Register it via `engine_registry.py` and run normally.
- **Cite results by `run_id`** (folder name) — it encodes the timestamp and engine so references are unambiguous.
- `eval/datasets/` contents are gitignored — regenerate from the generators above.
- `eval/runs/` are gitignored — reproduce from the same code + datasets.
- This `eval/README.md` is the only committed artifact in `eval/` — it documents the contract.
