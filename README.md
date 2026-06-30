# Khmer Document Extraction Pipeline

A **local-first OCR pipeline** that turns Khmer-language financial documents (PDFs / scans — e.g.
market-price bulletins and government budget reports) into **analyst-reviewable structured data** —
JSON, CSV, and Excel.

It pairs an automated extraction pipeline with a Streamlit review tool so a data analyst can correct the
machine's output before exporting. Everything runs **on-device** (no cloud APIs) so sensitive financial
documents never leave the machine.

> Personal internship R&D project (GDDE, Cambodia). It explores the best
> on-device workflow for Khmer table extraction; the evaluation harness backs the findings rather than
> being the goal itself.

---

## What it does

Documents flow through five stages (`IngestResult → PreprocessResult → SuryaResult → PostprocessResult →
ExportResult`):

| # | Stage | What happens |
|---|-------|--------------|
| 1 | **Ingest** | PDF / image → page images |
| 2 | **Preprocess** | OpenCV cleanup: deskew, stamp removal, sharpen, contrast, table-background flattening |
| 3 | **OCR** | Layout detection + Khmer recognition + table structure ([Surya](https://github.com/datalab-to/surya)) |
| 4 | **Post-process** | Deterministic Khmer text normalization (optional Qwen2.5-VL fallback) |
| 5 | **Export** | Document JSON + per-table CSV / Excel; multi-page tables stitched into one |

**Review UI (`app.py`):** upload → run → a side-by-side view (page image left, **editable tables**
right) → download Excel / CSV / JSON. Analysts can fix any cell, add/delete rows, and reset a table to
the original. Advanced OCR/preprocessing settings are tucked behind an expander so the common path stays
simple.

**Swappable engines:** Surya (default), Tesseract-`khm`, and a structure-aware **hybrid** (SLANet grid +
Surya row-strip recognition) for dense fragmented tables — selected via the `OCR_ENGINE` env var.

---

## Quickstart

**Prerequisites:** Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/), and Apple Silicon (Surya uses a
llama.cpp **Metal** backend). Tesseract with the `khm` language pack is optional (only for the Tesseract
engine).

```bash
uv sync                                  # install dependencies (pyproject.toml + uv.lock)

# --- Review UI ---
source setup-metal-macos.sh              # configure the Surya Metal backend (sets env vars)
uv run streamlit run app.py             # open the app, upload a document, click "Run Extraction"

# --- Command line (batch) ---
uv run python -m khmer_pipeline.pipeline input.pdf output/ [--dpi 200] [--no-deskew] [--no-qwen]

# --- Tests ---
uv run pytest -q
```

---

## Key results

The central finding: on dense real Khmer tables the bottleneck is **table structure / fragmentation, not
character recognition**. An off-the-shelf recognizer A/B (per-page recognition CER, lower = better;
see [`docs/REPORT.md`](docs/REPORT.md) §4.8):

| Engine | Mean CER | Notes |
|--------|----------|-------|
| **Surya** | **0.316** | best general recognizer |
| Hybrid (rowband) | 0.315 | ties overall; **wins on the dense table (0.667 → 0.288)** |
| Tesseract-`khm` | 0.576 | far behind on tables |
| Qwen2.5-VL-7B (4-bit, local) | 2.271 | collapsed — no off-the-shelf VLM beat Surya |

No turnkey off-the-shelf model beats Surya today, which motivates the queued fine-tuning work. Full
methodology, figures, and the fragmentation investigation are in
[`docs/REPORT.md`](docs/REPORT.md); the dated decision log is in
[`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).

---

## Repository map

| Path | What it is |
|------|------------|
| [`app.py`](app.py) | Streamlit review UI (the main user-facing tool) |
| [`src/khmer_pipeline/`](src/khmer_pipeline/) | The pipeline package — the 5 stages, swappable engines, synthetic-data generators, and the evaluation code |
| [`scripts/`](scripts/) | Research & evaluation one-offs — see [`scripts/README.md`](scripts/README.md) |
| [`docs/REPORT.md`](docs/REPORT.md) | Evaluation report (the write-up) |
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | R&D decision log (why things were built this way) |
| [`docs/`](docs/) | Also: `OPERATIONS.md`, `GLOSSARY.md`, `figures/`, and `superpowers/` (per-stage design specs/plans) |
| [`eval/`](eval/) | Evaluation harness + [`eval/README.md`](eval/README.md) (datasets/runs are gitignored) |
| [`tests/`](tests/) | pytest suite, mirroring `src/khmer_pipeline/` 1:1 |
| [`fonts/`](fonts/) | Khmer fonts for synthetic data (OFL-licensed) |
| `sample_data/` | **Gitignored** — real financial documents are never committed |
| [`CONTEXT.md`](CONTEXT.md) | Architecture deep-dive (stages, engine-swap design, memory management) |
| `setup-metal-macos.sh` / `stop-metal-macos.sh` | Start/stop the Surya Metal backend |

---

## Notes

- **Local-first / privacy:** all models run on-device; sensitive documents are never sent to a cloud
  service, and real data stays out of git (`sample_data/`, `eval/datasets/`, `eval/runs/` are ignored).
- **Architecture & conventions:** see [`CONTEXT.md`](CONTEXT.md) for how the stages fit together and how
  to swap engines.
