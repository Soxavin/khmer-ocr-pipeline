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

**Prerequisites:** Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/). The pipeline is **cross-platform**
and auto-selects its compute device (`src/khmer_pipeline/utils/device.py`): **CUDA** on NVIDIA, **MPS** on Apple
Silicon, **CPU** otherwise. On Apple Silicon, `source setup-metal-macos.sh` additionally enables Surya's
faster llama.cpp **Metal** backend. Tesseract with the `khm` language pack is optional (only for the
Tesseract engine); on Linux, the Docker image below bundles it.

```bash
uv sync                                  # install dependencies (pyproject.toml + uv.lock)

# --- Review UI (analyst tool) ---
source setup-metal-macos.sh              # configure the Surya Metal backend (sets env vars)
uv run streamlit run app.py             # open the app, upload a document, click "Run Extraction"
uv run streamlit run lab.py             # (optional) researcher lab — compare engines + inspect pipeline stages

# --- Command line (batch) ---
uv run python -m khmer_pipeline.pipeline input.pdf output/ [--dpi 200] [--no-deskew] [--no-qwen]

# --- Tests ---
uv run pytest -q          # (installs dev extras with: uv sync --extra dev)
```

### Running with Docker (Linux / NVIDIA GPU)

For non-Mac deployment, a Dockerfile packages the whole runtime (Tesseract, OpenCV libs, Python deps —
`mlx` is auto-excluded off-Mac). The **same image** uses the GPU when one is passed, or falls back to CPU:

```bash
docker build -t khmer-ocr .
docker run --gpus all -p 8501:8501 khmer-ocr   # NVIDIA GPU (CUDA) — needs the NVIDIA Container Toolkit
docker run           -p 8501:8501 khmer-ocr   # CPU fallback (no --gpus)
```

The container logs the selected device (`[device] using cuda|cpu`) on first OCR, then serves the app at
`http://localhost:8501`.

> **Two lanes:** **Mac users run natively** (`setup-metal-macos.sh` → Metal/MLX) — Docker can't access the
> Apple GPU. **Docker is the Linux / NVIDIA / CPU lane.** Both share the same code; the device is picked
> automatically.

---

## Key results

On dense real Khmer tables, **preprocessing is what makes or breaks table structure.** Fed a *raw* page,
Surya's layout fragments one dense table into ~8 regions; after the pipeline's preprocessing (contrast +
table-background flattening) it detects the table as **one clean region** (reproduced on two separate
bulletins). Scored against a hand-verified 75×9 ground-truth table under production (preprocessed)
conditions, **plain Surya is the best engine** and recovers the exact table shape:

| Engine (preprocessed) | Cell accuracy | Recall | Table CER | Pred dims |
|--------|--------------|--------|-----------|-----------|
| **Surya** | **0.259** | 0.623 | **0.249** | **75×9 (= GT)** |
| Hybrid (SLANet + row-strip) | 0.145 | 0.569 | 0.258 | 82×9 |
| Hybrid + DocLayout-YOLO | 0.135 | 0.561 | 0.279 | 79×9 |

A structure-focused effort (a SLANet-based **hybrid** engine and a **DocLayout-YOLO** layout detector) was
built to fix the *raw-image* fragmentation — but once preprocessing is on, Surya already handles the
structure, so those alternatives are unnecessary and score worse. The remaining gap is **recognition**, not
structure. On that axis, an off-the-shelf A/B (per-page recognition CER, lower = better) found no turnkey
model beats Surya:

| Engine | Mean recognition CER |
|--------|----------------------|
| **Surya** | **0.316** |
| Hybrid (rowband) | 0.315 |
| Tesseract-`khm` | 0.576 |
| Qwen2.5-VL-7B (4-bit, local) | 2.271 (collapsed) |

— which points future work at fine-tuning a Khmer recognizer. Full methodology and the dated decision log
(including the preprocessing revision, §2.25) are in [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) and
[`docs/REPORT.md`](docs/REPORT.md).

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
