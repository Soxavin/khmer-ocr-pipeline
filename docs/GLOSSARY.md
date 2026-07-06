# Glossary & Project Primer — Khmer OCR Pipeline

Plain-English reference for the terms, tools, and ideas in this project. Written so
you can answer "what is X and why?" quickly. For the formal decision history see
`PROJECT_LOG.md`; for how to run things see `OPERATIONS.md` and `eval/README.md`.

---

## 1. The big picture (one paragraph)
We take **Khmer government financial documents** (e.g. GDDE daily market-price tables)
as PDFs/scans and turn them into **accurate digital text and spreadsheet data**, running
**100% locally** on an Apple-Silicon Mac (no cloud, no API keys). The hard part isn't
reading Khmer letters — the OCR does that ~90%+ correctly — it's **preserving table
structure** (which number belongs to which row/column).

---

## 2. The pipeline — how a document is read (stage by stage)
Like a person reading a page:

| Stage | Plain meaning | Module |
|---|---|---|
| **1. Ingest** | Turn the PDF into page images | `ingest.py` |
| **2. Preprocess** | Clean the image (straighten, de-stamp, contrast) | `preprocess.py` |
| **3. OCR** | Read the page (two sub-steps below) | `engines/surya.py` |
| **4. Post-process** | Tidy/normalize the text | `postprocess.py` |
| **5. Export** | Save as JSON / CSV | `export.py` |

**OCR** = *Optical Character Recognition* = getting text out of a picture. It has **two
sub-steps**, and the difference between them is the key to this whole project:

- **Layout detection** — *"WHERE are things?"* Draws boxes around regions (paragraphs,
  tables, headers). It finds **regions**, not words.
- **Recognition** — *"WHAT does it say?"* Actually reads the characters inside each region.

---

## 3. Tools & models (the proper nouns)

- **Surya** — the main open-source OCR toolkit we use (layout detection + recognition +
  table reading). Version 0.20.
- **VLM** — *Vision-Language Model*. An AI that "looks" at an image and outputs text/HTML.
  Surya uses one to read a table region and emit it as an HTML `<table>`. It's powerful but
  **struggles when handed one big, dense table image** (it loses small digits) — this is the
  core limitation we hit.
- **llama-server** — the background process that runs Surya's VLM on the Mac's GPU. It stays
  alive between runs for speed (you can stop it with `stop-metal-macos.sh`).
- **Tesseract** — a classic, simpler OCR engine. We tested it as a **baseline** (a known
  reference point to compare Surya against). Finding: it reads some text but produces **no
  table structure** and garbles Khmer — confirming we need a structure-aware approach.
- **Qwen** (`Qwen2.5-7B-Instruct` via **MLX**) — a general Large Language Model we *experimented*
  with to "correct" OCR text. It didn't help (it's not trained for Khmer OCR) and is slow, so
  it's now **off by default**.
- **MLX** — Apple's machine-learning framework for running models on Apple-Silicon GPUs.
- **SLANet** — a small specialist model whose *only* job is to find a table's **grid**
  (row/column count + the coordinates of every cell). New, promising — see §5.
- **Khmer normalizer** (`utils/khmer_normalize.py`) — deterministic text cleanup: fixes invisible
  junk characters, Unicode ordering, duplicate marks. This is what replaced Qwen as the
  default post-processing.

---

## 4. The central problem: "table fragmentation"
On a dense table, Surya's **layout** step makes a mistake: instead of seeing **one table**,
it draws **many separate boxes** over chunks of it (real page 2 → **8 boxes**, a 2-row-band ×
4-column grid). Recognition then reads each box **independently**, so it reads "all the names"
in one box and "all the prices" in another — **destroying which value belongs to which row**.

> Analogy: cut a spreadsheet into 8 strips, shuffle them, read each aloud. All the right
> words, but you can't tell that "Rice = 4,000".

The letters are right (~90%); the **structure** is wrong. That's the bottleneck.

### What we tried to fix it (the "stitcher" experiments — all documented, all OFF by default)
- **Geometric stitcher** (`engines/table_stitch.py`) — glue the fragmented boxes back together
  *before* reading.
  - **Master mode** — glue all into one big box → the VLM choked on the huge crop (worse). ❌
  - **Row-band mode** — glue into full-width horizontal strips → best geometric attempt,
    improved structure a bit, but still a tradeoff (read fewer values). ⚠️
- **Lesson:** fixing *where* the table is isn't enough — the VLM can't read a wide dense
  Khmer table in one shot.

### The promising direction: "Hybrid B"
Use **SLANet** to get the **grid + each cell's coordinates**, then hand each **tiny single
cell** to **Surya** to read. Small cells read accurately *and* we know exactly which cell each
value belongs to. Validated in a prototype (SLANet recovered a 27×9 grid vs the true 28×9 on
the worst page). "Hybrid" = combining two specialist models (SLANet for structure + Surya for
Khmer text). This is the planned next build.

---

## 5. Evaluation & metrics (how we measure "better")
We score the OCR output against hand-checked **ground truth** using free, deterministic
metrics (no paid AI judge). Tools: `evaluation/run_benchmark.py` (runs OCR over a dataset),
`evaluation/analyze_benchmark.py` (summarizes numbers), `evaluation/visualize_benchmark.py` (makes charts).

| Term | Plain meaning | Direction |
|---|---|---|
| **Ground truth (GT)** | The correct answer we compare against (hand-verified) | — |
| **CER** | *Character Error Rate* — % of characters wrong | lower = better |
| **Cell_Accuracy** | Did the **right value land in the right cell**? (most important for tables) | higher = better |
| **Cell_Content_Recall** | Did we capture the value **at all** (even if misplaced)? | higher = better |
| **Table_CER / Text_CER** | CER measured over table text / body text separately | lower = better |
| **Document_CER** | CER over *all* page text pooled together | lower = better |
| **Paragraph_Recall** | Fraction of expected paragraph lines found | higher = better |
| **Paragraph_Leak** | Body text wrongly captured inside a table (a layout bug signal) | lower = better |
| **Tables_Found vs Expected** | How many tables detected vs how many really exist (the **fragmentation** signal) | should match |

> Tip for the mentor conversation: for *financial tables*, **Cell_Accuracy** and
> **Tables_Found vs Expected** matter more than raw CER, because a number in the wrong cell is
> worse than a slightly misspelled word.

> For the exact **formulas** and the *how/why* of each metric (plus the NFC / title-strip / row-alignment
> steps applied before scoring), see [`eval/README.md` §5.1](../eval/README.md#51-how-each-metric-is-computed--and-why-with-formulas).

---

## 6. Architecture & workflow concepts
- **Engine registry** (`engines/engine_registry.py`) — a "swappable socket" so we can switch OCR
  engines (`run_surya`, `run_tesseract`, future `hybrid`) via the `OCR_ENGINE` setting without
  touching the rest of the code. Every engine returns the **same shape of result**.
- **Eval harness** (`eval/`) — datasets + saved benchmark runs. Each run is one timestamped
  folder with results + a `manifest.json` (records *what/when/which code version*) so results
  are reproducible and citable.
- **Synthetic vs real data** — *synthetic* = computer-generated test tables (clean, known
  answers); *real* = actual GDDE PDFs (messy, the true test). The gap between them is a key
  thesis finding.
- **A/B test** — run the pipeline two ways (feature on vs off) and compare the metrics.
- **Deterministic** — same input always gives the same output (important for trustworthy
  measurement; note: the *OCR model itself* has slight run-to-run variance, so we lean on
  structural metrics).

---

## 7. Hardware terms
- **Apple Silicon / M4 Pro** — the Mac's chip. **24 GB unified memory** = RAM and GPU memory
  are shared, so big jobs can run out of room.
- **Metal / MPS** — Apple's GPU interface that Surya/PyTorch use for speed.
- **Per-page bounded memory** — we process pages one at a time and clear memory between them,
  so memory use stays low regardless of page count (stress-tested: 10 pages @ 300 DPI ≈ 2 GB).

---

## 8. Where things live (file map)
```
src/khmer_pipeline/
  ingest.py / preprocess.py / postprocess.py / export.py   # 4 of the 5 stages (top level)
  engines/surya.py          # the 5th stage: Surya OCR
  engines/engine_registry.py   # swappable OCR engine socket
  engines/tesseract_engine.py  # the Tesseract baseline engine
  utils/khmer_normalize.py  # deterministic Khmer text cleanup
  engines/table_stitch.py   # the (default-off) fragmentation stitcher experiments
  model_config.py           # model names + tunable thresholds
  evaluation/run_benchmark.py / analyze_benchmark.py / visualize_benchmark.py    # evaluation
app.py                      # the Streamlit web UI
eval/                       # datasets + benchmark runs + eval/README.md
docs/                       # PROJECT_LOG, OPERATIONS, FINAL_SPRINT_PLAN, this GLOSSARY
fonts/                      # vendored Khmer fonts (for reproducible synthetic data)
setup-metal-macos.sh / stop-metal-macos.sh   # start/stop the OCR backend
```

## 9. Common commands
```bash
source setup-metal-macos.sh                      # start the OCR backend (do this first)
uv run streamlit run app.py                      # launch the web UI
uv run python -m khmer_pipeline.evaluation.run_benchmark    # run the benchmark
uv run python -m khmer_pipeline.evaluation.analyze_benchmark   # summarize the latest run
uv run pytest -q                                 # run the test suite
bash stop-metal-macos.sh                         # stop the OCR backend when done
```
