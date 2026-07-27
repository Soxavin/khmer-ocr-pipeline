# Engine Evaluation — what we run, why, and what we measured

A standing reference for the OCR-engine question: *what is the best engine for GDDE financial
documents, and can anything beat or complement Surya?* Chronological detail lives in
`PROJECT_LOG.md` (§2.73–§2.85); this file is the summary you can hand someone cold.

Last updated: 2026-07-23.

---

## 1. The engines we ship

Every engine is a combination of two model types with opposite strengths:

- **Surya 2** (`surya-ocr` 0.20.0) — a 650M page-level vision-language model. Understands layout,
  placement, digits and Latin; **weak on dense Khmer** (~0.23–0.47 Khmer cell accuracy). Being
  generative, it is also **non-deterministic** on multi-page documents (±0.09 cell accuracy).
- **Kiri** (vendored CTC recognizer) — a per-cell line model, **fine-tuned for Khmer** (~0.92 on
  clean born-digital pages). Blind to layout; someone else must find the cells. Deterministic.

| engine key | structure from | text from | best for |
|---|---|---|---|
| `surya` | Surya VLM | Surya VLM | wide / number-heavy / scanned tables |
| `surya_kiri` | TableRec (geometry) | **Kiri**, every cell | narrow born-digital Khmer tables (ARDB bulletins) |
| `surya_kiri_vlm` | Surya VLM | Surya VLM, Kiri patches Khmer cells | mixed; slowest; least predictable |
| `auto` | — routes between the above — | | default |

`hybrid`, `tesseract` are superseded experiments kept for comparison.

## 2. Measured scorecard (verified GT)

Numbers are median of ≥1 run; ARDB is document-level, others per-page. See `eval/runs/`.

| document | class | surya | surya_kiri | best |
|---|---|---|---|---|
| ARDB 09.06 / 15.06 | born-digital, narrow, Khmer-dense | 0.58–0.64 | **0.959** | surya_kiri |
| budget p3 (TOFE) | born-digital, wide, numeric | **0.971** | 0.721 | surya |
| moc_gas p1 | **scanned ~124 DPI**, wide | **0.750** | 0.232 | surya |

The single most important fact: **no one engine wins everywhere.** `surya_kiri` owns the ARDB
bulletins (the daily common case) and loses badly on scans; `surya` is the reverse. This is why
`auto` exists and why the routing (below) matters more than any single-engine score.

## 3. Routing — `auto` (PROJECT_LOG §2.73, §2.75, §2.81)

`auto` decides per document. Two signals, checked in order:

1. **Pre-flight scan detection** (§2.81). A low-resolution raster scan is knowable from the PDF
   *before any inference* (`ingest.page_is_scanned`). Such documents route straight to `surya` —
   per-cell recognition cannot resolve Khmer diacritics at scan density. This is what fixed the
   original bug: on moc_gas, `auto` went **0.232 → 0.786** cell accuracy.
2. **Confidence fallback**. Otherwise run `surya_kiri`; if too many cells are low-confidence,
   re-run with `surya`. Works on budget p3 (correctly falls back).

Why not a single confidence threshold? Measured (§2.75): Kiri reports 0.222 low-confidence on the
ARDB page it *wins* and 0.231 on the moc_gas scan it *destroys* — 0.009 apart. A self-reported
confidence **cannot** detect a recognizer that is confidently wrong; the geometric pre-flight
signal can.

## 4. Engine fixes made along the way

- **Kiri wide-cell chunking** (§2.79). Kiri's fixed 48×640 input silently discarded any cell
  wider than ~13.3:1 (21% of moc_gas cells). Now over-wide cells are split at ink gaps and read in
  full. ARDB bit-identical; budget p3 Khmer 0.122 → 0.163. Does **not** rescue scanned pages —
  there the ceiling is source resolution, not truncation.
- **Auto-DPI logo bug** (§2.81). `resolve_auto_dpi` mistook a masthead logo for a page scan and
  returned 300 for every document. Fixed: a raster must cover ≥50% of the page to count.

## 5. Evaluation apparatus (so future results are trustworthy)

- **Script-independent structure metrics** (`row_alignment_rate`, `col_alignment_rate`,
  `col_count_match`) — a challenger with zero Khmer can still be judged on grid quality.
- **Column alignment** (§2.79) — a one-column offset used to zero every metric; an engine with
  184/184 perfect numeric recall was scoring 0.000. Fixed by aligning columns like rows.
- **GT circularity guard** (`gt_provenance.py`) — refuses to score an engine against ground truth
  its own model family drafted (our moc_gas GT is Gemini-drafted).
- **Free numeric GT** (`scripts/harvest_textlayer_gt.py`) — 711 model-free numeric cells harvested
  from born-digital PDF text layers, validated 222/222 against hand-verified budget p3.
- Score anything with `scripts/compare_engines_ab.py` (per-page GT, `--repeat` medians, stored
  grids, `--rescore`).

---

## 6. Challenger bake-off

**Framing set before testing** (GlotOCR Bench, arXiv 2604.12978): even frontier OCR models fail
beyond ~30 scripts, and Khmer is named among the under-served. So the realistic win from a
challenger is **structure + numbers + spans**, with Khmer text still coming from Surya or Kiri.
A model that owns the grid but not the Khmer is a *complement* success, not a rejection.

| challenger | size | status | verdict |
|---|---|---|---|
| **dots.ocr** | 3.0B | tested (Stage A) | **Not proven on Apple Silicon — see below.** Runs, but only one adverse data point completed. |
| PaddleOCR-VL-1.5 | 0.96B | queued | 3× smaller than dots.ocr; best local candidate to fit where dots.ocr OOM'd |
| Gemini (Flash) | API | blocked | needs `GEMINI_API_KEY`; free tier; **budget/ARDB only** (circularity guard) |
| Mistral OCR | API | queued | ~$0.001/page |
| DeepSeek-OCR-2 | 6.8GB (cached) | deprioritised | SAM encoder + 64-expert MoE — higher MPS risk than a dense model |

### dots.ocr — the honest verdict (§2.85)

**Why it was the most interesting candidate.** It emits tables as **HTML**, which our
`_parse_html_table_with_spans()` already consumes, and HTML natively carries `colspan`/`rowspan` —
the capability Surya 2 v0.20 *removed* (the §2.40 split-header limitation). Its own published
benchmarks (OmniDocBench, XDocParse) are genuinely strong.

**What we measured** (M4 Pro, MPS, one complete run — moc_gas table crop, full resolution):

| engine | cell | numeric | Khmer | structure |
|---|---|---|---|---|
| surya | **0.750** | **0.939** | **0.467** | — |
| **dots.ocr** | 0.286 | 0.333 | 0.133 | col-align **1.000**, row-align 0.929 |
| surya_kiri | 0.232 | 0.242 | 0.133 | — |

**Why this is NOT a fair verdict on dots.ocr, stated plainly:**

1. **Its benchmarks don't isolate Khmer.** Strong on broad (English/Chinese-heavy) document
   parsing ≠ strong on Khmer, the one script we need and the one GlotOCR flags as hard.
2. **The one complete score was our hardest document** — the 124-DPI scan, where even a Khmer
   *specialist* (Kiri) collapses to 0.13. dots.ocr actually **beat Kiri** here. Its structure
   score (col-align 1.000) confirms the GlotOCR split: **layout generalises across scripts,
   recognition does not.**
3. **The page it should ace — born-digital budget — OOM'd before finishing.** So dots.ocr was
   never measured on its best-case document type for us.
4. **MPS has documented correctness risk** and we ran it **off-label** as a table-crop recognizer,
   not its designed full-page mode.

So 0.286 is a **lower bound under adverse conditions**, not dots.ocr's ceiling. *Is it just not
for us?* — Unknown, honestly. It is not a drop-in Khmer win (nothing is), but whether it
complements Surya on structure/spans is **untested fairly**.

**Why we stopped rather than pushing the MLX port** (which was authorised): the MLX fallback fixes
speed and memory, but our blocker was **accuracy**, and a faster runtime cannot make a model read
Khmer better. The right way to test dots.ocr fairly is **Colab's T4** (CUDA, flash-attention,
full resolution, born-digital pages, full-page mode) — a notebook round-trip, not a local fix. If
a future session wants to settle the question, that is the experiment.

**Integration notes for whoever retries it** (each cost a debug cycle, none are on the model card):
`AutoProcessor` breaks under transformers 4.57 (build tokenizer + image processor directly); it
uses its own `<|user|>…<|endofuser|><|assistant|>` chat format, not Qwen's; image tokens are
`<|imgpad|>` × `grid_thw.prod()/merge_size²`; and the **vision tower has a separate
`attn_implementation`** that defaults to flash-attention → silently falls back to eager on Mac →
OOM. Working spike: `scratchpad/spike_dots_table.py`.

### Structural lesson for all VLM challengers

Autoregressive HTML generation costs output tokens proportional to **cell count**. A 34×16 budget
table is 544 cells, so these models scale worst exactly where our documents are hardest — the
opposite of what we want. Cloud APIs (no local memory) and the smallest local model (PaddleOCR-VL,
0.96B) are therefore tried before any larger local VLM.

---

## 7. Cloud engines — setup and the free-tier reality

Cloud engines are **opt-in and clearly labelled**: they send the page image to a third party. The
picker groups engines under **Local** / **Cloud** so an analyst never selects one by accident, and
the guidance text says plainly *"do not use for confidential documents."*

### Gemini (Google) — shipped engine + setup

**Status:** shipped as the `gemini` engine (`engines/gemini_engine.py`, §2.91), selectable in the
UI under **Cloud**. Model is `GEMINI_MODEL` (env), default `gemini-flash-latest`; set
`GEMINI_MODEL=gemini-flash-lite-latest` for higher quota / lower latency.

**First scored result (provisional — n=1, one page, one run):** `gemini-3.6-flash` on budget p3
(born-digital, hand-verified GT):

| engine | cell | numeric | Khmer | spans | secs |
|---|---|---|---|---|---|
| surya | 0.971 | **1.000** | 0.673 | none | ~50 |
| **gemini-3.6-flash** | 0.963 | 0.977 | **0.694** | **13 attrs** | 16 |
| surya_kiri | 0.721 | 0.550 | 0.122 | — | 27 |

Competitive with Surya on the born-digital best case, slightly ahead on Khmer, and it **restores
colspan/rowspan** (Surya 2 dropped them). Caveats before trusting: **n=1** on a single page;
Gemini's output schema is non-deterministic (two earlier calls varied bbox shape and key names —
`category`/`bbox` vs `label`/`box`), so run-to-run spread is unknown; and the decisive tests are
still open — **ARDB** (Khmer-dense, where surya_kiri hits 0.92) and **scanned** pages. moc_gas
cannot score Gemini (its GT is Gemini-drafted — circularity guard).

Is it free? **Free in money, not in data.** The free tier needs no credit card (~250–1500
requests/day, ~10–15/min on Flash), but **Google uses free-tier inputs and outputs to improve its
models, and human reviewers may see them** — the terms warn against confidential data. The paid
tier is data-private but requires billing. We ship the free tier *with the cloud label*; the eval
documents are already-published public bulletins, so benchmarking on it discloses nothing new.

Steps (one-time, ~3 minutes):
1. Go to **https://aistudio.google.com** and sign in with a Google account.
2. **Get API key** (left sidebar) → **Create API key** → let it create a new Google Cloud project.
   No credit card.
3. Copy the key and export it where the tools read it (mirrors the existing `OPENAI_API_KEY`
   pattern in `src/khmer_pipeline/evaluation/evaluate_judge.py`):
   ```bash
   export GEMINI_API_KEY="AIza…"        # add to your shell profile or a local .env
   ```
   The webapp backend reads the same variable from its environment.
4. Install the SDK (optional dependency group, like `openai`):
   ```bash
   uv add --optional eval-extras google-genai
   ```
5. Verify with the spike (public docs only — the moc_gas GT is Gemini-drafted, so the circularity
   guard refuses it):
   ```bash
   uv run --extra eval-extras python scripts/spike_gemini.py \
       --image eval/datasets/real/CambodiaBudgetExecutioninApr-2024_p3.png
   ```
   It runs one page, scores it against local GT via the shared HTML-table parser, and refuses any
   Gemini-drafted GT (moc_gas) so a circular score can't slip in.

Model: pin **`gemini-2.5-flash`** for reproducible benchmarking (the `-latest` alias drifts).

**EEA/UK/Switzerland exception:** the paid-services data terms apply to the free tier there too, so
the free tier will not activate without billing. Not relevant in Cambodia, noted for completeness.
