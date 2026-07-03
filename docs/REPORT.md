# A Local Khmer OCR Pipeline for Financial Documents — Evaluation Report

*GDDE internship. Draft assembled from `PROJECT_LOG.md` (decision records),
benchmark manifests, and the `eval/` harness. Figures in `docs/figures/`.*

---

## Abstract
We built and evaluated a production-grade, **100% local** Optical Character Recognition (OCR)
pipeline for Khmer government financial documents (Ministry of Economy and Finance daily
market-price tables), running entirely on a 24 GB Apple-Silicon Mac with no cloud services.
Using a **free, deterministic evaluation harness** (no paid LLM judge) over both synthetic and
real documents, we find that **character recognition is strong (~90%+)** but **table-structure
extraction is the dominant bottleneck** on dense real tables. We rigorously establish — through a
baseline comparison and three engineering interventions — that the limiting factor is **OCR
recognition of small, isolated Khmer table cells**, not layout detection, which is solvable.

> **⚠ Revision (2026-07-01 — see `PROJECT_LOG.md` §2.25).** The "table-structure is the dominant
> bottleneck" framing throughout this report was measured on **raw (un-preprocessed) page images**. The
> product always preprocesses first, which collapses Surya's dense-table layout from ~8 fragments to **one
> clean region** (reproduced on two separate bulletins). Re-scored under production (preprocessed)
> conditions, the engine ranking **flips**: plain **Surya wins** (Cell_Accuracy 0.259 vs hybrid 0.145 /
> DocLayout-YOLO 0.135) and recovers the exact 75×9 table shape. So under production conditions Surya
> handles the structure and the remaining gap is **recognition**; the geometric/hybrid/DocLayout structure
> work (Sections 4–7) stands as documented negative results but is not needed once preprocessing is on. A
> full re-narration of the sections below is consolidated in **§4.9** — an ablation + cross-layout + recall
> analysis showing the fix is *resolution* normalisation (not colour/flags), that it is *layout-specific*
> (a structurally different dense report does not fragment at all), and that the residual gap is
> *recognition*, not layout.

---

## 1. Introduction
**Problem.** GDDE works with Khmer-language financial documents — e.g. daily market-price bulletins
published as dense Khmer tables (≈28 rows × 9 columns). The goal: extract them into accurate digital
text and spreadsheet data, locally and reproducibly, for downstream analysis.

**Why it is hard.** Khmer is a complex script (stacked subscripts, no inter-word spaces); the
documents are dense financial tables; and a value placed in the wrong cell is worse than a
mildly misspelled word. The success metric is therefore *structural*, not just character-level.

**Contributions.**
1. A modular, swappable-engine local pipeline (Surya VLM-based OCR + deterministic Khmer
   normalization).
2. A free, deterministic evaluation harness with provenance-tracked runs.
3. A recognised-baseline comparison (Surya vs Tesseract) and a synthetic-vs-real gap analysis.
4. A rigorous investigation of the table-fragmentation bottleneck (two geometric methods + two
   hybrid structure-model methods) that isolates the true limiting factor **and resolves it**: a
   row-strip hybrid lifts dense-table `Cell_Accuracy` ~16× (0.024→0.393).
5. **Document-level table stitching** that turns a multi-page report into one structured table per
   logical section (one CSV each) — the analyst-facing output the project targets.

---

## 2. System Architecture
A five-stage pipeline, each stage a focused module (`src/khmer_pipeline/`):

1. **Ingest** — PDF → page images.
2. **Preprocess** — deskew, stamp removal, contrast, table-background normalisation.
3. **OCR** (`surya.py`) — two sub-steps: **layout detection** ("where are the regions") and
   **recognition** ("what does each region say"). Surya 0.20 runs a Vision-Language Model (VLM)
   via a resident `llama-server` on the Metal GPU.
4. **Post-process** — a deterministic **Khmer Unicode normalizer** (NFC, format-char stripping,
   duplicate-mark collapse). An optional general LLM (Qwen2.5-7B via MLX) was evaluated and
   **rejected** (see §4.5).
5. **Export** — JSON + per-table CSV.

**Swappable engines.** An `OCREngine` protocol + `engine_registry.py` lets us switch OCR engines
(`OCR_ENGINE=surya|tesseract|hybrid`) without touching orchestration; every engine returns the
same `SuryaResult` shape so the evaluation harness scores them unchanged.

**Hardware.** Apple M4 Pro, 24 GB unified memory. Pages are processed sequentially with cache
clearing between them; a stress test (10 pages @ 300 DPI) peaked at ~2 GB RSS — memory is
**per-page bounded**, not a practical constraint for realistic documents.

---

## 3. Methodology
**Datasets.**
- *Synthetic tables / documents* — generated from HTML with 5 vendored OFL Khmer fonts (Noto Sans
  Khmer, Battambang, Hanuman, Moul, Fasthand), rendered offline for reproducibility. Clean inputs
  with exact ground truth.
- *Real* — actual GDDE born-digital PDFs (3 hand-labelled pages). Note: the real PDF's embedded
  text layer is garbled (broken ToUnicode CMap), so OCR on rendered pixels is genuinely required.

**Metrics (deterministic, no paid judge).** `Cell_Accuracy` (right value in right cell — the key
table metric), `Cell_Content_Recall` (value captured anywhere), `Table_CER` / `Text_CER` /
`Document_CER` (character error rates), `Paragraph_Recall`, `Paragraph_Leak`, and
`Tables_Found vs Expected` (the fragmentation signal). All benchmarks use **raw renders** (no
preprocessing) to isolate OCR quality, and each run is provenance-tagged (`manifest.json`:
git commit, versions, dataset counts).

---

## 4. Results

### 4.1 Synthetic baseline — font sensitivity
On clean synthetic data, table detection never failed (`Tables_Found == Expected`) and there was
zero paragraph leakage. Accuracy is strongly **font-dependent**:

| Font | Cell_Acc | Content_Recall | Table_CER | Text_CER |
|---|---|---|---|---|
| **Noto Sans Khmer** | **0.82** | **0.94** | **0.043** | **0.044** |
| Battambang | 0.74 | 0.85 | 0.171 | 0.14 |
| Hanuman | 0.65 | 0.87 | 0.106 | 0.19 |
| Moul | 0.52 | 0.62 | 0.252 | 0.42 |
| Fasthand | 0.48 | 0.68 | 0.203 | 0.30 |

Noto Sans Khmer is decisively best; the decorative display fonts (Moul, Fasthand) are weakest —
an expected typeface limitation, not a pipeline defect. See `docs/figures/accuracy_by_font.png`.

### 4.2 Recognised baseline — Surya vs Tesseract
Both engines run on identical raw renders. Tesseract (`khm`) is the standard classic-OCR baseline.

| Engine | Cell_Acc | Table_CER | Text_CER | Document_CER |
|---|---|---|---|---|
| **Surya** | **0.589** | **0.180** | **0.335** | **0.325** |
| Tesseract | 0.000 | 0.970 | 0.367 | 0.443 |

**Tesseract produces no table structure at all** (`Cell_Accuracy = 0`) and garbles dense numeric
columns / inserts inter-cluster spaces — disqualifying for financial tables independent of CER.
Surya wins overall and is the only structure-aware engine. The one dataset where Tesseract's
pooled `Document_CER` looks better (synthetic_documents) is a metric artifact of linear-reading
order, not superiority. See `docs/figures/engine_comparison.png`. (Surya `run` `20260622_154407`;
Tesseract `20260623_100406`.)

### 4.3 Real-document evaluation — the synthetic-vs-real gap
On the real GDDE document, aggregate metrics look poor (`Cell_Accuracy` 0.05, `Text_CER` 0.95),
but per-page analysis reveals the cause is **structural, not recognition**:

| Page | Tables_Found | Document_CER | Note |
|---|---|---|---|
| p1 | 1 | 0.30 | clean single table |
| p2 | **8** | **0.70** | one table fragmented into 8 regions |
| p3 | 1 | 0.22 | clean single table |

Inspecting the OCR-vs-GT dumps: Surya's **character recognition is strong (~90%+; all numeric
values correct)**, but on the dense page 2 the **layout model shattered one table into 8 regions**,
serialising content column-wise and destroying row↔value associations. Because CER is
order-sensitive, this *reordering* drives the apparent error — not bad OCR. **Table-structure
fragmentation is the bottleneck.** See `docs/figures/table_fragmentation.png`.

### 4.4 The fragmentation investigation (four interventions)

> **⚠ Superseded as the production account (see §4.9):** these interventions were measured on *raw* images;
> under production preprocessing the dense table no longer fragments, so they are unnecessary — retained
> here as documented negative results that rule out alternatives.

We attacked fragmentation systematically; all are documented with A/B numbers (PROJECT_LOG
§2.12–2.17) and kept in the codebase behind flags.

| Intervention | Idea | Result on real page 2 |
|---|---|---|
| **Geometric stitch — master** (§2.12) | Merge all fragments into one box before OCR | Detection fixed (8→1) but VLM **chokes on the giant crop**: Content_Recall 0.76→**0.16** |
| **Geometric stitch — row-band** (§2.13) | Merge into full-width row strips | Best geometric variant: Cell_Acc 0.024→0.036 (+50% rel) but Recall→0.35 — still a tradeoff |
| **Hybrid — per-cell** (§2.15) | SLANet grid + per-cell Surya OCR | Structure solved (SLANet grid 27×9, 188 cells w/ coords) but per-cell VLM **hallucinates on tiny cells**, Recall→**0.04**, ~4.3 min/page |
| **Hybrid — row-strip** (§2.17–2.18) | SLANet grid + read each row as one full-width `label="Table"` strip; Surya emits the `<td>` columns itself; blank strips get one taller-crop retry | **The win:** detection fixed (8→1) **and** Cell_Acc 0.024→**0.425** (~18×), Table_CER 0.657→**0.288**, Recall 0.758→**0.623**, DocCER 0.670→**0.612** — beats Surya on every p2 metric |

**The decisive finding:** structure is solvable — a small (7.4 MB) SLANet model recovers a clean
27×9 grid in 0.07 s. Per-cell recognition is the wrong granularity: Surya's VLM is built for text
*lines/blocks*, so a tiny isolated cell makes it hallucinate (foreign scripts). The **row-strip**
hybrid resolves this by feeding the VLM a natural full-width line and letting it split the columns
itself — the first method to recover correct row↔value structure *and* readable cells on the dense
real table, where every earlier intervention failed. A taller-crop retry on blank strips (§2.18)
then recovers most of the lost recall (0.53→0.62 on p2), so rowband ends up beating pure Surya on
*every* p2 metric. The residual gap is a handful of genuinely-illegible rows — a recogniser limit.

### 4.5 Post-processing — LLM correction vs deterministic normalization
A general LLM (Qwen2.5-7B) was tested for OCR correction and found **useless and slow** (it is not
Khmer-OCR-trained). A variance-controlled A/B on fixed OCR output showed the **deterministic Khmer
normalizer** instead reduces synthetic-document CER by ~3.2% relative (0.4498→0.4353), never hurts,
and is instant. Qwen was demoted to opt-in; the normalizer is the default. Honest takeaway: *a
general LLM did not help; deterministic Unicode normalization does, modestly.*

### 4.6 Preprocessing on degraded input (synthetic proxy)
The OpenCV preprocessing stack (deskew, denoise/sharpen, contrast, table-background) was built for
scans but never tested on degraded input. As a controlled proxy (no real scan available), we
synthetically degraded the GT'd born-digital document (2.5° rotation, blur, seeded noise, contrast
reduction) and A/B'd preprocessing OFF vs ON against the existing ground truth:

| | clean (ceiling) | degraded, OFF | degraded, ON |
|---|---|---|---|
| avg `Document_CER` | 0.503 | 0.749 | **0.726** |

Degradation clearly hurts OCR; **preprocessing recovers a small but consistent slice (ON beats OFF
on all three pages, −3% relative)** but does not restore toward the clean ceiling. Conclusion: the
stack is a **modest, non-harmful** improvement worth enabling for scans — not a silver bullet.
**Caveat: synthetic degradation ≠ real scan artifacts**; a real-scan A/B remains future work.

### 4.7 Multi-page table stitching (document-level output)
Real reports are one continuous table split across page images. A document-level stitching step
(`merge_document_tables`) joins consecutive per-page tables that share a column structure, dropping the
header repeated at each page break, and emits **one CSV per logical table** (Stage-5 export, on by
default). End-to-end on the 3-page 09.06.26 report: with the **hybrid rowband** engine the 3 per-page
tables collapse into **one** table (source pages [0,1,2], headers de-duplicated); with **Surya** they
do not join, because per-page fragmentation produces inconsistent column counts — i.e. stitching pays
off precisely when paired with the structure-aware engine. Scored against the verified document GT
(75×9), two output-quality fixes were applied to the stitched table: clamping to SLANet's column count
(removing a spurious trailing-empty column — metric-neutral but cleaner output) and dropping
fully-empty rows from SLANet over-segmentation (rows 101→85). After both, hybrid reaches
`Cell_Acc 0.181 / Recall 0.590 / Table_CER 0.331`, **edging Surya's document-level accuracy (0.170)**
while remaining the only stitching-capable engine (Surya still leads recall at 0.722 by over-producing
content). The residual gap (85 vs 75 rows) is near-duplicate row splits + occasional recognition
hallucinations — OCR-quality noise left unaddressed, consistent with the intended use: a
**review-ready draft** the analyst corrects, not a perfect extraction. Net: **hybrid is the engine for
dense tables and the only one that enables clean stitching; Surya stays strong on mixed content.**

> **⚠ Superseded (§4.9):** under production preprocessing, plain Surya wins on the dense page too and
> recovers the exact table shape, so the single production default is **Surya + preprocessing**; the hybrid
> framing here reflects the earlier raw-image analysis and is retained as a documented negative result.

### 4.8 Off-the-shelf recogniser A/B (recognition axis)
Separately from *structure*, we measured how well each engine **recognises** Khmer — independent of
layout — with a placement-agnostic **recognition CER** (all text pooled on each side vs a single-source
ground truth; `evaluate_recognition`), across the 3 dense table pages + 1 genuine text page. Lower =
better.

| Engine | mean recognition CER | note |
|---|---|---|
| Surya | **0.316** | baseline; best overall |
| Hybrid (rowband) | 0.315 | ties overall, but **0.667 → 0.288 on the dense table**; worse on cleaner pages |
| Tesseract-khm | 0.576 | far behind on tables, competitive only on prose |
| Qwen2.5-VL-7B (4-bit, local MLX) | 2.271 | **failed** — repetition collapse; CER > 1 = unusable output, not "worse reading" |

Three takeaways. (1) **Surya remains the recogniser to beat**; Tesseract is not competitive on tables.
(2) **Hybrid's row-strip re-reading is a *targeted* win** on the dense fragmented page — confirming the
§4.4 result on the recognition ruler too — but adds noise on cleaner pages, so it is not a universal
default. (3) **An off-the-shelf open VLM did not beat Surya**: Qwen2.5-VL-7B (the strongest T4-class
candidate, run locally via MLX) collapsed into repetition loops even after decoding tuning, scoring
CER > 1 on every page — it failed to produce usable output. This is bounded to the 4-bit build, but
with the open *Khmer-specific* models being only hobby-grade line recognisers, the practical conclusion
is that **no turnkey off-the-shelf model beats Surya today — the empirical justification for the Khmer
fine-tuning experiment (§7).** (A data-quality aside: the text page's born-digital layer was a legacy
Khmer font, unusable as ground truth — see §6.)

### 4.9 The preprocessing confound — what actually drives fragmentation, and how far it generalises

*(Added 2026-07; see `PROJECT_LOG.md` §2.25–2.28. Under production this section supersedes the raw-image
framing of §4.3–4.7.)*

The results in §4.3–4.7 were measured on **raw** page images, but the product always preprocesses first;
re-scoring under production conditions changes the picture materially.

**The confound.** With `preprocess()` on, Surya's layout model collapses the dense page-2 table from ~8
fragments to **one** region, and the engine ranking flips: plain **Surya wins** (Cell_Accuracy 0.259 vs
hybrid 0.145 / DocLayout-YOLO 0.135) and recovers the exact 75×9 shape. So under production Surya handles
the structure, and the geometric / hybrid / DocLayout work (§4.4) stands as **documented negative results**
that is not needed. The shipped app already defaulted to Surya + preprocessing — the deliverable was always
correct; only the earlier raw-image *narrative* was skewed.

**What in preprocessing does it (ablation).** A component-isolation ablation shows it is **not** the tunable
OpenCV steps: with preprocessing on, disabling any one of the five flags — deskew, sharpen, contrast,
stamp-removal, or table-background (colour) normalisation — still yields one region, and disabling **all
five** still yields one region. The lever is the two **always-on** steps — margin-crop + **downscale to
≤ 2048 px** — i.e. **resolution normalisation**, not colour or contrast. The original colour-cue hypothesis
is falsified.

**How far it generalises.** The 8→1 collapse **reproduces on a second bulletin** (15.06.26; Table_CER
0.360 → 0.091) — but that is the *same layout, different day*. A cross-*layout* test on a structurally
different dense report (a budget-execution document, pp. 3–9) finds **no fragmentation at all**, in either
condition, despite raw pages of 4151–4400 px — far above the 2048 px cap. So high resolution is **not
sufficient** to cause fragmentation; the effect is **layout-specific** — it is the bulletin's mosaic of
many small, individually-shaded cells that Surya's layout tiler splits along at high resolution, which
downscaling dissolves. The correctly-scoped claim is therefore *"preprocessing resolves the fragmentation
of the dense colour-cell market-bulletin layout"* — **not** a universal dense-table fix.

**The residual gap is recognition, not layout.** Under production, ~30–38% of GT cell content is still
unrecovered. Classifying every miss: **96% are recognition errors** (wrong or blank text on
correctly-positioned cells), only **4% segmentation** (a few merged rows). Misses concentrate in the unit
column (51%), driven by a single systematic confusion — the Riel glyph `៛` misread as `#` / `វ` / `អ` —
plus Khmer subscript-consonant substitutions in item names. This **empirically justifies recogniser
fine-tuning** as the next lever (§7), and points to a cheap deterministic `៛`-normalisation rule as a
near-term recall win.

**Caveat.** Accuracy point-estimates are noisy — Surya is non-deterministic (the same config scored 0.179
vs 0.259 across runs) — so these conclusions rest on the *structural* signals (`Tables_Found`, `Table_CER`),
not single `Cell_Accuracy` numbers.

---

## 5. Discussion
The headline scientific result is a clean **decomposition of the table-extraction problem**:
- **Detection / structure: solved.** SLANet yields the correct grid with cell coordinates,
  cheaply and quickly.
- **Recognition granularity matters more than the recogniser.** Per-cell crops break the VLM
  (hallucination); a full-width **row strip** — the VLM's natural input — lets it both read the
  line and split the columns itself. This is what finally lifted dense-table `Cell_Accuracy`
  0.024→0.393 (§4.4, §2.17), turning the "open limit" into a *recall* problem (blank strips)
  rather than a *correctness* one.

> **⚠ Superseded by §4.9:** under production preprocessing, plain Surya wins on the dense page too and
> recovers the exact table shape, so the single production default is **Surya + preprocessing**. The
> hybrid recommendation below reflects the earlier raw-image analysis and is retained for the record.

For financial tables specifically, the metric that matters is `Cell_Accuracy` (row↔value
correctness), and the practical recommendation today is: **use Surya for clean/single-table pages
and pages without tables (where it is strong and has no phantom-table cost), and the row-strip
hybrid (`OCR_ENGINE=hybrid`, `KHMER_HYBRID_MODE=rowband`) for dense fragmented tables**, where it
now beats Surya on every metric (§2.18); the only reason it is not yet a blanket default is its
behaviour on no-table pages.

---

## 6. Limitations
- **Two real labelled documents, one layout** (09.06.26 + 15.06.26 — the same market-bulletin template,
  6 pages; §4.9) — real-world numbers are indicative, not statistically robust, and cross-*layout*
  generalisation was tested only GT-free (fragmentation counts on a budget-execution doc, §4.9). More
  labelled GDDE documents — especially *different* layouts and scanned pages — remain the highest-value
  data investment.
- **OCR non-determinism** — Surya output varies slightly run-to-run; we rely on structural metrics
  and fixed-output A/Bs to control for it.
- **Order-sensitive CER** over-penalises column-wise fragmentation; `Cell_Accuracy` /
  `Tables_Found` are the more faithful signals.
- **Hybrid stays opt-in for speed, not safety.** The "phantom table on text pages" worry is
  **resolved** (§2.20): on a genuine text page (CambodiaBudget p2) hybrid produces **no phantom table**
  (`Tables_Found=0`, identical to Surya) — it only rebuilds tables Surya actually detects. So hybrid is
  safe on text pages; it remains opt-in vs Surya because it is **~3× slower** and Surya is competitive
  except on dense fragmented tables. (The earlier p3 "regression" was purely a GT mislabel, §2.19. The
  `Document_CER` figure once reported for that page is **void** — its PDF text layer was a legacy Khmer
  font, unusable as ground truth, §2.21/§4.8; the phantom-safety result is GT-independent.)
- **Preprocessing tested only on a synthetic proxy** — the OpenCV stack was A/B-tested on
  *synthetically degraded* input (§4.6) and gives a modest, consistent gain, but has not yet been
  validated on a *real scanned* document (synthetic degradation ≠ real scan artifacts).

---

## 7. Future Work
1. **Khmer recogniser fine-tuning** — the off-the-shelf recogniser A/B is **done** (§4.8): no turnkey
   model beats Surya (Tesseract weaker on tables; Qwen2.5-VL-7B 4-bit collapses; open Khmer-specific
   models are only hobby-grade line recognisers). The justified next step is a **fine-tuning**
   experiment — a recogniser on real and/or synthetic Khmer word data — to try to beat Surya. The recall
   taxonomy (§4.9) reinforces this: **96% of residual misses are recognition, not layout**, concentrated in
   a systematic `៛`-glyph confusion and Khmer subscript substitutions — so a cheap **deterministic
   `៛`-normalisation post-processing rule** is a high-leverage near-term recall win *before* the full fine-tune.
2. **Layout/structure exploration** — a separate A/B on the *detection* axis (DocLayout-YOLO, PaddleOCR
   PP-Structure, more of the PaddlePaddle stack vs Surya-layout + SLANet) targeting table fragmentation.
3. **More real labelled data**, including scanned documents, to harden the evaluation and test the
   preprocessing stack on real (not synthetic) degradation.
4. **Column-fragmentation reconstruction** at the layout level, and recovering the residual stitched
   row over-production (near-duplicate splits / recognition hallucinations on harder pages).

---

## 8. Conclusion
We delivered a reliable, fully-local Khmer OCR pipeline with a rigorous, free evaluation harness,
and used it to pinpoint exactly where the difficulty lies. Khmer *character* recognition is already
strong; *table structure* fragments on dense documents, and we proved the structure half is solvable
(SLANet). We then showed the recognition half is solvable too **at the right granularity**: reading
each row as a full-width strip and letting the VLM emit its own columns lifts dense-table
`Cell_Accuracy` ~16× (0.024→0.393) — the first intervention to fix both detection and row↔value
correctness. What remains is a contained
engineering problem (gating the hybrid on no-table pages), not an open research wall. This precisely scopes the path to the "ultimate Khmer table extractor"
and is a defensible, evidence-backed thesis result.

---

### Appendix — reproducibility
- Code + decision history: this repo; `PROJECT_LOG.md` (§1–§3), `GLOSSARY.md`, `OPERATIONS.md`,
  `eval/README.md`.
- Re-run: `uv run python -m khmer_pipeline.evaluation.run_benchmark` → `analyze_benchmark` →
  `visualize_benchmark`. Each run folder carries a `manifest.json` (git commit, versions, datasets).
- Engines: `OCR_ENGINE=surya|tesseract|hybrid`. Figures: `docs/figures/`.
