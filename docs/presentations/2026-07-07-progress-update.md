# Progress-Update Presentation Prep — ARDB Market-Price Table Extraction

## 1. Problem → Proposed Solution → Scope

**Problem.** ARDB publishes daily Phnom Penh market-price tables as PDFs/scans. The data
(commodity, unit, wholesale/retail prices, % change) is trapped in images with **mixed
Khmer script + Arabic numerals + colored table backgrounds**. Analysts re-key it by hand.
Off-the-shelf OCR fails: Khmer-only models drop the Arabic price columns; general OCR
garbles the digits.

**Proposed solution (works, not necessarily optimal).** A local, swappable pipeline:
`ingest → preprocess → structure detection (Surya) → recognition → correction → JSON/CSV + review UI`.
The new recognition engine (`OCR_ENGINE=surya_kiri`): **Surya** locates the table + cells,
**Kiri OCR** (bilingual, Apache-2.0, run in pure-CTC "fast" mode) reads each cell, with
**per-cell Otsu binarization** to beat the low-contrast colored backgrounds.

**Scope (deliberately bounded).** Two real ARDB templates, 6 pages, verified ground truth.
Local-first (runs on a laptop, MEF-safe, no data leaves the machine). Deterministic metrics.
Out of scope for now: fine-tuning, broad multi-source generalization, production throughput.

## 2. Evaluation — metrics, what's wrong, is it good enough?

**Before any metric — three fairness steps (so we measure OCR quality, not artifacts):**
- **Unicode NFC normalization** of every GT and predicted cell. *Why:* Khmer can encode the same
  visual glyph with different codepoint orders; without NFC we'd count identical-looking text as
  "wrong." Also collapses whitespace.
- **Title-row stripping** (drop row 0 when it's a merged-colspan document title — first cell filled,
  rest empty). *Why:* the report title band isn't tabular data; scoring it would distort the numbers.
- **Monotonic row alignment** (GT→pred rows paired via `difflib.SequenceMatcher` on row signatures).
  *Why:* if the model inserts/drops one row (e.g. the split header), we align around it instead of
  letting a single off-by-one cascade into "every row below is wrong." Fair to structure slips.

**Notation.** After title-strip, GT grid has `R` rows × `C` cols; normalized cell = `ĝ(r,c)` for GT and
`p̂(r,c)` for pred. `A` = set of aligned `(i,j)` GT→pred row pairs. `𝟙[·]` = 1 if true else 0.

> **Slide-ready formula images** (transparent PNG, drop straight into any slide tool) live in
> [`docs/presentations/figures/`](figures/): `01_cell_accuracy.png`, `02_cell_content_recall.png`,
> `03_table_cer.png`, `04_levenshtein.png`. The LaTeX/plain-text below is the same math as source.

**Metric 1 — Cell_Accuracy (EXACT match).**
- *Formula:*

  $$\text{Cell\_Accuracy}=\frac{\displaystyle\sum_{(i,j)\in A}\ \sum_{c=1}^{C}\ \mathbb{1}\!\left[\hat g(i,c)=\hat p(j,c)\right]}{R\times C}$$

  ```text
                   Σ over aligned rows (i,j)  Σ over cols c   1[ ĝ(i,c) == p̂(j,c) ]
  Cell_Accuracy = ───────────────────────────────────────────────────────────────────
                                        R × C   (total GT cells)
  ```

  (denominator is the full GT cell count `R·C`; rows with no aligned pred pair contribute 0 to the
  numerator, so missing/extra rows are penalized).
- *Plain:* `correct_cells / total_GT_cells`, where a cell is correct only if the normalized strings are identical.
- *Why:* strictest, most honest "**can an analyst trust this cell without editing it?**" A financial cell
  is usable or it isn't — partial credit would overstate readiness. Headline production-readiness number.

**Metric 2 — Cell_Content_Recall (multiset content presence).**
- *Formula:* let `𝒢` = multiset of non-empty GT cell values, `𝒫` = multiset of predicted cell values,
  and `count_X(v)` = occurrences of value `v` in multiset `X`:

  $$\text{Recall}=\frac{\displaystyle\sum_{v\,\in\,\text{set}(\mathcal G)}\min\!\big(\text{count}_{\mathcal G}(v),\ \text{count}_{\mathcal P}(v)\big)}{|\mathcal G|}$$

  ```text
            Σ over distinct GT values v:  min( count_GT(v), count_pred(v) )
  Recall = ────────────────────────────────────────────────────────────────
                        |𝒢|   (number of non-empty GT cells)
  ```

  (`|𝒢|` = number of non-empty GT cells; position-independent; duplicates handled by the `min`).
- *Why:* **separates recognition from placement.** "12,000" read right but one row off fails accuracy yet
  still scores recall. So **Recall − Accuracy ≈ how much error is mis-placement vs mis-reading.** On p1:
  recall 0.75, accuracy 0.20 ⇒ content captured, the header-split *structure* is at fault.

**Metric 3 — Table_CER (character error rate).**
- *Formula:* flatten table row-major to one normalized string — `g` for GT, `p` for pred — then

  $$\text{CER}=\frac{\text{Lev}(g,\,p)}{|g|}$$

  where `|g|` = length of the GT string in Unicode codepoints, and `Lev` = Levenshtein edit distance
  (min. single-character insertions + deletions + substitutions), defined by the standard recurrence:

  $$\text{Lev}(i,j)=\begin{cases}\max(i,j)&\min(i,j)=0\\[2pt]\min\begin{cases}\text{Lev}(i-1,j)+1\\ \text{Lev}(i,j-1)+1\\ \text{Lev}(i-1,j-1)+\mathbb{1}[g_i\neq p_j]\end{cases}&\text{else}\end{cases}$$

  ```text
         Lev(g, p)            Lev = Levenshtein edit distance (ins + del + subst)
  CER = ───────────           |g| = length of GT string in Unicode codepoints
           |g|

  Lev(i,j) = max(i,j)                          if min(i,j) == 0
           = min( Lev(i-1, j)   + 1,           otherwise  ← delete
                  Lev(i,   j-1) + 1,                      ← insert
                  Lev(i-1, j-1) + [ g_i != p_j ] )        ← substitute (0 if chars equal)
  ```

  (edge cases in code: CER = 0 if both empty, 1 if GT empty but pred non-empty.)
- *Why:* character granularity shows **how close a wrong cell is**. `៛`→`អ` = 1 edit in ~40 chars ⇒ tiny
  CER despite failing exact-match — proof the residual errors are small and cheaply fixable. Also the
  **standard OCR metric**, so results compare to other systems / the literature.

**Why all three together:** their *divergence* is the finding. High recall + low accuracy ⇒ a
**structure/placement** problem; low CER + low accuracy ⇒ a **systematic glyph** problem. Neither metric
alone tells you *which*; the trio does — and that's the whole story of the ARDB results (numbers read
right, one glyph + one header-split explain the gaps).

**Head-to-head, both engines, production path, all 6 ARDB pages:**

| engine | Cell_Accuracy | Recall | Table_CER |
|---|---|---|---|
| `surya` (baseline) | 0.511 | 0.759 | 0.097 |
| `surya_kiri` (new) | **0.580** | 0.755 | **0.086** |

Honest read: the hybrid wins exact-match (+0.07) and CER, ties recall. It's a **modest,
situational** win — its real advantage is **robustness on structurally hard pages**
(data page p3: 0.75 vs 0.51, where Surya mis-counts rows).

**What is actually wrong (per-cell diagnosis on data page p2 = the best case, 0.790 / 0.055):**
The prices — the *actual data* — are read **essentially perfectly** (numeric cells CER ≈ 0.000).
The accuracy gap is dominated by **ONE systematic glyph error**: the riel symbol **`៛`** in the
unit column is misread as **`អ`/`#`** (`៛/គ.ក` → `អគ.ក`) in ~24 of 27 rows. Because exact-match
fails on any wrong glyph, this single symbol caps Cell_Accuracy at 0.79 while CER stays 5.5%.
Secondary: Khmer zero `០` vs Arabic `0` in %-cells; a couple of Khmer consonant confusions
(ត↔ព); occasional number-merge on the last, tightly-spaced row.

**One deterministic fix → big jump (before/after on p2):**

| | Cell_Accuracy | Recall | Table_CER |
|---|---|---|---|
| Raw model output | 0.790 | 0.787 | 0.055 |
| + `៛`-glyph & digit normalization | **0.918** | **0.919** | **0.018** |

**Is the workflow good on ARDB?** Yes on **data pages** (79% exact / 5.5% CER out of the box,
→ 92% / 1.8% with a one-line fix; prices correct). Weaker on **header pages** (p1: ~0.20 exact,
but 0.75 recall) — see limitations. Verdict: the pipeline already extracts the numbers analysts
need; remaining errors are understood and cheap to close.

## 3. Google AI Studio — how it could help

Google AI Studio (free web playground for Gemini multimodal models) fits our workflow in three ways,
best → least:
1. **Ground-truth bootstrapping (highest value).** Our current bottleneck is *hand-built* GT.
   Feed a page image → Gemini drafts the table → human verifies. Turns hours of keying into minutes
   of checking, so we can scale evaluation to many more pages/templates.
2. **Accuracy ceiling / benchmark.** Score Gemini on the same pages to see how much headroom our
   local pipeline still has — a principled target.
3. **Smarter correction layer.** image + our rough OCR → "fix the errors" (a stronger version of the
   Qwen correction stage already in the pipeline).

**Honest caveats to state (government context):** it's a **cloud API — data leaves the machine**,
which conflicts with the local-first/MEF-sensitivity stance for *production*; free-tier rate limits;
non-deterministic output needs verification. → Recommend it for **GT bootstrapping + benchmarking**,
keep the local pipeline as the deployable product. (Public market-price data is low-sensitivity, so
prototyping is fine.)

## 4. Context7 (dev tooling — clarify with mentor)

Context7 is a documentation tool (MCP server) that pulls **up-to-date, version-specific library docs**
into the AI coding assistant during development — helps keep Surya / PyTorch / transformers API usage
correct as those libraries change. It's a **development-productivity aid**, not part of the runtime
workflow or the results. (If the mentor meant something else by "Context7," worth confirming.)

## 5. Results to show on slides (artifacts ready)

> The raw per-cell dumps below embed ARDB ground-truth content, so they live under the **gitignored**
> `eval/runs/presentation_2026-07-07/` (durable on disk, never committed/pushed — same rule as
> `eval/datasets/`). This doc holds only aggregates + methodology.

- **Slide: metrics table** (§2 head-to-head) — surya_kiri vs surya on 6 ARDB pages.
- **Slide: output vs ground truth** — row-by-row PRED ‖ GT ‖ CER (data page p2). Numbers match; the
  visible diff is `៛`→`អ` in the unit column. (Full dump: `eval/runs/presentation_2026-07-07/CER_p2.md`.)
- **Slide: before/after the one-glyph fix** (§2) — 0.79→0.92 exact, CER 0.055→0.018.
  (Rescore detail: `eval/runs/presentation_2026-07-07/riel_fix_projection.txt`.)
- **Slide: the honest limitation** — p1 header page, PRED 25×9 vs GT 24×9 (multi-line header split).

## 6. Challenges, Limitations & Scaling (emphasize scaling)

**Challenges / limitations:**
- **Compute.** Trained/run on a laptop (M4, 24 GB unified memory, **no dedicated GPU/VRAM**). CTC loss
  isn't even implemented on Apple MPS (CPU fallback). Recognition is CPU-bound: **~30–45 s/page** (~240
  cells). Can't fine-tune large models locally.
- **Data.** Only 2 templates / 6 pages of verified GT — small; hand-built GT is the bottleneck.
- **Structure.** Multi-line column headers get split into extra rows (the p1 case); tightly-spaced rows
  occasionally merge digits.
- **Recognition.** One systematic glyph (`៛`) + Khmer/Arabic digit mixing — known, not yet fixed in the shipped engine.

**Scaling plan (the important part):**
- **Data scale-out:** use Google AI Studio to bootstrap GT → cover many more ARDB pages, dates, and
  other MEF/ARDB report formats; build a standing evaluation set.
- **Compute scale-out:** move fine-tuning to Colab / cloud GPU (Kiri ships `training.py`; we also have a
  from-scratch CRNN trainer). Fine-tune on ARDB-specific glyphs (the `៛` symbol, digit scripts) to close
  the residual gap the deterministic fix doesn't.
- **Throughput scale-out:** batch cell recognition + GPU inference (biggest single speed win); remove the
  engine's double layout pass (already flagged as a `TODO`); parallelize pages.
- **Immediate cheap win:** ship the deterministic `៛`/digit normalization in the correction stage → +0.13
  exact-match today, zero training.
- **Coverage:** generalize the structure step beyond the two templates; add per-template header handling.

**One-line takeaway for the mentor:** *"The pipeline already reads ARDB prices correctly (79%→92%
exact-match with one deterministic fix); the remaining work is scaling — more ground truth via Google AI
Studio, and cloud-GPU fine-tuning for the last glyph-level errors."*
