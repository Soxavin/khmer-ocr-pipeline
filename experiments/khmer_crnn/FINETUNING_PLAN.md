# Next Phase — Khmer Recognizer Fine-Tuning: Plan

*Forward-looking plan (not yet implemented). Self-contained so it can be handed
to other AI agents. Prior work it builds on: [FINDINGS.md](FINDINGS.md) (the
from-scratch CRNN exercise), and the main pipeline docs `docs/PROJECT_LOG.md` /
`docs/REPORT.md`.*

---

## 1. Context and motivation

The production pipeline (Surya OCR + a deterministic Khmer normalizer) handles
document **structure** well once preprocessing is applied; the residual
bottleneck is **recognition** (`PROJECT_LOG §2.25`: on the real market-price
docs, Cell Recall ~0.62, Table CER ~0.25; recall taxonomy §2.27: ~96% of misses
are recognition, not segmentation). Improving recognition is therefore the
highest-leverage next step, and the mentor endorsed fine-tuning.

The CRNN exercise ([FINDINGS.md](FINDINGS.md)) proved we can train a Khmer
recognizer end-to-end on the Mac (blank-collapse → 3.4% CER on Hanuman in 40
epochs) — but that is **from scratch on one font**. This phase moves to **real
fine-tuning**: warm-start a pretrained Khmer-capable recognizer and adapt it to
the GDDE financial-document domain.

## 2. Goal

A recognizer that **beats the Surya baseline on the real documents**, measured
by CER / Cell metrics on held-out real pages via the existing `evaluation/`
harness. **A negative result (cannot beat Surya) is still a valid thesis
finding** — it bounds what is achievable and justifies the shipped default.

## 3. Candidate models (ranked for our situation)

| # | Model | Why | Frictions / what to verify |
|---|---|---|---|
| 1 ⭐ | **mrrtmob/kiri-ocr** | **Bilingual EN+Khmer**, Apache-2.0 (MEF-safe), transformer CTC+attention, **verified charset covers Arabic digits + `%`/`.`/`,`/`-`/`/`/`()` + Latin + Khmer** (967 vocab). Reads our mixed cells structurally (see §3b). Fine-tunable (`training.py`), CPU-capable. | Off-the-shelf shows **digit-duplication** artifacts on prices (`7,800`→`7,8000`) → not yet trustworthy for exact figures; needs the real-cell eval (§7) + likely light fine-tune. **Mac packaging:** current build pins `onnxruntime-gpu` (no macOS ARM wheels) — install from git + use the CRAFT/legacy torch detector, or install plain `onnxruntime` for the DB detector. |
| 2 | **Our CRNN** ([train.py](train.py)) | License-free, fully controllable, runs on the Mac; **data-driven charset** — covers Khmer + Arabic + punctuation if trained on such data. Fallback if Kiri fine-tuning stalls. | Weaker raw ceiling; from scratch; needs the multi-font corpus (§5). |
| 3 | **Fine-tune Surya's recognizer** | Best production alignment — it is what we ship, and it **does** read Arabic numerals. | Newer Surya = single ~**650M VLM** (Qwen-style); code Apache-2.0 but **weights under modified AI-Pubs OpenRAIL-M** (**government/MEF use needs a license check**). Fine-tuning is not turnkey (contact hi@datalab.to). Heaviest option. |
| 4 | **songhieng/khmer-trocr-ocr-v1.0** | Existing **Khmer TrOCR** on HF; standard HuggingFace `Seq2SeqTrainer` fine-tuning; open. | Verify base size, **charset/tokenizer coverage of Arabic numerals + punctuation**, quality vs Kiri. |
| ref | **seanghay/KhmerOCR** | MIT, ONNX/CPU, clean API, 3M lines/800 fonts — excellent **Khmer-text** recognizer + `khmerocr_tools` synthetic-data reference. | **Not usable as our recognizer** — see §3a (Khmer-script-only charset). |
| — | ~~Qwen2.5-VL / VLMs~~ | — | **Excluded** — documented negative result (cost/complexity). |

**Recommended (revised after the Kiri OCR discovery):** lead = **mrrtmob/kiri-ocr**
— the only open, off-the-shelf model that both fits our charset *and* is
fine-tunable; evaluate it off-the-shelf on real cells (§7), then fine-tune on the
GDDE domain if the digit-quality gap warrants. **Our CRNN** is the controllable
fallback; **Surya fine-tune** the production-aligned option (pending MEF license).

### 3a. Why NOT seanghay/KhmerOCR (empirically tested 2026-07-06)

KhmerOCR looked like the obvious primary warm-start (MIT, ONNX/CPU, 3M lines/800
fonts, clean `khmerocr.recognize(pil) -> {text, confidence, font}`). We tested it
directly. **Its recognizer output vocabulary is 98 Khmer-script characters only**
(`TOKENS` in `khmerocr/__init__.py`): Khmer consonants/vowels/signs, `៛`, and
**Khmer** digits ០–៩. It has **no slots** for Arabic digits 0–9, `%`, `.`, `,`,
`/`, `-`, `(`, `)`, or spaces.

Run end-to-end on a **real** market-price page (`09.06.26_p2`), it captured the
Khmer row-number + product-name columns but **dropped all six Arabic-numeral
price/percentage columns** — ~67% of the table, and specifically the data the
document exists to report. On isolated cells it transliterates Arabic → Khmer
digits (e.g. `12,500` → `១២៥០០`, losing the comma; `7,800` → wrong value) and
drops all punctuation. This is **architectural, not a fine-tuning gap**: the
output layer cannot emit those characters.

**Root cause:** typical Khmer documents write numbers in Khmer numerals; our GDDE
financial tables use **Arabic** numerals with commas/decimals/percent — a
domain-specific mismatch that off-the-shelf Khmer OCR is not built for.

**Consequence for the ranking:** KhmerOCR drops from "primary" to a
**reference / Khmer-text-only baseline**. The lesson it hands us — *the required
charset is Khmer + Arabic digits + financial punctuation*.

### 3b. Why mrrtmob/kiri-ocr IS a fit (empirically tested 2026-07-06)

Found via `seanghay/awesome-khmer-language`. **Kiri OCR** is a **bilingual
English+Khmer** OCR library (Apache-2.0), a transformer with a hybrid CTC +
attention decoder, CRAFT/DB/legacy detectors, simple Python API
(`from kiri_ocr import OCR; ocr.extract_text(path)` and
`ocr.recognize_single_line_image(path)`), model auto-downloaded from HF
(`mrrtmob/kiri-ocr`).

**Charset — the thing KhmerOCR lacked.** Its `vocab.json` (967 entries) fully
covers **Arabic digits 0–9 (10/10)**, **`% . , - / ( )` (7/7)**, space, Latin,
and Khmer incl. `៛` and Khmer digits. So it is *architecturally capable* of our
mixed-script financial content.

**Recognizer test (isolated, on the same cells KhmerOCR failed):** Khmer product
names + Khmer digits **perfect**; a mixed `ពងមាន់ 340 500` cell and `-2.86%`
**perfect**; Arabic prices come out with the **right digits and punctuation but a
digit-duplication artifact** (`7,800`→`7,8000`, `12,500`→`122,5500`, `2.94%`→
`2.994%`); `៛`→`រ` on the riel sign. Confidences were uniformly low (~0.44),
possibly a synthetic-render/input-preprocessing mismatch. So: **capable, but the
off-the-shelf digit quality is not yet trustworthy for exact prices** — a
*quality* gap (fixable by fine-tuning / correct preprocessing), not a capability
gap.

**Practical notes:** (1) PyPI build is stale (dim-256 vs the HF checkpoint's
dim-384) — install from **git main** (v0.2.15). (2) git main pins
`onnxruntime-gpu` (no macOS ARM wheels); the **DB** detector needs it, but the
**CRAFT/legacy** detectors and the recognizer are pure torch (`device="cpu"`), so
Mac use is fine via those (or install plain `onnxruntime` for DB). (3) In our
**hybrid layer** Surya provides the cell crops, so Kiri's detector is irrelevant —
we only need `recognize_single_line_image` / `recognize_region`.

**Real-cell eval (end-to-end, DB detector, real page `09.06.26_p2`):** the DB
detector segmented ~237 cells cleanly. **Khmer product names + the `៛/គ្រាប់`
unit read perfectly** on real data (KhmerOCR mangled both), and Arabic digits are
*read* (not transliterated). **But the digit-duplication persists on real cells**
(`360`→`3660`, `7,800`→`7,8000` — every number gets a doubled digit) and the
low-value **percentage cells hallucinate** into Latin words (`0.00%`→`November`).
Confidence uniformly ~0.44. So **off-the-shelf it is NOT usable for these numeric
tables** as-called.

**RESOLVED — it was the decoder, not the model.** The duplication only happens
with the **attention decoder** (`decode_method="accurate"`, the default, and
`"beam"`). Switching to **`decode_method="fast"` (pure CTC head) eliminates it
entirely.** On isolated synthetic cells, `fast` reads `360`, `500`, `7,800`,
`12,500`, `2.94%`, `-2.86%`, and Khmer **all perfectly**. Re-run on the **real**
page with `fast`: Arabic prices, Khmer names, row numbers, and the `៛` unit read
**correctly at ~99% confidence** (up from ~0.44). The only remaining errors are
the small **percentage-change cells**, and those are a **DB-detector mis-crop**
(the recognizer reads `-2.86%` perfectly when cleanly cropped) — which **Surya's
superior table detection would fix** in the hybrid.

**Consequence for the ranking:** Kiri OCR (with `decode_method="fast"`) is a
**viable off-the-shelf recognizer** for our documents — the strongest result from
any off-the-shelf model. It becomes the immediate candidate for the
**Surya-detect + Kiri-recognize** hybrid (§7a), *before* any fine-tuning; fine-
tuning stays as an optional later quality lever (e.g. the `៛` unit occasionally
reads `!`/`អ`).

### 7a. Hybrid experiment: Surya-detect + Kiri-recognize (the near-term win)

Architecture (already supported by the swappable-engine registry + hybrid engine):
Surya detects table structure + cell boxes (its strength) → each cell crop →
`kiri_ocr` `recognize_single_line_image` / `recognize_region` with
**`decode_method="fast"`** → assemble into the table. Evaluate vs Surya-alone on
the real docs via the `evaluation/` harness (Cell metrics + CER). Optional per-cell
routing (Khmer cells → whichever wins; numeric → Surya) if Kiri doesn't beat Surya
everywhere. Packaging caveats for Kiri on Mac are in §3b.

**RESULT (prototyped 2026-07-06, real page `09.06.26_p2`, scored via `evaluate_table`).**
Winning recipe: **RAW page → Surya layout → `merge_table_regions` (8 fragments → 1
table) → Surya `TableRecPredictor` (243 cells, exact 27×9 grid) → crop each cell
from the RAW image → Kiri `recognize_single_line_image` with `decode_method="fast"`**.

| Config | Cell_Accuracy | Recall | CER |
|---|---|---|---|
| **Surya-cells + Kiri (raw + merge, +trailing-`.` fix)** | **0.444** | 0.430 | 0.328 |
| Surya-cells + Kiri (raw + merge, no fix) | 0.342 | 0.323 | 0.339 |
| Surya-alone baseline | 0.259 | 0.623 | 0.249 |
| Surya-cells + Kiri (preprocessed) | 0.255 | 0.234 | 0.364 |
| SLANet cells + Kiri | 0.041 | — | — |
| Kiri-DB / Surya-line-detect + Kiri (geom. grid) | 0.008–0.041 | — | — |

Key learnings:
- **The hybrid BEATS Surya-alone on exact Cell_Accuracy (0.342 vs 0.259, +32% rel).**
- **Structure must come from Surya `TableRecPredictor` + `merge_table_regions`** (gives
  cell polygons + row/col). SLANet and geometric grid-reconstruction both fail badly.
- **Preprocessing tension:** Surya's *structure* wants preprocessing (collapses the
  8-fragment split), but preprocessing *wrecks Kiri's recognition* (prices garble).
  Resolution: use the **RAW** image — `merge_table_regions` fixes the fragmentation
  structurally *without* preprocessing, so Kiri reads raw crops at its ~99% quality.
- **Artifact fixes:** stripping Kiri's **trailing `.`** on numbers + **no cell
  padding** (padding pulls in neighbours) is the best recipe.
- **%-cell root cause + fix (diagnosed by dumping the crops):** the two %-change
  columns are **low-contrast yellow-on-orange** text (price columns are yellow-on-
  **green** = high contrast). Boxes are correct (188×64px). **Plain grayscale and the
  pipeline's `_normalise_table_backgrounds` (desaturation) do NOT help** — they
  preserve luminance, so the low contrast survives. **Per-cell Otsu THRESHOLDING
  fixes it** (reads `-3.85%`/`0.00%`/`-2.86%`/`7.14%` perfectly): Otsu snaps the
  yellow(226)/orange(166) split to crisp black-on-white. (CLAHE failed — noised the
  good cells.) Add auto-polarity (`if binarized.mean()<127: invert`) so normal
  dark-on-light cells aren't flipped.
- **Broadened validation (6 market-price pages, 09.06.26 + 15.06.26 p1–p3), WITH
  per-cell Otsu — beats Surya on ALL THREE metrics:**

  | | Surya + Kiri + Otsu | Surya-alone |
  |---|---|---|
  | Cell_Accuracy (mean) | **0.580** | 0.259 |
  | Recall (mean) | **~0.75** | 0.623 |
  | CER (mean, lower=better) | **~0.09** | 0.249 |

  Data pages (p2/p3) hit **~0.79 Cell_Accuracy / ~0.05 CER**. Otsu helps *every*
  colored-background column, not just %. Only **p1 header pages** lag on exact
  accuracy (0.19–0.21) — but their Recall is 0.75, so it's purely a header/row-
  alignment issue, not recognition.
- **Net:** the **Surya-detect + Kiri-recognize + per-cell Otsu** hybrid is a
  **validated, no-fine-tune, local, Apache-2.0** recogniser that **beats Surya-alone
  on all metrics** across two real documents. Winning recipe: RAW page → Surya layout
  → `merge_table_regions` → `TableRecPredictor` cells → per-cell Otsu (+auto-polarity)
  → Kiri `recognize_single_line_image(decode_method="fast")` → strip trailing `.`, no
  padding. Open follow-ups: **p1 header/row-alignment**, full per-page Surya baseline,
  and productionising as `OCR_ENGINE=surya_kiri`.

## 4. Datasets (tiered by value)

1. **Small REAL labeled set from the actual GDDE market-price PDFs** — *highest
   value*. Even a few hundred hand-labeled line-crops is the true fine-tuning
   target and eval set. Extend the ground truth already built for the real docs.
   (Sensitive government data — stays gitignored, never committed/uploaded.)
2. **Synthetic multi-font corpus (we generate it)** — from the vendored fonts
   (§5). 5 fonts, biased toward **financial glyphs** (Khmer digits, `៛`, `%`,
   commas, `គ.ក`, etc.). The big lever we fully control; for pretraining/aug.
3. **`seanghay/khmer-hanuman-100k`** (already downloaded) — single font; useful
   for pretraining and as a sanity dataset.
4. **`seanghay/awesome-khmer-language` hub + khmer-ocr-benchmark-dataset** — a
   curated catalog of Khmer datasets, fonts (the 800+), and models. Survey for
   additional real/synthetic line data.
5. **Scan-degradation augmentation** — reuse `datagen/generate_degraded.py` to
   push synthetic lines toward real-scan quality (blur, noise, JPEG, skew).

## 5. The synthetic multi-font bridge (reuses existing code)

**We already have the pieces** — they just produce *pages*, not *lines*:

- `src/khmer_pipeline/utils/fonts.py` — **5 vendored OFL Khmer fonts** (Noto
  Sans Khmer, Battambang, Hanuman, Moul, Fasthand), base64-embedded, fully
  offline/deterministic; `font_face_style_tag(family)` gives a ready
  `@font-face` `<style>`.
- `src/khmer_pipeline/datagen/generate_synthetic_documents.py` /
  `generate_synthetic_tables.py` — render Khmer via **HTML/CSS + Playwright**
  into full-page PNG + structured GT JSON.

**Build a small `line renderer`** (new, e.g.
`datagen/generate_synthetic_lines.py`): given a text string + font, render a
single **tight-cropped line image** via the same Playwright + `font_face_style_tag`
pattern → `(PNG, string)`. Loop **text corpus × 5 fonts** → a line dataset in
the *exact* `(image, text)` format `train.py` already auto-detects.

- **Text source (elegant):** reuse the **Hanuman-100k labels themselves**,
  re-rendered across all 5 fonts → up to ~500k line-images, same content, 5×
  the glyph variety. Optionally mix in financial-vocabulary strings for domain
  match.
- **Scale target:** ~10⁴–10⁵ line-images for pretraining.

## 6. Recipe (phased)

1. **Verify + shortlist** (no compute): confirm KhmerOCR license/arch/weights;
   confirm Surya weight license for government use; pick the primary model.
2. **Synthetic multi-font corpus:** build the line renderer, generate the corpus
   (§5), with degradation augmentation.
3. **Baseline the pretrained models as-is** on the real doc line-crops (KhmerOCR,
   Surya) via the `evaluation/` harness — establishes the bar before any training.
4. **Fine-tune** the chosen model: warm-start → train on synthetic multi-font
   (and/or the pretrained checkpoint) → **fine-tune on the small real GDDE set**.
5. **Evaluate on held-out real docs** (CER + Cell metrics) vs the Surya baseline.
6. **Report:** fold results into `docs/REPORT.md` Track A; a negative result is
   still reported.

## 7. Evaluation

- Reuse the existing **`evaluation/` harness** (deterministic, free): recognition
  **CER** + **Cell_Accuracy / Cell_Content_Recall / Table_CER** on the real docs.
- Command shape (baseline vs fine-tuned): `scripts/eval_document.py "<stem>" --preprocess`.
- Compare against the Surya production default (Surya + preprocessing).

## 8. Compute

- **CRNN**: local (Mac, ~20 s/epoch on the curriculum config).
- **Surya / TrOCR / KhmerOCR fine-tune**: likely a GPU — **Colab** (T4/A100) per
  the project's local-first + Colab-for-heavy stance. Package the training as a
  Colab-ready notebook/script; keep the sensitive real data local (upload only
  the small labeled set to a private/ephemeral runtime if unavoidable, or train
  the domain-adaptation step locally).

## 9. Open decisions (to confirm before implementing)

1. **Primary model:** KhmerOCR (open, in-language) vs Surya (production-aligned,
   license-gated) — gated on the license/arch verification in §6.1.
2. **How much real GDDE data to hand-label** for the fine-tune/eval (a few
   hundred lines is a reasonable first target).
3. **Compute venue:** Colab vs local, per model size.
4. **Corpus scale + degradation strength** for the synthetic multi-font set.

## 10. Risks / caveats

- **Clean-font vs real-doc gap:** all pretrained Khmer models (KhmerOCR's 800
  fonts included) and our synthetic corpus use **clean rendered fonts**; the real
  GDDE docs carry the broken-CMap / legacy-font issue. But **OCR reads the image,
  not the broken text layer** — so the decisive test is always **eval on real doc
  images**, not assumptions about fonts.
- **License:** Surya weights may not be freely usable for a government deployment
  — resolve before investing in that path.
- **Data sensitivity:** real GDDE documents are sensitive government data — keep
  gitignored, do not upload to third-party services.
- **Fine-tuning may not beat Surya** — acceptable; it is a valid, reportable
  finding that bounds the achievable and validates the shipped default.

## 11. References

- seanghay on Hugging Face — https://huggingface.co/seanghay
- awesome-khmer-language — https://github.com/seanghay/awesome-khmer-language
- songhieng/khmer-trocr-ocr-v1.0 — https://huggingface.co/songhieng/khmer-trocr-ocr-v1.0
- Surya (datalab-to) — https://github.com/datalab-to/surya
