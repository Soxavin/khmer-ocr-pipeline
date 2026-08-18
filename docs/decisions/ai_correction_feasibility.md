# AI text correction — feasibility & decision memo

**Date:** 2026-07-28 · **Status:** Qwen corrector retired (§2.102); a Khmer-specific corrector is
documented future work.

## What existed
Stage 4 ("postprocess") bundled two things:

1. **An always-on deterministic layer** (kept) — Khmer Unicode normalization (`normalize_khmer`:
   NFC, ZWSP/BOM strip, canonical reorder, dup-diacritic collapse) + GDDE domain cell rules (៛
   riel-prefix repair, percent Khmer→Arabic digit fold) + foreign-script scrub + gridline-noise
   strip + **malformed-number flagging** (caps confidence, never rewrites digits). Runs on every
   extraction, including table cells.
2. **An "AI text correction" toggle** (removed) — a general `Qwen2.5-7B-Instruct-4bit` LLM (MLX)
   that rewrote **page-text blocks only** (never table cells) whose foreign-script ratio ≥ 0.15,
   via a batched JSON prompt.

## Why the Qwen toggle was retired
- **It never touched tables** — the numeric deliverable — only page prose.
- **It fired rarely and redundantly**: its only trigger was foreign-script prose, which the
  deterministic scrub already removes regardless.
- **Trust liability**: a general LLM can silently rewrite a correct string into a wrong one with no
  confidence signal — the opposite of the pipeline's stance everywhere else (numbers are *flagged*,
  never rewritten). For a numbers-must-be-trustworthy product that is a risk, not a feature.
- **Cost**: ~4GB model load + per-page inference on a 24GB Mac.

## What the literature says (post-OCR correction is a studied task)
- **The primary lever is a script-specialized recognizer, not a bolt-on corrector** (Rijhwani et al.
  2020, arXiv 2011.05402; Lexically-Aware Semi-Supervised 2021, arXiv 2111.02622). The project's own
  **Kiri fine-tune** is exactly this — the right primary lever; correction is secondary.
- **The standard corrector is a SMALL seq2seq trained on synthetic error pairs**, not a general chat
  LLM: byte-level **ByT5** (tokenizer-free, character-grained, good for diacritics) or char seq2seq
  ensembles (arXiv 2109.06264); ICDAR 2017/2019 Post-OCR competitions are the benchmarks.
  **RoundTripOCR** (ICON 2024, arXiv 2412.15248, github.com/harshvivek14/RoundTripOCR): inject
  realistic OCR errors into clean text → train a translation-style corrector.
- **LLMs can help but the evidence is domain-specific and risky** (Llama-2 vs BART, historical
  English newspapers) — every study flags over-correction/hallucination.

## Khmer-specific building blocks that exist
- **PrahokBART** (`nict-astrec-att/prahokbart_base`, also `prajdabre/prahokbart`; COLING 2025,
  arXiv 2512.13552) — a Khmer+English BART (~4.5B Khmer + 3.5B English tokens, Khmer word-seg +
  normalization baked in, "more efficient than mBART50"). Fine-tunable as an OCR-error→clean
  corrector. (Exact params/license: confirm from the HF card before any build.)
- **seanghay/tha** (ថា) — deterministic Khmer normalization/verbalization toolkit; complements the
  existing `normalize_khmer`, zero ML risk.
- **seanghay/awesome-khmer-language** — resource hub (spellcheckers, segmenters, G2P, corpora).

## If a corrector is revisited (recommended shape)
1. **Numbers stay flag-don't-rewrite** — no learned model touches numeric cells. Non-negotiable.
2. **Scope to page prose**, and make it purpose-built: fine-tune **ByT5-small** (safest first bet)
   or **PrahokBART** on **synthetic OCR-error pairs** (RoundTripOCR method) seeded from the
   project's own measured confusions (`postprocess.py` §2.33: ៛→អ/#/វ, percent digit folds).
3. **Real training data already exists**: the corrections corpus (`corrections.py`) captures analyst
   raw→verified edits — gold real-world error→correct pairs to train and eval on.
4. **Eval**: CER reduction on held-out prose; guardrails: ~0 over-correction on already-correct
   strings, zero leakage into numeric cells. **Kill criteria**: any over-correction/number leakage.
5. **Compute**: Colab-T4 fine-tune (hours); inference runs locally, far lighter than Qwen-7B.
   Integration is a drop-in via `ACTIVE_CORRECTION_ENGINE` (`engines/engine_registry.py`).

## Honest caveat
The payoff of any prose corrector lands on **page text**, not the **numeric tables** that are the
actual deliverable — so even a good corrector may be low-impact on the product. That, plus the trust
risk of the general LLM, is why Qwen was retired rather than replaced immediately. A Khmer-specific
prose corrector is a legitimate, defensible experiment for a future exploratory track, not an
urgent gap.

## Sources
arXiv 2011.05402 · 2111.02622 · 2109.06264 · RoundTripOCR (ACL ICON 2024 / arXiv 2412.15248) ·
ICDAR 2019 Post-OCR competition · PrahokBART (arXiv 2512.13552 / COLING 2025) · HF
nict-astrec-att/prahokbart_base · google/byt5-small · seanghay/tha · seanghay/awesome-khmer-language
