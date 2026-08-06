# Fine-tune evaluation figures

Four charts from the Gemma 4 E2B / Qwen3.5-0.8B ARDB fine-tuning work: the epoch sweep for
each model (2, 3, and 5 epochs on the `Soxavin/ardb-sft-v5` dataset) and how the best config
from each compares against the existing production OCR pipeline on real documents. Full
numbers and narrative live in `eval/gemma_finetune_runs.md`, `eval/qwen_finetune_runs.md`,
and their `.csv` companions — these charts are the visual summary, not a replacement for
those logs.

**The story across all four, in one paragraph**: the existing production OCR pipeline needed
no fine-tuning at all to outperform both models on real documents (chart 1) — that's the
headline. Both models struggle to produce even syntactically valid output on a meaningful
share of pages (chart 2), and that struggle doesn't resolve by training longer: for both
models independently, 3 epochs is the best point in the sweep, with both fewer and more
epochs making things worse (charts 2 and 3). Training itself was never the problem — every
run converged cleanly (chart 4) — so the open trade-off is generalization/output-reliability
at this data scale, not an optimizer or training-setup issue.

(The four images one level up in `docs/figures/` — `accuracy_by_font.png`, `cer_by_dataset.png`,
`engine_comparison.png`, `table_fragmentation.png` — are from earlier, separate OCR-engine
benchmarking work and aren't covered by this README; these fine-tune-eval figures live in their
own `finetune_eval/` subfolder to keep the two sets of charts from mixing together.)

---

## 1. `real_doc_comparison.png` — does either fine-tune actually work?

![Real-document comparison](real_doc_comparison.png)

**What it shows**: the existing OCR pipeline (Surya, no fine-tuning) vs. each model's
best-performing fine-tuned adapter, scored on the *same* 15 real, hand-verified ARDB pages —
documents neither model trained on. Left panel is metrics where a higher bar is better
(accuracy and match-rate); right panel is metrics where a lower bar is better (character
error rate). Qwen shows no bars at all — every one of its 15 pages failed to produce usable
output, so there's nothing to score, marked "no output" rather than a misleading zero.

**How to read it**: compare bar heights within each metric group (see the legend for which
color is which approach). The OCR pipeline is taller than Gemma on every "higher is better"
metric and shorter on every "lower is better" metric — meaning it wins on every axis measured.

**Takeaway**: this is the headline result. On real documents, the production OCR pipeline
currently outperforms both fine-tuned models, and Qwen doesn't produce usable structured
output at all. Gemma comes closer but isn't yet competitive with the existing pipeline.

---

## 2. `parse_failure_rate_by_run.png` — can the model produce valid output at all?

![Parse-failure rate by run](parse_failure_rate_by_run.png)

**What it shows**: for every training run in the epoch sweep, what fraction of validation
pages the model failed to even produce syntactically valid JSON for (1.0 = every page
failed, 0.0 = every page parsed). This is the most basic pass/fail signal — before asking
"is the text accurate," this asks "did the model produce a readable answer at all."

**How to read it**: the x-axis is **epoch count itself** (2, 3, 5) — not run order — so both
models' points at the same epoch count line up vertically and can be compared directly. Color
is model (blue = Gemma, green = Qwen, matching chart 1); line style is dataset version (solid
= `v5`, dashed = `v2` — Qwen only has `v5` data, so its line is always solid). Each point is
labeled with its step count, since two runs can share an epoch count but a different effective
batch size (see chart 3's note for a concrete example) — where two lines land on the exact
same point, their labels are stacked at different heights rather than overlapping.

**Takeaway**: for both models, 3 epochs is the best-performing point in the sweep — both
fewer (2) and more (5) epochs produced *more* failures, not fewer, and with epoch count on
the x-axis that V-shape is visible in both lines at a glance. That's the same pattern in both
models independently, which is why we didn't chase higher epoch counts further.

---

## 3. `cer_by_run.png` — when it *does* parse, how accurate is the text?

![CER by run](cer_by_run.png)

**What it shows**: Character Error Rate (CER) — roughly, "what fraction of characters would
need to change to turn the model's output into the correct answer," so lower is better —
broken out by content type (table text, section headers, page letterhead, etc.), for every
run and every model. Gemma and Qwen get their own panel with their own y-axis scale, because
one Qwen data point (a Section-Header CER over 5) would otherwise squash every other,
more relevant point into an unreadable cluster near zero.

**How to read it**: x-axis is epoch count, same convention as chart 2. Color is content
label (see the shared legend at top); line style is dataset version (solid `v5`, dashed
`v2` — same meaning as chart 2). Each small annotation next to a point (e.g. `n=6 (3ep, 78
steps)`) gives both how many pages that point's average is based on and which exact run it
came from. **The `n=` part matters** — a point based on `n=1` is a single data point, not a
trend, and shouldn't be read with the same confidence as one based on `n=12`. The italic
note at the bottom flags a specific case worth knowing about: Qwen's two "3 epoch" points
aren't duplicates — they used a different gradient-accumulation setting (different effective
batch size), which is why they have different step counts despite the same epoch count.

**Takeaway**: CER trends broadly track the parse-failure chart, but sample sizes shrink a
lot at the worse-performing runs (fewer pages parse → fewer pages to average), so the
higher-epoch points' CER numbers are on thinner evidence than the 3-epoch points'.

---

## 4. `loss_by_run.png` — did training itself work?

![Training loss by run](loss_by_run.png)

**What it shows**: the model's training loss at every step, for every run, on a log scale,
split into one panel per model (Gemma left, Qwen right) so each panel only has to hold that
model's own runs rather than all six overlaid on one axis.

**How to read it**: this is a training-process sanity check, not a quality result — it
answers "did the optimizer converge normally," not "is the model good." A clean, steadily-
dropping line means training ran without errors and the model fit its training data. It does
**not** mean the model generalizes well — one of the cleanest-looking curves in this
project's history (an earlier Gemma run, not shown on this specific chart) still produced 9
failures out of 9 validation pages. Use this chart alongside charts 2 and 3, never instead of
them. Line style still follows the solid-`v5`/dashed-`v2` convention from charts 2 and 3
(Gemma's panel has both; Qwen only has `v5` loss data logged, so its lines are all solid).

**Takeaway**: every run's loss dropped cleanly and as expected — training itself was never
the problem in any of these runs. Whatever is limiting output quality (see charts 1-3) is a
generalization/reliability issue, not a training-failure issue.

---

## A few things worth knowing before citing these numbers

- **Every run is a single seed.** Nothing here is averaged over repeated runs with different
  random seeds (Colab GPU time was a real, repeatedly-managed cost constraint throughout this
  project), so there are no error bars or confidence intervals on any chart. Read point-to-point
  differences as directional trends confirmed across two independent model families (see chart
  3's `n=` counts for how much data backs any one point), not as statistically tested results.
- **Colors are chosen to be colorblind-safe** (the Okabe-Ito palette) and every line/series that
  needs to survive grayscale printing also varies by marker shape or linestyle, not color alone.

## File formats

Each chart is written as both a `.png` (raster, for pasting directly into a document) and a
`.pdf` (vector, for print or any context where the image gets resized — it won't pixelate).
Use whichever your report tool handles better; the content is identical either way.

## Regenerating these figures

```bash
uv run python3 scripts/plot_run_metrics.py eval/gemma_finetune_runs.csv eval/qwen_finetune_runs.csv --out-dir docs/figures/finetune_eval
uv run python3 scripts/plot_real_doc_comparison.py eval/real_doc_eval.csv --out docs/figures/finetune_eval/real_doc_comparison.png
```

Both scripts read directly from the `eval/*.csv` run logs, so figures always reflect
whatever is currently logged there — re-run them after logging any new fine-tune result.
