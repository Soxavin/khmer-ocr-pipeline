# Qwen3.5-0.8B / ARDB fine-tune — run log

Durable record of each Colab training run for `scripts/colab_qwen35_finetune.ipynb`, so
results can be cited/compared later without scrolling back through chat history. Append a
new entry per full run — never edit past entries' results, only add corrections as notes if
a bug is later found to have affected them. Mirrors `eval/gemma_finetune_runs.md`'s
convention; the two stay separate files (not merged) since they track different models, but
`scripts/plot_run_metrics.py` can combine both CSVs into one comparison chart.

## Why this model is being fine-tuned despite a known base-model limitation

`docs/PROJECT_LOG.md` §2.104 documents a controlled isolation test (base checkpoint, zero
training steps): Qwen3.5-0.8B generates fluent **Thai** script instead of Khmer, on both a
plain-text translation prompt and this notebook's real image+instruction task. This is a
pretraining-time gap in the base model, not something a LoRA fine-tune on our dataset is
expected to fix on its own. The project's production fine-tune target reverted to Gemma 4 E2B
as a result (see `eval/gemma_finetune_runs.md`).

Fine-tuning Qwen3.5-0.8B is still being pursued here for two reasons: it's specifically
requested as a target to fine-tune, independent of whether it ends up matching Gemma's
quality; and it's a legitimate, presentable result either way — a documented, precisely
measured negative/partial result is still evidence of real experimental work, not just a
positive number.

## Training setup decisions — why, not just what

Non-default choices in `scripts/colab_qwen35_finetune.ipynb`'s config not already covered by
`eval/gemma_finetune_runs.md`'s shared reasoning (install cell structure, `max_steps`
sentinel, per-row progress printing, adapter-push-before-eval ordering — all identical
patterns, see that file). Qwen-specific choices:

- **16-bit LoRA, not QLoRA (`load_in_4bit=False`)** — Unsloth's own Qwen3.5 fine-tuning guide
  explicitly recommends against 4-bit training for this model family, unlike Gemma 4 E2B
  where QLoRA was the right call for T4 memory constraints.
- **`r=16, lora_alpha=16`** (vs. Gemma's `r=32`) — Unsloth's own default for this model; 0.8B
  has far less capacity than Gemma's 5.2B, so a larger rank is more likely to overfit the
  already-small dataset than help it.
- **`per_device_train_batch_size=2`** (vs. Gemma's `1`) — Unsloth's own default for this
  much smaller model, which leaves more per-step memory headroom on a T4/L4.
- **No `get_chat_template()` call** — the official Qwen3.5 (0.8B) Vision notebook doesn't
  call it either; Qwen3.5's own default chat template is already correct for this model.
- **`flash-linear-attention` + `causal_conv1d` install** — Qwen3.5's Gated DeltaNet
  hybrid-attention backbone (linear-attention/state-space, not a plain transformer) needs
  these specialized kernels; Gemma's install cell has no equivalent. `causal_conv1d`'s PyPI
  source build is a known failure point (`unslothai/unsloth` issue #4688) if the pinned
  version ever needs bumping — see the install cell's own comment for the from-source
  workaround.
- **Two base-model script-check probes, run before any training** (plain-text prompt, then
  the real image+instruction task) — re-verifies the §2.104 finding against the exact
  environment/model version actually loaded, rather than assuming it's still true. If this
  is ever fixed upstream, these probes are what would show it.
- **Script-detection diagnostic, two levels** — (1) whole-row, via direct Unicode
  Khmer-block-vs-Thai-block codepoint counting (exact for this binary disambiguation, no
  extra dependency); (2) per-region, broken down by label, using the same codepoint method
  **plus** a `fast-langdetect` (fasttext-based) cross-check as a second, independent signal.
  `fast-langdetect` is skipped on regions with no alphabetic content (bare numbers/
  punctuation) rather than forced into a guess, since its own docs note accuracy drops on
  short/non-linguistic text. This answers not just "did it fail" but "which fields, exactly,
  and do two independent methods agree."
- **`EPOCHS` sweep knob, starting at 3** — not yet empirically re-verified for this model
  (unlike Gemma's, which is grounded in 3 real runs' worth of evidence). Carried over as a
  starting point, to be re-swept against this model's own results, not assumed transferable.

## v3 — first trial, `Soxavin/ardb-sft-v3` (2026-08-03)

**Base-model probes (before any training)**: both probes generated fluent Thai instead of
Khmer at zero training steps — see `docs/PROJECT_LOG.md` §2.104 for the full transcript and
diagnosis. Reverted the project's production fine-tune target to Gemma 4 E2B as a result.
Adapter kept at `Soxavin/qwen35-ardb-lora-v3` as the documented finding, not deleted.

No full training run was completed against v3 — the base-model probes were conclusive enough
on their own to redirect effort, and v3 has since been superseded by v5's dataset fixes
(era-stratified splits, leakage fix — see `eval/gemma_finetune_runs.md` / the dataset
READMEs for detail).

## v5 — era-stratified, multi-year corpus

Dataset: `Soxavin/ardb-sft-v5` (47 non-frozen documents / 128 pages: 101 train / 9
validation / 18 test, both structural templates represented in every split).

**Run 1: 39 steps, 3 epochs, first run on `ardb-sft-v5`** (2026-08-05) — the first completed
Qwen3.5 training run of this project (the earlier v3 base-model probes never reached a full run;
see above). Loss dropped cleanly 0.528 → ~0.025-0.035, same clean shape as every Gemma run.
Training: 225.2s, peak reserved memory 7.582 GB / 22.034 GB (LoRA-only: 1.146 GB). Adapter
pushed to `Soxavin/qwen35-ardb-lora-v5-e3`.

**Not directly comparable step-for-step to Gemma Run 4, despite both being "3 epochs"**: Qwen's
`per_device_train_batch_size=2, gradient_accumulation_steps=4` gives an effective batch size of
8, vs Gemma's `1 × 4 = 4` — so 3 epochs over the same 101 rows took Qwen only 39 steps against
Gemma's 78. Half the gradient updates for the "same" epoch count is a real confound worth
carrying into any Gemma-vs-Qwen comparison, not just a footnote.

**Eval**: 5/9 JSON parse failures — notably worse than Gemma Run 4's 3/9 on the same validation
split. Script check (whole-row): 7 khmer / 2 thai / 0 neither — the §2.104 Thai-vs-Khmer
base-model finding **still reproduces after fine-tuning**, in 2 of 9 rows outright, plus
contamination inside otherwise-Khmer rows (per-region, among the 4 rows that parsed:
`Section-Header` 1 khmer/1 thai, `Table` 3 khmer/1 thai, `Page-Furniture` 1 khmer/0 thai —
`fast-langdetect` agrees on every count). `Section-Header` mean CER 1.012 (n=2, >1.0 means the
prediction is longer than the reference and mostly wrong), `Table` mean CER 0.548 (n=4),
`Page-Furniture` mean CER 5.182 (n=1, wildly verbose/wrong relative to a short reference); mean
bbox coordinate abs diff **122.86** (0-1000 scale) — roughly 27x worse than Gemma Run 4's 4.5.

**Read, from the inference-check sample (`doc_010` page 0)**: two failure modes, not one.
(1) **Content hallucination, not just wrong-word substitution** — the Section-Header was
replaced wholesale with an unrelated, fabricated title including a hallucinated date
(`ប្រចាំថ្ងៃទី២៣ ខែកក្កដា ឆ្នាំ២០២៦`, nowhere in the source), and table item names were replaced
with plausible-sounding but wrong Khmer commodity words throughout (`សាច់គោរស់` → generated
`ដើម្បីធញ្ញជាតិ`, etc.) — a more severe version of Gemma's word-substitution pattern, not
confined to isolated words. (2) **Numeric table content, in contrast, was largely correct** —
prices and percentages matched the expected values in most rows even where the item-name label
next to them was wrong, suggesting the model retains the table's numeric structure/alignment
better than its Khmer vocabulary for this content. (3) Label misclassification: the two
letterhead lines were emitted under `"Text"` rather than the expected `"Page-Furniture"` — a
category the model wasn't confident distinguishing from generic prose here.

**Plausible contributors, not yet isolated**: `max_length=2048` (this model's real, Unsloth-
enforced ceiling — see `docs/PROJECT_LOG.md` §2.107) truncates the longest Table targets
(~4000 chars) during training itself, which the model never sees complete; combined with half
the gradient updates of Gemma's "3 epochs," undertraining relative to Gemma is a more likely
primary explanation than the base-model Thai gap alone, though that gap is also confirmed still
present. Not yet controlled for individually — a next run holding effective batch size (or
step count) equal to Gemma's would separate "undertrained" from "still base-model-limited."

**Run 2: 78 steps, 3 epochs, `GRAD_ACCUM=2`, A100** (2026-08-05) — the step-matched rerun this
"next run" note above called for: `per_device_train_batch_size=2, gradient_accumulation_steps=2`
gives effective batch 4 and exactly 78 steps, identical to Gemma Run 4's step count (Run 1 above
used `GRAD_ACCUM=4`, effective batch 8, only 39 steps — not comparable). Trained on an **A100**,
not L4 — the L4 session stopped partway through and was abandoned, so this is a fresh retrain, not
a resumption. Training: 271.8s, peak reserved memory 6.443 GB / 79.251 GB (LoRA-only: 4.324 GB;
the 79.251 GB total confirms A100, vs. Run 1's L4 22.034 GB); loss dropped cleanly 0.604 → ~0.007-
0.02, same shape as every other run (full per-step curve in `eval/loss_history.csv`, `qwen/v5-
run2` — see `docs/figures/finetune_eval/loss_by_run.png`: notably, Qwen's loss sits roughly an order of
magnitude above Gemma's for most of training at this *same* step count, only converging close by
step 78, a first real signal for "undertrained relative to Gemma" independent of the eval numbers
below; the loss values are close to but not identical to an earlier same-config attempt, expected
GPU-to-GPU floating-point variation, not a discrepancy worth chasing). Adapter pushed to
`Soxavin/qwen35-ardb-lora-v5-e3-ga2` (overwrites the abandoned L4 attempt at the same repo).

**Eval** (scored with the bracket-repair-capable parser, see `docs/PROJECT_LOG.md` and the eval
cell's `_try_repair_json` — not yet with the `no_repeat_ngram_size`/timing speed fixes, which are
generation-time only and don't affect these results): **7/9 (78%) JSON parse failures — worse
than Run 1's 5/9 (56%) despite matching Gemma's step count**, and `recovered via bracket-repair
fallback: 0`. Script check (whole-row): 9/9 khmer, 0 thai, 0 other — the Thai-contamination issue
from Run 1 (7 khmer/2 thai) did **not** reproduce this run, a real improvement independent of the
parse-failure result. Where it did parse (2/9): `Table` mean CER 0.4252 (n=2), `Page-Furniture`
mean CER **0.0000** (n=2, exact match — the letterhead text itself is fully correct when it
parses, consistent with Run 1's finding that letterhead *content* was already right, just
mislabeled there); mean bbox abs diff 41.04 (n=6) — worse than Gemma Run 4's 4.5, but better than
Run 1's 122.86.

**Why bracket-repair recovered 0 of the 7 failures**: the inference-check sample (`doc_010` page
0, sampling-based, a separate generation from the greedy eval loop) revealed the raw output is
missing **both** the outer list's opening `[` *and* closing `]` — the model emits a bare comma-
separated sequence of `{...}` objects with no wrapping array at all, not just a dropped closing
bracket. The original `_try_repair_json` only handled the latter (required the text to already
start with `[`), so it correctly declined to touch this case and left it uncounted. Fixed:
`_try_repair_json` now finds the last complete object (last `}`), discards any trailing junk
after it (e.g. a stray end-of-turn token), and adds whichever of `[` / `]` is actually missing —
verified against both failure modes plus a still-must-fail mid-cut case. Applied to both models'
eval cells and `eval_finetune_real.py`. Not yet re-run against this adapter, so Run 2's 7/9 figure
above may improve once it is.

**Read, from the same inference-check sample**: two new observations beyond the missing-bracket
finding above. (1) **Table column-count mismatch**: the generated header has 7 columns (splitting
what should be one `លក់រាយ` ["retail"] column per date into two garbled sub-columns, `ល.រក` and
`ល.ក់ចង្កឹះ`, for both the 31-08-23 and 01-09-23 dates), while every data row still has only 6
cells — an internal header/row inconsistency distinct from anything seen in Run 1. (2)
**Recurring hallucinated date in the Section-Header**: the real header text was replaced with a
fabricated `... ប្រចាំថ្ងៃទី១ ខែកក្កដា ឆ្នាំ២០២៦` ("... daily, July 1 2026") — the *same* kind of
invented date Run 1's inference-check sample showed (`ថ្ងៃទី២៣ ខែកក្កដា ឆ្នាំ២០២៦`, July 23 2026).
Two data points isn't proof, but a recurring hallucinated-date pattern specifically in this field,
across two different runs, is worth watching for a third occurrence rather than dismissing as
one-off noise. (3) Letterhead text (`ធនាគារ ARDB`, `ដើម្បីកសិករនិងអភិវឌ្ឍន៍សេដ្ឋកិច្ចសង្គម`) is
again exactly correct, again mislabeled — `Section-Header` this time, vs. Run 1's `Text` — a third
distinct wrong label choice for the same content across two runs, suggesting the model has no
stable preference for this field's category, not just an occasional slip.

**Read, on the headline numbers**: matching Gemma's step count did not close the gap, and by
parse-failure rate it got *worse*, not better — this weighs against "undertrained relative to
Gemma" as the primary explanation and back toward the base-model Thai/script gap, the newly-found
missing-both-brackets format issue, or some other Qwen-specific limitation (Gated DeltaNet
architecture, `max_length=2048` truncating training targets) being the harder constraint. The
loss-curve gap noted above (Qwen sitting persistently higher than Gemma's at identical steps) is
consistent with this reading. Not conclusive from two runs, but the "just needs more/matched
steps" hypothesis is now the weaker of the two live explanations, not the stronger one — and
worth re-testing once the bracket-repair fix is re-run against this same adapter, since the true
parse-failure rate (excluding the format bug) is still unknown.

**Real-doc eval, same adapter (`Soxavin/qwen35-ardb-lora-v5-e3-ga2`)** (2026-08-05) — scored via
`scripts/colab_eval_real.ipynb` (generation, with the `no_repeat_ngram_size`/timing speed fixes
already applied) + `scripts/eval_finetune_real.py` (scoring, with the generalized bracket-repair
parser) against `eval/datasets/real`'s 15 real, hand-verified ARDB pages — tracked in
`eval/real_doc_eval.csv` alongside Gemma's same-format entry.

- **15/15 (100%) JSON parse failures — 0 recovered even with the generalized repair fix.** This
  is qualitatively different from anything seen before, Gemma's real-doc failures included: the
  raw output has broken key *names* (`"box_2 d"` with a stray space mid-key, `"box_1 d"`,
  `"bbox_2d\\"` — three different corruptions of the same field across two samples), mismatched
  quote characters (`"label\': Picture,` — no closing quote around the value at all), and
  parentheses used in place of brackets for coordinate arrays (`(160, 115, (830, 200)`). No
  bracket-level repair can fix a broken key name or the wrong bracket type — this is a different,
  deeper failure class than the missing-`[`/`]` bug, not a harder case of the same bug.
- **6/15 generations hit the 4096-token cap** (vs. Gemma's real-doc run, which never came close to
  its cap even before this trim); mean generation time 344.6s/page, total 5168.7s (~86 minutes)
  for 15 pages — far slower than Gemma's real-doc run, and one capped generation's raw text
  visibly degrades into complete gibberish (garbled HTML-tag-like fragments, a stray `�`
  replacement character) by the point it gets cut off, consistent with a runaway, ungrounded
  generation rather than a page that legitimately needed more tokens.
- **Read**: on real, out-of-distribution documents specifically, Qwen's output quality collapses
  far more severely than on the synthetic validation split (which, even at its worst — Run 2's
  7/9 failures — never showed broken key names or wrong bracket types, only the missing-brackets
  issue). It's also markedly worse than Gemma's real-doc showing, where even the 7 comprehensively-
  broken pages stayed structurally repairable once the bracket fix landed. Combined with the
  script/architecture concerns already raised in Run 1 and Run 2's "Read" sections, this is
  the strongest evidence yet that Qwen's gap to Gemma is not primarily an epoch-count or
  step-count issue — real-document generalization looks like the harder, more fundamental
  limitation for this model on this task.

**Run 3: 52 steps, 2 epochs, `GRAD_ACCUM=2`, L4** (2026-08-06) — the epoch-sweep continuation
from Gemma's own evidence (2 epochs clearly beat 5 on Gemma's v2 dataset), testing whether less
training helps Qwen the same way. Training: 260.3s, peak reserved memory 6.416 GB / 22.034 GB
(LoRA-only: 4.217 GB); loss dropped cleanly 0.604 → ~0.02-0.04 by step 52 (full curve in
`eval/loss_history.csv`, `qwen/v5-run3`) — not as low as either 3-epoch run's ~0.006-0.02, simply
because there are fewer steps to converge over, not itself a red flag. Adapter pushed to
`Soxavin/qwen35-ardb-lora-v5-e2-ga2`.

**Eval: this hypothesis did not hold — 2 epochs is dramatically worse, not better.**
**9/9 (100%) JSON parse failures**, up from Run 2's 7/9 (78%) and Run 1's 5/9 (56%) — parse
failure rate has now gotten monotonically *worse* as epochs went 3 (Run 1) → 3 (Run 2) → 2
(Run 3), the opposite direction from Gemma's own epoch sweep. `recovered via bracket-repair
fallback: 0` — same as Run 2, none of these are the bracket-only bug. **The Thai-script
contamination Run 2 had resolved reappeared**: whole-row script check 7 khmer / 2 thai — the
identical 7/2 split Run 1 showed, restored. Generation was also markedly slower and more
runaway: mean 401.1s/row (vs. no comparable per-row timing for Run 2, which used the
pre-instrumentation eval cell), **6 of 9 rows hit the 4096-token cap**.

**Read, from the inference-check sample (`doc_010` page 0)**: the raw output is not merely
malformed, it's substantially more degenerate than anything in Run 1 or Run 2 — half a dozen
different corrupted spellings of the same field across one sample (`"box_2 d"`, `"box₂d"`
[subscript Unicode 2], `"box₂_d"`, `"BOX₂D"`, `"row_2d"`, `"r o w_2 d"` with spaces between
individual letters), literal HTML `<table><tr><th>` tags bleeding into the middle of the JSON,
and the generation eventually devolves into a multi-hundred-token wall of bare numbers with no
JSON structure at all. This is a full breakdown of the output format, not a formatting quirk.

**Why this matters for the "undertrained vs. base-model-limited" question**: this result argues
*against* undertraining/overfitting framing entirely. If overfitting on a small dataset were the
primary issue (as it was for Gemma on v2), *less* training should have helped, not hurt. Instead,
less training let the base model's known Thai-generation tendency (§2.104) partially resurface
and made the output format collapse further — consistent with Qwen needing *more* exposure to
the target format to keep it constrained, not less. Combined with the real-doc eval's finding
(15/15 failures, comprehensively broken even on the 3-epoch adapter), the picture forming across
all three runs is that Qwen's core limitation is about grounding/format-stability under this
task, not a training-duration tradeoff the way it was for Gemma — the two models don't appear to
share the same failure mechanism, so the same epoch-sweep logic that helped one doesn't
transfer to the other.

**Run 4: 130 steps, 5 epochs, `GRAD_ACCUM=2`, L4** (2026-08-06) — the upward half of the sweep,
directly testing Run 3's own closing hypothesis: if Qwen needs *more* exposure to the target
format to stay constrained (not less), 5 epochs should look better than 3, not worse. It doesn't.

- `Num examples = 101 | Num Epochs = 5 | Total steps = 130`; training: 466.5s, peak reserved
  memory 6.416 GB / 22.034 GB (LoRA-only: 4.217 GB) — matches Run 3's memory figures almost to
  the decimal (same batch shape), a useful consistency check; loss dropped from 0.604109 to
  **0.004025**, the deepest fit of any Qwen run (Run 2's 3-epoch floor was ~0.007-0.02, Run 3's
  2-epoch floor was ~0.02-0.04 — monotonically lower as epochs increase, as expected).
- Adapter pushed to `Soxavin/qwen35-ardb-lora-v5-e5-ga2`.
- Eval (synthetic val, 9 rows): **9/9 JSON parse failures, 0 recovered** — same failure *rate* as
  Run 3, but far more expensive and, qualitatively, far worse. **7 of 9 rows hit the 4096-token
  cap outright** (an 8th reached 3879/4096, effectively also capped), and generation cost
  **3809.6s total, mean 423.3s/row** — by a wide margin the most expensive eval run in this
  project (previous worst: Run 3's 401.1s/row on a much shorter total). Whole-row script check:
  9/9 khmer, 0 thai — the Thai-contamination pattern from Runs 1 and 3 did *not* reproduce here,
  but there's no region-level script/CER breakdown at all this run, since zero rows parsed
  successfully.
- Inference-check sample (`val_dataset[0]`, `doc_010` page 0) is the most degenerate output seen
  anywhere in this project so far: beyond inconsistent key corruption (`box_2D`, `box_3D`,
  `box-3D` — three different capitalizations/separators for the same field in one sample), the
  generation invents an entirely new, fabricated tag vocabulary that appears nowhere in the
  training schema (`<T-Column>`, `<T-Cell>`, `<T-Row>`, `<T-Section-Header>`, `<T-Table>`,
  `<Box-2D>`, `<Box-3D>`) and runs on for thousands of characters of low-coherence
  Khmer-character soup without any sign of naturally terminating — consistent with the 7/9
  cap-hit rate above. This isn't a formatting slip, it's the model losing track of the output
  schema entirely partway through generation.
- **Read — this closes the epoch question for Qwen, and it lands exactly where Gemma's did**:
  three points now exist on Qwen's own v5 epoch sweep — 2 epochs (Run 3) → 9/9 failures, 3 epochs
  (Run 2) → 7/9, 5 epochs (this run) → 9/9 again, but qualitatively worse (runaway,
  never-terminating generation vs. Run 3's merely malformed-but-bounded output) and far more
  expensive to even evaluate. Like Gemma, **3 epochs is a non-monotonic local optimum** — both
  fewer and more epochs regress from it, and this specifically refutes Run 3's "more exposure
  should help" hypothesis rather than confirming it. Between the two models, this is now a
  genuinely symmetric finding, not a coincidence isolated to one architecture: on this dataset
  scale (101 train rows), 3 epochs is the best-performing point found in both models' sweeps, and
  there's no evidence pushing toward the mentor's ~10-epoch suggestion would help either one —
  if anything, the trend argues it would make both worse.
- Real-doc eval not run for this adapter, same reasoning as before, reinforced by cost this time:
  at 423.3s/row on 9 synthetic-val rows, a 15-page real-doc run would likely cost well over
  100 minutes of L4 time for a result the synthetic-val signal already makes clear.
