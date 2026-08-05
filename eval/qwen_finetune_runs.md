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

## v5 — era-stratified, multi-year corpus (pending)

Dataset: `Soxavin/ardb-sft-v5` (47 non-frozen documents / 128 pages: 101 train / 9
validation / 18 test, both structural templates represented in every split).

**Run 1: pending** — first full run against `ardb-sft-v5`. Before the full run: re-check the
two base-model probes (Probe 1/2 in the notebook) against this exact load — if the Thai-vs-
Khmer finding no longer reproduces, that's itself a significant, worth-flagging result. If it
does reproduce (expected, since it's a pretraining-time gap), proceed with the full run and
epoch sweep anyway, and record the per-region/per-label script breakdown here alongside the
usual CER/bbox/parse-failure numbers — the goal is a precise, presentable measurement either
way, not just a pass/fail.
