# Gemma 4 E2B / ARDB fine-tune — run log

Durable record of each Colab training run for `scripts/colab_gemma4_e2b_finetune.ipynb`,
so results can be cited/compared later without scrolling back through chat history. Append
a new entry per full run — never edit past entries' results, only add corrections as notes
if a bug is later found to have affected them (see the "real GT bug found and fixed" note
under v2 below for an example of this pattern).

## Training setup decisions — why, not just what

Every non-default choice in `scripts/colab_gemma4_e2b_finetune.ipynb`'s config, with the
reason it was made. Most are also inline code comments in the notebook itself; this section
exists so the reasoning survives even if a comment gets trimmed, and so it can be cited in
one place (e.g. for a mentor question) without re-reading every cell.

- **Framework: Unsloth, hand-edited notebook, not Unsloth Studio (no-code UI)** — mentor's
  explicit choice, specifically because Studio would hide the exact mechanics worth learning
  from this internship (dataset shaping, chat template, LoRA/QLoRA config, `SFTTrainer`
  setup). This is a pedagogical constraint, not a technical one.
- **`UNSLOTH_COMPILE_DISABLE=1`, set before `unsloth` is imported** — page targets vary a lot
  in length (long table + title + letterhead vs. a sparser page), which produces enough
  distinct compiled shapes to hit Dynamo's `recompile_limit`
  (`FailOnRecompileLimitHit`), crashing training outright. First attempted fix
  (`torch._dynamo.config.recompile_limit = 64` set beforehand) did NOT work — Unsloth's own
  compilation patch silently resets it after model load. Disabling the auto-compiler entirely
  is Unsloth's own documented fix for this failure mode.
- **`device_map={"": 0}`** — without it, `accelerate`'s automatic device_map inference threw
  `ValueError: "Some modules are dispatched on the CPU or the disk"` even though a 5.2B-param
  4-bit model (~2.6GB weights) trivially fits a 14.5GB T4. Root cause: likely an
  `accelerate`/`bitsandbytes` version mismatch from cell 1's unpinned installs — the "fix" is
  forcing the whole model onto GPU 0 rather than trusting the auto-dispatch decision.
- **4-bit QLoRA base + LoRA (`r=32`, `lora_alpha=32`, `target_modules="all-linear"`,
  `finetune_vision_layers/language_layers/attention_modules/mlp_modules=True`)** — the T4's
  16GB is the binding constraint; 4-bit quantization + modest LoRA rank is what makes full
  fine-tuning of a 5.2B vision-language model fit at all on a free-tier GPU. r=32 was kept
  deliberately modest rather than maxed out, since smoke-test memory headroom hadn't been
  pushed — raise it only if a smoke test comfortably fits and more capacity is wanted.
- **`per_device_train_batch_size=1`, `gradient_accumulation_steps=4`** — per-step batch size
  of 1 is itself a T4-memory constraint (vision-language models with image tokens are memory-
  heavy per example); gradient accumulation recovers a more stable effective batch size of 4
  without needing more VRAM per step.
- **`max_steps = 10 if SMOKE_TEST else -1`, never `None`** — `TrainingArguments._validate_args`
  unconditionally evaluates `max_steps > 0 and num_train_epochs > 0`, so `max_steps=None`
  crashes with a `TypeError` the instant it's compared — this only surfaced once
  `SMOKE_TEST=False` was actually tried, since the smoke path always sets a real int. `-1` is
  the library's real "unset" sentinel, not `None`.
  - **`num_train_epochs = 1 if SMOKE_TEST else 3`, changed from 5 → 3 (was 2 before that)** —
    dialed by direct evidence across 3 full runs, not a formula: 2 epochs/34 steps → 1/9 JSON
    parse failures (undertrained); 5 epochs/85 steps → 3/9; a repeat of that same 85-step
    config on the corrected dataset → 9/9 (all rows). Training loss dropped cleanly to
    ~0.002-0.005 in every 5-epoch run regardless — a textbook small-dataset (66 rows)
    overfitting signature: the model memorizes harder without becoming more reliable at the
    structural task (always emit every region, always close the JSON list). 3 epochs (51
    steps) is a deliberate middle point between the under- and over-trained extremes, not a
    final answer — see Run 4 below for whether it worked.
- **`max_length=4096`** — the longest observed page target (full table markdown + title +
  letterhead + footer, all in one combined JSON blob) runs up to ~2600 chars; 4096 leaves
  headroom once image tokens and the instruction text are added on top.
- **Eval `generate()` uses `do_sample=False` (greedy), while the separate inference-showcase
  cell keeps sampling (`temperature=1.0, top_p=0.95, top_k=64`)** — an earlier version of the
  eval cell used sampling too, which made CER numbers non-reproducible run to run (same
  checkpoint, different scores) — pointless for comparing runs against each other. Greedy
  decoding is deterministic, so it's the only mode that makes cross-run comparison valid. The
  qualitative showcase cell doesn't feed into any recorded number, so sampling there is fine
  (and arguably more representative of real inference use).
- **Eval `max_new_tokens=3200`** — sized off the same ~2600-char worst-case table blob, with
  extra headroom because Khmer tokenizes less efficiently per character than Latin text. A
  v1-era run at `max_new_tokens=1024` measurably truncated a 2093-char target at 1272 chars,
  which is why this cap is generous rather than tight.
- **Adapter pushed to the Hub (`e806b6bd`) immediately after training, before any eval
  cells** — added after a real Colab free-tier disconnect happened mid-eval and risked losing
  an entire completed training run, since `save_pretrained` alone doesn't survive the VM
  being torn down. Pushing right after `trainer.train()` — the expensive, hard-to-repeat part
  — means only the (rerunnable) eval cells are ever at risk from a disconnect, never the
  training itself.
- **Per-row progress printing in every `generate()`-in-a-loop cell** — added after repeatedly
  being asked whether a long-running cell was stuck. Eval does a full uncompiled greedy
  generation (up to 3200 tokens) per validation row, which has no output otherwise for
  minutes at a time; each cell now prints `[i/n] generating doc_id=... page=...` before each
  call so silence never has to be diagnosed as a hang.
- **Dataset schema itself (v1 split configs → v2 unified one-forward-pass JSON)** — not a
  training-config choice but the biggest single decision this arc: driven by mentor feedback
  that layout detection and transcription must happen in one forward pass, not two. Full
  rationale lives in the "v2 — unified page understanding" section below and in the plan file
  referenced from `project_gemma4_ardb_finetune` memory.

## v1 — split configs (transcription + layout), superseded 2026-07-28

Dataset: `Soxavin/ardb-gemma-sft-v1` (2 HF configs, 177 combined train rows). Superseded
after mentor feedback that layout detection and transcription must happen in one forward
pass, not two separate tasks — see `Soxavin/ardb-gemma-sft-v2` below.

**Run: 90 steps, 2 epochs** (`per_device_train_batch_size=1`, `gradient_accumulation_steps=4`)

- Training: 1423.2s, peak reserved memory 12.199 GB / 14.563 GB (LoRA-only: 0.185 GB)
- Eval (greedy decoding, `do_sample=False`): `layout` CER 0.071 (9 rows), `letterhead` CER
  0.000 (3 rows), `table` CER 0.309 (9 rows), `title` CER 0.201 (3 rows, small sample)
- Finding: `table`'s CER was partly a truncation artifact — `max_new_tokens=1024` cut off a
  2093-char target at 1272 chars. Fixed with per-task `max_new_tokens` caps, but the
  corrected number was never re-measured before the mentor's one-forward-pass feedback made
  this dataset design obsolete.

## v2 — unified page understanding (bbox + text, one forward pass)

Dataset: `Soxavin/ardb-gemma-sft-v2` (single flat schema, 66/9/9 train/val/test rows). One
instruction, one JSON list per page: `{"box_2d", "label", "text"}` per region.

**Run 1: 34 steps, 2 epochs** (2026-07-28, generic instruction wording already applied,
GT typo below NOT yet fixed)

- `Num examples = 66 | Num Epochs = 2 | Total steps = 34` (66 rows × 2 epochs ÷ 4
  grad-accum = 17 steps/epoch)
- Eval: 1/9 JSON parse failures; region count mismatches: 2 (`Text` missing on 2 pages);
  `Page-Furniture` CER 0.315 (16 matched), `Section-Header` CER 0.641 (3 matched), `Table`
  CER 0.399 (8 matched); mean bbox coordinate abs diff 4.8 (0-1000 scale)
- Read: bbox accuracy already good; text accuracy clearly undertrained relative to v1's
  isolated-field task (expected — joint prediction is harder, and only 34 steps).

**Run 2: 85 steps, 5 epochs** (2026-07-28, same dataset content as Run 1 — GT typo below
NOT yet fixed)

- `Num examples = 66 | Num Epochs = 5 | Total steps = 85`
- Training: loss 0.335 → ~0.002-0.005 by the end (steady decline, a few small bumps;
  ended low enough to suggest some overfitting risk on only 66 rows)
- Eval: 3/9 JSON parse failures (up from 1/9); region count mismatches: 0; `Page-Furniture`
  CER 0.216 (12 matched), `Table` CER 0.010 (6 matched, but see caveat below), `Text` CER
  0.000 (3 matched); mean bbox coordinate abs diff 1.2 (0-1000 scale, excellent)
- **Caveat on the CER numbers**: CER is only computed over rows that parsed successfully,
  so 3/9 failures are silently excluded from those averages — the "0.010" Table CER looks
  better than it should, since a third of the harder rows aren't counted at all.
- Inference-check spot errors found (real, not just CER-invisible noise): title day
  generated as "3" when it should be "2"; table header label "ល.ក" instead of "ល.រ";
  fabricated wholesale prices in cells that should be blank; a repeated "បោច" vs "បោត"
  character confusion across 3 rows (unrelated to the GT typo below — a different word).
- Diagnosed the JSON parse failures directly (not guessed): one failing row's raw
  (unparsed) output showed the model completely omitted the `Section-Header` region, then
  stopped generating right after `Table` without closing the JSON list — not a
  `max_new_tokens` truncation issue (generated length was well under the cap). A
  completeness/reliability gap, not a localization problem (bbox stayed excellent).

**2026-07-28 — real GT bug found and fixed (affects both runs above)**: the frozen
`09.06.26_p3` ground truth (the actual *template source* `build_ardb_template_sft.py`
reads for the Table's fixed row labels) had a typo: `បោតក្រហម(គ្រាប់)` should be
`ពោតក្រហម(គ្រាប់)`. Also present identically in `15.06.26_p3`. Since this is a *template*
label (not a per-document substituted field), it was baked verbatim into every training
row's Table text in both runs above. Fixed both GT files (both the `paragraphs` and
`tables.data` copies), rebuilt the dataset (same 84 ok / 1 quarantined / 2 skipped page
counts — only text content changed), repackaged, and re-pushed to
`Soxavin/ardb-gemma-sft-v2`. Full test suite green (909 passed) after the fix.

**Run 3: 85 steps, 5 epochs** (2026-07-29, first run against the GT-typo-corrected dataset;
a mid-training Colab disconnect occurred during the eval cells afterward, but the training
loss curve completed cleanly and looks unaffected)

- `Num examples = 66 | Num Epochs = 5 | Total steps = 85`; training: 2689.5s, peak reserved
  memory 12.162 GB / 14.563 GB (LoRA-only: 0.148 GB); loss 0.335 → ~0.002-0.005, same clean
  shape as Run 2
- Eval: **9/9 JSON parse failures — every validation row**, a sharp regression from Run 2's
  3/9 despite an identical config and a corrected dataset
- Inference-check spot errors, worse than Run 2's: only 1 `Page-Furniture` region emitted
  instead of 2 (both letterhead lines incorrectly merged into one entry's `text`), and that
  merged text was **hallucinated** — "ធនាគារ អរបឌ (ARDB)" / a different tagline, matching
  neither training data nor the real letterhead; `Section-Header` and `Table` boxes visibly
  wrong (not just slightly off, as in earlier runs — e.g. `Section-Header` box only 11 units
  tall vs. the expected ~44); table rows that should carry both wholesale and retail values
  collapsed to only 4 values each, as if the model lost the row-group column-count
  distinction it had handled correctly before
- **Read**: don't blame the disconnect — the loss curve is clean and looks like a normal
  completed run. The real signal is the 3-run trend below.

**Trend across all three v2 full runs** (steps ↔ JSON parse failures): 34 steps → 1/9;
85 steps → 3/9; 85 steps (repeat) → 9/9. More steps has consistently correlated with *worse*
structural reliability even as training loss drops cleanly every time — a small-dataset
(66 rows) overfitting signature: the model memorizes harder without generalizing more
reliably, and which specific way it breaks seems to vary a lot run to run at 5 epochs.

**2026-07-29 — epochs dialed back to 3** (51 steps), as a middle point between the
under-trained 2-epoch run and the apparently-overfit 5-epoch runs, based on the trend above.

## v5 — era-stratified, multi-year corpus

Dataset: `Soxavin/ardb-sft-v5` (47 non-frozen documents / 128 pages: 101 train / 9 validation /
18 test, both structural templates represented in every split — see that dataset's README for
the full breakdown; supersedes v2's date-clustered-only split).

**Run 4: 78 steps, 3 epochs, first run on `ardb-sft-v5`** (2026-08-05) — first full run on the
era-stratified, expanded dataset (v2's 66 train rows → v5's 101), and the first real test of the
epochs-dialed-back-to-3 decision made after Run 3's v2 finding above, now against different data.

- `Num examples = 101 | Num Epochs = 3 | Total steps = 78`; training: 884.2s, peak reserved
  memory 11.33 GB / 22.034 GB (LoRA-only: 3.465 GB) — note the 22.034 GB total is larger than
  Runs 1-3's T4 (14.563 GB total), so this session likely ran on a bigger GPU tier (e.g. L4),
  not a free T4 — worth knowing before comparing wall-clock training time across runs, though it
  doesn't affect the loss/CER numbers themselves; loss dropped cleanly 0.326 → ~0.003-0.01, same
  shape as every prior run.
- Adapter pushed to `Soxavin/gemma4-e2b-ardb-lora-v5-e3`.
- Eval: 3/9 JSON parse failures; region count mismatches: 1 (`doc_035 p2 Text`: expected 1, got
  0); `Page-Furniture` CER 0.000 (12 matched, perfect), `Section-Header` CER 0.263 (3 matched,
  small sample), `Table` CER 0.115 (6 matched); mean bbox coordinate abs diff 4.5 (0-1000 scale).
- **Read**: 3/9 parse failures on the larger, era-stratified dataset at 3 epochs lands at the
  same rate as Run 2's on the smaller v2 dataset at 5 epochs — a bigger, better-split dataset did
  not, on its own, fix the structural-completeness gap the epoch pullback was meant to address.
  One run isn't enough to re-open the epoch question (an epoch sweep against v5 itself is still
  the standing next step, see the plan), but 3 epochs shouldn't be read as "solved" just because
  it's lower than 5.
- Inference-check spot errors (`val_dataset[0]`, `doc_010` page 0 — this row parsed successfully,
  so it's not one of the 3 failures): text content is very close to ground truth (Page-Furniture
  and most of the Table match near-exactly), but with a recurring failure mode — the model swaps
  in a plausible-but-wrong Khmer word rather than misspelling one: `មាន់សាច់រស់ (CP)` → generated
  `មាន់សាច់បោច (CP)`; `មាន់ជម្រុះពង រស់` → generated `មាន់ជ្រៃប្រដាប់ រស់`; `សាច់ទា` → generated
  `ពងទា`; `សាច់ទាកាប៉ា` → generated `សាច់ឆ្អឹងជំនី`; plus one dropped word, `មាន់៣សាសន៍បោចហើយ` →
  generated `មាន់៣សាសន៍ហើយ`. Bounding boxes on this specific sample drifted more than the 4.5
  mean (Page-Furniture/Section-Header boxes off by roughly 11-21 units each here), so the
  aggregate average is being pulled down by other, tighter-matching rows — not every row sits
  close to the mean.
- **Candidate for Track E (post-hoc correction lexicon), not yet actionable**: the word-
  substitution pattern above is exactly the kind of systematic error the plan's
  correction-lexicon step targets, but per that plan's own rule a pair only qualifies once it
  recurs across ≥2 distinct documents — this is one document's worth of examples so far.
