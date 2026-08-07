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
- **Train-only image augmentation (v5 onward)**: brightness jitter ±10%, contrast jitter ±10%,
  and a 20%-chance light Gaussian blur (radius 0.1-0.3) — applied only to `converted_dataset`
  (the training split, via `convert_to_conversation(sample, train=True, rng=_aug_rng)`);
  `val_dataset` is read directly by the eval cells and never routed through it. Deliberately
  **no geometric augmentation** (rotation/crop/scale/affine) — every training row's `box_2d`
  targets are ground-truth coordinates that would need re-projecting to match any geometric
  transform, which this pass doesn't do, and table row/column alignment is exactly what the
  model needs to learn, so an unprojected transform would just make the label wrong rather than
  help. Identical implementation in `colab_qwen35_finetune.ipynb` (same parameters, same
  train/val split point), so it isn't a confound between the two models' results — see
  `docs/PROJECT_LOG.md` §2.106 for the original decision record.

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

**Real-doc eval, same adapter (`Soxavin/gemma4-e2b-ardb-lora-v5-e3`)** (2026-08-05) — scored via
`scripts/colab_eval_real.ipynb` (generation) + `scripts/eval_finetune_real.py` (scoring) against
`eval/datasets/real`'s 15 real, hand-verified ARDB pages — a genuinely out-of-distribution test
set (not template-substituted synthetic data the model's own training distribution resembles more
closely), tracked separately in `eval/real_doc_eval.csv` since its metric shape (whole-page
`table_cer`/`cell_accuracy`/`text_cer`/`document_cer`) doesn't match the per-label CER schema the
synthetic-val CSVs use.

**Update (2026-08-05, same predictions, rescored)**: `eval_finetune_real.py`'s `_try_repair_json`
was generalized after a Qwen run revealed a second failure mode (missing *both* the opening `[`
and closing `]`, not just the closing one — see `docs/PROJECT_LOG.md` and
`eval/qwen_finetune_runs.md`'s Run 2). Rescoring the *same* `real_predictions.json` (no
regeneration) with the fixed parser recovers all 7 previously-unparseable pages — **0/15 parse
failures now, down from 7/15**. The aggregate numbers get *worse*, not better, as a result: mean
`table_cer` 0.611 (was 0.270), mean `cell_accuracy` 0.348 (was 0.653). This is not a regression —
it's the previously-hidden failures finally being counted instead of excluded. All 7 newly-
recovered pages score exactly `table_cer=1.000, cell_accuracy=0.000` — not a partial improvement,
the worst possible score — meaning these pages' generations were comprehensively broken, both
structurally (missing brackets) *and* substantively (empty/unusable table content), not two
independent defects. The original 8 pages that already parsed keep their original scores
unchanged. See `eval/real_doc_eval.csv` for the corrected row (old numbers kept in its own note
for traceability, not silently overwritten).

- **The "page 2" pattern holds, and is now sharper**: the 7 comprehensively-broken pages are
  *exactly* the same 7 that failed to parse before the fix — 4 of 5 documents' 2nd page (typically
  a table continuation, longer/denser content), plus one document's 1st and 3rd pages instead.
  This reframes the earlier hypothesis: it's not "these pages have a JSON formatting quirk," it's
  "generation comprehensively breaks down on these specific pages" — the bracket omission is a
  symptom of the same failure, not a separate, milder bug.
- **Where generation actually succeeded** (the original 8 pages, scores unchanged): mean
  `table_cer` 0.270, mean `cell_accuracy` 0.653 (vs. mean `numeric_cell_accuracy` 0.742 — numeric
  content held up better than full-cell accuracy, the same "numbers survive better than labels"
  pattern seen in Qwen's runs). Mean `text_cer` (prose regions, all 15 pages) 1.099 — over 1.0,
  meaning prose predictions are on average longer than and substantially diverge from the
  reference.
- **Read**: real ARDB documents are meaningfully harder than the synthetic val split — but not
  because they use a different template. The entire corpus (2022-2026) reduces to only 2
  structural templates, and every real-doc-eval page uses one of the same 2 templates train/val
  does (verified: `_DEFAULT_EXCLUDE_STEMS` in `harvest_table_gt.py` excludes all 5 real-doc-eval
  source documents, including both template anchors, from train/val entirely — so these are
  unseen *pages*, not an unseen template family). The real difference is GT construction: train/
  val's fixed labels (column headers, commodity names, section headers) are copied verbatim from
  the template regardless of what's on the page, so those cells are correct by construction and
  don't actually test whether the model read them off the image; only the per-document numbers/
  dates are genuinely page-specific there. Real-doc GT is fully hand-verified per page, everything
  included, on pages the model never saw. So real-doc eval tests "can it read the fixed template
  content off real pixels, on unseen pages" — narrower than "generalizes to new structure," but
  still the harder and more meaningful test. The headline is now **generation comprehensively
  fails on roughly half the real pages** (7/15), not "half the pages have a parsing quirk." That's
  a starker, more accurate picture of current real-doc readiness than the pre-fix numbers
  suggested, and the p2/continuation-page pattern is the strongest lead so far for *why*.

**Run 5: 52 steps, 2 epochs, clean rerun** (2026-08-06) — a first attempt at this same config was
run and *withheld* from this log: step-1 loss (0.136) and LoRA-only peak memory (0.189 GB) were
both far below what a fresh run should show, strong evidence the Colab runtime had silently kept
the already-fine-tuned model loaded from a prior session (uploading a new notebook file does not
guarantee a new kernel/VM). User did `Runtime → Restart session` and reran; this entry is that
clean rerun.

- **Freshness confirmed, not just asserted**: step-1 loss 0.326041 and step-2 loss 0.340867 are
  *identical* to Run 4's (same values to 6 decimal places — expected, since both runs share the
  same seed, data order, and step-vs-epoch ratio of 26 steps/epoch, so the first couple of steps
  before the LR schedules diverge are fully deterministic); LoRA-only peak memory 3.446 GB is
  within noise of Run 4's 3.465 GB. Both numbers now land where a genuine fresh 2-epoch run
  should, resolving the contamination concern — this run's numbers are trustworthy.
- `Num examples = 101 | Num Epochs = 2 | Total steps = 52` (L4, `per_device=1, GRAD_ACCUM=4`,
  same effective batch 4 as every other v5 run); training: 592.8s, peak reserved memory 11.311 GB
  / 22.034 GB (LoRA-only: 3.446 GB); loss 0.326 → ~0.005-0.06 (noisier tail than Run 4 — two
  spikes back up to ~0.058 at steps 26 and 52, i.e. the last step of each epoch).
- Adapter pushed to `Soxavin/gemma4-e2b-ardb-lora-v5-e2` — **this overwrites the earlier
  contaminated attempt's push to the same adapter name**; nothing from that withheld run survives
  on the Hub.
- Eval (synthetic val, 9 rows): **9/9 JSON parse failures, 0 recovered via bracket-repair** —
  worse than Run 4's 3/9 at 3 epochs on the identical dataset/split. Generation was also far
  slower: 908.9s total, mean 101.0s/row (vs. no per-row timing captured for Run 4). Diagnostic
  sample (`doc_010` page 0): expected 2856 chars, generated only 1095 chars, and *every* region
  (Picture, both Page-Furniture lines, Section-Header, Table) came back `<region missing from
  prediction>` — not a bracket/formatting slip, the generated JSON simply didn't contain any
  region matching the expected labels.
- **Read — this changes the epoch-direction conclusion, not just adds a data point**: on the same
  `ardb-sft-v5` dataset and split, 3 epochs (Run 4) → 3/9 failures, 2 epochs (this run) → 9/9.
  Pulling epochs back made Gemma's structural reliability *worse*, the opposite of what the v2-era
  reasoning (66-row dataset, epochs pulled back 5→3 to fight overfitting) predicted. This mirrors
  Qwen's own finding on the same dataset (`eval/qwen_finetune_runs.md` Run 3: pulling Qwen back to
  2 epochs also made things worse, not better). Taken together, the two models now agree in
  direction: on `ardb-sft-v5`, less training hurts, not helps. That reopens the mentor's ~10-epoch
  suggestion as worth testing directly rather than dismissing it on the strength of the old v2
  overfitting evidence, which was measured on a much smaller dataset.
- Real-doc eval not run for this adapter (not requested this round — the synthetic-val result
  alone was clear enough not to justify the ~15-minute real-doc generation cost on a config
  already trending worse).

**Run 6: 130 steps, 5 epochs** (2026-08-06) — the upward half of the epoch sweep the Run 5 "Read"
proposed, testing whether more training (toward the mentor's ~10-epoch suggestion) reverses the
2-epoch regression.

- `Num examples = 101 | Num Epochs = 5 | Total steps = 130`; training: 1480.9s, peak reserved
  memory 11.33 GB / 22.034 GB (LoRA-only: 3.465 GB) — memory figures match Run 4's almost to the
  decimal (same batch shape), and step-1/step-2 loss (0.326041 / 0.340867) again match Runs 4 and
  5 exactly, confirming a fresh run, no contamination concern this time. Loss dropped to
  ~0.0015-0.006 by the end — lower than either Run 4's (~0.003-0.01) or Run 5's (~0.005-0.06)
  tail, i.e. the deepest fit of the three.
- Adapter pushed to `Soxavin/gemma4-e2b-ardb-lora-v5-e5`.
- Eval (synthetic val, 9 rows): **8/9 JSON parse failures, 1 recovered via bracket-repair** — worse
  than Run 4's 3/9, though marginally less bad than Run 5's 9/9. The one row that did parse
  (`doc_035` p2) still dropped 4 of its expected regions (Picture, one of two Page-Furniture
  lines, the entire Table, Text) — only `Page-Furniture` produced a scorable CER, and only 1
  matched row (CER 0.000, not statistically meaningful at n=1). No other label has any CER signal
  this run.
- Inference-check sample (`val_dataset[0]`, `doc_010` page 0) shows a qualitatively different, more
  severe failure mode than Runs 4/5: not just missing brackets, but per-region key corruption
  throughout (`"box_2x"`, `"box2d"` with no underscore, `"label"]=`, `=` used in place of `:`,
  mismatched quote styles within the same object) and the Table region's HTML degrades mid-
  generation into fabricated pseudo-XML tags (`<record>`, `<type>`, `<price>`) mixed with what
  reads as Vietnamese tokens (`"thuộc"`) — content that appears nowhere in this Khmer-only
  corpus. This looks like a genuine overfitting signature at the output-format level: extremely
  low training loss, but held-out generation doesn't degrade gracefully, it degrades into
  syntactically incoherent, cross-lingual noise.
- Generation was also slower and more variable: 1111.0s total, mean 123.4s/row (range 35.6s-451.0s
  per row — the widest spread seen in any Gemma run so far).
- **Read — this walks back Run 5's "reopens the ~10-epoch idea" conclusion, doesn't confirm it**:
  three points now exist on the v5 epoch sweep — 2 epochs → 9/9 failures, 3 epochs → 3/9, 5 epochs
  → 8/9. The relationship is **not monotonic**: 3 epochs is uniquely good, and both fewer *and*
  more epochs regress from it. Extending further toward the mentor's ~10 is not a natural next
  step on this evidence — 5 epochs already regressed sharply, so there's no basis yet to expect 10
  would reverse that rather than regress further. **3 epochs (`Soxavin/gemma4-e2b-ardb-lora-v5-e3`,
  Run 4) remains the strongest Gemma config found in this sweep** and the one to carry forward
  unless something else about the setup changes (more data, regularization, etc.) — not worth
  spending further Colab budget chasing higher epoch counts on the current data/config as-is.
- Real-doc eval not run for this adapter, same reasoning as Run 5: the synthetic-val result is
  unambiguous enough (worse than the current best on every axis, plus a new and more severe
  qualitative failure mode) that the ~15-18 minute real-doc generation cost isn't justified for a
  config that's already been superseded by Run 4's result.

**Local deployment investigation, `Soxavin/gemma4-e2b-ardb-lora-v5-e3`, webapp's `gemma_ardb`
engine** (2026-08-07) — first time this adapter was run through the actual local production
pipeline (`src/khmer_pipeline/engines/gemma_ardb_engine.py`, an isolated `uv run --no-project`
subprocess, distinct from the Colab-based real-doc eval above), rather than via Colab generation
scripts. Surfaced one real infra bug (now fixed) and, once fixed, reproduced the same
comprehensive-failure signature the Colab real-doc eval already found — narrowing the cause from
"maybe deployment" to "the adapter itself."

- **Bug found and fixed: LoRA attach crash outside Unsloth.** `gemma_ardb_infer.py` originally
  loaded `unsloth/gemma-4-E2B-it` + this adapter via plain
  `transformers.AutoModelForImageTextToText` and `peft.PeftModel.from_pretrained` (no Unsloth).
  This crashed every time: `ValueError: Target
  module Gemma4ClippableLinear(...) is not supported. Currently, only the following modules are
  supported: torch.nn.Linear, ...` — `unsloth/gemma-4-E2B-it` uses Unsloth's own layer classes,
  which stock `peft` doesn't recognize as LoRA-injectable outside an Unsloth environment. Not a
  training-run problem (Unsloth's own `FastVisionModel.get_peft_model` handled this fine during
  training) — purely a local-inference deployment gap.
- **Fix: merge once, in Colab, deploy the merged checkpoint.** Added a "Merge for local inference"
  section to `scripts/colab_gemma4_e2b_finetune.ipynb` that loads the adapter via
  `FastVisionModel.from_pretrained` (Unsloth, where LoRA attachment works) and calls
  `push_to_hub_merged(..., save_method="merged_16bit")`, producing a plain, non-PEFT checkpoint at
  `Soxavin/gemma4-e2b-ardb-merged-v5-e3`. `gemma_ardb_engine.py`'s `base_model_id` now points there;
  `gemma_ardb_infer.py` loads it directly with `AutoModelForImageTextToText.from_pretrained`, no
  PEFT step, no Unsloth dependency at inference time.
- **First real run through the fixed pipeline**: no crash, ~80s end-to-end on one page
  (`eval/datasets/real/…_០៣_០៤_សីហា_ឆ្នាំ២០២២…_p1.png` — one of the real-doc-eval corpus's 15
  pages), confirming the deployment path itself works mechanically. Output, however, was
  malformed JSON: hallucinated/wrong field names throughout (`box_2D` instead of `box_2d`,
  `label_2d_text`, a stray `-text` key, mismatched brackets), and the Table region's HTML degraded
  into broken/invented tags mid-generation — not a clean parse failure, a structurally incoherent
  one.
- **Root-cause investigation — two hypotheses raised, both disproven with direct evidence:**
  1. *Chat-template mismatch* (training used Unsloth's `get_chat_template(processor, "gemma-4")`;
     the merge step's `FastVisionModel.from_pretrained` silently reloaded the base model's default
     template instead — confirmed via file hash: merged repo's `chat_template.jinja` matched the
     *base* model's (18,810 bytes, sha256 `241c50d8…`), not the adapter's actual trained-with
     template (2,375 bytes, `b728115a…`) it should have carried over). **Fixed** (re-uploaded the
     adapter's actual `chat_template.jinja` + `tokenizer_config.json` onto the merged repo,
     verified hashes now match) — but re-running produced **byte-identical** output to before the
     fix. Confirmed the rendered prompt text itself (`processor.apply_chat_template(...)`, both
     template revisions, same input messages) was already identical, 385 characters, both before
     and after — the two templates happen to collapse to the same rendered string for this single-
     turn, one-image, no-system-prompt message shape. Hypothesis disproven by direct text diff, not
     inference.
  2. *Merge/dequantization precision loss* (4-bit QLoRA base dequantized + LoRA delta baked in
     during the merge — could a small model's fragile schema-following degrade from that numerical
     roundtrip?). Tested by loading the adapter the *unmerged* way, exactly as training did
     (`FastVisionModel.from_pretrained(adapter_repo, load_in_4bit=True)`, no merge step at all,
     via a new standalone Colab cell), on the identical page. Result: **the same failure family** —
     hallucinated field names (`box_2D`, `label_2d_text`, `text_content`), broken/mismatched HTML
     table tags, even a stray `model_` token leaking into the output. Two independent loading paths
     (Unsloth-native unmerged vs. our merged checkpoint) fail the same way. Hypothesis disproven.
- **Read**: neither infra hypothesis holds — the deployment pipeline (subprocess isolation, merge
  step, chat template) is confirmed correct and introduces no additional degradation beyond
  whatever the adapter itself already does. This corroborates and sharpens the Real-doc eval
  section above's "generation comprehensively fails on roughly half the real pages" finding: it's
  not a page-2/continuation-page-specific issue as that section's original hypothesis suggested —
  this test page is `p1` (a document's *first* page, not a continuation page) and still failed
  comprehensively, which is a genuine counter-example to that pattern, not a confirmation of it.
  The practical upshot for the local UI: `gemma_ardb` is wired, documented, and fails soft with the
  garbled output visibly surfaced (not silently blank) when this happens — the same
  "labeled trial, not production" posture already used for `qwen_ardb`, now confirmed warranted
  for Gemma too rather than being unnecessarily cautious phrasing.
