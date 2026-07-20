# Human-in-the-loop learning & the monthly retrain

*How analyst corrections become training data, and the procedure for retraining on
them safely. Companion to `PROJECT_LOG.md` (decisions) and
`experiments/layout_yolo/README.md` (the layout track).*

---

## Why this exists

Every day an analyst uses the workspace, they fix cells the model got wrong. Those
fixes are the scarcest and most valuable data in the project: **verified human
labels for real, in-domain failures** — the misread ៛, the slipped digit, the
long-tail cases that no synthetic corpus reproduces well. Until now they were
thrown away (the webapp kept edits in in-memory session state only).

This is the mentor's *continual learning* directive delivered as working
infrastructure plus a procedure, rather than a survey.

**Honest scope.** What is built is the **capture, curation and ingestion** path,
demonstrated end to end. It is **not** a completed retrain, and no accuracy gain is
claimed from it: with a 7-page golden set and no accumulated correction history, any
batch today is far too small to move the metrics. The value is that corrections now
*accumulate* in a trainable form instead of evaporating.

---

## The loop

```
analyst fixes a cell in the workspace
        │
        ▼
[1] CAPTURE   corrections.capture_corrections()
              diff model output vs the corrected grid
        │     → crop PNG + corrected text  (+ nested provenance)
        ▼
[2] CURATE    gold-standard rule · cosmetic-edit filter · validate.py flags
        │
        ▼
[3] ACCUMULATE   corrections/corrections.jsonl  (append-only, gitignored)
        │
        ▼   (when a batch is worth training on)
[4] BUILD     build_trainset.py --corrections
        │     mixes with real + hanuman + targeted-synthetic
        ▼
[5] TRAIN     train_kiri.py
        │
        ▼
[6] GATE      gate_ab.py on the golden set   ← deploy ONLY if it wins
        │
        ▼
   promote weights to models/kiri_finetuned/
```

Steps 4–6 are the path §2.39 already validated end to end; steps 1–3 are what this
work added.

---

## [1] Capture — what gets recorded

`src/khmer_pipeline/corrections.py`

Each captured fix is one JSONL line. The top-level keys are exactly what
`build_trainset.py` consumes; everything else is nested under `provenance` so a
future retrain can filter without a data-cleaning pass:

```json
{"image": "crops/ardb_p0_t0_r12_c3.png",
 "text": "៛/គ.ក",
 "origin": "correction",
 "provenance": {"prediction": "អ/គ.ក", "flags": ["sequence_illegal", "low_conf"],
                "confidence": 0.41, "source": "ardb.pdf", "page": 0, "table": 0,
                "row": 12, "col": 3, "bbox": [x0, y0, x1, y1],
                "engine": "surya_kiri", "timestamp": "…Z"}}
```

**Crops come from the frame the recognizer actually read** (the geometric-only
`recognition_page_images`), not the photometric display frame — §2.30 measured that
photometric normalization degrades Kiri, so a training crop must look like an
inference crop.

**Prerequisite that had to be built:** per-cell geometry. `surya_kiri` crops every
cell to feed Kiri but used to discard the box (`_build_table_from_grid` wrote
`bbox: []`), leaving corrections unpairable with pixels. Boxes are now threaded
through in page space alongside confidence, keyed identically so text, confidence
and geometry can never drift apart.

**Only `surya_kiri` can be captured from** — it is the sole per-cell-cropping
engine; `surya`'s VLM path has no per-cell geometry. Cells without a usable box are
skipped, never crashed on.

---

## [2] Curate — why not everything is training data

Training on every keystroke would teach the recognizer an analyst's formatting
habits instead of fixing its character errors. Three rules, in order:

1. **The gold-standard rule.** Only analyst-**verified** tables are captured. The
   model never learns from its own unverified output — that is how a pipeline
   amplifies its own mistakes.
2. **Cosmetic edits are dropped.** Differences that vanish under normalization are
   not recognition errors.
   > **Subtlety worth knowing:** `khmer_normalize` deliberately *preserves* ZWNJ /
   > ZWJ because they affect Khmer shaping. That is correct for text handling, but
   > it means two visually identical strings differing only by a joiner compare
   > **unequal** — and would have become a bogus training pair. `corrections.py`
   > therefore folds invisible characters *for the equality test only*; the stored
   > text stays the analyst's exact string, and the normalizer is untouched.
3. **Flags are recorded, not filtered.** `validate.py`'s taxonomy (`low_conf`,
   `sequence_illegal`, `digit_mixed`, `numeric_unparseable`, `numeric_mismatch`,
   `structure_ragged`) travels with each record, so error classes can be selected at
   build time instead of re-derived from raw text.

---

## [3–6] The monthly retrain procedure

Run when a meaningful batch has accumulated (order hundreds of corrections, not a
handful — see the volume caveat).

```bash
# 3. inspect what has accumulated
wc -l corrections/corrections.jsonl

# 3b. VISUAL GATE — never skip before training on new crops
uv run python scripts/verify_corrections.py --out corrections
open corrections/contact_sheet.html
```

The contact sheet labels each crop `prediction → correction` and groups by layout
path. **Scan it.** An off-by-origin bbox produces plausible-looking but shifted
crops that would silently poison the fine-tune; a systematic drift is obvious in
seconds here and invisible in the JSONL.

```bash
# 4. build the trainset (optionally selecting error classes)
uv run python experiments/kiri_finetune/build_trainset.py \
    --out experiments/kiri_finetune/trainset_$(date +%Y%m) \
    --corrections corrections \
    [--corrections-flags sequence_illegal,digit_mixed]

# 5. train (see reference_no_local_vision_training: Kiri is small enough to train
#    locally; vision/layout models must go to Colab)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python experiments/kiri_finetune/train_kiri.py \
    --trainset experiments/kiri_finetune/trainset_$(date +%Y%m)

# 6. REGRESSION GATE — the deploy decision
KHMER_KIRI_WEIGHTS=stock        uv run python experiments/kiri_finetune/gate_ab.py --tag baseline
KHMER_KIRI_WEIGHTS=<candidate>  uv run python experiments/kiri_finetune/gate_ab.py --tag candidate
```

**Promote only if the candidate wins on the golden set** — Recall + Numeric_Cell_
Accuracy, no per-page regression. A candidate that improves the average while
regressing a page is a no-go. Record the decision in `PROJECT_LOG.md` either way; a
documented negative result is still evidence (§2.24 precedent).

Corrections are **append-only** and never deleted after a retrain: the same corpus
is reused in later batches, and provenance makes it auditable.

---

## Limits, stated plainly

- **Volume.** A handful of corrections will not move the golden-set numbers. This is
  infrastructure that compounds with use, not an immediate accuracy win.
- **Capture is `surya_kiri`-only** (the only engine with per-cell geometry).
- **Recognition only.** Corrections teach Kiri to *read* better. They do not fix
  table **structure** errors (wrong row/column splits) — that is the layout track
  (`experiments/layout_yolo/README.md`).
- **Provenance mixing.** The existing "real" crops come from raw 200-DPI renders
  while correction crops come from the preprocessed geometric frame. Note this in
  any datacard; do not silently blend provenance.
- **Confidently-wrong cells** are captured only if an analyst happens to notice
  them — the loop inherits whatever the review workflow surfaces.
- **Privacy.** `corrections/` holds real GDDE financial-document crops and is
  gitignored, same handling as `corpus/`. It must not be published without the same
  sign-off the source documents need.

---

## Status

| stage | state |
|---|---|
| per-cell geometry | **built** (round-trip verified on real pages) |
| capture + curation | **built**, 11 tests |
| trainset ingestion | **built** (`--corrections`, flag filtering) |
| loop demo + visual gate | **built** (`scripts/verify_corrections.py`, both layout paths) |
| webapp save-hook | **deferred** — wiring `capture_corrections` into the save/verify path |
| a real retrain on captured data | **not done** — needs accumulated volume |

The one piece between this and a live loop is the webapp hook: call
`capture_corrections` when an analyst verifies a table, writing to `corrections/`.
