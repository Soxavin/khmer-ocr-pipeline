# Track A — fine-tuning a layout detector (what happens and why)

*A walkthrough for us, not a formal doc. Status: dataset published to HF; mentor trains on L4.*

## Why we're doing this

§2.37 measured our biggest reliability problem: **Surya's layout stage is non-deterministic** —
on the wide budget table its column count swings 14–21 between identical runs, and one run found
no table at all. Recognition is fine; *finding the table box consistently* is not. A small YOLO
detector fine-tuned on our own documents is deterministic: same page in → same boxes out, every
time. That's the whole bet. If it also *finds* the boxes as well as Surya does, it wins.

## The moving parts

1. **The dataset** (done): `ardb-layout-coco-v1` on HF — 91 pages, human-corrected boxes,
   5 classes, exactly one `Table` box per page *including the header row and label columns*
   (§2.24 failed precisely because an off-the-shelf model clipped label columns).
2. **The training** (mentor, L4): standard Ultralytics fine-tune. The whole thing is ~4 commands:

   ```python
   # one-time: convert our COCO splits to YOLO layout (or export YOLOv8 from Roboflow directly)
   # then:
   from ultralytics import YOLO
   model = YOLO("yolo11s.pt")            # small model — 73 train images can't feed a big one
   model.train(data="data.yaml", epochs=100, imgsz=960, patience=20, seed=0)
   # artifacts: runs/detect/train/weights/best.pt  ← this file is what comes back to us
   ```

   Expect minutes-to-an-hour on L4. `imgsz=960` is a compromise: our pages are 2000px, tables
   are large so even 640 works, but thin gridlines resolve better at higher res.
3. **The return path** (built, commit `44366b8`): drop `best.pt` anywhere local and run the
   pipeline with

   ```bash
   KHMER_LAYOUT_DETECTOR=doclayout KHMER_LAYOUT_WEIGHTS=/path/to/best.pt uv run python -m khmer_pipeline.pipeline <pdf>
   ```

   (needs `uv add 'ultralytics>=8.3,<9'` once, when the weights arrive).

## How we judge it (the two gates — same pattern as every model this sprint)

**Gate 1 — cheap probe.** Ultralytics prints mAP@50 per class on our test split at the end of
training; the mentor can read it straight off the run. What we want: `Table` mAP@50 ≳ 0.9
(single-template data, should be easy) — and, more importantly, when we run the detector twice
on the same page, **identical boxes** (this determinism is the point; Surya can't do it).

**Gate 2 — end-to-end A/B (the one that decides).** A good box score can still lose the
extraction war (§2.24's detector had beautiful boxes that amputated the label column). So:
run the full harness on ALL ground-truth docs — ARDB *and* the budget table — 3 runs each,
YOLO-layout vs Surya-layout, comparing **Recall + Numeric_Cell_Accuracy** (the honest pair;
Cell_Accuracy inflates on sparse tables).

- **GO** = structure stability improves (stable column counts run-to-run) with no recall loss
  → the engine switch becomes a documented option for ARDB-type docs.
- **NO-GO** = document the negative result with numbers (§2.24 precedent — a NO-GO with
  evidence is still thesis material, and it cost us almost nothing because the dataset
  exists regardless).

## Honest expectations

- The training data is one template (ARDB dailies). The detector will likely be an **ARDB
  specialist** — great on dailies, unknown on anything else. That's fine: Surya stays the
  default engine; the fine-tuned detector is an option, per the modular-pipeline philosophy.
- The real generalization test is the budget-table GT doc, which the model never saw. If it
  holds up there too, that's a bonus worth a highlighted line in the report.
- Monthly retrain story (mentor's ask): new month's PDFs → pseudo-labeler → Roboflow correction
  (~2–4 h) → re-run the same training command → re-run the same two gates. The Week-4 runbook
  formalizes exactly this loop.
