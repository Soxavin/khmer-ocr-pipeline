# Track A — fine-tuning a layout detector (what happens and why)

*A walkthrough for us, not a formal doc. Status: dataset published to HF; integration built
and verified; **training pending on Colab**.*

## Why we're doing this

§2.37 measured our biggest reliability problem: **Surya's layout stage is non-deterministic** —
on the wide budget table its column count swings 14–21 between identical runs, and one run found
no table at all. Recognition is fine; *finding the table box consistently* is not. A detector
fine-tuned on our own documents is deterministic: same page in → same boxes out, every time.
That's the whole bet. If it also *finds* the boxes as well as Surya does, it wins.

## What we fine-tune (and what we don't)

We fine-tune **DocLayout-YOLO from the DocStructBench-pretrained checkpoint**
(`juliozhao/DocLayout-YOLO-DocStructBench`) — a model that already knows what a document
table looks like. We do **not** train from generic COCO weights (`yolo11s.pt`): 73 training
images cannot teach "what a table is" from scratch, and §2.24 already showed an
off-the-shelf document detector is only a *starting point*, not an answer. This is the
mentor's directive read literally — *fine-tune on our docs*.

## The moving parts

1. **The dataset** (done): `ardb-layout-coco-v1` on HF / `eval/datasets/layout_v1_corrected`
   locally — 84/9/9 pages, human-corrected boxes, 5 classes, exactly one `Table` box per page
   *including the header row and label columns* (§2.24 failed precisely because an
   off-the-shelf model clipped label columns).

2. **The training** — **Colab T4, `scripts/colab_layout_finetune.ipynb`**.
   **Do not train locally**: `imgsz=1024` freezes the 24GB Mac, which already hosts
   PyTorch (Surya) + MLX. The notebook trains, reads Gate 1 off the run, exports the ONNX,
   and downloads it — the whole loop, ~10–20 min on a free T4.

3. **The return path** (built, verified): the notebook returns `khmer_layout_best.onnx`.
   Drop it anywhere local and:

   ```bash
   KHMER_LAYOUT_WEIGHTS=/path/khmer_layout_best.onnx uv run python -m khmer_pipeline.pipeline <pdf>
   ```

   **Why ONNX and not `best.pt`:** training uses the `doclayout_yolo` research fork, which
   pickles its own module paths into the checkpoint — stock `ultralytics` dies with
   `ModuleNotFoundError: doclayout_yolo`. And the fork cannot join the project venv either:
   it requires `opencv-python`, which collides with our `opencv-python-headless` (both ship
   `cv2`). ONNX has no Python class dependency, so the weights cross the venv boundary
   cleanly — and cost us **zero new production dependencies**, since `rapid_layout` +
   `onnxruntime` are already installed and `layout_detect.py` already speaks
   `doclayout_docstructbench`. (`rapid_layout` reads its class list from an ONNX `character`
   metadata key; the notebook and `export_onnx.py` inject it, else it raises
   `KeyError: 'character'`.)

   `.pt` weights still route to Ultralytics, so a stock-YOLO fine-tune (e.g. the mentor's L4
   run) also works — `layout_detect.py` picks the backend from the file extension.

## How we judge it (the two gates — same pattern as every model this sprint)

**Gate 1 — cheap probe.** The notebook prints test-split mAP@50 per class. What we want:
`Table` mAP@50 ≳ 0.9 (single-template data, should be easy) — and, more importantly,
**identical boxes across repeated runs**. Determinism is the point; Surya can't do it.
(Already confirmed for this architecture: the *pretrained* checkpoint gives byte-identical
boxes across 3 runs through `detect_table_boxes`.)

**Gate 2 — end-to-end A/B (the one that decides).** `experiments/layout_yolo/gate.py`.
A good box score can still lose the extraction war (§2.24's detector had beautiful boxes
that amputated the label column). It runs every GT doc — ARDB *and* the budget table —
× {surya, surya_kiri} × {layout on, off} × 3 runs, comparing **Recall +
Numeric_Cell_Accuracy** (the honest pair; Cell_Accuracy inflates on sparse tables) and
reporting run-to-run **shape stability**.

```bash
KHMER_LAYOUT_WEIGHTS=/path/khmer_layout_best.onnx uv run python experiments/layout_yolo/gate.py
```

Inference only — safe to run locally.

- **GO** = structure stability improves (stable dims run-to-run) with no recall loss
  → the engine switch becomes a documented option for ARDB-type docs.
- **NO-GO** = document the negative result with numbers (§2.24 precedent — a NO-GO with
  evidence is still thesis material, and it cost us almost nothing because the dataset
  exists regardless).

Scores come from the **§2.42-fixed** row aligner. Pre-§2.42 numbers are not comparable.

## Honest expectations

- The training data is one template (ARDB dailies). The detector will likely be an **ARDB
  specialist** — great on dailies, unknown on anything else. That's fine: Surya stays the
  default engine; the fine-tuned detector is an option, per the modular-pipeline philosophy.
- The real generalization test is the budget-table GT doc, which the model never saw. If it
  holds up there too, that's a bonus worth a highlighted line in the report.
- **Licence caveat, unresolved:** DocLayout-YOLO is an Ultralytics derivative and the export
  carries `AGPL-3.0`. Ultralytics asserts AGPL reaches models trained with their code, and our
  deliverable serves a web UI (AGPL §13's network clause). Serving via ONNX means no AGPL
  *code* ships, but the weights' provenance needs a decision before anything is published —
  same class of question as Surya's OpenRAIL-M weights. Raise with mentor/GDDE.
- Monthly retrain story (mentor's ask): new month's PDFs → pseudo-labeler → Roboflow correction
  (~2–4 h) → re-run the notebook → re-run the same two gates. The Week-4 runbook formalizes
  exactly this loop.
