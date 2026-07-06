# Khmer CRNN Training Exercise — Findings

*A self-contained write-up of what was built, measured, and learned. Sibling
docs: [README.md](README.md) (how to run), [FINETUNING_PLAN.md](FINETUNING_PLAN.md) (next phase).*

---

## 1. Purpose and framing

This was a **learning + benchmarking exercise**, adapted from a mentor-provided
school assignment. The goal was **not** to ship a model, but to:

1. **Learn the training mechanics** of a Khmer text recognizer end-to-end.
2. **Measure real per-epoch training time on a MacBook** (Apple M4 Pro / MPS).
3. Do it with proper experimental rigor (validation split, no leakage, metrics).

**Two framings that must stay clear:**

- **This is training *from scratch*, not fine-tuning.** The model
  (`resnet34(weights=None)` + BiRNN + CTC) learns to read Khmer from zero. That
  distinction matters: the real "fine-tuning phase" (warm-starting a pretrained
  Khmer model) is a *separate, later* effort — see [FINETUNING_PLAN.md](FINETUNING_PLAN.md).
- **Single font, by design.** Training data is `seanghay/khmer-hanuman-100k`
  (one font, Hanuman). The resulting model will **not** read the real GDDE
  documents, which use other/legacy fonts. This exercise proves the *machinery*,
  not a production recognizer.

## 2. What the model is

A standard **CRNN** recognizer:

- **CNN stem:** ResNet (18 or 34), with the layer3/layer4 strides surgically
  changed to `(2,1)` so the **width (time) axis is preserved** — CTC needs many
  time-steps.
- **Sequence head:** a bidirectional **GRU or LSTM** reads the CNN feature
  columns left-to-right.
- **Loss:** **CTC** (Connectionist Temporal Classification), which aligns a
  variable-length text label to the image columns *without* needing a bounding
  box per character.

## 3. Adaptations we made to the starter script (and why)

The mentor provided a working starter script written for a **CUDA/Blackwell GPU**
environment. Adapting it to **our setup (Apple Silicon / MPS)** and **our research
goals** (validation, metrics, benchmarking) meant the following changes — kept
here as a record of what we changed and why (all in [train.py](train.py)):

| Starter script (as given) | Our adaptation — and why |
|---|---|
| Hardcoded `device="cuda"` | Portable device select (reuses `khmer_pipeline.utils.device.detect_device`) → picks **MPS** on the Mac |
| `GradScaler`/`autocast('cuda')` — CUDA-specific | fp32 on MPS (no scaler); AMP path kept only under CUDA |
| `torch.compile` (for Blackwell) | Opt-in flag, **skipped on MPS** (limited support there) |
| Single slice, no train/val split | Split **before** any augmentation; vocab built from **train split only** — so we can measure overfitting without leakage |
| No augmentation | Added mild, **train-only** affine/blur/jitter |
| No validation metric | Added greedy CTC decode + **validation CER** each epoch |
| No timing | Added warmup-aware per-epoch timing + throughput (our benchmark goal) |
| No seeding/checkpoints | Seeded; saves best-CER checkpoint + `metrics.csv` + `summary.json` |
| No explicit CTC length check | Added a `T ≥ label length` check for our long-line data (see §4) |

## 4. The key discovery — matching the model to *this* dataset

Adding a **CTC-feasibility check** (output time-steps `T` must be ≥ the label
length) surfaced a mismatch on the real data and forced us to actually look at it:

- **Labels are sentences, not words:** median **41 characters**, up to **139**.
- **Images are long horizontal strips:** median width **~1068 px** (up to 3514).

The starter script's default `Resize((48, 256))` suits short labels, but for
**this** long-line dataset it doesn't fit, for **two** reasons:
1. It squishes a ~1068 px line into 256 px — hurting legibility.
2. 256 px → only **32 time-steps**, but the *median* label is 41 chars → **CTC is
   infeasible for more than half the dataset**. Because CTC uses
   `zero_infinity=True`, those over-length samples would contribute **zero
   gradient** rather than raise an error — so without a check the mismatch is
   invisible (the model quietly trains on the short-label subset only).

**Our adaptation:** widen the input to **1024 px** (→ 128 time-steps, matching
the data's natural aspect ratio) and **filter** the few labels longer than `T`
(printed loudly, not dropped silently). A `--max-label-len` knob allows a
shorter-label curriculum.

> **Takeaway for the report:** adding the feasibility check turned an otherwise
> invisible data/config mismatch into an explicit, understood design decision.
> This is the single most valuable engineering outcome of the exercise.

## 5. Results

Environment: Apple **M4 Pro, MPS**, PyTorch. **Note:** `aten::_ctc_loss` is not
implemented on MPS, so the CTC loss runs on **CPU fallback**
(`PYTORCH_ENABLE_MPS_FALLBACK=1`); the ResNet fwd/bwd stays on MPS. All Mac
epoch times therefore include a per-step CPU round-trip for the loss.

### 5a. Benchmark — per-epoch training time (the primary deliverable)

Default config (ResNet34 + GRU, batch 64, 48×1024 images, ~9k Hanuman lines):

- **~121 s/epoch**, **~75 samples/sec**, ~850 ms/step.
- **Stable across epochs (123.9 → 120.9 → 119.2 s) — no thermal throttling**;
  the machine stayed cool over the run (the throttle proxy correctly stayed
  silent).

### 5b. Ablations — what drives epoch time

| Backbone | RNN | sec/epoch | samples/sec |
|---|---|---|---|
| resnet34 | gru | 121.4 | 74.9 |
| resnet34 | **lstm** | 111.8 | 80.3 |
| **resnet18** | gru | **76.0** | 121.0 |

- **Backbone dominates speed:** resnet18 is **~1.6× faster** than resnet34.
- **GRU vs LSTM barely matters for speed** here — cost is dominated by the CNN
  and the CPU-side CTC loss, not the RNN.

### 5c. Convergence — does it actually learn? (`runs/converge1`)

Config chosen to be *learnable* (isolate whether the pipeline is sound): ResNet18,
64×512 images, **labels ≤ 40 chars** (curriculum), LR 5e-4, 40 epochs (~20 s/epoch).

| Epoch | val CER | Note |
|---|---|---|
| 1–3 | **1.00** | Blank collapse (predicts empty) |
| **4** | **0.969** | 🔑 **Breakout — first drop below 1.0** |
| 7 | 0.547 | Rapidly learning |
| 10 | 0.188 | 81% chars correct |
| 20 | 0.059 | 94% correct |
| 40 | **0.034** | **96.6% character accuracy** |

Learning curve: [runs/converge1/learning_curve.png](runs/converge1/learning_curve.png).

**This proves the pipeline is sound.** The earlier `val_cer = 1.0` was **not a
bug** — it is the classic CTC dynamic of sitting at all-blank until it "aligns,"
then dropping fast. Given a learnable setup it reaches **3.4% CER** on the
Hanuman font.

**Full-length task also converges.** A second run with **no label cap** (full
sentence-length labels, 1024 px, ResNet18, 40 epochs, ~90 s/epoch) reached **3.7%
CER** (best, ~epoch 35), breaking out of blank-collapse at epoch 3. So the model
learns the *hard* task nearly as well as the curriculum — the pipeline is sound
on real-length Khmer lines, not just short ones.
(`runs/converge_full/`.)

## 6. Honest limitations

- **Single font (Hanuman).** Will not read the real GDDE documents.
- **The 3.4% CER is on the easy task:** labels ≤ 40 chars, taller 64 px images,
  and a 483-sample validation subset — not the full sentence-length problem.
- **Not fine-tuning:** trained from scratch; a production effort would warm-start
  a pretrained Khmer model (see [FINETUNING_PLAN.md](FINETUNING_PLAN.md)).
- **CTC-on-CPU on Mac** means these epoch times are not a fully-native-MPS
  measurement.

## 7. Reproduce

```bash
uv sync --extra experiments        # torchvision, datasets, psutil (pinned)

# smoke test
uv run python experiments/khmer_crnn/train.py --limit 512 --epochs 1 --max-steps 5

# benchmark (default config)
uv run python experiments/khmer_crnn/train.py --limit 10000 --epochs 3

# convergence (learnable curriculum)
uv run python experiments/khmer_crnn/train.py \
  --backbone resnet18 --img-h 64 --img-w 512 --max-label-len 40 \
  --lr 5e-4 --epochs 40 --run-name converge1

# plot any run's learning curve (matplotlib is in the `dev` extra)
uv run --extra dev python experiments/khmer_crnn/plot_metrics.py experiments/khmer_crnn/runs/converge1
```

Each run writes to `runs/<name>/`: `summary.json`, `metrics.csv`, `best.pt`,
and (via plot_metrics) `learning_curve.png`.
