# Khmer CRNN trainer (learning + Mac benchmark)

A from-scratch **CRNN** (ResNet stem → BiRNN → CTC) recognizer for Khmer text,
adapted from a mentor-provided starter script (originally CUDA-oriented) to our
Mac + research use-case. The point of this experiment is **to learn the training
mechanics** and to **benchmark real epoch time on Apple Silicon (MPS)** — *not*
to ship a production model. See [FINDINGS.md](FINDINGS.md) for the adaptations
we made and why.

> **This is training-from-scratch, not fine-tuning.** `resnet34(weights=None)`
> learns to read Khmer from zero. And it trains on a **single font** (Hanuman,
> via `seanghay/khmer-hanuman-100k`), so it will **not** transfer to the real
> GDDE documents, which use different/legacy fonts. Both facts are deliberate
> for a learning/benchmarking exercise.

## Install

The heavy deps (`torchvision`, `datasets`, `psutil`) live in an isolated
optional group so the core pipeline env stays lean:

```bash
uv sync --extra experiments
```

## Run

```bash
# Smoke test — a few steps, tiny slice, proves it runs end-to-end
uv run python experiments/khmer_crnn/train.py --limit 512 --epochs 1 --max-steps 5

# The benchmark (the primary deliverable) — real epoch time on this machine
uv run python experiments/khmer_crnn/train.py --limit 10000 --epochs 2

# Ablations to learn from
uv run python experiments/khmer_crnn/train.py --rnn lstm          # vs default gru
uv run python experiments/khmer_crnn/train.py --backbone resnet18 # lighter/faster
uv run python experiments/khmer_crnn/train.py --batch-size 32     # sweep batch size
uv run python experiments/khmer_crnn/train.py --no-augment        # augmentation off
```

> **First run downloads the full dataset.** `load_dataset("seanghay/khmer-hanuman-100k")`
> pulls the whole dataset before `--limit` selects a slice — expect a large,
> slow first download; it is cached afterward.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--limit` | 10000 | rows loaded from the split |
| `--val-frac` | 0.1 | held out for validation (split happens **before** augmentation) |
| `--rnn` | `gru` | `gru` or `lstm` |
| `--backbone` | `resnet34` | or `resnet18` (faster epochs) |
| `--batch-size` | 64 | starter script used 256; lowered for Mac unified memory |
| `--epochs` | 2 | ~2 epochs is enough for a timing benchmark |
| `--max-steps` | none | cap steps/epoch for quick smoke runs |
| `--augment` / `--no-augment` | on | mild, **train-only** affine/blur/jitter |
| `--aug-strength` | 1.0 | multiplier on augmentation magnitude |
| `--lr` / `--weight-decay` | 1e-3 / 1e-4 | AdamW |
| `--dropout` | 0.2 | inter-layer RNN dropout |
| `--compile` | off | `torch.compile`; skipped on MPS (unstable) |
| `--seed` | 42 | seeds python/numpy/torch |
| `--out-dir` | `experiments/khmer_crnn/runs` | per-run checkpoints + metrics land here |

## Reading the output

Each run writes to `runs/<run-name>/`:
- **`summary.json`** — the headline benchmark: device, precision, backbone/rnn,
  `avg_epoch_sec`, steady-state `samples/sec`, `best_val_cer`.
- **`metrics.csv`** — one row per epoch (`train_loss`, `val_cer`, `epoch_sec`,
  `samples_per_sec`, `ms_per_step`) so GRU-vs-LSTM / batch-size runs are comparable.
- **`best.pt`** — checkpoint at the best `val_cer` (weights + vocab + args).

**Overfitting signal:** watch `train_loss` ↓ while `val_cer` ↑ across epochs.
**Throttling proxy:** the per-epoch `samples/sec` — a sustained drop after the
warmup epoch is flagged in the console (`[throttle-proxy] ...`).

## Notes / caveats

- **CTC loss on MPS falls back to CPU.** `aten::_ctc_loss` is unimplemented on
  the MPS backend, so the script sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before
  importing torch. The ResNet forward/backward runs natively on MPS; only the
  loss round-trips to CPU each step. This is **surfaced in the run summary**
  (`ctc_loss_device: cpu-fallback`) so the epoch-time benchmark is honest —
  interpret Mac epoch times with that CPU round-trip in mind.
- **True GPU temperature needs privilege.** `psutil.sensors_temperatures()` is
  typically empty on macOS; the honest non-sudo signal is throughput drift
  (above). For real thermals: `sudo powermetrics --samplers smc`.
- **CTC feasibility is asserted at startup:** output timesteps `T` must be ≥ the
  longest target; otherwise `zero_infinity=True` would silently zero those
  gradients. If it fails, increase `--img-w` or use a lighter backbone.
