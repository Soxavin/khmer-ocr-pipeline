"""Khmer CRNN (ResNet stem + BiRNN + CTC) trainer — adapted for our Mac + research use-case.

Starting point: a mentor-provided starter script that trains a ResNet34 + BiGRU
+ CTC recognizer from scratch on `seanghay/khmer-hanuman-100k`, written for a
CUDA/Blackwell GPU. We adapted it to our setup (Apple Silicon / MPS) and our
research goals — portability, a validation split + metrics, and per-epoch
benchmarking. This is a *from-scratch training* exercise, not fine-tuning: the
point is to learn the training mechanics and benchmark real epoch time on the
Mac, not to ship a model. See FINDINGS.md for the list of adaptations and why.

Read top to bottom — each section is self-contained and commented for that
purpose (favour clarity over abstraction).

Run the smoke test:
    uv run python experiments/khmer_crnn/train.py --limit 512 --epochs 1 --max-steps 5

See README.md in this directory for the full flag reference and how to read
the benchmark output.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

# CTC loss (aten::_ctc_loss) is not implemented for the MPS backend, so on
# Apple Silicon that single op must fall back to CPU. This must be set BEFORE
# importing torch; it is ignored on CUDA/CPU. Consequence: on MPS the epoch-time
# benchmark includes a per-step CPU round-trip for the loss (surfaced in the run
# summary) — the ResNet forward/backward still runs natively on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Reuse the repo's device selection — the one thing we're allowed to import
# from the shipped package. Do NOT hardcode "cuda" anywhere below.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from khmer_pipeline.utils.device import detect_device  # noqa: E402

from metrics import character_error_rate, greedy_ctc_decode  # noqa: E402

try:
    import psutil
except ImportError:  # optional; thermal proxy just degrades gracefully
    psutil = None


# ─────────────────────────────────────────────────────────────────────────
# 1. Config / argparse
# ─────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    p.add_argument("--dataset", default="seanghay/khmer-hanuman-100k", help="HF dataset id")
    p.add_argument("--limit", type=int, default=10000, help="rows to load from the dataset split")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of --limit held out for validation")
    p.add_argument("--max-label-len", type=int, default=None,
                    help="also drop labels longer than this many chars (curriculum: start short to "
                         "help CTC escape the all-blank collapse). Effective cap = min(T, this).")
    p.add_argument("--img-h", type=int, default=48, help="resize height fed to the CNN")
    p.add_argument("--img-w", type=int, default=1024,
                    help="resize width. This dataset is long text-lines (median ~1068px, labels up to "
                         "~139 chars); CTC timesteps grow ~img_w/8, so 1024 -> ~128 steps fits ~99%% of "
                         "labels. We raised this from the starter script's 256 (which suits short labels).")

    # Augmentation (train split only)
    p.add_argument("--augment", dest="augment", action="store_true", default=True,
                    help="enable mild train-only augmentation (default: on)")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                    help="disable train-only augmentation")
    p.add_argument("--aug-strength", type=float, default=1.0,
                    help="multiplier on augmentation magnitude (1.0 = defaults below)")

    # Model
    p.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet34")
    p.add_argument("--rnn", choices=["gru", "lstm"], default="gru")
    p.add_argument("--hidden", type=int, default=256, help="BiRNN hidden size (per direction)")
    p.add_argument("--rnn-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2, help="inter-layer RNN dropout")

    # Optimization
    p.add_argument("--batch-size", type=int, default=64,
                    help="starter script used 256; lowered for Mac unified memory")
    p.add_argument("--epochs", type=int, default=2, help="~2 epochs is enough for a timing benchmark")
    p.add_argument("--max-steps", type=int, default=None,
                    help="cap steps per epoch (quick smoke runs); None = full epoch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)

    # Precision / compile
    p.add_argument("--compile", action="store_true", default=False,
                    help="torch.compile — opt-in, off by default, skipped entirely on MPS")

    # Repro / IO
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="experiments/khmer_crnn/runs", help="checkpoints + metrics CSV land here")
    p.add_argument("--run-name", default=None, help="defaults to a timestamp")

    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no CUDA device present


# ─────────────────────────────────────────────────────────────────────────
# 2. Data: leakage-safe split, train-only augmentation, vocab from train only
# ─────────────────────────────────────────────────────────────────────────
#
# The starter script loaded a single slice with no train/val split; for our
# research goals we add a held-out validation set so we can detect overfitting.
# With a split, the order matters to avoid leakage (building the vocab from the
# full slice would leak val characters into training):
#   1. load raw rows
#   2. split indices (fixed seed) BEFORE anything else touches the data
#   3. build vocab from the train indices only
#   4. THEN construct datasets/transforms (augmentation is train-only)

def load_rows(dataset_id: str, limit: int):
    from datasets import load_dataset

    print(f"Step 1: Downloading/Loading dataset ({dataset_id})...")
    ds = load_dataset(dataset_id, split="train").select(range(limit))
    print(f"Loaded {len(ds)} samples.")

    text_col = next(c for c in ds.column_names if c in ("text", "label", "ground_truth"))
    img_col = next(c for c in ds.column_names if c in ("image", "img", "pixel_values"))
    return ds, text_col, img_col


def split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle + split BEFORE any vocab/augmentation logic touches the data."""
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def build_vocab(ds, text_col, train_idx: list[int]) -> tuple[dict[str, int], dict[int, str], int]:
    """Vocab built from the TRAIN split only. Val OOV chars are simply
    unencodable and therefore always count as CER errors (see metrics.py) —
    that's the intended, honest behavior, not a bug to work around.
    """
    train_chars = sorted({ch for i in train_idx for ch in ds[i][text_col]})
    c2i = {c: i + 1 for i, c in enumerate(train_chars)}  # blank = 0
    i2c = {i + 1: c for i, c in enumerate(train_chars)}
    vocab_size = len(train_chars) + 1
    print(f"Vocab built from train split only: {len(train_chars)} chars (+1 blank) = {vocab_size}")
    return c2i, i2c, vocab_size


def build_transform(img_h: int, img_w: int, augment: bool, strength: float) -> T.Compose:
    """Val transform is deterministic resize+normalize only. Train transform
    additionally applies mild, controlled augmentation — small affine jitter,
    slight blur, brightness/contrast — to mimic scan noise without breaking
    glyph shapes (Khmer subscripts/diacritics are small and easy to wreck
    with aggressive augmentation).
    """
    ops = [T.Resize((img_h, img_w))]
    if augment:
        ops += [
            T.RandomAffine(
                degrees=2 * strength,
                translate=(0.02 * strength, 0.02 * strength),
                scale=(1 - 0.05 * strength, 1 + 0.05 * strength),
                shear=2 * strength,
                fill=255,
            ),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6 * strength + 0.1))], p=0.3),
            T.ColorJitter(brightness=0.15 * strength, contrast=0.15 * strength),
        ]
    ops += [
        T.Grayscale(1),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ]
    return T.Compose(ops)


class KhmerImgDataset(Dataset):
    """Wraps a subset of the HF dataset (by index list) with a fixed transform.

    `augment` is a plain bool baked in at construction time — never toggled
    per-item — so train/val never accidentally share augmentation state.
    """

    def __init__(self, hf_ds, indices: list[int], text_col: str, img_col: str,
                 c2i: dict[str, int], img_h: int, img_w: int, augment: bool, aug_strength: float):
        self.ds = hf_ds
        self.indices = indices
        self.text_col = text_col
        self.img_col = img_col
        self.c2i = c2i
        self.transform = build_transform(img_h, img_w, augment, aug_strength)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.ds[self.indices[i]]
        img = row[self.img_col]
        if not isinstance(img, Image.Image):
            img = Image.open(BytesIO(img))
        x = self.transform(img.convert("RGB"))
        # OOV chars (only possible in val, by construction) are dropped here;
        # they still count against the reference string in CER (see below).
        target = [self.c2i[c] for c in row[self.text_col] if c in self.c2i]
        return x, torch.tensor(target, dtype=torch.long), row[self.text_col]


def pad_collate(batch):
    xs, ts, raw_texts = zip(*batch)
    imgs = torch.stack(xs)
    tgt_lens = torch.tensor([len(t) for t in ts], dtype=torch.long)
    targets = nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    return imgs, targets, tgt_lens, list(raw_texts)


# ─────────────────────────────────────────────────────────────────────────
# 3. Model: ResNet stem w/ stride-(2,1) surgery + switchable BiGRU/BiLSTM
# ─────────────────────────────────────────────────────────────────────────
#
# The stride-(2,1) surgery on layer3/layer4 keeps the time (width) axis from
# collapsing too fast, since CTC needs T (output timesteps) to be at least
# as long as the longest target sequence. We add an explicit check for this
# (see compute_timesteps + the length filter): with zero_infinity=True, any
# sample where T < target_len contributes zero gradient instead of raising, so
# on our long-line dataset it's worth surfacing rather than leaving implicit.

class KhmerCRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 256, backbone: str = "resnet34",
                 rnn_type: str = "gru", rnn_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        backbone_fn = {"resnet18": tv.resnet18, "resnet34": tv.resnet34}[backbone]
        rn = backbone_fn(weights=None)
        rn.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2)
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4
        for block_group in (self.layer3, self.layer4):
            for block in block_group:
                if hasattr(block, "conv1") and block.conv1.stride != (1, 1):
                    block.conv1.stride = (2, 1)
                if block.downsample is not None:
                    block.downsample[0].stride = (2, 1)
        self.vpool = nn.AdaptiveAvgPool2d((1, None))

        rnn_out_channels = rn.layer4[-1].conv2.out_channels if backbone == "resnet18" else 512
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        rnn_dropout = dropout if rnn_layers > 1 else 0.0  # nn.RNN warns/no-ops dropout with 1 layer
        self.rnn = rnn_cls(rnn_out_channels, hidden, num_layers=rnn_layers,
                            bidirectional=True, dropout=rnn_dropout)
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.vpool(f)                    # (B, C, 1, W)
        f = f.squeeze(2).permute(2, 0, 1)     # (W=T, B, C)
        out, _ = self.rnn(f)
        return self.fc(out)                  # (T, B, vocab_size)


def compute_timesteps(model: nn.Module, img_h: int, img_w: int, device: str) -> int:
    """Return the number of CTC output timesteps T the model emits for the given
    input width (one dummy forward). CTC requires T >= label length, so T is
    what decides which samples are learnable — see the length filter in main().
    We measure it explicitly because on this long-line dataset most labels don't
    fit the original 256px width, and zero_infinity=True would otherwise mask
    that by nulling those samples' gradients rather than erroring.
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, img_h, img_w, device=device)
        out = model(dummy)
    model.train()
    T_steps = int(out.shape[0])
    assert T_steps > 0
    return T_steps


# ─────────────────────────────────────────────────────────────────────────
# 4. Precision / device portability
# ─────────────────────────────────────────────────────────────────────────
#
# GradScaler is CUDA-only (it exists to counteract fp16 underflow, which is
# a CUDA-tensor-core-era concern). MPS gets plain fp32 by default — no
# autocast, no scaler. torch.compile is opt-in and explicitly skipped on
# MPS (unstable/limited support as of this torch version).

@dataclass
class PrecisionPlan:
    device: str
    use_amp: bool
    amp_dtype: torch.dtype | None
    use_scaler: bool


def make_precision_plan(device: str) -> PrecisionPlan:
    if device == "cuda":
        return PrecisionPlan(device, use_amp=True, amp_dtype=torch.bfloat16, use_scaler=False)
    # MPS and CPU: fp32, no autocast, no scaler. (bf16 autocast on MPS is
    # possible but flaky across torch versions — keep default path simple.)
    return PrecisionPlan(device, use_amp=False, amp_dtype=None, use_scaler=False)


def maybe_compile(model: nn.Module, device: str, compile_flag: bool) -> nn.Module:
    if not compile_flag:
        return model
    if device == "mps":
        print("[compile] --compile requested but skipped: torch.compile is unstable on MPS.")
        return model
    print("[compile] compiling model (torch.compile)...")
    return torch.compile(model)


# ─────────────────────────────────────────────────────────────────────────
# 5. Thermal / throttling proxy (non-sudo)
# ─────────────────────────────────────────────────────────────────────────
#
# True Apple Silicon GPU temperature requires `sudo powermetrics --samplers
# smc` — not viable to shell out to from an unprivileged training script.
# psutil.sensors_temperatures() is frequently EMPTY on macOS (no exposed
# sensors via the standard API), so we treat it as best-effort and fall back
# to the honest signal we *can* measure without privilege: throughput
# degradation across epochs. Sustained throttling shows up as a falling
# samples/sec trend even though the workload per step is constant.

def read_temps() -> dict:
    if psutil is None or not hasattr(psutil, "sensors_temperatures"):
        return {}
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return {}
    return {name: [t.current for t in entries] for name, entries in (temps or {}).items()}


def detect_throttling(samples_per_sec_history: list[float], drop_threshold: float = 0.15) -> str | None:
    """Flag a throttling proxy if throughput has dropped >drop_threshold
    relative to the best epoch seen so far (after the first, warmup epoch).
    """
    if len(samples_per_sec_history) < 3:
        return None
    warmed = samples_per_sec_history[1:]  # drop epoch 0 (warmup/compile overhead)
    best = max(warmed[:-1])
    latest = warmed[-1]
    if best > 0 and (best - latest) / best > drop_threshold:
        return f"throughput dropped {100 * (best - latest) / best:.0f}% vs best epoch (possible throttling)"
    return None


# ─────────────────────────────────────────────────────────────────────────
# 6. Validation: greedy decode + CER
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader, i2c: dict[int, str], device: str) -> float:
    model.eval()
    all_preds, all_refs = [], []
    for imgs, targets, tgt_lens, raw_texts in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        log_probs = logits.log_softmax(2)
        preds = greedy_ctc_decode(log_probs, i2c, blank=0)
        all_preds.extend(preds)
        all_refs.extend(raw_texts)
    model.train()
    return character_error_rate(all_preds, all_refs)


# ─────────────────────────────────────────────────────────────────────────
# 7. Training loop with warmup-aware per-epoch benchmarking
# ─────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optim, ctc_loss, precision: PrecisionPlan, device: str,
                     max_steps: int | None, epoch: int, epochs: int) -> tuple[float, float, float, float]:
    """Returns (avg_loss, epoch_sec, samples_per_sec, ms_per_step)."""
    model.train()
    total_loss, n_steps, n_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
    t0 = time.perf_counter()
    for step, (imgs, targets, tgt_lens, _raw_texts) in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        if precision.use_amp:
            with torch.autocast(device_type=device, dtype=precision.amp_dtype):
                logits = model(imgs)
                T_steps, B, _V = logits.shape
                log_probs = logits.log_softmax(2)
                input_lens = torch.full((B,), T_steps, dtype=torch.long, device=device)
                loss = ctc_loss(log_probs, targets, input_lens, tgt_lens)
            loss.backward()
            optim.step()
        else:
            logits = model(imgs)
            T_steps, B, _V = logits.shape
            log_probs = logits.log_softmax(2)
            input_lens = torch.full((B,), T_steps, dtype=torch.long, device=device)
            loss = ctc_loss(log_probs, targets, input_lens, tgt_lens)
            loss.backward()
            optim.step()

        total_loss += loss.item()
        n_steps += 1
        n_samples += B
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # MPS/CUDA both run async; sync before stopping the clock so epoch_sec
    # reflects actual compute time, not just kernel-launch time.
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    epoch_sec = time.perf_counter() - t0

    avg_loss = total_loss / max(n_steps, 1)
    samples_per_sec = n_samples / epoch_sec if epoch_sec > 0 else 0.0
    ms_per_step = 1000 * epoch_sec / max(n_steps, 1)
    return avg_loss, epoch_sec, samples_per_sec, ms_per_step


# ─────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    device = detect_device()
    precision = make_precision_plan(device)
    print(f"[device] using {device} | amp={precision.use_amp} dtype={precision.amp_dtype} "
          f"scaler={precision.use_scaler}")
    if device == "mps":
        print("[mps] aten::_ctc_loss is unimplemented on MPS — the CTC loss runs on CPU via "
              "PYTORCH_ENABLE_MPS_FALLBACK; epoch time includes that per-step round-trip "
              "(ResNet fwd/bwd stays on MPS).")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] writing checkpoints + metrics to {out_dir}")

    # ---- Data: split FIRST, vocab from train only, THEN augmented datasets ----
    ds, text_col, img_col = load_rows(args.dataset, args.limit)
    train_idx, val_idx = split_indices(len(ds), args.val_frac, args.seed)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val (seed={args.seed})")

    # Vocab from the (pre-filter) train split — a few extra classes from
    # dropped long samples are harmless (CTC just never targets them).
    c2i, i2c, vocab_size = build_vocab(ds, text_col, train_idx)

    # ---- Model (built before datasets so we can size the CTC time axis) ----
    model = KhmerCRNN(vocab_size, hidden=args.hidden, backbone=args.backbone,
                       rnn_type=args.rnn, rnn_layers=args.rnn_layers, dropout=args.dropout).to(device)
    T_steps = compute_timesteps(model, args.img_h, args.img_w, device)

    # CTC needs T >= label length. This dataset is long text-LINES (labels up to
    # ~139 chars), so at a given width only labels <= T are learnable. Drop the
    # longer ones loudly rather than let zero_infinity silently null their
    # gradients. Widen --img-w to keep more of them (T grows ~ img_w / 8).
    cap = T_steps if args.max_label_len is None else min(T_steps, args.max_label_len)

    def _fits(i: int) -> bool:
        return len(ds[i][text_col]) <= cap

    n_tr0, n_va0 = len(train_idx), len(val_idx)
    train_idx = [i for i in train_idx if _fits(i)]
    val_idx = [i for i in val_idx if _fits(i)]
    print(f"[ctc-filter] T={T_steps} timesteps (img_w={args.img_w}), label cap={cap}: kept "
          f"{len(train_idx)}/{n_tr0} train, {len(val_idx)}/{n_va0} val "
          f"(dropped labels longer than {cap} chars)")
    if not train_idx:
        raise SystemExit(f"No training samples fit T={T_steps}. Increase --img-w.")
    max_target_len = max(len(ds[i][text_col]) for i in train_idx)
    print(f"[ctc-check] T={T_steps} >= longest kept label {max_target_len} "
          f"(ratio {T_steps / max(max_target_len, 1):.2f}x) — OK")

    train_ds = KhmerImgDataset(ds, train_idx, text_col, img_col, c2i,
                                args.img_h, args.img_w, augment=args.augment, aug_strength=args.aug_strength)
    val_ds = KhmerImgDataset(ds, val_idx, text_col, img_col, c2i,
                              args.img_h, args.img_w, augment=False, aug_strength=0.0)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate,
        num_workers=0,
    )

    model = maybe_compile(model, device, args.compile)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Train ----
    metrics_rows = []
    samples_per_sec_history = []
    best_val_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        avg_loss, epoch_sec, samples_per_sec, ms_per_step = train_one_epoch(
            model, train_loader, optim, ctc_loss, precision, device,
            args.max_steps, epoch, args.epochs,
        )
        val_cer = run_validation(model, val_loader, i2c, device)
        samples_per_sec_history.append(samples_per_sec)

        throttle_flag = detect_throttling(samples_per_sec_history)
        temps = read_temps()

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | val_cer={val_cer:.4f} | "
              f"epoch_sec={epoch_sec:.2f} | samples/sec={samples_per_sec:.1f} | ms/step={ms_per_step:.1f}")
        if throttle_flag:
            print(f"  [throttle-proxy] {throttle_flag}")
        if temps:
            print(f"  [sensors] {temps}")
        elif epoch == 1:
            print("  [sensors] psutil.sensors_temperatures() empty/unavailable on this OS "
                  "(expected on macOS) — true GPU temp needs `sudo powermetrics --samplers smc`.")

        metrics_rows.append({
            "epoch": epoch, "train_loss": avg_loss, "val_cer": val_cer,
            "epoch_sec": epoch_sec, "samples_per_sec": samples_per_sec, "ms_per_step": ms_per_step,
        })

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save({
                "model_state": model.state_dict(),
                "vocab": {"c2i": c2i, "i2c": i2c, "vocab_size": vocab_size},
                "args": vars(args),
                "epoch": epoch,
                "val_cer": val_cer,
            }, out_dir / "best.pt")

    # ---- Metrics CSV (so GRU-vs-LSTM / batch-size sweeps are comparable) ----
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_cer", "epoch_sec", "samples_per_sec", "ms_per_step"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    # ---- Benchmark summary (the primary deliverable) ----
    steady_state = samples_per_sec_history[1:] if len(samples_per_sec_history) > 1 else samples_per_sec_history
    summary = {
        "device": device,
        "precision": "amp-bf16" if precision.use_amp else "fp32",
        "ctc_loss_device": "cpu-fallback (MPS unsupported)" if device == "mps" else device,
        "backbone": args.backbone,
        "rnn": args.rnn,
        "batch_size": args.batch_size,
        "epochs_run": len(metrics_rows),
        "avg_epoch_sec": sum(r["epoch_sec"] for r in metrics_rows) / len(metrics_rows),
        "avg_samples_per_sec_steady_state": sum(steady_state) / len(steady_state) if steady_state else 0.0,
        "best_val_cer": best_val_cer,
    }
    print("\n=== Benchmark summary ===")
    for k, v in summary.items():
        print(f"  {k:>32}: {v}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Metrics: {csv_path}  Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
