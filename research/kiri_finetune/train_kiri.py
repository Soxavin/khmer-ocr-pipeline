"""Fine-tune the vendored Kiri CTC recognizer on the assembled trainset (Track B).

Starts from the pinned HF checkpoint, trains the full model with CTC loss on
(cell-image, text) pairs, evaluates per error class each epoch (riel units,
decimal percents, empty rejection, plus overall CER), and saves the best
checkpoint as <out>/model.safetensors + vocab.json — directly loadable via
KHMER_KIRI_WEIGHTS=<out>.

Usage (repo root; MPS fallback is set automatically):
    uv run python experiments/kiri_finetune/train_kiri.py \
        --data experiments/kiri_finetune/trainset --out experiments/kiri_finetune/run1 \
        [--epochs 8] [--batch 32] [--lr 1e-4] [--limit N]
"""

from __future__ import annotations

import os

# CTC loss is unimplemented on Apple-Silicon MPS; must be set BEFORE importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from khmer_pipeline.engines.kiri_vendor import loader as kiri_loader
from khmer_pipeline.engines.kiri_vendor.model import preprocess_pil

_PERCENT_RE = re.compile(r"^[+-]?\d+\.\d+%$")


def _bucket(text: str) -> str:
    """Error-class bucket for per-class eval (matches the measured taxonomy)."""
    if not text:
        return "empty"
    if "៛" in text:
        return "riel"
    if _PERCENT_RE.match(text):
        return "decimal_percent"
    if "." in text and any(c.isdigit() for c in text):
        return "decimal_number"
    return "other"


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


class PairDataset(Dataset):
    def __init__(self, jsonl: Path, cfg, tokenizer, limit: int | None = None):
        self.rows = [json.loads(l) for l in jsonl.read_text().splitlines()][:limit]
        self.cfg, self.tok = cfg, tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(_REPO / r["image"])
        tensor = preprocess_pil(self.cfg, img)[0]  # (1, H, W)
        ids = [self.tok.token_to_id.get(ch, self.tok.unk_id) + self.tok.ctc_offset
               for ch in r["text"]]
        return tensor, torch.tensor(ids, dtype=torch.long), r["text"]


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch]) if any(len(b[1]) for b in batch) \
        else torch.zeros(0, dtype=torch.long)
    target_lens = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    texts = [b[2] for b in batch]
    return imgs, targets, target_lens, texts


@torch.inference_mode()
def evaluate(model, tokenizer, loader_, device) -> dict:
    """Val pass: overall CER + per-bucket exact-match accuracy and CER."""
    model.eval()
    stats: dict[str, list] = {}
    for imgs, _, _, texts in loader_:
        mem = model.encode(imgs.to(device))
        preds = model.ctc_head(mem).argmax(dim=-1)
        for row_ids, ref in zip(preds.tolist(), texts):
            hyp = tokenizer.decode_ctc(row_ids).strip()
            b = _bucket(ref)
            stats.setdefault(b, []).append((ref == hyp, _cer(ref, hyp)))
    report = {}
    all_pairs = [p for v in stats.values() for p in v]
    report["overall"] = {"n": len(all_pairs),
                         "acc": sum(a for a, _ in all_pairs) / max(1, len(all_pairs)),
                         "cer": sum(c for _, c in all_pairs) / max(1, len(all_pairs))}
    for b, pairs in sorted(stats.items()):
        report[b] = {"n": len(pairs), "acc": sum(a for a, _ in pairs) / len(pairs),
                     "cer": sum(c for _, c in pairs) / len(pairs)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Kiri CTC on the trainset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None, help="cap samples (smoke test)")
    parser.add_argument("--eval-only", action="store_true", help="baseline eval, no training")
    args = parser.parse_args()
    data = args.data if args.data.is_absolute() else _REPO / args.data
    out = args.out if args.out.is_absolute() else _REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, cfg, tokenizer = kiri_loader.load_kiri_model(device=device, verbose=True)
    # vocab must ship next to the fine-tuned weights for the KHMER_KIRI_WEIGHTS loader
    base_path = kiri_loader._download_from_hf(kiri_loader._HF_REPO)
    shutil.copy2(kiri_loader._find_vocab(base_path), out / "vocab.json")

    train_ds = PairDataset(data / "labels_train.jsonl", cfg, tokenizer, args.limit)
    val_ds = PairDataset(data / "labels_val.jsonl", cfg, tokenizer,
                         args.limit and max(200, args.limit // 8))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=_collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, collate_fn=_collate)
    print(f"train {len(train_ds)} / val {len(val_ds)} on {device}")

    baseline = evaluate(model, tokenizer, val_dl, device)
    print("BASELINE:", json.dumps(baseline, indent=2))
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    if args.eval_only:
        return

    ctc = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    best_cer = baseline["overall"]["cer"]

    from safetensors.torch import save_file

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.perf_counter(), 0.0
        for step, (imgs, targets, target_lens, _) in enumerate(train_dl, 1):
            mem = model.encode(imgs.to(device))
            log_probs = model.ctc_head(mem).log_softmax(-1).permute(1, 0, 2)  # (T,B,C)
            input_lens = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, input_lens, target_lens)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"  e{epoch} s{step}/{len(train_dl)} loss {running / step:.4f}")
        sched.step()

        report = evaluate(model, tokenizer, val_dl, device)
        print(f"epoch {epoch} ({time.perf_counter() - t0:.0f}s) "
              f"loss {running / max(1, len(train_dl)):.4f} "
              f"val CER {report['overall']['cer']:.4f}", json.dumps(report))
        (out / f"epoch_{epoch}.json").write_text(json.dumps(report, indent=2))
        if report["overall"]["cer"] < best_cer:
            best_cer = report["overall"]["cer"]
            save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                      str(out / "model.safetensors"))
            print(f"  ✓ new best (CER {best_cer:.4f}) → {out / 'model.safetensors'}")

    print(f"Done. Best val CER {best_cer:.4f}. "
          f"Use with: KHMER_KIRI_WEIGHTS={out} OCR_ENGINE=surya_kiri")


if __name__ == "__main__":
    main()
