"""Track A: fine-tune DocLayout-YOLO (DocStructBench-pretrained) on our corrected labels.

Starting point is the document-layout-pretrained checkpoint — the §2.24 off-the-shelf
loser, now fine-tuned on OUR documents per the mentor directive. The doclayout-yolo
package is a research fork, so run ISOLATED from the project venv (mlx-vlm precedent):

    uv run --no-project --with doclayout-yolo --with huggingface-hub \
        python experiments/layout_yolo/finetune_doclayout.py \
        [--dataset eval/datasets/layout_v1_corrected] [--epochs 100] [--device mps]

Outputs land in experiments/layout_yolo/runs/ (gitignored); best weights printed at end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_HF_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
_HF_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DocLayout-YOLO on the layout dataset.")
    parser.add_argument("--dataset", type=Path,
                        default=_REPO / "eval/datasets/layout_v1_corrected")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)  # pretrained resolution
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else _REPO / args.dataset

    # Roboflow's data.yaml paths are relative to itself; rewrite with absolute paths.
    names = next(l for l in (dataset / "data.yaml").read_text().splitlines()
                 if l.startswith("names:"))
    local_yaml = _REPO / "experiments/layout_yolo/data_local.yaml"
    local_yaml.write_text(
        f"train: {dataset / 'train/images'}\n"
        f"val: {dataset / 'valid/images'}\n"
        f"test: {dataset / 'test/images'}\n"
        f"nc: 5\n{names}\n")

    from huggingface_hub import hf_hub_download
    from doclayout_yolo import YOLOv10

    ckpt = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
    print(f"pretrained checkpoint: {ckpt}")
    model = YOLOv10(ckpt)
    # nc mismatch (10 DocStructBench classes → our 5) is handled by the trainer:
    # the detection head is rebuilt from data.yaml, the backbone transfers.
    model.train(data=str(local_yaml), epochs=args.epochs, imgsz=args.imgsz,
                patience=20, seed=0, device=args.device, workers=2, batch=8,
                project=str(_REPO / "experiments/layout_yolo/runs"), name="doclayout_ft")
    metrics = model.val(data=str(local_yaml), split="test", device=args.device)
    print("\n--- test-split metrics ---")
    print(f"mAP50 (all classes): {metrics.box.map50:.3f}")
    for i, name in metrics.names.items():
        if i < len(metrics.box.ap50):
            print(f"  {name}: mAP50={metrics.box.ap50[i]:.3f}")
    print(f"best weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
