"""Export a fine-tuned DocLayout-YOLO checkpoint to ONNX for the production pipeline.

Why ONNX and not the .pt: the training fork (`doclayout_yolo`) pickles its own module
paths into the checkpoint, so stock `ultralytics` cannot load it
(ModuleNotFoundError: doclayout_yolo) — and installing the fork into the project venv
is not an option either, since it requires `opencv-python` which collides with the
project's `opencv-python-headless` (both provide `cv2`). ONNX carries no Python class
dependency, so the weights cross the venv boundary cleanly and production needs NO new
dependency: `rapid_layout` + `onnxruntime` are already installed and `layout_detect.py`
already speaks `doclayout_docstructbench`.

rapid_layout reads the class list from a `character` metadata key (its bundled models
carry one; an exported model does not), so this injects it — otherwise loading fails
with KeyError: 'character'.

Run ISOLATED from the project venv, like the trainer:

    uv run --no-project --with doclayout-yolo --with onnx --with onnxslim \
        --with onnxscript --with onnxruntime \
        python experiments/layout_yolo/export_onnx.py --weights <run>/weights/best.pt

Prints the .onnx path to hand to KHMER_LAYOUT_WEIGHTS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _REPO / "eval/datasets/layout_v1_corrected/data.yaml"


def _class_names(data_yaml: Path) -> list[str]:
    """Read the ordered class list out of a YOLO data.yaml `names:` line."""
    line = next(l for l in data_yaml.read_text().splitlines() if l.startswith("names:"))
    inner = line.split("names:", 1)[1].strip().strip("[]")
    return [n.strip().strip("'\"") for n in inner.split(",") if n.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export fine-tuned DocLayout-YOLO to ONNX.")
    ap.add_argument("--weights", type=Path, required=True, help="best.pt from the fine-tune run")
    ap.add_argument("--data", type=Path, default=_DEFAULT_DATA, help="data.yaml (for class names)")
    ap.add_argument("--imgsz", type=int, default=1024, help="must match the training imgsz")
    args = ap.parse_args()

    names = _class_names(args.data)
    print(f"classes ({len(names)}): {names}")

    from doclayout_yolo import YOLOv10

    model = YOLOv10(str(args.weights))
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=13))

    import onnx

    m = onnx.load(str(onnx_path))
    # Drop a pre-existing key first: metadata_props is a repeated field, so a second
    # 'character' entry would shadow rather than replace.
    for i, prop in enumerate(list(m.metadata_props)):
        if prop.key == "character":
            del m.metadata_props[i]
            break
    entry = m.metadata_props.add()
    entry.key = "character"
    entry.value = "\n".join(names)
    onnx.save(m, str(onnx_path))

    print(f"\nexported: {onnx_path}")
    print(f"use it with:  KHMER_LAYOUT_WEIGHTS={onnx_path}")


if __name__ == "__main__":
    main()
