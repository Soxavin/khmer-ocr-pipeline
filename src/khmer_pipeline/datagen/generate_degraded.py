from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

# Synthetic scan-like degradation of clean (born-digital) page renders, to A/B-test
# the preprocessing stack against EXISTING ground truth (no new labeling). This is a
# proxy: synthetic degradation != real scan artifacts. Applied: contrast reduction
# (exercises normalise/sharpen) -> gaussian blur -> seeded gaussian noise -> small
# rotation (exercises deskew; > _DESKEW_MIN_ANGLE_DEG=0.5). Rotation last so the
# border fill is clean white (documents are white-background).

_DEGRADE_ROTATION_DEG = 2.5
_DEGRADE_BLUR_SIGMA = 0.8
_DEGRADE_NOISE_STD = 12.0
_DEGRADE_CONTRAST = 0.85  # scale toward mid-gray (128)


def degrade_page(img: np.ndarray, seed: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    out = (out - 128.0) * _DEGRADE_CONTRAST + 128.0
    out = cv2.GaussianBlur(out, (0, 0), _DEGRADE_BLUR_SIGMA)
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0.0, _DEGRADE_NOISE_STD, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), _DEGRADE_ROTATION_DEG, 1.0)
    out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scan-like degraded copies of a dataset.")
    parser.add_argument("--real-dir", type=Path, default=Path("eval/datasets/real"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/datasets/real_degraded"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(args.real_dir.glob("*.png"))
    for i, png in enumerate(pngs):
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        deg = degrade_page(img, seed=args.seed + i)
        cv2.imwrite(str(args.output_dir / png.name), cv2.cvtColor(deg, cv2.COLOR_RGB2BGR))
    # ground truth is unchanged by degradation — copy verbatim
    for gt in sorted(args.real_dir.glob("*_ground_truth.json")):
        shutil.copy(gt, args.output_dir / gt.name)
    print(f"Degraded {len(pngs)} image(s) → {args.output_dir}")


if __name__ == "__main__":
    main()
