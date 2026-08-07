"""Combine already-packaged single-config HF dataset folders into one multi-config repo
folder (data/<config>/<split>-*.parquet per config), so related datasets with different
schemas can share one Hugging Face repo/card instead of living in separate repos.

Usage:
    uv run python scripts/package_multiconfig_dataset.py \
        --config transcription eval/datasets/ardb_table_sft_v1_hf \
        --config layout eval/datasets/ardb_layout_detection_sft_v1_hf \
        --out eval/datasets/ardb_gemma_sft_v1_hf
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def combine(configs: list[tuple[str, Path]], out_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, src_hf_dir in configs:
        src_data = src_hf_dir / "data"
        dst_data = out_dir / "data" / name
        dst_data.mkdir(parents=True, exist_ok=True)
        files = sorted(src_data.glob("*.parquet"))
        for f in files:
            shutil.copy2(f, dst_data / f.name)
        counts[name] = len(files)
        print(f"{name}: copied {len(files)} parquet files from {src_hf_dir}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine packaged HF dataset folders into one multi-config repo folder.")
    parser.add_argument("--config", nargs=2, action="append", metavar=("NAME", "HF_DIR"),
                        required=True, dest="configs",
                        help="Repeatable: --config <name> <path-to-packaged-hf-dir>")
    parser.add_argument("--out", type=Path, required=True, help="Combined HF upload folder")
    args = parser.parse_args()
    configs = [(name, Path(hf_dir)) for name, hf_dir in args.configs]
    combine(configs, args.out)


if __name__ == "__main__":
    main()
