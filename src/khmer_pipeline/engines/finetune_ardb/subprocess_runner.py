"""Shared isolated-subprocess runner for the ARDB fine-tune engines.

Gemma 4 needs transformers>=5.5; Qwen3.5's fine-tune notebook needs a matching
newer torch/transformers pin set too — both conflict with Surya's
transformers>=4.56.1,<5.0 pin (pyproject.toml), so neither can run in-process.
Mirrors the isolated-script precedent in scripts/mlx_recognizer.py (which
sidesteps mlx-vlm's own transformers>=5.1 requirement the same way), generalized
so each model supplies its own package pins + infer script via a `FineTuneConfig`
rather than hardcoding one model's requirements here.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

TableTextFormat = Literal["html", "markdown"]

# Cold-start (model download + load) can take minutes on first run; generous but
# bounded so a hung subprocess doesn't hang the whole extraction run forever.
_TIMEOUT_S = 600


@dataclass(frozen=True)
class FineTuneConfig:
    """Everything a per-model infer script + the shared runner need to invoke one
    isolated inference call. One instance per model — see gemma_ardb_engine.py /
    qwen_ardb_engine.py for the concrete configs."""
    base_model_id: str
    adapter_repo_id: str
    infer_script: str                    # path to the per-model script, repo-root-relative
    extra_pins: list[str] = field(default_factory=list)   # e.g. ["transformers>=5.5,<6", "peft>=0.13,<1"]
    table_text_format: TableTextFormat = "html"


def run_isolated_inference(page_image: np.ndarray, config: FineTuneConfig) -> str:
    """Run one page image through the isolated subprocess, returning its raw
    stdout text (the model's raw generation — parsing happens in the caller via
    finetune_ardb.parsing.parse_regions). Raises RuntimeError with the
    subprocess's stderr on a non-zero exit or timeout; never returns garbage
    silently — the engine layer is what fails soft per page, not this runner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "page.png"
        Image.fromarray(page_image).save(img_path)

        cmd = ["uv", "run", "--no-project"]
        for pin in config.extra_pins:
            cmd += ["--with", pin]
        cmd += [
            "python", config.infer_script,
            "--base-model", config.base_model_id,
            "--adapter", config.adapter_repo_id,
            "--image", str(img_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{config.infer_script} timed out after {_TIMEOUT_S}s"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"{config.infer_script} exited {proc.returncode}: {proc.stderr.strip()}"
            )
        return proc.stdout
