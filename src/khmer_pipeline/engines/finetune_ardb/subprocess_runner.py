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

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
from PIL import Image

TableTextFormat = Literal["html", "markdown"]

# Cold-start (model download + load) can take minutes on first run; generous but
# bounded so a hung subprocess doesn't hang the whole extraction run forever.
# Per-model override via FineTuneConfig.timeout_s (e.g. Qwen's known-slow, may-
# hit-its-token-cap generations need longer than this default).
_DEFAULT_TIMEOUT_S = 600

# How often to check `is_cancelled` / the timeout while the subprocess runs.
_POLL_INTERVAL_S = 0.2
# Grace period between SIGTERM and SIGKILL when tearing down a cancelled run.
_KILL_GRACE_S = 5


class InferenceCancelled(RuntimeError):
    """Raised when `is_cancelled` reports True while the isolated subprocess is
    still running. The subprocess's whole process group is killed *before* this
    is raised, so Stop actually frees the CPU/GPU/memory it was using instead of
    letting it run to completion or timeout in the background."""


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
    timeout_s: int = _DEFAULT_TIMEOUT_S


def run_isolated_inference(
    page_image: np.ndarray,
    config: FineTuneConfig,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    """Run one page image through the isolated subprocess, returning its raw
    stdout text (the model's raw generation — parsing happens in the caller via
    finetune_ardb.parsing.parse_regions). Raises RuntimeError with the
    subprocess's stderr on a non-zero exit or timeout; never returns garbage
    silently — the engine layer is what fails soft per page, not this runner.
    If `is_cancelled` starts returning True while the subprocess is still
    running, it (and its whole process group — `uv run`'s actual model child,
    not just the wrapper) is killed and InferenceCancelled is raised."""
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
        # New session -> own process group, so killing it also kills `uv run`'s
        # child interpreter (the actual model process), not just the wrapper.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        start = time.monotonic()
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=_POLL_INTERVAL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            if is_cancelled is not None and is_cancelled():
                _kill_process_group(proc)
                raise InferenceCancelled(f"{config.infer_script} cancelled by user")
            if time.monotonic() - start > config.timeout_s:
                _kill_process_group(proc)
                raise RuntimeError(f"{config.infer_script} timed out after {config.timeout_s}s")

        if proc.returncode != 0:
            raise RuntimeError(
                f"{config.infer_script} exited {proc.returncode}: {stderr.strip()}"
            )
        return stdout


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM the subprocess's whole process group, escalating to SIGKILL if it
    hasn't exited within the grace period."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=_KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=_KILL_GRACE_S)
