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
    subprocess's stderr on a non-zero exit that produced NO stdout — a real
    failure, nothing to salvage. A non-zero exit that DID produce stdout is
    still returned rather than discarded: observed live, the infer scripts can
    finish generating (weights loaded, output already written) and then die
    with SIGABRT (multiprocessing's resource_tracker crashing during interpreter
    shutdown, exit 134) — a cleanup-time crash, not a generation failure.
    Discarding a real generation because Python's own teardown code crashed
    afterward would silently turn a working run into an empty page; the
    caller's JSON-parse fallback already handles a malformed/truncated result
    correctly (parse_regions returning None -> raw text preserved), so handing
    it whatever stdout exists is strictly more informative than raising.
    If `is_cancelled` starts returning True while the subprocess is still
    running, it (and its whole process group — `uv run`'s actual model child,
    not just the wrapper) is killed and InferenceCancelled is raised."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "page.png"
        Image.fromarray(page_image).save(img_path)

        # --isolated is load-bearing, not belt-and-braces: without it `uv run
        # --no-project` REUSES whatever environment it discovers (the project's
        # .venv via an inherited VIRTUAL_ENV, or a conda base) and layers --with
        # packages on top as an overlay. That mixes two torch builds in one
        # process: the overlay's torch plus the project venv's torchvision, which
        # registers C++ ops against the torch it was compiled for and dies with
        # "operator torchvision::nms does not exist" the moment transformers
        # imports it. Qwen hit this on every run (torch==2.10.0 pin, guaranteed
        # mismatch); Gemma only escaped because its unpinned `torch` happened to
        # resolve close enough to the leaked torchvision. --isolated builds a
        # genuinely standalone env, so the pins in extra_pins are the whole
        # dependency picture.
        cmd = ["uv", "run", "--isolated", "--no-project"]
        for pin in config.extra_pins:
            cmd += ["--with", pin]
        cmd += [
            "python", config.infer_script,
            "--base-model", config.base_model_id,
            "--adapter", config.adapter_repo_id,
            "--image", str(img_path),
        ]
        # Scrub the venv pointers too: --isolated governs what uv builds, but a
        # leftover VIRTUAL_ENV/PYTHONPATH can still put the parent's site-packages
        # on the child's sys.path once python starts.
        env = {k: v for k, v in os.environ.items()
               if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
        # New session -> own process group, so killing it also kills `uv run`'s
        # child interpreter (the actual model process), not just the wrapper.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, env=env,
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

        if proc.returncode != 0 and not stdout.strip():
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
