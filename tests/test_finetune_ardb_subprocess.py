"""Tests for the shared isolated-subprocess runner. Mocks subprocess.run — never
actually shells out (no real uv/model download in unit tests)."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from khmer_pipeline.engines.finetune_ardb.subprocess_runner import (
    FineTuneConfig,
    run_isolated_inference,
)


def _config(**overrides) -> FineTuneConfig:
    defaults = dict(
        base_model_id="unsloth/gemma-4-E2B-it",
        adapter_repo_id="Soxavin/gemma4-e2b-ardb-lora-v5-e3",
        infer_script="src/khmer_pipeline/engines/finetune_ardb/gemma_ardb_infer.py",
        extra_pins=["transformers>=5.5,<6", "peft>=0.13,<1"],
        table_text_format="html",
    )
    defaults.update(overrides)
    return FineTuneConfig(**defaults)


def _img() -> np.ndarray:
    return np.full((10, 10, 3), 255, dtype=np.uint8)


def test_invokes_uv_run_no_project_with_pins(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout='[{"label":"Text","text":"ok","box_2d":[0,0,1,1]}]', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = run_isolated_inference(_img(), _config())
    cmd = captured["cmd"]
    assert cmd[:3] == ["uv", "run", "--no-project"]
    assert "--with" in cmd and "transformers>=5.5,<6" in cmd
    assert "--with" in cmd and "peft>=0.13,<1" in cmd
    assert "src/khmer_pipeline/engines/finetune_ardb/gemma_ardb_infer.py" in cmd
    assert "ok" in out


def test_passes_adapter_repo_id_as_argument(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_isolated_inference(_img(), _config(adapter_repo_id="Soxavin/qwen35-ardb-lora-v5-e3-ga2"))
    assert "Soxavin/qwen35-ardb-lora-v5-e3-ga2" in captured["cmd"]


def test_nonzero_exit_raises_runtimeerror(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ModuleNotFoundError: peft")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="peft"):
        run_isolated_inference(_img(), _config())


def test_timeout_raises_runtimeerror(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out"):
        run_isolated_inference(_img(), _config())
