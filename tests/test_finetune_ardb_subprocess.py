"""Tests for the shared isolated-subprocess runner. Mocks subprocess.Popen and
os.getpgid/os.killpg — never actually shells out (no real uv/model download in
unit tests) and never sends real signals."""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from khmer_pipeline.engines.finetune_ardb.subprocess_runner import (
    FineTuneConfig,
    InferenceCancelled,
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


class FakeProc:
    """Stand-in for subprocess.Popen. `timeouts_before_done` controls how many
    times communicate() raises TimeoutExpired before returning; `hangs` makes it
    raise forever (simulating a still-running process) so tests can drive
    cancellation/timeout without a real subprocess or real sleeping."""

    def __init__(self, cmd, returncode=0, stdout="[]", stderr="", timeouts_before_done=0, hangs=False):
        self.pid = 4242
        self.args = cmd
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._timeouts_before_done = timeouts_before_done
        self._hangs = hangs
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.communicate_calls = 0

    def communicate(self, timeout=None):
        self.communicate_calls += 1
        if self._hangs or self._timeouts_before_done > 0:
            if not self._hangs:
                self._timeouts_before_done -= 1
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.returncode = self._returncode
        return self._stdout, self._stderr

    def wait(self, timeout=None):
        # Called only after the group has been signaled; simulate a clean exit.
        self.returncode = self._returncode if self._returncode is not None else -15
        return self.returncode


def _patch_process_control(monkeypatch):
    """Kill-related OS calls never need to touch a real process in these tests."""
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
    return killpg_calls


def test_invokes_uv_run_isolated_no_project_with_pins(monkeypatch):
    _patch_process_control(monkeypatch)
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc(cmd, stdout='[{"label":"Text","text":"ok","box_2d":[0,0,1,1]}]')

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = run_isolated_inference(_img(), _config())
    cmd = captured["cmd"]
    assert cmd[:4] == ["uv", "run", "--isolated", "--no-project"]
    assert "--with" in cmd and "transformers>=5.5,<6" in cmd
    assert "--with" in cmd and "peft>=0.13,<1" in cmd
    assert "src/khmer_pipeline/engines/finetune_ardb/gemma_ardb_infer.py" in cmd
    assert "ok" in out


def test_subprocess_env_drops_the_parents_venv_pointers(monkeypatch):
    """Without this the child inherits VIRTUAL_ENV/PYTHONPATH and can put the
    project venv's site-packages back on sys.path — the exact leak that mixed two
    torch builds and killed every Qwen run with 'operator torchvision::nms does
    not exist'."""
    _patch_process_control(monkeypatch)
    monkeypatch.setenv("VIRTUAL_ENV", "/some/project/.venv")
    monkeypatch.setenv("PYTHONPATH", "/some/project/src")
    monkeypatch.setenv("PATH", "/usr/bin")  # unrelated vars must still pass through
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return FakeProc(cmd, stdout="[]")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    run_isolated_inference(_img(), _config())
    env = captured["env"]
    assert env is not None, "must pass an explicit env, not inherit the parent's"
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env.get("PATH") == "/usr/bin"


def test_passes_adapter_repo_id_as_argument(monkeypatch):
    _patch_process_control(monkeypatch)
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc(cmd)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    run_isolated_inference(_img(), _config(adapter_repo_id="Soxavin/qwen35-ardb-lora-v5-e3-ga2"))
    assert "Soxavin/qwen35-ardb-lora-v5-e3-ga2" in captured["cmd"]


def test_nonzero_exit_raises_runtimeerror(monkeypatch):
    _patch_process_control(monkeypatch)
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda cmd, **kwargs: FakeProc(cmd, returncode=1, stdout="", stderr="ModuleNotFoundError: peft"),
    )
    with pytest.raises(RuntimeError, match="peft"):
        run_isolated_inference(_img(), _config())


def test_timeout_raises_runtimeerror_with_configured_value(monkeypatch):
    _patch_process_control(monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: FakeProc(cmd, hangs=True))
    # Jump straight past any timeout after the first poll — no real waiting.
    times = iter([0, 10_000])
    monkeypatch.setattr(
        "khmer_pipeline.engines.finetune_ardb.subprocess_runner.time.monotonic",
        lambda: next(times, 10_000),
    )
    with pytest.raises(RuntimeError, match="timed out after 1200s"):
        run_isolated_inference(_img(), _config(timeout_s=1200))


def test_timeout_kills_the_process_group(monkeypatch):
    killpg_calls = _patch_process_control(monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: FakeProc(cmd, hangs=True))
    times = iter([0, 10_000])
    monkeypatch.setattr(
        "khmer_pipeline.engines.finetune_ardb.subprocess_runner.time.monotonic",
        lambda: next(times, 10_000),
    )
    with pytest.raises(RuntimeError):
        run_isolated_inference(_img(), _config(timeout_s=600))
    assert killpg_calls  # at least the SIGTERM


def test_cancellation_kills_process_and_raises_inference_cancelled(monkeypatch):
    killpg_calls = _patch_process_control(monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: FakeProc(cmd, hangs=True))
    with pytest.raises(InferenceCancelled):
        run_isolated_inference(_img(), _config(), is_cancelled=lambda: True)
    assert killpg_calls


def test_cancellation_not_triggered_when_process_finishes_first(monkeypatch):
    _patch_process_control(monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: FakeProc(cmd, stdout="[]"))
    calls = {"n": 0}

    def is_cancelled():
        calls["n"] += 1
        return True  # would cancel if it were ever consulted after completion

    out = run_isolated_inference(_img(), _config(), is_cancelled=is_cancelled)
    assert out == "[]"


def test_kill_process_group_escalates_to_sigkill_when_sigterm_does_not_stop_it(monkeypatch):
    killpg_calls = []
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    class StubbornProc:
        pid = 99
        args = ["uv"]

        def __init__(self):
            self._term_attempts = 0

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(self.args, timeout)

        def wait(self, timeout=None):
            self._term_attempts += 1
            if self._term_attempts == 1:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: StubbornProc())
    with pytest.raises(InferenceCancelled):
        run_isolated_inference(_img(), _config(), is_cancelled=lambda: True)
    signals_sent = [sig for _, sig in killpg_calls]
    assert signals_sent == [15, 9]  # SIGTERM then SIGKILL
