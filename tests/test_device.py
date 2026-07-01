from __future__ import annotations
import os
from unittest.mock import patch
import pytest


def _reset_configured():
    # Reset the module-level _configured flag between tests so print guard doesn't leak.
    import khmer_pipeline.device as dev_mod
    dev_mod._configured = False


@pytest.fixture(autouse=True)
def reset_device_module():
    _reset_configured()
    yield
    _reset_configured()


# --- detect_device ---

def test_detect_device_cuda():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.backends.mps.is_available", return_value=False):
        from khmer_pipeline.device import detect_device
        assert detect_device() == "cuda"


def test_detect_device_mps():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        from khmer_pipeline.device import detect_device
        assert detect_device() == "mps"


def test_detect_device_cpu():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        from khmer_pipeline.device import detect_device
        assert detect_device() == "cpu"


# --- configure_runtime ---

def test_configure_runtime_sets_torch_device(monkeypatch):
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.delenv("SURYA_INFERENCE_BACKEND", raising=False)
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        from khmer_pipeline.device import configure_runtime
        result = configure_runtime()
    assert result == "cpu"
    assert os.environ.get("TORCH_DEVICE") == "cpu"


def test_configure_runtime_noop_when_surya_backend_set(monkeypatch):
    monkeypatch.setenv("SURYA_INFERENCE_BACKEND", "llamacpp")
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    from khmer_pipeline.device import configure_runtime
    result = configure_runtime()
    # Should not set TORCH_DEVICE and should return the fallback label
    assert os.environ.get("TORCH_DEVICE") is None
    assert result == "metal/llamacpp"


def test_configure_runtime_noop_when_torch_device_already_set(monkeypatch):
    monkeypatch.delenv("SURYA_INFERENCE_BACKEND", raising=False)
    monkeypatch.setenv("TORCH_DEVICE", "cpu")
    from khmer_pipeline.device import configure_runtime
    result = configure_runtime()
    # Should return the pre-set value unchanged
    assert result == "cpu"
    assert os.environ.get("TORCH_DEVICE") == "cpu"


def test_configure_runtime_detects_cuda_when_available(monkeypatch):
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.delenv("SURYA_INFERENCE_BACKEND", raising=False)
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.backends.mps.is_available", return_value=False):
        from khmer_pipeline.device import configure_runtime
        result = configure_runtime()
    assert result == "cuda"
    assert os.environ.get("TORCH_DEVICE") == "cuda"
