from __future__ import annotations
import os

# OS/GPU-aware device selection. Surya + torch read TORCH_DEVICE from the env;
# we set it once so the pipeline uses CUDA on NVIDIA, MPS on Apple Silicon, else CPU.

_configured = False


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_runtime() -> str:
    # Respect an explicit Metal/llama.cpp setup (setup-metal-macos.sh) or a
    # user-set TORCH_DEVICE; otherwise auto-select and export TORCH_DEVICE.
    global _configured
    if os.environ.get("SURYA_INFERENCE_BACKEND"):
        return os.environ.get("TORCH_DEVICE", "metal/llamacpp")
    dev = os.environ.get("TORCH_DEVICE")
    if not dev:
        dev = detect_device()
        os.environ["TORCH_DEVICE"] = dev
        if not _configured:
            print(f"[device] using {dev}", flush=True)
            _configured = True
    return dev
