# src/khmer_pipeline/memory.py
from __future__ import annotations
import gc
import logging

logger = logging.getLogger(__name__)

def clear_device_cache() -> None:
    """
    Explicitly clears CPU, MPS (Apple Silicon), CUDA, and MLX memory caches.
    Crucial for preventing Out-Of-Memory (OOM) errors during heavy multi-stage
    ML inference on both Apple Silicon (24GB unified memory) and NVIDIA GPUs.
    """
    # 1. Standard Python garbage collection
    gc.collect()

    # 2. Surya 0.20+ runs via a C++ llama-server process and manages its own VRAM;
    #    PyTorch MPS cache flush is no longer needed for the OCR stage.

    # 3. Clear CUDA cache (Linux/NVIDIA GPU path)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # 4. Clear MLX cache (used by Qwen VLM fallback, Apple Silicon only)
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass # MLX might not be installed