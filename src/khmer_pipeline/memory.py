# src/khmer_pipeline/memory.py
from __future__ import annotations
import gc
import logging

logger = logging.getLogger(__name__)

def clear_device_cache() -> None:
    """
    Explicitly clears CPU, MPS (Apple Silicon), and MLX memory caches.
    Crucial for preventing Out-Of-Memory (OOM) errors on 24GB unified 
    memory Macs during heavy multi-stage ML inference.
    """
    # 1. Standard Python garbage collection
    gc.collect()
    
    # 2. Clear PyTorch MPS cache (used by Surya OCR)
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass # PyTorch might not be installed or MPS not available
        
    # 3. Clear MLX cache (used by Qwen VLM fallback)
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass # MLX might not be installed