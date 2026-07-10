"""HF download + safetensors loading for the vendored Kiri recognizer.

Adapted from kiri_ocr/core.py (Apache-2.0, mrrtmob/kiri-ocr). Stripped of
detector logic and onnxruntime references.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch

from .model import CFG, CharTokenizer, KiriOCR

_HF_REPO = "mrrtmob/kiri-ocr"
# Local fine-tuned weights override (Track B): a model.safetensors file or a folder
# holding model.safetensors + vocab.json. Unset → pinned HF snapshot as always.
_LOCAL_WEIGHTS_ENV = "KHMER_KIRI_WEIGHTS"
# Pin the known-good snapshot (dim-384 checkpoint, PROJECT_LOG §2.30). Without a
# pinned revision, an upstream re-push of model.safetensors would silently change
# production OCR output on a fresh machine / cleared cache. Model AND vocab are
# fetched from the SAME snapshot so `_find_vocab`'s "vocab sits beside the model"
# assumption holds.
_HF_REVISION = "3a3819874ad67a3a9624d5d994c46649060d7dc9"
_MODEL_CACHE: dict[tuple, dict] = {}  # (model_path, device) → {model, cfg, tokenizer}


def _download_from_hf(repo_id: str, verbose: bool = False) -> str:
    """Download model.safetensors + vocab.json from HuggingFace Hub.
    Returns the local path to the safetensors file.

    Deliberately does NOT fetch config.json / model_meta.json: they are stale for
    this checkpoint (describe an older dim-256 variant) and we always infer the
    architecture from the weights — fetching them only risks other tools picking
    up the wrong config from the shared HF cache."""
    from huggingface_hub import hf_hub_download

    if verbose:
        print(f"[Kiri] Downloading from HuggingFace: {repo_id}")

    for filename in ["vocab.json", "vocab_auto.json"]:
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, revision=_HF_REVISION)
        except Exception:
            pass

    # Model weights (prefer safetensors)
    for model_name in ["model.safetensors", "model.pt"]:
        try:
            return hf_hub_download(repo_id=repo_id, filename=model_name, revision=_HF_REVISION)
        except Exception:
            pass

    raise FileNotFoundError(f"[Kiri] No model file found in HF repo {repo_id}")


def _local_weights_path() -> Optional[Path]:
    """Resolve KHMER_KIRI_WEIGHTS to a safetensors file, or None when unset.
    A directory resolves to <dir>/model.safetensors; missing paths fail loudly
    (a fine-tune swap must never silently fall back to the stock model)."""
    raw = os.environ.get(_LOCAL_WEIGHTS_ENV)
    if not raw:
        return None
    path = Path(raw)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(
            f"[Kiri] {_LOCAL_WEIGHTS_ENV}={raw!r} but no weights file at {path}")
    return path


def _find_vocab(model_dir: str, verbose: bool = False) -> Optional[str]:
    """Locate vocab.json near the model checkpoint (HF cache or local)."""
    candidates = [
        Path(model_dir).parent / "vocab.json",
        Path(model_dir).parent / "vocab_auto.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    if verbose:
        print(f"[Kiri] ⚠️ No vocab.json found near {model_dir}")
    return None


def _infer_config(state_dict: dict, cfg: CFG) -> None:
    """Patch CFG dims/depth from safetensors weight shapes so the architecture
    matches the checkpoint exactly.

    The weights are the only trustworthy source: the repo's config.json is stale
    (a different dim-256 variant) and uses a non-CFG schema. Head counts are NOT
    guessed by divisibility (384 divides by 16, but the model was trained with 8)
    — they stay at the CFG default unless a reliable value is available.
    """
    import re

    def _shape(key):
        t = state_dict.get(key)
        return tuple(t.shape) if (t is not None and hasattr(t, "shape")) else None

    # --- embedding dims (reliable: layer-norm weight shape) ---
    if (s := _shape("enc_ln.weight")):
        cfg.ENC_DIM = s[0]
    if (s := _shape("dec_ln.weight")):
        cfg.DEC_DIM = s[0]

    # --- FF dims (first linear in each MLP block) ---
    if (s := _shape("enc.layers.0.linear1.weight")):
        cfg.ENC_FF = s[0]
    if (s := _shape("dec.layers.0.linear1.weight")):
        cfg.DEC_FF = s[0]

    # --- layer counts (number of distinct enc/dec transformer blocks) ---
    def _count_layers(prefix: str) -> int:
        idxs = {int(m.group(1)) for k in state_dict
                if (m := re.match(rf"{prefix}\.layers\.(\d+)\.", k))}
        return len(idxs)

    if (n := _count_layers("enc")):
        cfg.ENC_LAYERS = n
    if (n := _count_layers("dec")):
        cfg.DEC_LAYERS = n

    # --- attention heads: head_dim is 64, so heads = dim // 64 (matches the
    # upstream loader). Do NOT guess by divisibility — 384 divides by 8/16 too,
    # but the model was trained with 6 heads (config.json's "8" is stale). ---
    def _heads_from_inproj(key: str, dim: int) -> int:
        s = _shape(key)
        total = (s[0] // 3) if s else dim  # in_proj packs [q;k;v] → 3*dim rows
        if total % 64 == 0:
            return total // 64
        if total % 32 == 0:
            return total // 32
        return cfg.ENC_HEADS
    cfg.ENC_HEADS = _heads_from_inproj("enc.layers.0.self_attn.in_proj_weight", cfg.ENC_DIM)
    cfg.DEC_HEADS = _heads_from_inproj("dec.layers.0.self_attn.in_proj_weight", cfg.DEC_DIM)


def load_kiri_model(device: str = "cpu", verbose: bool = False) -> tuple[KiriOCR, CFG, CharTokenizer]:
    """Download (if needed) and load the Kiri recognizer. Results are cached per device."""
    local = _local_weights_path()
    cache_key = (str(local) if local else _HF_REPO, device)
    if cache_key in _MODEL_CACHE:
        if verbose:
            print("[Kiri] ⚡ Reusing cached model")
        cached = _MODEL_CACHE[cache_key]
        return cached["model"], cached["cfg"], cached["tokenizer"]

    if local is not None:
        model_path = str(local)
        if verbose:
            print(f"[Kiri] Using LOCAL fine-tuned weights: {model_path}")
    else:
        try:
            model_path = _download_from_hf(_HF_REPO, verbose=verbose)
        except Exception as e:
            print(f"[Kiri] ❌ Failed to download model from HuggingFace: {e}", file=sys.stderr)
            raise

    vocab_path = _find_vocab(model_path, verbose=verbose)
    if not vocab_path:
        raise FileNotFoundError(f"[Kiri] Could not find vocab.json near {model_path}")

    if verbose:
        print(f"[Kiri] Loading model from {model_path}")

    from safetensors.torch import load_file as safetensors_load

    state_dict = safetensors_load(model_path, device=device)

    # Always infer architecture from the weights. The repo's config.json is stale
    # (describes an older dim-256 variant) and uses a non-CFG schema, so trusting
    # it silently mis-sizes the model and load_state_dict(strict=False) then leaves
    # whole modules randomly initialised → garbage recognition.
    cfg = CFG()
    _infer_config(state_dict, cfg)

    # Check for decoder positional encoding (newer models have it)
    has_dec_pos_enc = any("dec_pos_enc" in k for k in state_dict)

    tokenizer = CharTokenizer(vocab_path, cfg)
    model = KiriOCR(cfg, tokenizer, use_dec_pos_enc=has_dec_pos_enc).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # The CTC decode path (stem → encoder → ctc_head) MUST load fully; any missing
    # key there means a silent architecture mismatch producing random weights.
    ctc_path_missing = [k for k in missing
                        if k.startswith(("stem.", "enc.", "enc_ln", "ctc_head", "pos"))]
    if ctc_path_missing:
        raise RuntimeError(
            f"[Kiri] CTC-path weights failed to load ({len(ctc_path_missing)} keys, "
            f"e.g. {ctc_path_missing[:3]}). Architecture does not match the checkpoint."
        )
    model.eval()

    if verbose:
        print(f"[Kiri] ✓ Loaded (vocab: {tokenizer.vocab_size} chars, enc_dim={cfg.ENC_DIM}, "
              f"enc_layers={cfg.ENC_LAYERS}, heads={cfg.ENC_HEADS})")

    _MODEL_CACHE[cache_key] = {"model": model, "cfg": cfg, "tokenizer": tokenizer}
    return model, cfg, tokenizer
