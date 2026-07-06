"""Vendored Kiri OCR model architecture — CNN stem + Transformer encoder + CTC head.

Extracted from kiri_ocr/model.py (Apache-2.0, mrrtmob/kiri-ocr v0.2.15).
Kept minimal: only the CTC (fast) decode path is wired; the attention decoder,
beam search, and detector are excluded to avoid onnxruntime-gpu.

When the HF checkpoint provides different dims (e.g. dim=384), CFG is patched
at load time so the architecture matches the weights.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CFG:
    """Model hyperparameters. Defaults match the mrrtmob/kiri-ocr dim-384 checkpoint;
    patched at load time if the safetensors metadata says otherwise."""
    IMG_H: int = 48
    IMG_W: int = 640
    MAX_DEC_LEN: int = 512
    UNK_TOKEN: str = "<unk>"
    COLLAPSE_WHITESPACE: bool = True
    UNICODE_NFC: bool = True

    # Defaults match the mrrtmob/kiri-ocr checkpoint (snapshot 3a381987, dim-384).
    # NOTE: the repo's config.json is STALE (describes an older dim-256/4-layer
    # variant) and uses a non-CFG schema, so it must NOT be trusted — the loader
    # infers depth/dims from the safetensors weights instead. See loader.py.
    # NOTE: heads = dim // 64 (head_dim 64) → 6 for dim-384. config.json says 8,
    # but that value is stale (like its dim/layers); the checkpoint was trained
    # with 6 heads, which is what the upstream loader infers and what decodes
    # correctly. The loader re-infers all of these from the weights.
    ENC_DIM: int = 384
    ENC_LAYERS: int = 6
    ENC_HEADS: int = 6
    ENC_FF: int = 1536
    DROPOUT: float = 0.15

    USE_DECODER: bool = True
    DEC_DIM: int = 384
    DEC_LAYERS: int = 4
    DEC_HEADS: int = 6
    DEC_FF: int = 1536

    USE_CTC: bool = True
    USE_LM: bool = True
    USE_LM_FUSION_EVAL: bool = True
    LM_FUSION_ALPHA: float = 0.35
    USE_FP16: bool = True

    MEM_MAX_LEN_RATIO: float = 1.5
    DEC_MAX_LEN_RATIO: float = 2.5
    DEC_MAX_LEN_PAD: int = 10
    BEAM_WIDTH: int = 5
    BEAM_LEN_PENALTY: float = 0.0


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (1D – decoder, 2D – encoder)
# ---------------------------------------------------------------------------

class SinusoidalPosEnc1D(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class PosEnc2D(nn.Module):
    """2D sinusoidal positional encoding — copied verbatim from kiri_ocr.model
    (Apache-2.0). The exact channel-interleave / permute layout is load-bearing:
    a paraphrased version silently produces a different encoding and garbage OCR."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def _make_pe(self, length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(length, dtype=dtype, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=dtype, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros((length, dim), dtype=dtype, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        num_feats = c // 2
        if num_feats == 0:
            return x
        pe_y = self._make_pe(h, num_feats, x.device, x.dtype)
        pe_x = self._make_pe(w, num_feats, x.device, x.dtype)
        pe_y = pe_y.unsqueeze(2).repeat(1, 1, w)
        pe_x = pe_x.transpose(0, 1).unsqueeze(0).repeat(h, 1, 1)
        pe = torch.cat([pe_y, pe_x], dim=1)
        pe = pe.permute(1, 0, 2)
        if pe.size(0) < c:
            pad = torch.zeros((c - pe.size(0), h, w), device=x.device, dtype=x.dtype)
            pe = torch.cat([pe, pad], dim=0)
        return x + pe.unsqueeze(0)


# ---------------------------------------------------------------------------
# CNN stem
# ---------------------------------------------------------------------------

class ConvStem(nn.Module):
    """CNN stem — copied verbatim from kiri_ocr.model (Apache-2.0).

    Load-bearing details a reconstruction gets wrong: the activation is SiLU
    (not GELU), the strides are mixed — (1,1), (2,2), (2,2), (2,1) — so width is
    downsampled less than height, the channel progression is 1→48→96→160→dim, the
    module is named ``net`` (matching ``stem.net.*`` keys), and a trailing Dropout2d
    keeps the Sequential indices aligned with the checkpoint (BN at 1/4/7/10)."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, (2, 2), 1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 160, 3, (2, 2), 1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, dim, 3, (2, 1), 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Image preprocessing (PIL → tensor)
# ---------------------------------------------------------------------------

class ResizeKeepRatioPadNoCrop:
    """Copied verbatim from kiri_ocr.model (Apache-2.0). Scales by HEIGHT only
    (not min-ratio), resizes with BILINEAR, crops overflow width, and left-pads
    short lines with gray 128 — all load-bearing for matching training-time input."""

    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def __call__(self, img: Image.Image) -> Image.Image:
        iw, ih = img.size
        if ih <= 0 or iw <= 0:
            return img.resize((self.w, self.h), Image.BILINEAR)

        scale = self.h / float(ih)
        nw = max(1, int(round(iw * scale)))
        img = img.resize((nw, self.h), Image.BILINEAR)

        if nw >= self.w:
            return img.crop((0, 0, self.w, self.h))

        # Left-aligned padding with gray (128)
        new_img = Image.new("L", (self.w, self.h), 128)
        new_img.paste(img, (0, 0))
        return new_img


def preprocess_pil(cfg: CFG, pil: Image.Image) -> torch.Tensor:
    """Convert a PIL image into the (1, 1, H, W) tensor KiriOCR expects."""
    img = pil.convert("L")
    img = ResizeKeepRatioPadNoCrop(cfg.IMG_H, cfg.IMG_W)(img)
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    return img_tensor.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Character tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, vocab_path: str, cfg: CFG):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_raw: dict[str, int] = json.load(f)

        if cfg.UNK_TOKEN not in vocab_raw:
            vocab_raw[cfg.UNK_TOKEN] = max(vocab_raw.values(), default=-1) + 1

        items = sorted(vocab_raw.items(), key=lambda kv: kv[1])
        self.token_to_id = {tok: i for i, (tok, _) in enumerate(items)}
        self.id_to_token = {i: tok for i, (tok, _) in enumerate(items)}

        self.unk_token = cfg.UNK_TOKEN
        self.unk_id = self.token_to_id[cfg.UNK_TOKEN]
        self.blank_id = 0
        self.pad_id = 1
        self.ctc_offset = 2
        self.vocab_size = len(self.token_to_id)
        self.ctc_classes = self.vocab_size + self.ctc_offset

        self.dec_pad = 0
        self.dec_bos = 1
        self.dec_eos = 2
        self.dec_offset = 3
        self.dec_vocab = self.vocab_size + self.dec_offset

    def decode_ctc(self, ids: list[int]) -> str:
        """CTC greedy decode: collapse repeats, drop blanks."""
        result: list[str] = []
        prev = self.blank_id
        for idx in ids:
            if idx != self.blank_id and idx != prev:
                token_id = idx - self.ctc_offset
                if 0 <= token_id < self.vocab_size:
                    result.append(self.id_to_token[token_id])
                else:
                    result.append(self.unk_token)
            prev = idx
        return "".join(result)

    def dec_to_ctc_id(self, dec_id: int) -> int:
        if dec_id in (self.dec_pad, self.dec_bos, self.dec_eos):
            return self.blank_id
        raw_id = dec_id - self.dec_offset
        if 0 <= raw_id < self.vocab_size:
            return raw_id + self.ctc_offset
        return self.unk_id + self.ctc_offset


# ---------------------------------------------------------------------------
# Main OCR model (CNN + Transformer encoder + CTC head only)
# ---------------------------------------------------------------------------

class KiriOCR(nn.Module):
    """Vision Transformer OCR model. Only the CTC path is used by the hybrid engine
    (decode_method='fast'); the attention decoder is initialised but never called."""

    def __init__(self, cfg: CFG, tok: CharTokenizer, use_dec_pos_enc: bool = True):
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        self.use_dec_pos_enc = use_dec_pos_enc
        d = cfg.DROPOUT

        self.stem = ConvStem(cfg.ENC_DIM, d)
        self.pos2d = PosEnc2D(cfg.ENC_DIM)

        self.enc_ln_in = nn.LayerNorm(cfg.ENC_DIM)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.ENC_DIM, nhead=cfg.ENC_HEADS,
            dim_feedforward=cfg.ENC_FF, dropout=d,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.ENC_LAYERS, enable_nested_tensor=False)
        self.enc_ln = nn.LayerNorm(cfg.ENC_DIM)

        if cfg.USE_CTC:
            self.ctc_head = nn.Sequential(
                nn.LayerNorm(cfg.ENC_DIM),
                nn.Dropout(d),
                nn.Linear(cfg.ENC_DIM, tok.ctc_classes),
            )

        # Decoder components (kept so state_dict keys match; never called in fast path)
        self.mem_proj = nn.Linear(cfg.ENC_DIM, cfg.DEC_DIM, bias=False)
        self.dec_emb = nn.Embedding(tok.dec_vocab, cfg.DEC_DIM)
        if use_dec_pos_enc:
            self.dec_pos_enc = SinusoidalPosEnc1D(dim=cfg.DEC_DIM, max_len=cfg.MAX_DEC_LEN + 10, dropout=d)
        else:
            self.dec_pos_enc = None

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.DEC_DIM, nhead=cfg.DEC_HEADS,
            dim_feedforward=cfg.DEC_FF, dropout=d,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=cfg.DEC_LAYERS)
        self.dec_ln = nn.LayerNorm(cfg.DEC_DIM)
        self.dec_head = nn.Linear(cfg.DEC_DIM, tok.dec_vocab)

        if cfg.USE_LM:
            self.lm_head = nn.Linear(cfg.DEC_DIM, tok.dec_vocab)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """CNN stem + 2D positional encoding → Transformer encoder → LayerNorm."""
        x = self.stem(imgs)
        x = self.pos2d(x)
        x = F.adaptive_avg_pool2d(x, (1, x.size(-1)))
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.enc_ln_in(x)
        x = self.enc(x)
        x = self.enc_ln(x)
        return x


# ---------------------------------------------------------------------------
# CTC greedy decode (decode_method="fast")
# ---------------------------------------------------------------------------

@torch.inference_mode()
def greedy_ctc_decode(model: KiriOCR, image_tensor: torch.Tensor,
                      tokenizer: CharTokenizer, cfg: CFG) -> Tuple[str, float]:
    """Pure CTC greedy decode — no attention decoder, no beam search."""
    mem = model.encode(image_tensor)
    ctc_logits = model.ctc_head(mem)  # (1, T, vocab)
    pred_ids = ctc_logits.argmax(dim=-1)[0].tolist()
    text = tokenizer.decode_ctc(pred_ids)
    if cfg.COLLAPSE_WHITESPACE:
        text = " ".join(text.split())
    # Confidence: mean of max softmax probabilities over non-blank positions
    probs = F.softmax(ctc_logits, dim=-1)
    max_probs = probs.max(dim=-1).values[0]
    non_blank = [p for i, p in enumerate(max_probs.tolist()) if pred_ids[i] != tokenizer.blank_id]
    conf = float(sum(non_blank) / len(non_blank)) if non_blank else 0.0
    return text, conf
