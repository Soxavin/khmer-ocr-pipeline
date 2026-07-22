"""
Model configuration for the Khmer OCR pipeline.
To swap a model, change the constant here and update the corresponding
stage file (surya.py for Stage 3, postprocess.py for Stage 4).

Stage 3 options: "surya-2" (surya-ocr 0.20.x)
Stage 4 options: "qwen2.5-7b-instruct-4bit-mlx"
"""
from __future__ import annotations

# Stage 3 — Layout detection, OCR, table recognition.
# Surya 2 (surya-ocr 0.20.x) is ONE 650M VLM serving layout + OCR + table-rec
# through a shared SuryaInferenceManager — not the separate layout/recognition
# checkpoints of 0.17.x. Kept as a single constant so this can't drift back into
# describing an architecture we no longer run.
STAGE3_MODEL: str = "surya-2"
STAGE3_CHECKPOINT: str = "datalab-to/surya-ocr-2"  # surya.settings.SURYA_MODEL_CHECKPOINT

# Stage 4 — Post-processing correction model
STAGE4_MODEL: str = "qwen2.5-7b-instruct-4bit-mlx"
STAGE4_MODEL_PATH: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# Thresholds — tunable without touching stage logic
ANOMALY_THRESHOLD: float = 0.15
CONFIDENCE_LOW: float = 0.5   # below this = low confidence (red in UI)
CONFIDENCE_MID: float = 0.8   # below this = medium confidence (yellow in UI)

# Per-CELL confidence buckets for the table confidence view (app.py), from the
# §2.33 calibration on real docs: <0.80 ≈ 35% correct (red), 0.80–0.95 ≈ 67%
# (amber), ≥0.95 ≈ 85% (untinted). CELL_CONF_LOW matches surya_kiri's
# _LOW_CONF_THRESHOLD warning cutoff — keep them in sync.
CELL_CONF_LOW: float = 0.80
CELL_CONF_MID: float = 0.95
