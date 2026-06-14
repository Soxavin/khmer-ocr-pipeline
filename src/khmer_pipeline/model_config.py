"""
Model configuration for the Khmer OCR pipeline.
To swap a model, change the constant here and update the corresponding
stage file (surya.py for Stage 3, postprocess.py for Stage 4).

Stage 3 options: "surya-0.17.1"
Stage 4 options: "qwen2.5-7b-instruct-4bit-mlx"
"""

# Stage 3 — Layout detection, OCR, table recognition
STAGE3_MODEL: str = "surya-0.17.1"
STAGE3_CHECKPOINT_LAYOUT: str = "vikp/surya_layout3"  # populated from surya settings
STAGE3_CHECKPOINT_RECOGNITION: str = "vikp/surya_rec2"

# Stage 4 — Post-processing correction model
STAGE4_MODEL: str = "qwen2.5-7b-instruct-4bit-mlx"
STAGE4_MODEL_PATH: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# Thresholds — tunable without touching stage logic
ANOMALY_THRESHOLD: float = 0.15
CONFIDENCE_LOW: float = 0.5   # below this = low confidence (red in UI)
CONFIDENCE_MID: float = 0.8   # below this = medium confidence (yellow in UI)
