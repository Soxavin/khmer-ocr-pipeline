"""Vendored Kiri OCR recognizer subset — model architecture, preprocessing, and CTC decode.
Keeps only what the surya_kiri hybrid engine needs; detector code excluded to avoid the
onnxruntime-gpu dependency that has no macOS ARM wheels."""
