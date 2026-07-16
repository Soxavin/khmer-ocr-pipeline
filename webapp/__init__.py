"""NiceGUI review UI for the Khmer OCR pipeline.

Presentation layer only — it imports and calls the same pipeline functions as the
Streamlit `app.py` (ingest → preprocess → OCR → postprocess → export), reusing every
dataclass unchanged. Run with `uv run python -m webapp.main`.
"""
