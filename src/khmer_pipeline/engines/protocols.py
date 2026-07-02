from __future__ import annotations
from typing import Callable, Optional, Protocol
from ..models import PreprocessResult, SuryaResult, PostprocessResult

class OCREngine(Protocol):
    """Interface for Stage 3: Layout, Text OCR, and Table Recognition."""
    def __call__(
        self,
        result: PreprocessResult,
        on_page: Optional[Callable[[int, int], None]] = None,
    ) -> SuryaResult: ...

class CorrectionEngine(Protocol):
    """Interface for Stage 4: Rule-based + VLM fallback text correction."""
    def __call__(
        self,
        result: SuryaResult,
        skip_qwen: bool = False,
        anomaly_threshold: float = 0.15,
    ) -> PostprocessResult: ...
