from __future__ import annotations

from upsonic.ocr.ocr import OCR, infer_provider
from upsonic.ocr.base import (
    OCRProvider,
    OCRConfig,
    OCRResult,
    OCRMetrics,
    OCRTextBlock,
    BoundingBox,
)
from upsonic.ocr.exceptions import (
    OCRError,
    OCRProviderError,
    OCRFileNotFoundError,
    OCRUnsupportedFormatError,
    OCRProcessingError,
)

try:
    from upsonic.ocr.paddleocr import (
        PaddleOCRConfig,
        PaddleOCRProvider,
        PPStructureV3Provider,
        PPChatOCRv4Provider,
        PaddleOCRVLProvider,
        PaddleOCR,
        PPStructureV3,
        PPChatOCRv4,
        PaddleOCRVL,
    )
    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PADDLEOCR_AVAILABLE = False

__all__ = [
    "OCR",
    "infer_provider",
    "OCRProvider",
    "OCRConfig",
    "OCRResult",
    "OCRMetrics",
    "OCRTextBlock",
    "BoundingBox",
    "OCRError",
    "OCRProviderError",
    "OCRFileNotFoundError",
    "OCRUnsupportedFormatError",
    "OCRProcessingError",
]

if _PADDLEOCR_AVAILABLE:
    __all__.extend([
        "PaddleOCRConfig",
        "PaddleOCRProvider",
        "PPStructureV3Provider",
        "PPChatOCRv4Provider",
        "PaddleOCRVLProvider",
        "PaddleOCR",
        "PPStructureV3",
        "PPChatOCRv4",
        "PaddleOCRVL",
    ])

