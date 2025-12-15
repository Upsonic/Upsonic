from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ocr import OCR, infer_provider
    from .base import (
        OCRProvider,
        OCRConfig,
        OCRResult,
        OCRMetrics,
        OCRTextBlock,
        BoundingBox,
    )
    from .exceptions import (
        OCRError,
        OCRProviderError,
        OCRFileNotFoundError,
        OCRUnsupportedFormatError,
        OCRProcessingError,
    )
    try:
        from .paddleocr import (
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
    except ImportError:
        pass

def _get_base_classes():
    """Lazy import of base OCR classes."""
    from .ocr import OCR, infer_provider
    from .base import (
        OCRProvider,
        OCRConfig,
        OCRResult,
        OCRMetrics,
        OCRTextBlock,
        BoundingBox,
    )
    
    return {
        'OCR': OCR,
        'infer_provider': infer_provider,
        'OCRProvider': OCRProvider,
        'OCRConfig': OCRConfig,
        'OCRResult': OCRResult,
        'OCRMetrics': OCRMetrics,
        'OCRTextBlock': OCRTextBlock,
        'BoundingBox': BoundingBox,
    }

def _get_exception_classes():
    """Lazy import of exception classes."""
    from .exceptions import (
        OCRError,
        OCRProviderError,
        OCRFileNotFoundError,
        OCRUnsupportedFormatError,
        OCRProcessingError,
    )
    
    return {
        'OCRError': OCRError,
        'OCRProviderError': OCRProviderError,
        'OCRFileNotFoundError': OCRFileNotFoundError,
        'OCRUnsupportedFormatError': OCRUnsupportedFormatError,
        'OCRProcessingError': OCRProcessingError,
    }

def _get_paddleocr_classes():
    """Lazy import of PaddleOCR classes (optional dependency)."""
    try:
        from .paddleocr import (
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
        
        return {
            'PaddleOCRConfig': PaddleOCRConfig,
            'PaddleOCRProvider': PaddleOCRProvider,
            'PPStructureV3Provider': PPStructureV3Provider,
            'PPChatOCRv4Provider': PPChatOCRv4Provider,
            'PaddleOCRVLProvider': PaddleOCRVLProvider,
            'PaddleOCR': PaddleOCR,
            'PPStructureV3': PPStructureV3,
            'PPChatOCRv4': PPChatOCRv4,
            'PaddleOCRVL': PaddleOCRVL,
        }
    except ImportError:
        return {}

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Base classes
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    # Exception classes
    exception_classes = _get_exception_classes()
    if name in exception_classes:
        return exception_classes[name]
    
    # PaddleOCR classes (optional)
    paddleocr_classes = _get_paddleocr_classes()
    if name in paddleocr_classes:
        return paddleocr_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

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
    # PaddleOCR classes (optional, available if paddleocr is installed)
    "PaddleOCRConfig",
    "PaddleOCRProvider",
    "PPStructureV3Provider",
    "PPChatOCRv4Provider",
    "PaddleOCRVLProvider",
    "PaddleOCR",
    "PPStructureV3",
    "PPChatOCRv4",
    "PaddleOCRVL",
]

