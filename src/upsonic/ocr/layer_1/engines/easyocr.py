from __future__ import annotations

import os
import ssl
from contextlib import contextmanager
from typing import Iterator, List, Optional

import numpy as np

from upsonic.ocr.base import OCRProvider, OCRConfig, OCRResult, OCRTextBlock, BoundingBox
from upsonic.ocr.exceptions import OCRProviderError, OCRProcessingError
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    _EASYOCR_AVAILABLE = False


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@contextmanager
def _optional_ssl_bypass(disable: bool) -> Iterator[None]:
    """Optionally swap in an unverified default SSL context.

    EasyOCR downloads models on first use and routes the request through
    ``urllib`` with the process-wide default SSL context. Some corporate
    networks ship intermediate CAs that the system trust store cannot
    validate, so historically this code unconditionally replaced
    ``ssl._create_default_https_context`` with the unverified variant for
    the duration of ``easyocr.Reader(...)``. That is a global, process-wide
    side effect — any concurrent HTTPS call in another thread inherits the
    unverified context for the same window.

    The bypass is now opt-out via ``UPSONIC_OCR_DISABLE_SSL_BYPASS``. The
    preferred long-term fix is to pre-download EasyOCR models with
    ``model_storage_directory`` so the runtime never needs to fetch over
    HTTPS at all (see :class:`EasyOCREngine` docstring).
    """
    if disable:
        yield
        return

    logger.warning(
        "EasyOCREngine is temporarily replacing the process-wide default "
        "SSL context with an unverified one to allow EasyOCR model download. "
        "Concurrent HTTPS calls in other threads will inherit the unverified "
        "context until Reader init completes. Set "
        "UPSONIC_OCR_DISABLE_SSL_BYPASS=1 to disable this behavior, or "
        "pre-download models with `model_storage_directory` to avoid it."
    )
    original_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        yield
    finally:
        ssl._create_default_https_context = original_context


class EasyOCREngine(OCRProvider):
    """EasyOCR engine for text extraction.

    EasyOCR is a ready-to-use OCR with 80+ supported languages.
    It uses deep learning models for high-accuracy text detection and recognition.

    Example:
        >>> from upsonic.ocr.layer_1.engines import EasyOCREngine
        >>> ocr = EasyOCREngine(languages=['en'], rotation_fix=True)
        >>> text = ocr.get_text('document.pdf')
    """
    
    def __init__(
        self, 
        config: Optional[OCRConfig] = None, 
        gpu: bool = False,
        model_storage_directory: Optional[str] = None,
        download_enabled: bool = True,
        **kwargs
    ):
        """Initialize EasyOCR provider.

        Args:
            config: OCRConfig object
            gpu: Whether to use GPU acceleration
            model_storage_directory: Path to directory where models are stored/downloaded.
                If None, uses EasyOCR's default location (~/.EasyOCR/model). Pre-populating
                this directory at deploy time is the recommended way to avoid running the
                first-use model download with global SSL verification disabled (see below).
            download_enabled: Whether to allow automatic model downloads (default: True)
            **kwargs: Additional configuration arguments

        Notes:
            On first use, EasyOCR downloads its models over HTTPS. To tolerate corporate
            networks with broken intermediate CAs, ``EasyOCREngine`` temporarily replaces
            the process-wide default SSL context with an unverified one for the duration
            of ``easyocr.Reader(...)``. This is a global side effect — concurrent HTTPS
            calls in other threads inherit the unverified context for the same window,
            and a warning is logged each time it happens. Set
            ``UPSONIC_OCR_DISABLE_SSL_BYPASS=1`` (truthy values: ``1``, ``true``, ``yes``,
            ``on``) to disable the bypass and rely on the system trust store, or
            pre-populate ``model_storage_directory`` so the runtime never needs to
            download models at all.
        """
        self.gpu = gpu
        self.model_storage_directory = model_storage_directory
        self.download_enabled = download_enabled
        self._reader = None
        super().__init__(config, **kwargs)
    
    @property
    def name(self) -> str:
        return "easyocr"
    
    @property
    def supported_languages(self) -> List[str]:
        """EasyOCR supports 80+ languages."""
        return [
            'en', 'zh', 'ja', 'ko', 'th', 'vi', 'ar', 'ru', 'de', 'fr', 
            'es', 'pt', 'it', 'nl', 'pl', 'tr', 'hi', 'bn', 'ta', 'te',
            'mr', 'ne', 'pa', 'si', 'ur', 'fa', 'he', 'el', 'cs', 'da',
            'fi', 'hu', 'id', 'ms', 'no', 'ro', 'sv', 'uk', 'bg', 'hr',
            'lt', 'lv', 'et', 'ga', 'is', 'mk', 'mt', 'sk', 'sl', 'sq',
        ]
    
    def _validate_dependencies(self) -> None:
        """Validate that EasyOCR is installed."""
        if not _EASYOCR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="easyocr",
                install_command='pip install easyocr',
                feature_name="EasyOCR provider"
            )
    
    def _get_reader(self):
        """Get or create EasyOCR reader instance (thread-safe)."""
        if self._reader is not None:
            return self._reader
        with self._reader_lock:
            # Double-check after acquiring lock
            if self._reader is not None:
                return self._reader
            try:
                from upsonic.utils.printing import ocr_language_not_supported, ocr_loading, ocr_initialized

                # Check language support
                unsupported_langs = [lang for lang in self.config.languages if lang not in self.supported_languages]
                if unsupported_langs:
                    ocr_language_not_supported(
                        provider_name="EasyOCR",
                        unsupported_langs=unsupported_langs,
                        supported_langs=self.supported_languages,
                        help_url="https://www.jaided.ai/easyocr/"
                    )
                    raise OCRProviderError(
                        f"Language(s) not supported by EasyOCR: {', '.join(unsupported_langs)}",
                        error_code="UNSUPPORTED_LANGUAGE"
                    )

                # Show loading message
                extra_info = {
                    "GPU": "Enabled" if self.gpu else "Disabled",
                    "Note": "First run will download models"
                }
                ocr_loading("EasyOCR", self.config.languages, extra_info)

                # EasyOCR fetches model weights from the network on first use.
                # The bypass is opt-out via UPSONIC_OCR_DISABLE_SSL_BYPASS; see the
                # `_optional_ssl_bypass` context manager and the class docstring.
                disable_bypass = _truthy_env("UPSONIC_OCR_DISABLE_SSL_BYPASS")

                with _optional_ssl_bypass(disable=disable_bypass):
                    # Build Reader arguments
                    reader_kwargs = {
                        'gpu': self.gpu,
                        'verbose': False,
                        'download_enabled': self.download_enabled
                    }

                    # Add custom model storage directory if provided
                    if self.model_storage_directory:
                        reader_kwargs['model_storage_directory'] = self.model_storage_directory

                    self._reader = easyocr.Reader(
                        self.config.languages,
                        **reader_kwargs
                    )

                    ocr_initialized("EasyOCR")

            except Exception as e:
                raise OCRProviderError(
                    f"Failed to initialize EasyOCR reader: {str(e)}",
                    error_code="READER_INIT_FAILED",
                    original_error=e
                )
        return self._reader
    
    def _process_image(self, image, **kwargs) -> OCRResult:
        """Process a single image with EasyOCR.
        
        Args:
            image: PIL Image object
            **kwargs: Additional arguments (paragraph, detail, etc.)
            
        Returns:
            OCRResult object
        """
        try:
            reader = self._get_reader()
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Perform OCR
            # detail=1 returns bounding boxes and confidence scores
            results = reader.readtext(
                img_array,
                detail=1,
                paragraph=kwargs.get('paragraph', False),
                min_size=kwargs.get('min_size', 10),
                text_threshold=kwargs.get('text_threshold', 0.7),
                low_text=kwargs.get('low_text', 0.4),
                link_threshold=kwargs.get('link_threshold', 0.4),
                canvas_size=kwargs.get('canvas_size', 2560),
                mag_ratio=kwargs.get('mag_ratio', 1.0),
            )
            
            # Process results
            blocks = []
            text_parts = []
            confidences = []
            
            for bbox_coords, text, confidence in results:
                # Filter by confidence threshold
                if confidence < self.config.confidence_threshold:
                    continue
                
                # Extract bounding box coordinates
                # bbox_coords is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                
                bbox = BoundingBox(
                    x=min(x_coords),
                    y=min(y_coords),
                    width=max(x_coords) - min(x_coords),
                    height=max(y_coords) - min(y_coords),
                    confidence=confidence
                )
                
                block = OCRTextBlock(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    language=None  # EasyOCR doesn't return per-block language
                )
                
                blocks.append(block)
                text_parts.append(text)
                confidences.append(confidence)
            
            # Combine text
            combined_text = " ".join(text_parts) if text_parts else ""
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return OCRResult(
                text=combined_text,
                blocks=blocks,
                confidence=avg_confidence,
                page_count=1,
                provider=self.name
            )
            
        except Exception as e:
            if isinstance(e, OCRProviderError):
                raise
            raise OCRProcessingError(
                f"EasyOCR processing failed: {str(e)}",
                error_code="EASYOCR_PROCESSING_FAILED",
                original_error=e
            )

