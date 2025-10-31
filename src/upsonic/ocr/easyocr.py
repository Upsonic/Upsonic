from __future__ import annotations

from typing import List, Optional
import numpy as np

from upsonic.ocr.base import OCRProvider, OCRConfig, OCRResult, OCRTextBlock, BoundingBox
from upsonic.ocr.exceptions import OCRProviderError, OCRProcessingError

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    _EASYOCR_AVAILABLE = False


class EasyOCR(OCRProvider):
    """EasyOCR provider for text extraction.
    
    EasyOCR is a ready-to-use OCR with 80+ supported languages.
    It uses deep learning models for high-accuracy text detection and recognition.
    
    Example:
        >>> from upsonic.ocr.easyocr import EasyOCR
        >>> ocr = EasyOCR(languages=['en'], rotation_fix=True)
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
                If None, uses EasyOCR's default location (~/.EasyOCR/model)
            download_enabled: Whether to allow automatic model downloads (default: True)
            **kwargs: Additional configuration arguments
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
        """Get or create EasyOCR reader instance."""
        if self._reader is None:
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
                
                # Handle SSL certificate issues during model download
                # EasyOCR downloads models on first use
                import ssl
                import urllib.request
                
                # Save original SSL context
                original_context = ssl._create_default_https_context
                
                try:
                    # Temporarily disable SSL verification for model download
                    ssl._create_default_https_context = ssl._create_unverified_context
                    
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
                finally:
                    # Restore original SSL context
                    ssl._create_default_https_context = original_context
                    
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

