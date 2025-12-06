from __future__ import annotations

from typing import Type, Union, Optional, Dict, Any
from pathlib import Path

from upsonic.ocr.base import OCRProvider, OCRConfig, OCRResult, OCRMetrics
from upsonic.ocr.exceptions import OCRError


class OCR:
    """Unified OCR interface for multiple OCR providers.
    
    This class provides a high-level interface for performing OCR with
    different providers while maintaining a consistent API.
    
    Example:
        >>> from upsonic import OCR
        >>> from upsonic.ocr.deepseek import DeepSeekOCR
        >>> 
        >>> my_ocr = OCR(DeepSeekOCR, rotation_fix=True, api_key="your-key")
        >>> text = my_ocr.get_text("document.pdf")
        >>> print(text)
        
        >>> # Or with EasyOCR
        >>> from upsonic.ocr.easyocr import EasyOCR
        >>> my_ocr = OCR(EasyOCR, languages=['en', 'zh'], gpu=True)
        >>> text = my_ocr.get_text("chinese_document.png")
    """
    
    def __init__(
        self,
        provider_class: Type[OCRProvider],
        config: Optional[OCRConfig] = None,
        **kwargs
    ):
        """Initialize OCR with a specific provider.
        
        Args:
            provider_class: The OCR provider class to use
                (e.g., EasyOCR, RapidOCR, TesseractOCR, DeepSeekOCR)
            config: Optional OCRConfig object
            **kwargs: Additional provider-specific arguments and config overrides
                Common arguments:
                - rotation_fix: bool - Enable automatic rotation correction
                - enhance_contrast: bool - Enhance image contrast
                - remove_noise: bool - Apply noise reduction
                - languages: List[str] - Languages to detect
                - confidence_threshold: float - Minimum confidence (0.0-1.0)
                - pdf_dpi: int - DPI for PDF rendering (default: 300)
        
        Raises:
            OCRError: If provider initialization fails
        """
        if not issubclass(provider_class, OCRProvider):
            raise OCRError(
                f"Provider class must be a subclass of OCRProvider, got {provider_class}",
                error_code="INVALID_PROVIDER"
            )
        
        # Build config from kwargs if not provided
        if config is None:
            config_kwargs = {}
            config_fields = {
                'languages', 'confidence_threshold', 'rotation_fix',
                'enhance_contrast', 'remove_noise', 'pdf_dpi',
                'preserve_formatting'
            }
            
            # Extract config-related kwargs
            for key in config_fields:
                if key in kwargs:
                    config_kwargs[key] = kwargs.pop(key)
            
            if config_kwargs:
                config = OCRConfig(**config_kwargs)
        
        # Initialize provider
        try:
            if config:
                self.provider = provider_class(config=config, **kwargs)
            else:
                self.provider = provider_class(**kwargs)
        except Exception as e:
            raise OCRError(
                f"Failed to initialize OCR provider: {str(e)}",
                error_code="PROVIDER_INIT_FAILED",
                original_error=e
            )
    
    def get_text(self, file_path: Union[str, Path], **kwargs) -> str:
        """Extract text from an image or PDF file.
        
        This is the main method for the text extraction.
        
        Args:
            file_path: Path to the image or PDF file
                Supported formats:
                - Images: .png, .jpg, .jpeg, .bmp, .tiff, .tif, .webp
                - Documents: .pdf
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Extracted text as a string
            
        Raises:
            OCRFileNotFoundError: If the file doesn't exist
            OCRUnsupportedFormatError: If the file format is not supported
            OCRProcessingError: If OCR processing fails
            
        Example:
            >>> text = ocr.get_text("invoice.pdf")
            >>> text = ocr.get_text("receipt.jpg")
        """
        return self.provider.get_text(file_path, **kwargs)
    
    def process_file(self, file_path: Union[str, Path], **kwargs) -> OCRResult:
        """Process a file and return detailed OCR results.
        
        Use this method when you need detailed information like confidence scores,
        bounding boxes, and per-block text.
        
        Args:
            file_path: Path to the image or PDF file
            **kwargs: Additional provider-specific arguments
            
        Returns:
            OCRResult object with detailed information
            
        Example:
            >>> result = ocr.process_file("document.pdf")
            >>> print(f"Text: {result.text}")
            >>> print(f"Confidence: {result.confidence:.2%}")
            >>> print(f"Pages: {result.page_count}")
            >>> print(f"Processing time: {result.processing_time_ms:.2f}ms")
            >>> for block in result.blocks:
            ...     print(f"  Block: {block.text[:50]}... (conf: {block.confidence:.2%})")
        """
        return self.provider.process_file(file_path, **kwargs)
    
    def get_metrics(self) -> OCRMetrics:
        """Get current metrics for the OCR provider.
        
        Returns:
            OCRMetrics object with statistics
            
        Example:
            >>> metrics = ocr.get_metrics()
            >>> print(f"Files processed: {metrics.files_processed}")
            >>> print(f"Total pages: {metrics.total_pages}")
            >>> print(f"Avg confidence: {metrics.average_confidence:.2%}")
        """
        return self.provider.get_metrics()
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self.provider.reset_metrics()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the OCR provider.
        
        Returns:
            Dictionary with provider information
            
        Example:
            >>> info = ocr.get_info()
            >>> print(f"Provider: {info['name']}")
            >>> print(f"Languages: {', '.join(info['supported_languages'][:5])}...")
        """
        return self.provider.get_info()
    
    @property
    def name(self) -> str:
        """Get the name of the current OCR provider."""
        return self.provider.name
    
    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return self.provider.supported_languages
    
    @property
    def config(self) -> OCRConfig:
        """Get the current configuration."""
        return self.provider.config
    
    def __repr__(self) -> str:
        return f"OCR(provider={self.provider.name})"


def infer_provider(provider_name: str, **kwargs) -> OCR:
    """Create an OCR instance by provider name.
    
    This is a convenience function for creating OCR instances without
    importing the provider class directly.
    
    Args:
        provider_name: Name of the provider ('easyocr', 'rapidocr', 'tesseract', 'deepseek')
        **kwargs: Arguments to pass to the OCR constructor
        
    Returns:
        OCR instance
        
    Raises:
        OCRError: If provider name is unknown
        
    Example:
        >>> ocr = infer_provider('easyocr', languages=['en'], rotation_fix=True)
        >>> text = ocr.get_text('document.pdf')
    """
    provider_map = {
        'easyocr': 'upsonic.ocr.easyocr.EasyOCR',
        'rapidocr': 'upsonic.ocr.rapidocr.RapidOCR',
        'tesseract': 'upsonic.ocr.tesseract.TesseractOCR',
        'deepseek': 'upsonic.ocr.deepseek.DeepSeekOCR',
        'deepseek_ocr': 'upsonic.ocr.deepseek.DeepSeekOCR',
        'deepseek_ollama': 'upsonic.ocr.deepseek_ollama.DeepSeekOllamaOCR',
        'paddleocr': 'upsonic.ocr.paddleocr.PaddleOCR',
        'paddle': 'upsonic.ocr.paddleocr.PaddleOCR',
        'ppstructurev3': 'upsonic.ocr.paddleocr.PPStructureV3',
        'pp_structure_v3': 'upsonic.ocr.paddleocr.PPStructureV3',
        'ppchatocrv4': 'upsonic.ocr.paddleocr.PPChatOCRv4',
        'pp_chat_ocr_v4': 'upsonic.ocr.paddleocr.PPChatOCRv4',
        'paddleocr_vl': 'upsonic.ocr.paddleocr.PaddleOCRVL',
        'paddleocrvl': 'upsonic.ocr.paddleocr.PaddleOCRVL',
    }
    
    provider_name_lower = provider_name.lower()
    if provider_name_lower not in provider_map:
        raise OCRError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {', '.join(provider_map.keys())}",
            error_code="UNKNOWN_PROVIDER"
        )
    
    module_path, class_name = provider_map[provider_name_lower].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    provider_class = getattr(module, class_name)
    
    return OCR(provider_class, **kwargs)

