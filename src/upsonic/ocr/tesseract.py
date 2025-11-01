from __future__ import annotations

from typing import List, Optional, Dict, Any

from upsonic.ocr.base import OCRProvider, OCRConfig, OCRResult, OCRTextBlock, BoundingBox
from upsonic.ocr.exceptions import OCRProviderError, OCRProcessingError

try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    _PYTESSERACT_AVAILABLE = False


class TesseractOCR(OCRProvider):
    """Tesseract OCR provider for text extraction.
    
    Tesseract is Google's open-source OCR engine with support for 100+ languages.
    It's one of the most accurate open-source OCR engines available.
    
    Note: Requires Tesseract to be installed on the system.
    - Ubuntu/Debian: sudo apt-get install tesseract-ocr
    - macOS: brew install tesseract
    - Windows: Download installer from GitHub
    
    Example:
        >>> from upsonic.ocr.tesseract import TesseractOCR
        >>> ocr = TesseractOCR(languages=['eng'], rotation_fix=True)
        >>> text = ocr.get_text('document.pdf')
    """
    
    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        tesseract_cmd: Optional[str] = None,
        tessdata_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize Tesseract OCR provider.
        
        Args:
            config: OCRConfig object
            tesseract_cmd: Path to tesseract executable (optional)
            tessdata_dir: Path to custom tessdata directory containing trained data files.
                If None, uses Tesseract's default location or TESSDATA_PREFIX environment variable
            **kwargs: Additional configuration arguments
        """
        self.tesseract_cmd = tesseract_cmd
        self.tessdata_dir = tessdata_dir
        super().__init__(config, **kwargs)
        
        # Set tesseract command if provided
        if self.tesseract_cmd and _PYTESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        
        # Set tessdata directory if provided
        if self.tessdata_dir and _PYTESSERACT_AVAILABLE:
            import os
            # Set environment variable for tessdata path
            os.environ['TESSDATA_PREFIX'] = self.tessdata_dir
    
    @property
    def name(self) -> str:
        return "tesseract"
    
    @property
    def supported_languages(self) -> List[str]:
        """Tesseract supports 100+ languages."""
        # Common language codes - full list depends on installed language packs
        return [
            'eng', 'fra', 'deu', 'spa', 'por', 'ita', 'nld', 'pol', 'rus',
            'jpn', 'chi_sim', 'chi_tra', 'kor', 'ara', 'hin', 'ben', 'tel',
            'mar', 'tam', 'guj', 'kan', 'mal', 'pan', 'tha', 'vie', 'tur',
            'heb', 'fas', 'ukr', 'ell', 'ces', 'dan', 'fin', 'hun', 'ind',
            'msa', 'nor', 'ron', 'swe', 'bul', 'hrv', 'lit', 'lav', 'est',
            'slk', 'slv', 'srp', 'cat', 'glg', 'eus', 'isl', 'gle', 'mlt',
            'cym', 'sqi', 'aze', 'bel', 'kat', 'arm', 'mkd', 'mon', 'uzb',
        ]
    
    def _validate_dependencies(self) -> None:
        """Validate that pytesseract is installed."""
        if not _PYTESSERACT_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pytesseract",
                install_command='pip install pytesseract',
                feature_name="Tesseract OCR provider"
            )
        
        from upsonic.utils.printing import ocr_language_not_supported, ocr_loading, ocr_initialized, warning_log
        
        try:
            version = pytesseract.get_tesseract_version()
            extra_info = {"Version": version.public}
            ocr_loading("Tesseract OCR", self.config.languages, extra_info)
        except Exception as e:
            raise OCRProviderError(
                "Tesseract is not installed or not in PATH. "
                "Please install Tesseract OCR:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "  macOS: brew install tesseract\n"
                "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki",
                error_code="TESSERACT_NOT_INSTALLED",
                original_error=e
            )
        
        try:
            available_langs = pytesseract.get_languages(config='')
            unsupported_langs = [lang for lang in self.config.languages if lang not in available_langs]
            if unsupported_langs:
                ocr_language_not_supported(
                    provider_name="Tesseract OCR",
                    unsupported_langs=unsupported_langs,
                    supported_langs=available_langs,
                    help_url="https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html"
                )
                raise OCRProviderError(
                    f"Language pack(s) not installed for Tesseract: {', '.join(unsupported_langs)}",
                    error_code="UNSUPPORTED_LANGUAGE"
                )
            ocr_initialized("Tesseract OCR")
        except OCRProviderError:
            raise
        except Exception:
            warning_log("Could not verify language pack availability", "Tesseract OCR")
    
    def _get_tesseract_config(self, **kwargs) -> str:
        """Build Tesseract configuration string.
        
        Args:
            **kwargs: Additional Tesseract-specific options
            
        Returns:
            Configuration string for Tesseract
        """
        config_parts = []
        
        # Page segmentation mode (PSM)
        psm = kwargs.get('psm', 3)  # Default: 3 = Fully automatic page segmentation
        config_parts.append(f'--psm {psm}')
        
        # OCR Engine Mode (OEM)
        oem = kwargs.get('oem', 3)  # Default: 3 = Both legacy and LSTM engines
        config_parts.append(f'--oem {oem}')
        
        # Add tessdata directory if provided
        if self.tessdata_dir:
            config_parts.append(f'--tessdata-dir "{self.tessdata_dir}"')
        
        # Additional custom config
        custom_config = kwargs.get('custom_config', '')
        if custom_config:
            config_parts.append(custom_config)
        
        return ' '.join(config_parts)
    
    def _process_image(self, image, **kwargs) -> OCRResult:
        """Process a single image with Tesseract.
        
        Args:
            image: PIL Image object
            **kwargs: Additional arguments (psm, oem, custom_config, etc.)
            
        Returns:
            OCRResult object
        """
        try:
            # Prepare language string for Tesseract
            # Tesseract uses '+' to separate multiple languages
            lang_string = '+'.join(self.config.languages)
            
            tesseract_config = self._get_tesseract_config(**kwargs)
            
            # Get detailed OCR data with bounding boxes
            data = pytesseract.image_to_data(
                image,
                lang=lang_string,
                config=tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            blocks = []
            text_parts = []
            confidences = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text:
                    continue
                
                confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 range
                
                # Filter by confidence threshold
                if confidence < self.config.confidence_threshold:
                    continue
                
                # Extract bounding box
                x = float(data['left'][i])
                y = float(data['top'][i])
                w = float(data['width'][i])
                h = float(data['height'][i])
                
                bbox = BoundingBox(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=confidence
                )
                
                block = OCRTextBlock(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    language=None  # Tesseract doesn't return per-block language
                )
                
                blocks.append(block)
                text_parts.append(text)
                confidences.append(confidence)
            
            # Also get the full text with layout preserved (if requested)
            if self.config.preserve_formatting:
                full_text = pytesseract.image_to_string(
                    image,
                    lang=lang_string,
                    config=tesseract_config
                ).strip()
            else:
                full_text = " ".join(text_parts) if text_parts else ""
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                blocks=blocks,
                confidence=avg_confidence,
                page_count=1,
                provider=self.name
            )
            
        except Exception as e:
            if isinstance(e, OCRProviderError):
                raise
            raise OCRProcessingError(
                f"Tesseract processing failed: {str(e)}",
                error_code="TESSERACT_PROCESSING_FAILED",
                original_error=e
            )
    
    def get_available_languages(self) -> List[str]:
        """Get list of languages installed on the system."""
        try:
            langs = pytesseract.get_languages(config='')
            return [lang for lang in langs if lang != 'osd']  # Exclude orientation/script detection
        except Exception:
            return self.supported_languages
    
    def get_tesseract_version(self) -> str:
        """Get the installed Tesseract version."""
        try:
            return pytesseract.get_tesseract_version().public
        except Exception:
            return "unknown"
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this OCR provider."""
        info = super().get_info()
        info['tesseract_version'] = self.get_tesseract_version()
        info['available_languages'] = self.get_available_languages()
        return info

