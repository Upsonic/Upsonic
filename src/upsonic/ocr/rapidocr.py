from __future__ import annotations

from typing import List, Optional
import numpy as np

from upsonic.ocr.base import OCRProvider, OCRConfig, OCRResult, OCRTextBlock, BoundingBox
from upsonic.ocr.exceptions import OCRProviderError, OCRProcessingError

try:
    from rapidocr_onnxruntime import RapidOCR as RapidOCREngine
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    try:
        from rapidocr_openvino import RapidOCR as RapidOCREngine
        _RAPIDOCR_AVAILABLE = True
    except ImportError:
        RapidOCREngine = None
        _RAPIDOCR_AVAILABLE = False


class RapidOCR(OCRProvider):
    """RapidOCR provider for fast text extraction.
    
    RapidOCR is a lightweight OCR library based on ONNX Runtime.
    It provides fast inference with support for multiple backends.
    
    Example:
        >>> from upsonic.ocr.rapidocr import RapidOCR
        >>> ocr = RapidOCR(languages=['en', 'ch'], rotation_fix=True)
        >>> text = ocr.get_text('document.png')
    """
    
    def __init__(
        self, 
        config: Optional[OCRConfig] = None,
        det_model_path: Optional[str] = None,
        rec_model_path: Optional[str] = None,
        cls_model_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize RapidOCR provider.
        
        Args:
            config: OCRConfig object
            det_model_path: Path to custom text detection model file (ONNX format)
            rec_model_path: Path to custom text recognition model file (ONNX format)
            cls_model_path: Path to custom text direction classifier model file (ONNX format)
            **kwargs: Additional configuration arguments
        """
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.cls_model_path = cls_model_path
        self._engine = None
        super().__init__(config, **kwargs)
    
    @property
    def name(self) -> str:
        return "rapidocr"
    
    @property
    def supported_languages(self) -> List[str]:
        """RapidOCR primarily supports Chinese and English."""
        return [
            'en', 'ch', 'chinese_cht', 'japan', 'korean',
            'ta', 'te', 'ka', 'latin', 'arabic', 'cyrillic',
            'devanagari'
        ]
    
    def _validate_dependencies(self) -> None:
        """Validate that RapidOCR is installed."""
        if not _RAPIDOCR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="rapidocr-onnxruntime",
                install_command='pip install rapidocr-onnxruntime',
                feature_name="RapidOCR provider"
            )
    
    def _get_engine(self):
        """Get or create RapidOCR engine instance."""
        if self._engine is None:
            from upsonic.utils.printing import ocr_language_not_supported, ocr_loading, ocr_initialized
            
            unsupported_langs = [lang for lang in self.config.languages if lang not in self.supported_languages]
            if unsupported_langs:
                ocr_language_not_supported(
                    provider_name="RapidOCR",
                    unsupported_langs=unsupported_langs,
                    supported_langs=self.supported_languages,
                    help_url=None
                )
                raise OCRProviderError(
                    f"Language(s) not supported by RapidOCR: {', '.join(unsupported_langs)}",
                    error_code="UNSUPPORTED_LANGUAGE"
                )
            
            extra_info = {
                "Note": "Primarily supports Chinese and English"
            }
            
            # Add model path info if custom models are provided
            if self.det_model_path or self.rec_model_path or self.cls_model_path:
                model_info = []
                if self.det_model_path:
                    model_info.append(f"det={self.det_model_path}")
                if self.rec_model_path:
                    model_info.append(f"rec={self.rec_model_path}")
                if self.cls_model_path:
                    model_info.append(f"cls={self.cls_model_path}")
                extra_info["Custom Models"] = ", ".join(model_info)
            
            ocr_loading("RapidOCR", self.config.languages, extra_info)
            
            try:
                # Build engine initialization arguments
                engine_kwargs = {}
                
                # Add custom model paths if provided
                if self.det_model_path:
                    engine_kwargs['Det.model_path'] = self.det_model_path
                if self.rec_model_path:
                    engine_kwargs['CLs.model_path'] = self.rec_model_path
                if self.cls_model_path:
                    engine_kwargs['Rec.model_path'] = self.cls_model_path
                
                self._engine = RapidOCREngine(**engine_kwargs) if engine_kwargs else RapidOCREngine()
                ocr_initialized("RapidOCR")
            except Exception as e:
                raise OCRProviderError(
                    f"Failed to initialize RapidOCR engine: {str(e)}",
                    error_code="ENGINE_INIT_FAILED",
                    original_error=e
                )
        return self._engine
    
    def _process_image(self, image, **kwargs) -> OCRResult:
        """Process a single image with RapidOCR.
        
        Args:
            image: PIL Image object
            **kwargs: Additional arguments
            
        Returns:
            OCRResult object
        """
        try:
            engine = self._get_engine()
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Perform OCR
            # RapidOCR returns: (dt_boxes, rec_res, time_dict) or (dt_boxes, rec_res) or None
            result = engine(img_array)
            
            if result is None or not result[0]:
                # No text detected
                return OCRResult(
                    text="",
                    blocks=[],
                    confidence=0.0,
                    page_count=1,
                    provider=self.name
                )
            
            # Handle different return formats
            if len(result) == 3:
                dt_boxes, rec_res, time_dict = result
            else:
                dt_boxes, rec_res = result
                time_dict = {}
            
            # Process results
            blocks = []
            text_parts = []
            confidences = []
            
            # RapidOCR format: dt_boxes is a list where each item is [box_coords, text, confidence_str]
            for item in dt_boxes:
                # Each item is [box_coords, text, confidence_str]
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                
                box_coords, text, confidence_str = item[0], item[1], item[2]
                
                # Convert confidence string to float
                try:
                    confidence = float(confidence_str)
                except (ValueError, TypeError):
                    confidence = 0.0
                
                # Filter by confidence threshold
                if confidence < self.config.confidence_threshold:
                    continue
                
                # Extract bounding box
                # box_coords is array of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in box_coords]
                y_coords = [point[1] for point in box_coords]
                
                bbox = BoundingBox(
                    x=float(min(x_coords)),
                    y=float(min(y_coords)),
                    width=float(max(x_coords) - min(x_coords)),
                    height=float(max(y_coords) - min(y_coords)),
                    confidence=float(confidence)
                )
                
                block = OCRTextBlock(
                    text=text,
                    confidence=float(confidence),
                    bbox=bbox,
                    language=None
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
                provider=self.name,
                metadata={'processing_time': time_dict}
            )
            
        except Exception as e:
            if isinstance(e, OCRProviderError):
                raise
            raise OCRProcessingError(
                f"RapidOCR processing failed: {str(e)}",
                error_code="RAPIDOCR_PROCESSING_FAILED",
                original_error=e
            )

