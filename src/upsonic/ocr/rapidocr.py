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
    
    def __init__(self, config: Optional[OCRConfig] = None, **kwargs):
        """Initialize RapidOCR provider.
        
        Args:
            config: OCRConfig object
            **kwargs: Additional configuration arguments
        """
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
            try:
                self._engine = RapidOCREngine()
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

