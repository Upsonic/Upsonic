"""PaddleOCR provider

This module provides comprehensive wrappers for all PaddleOCR pipelines:
- PaddleOCR: General OCR with PP-OCRv3/v4/v5 models
- PPStructureV3: Advanced document structure recognition
- PPChatOCRv4: Chat-based OCR with multimodal capabilities
- PaddleOCRVL: Vision-Language OCR for complex documents

All providers are fully compatible with the Upsonic AI agent framework.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

from upsonic.ocr.base import (
    OCRProvider, 
    OCRConfig, 
    OCRResult, 
    OCRTextBlock,
    BoundingBox
)
from upsonic.ocr.exceptions import OCRError, OCRProcessingError


class PaddleOCRConfig(OCRConfig):
    """Extended configuration for PaddleOCR providers with all pipeline-specific settings.
    
    This config extends the base OCRConfig to include all PaddleOCR-specific parameters
    while maintaining compatibility with the unified OCR interface.
    """
    
    # Model configurations
    doc_orientation_classify_model_name: Optional[str] = None
    doc_orientation_classify_model_dir: Optional[str] = None
    doc_unwarping_model_name: Optional[str] = None
    doc_unwarping_model_dir: Optional[str] = None
    text_detection_model_name: Optional[str] = None
    text_detection_model_dir: Optional[str] = None
    textline_orientation_model_name: Optional[str] = None
    textline_orientation_model_dir: Optional[str] = None
    text_recognition_model_name: Optional[str] = None
    text_recognition_model_dir: Optional[str] = None
    
    # Batch sizes
    textline_orientation_batch_size: Optional[int] = None
    text_recognition_batch_size: Optional[int] = None
    
    # Feature toggles
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    use_textline_orientation: Optional[bool] = None
    
    # Text detection parameters
    text_det_limit_side_len: Optional[int] = None
    text_det_limit_type: Optional[str] = None
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None
    text_det_input_shape: Optional[tuple] = None
    
    # Text recognition parameters
    text_rec_score_thresh: Optional[float] = None
    return_word_box: Optional[bool] = None
    text_rec_input_shape: Optional[tuple] = None
    
    # Language and version
    lang: Optional[str] = None
    ocr_version: Optional[str] = None


class BasePaddleOCRProvider(OCRProvider):
    """Base class for all PaddleOCR providers with shared functionality.
    
    This abstract base class provides common functionality for all PaddleOCR
    pipeline implementations, ensuring consistent behavior across providers.
    """
    
    def __init__(
        self, 
        config: Optional[Union[OCRConfig, PaddleOCRConfig]] = None,
        **kwargs
    ):
        """Initialize the PaddleOCR provider.
        
        Args:
            config: OCRConfig or PaddleOCRConfig object
            **kwargs: Additional PaddleOCR-specific parameters
        """
        # Convert OCRConfig to PaddleOCRConfig if needed
        if config and not isinstance(config, PaddleOCRConfig):
            config_dict = config.model_dump()
            config_dict.update(kwargs)
            config = PaddleOCRConfig(**config_dict)
            kwargs = {}
        elif config is None:
            config = PaddleOCRConfig(**kwargs)
            kwargs = {}
        
        super().__init__(config=config, **kwargs)
        self._paddle_instance = None
        self._paddle_kwargs = kwargs
        self._initialize_paddle()
    
    def _validate_dependencies(self) -> None:
        """Validate that PaddleOCR is installed."""
        try:
            import paddleocr
            self._paddleocr_available = True
        except ImportError:
            self._paddleocr_available = False
            from upsonic.utils.printing import import_error
            import_error(
                package_name="paddleocr",
                install_command='pip install paddleocr',
                feature_name="PaddleOCR"
            )
    
    def _initialize_paddle(self) -> None:
        """Initialize the PaddleOCR instance. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_paddle")
    
    def _build_paddle_params(self, config: PaddleOCRConfig, **kwargs) -> Dict[str, Any]:
        """Build parameters dictionary for PaddleOCR initialization.
        
        Args:
            config: PaddleOCRConfig object
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of parameters for PaddleOCR
        """
        params = {}
        
        # Add all non-None config values
        config_dict = config.model_dump()
        for key, value in config_dict.items():
            if value is not None:
                params[key] = value
        
        # Override with kwargs
        params.update(kwargs)
        
        # Remove base OCRConfig fields that aren't valid for PaddleOCR
        base_config_fields = {
            'languages', 'confidence_threshold', 'rotation_fix',
            'enhance_contrast', 'remove_noise', 'pdf_dpi', 
            'preserve_formatting'
        }
        params = {k: v for k, v in params.items() if k not in base_config_fields}
        
        # Remove any pydantic internal fields
        internal_fields = {'__class__', '__dict__', '__doc__', '__module__'}
        params = {k: v for k, v in params.items() if k not in internal_fields}
        
        return params
    
    def _process_image(self, image, **kwargs) -> OCRResult:
        """Process a single image and extract text using PaddleOCR.
        
        Args:
            image: PIL Image object
            **kwargs: Additional processing arguments
            
        Returns:
            OCRResult object with extracted text and metadata
        """
        if not self._paddle_instance:
            raise OCRError(
                "PaddleOCR instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        start_time = time.time()
        
        try:
            import numpy as np
            import tempfile
            
            # Save image to temporary file for PaddleOCR (it expects file path)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name, format='PNG')
                tmp_path = tmp.name
            
            try:
                # Process with PaddleOCR using predict() method
                paddle_result = self._paddle_instance.predict(tmp_path, **kwargs)
                
                # Extract text and metadata from result
                text, blocks, confidence = self._extract_paddle_predict_result(paddle_result)
                
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                text=text,
                blocks=blocks,
                confidence=confidence,
                page_count=1,
                processing_time_ms=processing_time,
                provider=self.name,
                metadata={
                    'paddle_result': paddle_result,
                    'config': self.config.model_dump()
                }
            )
            
        except Exception as e:
            raise OCRProcessingError(
                f"PaddleOCR processing failed: {str(e)}",
                error_code="PADDLE_PROCESSING_FAILED",
                original_error=e
            )
    
    def _extract_paddle_predict_result(self, paddle_result: Any) -> tuple[str, List[OCRTextBlock], float]:
        """Extract text from PaddleOCR predict() result.
        
        The predict() method returns result objects which are dict-like with:
        - rec_texts: list of recognized text strings
        - rec_scores: list of confidence scores
        - dt_polys: detection polygons (bounding boxes)
        - rec_boxes: recognition boxes
        
        Args:
            paddle_result: List of result dicts from paddle.predict()
            
        Returns:
            Tuple of (text, blocks, confidence)
        """
        text_parts = []
        blocks = []
        total_confidence = 0.0
        count = 0
        
        # Handle None or empty results
        if not paddle_result:
            return "", [], 0.0
        
        try:
            # PaddleOCR predict() returns a list of result objects (one per page/image)
            for page_idx, res in enumerate(paddle_result):
                # Handle dict-like objects
                if hasattr(res, 'get'):
                    rec_texts = res.get('rec_texts', [])
                    rec_scores = res.get('rec_scores', [])
                    dt_polys = res.get('dt_polys', [])
                    rec_boxes = res.get('rec_boxes', [])
                else:
                    # Fallback for non-dict results
                    rec_texts = getattr(res, 'rec_texts', [])
                    rec_scores = getattr(res, 'rec_scores', [])
                    dt_polys = getattr(res, 'dt_polys', [])
                    rec_boxes = getattr(res, 'rec_boxes', [])
                
                # Ensure we have lists
                if rec_texts is None:
                    rec_texts = []
                if rec_scores is None:
                    rec_scores = []
                if dt_polys is None:
                    dt_polys = []
                
                # Process each recognized text
                for i, text in enumerate(rec_texts):
                    if text and str(text).strip():  # Skip empty or whitespace-only texts
                        text_str = str(text).strip()
                        text_parts.append(text_str)
                        
                        # Get confidence score
                        try:
                            conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                        except (ValueError, TypeError, IndexError):
                            conf = 1.0
                        
                        total_confidence += conf
                        count += 1
                        
                        # Create bounding box if available
                        bbox = None
                        if i < len(dt_polys) and dt_polys[i] is not None:
                            try:
                                poly = dt_polys[i]
                                # Handle different polygon formats
                                if isinstance(poly, (list, tuple)) and len(poly) > 0:
                                    # poly is a list of [x, y] points
                                    x_coords = [float(p[0]) for p in poly if len(p) >= 2]
                                    y_coords = [float(p[1]) for p in poly if len(p) >= 2]
                                    
                                    if x_coords and y_coords:
                                        x = min(x_coords)
                                        y = min(y_coords)
                                        width = max(x_coords) - x
                                        height = max(y_coords) - y
                                        bbox = BoundingBox(
                                            x=x,
                                            y=y,
                                            width=width,
                                            height=height,
                                            confidence=conf
                                        )
                            except (IndexError, TypeError, ValueError, AttributeError) as e:
                                # Skip bbox creation on error
                                pass
                        
                        # Create text block
                        blocks.append(OCRTextBlock(
                            text=text_str,
                            confidence=conf,
                            bbox=bbox,
                            page_number=page_idx + 1
                        ))
                    
        except Exception as e:
            # Fallback: try to get string representation
            try:
                if isinstance(paddle_result, (list, tuple)) and paddle_result:
                    text_str = str(paddle_result[0])
                else:
                    text_str = str(paddle_result)
                
                if text_str.strip():
                    text_parts = [text_str]
                    blocks = [OCRTextBlock(text=text_str, confidence=1.0, page_number=1)]
                    count = 1
                    total_confidence = 1.0
            except:
                # Complete fallback
                pass
        
        combined_text = "\n".join(text_parts) if text_parts else ""
        avg_confidence = total_confidence / count if count > 0 else 0.0
        
        return combined_text, blocks, avg_confidence
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported languages based on OCR version."""
        # PP-OCRv5 supported languages
        v5_langs = [
            'ch', 'en', 'fr', 'de', 'japan', 'korean', 'chinese_cht', 
            'af', 'it', 'es', 'bs', 'pt', 'cs', 'cy', 'da', 'et', 
            'ga', 'hr', 'hu', 'rslatin', 'id', 'oc', 'is', 'lt', 
            'mi', 'ms', 'nl', 'no', 'pl', 'sk', 'sl', 'sq', 'sv', 
            'sw', 'tl', 'tr', 'uz', 'la', 'ru', 'be', 'uk'
        ]
        
        # Check configured version
        ocr_version = getattr(self.config, 'ocr_version', None)
        if ocr_version == 'PP-OCRv4':
            return ['ch', 'en']
        elif ocr_version == 'PP-OCRv3':
            # PP-OCRv3 supports many more languages
            return v5_langs + [
                'abq', 'ady', 'ang', 'ar', 'ava', 'az', 'bg', 'bgc', 
                'bh', 'bho', 'che', 'gom', 'hi', 'inh', 'ka', 'kbd', 
                'ku', 'lbe', 'lez', 'lv', 'mah', 'mai', 'mn', 'mr', 
                'mt', 'ne', 'new', 'pi', 'ro', 'rs_cyrillic', 'sa', 
                'sck', 'ta', 'tab', 'te', 'ug', 'ur', 'vi', 'fa'
            ]
        
        # Default to PP-OCRv5
        return v5_langs


class PaddleOCRProvider(BasePaddleOCRProvider):
    """General OCR Pipeline using PaddleOCR (PP-OCRv3/v4/v5).
    
    This provider wraps the general PaddleOCR pipeline which supports:
    - PP-OCRv5 (default): Best accuracy with 13% improvement over v4
    - PP-OCRv4: Good balance of speed and accuracy
    - PP-OCRv3: Supports most languages
    
    Example:
        >>> from upsonic import OCR
        >>> from upsonic.ocr.paddleocr import PaddleOCRProvider
        >>> 
        >>> ocr = OCR(PaddleOCRProvider, lang='en', ocr_version='PP-OCRv5')
        >>> text = ocr.get_text('document.pdf')
        >>> print(text)
    """
    
    def process_file(self, file_path: Union[str, Path], **kwargs) -> OCRResult:
        """Process a file directly using PaddleOCR.
        
        Override the base class method to work directly with file paths.
        PaddleOCR can handle PDFs directly, but we optimize for better reliability.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional provider-specific arguments
            
        Returns:
            OCRResult object with detailed information
        """
        from upsonic.ocr.utils import validate_file_path, is_pdf, pdf_to_images
        import tempfile
        import os
        
        start_time = time.time()
        
        # Validate file exists
        path = validate_file_path(file_path)
        
        try:
            # Check if it's a PDF
            if is_pdf(path):
                # Use lower DPI for faster processing while maintaining accuracy
                # PaddleOCR works well with 200 DPI, which is much faster than 300
                dpi = min(self.config.pdf_dpi, 200)
                
                print(f"📄 Converting PDF to images (DPI={dpi})...")
                images = pdf_to_images(path, dpi=dpi)
                print(f"  ✓ Converted to {len(images)} page(s)")
                
                # Process each page as an image
                all_results = []
                print(f"🔍 Processing pages with PaddleOCR...")
                
                for page_num, image in enumerate(images, 1):
                    print(f"  • Page {page_num}/{len(images)}...", end=' ', flush=True)
                    
                    # Save image to temp file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        # Use JPEG for faster I/O and smaller temp files
                        image.save(tmp.name, format='PNG')
                        tmp_path = tmp.name
                    
                    try:
                        # Process with PaddleOCR
                        paddle_result = self._paddle_instance.predict(tmp_path, **kwargs)
                        all_results.extend(paddle_result)
                        print("✓")
                    finally:
                        # Clean up temp file immediately
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                # Extract text and metadata from all results
                text, blocks, confidence = self._extract_paddle_predict_result(all_results)
                
                # Update page numbers for blocks
                if blocks and len(images) > 1:
                    blocks_per_page = len(blocks) // len(images)
                    for i, block in enumerate(blocks):
                        block.page_number = (i // blocks_per_page) + 1 if blocks_per_page > 0 else (i % len(images)) + 1
                
                page_count = len(images)
                
            else:
                # For images, process directly without temp files
                print(f"🖼️  Processing image with PaddleOCR...")
                paddle_result = self._paddle_instance.predict(str(path), **kwargs)
                
                # Extract text and metadata from result
                text, blocks, confidence = self._extract_paddle_predict_result(paddle_result)
                
                page_count = len(paddle_result) if paddle_result else 1
                print(f"  ✓ Complete!")
            
            processing_time = (time.time() - start_time) * 1000
            
            result = OCRResult(
                text=text,
                blocks=blocks,
                confidence=confidence,
                page_count=page_count,
                processing_time_ms=processing_time,
                provider=self.name,
                metadata={
                    'file_path': str(file_path),
                    'config': self.config.model_dump()
                }
            )
            
            # Update metrics
            self._metrics.total_pages += page_count
            self._metrics.total_characters += len(text)
            self._metrics.processing_time_ms += processing_time
            self._metrics.files_processed += 1
            if blocks:
                self._metrics.average_confidence = sum(b.confidence for b in blocks) / len(blocks)
            
            return result
            
        except Exception as e:
            raise OCRProcessingError(
                f"PaddleOCR processing failed: {str(e)}",
                error_code="PADDLE_PROCESSING_FAILED",
                original_error=e
            )
    
    def __init__(
        self,
        config: Optional[Union[OCRConfig, PaddleOCRConfig]] = None,
        # Model configurations - Use model_dir to specify custom model directory paths
        # Use model_name to specify model identifier, or leave None to use defaults
        doc_orientation_classify_model_name: Optional[str] = None,
        doc_orientation_classify_model_dir: Optional[str] = None,
        doc_unwarping_model_name: Optional[str] = None,
        doc_unwarping_model_dir: Optional[str] = None,
        text_detection_model_name: Optional[str] = None,
        text_detection_model_dir: Optional[str] = None,  # Path to custom text detection model directory
        textline_orientation_model_name: Optional[str] = None,
        textline_orientation_model_dir: Optional[str] = None,
        text_recognition_model_name: Optional[str] = None,
        text_recognition_model_dir: Optional[str] = None,  # Path to custom text recognition model directory
        # Batch sizes
        textline_orientation_batch_size: Optional[int] = None,
        text_recognition_batch_size: Optional[int] = None,
        # Feature toggles
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        # Text detection parameters
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_det_input_shape: Optional[tuple] = None,
        # Text recognition parameters
        text_rec_score_thresh: Optional[float] = None,
        return_word_box: Optional[bool] = None,
        text_rec_input_shape: Optional[tuple] = None,
        # Language and version
        lang: Optional[str] = None,
        ocr_version: Optional[str] = None,
        **kwargs
    ):
        """Initialize PaddleOCR provider with all configuration options."""
        # Build kwargs for config
        local_kwargs = {
            k: v for k, v in locals().items() 
            if k not in ['self', 'config', 'kwargs'] and v is not None
        }
        local_kwargs.update(kwargs)
        
        super().__init__(config=config, **local_kwargs)
    
    def _initialize_paddle(self) -> None:
        """Initialize the PaddleOCR instance with all parameters."""
        try:
            from paddleocr import PaddleOCR
            from upsonic.utils.printing import ocr_language_warning, ocr_loading, ocr_initialized
            
            # Check language support
            lang = getattr(self.config, 'lang', 'en')
            if lang and lang not in self.supported_languages:
                ocr_language_warning(
                    provider_name="PaddleOCR",
                    warning_langs=[lang],
                    best_supported=self.supported_languages[:30]
                )
            
            # Show loading message
            ocr_version = getattr(self.config, 'ocr_version', 'PP-OCRv5')
            extra_info = {
                "Version": ocr_version,
                "Note": "First run will download models (~10-50MB per model)"
            }
            ocr_loading(f"PaddleOCR ({ocr_version})", [lang], extra_info)
            
            # Build parameters
            paddle_params = self._build_paddle_params(self.config, **self._paddle_kwargs)
            
            # Initialize PaddleOCR
            self._paddle_instance = PaddleOCR(**paddle_params)
            
            ocr_initialized(f"PaddleOCR ({ocr_version})")
            
        except Exception as e:
            raise OCRError(
                f"Failed to initialize PaddleOCR: {str(e)}",
                error_code="PADDLE_INIT_FAILED",
                original_error=e
            )
    
    @property
    def name(self) -> str:
        """Get the provider name."""
        version = getattr(self.config, 'ocr_version', 'v5')
        return f"PaddleOCR-{version}"
    
    def predict(self, input: Union[str, Path], **kwargs) -> List[Any]:
        """Predict using PaddleOCR on input file.
        
        This method directly calls the PaddleOCR predict() method and returns
        the raw results, which include text, confidence, and bounding boxes.
        
        Args:
            input: Path to input file (image or PDF)
            **kwargs: Additional prediction parameters
            
        Returns:
            List of prediction results
        """
        if not self._paddle_instance:
            raise OCRError(
                "PaddleOCR instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.predict(str(input), **kwargs)


class PPStructureV3Provider(BasePaddleOCRProvider):
    """PP-StructureV3 Pipeline for advanced document structure recognition.
    
    This provider includes comprehensive document analysis capabilities:
    - Layout detection
    - Table recognition (wired and wireless)
    - Chart recognition
    - Seal text detection and recognition
    - Formula recognition
    - General OCR for text regions
    
    Example:
        >>> from upsonic import OCR
        >>> from upsonic.ocr.paddleocr import PPStructureV3Provider
        >>> 
        >>> ocr = OCR(PPStructureV3Provider, 
        ...           use_table_recognition=True,
        ...           use_formula_recognition=True,
        ...           lang='en')
        >>> result = ocr.process_file('complex_document.pdf')
        >>> print(result.text)
    """
    
    def __init__(
        self,
        config: Optional[Union[OCRConfig, PaddleOCRConfig]] = None,
        # Layout detection
        layout_detection_model_name: Optional[str] = None,
        layout_detection_model_dir: Optional[str] = None,
        layout_threshold: Optional[float] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[float] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        # Chart recognition
        chart_recognition_model_name: Optional[str] = None,
        chart_recognition_model_dir: Optional[str] = None,
        chart_recognition_batch_size: Optional[int] = None,
        # Region detection
        region_detection_model_name: Optional[str] = None,
        region_detection_model_dir: Optional[str] = None,
        # Document preprocessing
        doc_orientation_classify_model_name: Optional[str] = None,
        doc_orientation_classify_model_dir: Optional[str] = None,
        doc_unwarping_model_name: Optional[str] = None,
        doc_unwarping_model_dir: Optional[str] = None,
        # Text detection
        text_detection_model_name: Optional[str] = None,
        text_detection_model_dir: Optional[str] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        # Text line orientation
        textline_orientation_model_name: Optional[str] = None,
        textline_orientation_model_dir: Optional[str] = None,
        textline_orientation_batch_size: Optional[int] = None,
        # Text recognition
        text_recognition_model_name: Optional[str] = None,
        text_recognition_model_dir: Optional[str] = None,
        text_recognition_batch_size: Optional[int] = None,
        text_rec_score_thresh: Optional[float] = None,
        # Table recognition
        table_classification_model_name: Optional[str] = None,
        table_classification_model_dir: Optional[str] = None,
        wired_table_structure_recognition_model_name: Optional[str] = None,
        wired_table_structure_recognition_model_dir: Optional[str] = None,
        wireless_table_structure_recognition_model_name: Optional[str] = None,
        wireless_table_structure_recognition_model_dir: Optional[str] = None,
        wired_table_cells_detection_model_name: Optional[str] = None,
        wired_table_cells_detection_model_dir: Optional[str] = None,
        wireless_table_cells_detection_model_name: Optional[str] = None,
        wireless_table_cells_detection_model_dir: Optional[str] = None,
        table_orientation_classify_model_name: Optional[str] = None,
        table_orientation_classify_model_dir: Optional[str] = None,
        # Seal text recognition
        seal_text_detection_model_name: Optional[str] = None,
        seal_text_detection_model_dir: Optional[str] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_text_recognition_model_name: Optional[str] = None,
        seal_text_recognition_model_dir: Optional[str] = None,
        seal_text_recognition_batch_size: Optional[int] = None,
        seal_rec_score_thresh: Optional[float] = None,
        # Formula recognition
        formula_recognition_model_name: Optional[str] = None,
        formula_recognition_model_dir: Optional[str] = None,
        formula_recognition_batch_size: Optional[int] = None,
        # Feature toggles
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        use_formula_recognition: Optional[bool] = None,
        use_chart_recognition: Optional[bool] = None,
        use_region_detection: Optional[bool] = None,
        # Language and version
        lang: Optional[str] = None,
        ocr_version: Optional[str] = None,
        **kwargs
    ):
        """Initialize PP-StructureV3 provider with comprehensive configuration.
        
        Args:
            config: Optional OCRConfig or PaddleOCRConfig object
            layout_detection_model_name: Name of the layout detection model
            layout_detection_model_dir: Path to the layout detection model
            layout_threshold: Score threshold for layout detection
            layout_nms: Whether to use NMS in layout detection
            layout_unclip_ratio: Expansion coefficient for layout detection
            layout_merge_bboxes_mode: Overlapping box filtering method
            chart_recognition_model_name: Name of the chart recognition model
            chart_recognition_model_dir: Path to the chart recognition model
            chart_recognition_batch_size: Batch size for chart recognition
            region_detection_model_name: Name of the region detection model
            region_detection_model_dir: Path to the region detection model
            [... all other parameters similar to __init__ signature ...]
            lang: Language code for OCR
            ocr_version: PP-OCR version to use
            **kwargs: Additional provider-specific arguments
        """
        local_kwargs = {
            k: v for k, v in locals().items() 
            if k not in ['self', 'config', 'kwargs'] and v is not None
        }
        local_kwargs.update(kwargs)
        
        super().__init__(config=config, **local_kwargs)
    
    def _initialize_paddle(self) -> None:
        """Initialize the PP-StructureV3 instance."""
        try:
            from paddleocr import PPStructureV3
            from upsonic.utils.printing import ocr_loading, ocr_initialized
            
            # Show loading message
            features = []
            if getattr(self.config, 'use_table_recognition', None):
                features.append("table recognition")
            if getattr(self.config, 'use_formula_recognition', None):
                features.append("formula recognition")
            if getattr(self.config, 'use_seal_recognition', None):
                features.append("seal recognition")
            if getattr(self.config, 'use_chart_recognition', None):
                features.append("chart recognition")
            
            extra_info = {
                "Features": ", ".join(features) if features else "basic",
                "Note": "First run will download multiple specialized models"
            }
            ocr_loading("PP-StructureV3", [getattr(self.config, 'lang', 'en')], extra_info)
            
            paddle_params = self._build_paddle_params(self.config, **self._paddle_kwargs)
            
            self._paddle_instance = PPStructureV3(**paddle_params)
            
            ocr_initialized("PP-StructureV3")
            
        except Exception as e:
            raise OCRError(
                f"Failed to initialize PP-StructureV3: {str(e)}",
                error_code="PADDLE_STRUCTURE_INIT_FAILED",
                original_error=e
            )

    
    @property
    def name(self) -> str:
        """Get the provider name."""
        return "PP-StructureV3"
    
    def predict(self, input: Union[str, Path], **kwargs) -> List[Any]:
        """Predict using PP-StructureV3 on input file.
        
        Args:
            input: Path to input file (image or PDF)
            **kwargs: Additional prediction parameters
            
        Returns:
            List of prediction results with markdown and visual info
        """
        if not self._paddle_instance:
            raise OCRError(
                "PP-StructureV3 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.predict(str(input), **kwargs)
    
    def concatenate_markdown_pages(self, markdown_list: List[Dict[str, Any]]) -> str:
        """Concatenate markdown from multiple pages.
        
        Args:
            markdown_list: List of markdown dictionaries from predict results
            
        Returns:
            Concatenated markdown text
        """
        if not self._paddle_instance:
            raise OCRError(
                "PP-StructureV3 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.concatenate_markdown_pages(markdown_list)


class PPChatOCRv4Provider(BasePaddleOCRProvider):
    """PP-ChatOCRv4 Pipeline for document understanding with chat capabilities.
    
    This provider combines OCR with multimodal language models for:
    - Document information extraction
    - Key-value pair extraction
    - Question answering over documents
    - Table and seal recognition
    - Layout-aware text extraction
    
    Example:
        >>> from upsonic import OCR
        >>> from upsonic.ocr.paddleocr import PPChatOCRv4Provider
        >>> 
        >>> ocr = OCR(PPChatOCRv4Provider,
        ...           use_table_recognition=True,
        ...           use_seal_recognition=True,
        ...           mllm_chat_bot_config={'api_key': 'your-key'})
        >>> result = ocr.process_file('invoice.pdf')
        >>> print(result.text)
        >>> 
        >>> # Or use advanced features
        >>> visual_result = ocr.provider.visual_predict('document.pdf')
        >>> vector_info = ocr.provider.build_vector(visual_result)
        >>> chat_result = ocr.provider.chat(key_list=['name', 'date'], visual_info=visual_result)
    """
    
    def __init__(
        self,
        config: Optional[Union[OCRConfig, PaddleOCRConfig]] = None,
        # Layout detection
        layout_detection_model_name: Optional[str] = None,
        layout_detection_model_dir: Optional[str] = None,
        # Document preprocessing
        doc_orientation_classify_model_name: Optional[str] = None,
        doc_orientation_classify_model_dir: Optional[str] = None,
        doc_unwarping_model_name: Optional[str] = None,
        doc_unwarping_model_dir: Optional[str] = None,
        # Text detection
        text_detection_model_name: Optional[str] = None,
        text_detection_model_dir: Optional[str] = None,
        # Text line orientation
        textline_orientation_model_name: Optional[str] = None,
        textline_orientation_model_dir: Optional[str] = None,
        textline_orientation_batch_size: Optional[int] = None,
        # Text recognition
        text_recognition_model_name: Optional[str] = None,
        text_recognition_model_dir: Optional[str] = None,
        text_recognition_batch_size: Optional[int] = None,
        # Table structure recognition
        table_structure_recognition_model_name: Optional[str] = None,
        table_structure_recognition_model_dir: Optional[str] = None,
        # Seal text recognition
        seal_text_detection_model_name: Optional[str] = None,
        seal_text_detection_model_dir: Optional[str] = None,
        seal_text_recognition_model_name: Optional[str] = None,
        seal_text_recognition_model_dir: Optional[str] = None,
        seal_text_recognition_batch_size: Optional[int] = None,
        # Feature toggles
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        # Layout parameters
        layout_threshold: Optional[float] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[float] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        # Text detection parameters
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        # Seal detection parameters
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        # Chat bot configurations
        retriever_config: Optional[Dict[str, Any]] = None,
        mllm_chat_bot_config: Optional[Dict[str, Any]] = None,
        chat_bot_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize PP-ChatOCRv4 provider with chat capabilities.
        
        Args:
            config: Optional OCRConfig or PaddleOCRConfig object
            layout_detection_model_name: Name of the layout detection model
            layout_detection_model_dir: Path to the layout detection model
            [... all other model and configuration parameters ...]
            retriever_config: Configuration for the retriever component
            mllm_chat_bot_config: Configuration for the multimodal LLM
            chat_bot_config: Configuration for the chat bot
            **kwargs: Additional provider-specific arguments
        """
        local_kwargs = {
            k: v for k, v in locals().items() 
            if k not in ['self', 'config', 'kwargs'] and v is not None
        }
        local_kwargs.update(kwargs)
        
        super().__init__(config=config, **local_kwargs)
    
    def _initialize_paddle(self) -> None:
        """Initialize the PP-ChatOCRv4 instance."""
        try:
            from paddleocr import PPChatOCRv4Doc
            from upsonic.utils.printing import ocr_loading, ocr_initialized
            
            # Show loading message
            features = []
            if getattr(self.config, 'use_table_recognition', None):
                features.append("table recognition")
            if getattr(self.config, 'use_seal_recognition', None):
                features.append("seal recognition")
            
            extra_info = {
                "Features": ", ".join(features) if features else "basic",
                "Capabilities": "Information extraction, Q&A, key-value pairs",
                "Note": "First run will download OCR and multimodal LLM models"
            }
            ocr_loading("PP-ChatOCRv4", [getattr(self.config, 'lang', 'en')], extra_info)
            
            paddle_params = self._build_paddle_params(self.config, **self._paddle_kwargs)
            
            self._paddle_instance = PPChatOCRv4Doc(**paddle_params)
            
            ocr_initialized("PP-ChatOCRv4")
            
        except Exception as e:
            raise OCRError(
                f"Failed to initialize PP-ChatOCRv4: {str(e)}",
                error_code="PADDLE_CHAT_INIT_FAILED",
                original_error=e
            )
    
    
    @property
    def name(self) -> str:
        """Get the provider name."""
        return "PP-ChatOCRv4"
    
    # Advanced PPChatOCRv4 Methods
    
    def visual_predict(
        self,
        input: Union[str, Path],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        layout_threshold: Optional[float] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[float] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform visual prediction on input document.
        
        This method extracts visual information including text, tables, and seals
        from the document using the configured OCR and layout analysis models.
        
        Args:
            input: Path to input file (image or PDF)
            use_doc_orientation_classify: Whether to classify document orientation
            use_doc_unwarping: Whether to unwarp document
            use_textline_orientation: Whether to classify text line orientation
            use_seal_recognition: Whether to recognize seals
            use_table_recognition: Whether to recognize tables
            layout_threshold: Score threshold for layout detection
            layout_nms: Whether to use NMS in layout detection
            layout_unclip_ratio: Expansion coefficient for layout detection
            layout_merge_bboxes_mode: Overlapping box filtering method
            text_det_limit_side_len: Limit on text detection input side length
            text_det_limit_type: How to apply side length limit
            text_det_thresh: Detection pixel threshold
            text_det_box_thresh: Detection box threshold
            text_det_unclip_ratio: Text detection expansion coefficient
            text_rec_score_thresh: Text recognition score threshold
            seal_det_limit_side_len: Limit on seal detection input side length
            seal_det_limit_type: How to apply seal side length limit
            seal_det_thresh: Seal detection pixel threshold
            seal_det_box_thresh: Seal detection box threshold
            seal_det_unclip_ratio: Seal detection expansion coefficient
            seal_rec_score_thresh: Seal recognition score threshold
            **kwargs: Additional arguments
            
        Returns:
            List of dictionaries containing visual_info and layout_parsing_result
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        params = {k: v for k, v in locals().items() if k not in ['self', 'input', 'kwargs'] and v is not None}
        params.update(kwargs)
        
        return self._paddle_instance.visual_predict(str(input), **params)
    
    def visual_predict_iter(
        self,
        input: Union[str, Path],
        **kwargs
    ):
        """Iterator version of visual_predict for memory-efficient processing.
        
        Args:
            input: Path to input file (image or PDF)
            **kwargs: Same arguments as visual_predict
            
        Yields:
            Dictionaries containing visual_info and layout_parsing_result
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.visual_predict_iter(str(input), **kwargs)
    
    def build_vector(
        self,
        visual_info: List[Dict[str, Any]],
        min_characters: int = 3500,
        block_size: int = 300,
        flag_save_bytes_vector: bool = False,
        retriever_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build vector embeddings for text content.
        
        Args:
            visual_info: List of visual information dictionaries from visual_predict
            min_characters: Minimum characters for chunking
            block_size: Size of text blocks for embedding
            flag_save_bytes_vector: Whether to save vectors as bytes
            retriever_config: Configuration for the retriever/embedding model
            
        Returns:
            Dictionary containing vector information
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.build_vector(
            visual_info=visual_info,
            min_characters=min_characters,
            block_size=block_size,
            flag_save_bytes_vector=flag_save_bytes_vector,
            retriever_config=retriever_config
        )
    
    def mllm_pred(
        self,
        input: Union[str, Path],
        key_list: List[str],
        mllm_chat_bot_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get predictions from multimodal large language model.
        
        Args:
            input: Path to input file
            key_list: List of keys to extract
            mllm_chat_bot_config: Configuration for multimodal LLM
            
        Returns:
            Dictionary containing MLLM predictions
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.mllm_pred(
            input=str(input),
            key_list=key_list,
            mllm_chat_bot_config=mllm_chat_bot_config
        )
    
    def chat(
        self,
        key_list: List[str],
        visual_info: List[Dict[str, Any]],
        use_vector_retrieval: bool = True,
        vector_info: Optional[Dict[str, Any]] = None,
        min_characters: int = 3500,
        text_task_description: Optional[str] = None,
        text_output_format: Optional[str] = None,
        text_rules_str: Optional[str] = None,
        text_few_shot_demo_text_content: Optional[str] = None,
        text_few_shot_demo_key_value_list: Optional[List] = None,
        table_task_description: Optional[str] = None,
        table_output_format: Optional[str] = None,
        table_rules_str: Optional[str] = None,
        table_few_shot_demo_text_content: Optional[str] = None,
        table_few_shot_demo_key_value_list: Optional[List] = None,
        mllm_predict_info: Optional[Dict[str, Any]] = None,
        mllm_integration_strategy: str = "integration",
        chat_bot_config: Optional[Dict[str, Any]] = None,
        retriever_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract key information using chat-based approach.
        
        Args:
            key_list: List of keys to extract
            visual_info: Visual information from visual_predict
            use_vector_retrieval: Whether to use vector retrieval
            vector_info: Vector information from build_vector
            min_characters: Minimum characters for processing
            text_task_description: Task description for text extraction
            text_output_format: Output format for text
            text_rules_str: Rules for text extraction
            text_few_shot_demo_text_content: Few-shot demo text content
            text_few_shot_demo_key_value_list: Few-shot demo key-value pairs
            table_task_description: Task description for table extraction
            table_output_format: Output format for tables
            table_rules_str: Rules for table extraction
            table_few_shot_demo_text_content: Few-shot demo table content
            table_few_shot_demo_key_value_list: Few-shot demo table key-value pairs
            mllm_predict_info: MLLM prediction information
            mllm_integration_strategy: Strategy for integrating MLLM results
            chat_bot_config: Configuration for chat bot
            retriever_config: Configuration for retriever
            
        Returns:
            Dictionary containing extracted key-value pairs
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        params = {k: v for k, v in locals().items() if k not in ['self'] and v is not None}
        
        return self._paddle_instance.chat(**params)
    
    def save_vector(
        self,
        vector_info: Dict[str, Any],
        save_path: Union[str, Path],
        retriever_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save vector information to disk.
        
        Args:
            vector_info: Vector information from build_vector
            save_path: Path to save the vectors
            retriever_config: Configuration for retriever
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.save_vector(
            vector_info=vector_info,
            save_path=str(save_path),
            retriever_config=retriever_config
        )
    
    def load_vector(
        self,
        data_path: Union[str, Path],
        retriever_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load vector information from disk.
        
        Args:
            data_path: Path to load vectors from
            retriever_config: Configuration for retriever
            
        Returns:
            Dictionary containing vector information
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.load_vector(
            data_path=str(data_path),
            retriever_config=retriever_config
        )
    
    def save_visual_info_list(
        self,
        visual_info: List[Dict[str, Any]],
        save_path: Union[str, Path]
    ) -> None:
        """Save visual information list to disk.
        
        Args:
            visual_info: Visual information from visual_predict
            save_path: Path to save the visual info
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.save_visual_info_list(
            visual_info=visual_info,
            save_path=str(save_path)
        )
    
    def load_visual_info_list(
        self,
        data_path: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """Load visual information list from disk.
        
        Args:
            data_path: Path to load visual info from
            
        Returns:
            List of visual information dictionaries
        """
        if not self._paddle_instance:
            raise OCRError(
                "PPChatOCRv4 instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.load_visual_info_list(data_path=str(data_path))


class PaddleOCRVLProvider(BasePaddleOCRProvider):
    """PaddleOCR-VL Pipeline for Vision-Language document understanding.
    
    This provider leverages vision-language models for:
    - Complex document layout understanding
    - Chart and graph interpretation
    - Multi-modal content extraction
    - Format-preserving text extraction
    - Markdown output generation
    
    Example:
        >>> from upsonic import OCR
        >>> from upsonic.ocr.paddleocr import PaddleOCRVLProvider
        >>> 
        >>> ocr = OCR(PaddleOCRVLProvider,
        ...           use_layout_detection=True,
        ...           use_chart_recognition=True,
        ...           format_block_content=True,
        ...           vl_rec_backend='local')
        >>> markdown_text = ocr.get_text('research_paper.pdf')
        >>> print(markdown_text)
    """
    
    def __init__(
        self,
        config: Optional[Union[OCRConfig, PaddleOCRConfig]] = None,
        # Layout detection
        layout_detection_model_name: Optional[str] = None,
        layout_detection_model_dir: Optional[str] = None,
        layout_threshold: Optional[float] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[float] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        # Vision-Language recognition
        vl_rec_model_name: Optional[str] = None,
        vl_rec_model_dir: Optional[str] = None,
        vl_rec_backend: Optional[str] = None,
        vl_rec_server_url: Optional[str] = None,
        vl_rec_max_concurrency: Optional[int] = None,
        # Document preprocessing
        doc_orientation_classify_model_name: Optional[str] = None,
        doc_orientation_classify_model_dir: Optional[str] = None,
        doc_unwarping_model_name: Optional[str] = None,
        doc_unwarping_model_dir: Optional[str] = None,
        # Feature toggles
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_layout_detection: Optional[bool] = None,
        use_chart_recognition: Optional[bool] = None,
        format_block_content: Optional[bool] = None,
        # VLM parameters
        use_queues: Optional[bool] = None,
        prompt_label: Optional[str] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs
    ):
        """Initialize PaddleOCR-VL provider with vision-language capabilities.
        
        Args:
            config: Optional OCRConfig or PaddleOCRConfig object
            layout_detection_model_name: Name of the layout detection model
            layout_detection_model_dir: Path to the layout detection model
            layout_threshold: Score threshold for layout detection
            layout_nms: Whether to use NMS in layout detection
            layout_unclip_ratio: Expansion coefficient for layout detection
            layout_merge_bboxes_mode: Overlapping box filtering method
            vl_rec_model_name: Name of the VL recognition model
            vl_rec_model_dir: Path to the VL recognition model
            vl_rec_backend: Backend for VL recognition ('local', 'server', etc.)
            vl_rec_server_url: Server URL for VL recognition
            vl_rec_max_concurrency: Maximum concurrency for VLM requests
            doc_orientation_classify_model_name: Name of document orientation model
            doc_orientation_classify_model_dir: Path to document orientation model
            doc_unwarping_model_name: Name of document unwarping model
            doc_unwarping_model_dir: Path to document unwarping model
            use_doc_orientation_classify: Whether to classify document orientation
            use_doc_unwarping: Whether to unwarp documents
            use_layout_detection: Whether to detect layout
            use_chart_recognition: Whether to recognize charts
            format_block_content: Whether to format content as Markdown
            use_queues: Whether to use queues for async processing
            prompt_label: Prompt label for the VLM
            repetition_penalty: Repetition penalty for VLM sampling
            temperature: Temperature for VLM sampling
            top_p: Top-p parameter for VLM sampling
            min_pixels: Minimum pixels for image preprocessing
            max_pixels: Maximum pixels for image preprocessing
            **kwargs: Additional provider-specific arguments
        """
        local_kwargs = {
            k: v for k, v in locals().items() 
            if k not in ['self', 'config', 'kwargs'] and v is not None
        }
        local_kwargs.update(kwargs)
        
        super().__init__(config=config, **local_kwargs)
    
    def _initialize_paddle(self) -> None:
        """Initialize the PaddleOCR-VL instance."""
        try:
            from paddleocr import PaddleOCRVL
            from upsonic.utils.printing import ocr_loading, ocr_initialized
            
            # Show loading message
            backend = getattr(self.config, 'vl_rec_backend', 'local')
            
            features = []
            if getattr(self.config, 'use_layout_detection', None):
                features.append("layout detection")
            if getattr(self.config, 'use_chart_recognition', None):
                features.append("chart recognition")
            if getattr(self.config, 'format_block_content', None):
                features.append("markdown formatting")
            
            extra_info = {
                "Backend": backend,
                "Features": ", ".join(features) if features else "basic",
                "Note": "First run will download vision-language models (may be large)"
            }
            ocr_loading("PaddleOCR-VL", [getattr(self.config, 'lang', 'en')], extra_info)
            
            paddle_params = self._build_paddle_params(self.config, **self._paddle_kwargs)
            
            self._paddle_instance = PaddleOCRVL(**paddle_params)
            
            ocr_initialized("PaddleOCR-VL")
            
        except Exception as e:
            raise OCRError(
                f"Failed to initialize PaddleOCR-VL: {str(e)}",
                error_code="PADDLE_VL_INIT_FAILED",
                original_error=e
            )
    
    
    @property
    def name(self) -> str:
        """Get the provider name."""
        return "PaddleOCR-VL"
    
    def predict(self, input: Union[str, Path], **kwargs) -> List[Any]:
        """Predict using PaddleOCR-VL on input file.
        
        Args:
            input: Path to input file (image or PDF)
            **kwargs: Additional prediction parameters
            
        Returns:
            List of prediction results with markdown and visual info
        """
        if not self._paddle_instance:
            raise OCRError(
                "PaddleOCR-VL instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.predict(str(input), **kwargs)
    
    def concatenate_markdown_pages(self, markdown_list: List[Dict[str, Any]]) -> str:
        """Concatenate markdown from multiple pages.
        
        Args:
            markdown_list: List of markdown dictionaries from predict results
            
        Returns:
            Concatenated markdown text
        """
        if not self._paddle_instance:
            raise OCRError(
                "PaddleOCR-VL instance not initialized",
                error_code="PROVIDER_NOT_INITIALIZED"
            )
        
        return self._paddle_instance.concatenate_markdown_pages(markdown_list)


PaddleOCR = PaddleOCRProvider
PPStructureV3 = PPStructureV3Provider
PPChatOCRv4 = PPChatOCRv4Provider
PaddleOCRVL = PaddleOCRVLProvider
