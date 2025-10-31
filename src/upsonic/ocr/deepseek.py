from __future__ import annotations

from typing import List, Optional, Dict, Any
import os

from upsonic.ocr.base import OCRProvider, OCRConfig, OCRResult, OCRTextBlock
from upsonic.ocr.exceptions import OCRProviderError, OCRProcessingError

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
    
    # Try to import NGramPerReqLogitsProcessor (may not be available in all vLLM versions)
    try:
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
        _NGRAM_PROCESSOR_AVAILABLE = True
    except ImportError:
        NGramPerReqLogitsProcessor = None
        _NGRAM_PROCESSOR_AVAILABLE = False
except ImportError:
    LLM = None
    SamplingParams = None
    NGramPerReqLogitsProcessor = None
    _VLLM_AVAILABLE = False
    _NGRAM_PROCESSOR_AVAILABLE = False


class DeepSeekOCR(OCRProvider):
    """DeepSeek OCR provider using DeepSeek-OCR model with vLLM.
    
    This provider uses DeepSeek's specialized OCR model (deepseek-ai/DeepSeek-OCR)
    running locally via vLLM. It provides high-quality OCR with support for complex
    layouts and table structures.
    
    **Requirements:**
    - vLLM with DeepSeek-OCR support (may require specific vLLM version or custom build)
    - Additional dependencies: `pip install addict matplotlib`
    - Sufficient GPU memory to run the model (typically 16GB+ VRAM)
    
    **Note:** DeepSeek-OCR model architecture may not be supported in all vLLM versions.
    As of vLLM 0.11.0, the DeepseekOCRForCausalLM architecture is not in the standard
    supported list. You may need a custom vLLM build or an updated version.
    
    Example:
        >>> from upsonic.ocr.deepseek import DeepSeekOCR
        >>> ocr = DeepSeekOCR(rotation_fix=True)
        >>> text = ocr.get_text('document.png')
    """
    
    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        model_name: str = "deepseek-ai/DeepSeek-OCR",
        prompt: str = "<image>\nFree OCR.",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 30,
        window_size: int = 90,
        **kwargs
    ):
        """Initialize DeepSeek OCR provider.
        
        Args:
            config: OCRConfig object
            model_name: Model name or path to custom model. Can be:
                - HuggingFace model identifier (e.g., "deepseek-ai/DeepSeek-OCR")
                - Local path to a model directory
                - Any model identifier supported by vLLM
                Default: "deepseek-ai/DeepSeek-OCR"
            prompt: OCR prompt template (default: "<image>\nFree OCR.")
            temperature: Sampling temperature (default: 0.0 for deterministic output)
            max_tokens: Maximum tokens to generate (default: 8192)
            ngram_size: N-gram size for logits processor (default: 30)
            window_size: Window size for logits processor (default: 90)
            **kwargs: Additional configuration arguments
        """
        self.model_name = model_name
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ngram_size = ngram_size
        self.window_size = window_size
        self._llm = None
        self._sampling_params = None
        super().__init__(config, **kwargs)
    
    @property
    def name(self) -> str:
        return "deepseek_ocr"
    
    @property
    def supported_languages(self) -> List[str]:
        """DeepSeek-OCR supports multiple languages."""
        return [
            'en', 'zh', 'ja', 'ko', 'es', 'fr', 'de', 'it', 'pt', 'ru',
            'ar', 'hi', 'th', 'vi', 'id', 'ms', 'tr', 'pl', 'nl', 'uk'
        ]
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are installed."""
        if not _VLLM_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="vllm",
                install_command='pip install vllm',
                feature_name="DeepSeek OCR provider"
            )
        
        missing_deps = []
        try:
            import addict
        except ImportError:
            missing_deps.append('addict')
        
        try:
            import matplotlib
        except ImportError:
            missing_deps.append('matplotlib')
        
        if missing_deps:
            from upsonic.utils.printing import import_error
            import_error(
                package_name=', '.join(missing_deps),
                install_command=f'pip install {" ".join(missing_deps)}',
                feature_name="DeepSeek OCR provider (additional dependencies)"
            )
    
    def _get_llm(self) -> LLM:
        """Get or create vLLM instance."""
        if self._llm is None:
            if not _VLLM_AVAILABLE:
                raise OCRProviderError(
                    "vLLM is not available. Please install vLLM: pip install vllm",
                    error_code="VLLM_NOT_AVAILABLE"
                )
            
            from upsonic.utils.printing import ocr_language_warning, ocr_loading, ocr_initialized
            
            unsupported_langs = [lang for lang in self.config.languages if lang not in self.supported_languages]
            if unsupported_langs:
                ocr_language_warning(
                    provider_name="DeepSeek-OCR",
                    warning_langs=unsupported_langs,
                    best_supported=self.supported_languages
                )
            
            extra_info = {
                "Model": self.model_name,
                "Temperature": str(self.temperature),
                "Max tokens": str(self.max_tokens),
                "Note": "This may take a few minutes on first run"
            }
            ocr_loading("DeepSeek-OCR", self.config.languages, extra_info)
            
            try:
                llm_kwargs = {
                    'model': self.model_name,
                    'enable_prefix_caching': False,
                    'mm_processor_cache_gb': 0,
                }
                
                # Add logits processors only if available
                if _NGRAM_PROCESSOR_AVAILABLE and NGramPerReqLogitsProcessor is not None:
                    llm_kwargs['logits_processors'] = [NGramPerReqLogitsProcessor]
                
                self._llm = LLM(**llm_kwargs)
                ocr_initialized("DeepSeek-OCR")
            except Exception as e:
                error_msg = str(e)
                
                if 'not supported for now' in error_msg or 'DeepseekOCRForCausalLM' in error_msg:
                    raise OCRProviderError(
                        f"[UNSUPPORTED_MODEL_ARCHITECTURE] DeepSeek-OCR model architecture is not supported in your vLLM version. "
                        f"The '{self.model_name}' model requires either:\n"
                        f"  1. A newer version of vLLM that supports DeepseekOCRForCausalLM\n"
                        f"  2. A custom vLLM build with DeepSeek-OCR support\n"
                        f"  3. Use an alternative model or OCR provider (RapidOCR, EasyOCR, Tesseract)\n"
                        f"Original error: {error_msg}",
                        error_code="UNSUPPORTED_MODEL_ARCHITECTURE",
                        original_error=e
                    )
                
                raise OCRProviderError(
                    f"Failed to initialize DeepSeek-OCR model: {error_msg}",
                    error_code="MODEL_INIT_FAILED",
                    original_error=e
                )
        return self._llm
    
    def _get_sampling_params(self) -> SamplingParams:
        """Get or create sampling parameters."""
        if self._sampling_params is None:
            # Build sampling params
            params_kwargs = {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'skip_special_tokens': False,
            }
            
            # Add ngram parameters only if the processor is available
            if _NGRAM_PROCESSOR_AVAILABLE:
                params_kwargs['extra_args'] = dict(
                    ngram_size=self.ngram_size,
                    window_size=self.window_size,
                    whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
                )
            
            self._sampling_params = SamplingParams(**params_kwargs)
        return self._sampling_params
    
    def _process_image(self, image, **kwargs) -> OCRResult:
        """Process a single image with DeepSeek-OCR model.
        
        Args:
            image: PIL Image object
            **kwargs: Additional arguments (prompt customization, etc.)
            
        Returns:
            OCRResult object
        """
        try:
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            prompt = kwargs.get('prompt', self.prompt)
            
            model_input = {
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            }
            
            llm = self._get_llm()
            sampling_params = self._get_sampling_params()
            
            model_outputs = llm.generate([model_input], sampling_params)
            
            extracted_text = model_outputs[0].outputs[0].text.strip()
            
            block = OCRTextBlock(
                text=extracted_text,
                confidence=1.0,  # DeepSeek-OCR doesn't provide confidence scores
                bbox=None,
                language=None
            )
            
            return OCRResult(
                text=extracted_text,
                blocks=[block],
                confidence=1.0,
                page_count=1,
                provider=self.name,
                metadata={
                    'model': self.model_name,
                    'prompt': prompt,
                }
            )
            
        except Exception as e:
            if isinstance(e, OCRProviderError):
                raise
            raise OCRProcessingError(
                f"DeepSeek OCR processing failed: {str(e)}",
                error_code="DEEPSEEK_PROCESSING_FAILED",
                original_error=e
            )
    
    def process_images_batch(self, images: List, **kwargs) -> List[OCRResult]:
        """Process multiple images in a batch for better performance.
        
        Args:
            images: List of PIL Image objects
            **kwargs: Additional arguments
            
        Returns:
            List of OCRResult objects
        """
        if not images:
            return []
        
        try:
            processed_images = []
            for img in images:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                processed_images.append(img)
            
            prompt = kwargs.get('prompt', self.prompt)
            
            model_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": img}
                }
                for img in processed_images
            ]
            
            llm = self._get_llm()
            sampling_params = self._get_sampling_params()
            
            model_outputs = llm.generate(model_inputs, sampling_params)
            
            results = []
            for i, output in enumerate(model_outputs):
                extracted_text = output.outputs[0].text.strip()
                
                block = OCRTextBlock(
                    text=extracted_text,
                    confidence=1.0,
                    bbox=None,
                    language=None,
                    page_number=i + 1
                )
                
                result = OCRResult(
                    text=extracted_text,
                    blocks=[block],
                    confidence=1.0,
                    page_count=1,
                    provider=self.name,
                    metadata={
                        'model': self.model_name,
                        'prompt': prompt,
                        'batch_index': i,
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            if isinstance(e, OCRProviderError):
                raise
            raise OCRProcessingError(
                f"DeepSeek OCR batch processing failed: {str(e)}",
                error_code="DEEPSEEK_BATCH_PROCESSING_FAILED",
                original_error=e
            )
    
    def process_file(self, file_path, **kwargs):
        """Process a file with optimized batch processing for PDFs.
        
        This override provides efficient batch processing when dealing with
        multi-page PDFs, processing all pages in a single batch.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments
            
        Returns:
            OCRResult object
        """
        from upsonic.ocr.utils import prepare_file_for_ocr
        import time
        
        start_time = time.time()
        
        processing_config = {
            'rotation_fix': kwargs.get('rotation_fix', self.config.rotation_fix),
            'enhance_contrast': kwargs.get('enhance_contrast', self.config.enhance_contrast),
            'remove_noise': kwargs.get('remove_noise', self.config.remove_noise),
            'pdf_dpi': kwargs.get('pdf_dpi', self.config.pdf_dpi),
        }
        
        images = prepare_file_for_ocr(file_path, **processing_config)
        
        if len(images) > 1:
            page_results = self.process_images_batch(images, **kwargs)
            
            all_blocks = []
            all_text_parts = []
            total_confidence = 0.0
            
            for page_num, result in enumerate(page_results, start=1):
                for block in result.blocks:
                    block.page_number = page_num
                    all_blocks.append(block)
                
                all_text_parts.append(result.text)
                total_confidence += result.confidence
            
            combined_text = "\n\n".join(all_text_parts)
            avg_confidence = total_confidence / len(page_results) if page_results else 0.0
            
        else:
            result = self._process_image(images[0], **kwargs)
            combined_text = result.text
            all_blocks = result.blocks
            avg_confidence = result.confidence
        
        processing_time = (time.time() - start_time) * 1000
        
        final_result = OCRResult(
            text=combined_text,
            blocks=all_blocks,
            confidence=avg_confidence,
            page_count=len(images),
            processing_time_ms=processing_time,
            provider=self.name,
            metadata={
                'file_path': str(file_path),
                'model': self.model_name,
                'batch_processed': len(images) > 1,
                'config': self.config.model_dump(),
            }
        )
        
        self._metrics.total_pages += len(images)
        self._metrics.total_characters += len(combined_text)
        self._metrics.processing_time_ms += processing_time
        self._metrics.files_processed += 1
        if all_blocks:
            self._metrics.average_confidence = sum(b.confidence for b in all_blocks) / len(all_blocks)
        
        return final_result

