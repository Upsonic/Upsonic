from __future__ import annotations

from typing import List, Optional
from pathlib import Path
import tempfile
import time
import threading

from upsonic.ocr.base import OCRProvider, OCRConfig, OCRResult, OCRTextBlock
from upsonic.ocr.exceptions import OCRProviderError, OCRProcessingError

try:
    from ollama import Client
    _OLLAMA_AVAILABLE = True
except ImportError:
    Client = None
    _OLLAMA_AVAILABLE = False


class DeepSeekOllamaOCR(OCRProvider):
    """DeepSeek OCR provider using Ollama with deepseek-ocr model.
    
    This provider uses DeepSeek's OCR model running locally via Ollama.
    It provides high-quality OCR with support for complex layouts.
    
    **Requirements:**
    - Ollama installed and running locally
    - DeepSeek OCR model: `ollama pull deepseek-ocr:3b`
    - Python ollama package: `pip install ollama`
    
    Example:
        >>> from upsonic.ocr.deepseek_ollama import DeepSeekOllamaOCR
        >>> ocr = DeepSeekOllamaOCR(rotation_fix=True)
        >>> text = ocr.get_text('document.png')
    """
    
    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        host: str = 'http://localhost:11434',
        model: str = 'deepseek-ocr:3b',
        prompt: str = r"\nFree OCR.",
        timeout: float = 60.0,
        **kwargs
    ):
        """Initialize DeepSeek Ollama OCR provider.
        
        Args:
            config: OCRConfig object
            host: Ollama server host (default: 'http://localhost:11434')
            model: Model name (default: 'deepseek-ocr:3b')
            prompt: OCR prompt template (default: r"\nFree OCR.")
            timeout: Timeout in seconds for streaming response (default: 60.0)
            **kwargs: Additional configuration arguments
        """
        self.host = host
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self._client = None
        super().__init__(config, **kwargs)
    
    @property
    def name(self) -> str:
        return "deepseek_ollama_ocr"
    
    @property
    def supported_languages(self) -> List[str]:
        """DeepSeek-OCR supports multiple languages."""
        return [
            'en', 'zh', 'ja', 'ko', 'es', 'fr', 'de', 'it', 'pt', 'ru',
            'ar', 'hi', 'th', 'vi', 'id', 'ms', 'tr', 'pl', 'nl', 'uk'
        ]
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are installed."""
        if not _OLLAMA_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="ollama",
                install_command='pip install ollama',
                feature_name="DeepSeek Ollama OCR provider"
            )
    
    def _get_client(self) -> Client:
        """Get or create Ollama client instance."""
        if self._client is None:
            if not _OLLAMA_AVAILABLE:
                raise OCRProviderError(
                    "Ollama is not available. Please install ollama: pip install ollama",
                    error_code="OLLAMA_NOT_AVAILABLE"
                )
            
            from upsonic.utils.printing import ocr_language_warning, ocr_loading, ocr_initialized
            
            unsupported_langs = [lang for lang in self.config.languages if lang not in self.supported_languages]
            if unsupported_langs:
                ocr_language_warning(
                    provider_name="DeepSeek-OCR (Ollama)",
                    warning_langs=unsupported_langs,
                    best_supported=self.supported_languages
                )
            
            extra_info = {
                "Model": self.model,
                "Host": self.host,
                "Backend": "Ollama",
                "Note": "Make sure Ollama is running and the model is pulled"
            }
            ocr_loading("DeepSeek-OCR (Ollama)", self.config.languages, extra_info)
            
            try:
                self._client = Client(host=self.host)
                ocr_initialized("DeepSeek-OCR (Ollama)")
            except Exception as e:
                raise OCRProviderError(
                    f"Failed to initialize Ollama client: {str(e)}. "
                    f"Make sure Ollama is running at {self.host}",
                    error_code="CLIENT_INIT_FAILED",
                    original_error=e
                )
        return self._client
    
    def _process_image(self, image, **kwargs) -> OCRResult:
        """Process a single image with DeepSeek-OCR model via Ollama.
        
        Args:
            image: PIL Image object
            **kwargs: Additional arguments (prompt customization, etc.)
            
        Returns:
            OCRResult object
        """
        try:
            # Convert image to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Save image to temporary file since Ollama needs file path
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, format='PNG')
                tmp_path = tmp_file.name
            
            try:
                prompt = kwargs.get('prompt', self.prompt)
                timeout = kwargs.get('timeout', self.timeout)
                client = self._get_client()
                
                # Variables for timeout control
                result_data = {
                    'text': '',
                    'timeout_reached': False,
                    'processing_time': 0.0,
                    'chunk_count': 0
                }
                start_time = time.time()
                stop_flag = threading.Event()
                
                def stream_with_timeout():
                    """Stream response in a separate thread"""
                    try:
                        stream = client.chat(
                            model=self.model,
                            messages=[
                                {
                                    'role': 'user',
                                    'content': prompt,
                                    'images': [tmp_path],
                                }
                            ],
                            stream=True,
                        )
                        
                        for chunk in stream:
                            result_data['chunk_count'] += 1
                            
                            # Check timeout in thread
                            current_time = time.time()
                            elapsed = current_time - start_time
                            
                            if elapsed > timeout:
                                result_data['timeout_reached'] = True
                                break
                            
                            # Check if we should stop from outside
                            if stop_flag.is_set():
                                result_data['timeout_reached'] = True
                                break
                            
                            # Add chunk content
                            if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                                if chunk.message.content:
                                    result_data['text'] += chunk.message.content
                    except Exception:
                        pass
                    finally:
                        result_data['processing_time'] = time.time() - start_time
                
                # Start streaming thread
                stream_thread = threading.Thread(target=stream_with_timeout, daemon=True)
                stream_thread.start()
                
                # Wait with timeout
                stream_thread.join(timeout=timeout + 0.5)
                
                # If thread is still alive, force stop
                if stream_thread.is_alive():
                    stop_flag.set()
                    result_data['timeout_reached'] = True
                    # Wait a bit more for graceful shutdown
                    stream_thread.join(timeout=0.5)
                
                # Ensure processing_time is set
                if result_data['processing_time'] == 0.0:
                    result_data['processing_time'] = time.time() - start_time
                
                extracted_text = result_data['text'].strip()
                
                block = OCRTextBlock(
                    text=extracted_text,
                    confidence=1.0,
                    bbox=None,
                    language=None
                )
                
                metadata = {
                    'model': self.model,
                    'host': self.host,
                    'backend': 'ollama',
                    'prompt': prompt,
                    'streaming': True,
                    'timeout_reached': result_data['timeout_reached'],
                    'timeout_duration': timeout,
                    'processing_time': result_data['processing_time'],
                    'chunk_count': result_data['chunk_count'],
                }
                
                return OCRResult(
                    text=extracted_text,
                    blocks=[block],
                    confidence=1.0,
                    page_count=1,
                    provider=self.name,
                    metadata=metadata
                )
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            
        except Exception as e:
            if isinstance(e, OCRProviderError):
                raise
            raise OCRProcessingError(
                f"DeepSeek Ollama OCR processing failed: {str(e)}",
                error_code="DEEPSEEK_OLLAMA_PROCESSING_FAILED",
                original_error=e
            )
    
    def process_file(self, file_path, **kwargs):
        """Process a file and preserve timeout metadata.
        
        Override base class to preserve streaming and timeout metadata.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments (including timeout)
            
        Returns:
            OCRResult object with timeout metadata
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
        
        # Process first image (single page for now)
        if images:
            result = self._process_image(images[0], **kwargs)
            
            # Update metrics
            self._metrics.total_pages += 1
            self._metrics.total_characters += len(result.text)
            self._metrics.processing_time_ms += result.metadata.get('processing_time', 0) * 1000
            self._metrics.files_processed += 1
            if result.blocks:
                self._metrics.average_confidence = sum(b.confidence for b in result.blocks) / len(result.blocks)
            
            # Add file_path to metadata
            result.metadata['file_path'] = str(file_path)
            result.metadata['config'] = self.config.model_dump()
            
            return result
        
        # Empty result if no images
        return OCRResult(
            text="",
            blocks=[],
            confidence=0.0,
            page_count=0,
            provider=self.name,
            metadata={'file_path': str(file_path), 'error': 'No images found'}
        )

