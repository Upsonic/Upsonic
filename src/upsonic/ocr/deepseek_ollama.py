from __future__ import annotations

from typing import List, Optional
from pathlib import Path
import tempfile

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
        prompt: str = '<image>\nFree OCR.',
        **kwargs
    ):
        """Initialize DeepSeek Ollama OCR provider.
        
        Args:
            config: OCRConfig object
            host: Ollama server host (default: 'http://localhost:11434')
            model: Model name (default: 'deepseek-ocr:3b')
            prompt: OCR prompt template (default: '<image>\nFree OCR.')
            **kwargs: Additional configuration arguments
        """
        self.host = host
        self.model = model
        self.prompt = prompt
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
                client = self._get_client()
                
                response = client.chat(
                    model=self.model,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [tmp_path],
                        }
                    ],
                )
                
                extracted_text = response.message.content.strip() if response.message.content else ""
                
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
                        'model': self.model,
                        'host': self.host,
                        'backend': 'ollama',
                        'prompt': prompt,
                    }
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

