from __future__ import annotations
import asyncio
import json
from typing import List, Dict, Any, Optional
import time

try:
    import aiohttp
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

from pydantic import Field, field_validator
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode
from ..utils.package.exception import ConfigurationError, ModelConnectionError
from upsonic.utils.printing import info_log, debug_log, warning_log


class OllamaEmbeddingConfig(EmbeddingConfig):
    """Configuration for Ollama embedding models."""
    
    base_url: str = Field("http://localhost:11434", description="Ollama server URL")
    
    model_name: str = Field("nomic-embed-text", description="Ollama embedding model name")
    
    auto_pull_model: bool = Field(True, description="Automatically pull model if not available")
    keep_alive: Optional[str] = Field("5m", description="Keep model loaded for duration")
    
    temperature: Optional[float] = Field(None, description="Model temperature")
    top_p: Optional[float] = Field(None, description="Top-p sampling")
    num_ctx: Optional[int] = Field(None, description="Context window size")
    
    request_timeout: float = Field(120.0, description="Request timeout in seconds")
    connection_timeout: float = Field(10.0, description="Connection timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    
    enable_keep_alive: bool = Field(True, description="Keep model loaded between requests")
    enable_model_preload: bool = Field(True, description="Preload model on startup")
    
    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v):
        """Validate Ollama base URL format."""
        if not v.startswith(('http://', 'https://')):
            v = f"http://{v}"
        
        v = v.rstrip('/')
        
        return v
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate and suggest Ollama model names."""
        known_models = [
            "nomic-embed-text",
            "mxbai-embed-large",
            "snowflake-arctic-embed",
            "all-minilm",
            "bge-large",
            "bge-m3"
        ]
        
        if v in known_models:
            return v
        
        if ':' in v:
            return v
        
        warning_log(f"'{v}' may not be a standard Ollama embedding model. Known models: {known_models}", context="OllamaEmbedding")
        return v


class OllamaEmbedding(EmbeddingProvider):
    
    config: OllamaEmbeddingConfig
    
    def __init__(self, config: Optional[OllamaEmbeddingConfig] = None, **kwargs):
        if not HTTP_AVAILABLE:
            raise ConfigurationError(
                "HTTP libraries not found. Install with: pip install aiohttp requests",
                error_code="DEPENDENCY_MISSING"
            )
        
        if config is None:
            config = OllamaEmbeddingConfig(**kwargs)
        
        super().__init__(config=config)
        
        self._setup_http_session()
        
        self._model_info: Optional[Dict[str, Any]] = None
        
        self._last_health_check = 0
        self._health_check_interval = 60
        
        self._initialize_sync()
    
    def _setup_http_session(self):
        """Setup HTTP session for requests."""
        self.session = requests.Session()
        
        self.session.timeout = (self.config.connection_timeout, self.config.request_timeout)
        
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _initialize_sync(self):
        """Initialize Ollama connection and model synchronously."""
        try:
            if self._check_ollama_health_sync():
                info_log("Connected to Ollama server", context="OllamaEmbedding")
                
                if self.config.auto_pull_model:
                    self._ensure_model_available_sync()
                
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Ollama initialization failed: {e}", "OllamaProvider")
    
    async def _initialize_async(self):
        """Initialize Ollama connection and model asynchronously."""
        try:
            await self._check_ollama_health()
            
            if self.config.auto_pull_model:
                await self._ensure_model_available()
            
            if self.config.enable_model_preload:
                await self._preload_model()
                
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Ollama initialization failed: {e}", "OllamaProvider")
    
    def _check_ollama_health_sync(self) -> bool:
        """Check if Ollama server is healthy synchronously."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/version",
                timeout=self.config.connection_timeout
            )
            if response.status_code == 200:
                version_info = response.json()
                from upsonic.utils.printing import connection_info
                connection_info("Ollama", version_info.get('version', 'unknown'))
                return True
            else:
                debug_log(f"Ollama health check failed with status: {response.status_code}", context="OllamaEmbedding")
                return False
                
        except Exception as e:
            debug_log(f"Ollama server not accessible: {e}", context="OllamaEmbedding")
            return False
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama server is healthy asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/api/version",
                    timeout=aiohttp.ClientTimeout(total=self.config.connection_timeout)
                ) as response:
                    if response.status == 200:
                        version_info = await response.json()
                        from upsonic.utils.printing import connection_info
                        connection_info("Ollama", version_info.get('version', 'unknown'))
                        return True
                    else:
                        from upsonic.utils.printing import error_log
                        error_log(f"Ollama health check failed with status: {response.status}", "OllamaProvider")
                        return False
                        
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Ollama server not accessible: {e}", "OllamaProvider")
            return False
    
    def _ensure_model_available_sync(self):
        """Ensure the embedding model is available synchronously."""
        try:
            models = self._list_models_sync()
            model_names = [model.get('name', '') for model in models]
            
            model_found = False
            for model_name in model_names:
                if self.config.model_name in model_name or model_name.startswith(self.config.model_name):
                    model_found = True
                    from upsonic.utils.printing import success_log
                    success_log(f"Model {self.config.model_name} found: {model_name}", "OllamaProvider")
                    break
            
            if not model_found:
                from upsonic.utils.printing import info_log
                info_log(f"Model {self.config.model_name} not found, pulling...", "OllamaProvider")
                self._pull_model_sync()
            
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Error checking/pulling model: {e}", "OllamaProvider")
    
    async def _ensure_model_available(self):
        """Ensure the embedding model is available asynchronously."""
        try:
            models = await self._list_models()
            model_names = [model.get('name', '') for model in models]
            
            model_found = False
            for model_name in model_names:
                if self.config.model_name in model_name or model_name.startswith(self.config.model_name):
                    model_found = True
                    from upsonic.utils.printing import success_log
                    success_log(f"Model {self.config.model_name} found: {model_name}", "OllamaProvider")
                    break
            
            if not model_found:
                from upsonic.utils.printing import info_log
                info_log(f"Model {self.config.model_name} not found, pulling...", "OllamaProvider")
                await self._pull_model()
            
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Error checking/pulling model: {e}", "OllamaProvider")
    
    def _list_models_sync(self) -> List[Dict[str, Any]]:
        """List available models in Ollama synchronously."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.request_timeout
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            else:
                return []
                
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Error listing models: {e}", "OllamaProvider")
            return []
    
    async def _list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('models', [])
                    else:
                        return []
                        
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Error listing models: {e}", "OllamaProvider")
            return []
    
    def _pull_model_sync(self):
        """Pull the embedding model synchronously."""
        try:
            pull_payload = {"name": self.config.model_name}
            
            response = self.session.post(
                f"{self.config.base_url}/api/pull",
                json=pull_payload,
                timeout=600
            )
            
            if response.status_code == 200:
                info_log(f"Successfully pulled model: {self.config.model_name}", context="OllamaEmbedding")
            else:
                raise ModelConnectionError(f"Failed to pull model: HTTP {response.status_code}")
                
        except Exception as e:
            raise ModelConnectionError(
                f"Model pull failed: {str(e)}",
                error_code="OLLAMA_PULL_ERROR",
                original_error=e
            )
    
    async def _pull_model(self):
        """Pull the embedding model asynchronously."""
        try:
            pull_payload = {"name": self.config.model_name}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/pull",
                    json=pull_payload,
                    timeout=aiohttp.ClientTimeout(total=600)
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    progress = json.loads(line.decode('utf-8'))
                                    if 'status' in progress:
                                        debug_log(f"Pull progress: {progress['status']}", context="OllamaEmbedding")
                                        if progress.get('status') == 'success':
                                            info_log(f"Successfully pulled model: {self.config.model_name}", context="OllamaEmbedding")
                                            break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        raise ModelConnectionError(f"Failed to pull model: HTTP {response.status}")
                        
        except Exception as e:
            raise ModelConnectionError(
                f"Model pull failed: {str(e)}",
                error_code="OLLAMA_PULL_ERROR",
                original_error=e
            )
    
    async def _preload_model(self):
        """Preload the model to keep it in memory."""
        try:
            await self._embed_batch(["preload"], EmbeddingMode.QUERY)
            info_log(f"Model {self.config.model_name} preloaded successfully", context="OllamaEmbedding")
            
        except Exception as e:
            debug_log(f"Model preload failed: {e}", context="OllamaEmbedding")
    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """Ollama supports document and query modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY, EmbeddingMode.SYMMETRIC]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get Ollama pricing info (local execution is free)."""
        return {
            "per_million_tokens": 0.0,
            "currency": "USD",
            "note": "Local Ollama execution - no API costs",
            "compute_cost": "Uses local compute resources"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Ollama model."""
        if self._model_info is None:
            self._model_info = {
                "model_name": self.config.model_name,
                "provider": "Ollama",
                "type": "embedding",
                "execution_mode": "Local",
                "base_url": self.config.base_url,
                "keep_alive": self.config.keep_alive,
                "auto_pull": self.config.auto_pull_model
            }
            
            try:
                pass
            except Exception:
                pass
        
        return self._model_info
    
    async def _make_embedding_request(self, texts: List[str]) -> Dict[str, Any]:
        """Make embedding request to Ollama API."""
        payload = {
            "model": self.config.model_name,
            "prompt": texts[0] if len(texts) == 1 else texts,
        }
        
        if self.config.keep_alive is not None:
            payload["keep_alive"] = self.config.keep_alive
        
        options = {}
        if self.config.temperature is not None:
            options["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            options["top_p"] = self.config.top_p
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        
        if options:
            payload["options"] = options
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.base_url}/api/embeddings",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise ModelConnectionError(
                        f"Ollama API error (HTTP {response.status}): {error_text}",
                        error_code=f"OLLAMA_HTTP_{response.status}"
                    )
    
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """
        Embed a batch of texts using Ollama.
        
        Args:
            texts: List of text strings to embed
            mode: Embedding mode
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            if not await self._check_ollama_health():
                raise ModelConnectionError(
                    "Ollama server is not accessible",
                    error_code="OLLAMA_HEALTH_CHECK_FAILED"
                )
            self._last_health_check = current_time
        
        try:
            all_embeddings = []
            
            for text in texts:
                response = await self._make_embedding_request([text])
                
                if "embedding" in response:
                    all_embeddings.append(response["embedding"])
                elif "embeddings" in response:
                    all_embeddings.extend(response["embeddings"])
                else:
                    raise ModelConnectionError("Unexpected response format from Ollama")
                
                await asyncio.sleep(0.01)
            
            return all_embeddings
            
        except Exception as e:
            if "connection" in str(e).lower():
                raise ModelConnectionError(
                    f"Failed to connect to Ollama server at {self.config.base_url}: {str(e)}",
                    error_code="OLLAMA_CONNECTION_ERROR",
                    original_error=e
                )
            else:
                raise ModelConnectionError(
                    f"Ollama embedding failed: {str(e)}",
                    error_code="OLLAMA_EMBEDDING_ERROR",
                    original_error=e
                )
    
    async def validate_connection(self) -> bool:
        """Validate Ollama connection and model access."""
        try:
            if not await self._check_ollama_health():
                return False
            
            test_result = await self._embed_batch(["test connection"], EmbeddingMode.QUERY)
            return len(test_result) > 0 and len(test_result[0]) > 0
            
        except Exception as e:
            debug_log(f"Ollama connection validation failed: {str(e)}", context="OllamaEmbedding")
            return False
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get Ollama server information."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.base_url}/api/version") as response:
                    version_info = await response.json() if response.status == 200 else {}
                
                models = await self._list_models()
                
                return {
                    "server_url": self.config.base_url,
                    "version": version_info.get("version", "unknown"),
                    "available_models": len(models),
                    "models": [model.get("name") for model in models],
                    "target_model": self.config.model_name,
                    "model_loaded": any(
                        self.config.model_name in model.get("name", "") 
                        for model in models
                    )
                }
                
        except Exception as e:
            return {"error": f"Failed to get server info: {e}"}
    
    async def get_model_details(self) -> Dict[str, Any]:
        """Get detailed information about the current model."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/show",
                    json={"name": self.config.model_name}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"Model not found: {self.config.model_name}"}
                        
        except Exception as e:
            return {"error": f"Failed to get model details: {e}"}

    async def close(self):
        """
        Clean up Ollama HTTP sessions and connections.
        """
        if hasattr(self, 'session') and self.session:
            try:
                if hasattr(self.session, 'aclose'):
                    await self.session.aclose()
                elif hasattr(self.session, 'close'):
                    self.session.close()
                
                del self.session
                self.session = None
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Error closing Ollama HTTP session: {e}", "OllamaProvider")
    
    def __del__(self):
        """
        Destructor to ensure session is closed when object is garbage collected.
        """
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
            except Exception:
                pass  # Ignore errors during garbage collection
        
        if hasattr(self, '_model_info'):
            self._model_info = None


def create_nomic_embedding(**kwargs) -> OllamaEmbedding:
    """Create Nomic Embed Text embedding provider."""
    config = OllamaEmbeddingConfig(
        model_name="nomic-embed-text",
        **kwargs
    )
    return OllamaEmbedding(config=config)


def create_mxbai_embedding(**kwargs) -> OllamaEmbedding:
    """Create MXBAI Large embedding provider."""
    config = OllamaEmbeddingConfig(
        model_name="mxbai-embed-large",
        **kwargs
    )
    return OllamaEmbedding(config=config)


def create_arctic_embedding(**kwargs) -> OllamaEmbedding:
    """Create Snowflake Arctic embedding provider."""
    config = OllamaEmbeddingConfig(
        model_name="snowflake-arctic-embed",
        **kwargs
    )
    return OllamaEmbedding(config=config)


def create_ollama_embedding_with_auto_setup(
    model_name: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    **kwargs
) -> OllamaEmbedding:
    """Create Ollama embedding with automatic model setup."""

    config_kwargs = {
        "model_name": model_name,
        "base_url": base_url,
        "auto_pull_model": kwargs.pop('auto_pull_model', True),
        "enable_model_preload": kwargs.pop('enable_model_preload', True)
    }

    config_kwargs.update(kwargs)
    
    config = OllamaEmbeddingConfig(**config_kwargs)
    return OllamaEmbedding(config=config)
