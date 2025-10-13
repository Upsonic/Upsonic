from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Iterator
import numpy as np
import asyncio

try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    from fastembed.common import OnnxProvider
    import onnxruntime as ort
    FASTEMBED_AVAILABLE = True
except ImportError:
    TextEmbedding = None
    SparseTextEmbedding = None
    OnnxProvider = None
    ort = None
    FASTEMBED_AVAILABLE = False

from pydantic import Field, field_validator
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode
from ..utils.package.exception import ConfigurationError, ModelConnectionError
from upsonic.utils.printing import info_log, debug_log


class FastEmbedConfig(EmbeddingConfig):
    """Configuration for FastEmbed models."""
    
    model_name: str = Field("BAAI/bge-small-en-v1.5", description="FastEmbed model name")
    
    cache_dir: Optional[str] = Field(None, description="Model cache directory")
    threads: Optional[int] = Field(None, description="Number of threads (auto-detected if None)")
    
    providers: List[str] = Field(
        default_factory=lambda: ["CPUExecutionProvider"], 
        description="ONNX execution providers"
    )
    
    enable_gpu: bool = Field(False, description="Enable GPU acceleration if available")
    enable_parallel_processing: bool = Field(True, description="Enable parallel text processing")
    doc_embed_type: str = Field("default", description="Document embedding type (default, passage)")
    
    max_memory_mb: Optional[int] = Field(None, description="Maximum memory usage in MB")
    model_warmup: bool = Field(True, description="Warm up model on initialization")
    
    enable_sparse_embeddings: bool = Field(False, description="Use sparse embeddings for better performance")
    sparse_model_name: Optional[str] = Field(None, description="Sparse model name if different from dense")
    
    @field_validator('providers')
    @classmethod
    def validate_providers(cls, v):
        """Validate ONNX providers."""
        valid_providers = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider", 
            "ROCMExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider"
        ]
        
        for provider in v:
            if provider not in valid_providers:
                raise ValueError(f"Invalid ONNX provider: {provider}. Valid options: {valid_providers}")
        
        return v
    
    @field_validator('doc_embed_type')
    @classmethod
    def validate_embed_type(cls, v):
        """Validate document embedding type."""
        valid_types = ["default", "passage"]
        if v not in valid_types:
            raise ValueError(f"Invalid doc_embed_type: {v}. Valid options: {valid_types}")
        return v


class FastEmbedProvider(EmbeddingProvider):
    
    config: FastEmbedConfig
    
    def __init__(self, config: Optional[FastEmbedConfig] = None, **kwargs):
        if not FASTEMBED_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="fastembed",
                install_command='pip install fastembed',
                feature_name="fastembed provider"
            )

        if config is None:
            config = FastEmbedConfig(**kwargs)

        super().__init__(config=config)
        
        self._setup_providers()
        
        self._initialize_models()
        
        self._model_info: Optional[Dict[str, Any]] = None
    
    def _setup_providers(self):
        """Setup ONNX execution providers based on configuration."""
        self.providers = self.config.providers.copy()
        
        if self.config.enable_gpu:
            try:
                available_providers = ort.get_available_providers()
                
                if "CUDAExecutionProvider" in available_providers and "CUDAExecutionProvider" not in self.providers:
                    self.providers.insert(0, "CUDAExecutionProvider")
                    debug_log("CUDA provider available and enabled", context="FastEmbed")
                elif "ROCMExecutionProvider" in available_providers and "ROCMExecutionProvider" not in self.providers:
                    self.providers.insert(0, "ROCMExecutionProvider")
                    debug_log("ROCm provider available and enabled", context="FastEmbed")
                elif "CoreMLExecutionProvider" in available_providers and "CoreMLExecutionProvider" not in self.providers:
                    self.providers.insert(0, "CoreMLExecutionProvider")
                    debug_log("CoreML provider available and enabled", context="FastEmbed")
                else:
                    debug_log("GPU requested but no GPU providers available, using CPU", context="FastEmbed")
                    
            except ImportError:
                debug_log("ONNXRuntime not available for provider detection", context="FastEmbed")
        
        info_log(f"FastEmbed using providers: {self.providers}", context="FastEmbed")
    
    def _initialize_models(self):
        """Initialize FastEmbed models."""
        try:
            init_params = {
                "model_name": self.config.model_name,
                "cache_dir": self.config.cache_dir,
                "threads": self.config.threads,
                "providers": self.providers
            }
            
            init_params = {k: v for k, v in init_params.items() if v is not None}
            
            info_log(f"Initializing FastEmbed model: {self.config.model_name}", context="FastEmbed")
            self.embedding_model = TextEmbedding(**init_params)
            
            self.sparse_model = None
            if self.config.enable_sparse_embeddings:
                sparse_model_name = self.config.sparse_model_name or self.config.model_name
                try:
                    sparse_init_params = init_params.copy()
                    sparse_init_params["model_name"] = sparse_model_name
                    self.sparse_model = SparseTextEmbedding(**sparse_init_params)
                    from upsonic.utils.printing import success_log
                    success_log(f"Initialized sparse embedding model: {sparse_model_name}", "FastEmbedProvider")
                except Exception as e:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Could not initialize sparse model: {e}", "FastEmbedProvider")
            
            if self.config.model_warmup:
                self._warmup_models()
            
            from upsonic.utils.printing import success_log
            success_log("FastEmbed models initialized successfully", "FastEmbedProvider")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize FastEmbed model {self.config.model_name}: {str(e)}",
                error_code="MODEL_INIT_ERROR",
                original_error=e
            )
    
    def _warmup_models(self):
        """Warm up models with sample data."""
        try:
            warmup_texts = ["This is a warmup text.", "FastEmbed initialization test."]
            
            list(self.embedding_model.embed(warmup_texts))
            
            if self.sparse_model:
                list(self.sparse_model.embed(warmup_texts))
            
            from upsonic.utils.printing import success_log
            success_log("Model warmup completed", "FastEmbedProvider")
            
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Model warmup failed: {e}", "FastEmbedProvider")
    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """FastEmbed supports all embedding modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY, EmbeddingMode.SYMMETRIC]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get FastEmbed pricing info (local execution is free)."""
        return {
            "per_million_tokens": 0.0,
            "currency": "USD",
            "note": "Local model execution - no API costs",
            "compute_cost": "Uses local compute resources"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current FastEmbed model."""
        if self._model_info is None:
            info = {
                "model_name": self.config.model_name,
                "provider": "FastEmbed (Qdrant)",
                "type": "embedding",
                "execution_mode": "Local",
                "providers": self.providers,
                "threads": self.config.threads or "auto",
                "sparse_embeddings": self.config.enable_sparse_embeddings
            }
            
            try:
                if hasattr(self.embedding_model, 'model'):
                    model = self.embedding_model.model
                    if hasattr(model, 'get_input_details'):
                        input_details = model.get_input_details()
                        if input_details:
                            info["input_shape"] = input_details[0].get('shape', 'unknown')
                    
                    if hasattr(model, 'get_output_details'):
                        output_details = model.get_output_details()
                        if output_details:
                            info["output_shape"] = output_details[0].get('shape', 'unknown')
                            if len(output_details[0].get('shape', [])) > 1:
                                info["dimensions"] = output_details[0]['shape'][-1]
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Could not get detailed model info: {e}", "FastEmbedProvider")
            
            if self.sparse_model:
                info["sparse_model"] = {
                    "enabled": True,
                    "model_name": self.config.sparse_model_name or self.config.model_name
                }
            
            self._model_info = info
        
        return self._model_info
    
    def _process_embeddings(self, embeddings_iterator: Iterator[np.ndarray]) -> List[List[float]]:
        """Process embeddings iterator into list format."""
        embeddings = []
        for embedding in embeddings_iterator:
            if isinstance(embedding, np.ndarray):
                embeddings.append(embedding.tolist())
            else:
                embeddings.append(list(embedding))
        return embeddings
    
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """
        Embed a batch of texts using FastEmbed.
        
        Args:
            texts: List of text strings to embed
            mode: Embedding mode (affects processing strategy)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            if self.config.enable_sparse_embeddings and self.sparse_model:
                embeddings_iterator = self.sparse_model.embed(texts)
                embeddings = self._process_embeddings(embeddings_iterator)
            else:
                if mode == EmbeddingMode.QUERY:
                    embeddings_iterator = self.embedding_model.query_embed(texts)
                elif self.config.doc_embed_type == "passage":
                    embeddings_iterator = self.embedding_model.passage_embed(texts)
                else:
                    embeddings_iterator = self.embedding_model.embed(texts)
                
                embeddings = self._process_embeddings(embeddings_iterator)
            
            if self.config.normalize_embeddings:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            raise ModelConnectionError(
                f"FastEmbed embedding failed: {str(e)}",
                error_code="FASTEMBED_ERROR",
                original_error=e
            )
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length."""
        normalized = []
        for embedding in embeddings:
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                normalized.append((embedding_array / norm).tolist())
            else:
                normalized.append(embedding)
        return normalized
    
    async def validate_connection(self) -> bool:
        """Validate FastEmbed model is working."""
        try:
            test_result = await self._embed_batch(["test connection"], EmbeddingMode.QUERY)
            return len(test_result) > 0 and len(test_result[0]) > 0
            
        except Exception as e:
            debug_log(f"FastEmbed validation failed: {str(e)}", context="FastEmbed")
            return False
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance and resource usage information."""
        info = {
            "model_name": self.config.model_name,
            "providers": self.providers,
            "threads": self.config.threads or "auto",
            "parallel_processing": self.config.enable_parallel_processing,
            "sparse_embeddings": self.config.enable_sparse_embeddings,
            "memory_limit_mb": self.config.max_memory_mb or "unlimited"
        }
        
        try:
            import psutil
            process = psutil.Process()
            info.update({
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": process.cpu_percent()
            })
        except ImportError:
            info["system_info"] = "psutil not available for detailed metrics"
        
        return info
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available FastEmbed models."""
        try:
            models = []
            for model_info in TextEmbedding.list_supported_models():
                models.append({
                    "model_name": model_info.get("model", "unknown"),
                    "dimensions": model_info.get("dim", "unknown"),
                    "description": model_info.get("description", ""),
                    "size_mb": model_info.get("size_in_GB", 0) * 1024 if model_info.get("size_in_GB") else "unknown",
                    "sources": model_info.get("sources", []),
                    "model_file": model_info.get("model_file", "")
                })
            
            return models
            
        except Exception as e:
            debug_log(f"Could not list FastEmbed models: {e}", context="FastEmbed")
            return []
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about model caching."""
        base_info = super().get_cache_info()
        
        cache_dir = self.config.cache_dir or os.path.expanduser("~/.cache/fastembed")
        
        try:
            if os.path.exists(cache_dir):
                total_size = 0
                file_count = 0
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                
                base_info.update({
                    "model_cache_dir": cache_dir,
                    "cache_size_mb": total_size / 1024 / 1024,
                    "cached_files": file_count
                })
            else:
                base_info.update({
                    "model_cache_dir": cache_dir,
                    "cache_exists": False
                })
        except Exception as e:
            base_info["cache_error"] = str(e)
        
        return base_info

    async def close(self):
        """
        Clean up FastEmbed models and clear memory.
        """
        if hasattr(self, 'embedding_model') and self.embedding_model:
            try:
                if hasattr(self.embedding_model, 'aclose'):
                    await self.embedding_model.aclose()
                elif hasattr(self.embedding_model, 'close'):
                    if asyncio.iscoroutinefunction(self.embedding_model.close):
                        await self.embedding_model.close()
                    else:
                        self.embedding_model.close()
                
                del self.embedding_model
                self.embedding_model = None
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Error closing FastEmbed model: {e}", "FastEmbedProvider")
        
        if hasattr(self, 'sparse_model') and self.sparse_model:
            try:
                if hasattr(self.sparse_model, 'aclose'):
                    await self.sparse_model.aclose()
                elif hasattr(self.sparse_model, 'close'):
                    if asyncio.iscoroutinefunction(self.sparse_model.close):
                        await self.sparse_model.close()
                    else:
                        self.sparse_model.close()
                
                del self.sparse_model
                self.sparse_model = None
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Error closing FastEmbed sparse model: {e}", "FastEmbedProvider")
        
        try:
            import onnxruntime as ort
            ort.get_session().close() if hasattr(ort, 'get_session') else None
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not clear ONNX runtime cache: {e}", "FastEmbedProvider")
        
        await super().close()


def create_bge_small_embedding(**kwargs) -> FastEmbedProvider:
    """Create BGE-small FastEmbed provider (fast and efficient)."""
    config = FastEmbedConfig(
        model_name="BAAI/bge-small-en-v1.5",
        **kwargs
    )
    return FastEmbedProvider(config=config)


def create_bge_large_embedding(**kwargs) -> FastEmbedProvider:
    """Create BGE-large FastEmbed provider (high quality)."""
    config = FastEmbedConfig(
        model_name="BAAI/bge-large-en-v1.5",
        **kwargs
    )
    return FastEmbedProvider(config=config)


def create_e5_embedding(**kwargs) -> FastEmbedProvider:
    """Create E5 FastEmbed provider (multilingual)."""
    config = FastEmbedConfig(
        model_name="intfloat/e5-large-v2",
        **kwargs
    )
    return FastEmbedProvider(config=config)


def create_sparse_embedding(**kwargs) -> FastEmbedProvider:
    """Create sparse embedding provider for efficiency."""
    config = FastEmbedConfig(
        model_name="prithivida/Splade_PP_en_v1",
        enable_sparse_embeddings=True,
        **kwargs
    )
    return FastEmbedProvider(config=config)


def create_gpu_accelerated_embedding(
    model_name: str = "BAAI/bge-large-en-v1.5",
    **kwargs
) -> FastEmbedProvider:
    """Create GPU-accelerated FastEmbed provider."""
    config = FastEmbedConfig(
        model_name=model_name,
        enable_gpu=True,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        **kwargs
    )
    return FastEmbedProvider(config=config)
