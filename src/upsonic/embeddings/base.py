from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import time
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, ConfigDict
from ..schemas.data_models import Chunk

from ..utils.package.exception import ConfigurationError, ModelConnectionError
from ..utils.printing import console, Panel, Table


class EmbeddingMode(str, Enum):
    """Embedding operation modes for different use cases."""
    DOCUMENT = "document"
    QUERY = "query"
    SYMMETRIC = "symmetric"
    CLUSTERING = "clustering"


class EmbeddingMetrics(BaseModel):
    """Metrics and statistics for embedding operations."""
    total_chunks: int = 0
    total_tokens: int = 0
    embedding_time_ms: float = 0
    avg_time_per_chunk: float = 0
    dimension: int = 0
    model_name: str = ""
    provider: str = ""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EmbeddingConfig(BaseModel):
    """Base configuration for all embedding providers."""
    model_name: str = Field(..., description="The name of the embedding model")
    dimension: Optional[int] = Field(None, description="Expected embedding dimension (for validation)")
    batch_size: int = Field(100, description="Batch size for document embedding")
    max_retries: int = Field(3, description="Maximum number of retries on failure")
    retry_delay: float = Field(1.0, description="Initial delay between retries")
    timeout: float = Field(30.0, description="Timeout for embedding operations")
    normalize_embeddings: bool = Field(True, description="Whether to normalize embeddings to unit length")
    show_progress: bool = Field(True, description="Whether to show progress during batch operations")
    cache_embeddings: bool = Field(False, description="Whether to cache embeddings")
    truncate_input: bool = Field(True, description="Whether to truncate inputs that exceed max length")
    max_input_length: Optional[int] = Field(None, description="Maximum input length in tokens/characters")
    
    enable_retry_with_backoff: bool = Field(True, description="Enable exponential backoff for retries")
    enable_adaptive_batching: bool = Field(True, description="Dynamically adjust batch size based on performance")
    enable_compression: bool = Field(False, description="Enable embedding compression for storage efficiency")
    compression_ratio: float = Field(0.5, description="Target compression ratio (0.0-1.0)")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EmbeddingProvider(BaseModel, ABC):
    """
    The abstract contract for all embedding model providers.

    This base class provides a comprehensive framework for embedding operations
    with advanced features like batching, caching, progress tracking, and error handling.
    """
    
    config: EmbeddingConfig
    _cache: Dict[str, List[float]] = {}
    _metrics: EmbeddingMetrics = None
    _executor: Optional[ThreadPoolExecutor] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    def __init__(self, **data):
        super().__init__(**data)
        self._metrics = EmbeddingMetrics(provider=self.__class__.__name__)
        if hasattr(self.config, 'model_name'):
            self._metrics.model_name = self.config.model_name
    
    @abstractmethod
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """
        Internal method to embed a batch of texts.

        This is the core method that each provider must implement. It handles the actual
        API calls or model inference for a batch of texts.

        Args:
            texts: List of text strings to embed
            mode: The embedding mode to use

        Returns:
            List of embedding vectors
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model metadata like dimensions, max tokens, etc.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def supported_modes(self) -> List[EmbeddingMode]:
        """List of embedding modes supported by this provider."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def pricing_info(self) -> Dict[str, float]:
        """Get pricing information for the embedding model."""
        raise NotImplementedError
    
    async def embed_documents(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Creates vector embeddings for a batch of Chunks with advanced features.
        
        This method provides intelligent batching, progress tracking, caching,
        and comprehensive error handling.
        
        Args:
            chunks: List of Chunk objects to be embedded
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
        
        texts = [chunk.text_content for chunk in chunks]
        return await self.embed_texts(texts, mode=EmbeddingMode.DOCUMENT)
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Creates a vector embedding for a single query string.
        
        This method is optimized for real-time query embedding with caching support.
        
        Args:
            query: The user's query string
            
        Returns:
            Single embedding vector
        """
        embeddings = await self.embed_texts([query], mode=EmbeddingMode.QUERY)
        return embeddings[0] if embeddings else []
    

    async def embed_texts(
        self, 
        texts: List[str], 
        mode: EmbeddingMode = EmbeddingMode.DOCUMENT,
        show_progress: Optional[bool] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[List[float]]:
        """
        Embed a list of texts with advanced features.
        
        This is the main method that handles batching, caching, progress tracking,
        and error recovery for embedding operations.

        Args:
            texts: List of text strings to embed
            mode: The embedding mode to use
            show_progress: Override the default progress display setting
            metadata: Optional metadata for each text (for caching keys)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        start_time = time.time()
        self._metrics.total_chunks = len(texts)
        
        if self.config.cache_embeddings:
            embeddings, uncached_indices = self._check_cache(texts, metadata)
            if not uncached_indices:
                return embeddings
            uncached_texts = [texts[i] for i in uncached_indices]
        else:
            embeddings = [None] * len(texts)
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        all_embeddings = []
        show_progress = show_progress if show_progress is not None else self.config.show_progress
        
        if show_progress and len(uncached_texts) > 1:
            progress_table = self._create_progress_display(len(uncached_texts))
        
        for i in range(0, len(uncached_texts), self.config.batch_size):
            batch = uncached_texts[i:i + self.config.batch_size]
            batch_embeddings = await self._embed_with_retry(batch, mode)
            all_embeddings.extend(batch_embeddings)
            
            if show_progress and len(uncached_texts) > 1:
                self._update_progress(i + len(batch), len(uncached_texts))
        
        if self.config.cache_embeddings:
            self._update_cache(uncached_texts, all_embeddings, metadata, uncached_indices)
            for idx, embedding in zip(uncached_indices, all_embeddings):
                embeddings[idx] = embedding
        else:
            embeddings = all_embeddings
        
        if self.config.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        if self.config.enable_compression:
            embeddings = self._compress_embeddings(embeddings)
        
        self._metrics.embedding_time_ms = (time.time() - start_time) * 1000
        self._metrics.avg_time_per_chunk = self._metrics.embedding_time_ms / len(texts)
        self._metrics.dimension = len(embeddings[0]) if embeddings else 0
        
        return embeddings
    
    async def _embed_with_retry(self, texts: List[str], mode: EmbeddingMode) -> List[List[float]]:
        """Handle retries with exponential backoff."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await self._embed_batch(texts, mode)
            except Exception as e:
                if isinstance(e, ConfigurationError):
                    raise
                last_error = e
                if attempt < self.config.max_retries - 1:
                    if self.config.enable_retry_with_backoff:
                        delay = self.config.retry_delay * (2 ** attempt)
                    else:
                        delay = self.config.retry_delay
                    await asyncio.sleep(delay)
                    
                    if self.config.enable_adaptive_batching and len(texts) > 1:
                        mid = len(texts) // 2
                        first_half = await self._embed_with_retry(texts[:mid], mode)
                        second_half = await self._embed_with_retry(texts[mid:], mode)
                        return first_half + second_half
        
        raise ModelConnectionError(
            f"Failed to embed texts after {self.config.max_retries} attempts: {str(last_error)}",
            error_code="EMBEDDING_FAILED",
            original_error=last_error
        )
    
    def _check_cache(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]]) -> Tuple[List[Optional[List[float]]], List[int]]:
        """Check cache for existing embeddings."""
        embeddings = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, metadata[i] if metadata else None)
            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
            else:
                embeddings.append(None)
                uncached_indices.append(i)
        
        return embeddings, uncached_indices
    
    def _update_cache(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]], indices: List[int]):
        """Update the embedding cache."""
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            meta_idx = indices[i] if indices else i
            cache_key = self._get_cache_key(text, metadata[meta_idx] if metadata else None)
            self._cache[cache_key] = embedding
    
    def _get_cache_key(self, text: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Generate a cache key for a text."""
        key_parts = [self.config.model_name, text]
        if metadata:
            key_parts.append(str(sorted(metadata.items())))
        return "|".join(key_parts)
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length."""
        normalized = []
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized.append([x / norm for x in embedding])
            else:
                normalized.append(embedding)
        return normalized
    
    def _compress_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Apply dimensionality reduction for storage efficiency."""
        return embeddings
    
    def _create_progress_display(self, total: int) -> Table:
        """Create a progress display table."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Time", style="blue")
        
        console.print(Panel(table, title="Embedding Progress", expand=False))
        return table
    
    def _update_progress(self, current: int, total: int):
        """Update progress display."""
        progress_pct = (current / total) * 100
        elapsed = self._metrics.embedding_time_ms / 1000
        eta = (elapsed / current) * (total - current) if current > 0 else 0
        
        pass
    
    async def validate_connection(self) -> bool:
        """Validate that the embedding provider is properly configured and accessible."""
        try:
            test_embedding = await self.embed_texts(["test"], show_progress=False)
            return len(test_embedding) > 0 and len(test_embedding[0]) > 0
        except Exception:
            return False
    
    def get_metrics(self) -> EmbeddingMetrics:
        """Get current metrics for this embedding provider."""
        return self._metrics.copy()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        return {
            "enabled": self.config.cache_embeddings,
            "size": len(self._cache),
            "memory_mb": sum(len(str(v)) for v in self._cache.values()) / (1024 * 1024)
        }
    
    async def warmup(self, sample_texts: Optional[List[str]] = None):
        """Warm up the embedding model with sample texts."""
        if sample_texts is None:
            sample_texts = ["Hello world", "This is a test"]
        
        await self.embed_texts(sample_texts, show_progress=False)
    
    def estimate_cost(self, num_texts: int, avg_text_length: int = 100) -> Dict[str, float]:
        """Estimate the cost of embedding a given number of texts."""
        pricing = self.pricing_info
        if not pricing:
            return {"estimated_cost": 0.0, "message": "Pricing information not available"}
        
        avg_tokens = avg_text_length / 4
        total_tokens = num_texts * avg_tokens
        
        cost_per_million = pricing.get("per_million_tokens", 0.0)
        estimated_cost = (total_tokens / 1_000_000) * cost_per_million
        
        return {
            "estimated_cost": estimated_cost,
            "estimated_tokens": total_tokens,
            "price_per_million_tokens": cost_per_million
        }
    
    async def close(self):
        """
        Clean up resources and close connections.
        
        This method should be called when the provider is no longer needed
        to prevent resource leaks.
        """
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None