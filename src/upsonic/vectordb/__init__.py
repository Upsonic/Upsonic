from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import (
    BaseVectorDBProvider,
)

from .config import (
    # Base configs
    BaseVectorDBConfig,
    DistanceMetric,
    IndexType,
    Mode,
    ConnectionConfig,
    
    # Index configs
    HNSWIndexConfig,
    IVFIndexConfig,
    FlatIndexConfig,
    
    # Payload configs
    PayloadFieldConfig,
    
    # Provider configs
    ChromaConfig,
    FaissConfig,
    QdrantConfig,
    PineconeConfig,
    MilvusConfig,
    WeaviateConfig,
    PgVectorConfig,
    
    # Factory function
    create_config,
)

if TYPE_CHECKING:
    from .providers.chroma import ChromaProvider
    from .providers.faiss import FaissProvider
    from .providers.pinecone import PineconeProvider
    from .providers.qdrant import QdrantProvider
    from .providers.milvus import MilvusProvider
    from .providers.weaviate import WeaviateProvider
    from .providers.pgvector import PgVectorProvider

# Provider class mapping for lazy imports
_PROVIDER_MAP = {
    'ChromaProvider': '.providers.chroma',
    'FaissProvider': '.providers.faiss',
    'PineconeProvider': '.providers.pinecone',
    'QdrantProvider': '.providers.qdrant',
    'MilvusProvider': '.providers.milvus',
    'WeaviateProvider': '.providers.weaviate',
    'PgVectorProvider': '.providers.pgvector',
}

# Cache for lazily imported providers
_provider_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy import of provider classes."""
    # Check cache first
    if name in _provider_cache:
        return _provider_cache[name]
    
    # Check if it's a provider class
    if name in _PROVIDER_MAP:
        module_path = _PROVIDER_MAP[name]
        try:
            # Import the module dynamically
            from importlib import import_module
            module = import_module(module_path, package=__package__)
            provider_class = getattr(module, name)
            # Cache it for future access
            _provider_cache[name] = provider_class
            return provider_class
        except (ImportError, AttributeError) as e:
            raise AttributeError(
                f"module '{__name__}' has no attribute '{name}'. "
                f"Failed to import provider: {e}"
            ) from e
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Base classes
    'BaseVectorDBProvider',
    
    # Provider classes
    'ChromaProvider',
    'FaissProvider',
    'PineconeProvider',
    'QdrantProvider',
    'MilvusProvider',
    'WeaviateProvider',
    'PgVectorProvider',
    
    # Config classes
    'BaseVectorDBConfig',
    'DistanceMetric',
    'IndexType',
    'Mode',
    'ConnectionConfig',
    'HNSWIndexConfig',
    'IVFIndexConfig',
    'FlatIndexConfig',
    'PayloadFieldConfig',
    'ChromaConfig',
    'FaissConfig',
    'QdrantConfig',
    'PineconeConfig',
    'MilvusConfig',
    'WeaviateConfig',
    'PgVectorConfig',
    'create_config',
]


