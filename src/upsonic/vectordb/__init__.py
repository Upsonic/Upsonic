from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseVectorDBProvider
    from .store import VectorStore
    from .config import (
        BaseVectorDBConfig,
        DistanceMetric,
        IndexType,
        Mode,
        ConnectionConfig,
        HNSWIndexConfig,
        IVFIndexConfig,
        FlatIndexConfig,
        PayloadFieldConfig,
        ChromaConfig,
        FaissConfig,
        QdrantConfig,
        PineconeConfig,
        MilvusConfig,
        WeaviateConfig,
        PgVectorConfig,
        SuperMemoryConfig,
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
    from .providers.supermemory import SuperMemoryProvider

# Provider class mapping for lazy imports
_PROVIDER_MAP = {
    'ChromaProvider': '.providers.chroma',
    'FaissProvider': '.providers.faiss',
    'PineconeProvider': '.providers.pinecone',
    'QdrantProvider': '.providers.qdrant',
    'MilvusProvider': '.providers.milvus',
    'WeaviateProvider': '.providers.weaviate',
    'PgVectorProvider': '.providers.pgvector',
    'SuperMemoryProvider': '.providers.supermemory',
}

# Cache for lazily imported providers and configs
_provider_cache: dict[str, Any] = {}
_config_cache: dict[str, Any] = {}

def _get_base_classes():
    """Lazy import of base classes."""
    from .base import BaseVectorDBProvider
    return {
        'BaseVectorDBProvider': BaseVectorDBProvider,
    }

def _get_config_classes():
    """Lazy import of config classes."""
    if _config_cache:
        return _config_cache
    
    from .config import (
        BaseVectorDBConfig,
        DistanceMetric,
        IndexType,
        Mode,
        ConnectionConfig,
        HNSWIndexConfig,
        IVFIndexConfig,
        FlatIndexConfig,
        PayloadFieldConfig,
        ChromaConfig,
        FaissConfig,
        QdrantConfig,
        PineconeConfig,
        MilvusConfig,
        WeaviateConfig,
        PgVectorConfig,
        SuperMemoryConfig,
        create_config,
    )
    
    _config_cache.update({
        'BaseVectorDBConfig': BaseVectorDBConfig,
        'DistanceMetric': DistanceMetric,
        'IndexType': IndexType,
        'Mode': Mode,
        'ConnectionConfig': ConnectionConfig,
        'HNSWIndexConfig': HNSWIndexConfig,
        'IVFIndexConfig': IVFIndexConfig,
        'FlatIndexConfig': FlatIndexConfig,
        'PayloadFieldConfig': PayloadFieldConfig,
        'ChromaConfig': ChromaConfig,
        'FaissConfig': FaissConfig,
        'QdrantConfig': QdrantConfig,
        'PineconeConfig': PineconeConfig,
        'MilvusConfig': MilvusConfig,
        'WeaviateConfig': WeaviateConfig,
        'PgVectorConfig': PgVectorConfig,
        'SuperMemoryConfig': SuperMemoryConfig,
        'create_config': create_config,
    })
    
    return _config_cache

def __getattr__(name: str) -> Any:
    """Lazy import of provider, config, and store classes."""
    # Check base classes first
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]

    # VectorStore convenience wrapper
    if name == "VectorStore":
        from .store import VectorStore
        return VectorStore
    
    # Check config classes
    config_classes = _get_config_classes()
    if name in config_classes:
        return config_classes[name]
    
    # Check provider cache
    if name in _provider_cache:
        return _provider_cache[name]
    
    # Check if it's a provider class
    if name in _PROVIDER_MAP:
        module_path = _PROVIDER_MAP[name]
        try:
            from importlib import import_module
            module = import_module(module_path, package=__package__)
            provider_class = getattr(module, name)
            _provider_cache[name] = provider_class
            return provider_class
        except (ImportError, AttributeError) as e:
            raise AttributeError(
                f"module '{__name__}' has no attribute '{name}'. "
                f"Failed to import provider: {e}"
            ) from e
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )


__all__ = [
    # Base classes
    'BaseVectorDBProvider',

    # High-level wrapper
    'VectorStore',
    
    # Provider classes
    'ChromaProvider',
    'FaissProvider',
    'PineconeProvider',
    'QdrantProvider',
    'MilvusProvider',
    'WeaviateProvider',
    'PgVectorProvider',
    'SuperMemoryProvider',
    
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
    'SuperMemoryConfig',
    'create_config',
]


