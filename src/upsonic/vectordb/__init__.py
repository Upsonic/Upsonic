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

def _get_provider_classes():
    """Get provider classes with lazy importing."""
    providers = {}
    
    try:
        from .providers.chroma import ChromaProvider
        providers['ChromaProvider'] = ChromaProvider
    except ImportError:
        pass
    
    try:
        from .providers.faiss import FaissProvider
        providers['FaissProvider'] = FaissProvider
    except ImportError:
        pass
    
    try:
        from .providers.pinecone import PineconeProvider
        providers['PineconeProvider'] = PineconeProvider
    except ImportError:
        pass
    
    try:
        from .providers.qdrant import QdrantProvider
        providers['QdrantProvider'] = QdrantProvider
    except ImportError:
        pass
    
    try:
        from .providers.milvus import MilvusProvider
        providers['MilvusProvider'] = MilvusProvider
    except ImportError:
        pass
    
    try:
        from .providers.weaviate import WeaviateProvider
        providers['WeaviateProvider'] = WeaviateProvider
    except ImportError:
        pass
    
    try:
        from .providers.pgvector import PgVectorProvider
        providers['PgVectorProvider'] = PgVectorProvider
    except ImportError:
        pass
    
    return providers


def __getattr__(name: str) -> Any:
    """Lazy import of provider classes."""
    providers = _get_provider_classes()
    if name in providers:
        return providers[name]
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


