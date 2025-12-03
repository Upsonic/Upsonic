import pydantic
from enum import Enum
from typing import Optional, Dict, Any, List, Literal, Union
from abc import ABC

# Shared configuration for immutability
PydanticConfig = pydantic.ConfigDict(frozen=True, extra='forbid')

# Import logger for warnings
from upsonic.utils.logging_config import get_logger
logger = get_logger(__name__)


class Mode(str, Enum):
    """Operational mode of the vector database."""
    CLOUD = 'cloud'
    LOCAL = 'local'
    EMBEDDED = 'embedded'
    IN_MEMORY = 'in_memory'


class DistanceMetric(str, Enum):
    """Similarity calculation algorithms."""
    COSINE = 'Cosine'
    EUCLIDEAN = 'Euclidean'
    DOT_PRODUCT = 'DotProduct'


class IndexType(str, Enum):
    """Core Approximate Nearest Neighbor (ANN) index algorithms."""
    HNSW = 'HNSW'
    IVF_FLAT = 'IVF_FLAT'
    FLAT = 'FLAT'


class BaseVectorDBConfig(pydantic.BaseModel, ABC):
    """
    Base configuration containing only the attributes that ALL vector database
    providers require. This ensures a minimal, clean interface.
    """
    model_config = PydanticConfig
    
    # Essential collection settings
    collection_name: str = "default_collection"
    vector_size: int
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    recreate_if_exists: bool = False
    
    # Common search defaults
    default_top_k: int = 10
    default_similarity_threshold: Optional[float] = None
    
    # Common search capabilities (used by most providers)
    dense_search_enabled: bool = True
    full_text_search_enabled: bool = True
    hybrid_search_enabled: bool = True
    default_hybrid_alpha: float = 0.5
    default_fusion_method: Literal['rrf', 'weighted'] = 'weighted'
    
    # Common metadata and indexing (used by most providers)
    provider_name: Optional[str] = None
    provider_description: Optional[str] = None
    provider_id: Optional[str] = None

    default_metadata: Optional[Dict[str, Any]] = None
    auto_generate_content_id: bool = True
    indexed_fields: Optional[List[Union[str, Dict[str, Any]]]] = None  # Can be ["field"] or [{"field": "name", "type": "keyword"}]
    
    @pydantic.field_validator('default_similarity_threshold')
    @classmethod
    def validate_similarity_threshold(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v


class ConnectionConfig(pydantic.BaseModel):
    """Connection settings for cloud/local/embedded modes."""
    model_config = PydanticConfig
    mode: Mode
    
    # Cloud/Local connection
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[pydantic.SecretStr] = None
    use_tls: bool = True
    
    # Advanced connection options
    grpc_port: Optional[int] = None
    prefer_grpc: bool = False
    https: Optional[bool] = None
    prefix: Optional[str] = None  # URL path prefix
    timeout: Optional[float] = None  # Request timeout in seconds
    
    # Alternative connection modes
    url: Optional[str] = None  # Full URL (overrides host/port)
    location: Optional[str] = None  # Special location string (e.g., ":memory:")
    
    # Embedded/Local storage
    db_path: Optional[str] = None
    
    @pydantic.model_validator(mode='after')
    def validate_connection_params(self):
        """Ensure required parameters are provided based on mode."""
        if self.mode == Mode.CLOUD:
            if not self.api_key and not self.url:
                raise ValueError("api_key is required for CLOUD mode (unless full url is provided)")
        elif self.mode == Mode.LOCAL:
            if not self.url and (not self.host or self.port is None):
                raise ValueError("host and port are required for LOCAL mode (unless full url is provided)")
        elif self.mode == Mode.EMBEDDED:
            if not self.db_path:
                raise ValueError("db_path is required for EMBEDDED mode")
        return self




class HNSWIndexConfig(pydantic.BaseModel):
    """HNSW index tuning parameters."""
    model_config = PydanticConfig
    type: Literal[IndexType.HNSW] = IndexType.HNSW
    m: int = 16
    ef_construction: int = 200
    ef_search: Optional[int] = None


class IVFIndexConfig(pydantic.BaseModel):
    """IVF index tuning parameters."""
    model_config = PydanticConfig
    type: Literal[IndexType.IVF_FLAT] = IndexType.IVF_FLAT
    nlist: int = 100
    nprobe: Optional[int] = None


class FlatIndexConfig(pydantic.BaseModel):
    """Flat (brute-force) index - no tuning needed."""
    model_config = PydanticConfig
    type: Literal[IndexType.FLAT] = IndexType.FLAT


IndexConfig = Union[HNSWIndexConfig, IVFIndexConfig, FlatIndexConfig]




class ChromaConfig(BaseVectorDBConfig):
    """
    Configuration for ChromaDB provider.
    Supports only HNSW and FLAT indexes.
    
    Note: ChromaDB automatically indexes all metadata fields, so no explicit indexing config is needed.
    """
    connection: ConnectionConfig
    index: Union[HNSWIndexConfig, FlatIndexConfig] = HNSWIndexConfig()
    
    # Chroma-specific
    tenant: Optional[str] = None
    database: Optional[str] = None
    
    @pydantic.model_validator(mode='after')
    def validate_chroma_config(self):
        """Validate Chroma-specific constraints."""
        # Chroma doesn't support IVF
        if isinstance(self.index, IVFIndexConfig):
            raise ValueError("Chroma does not support IVF index type")
        
        return self
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ChromaConfig':
        """Create ChromaConfig from a dictionary."""
        return cls(**config_dict)


class FaissConfig(BaseVectorDBConfig):
    """
    Configuration for FAISS provider.
    Supports all index types and quantization.
    
    FAISS is a pure vector similarity library, so metadata indexing and filtering
    are implemented as custom Python structures for fast lookups.
    """
    # FAISS always needs a path for persistence (except in-memory)
    db_path: Optional[str] = None
    
    # Index configuration
    index: IndexConfig = HNSWIndexConfig()
    
    # FAISS-specific features
    normalize_vectors: bool = True  # Auto-normalize for cosine similarity
    quantization_type: Optional[Literal['scalar', 'product']] = None
    quantization_bits: int = 8
    
    
    @pydantic.model_validator(mode='after')
    def validate_faiss_config(self):
        """Ensure normalize_vectors is True for cosine similarity."""
        if self.distance_metric == DistanceMetric.COSINE and not self.normalize_vectors:
            raise ValueError("normalize_vectors must be True when using COSINE distance metric")
        return self
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FaissConfig':
        """Create FaissConfig from a dictionary."""
        return cls(**config_dict)


class PayloadFieldConfig(pydantic.BaseModel):
    """
    Advanced configuration for a payload field with explicit type and indexing.
    
    Use this for fine-grained control over field indexing.
    """
    model_config = PydanticConfig
    field_name: str
    field_type: Literal['text', 'keyword', 'integer', 'float', 'boolean', 'geo']
    indexed: bool = True  # Changed default to True (if you're defining it, you want it indexed)
    params: Optional[Dict[str, Any]] = None  # Custom Qdrant index params
    

class QdrantConfig(BaseVectorDBConfig):
    """
    Configuration for Qdrant provider.
    Supports HNSW and FLAT indexes with advanced features.
    """
    connection: ConnectionConfig
    index: Union[HNSWIndexConfig, FlatIndexConfig] = HNSWIndexConfig()
    
    # Qdrant-specific
    quantization_config: Optional[Dict[str, Any]] = None
    on_disk_payload: bool = False
    write_consistency_factor: int = 1
    shard_number: Optional[int] = None
    replication_factor: Optional[int] = None
    
    # Payload schema and indexing
    # Advanced approach: explicit field configurations with types and params
    payload_field_configs: Optional[List[PayloadFieldConfig]] = None  # Fine-grained control
    
    # Named vectors for sparse/dense hybrid search
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    use_sparse_vectors: bool = False  # Enable sparse vector support
    
    @pydantic.model_validator(mode='after')
    def validate_qdrant_config(self):
        """Validate Qdrant-specific constraints."""
        if isinstance(self.index, IVFIndexConfig):
            raise ValueError("Qdrant does not support IVF index type")
        
        # If sparse vectors enabled, validate hybrid search is enabled
        if self.use_sparse_vectors and not self.hybrid_search_enabled:
            raise ValueError("use_sparse_vectors requires hybrid_search_enabled to be True")
        
        return self
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QdrantConfig':
        """Create QdrantConfig from a dictionary."""
        return cls(**config_dict)


class PineconeConfig(BaseVectorDBConfig):
    """
    Configuration for Pinecone provider.
    Cloud-only vector database with comprehensive metadata and indexing support.
    
    Supports both ServerlessSpec and PodSpec configurations.
    """
    # Pinecone connection
    api_key: pydantic.SecretStr
    
    # Spec can be ServerlessSpec, PodSpec, or dict
    spec: Optional[Union[Dict[str, Any], Any]] = None  # ServerlessSpec, PodSpec, or dict
    
    # Environment (for backward compatibility, used if spec not provided)
    environment: Optional[str] = None  # Format: "cloud-region" (e.g., "aws-us-east-1")
    
    # Pinecone-specific
    namespace: Optional[str] = None
    metric: Literal['cosine', 'euclidean', 'dotproduct'] = 'cosine'
    
    # Pod configuration (for PodSpec)
    pods: Optional[int] = None
    pod_type: Optional[str] = None
    replicas: Optional[int] = None
    shards: Optional[int] = None
    
    # Advanced Pinecone client settings
    host: Optional[str] = None
    additional_headers: Optional[Dict[str, str]] = None
    pool_threads: Optional[int] = 1
    index_api: Optional[Any] = None
    
    # Sparse vector support for hybrid search (single index approach)
    use_sparse_vectors: bool = False  # Enable sparse vector support in single index
    sparse_encoder_model: str = "pinecone-sparse-english-v0"  # Model for sparse vector generation
    
    # Batch processing
    batch_size: int = 100  # Batch size for upsert operations
    show_progress: bool = False  # Show progress during batch operations
    
    # Timeout settings
    timeout: Optional[int] = None  # Request timeout in seconds (for index operations)
    
    # Reranking
    reranker: Optional[Any] = None  # Reranker instance for post-processing results
    
    def model_post_init(self, __context):
        """Map distance metric to Pinecone format and validate configuration."""
        metric_map = {
            DistanceMetric.COSINE: 'cosine',
            DistanceMetric.EUCLIDEAN: 'euclidean',
            DistanceMetric.DOT_PRODUCT: 'dotproduct'
        }
        # Use object.__setattr__ to bypass frozen model restriction
        object.__setattr__(self, 'metric', metric_map[self.distance_metric])
        
        # Sync hybrid search settings - use base class hybrid_search_enabled
        if self.use_sparse_vectors:
            object.__setattr__(self, 'hybrid_search_enabled', True)
        
        # CRITICAL: Hybrid search requires dotproduct metric (uses base class hybrid_search_enabled)
        if self.hybrid_search_enabled and self.metric != 'dotproduct':
            logger.warning(f"Hybrid search requires dotproduct metric. Updating from '{self.metric}' to 'dotproduct'.")
            object.__setattr__(self, 'metric', 'dotproduct')
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PineconeConfig':
        """Create PineconeConfig from a dictionary."""
        return cls(**config_dict)


class MilvusConfig(BaseVectorDBConfig):
    """
    Configuration for Milvus provider.
    Supports advanced indexing, consistency, and hybrid search capabilities.
    
    Milvus supports both dense and sparse vectors for hybrid search scenarios.
    Sparse vectors enable full-text search capabilities when combined with dense vectors.
    """
    connection: ConnectionConfig
    index: IndexConfig = HNSWIndexConfig()
    
    # Milvus-specific consistency
    consistency_level: Literal['Strong', 'Bounded', 'Session', 'Eventually'] = 'Bounded'
    
    # Advanced index configuration
    index_params: Optional[Dict[str, Any]] = None  # Override automatic index params
    
    # Sparse vector support for full-text and hybrid search
    use_sparse_vectors: bool = False  # Enable sparse vector support
    
    # Vector field names
    dense_vector_field: str = "dense_vector"
    sparse_vector_field: str = "sparse_vector"
    
    # Search parameters (default search params if not provided in search calls)
    search_params: Optional[Dict[str, Any]] = None
    
    # Hybrid search ranking
    rrf_k: int = 60  # k parameter for RRFRanker
    
    # Batch processing
    batch_size: int = 100  # Batch size for upsert operations
    
    @pydantic.model_validator(mode='after')
    def validate_milvus_config(self):
        """Validate Milvus-specific constraints."""
        # If sparse vectors enabled, validate hybrid search is enabled
        if self.use_sparse_vectors and not self.hybrid_search_enabled:
            logger.warning("use_sparse_vectors is enabled but hybrid_search_enabled is False. Enabling hybrid_search_enabled.")
            object.__setattr__(self, 'hybrid_search_enabled', True)
        
        return self
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MilvusConfig':
        """Create MilvusConfig from a dictionary."""
        return cls(**config_dict)


class WeaviateConfig(BaseVectorDBConfig):
    """
    Configuration for Weaviate provider.
    Schema-based vector database with modules and advanced features.
    
    This config supports all Weaviate v4 features including multi-tenancy,
    replication, sharding, inverted indexing, and generative/reranker modules.
    """
    connection: ConnectionConfig
    index: Union[HNSWIndexConfig, FlatIndexConfig] = HNSWIndexConfig()
    
    # Collection description
    description: Optional[str] = None
    
    # Weaviate-specific: Multi-tenancy
    namespace: Optional[str] = None  # For multi-tenancy support (tenant name)
    multi_tenancy_enabled: bool = False  # Enable multi-tenancy for the collection
    
    # Schema configuration
    properties: Optional[List[Dict[str, Any]]] = None  # Custom properties beyond standard fields
    references: Optional[List[Dict[str, Any]]] = None  # Cross-references to other collections
    
    # Inverted index configuration (for BM25 tuning)
    inverted_index_config: Optional[Dict[str, Any]] = None  # e.g., {'bm25': {'k1': 1.2, 'b': 0.75}}
    
    # Advanced Weaviate features
    replication_config: Optional[Dict[str, Any]] = None  # e.g., {'factor': 3, 'asyncEnabled': True}
    sharding_config: Optional[Dict[str, Any]] = None  # e.g., {'virtualPerPhysical': 128, 'desiredCount': 2}
    
    # AI modules (optional)
    generative_config: Optional[Dict[str, Any]] = None  # e.g., {'provider': 'openai', 'model': 'gpt-4'}
    reranker_config: Optional[Dict[str, Any]] = None  # e.g., {'provider': 'cohere', 'model': 'rerank-english-v2.0'}
    
    # API keys for generative and reranker modules (optional - can also use env vars)
    # Format: {'provider_name': 'api_key_value'}
    api_keys: Optional[Dict[str, str]] = None  # e.g., {'openai': 'sk-...', 'cohere': '...'}
    
    @pydantic.model_validator(mode='after')
    def validate_weaviate_config(self):
        """Validate Weaviate-specific constraints."""
        # Weaviate doesn't support IVF
        if isinstance(self.index, IVFIndexConfig):
            raise ValueError("Weaviate does not support IVF index type")
        
        # If namespace is specified, multi_tenancy should be enabled
        if self.namespace and not self.multi_tenancy_enabled:
            # Auto-enable multi-tenancy if namespace is provided
            object.__setattr__(self, 'multi_tenancy_enabled', True)
        
        return self
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WeaviateConfig':
        """Create WeaviateConfig from a dictionary."""
        return cls(**config_dict)


class PgVectorConfig(BaseVectorDBConfig):
    """
    Configuration for PgVector provider.
    PostgreSQL extension for vector similarity.
    
    This config supports comprehensive metadata management, flexible indexing,
    and hybrid search capabilities using pgvector and PostgreSQL full-text search.
    """
    # PostgreSQL connection
    connection_string: pydantic.SecretStr
    
    # PgVector-specific
    schema_name: str = "public"
    table_name: Optional[str] = None  # Uses collection_name if not specified
    
    # Index configuration (uses shared index configs)
    index: Union[HNSWIndexConfig, IVFIndexConfig] = HNSWIndexConfig()
    
    # Full-text search configuration
    content_language: str = "english"  # Language for full-text search (e.g., 'english', 'spanish')
    prefix_match: bool = False  # Enable prefix matching for full-text search (appends * to words)
    
    # Schema version for migrations
    schema_version: int = 1
    auto_upgrade_schema: bool = False
    
    # Batch processing
    batch_size: int = 100  # Batch size for upsert operations
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    
    @pydantic.field_validator('table_name')
    @classmethod
    def set_table_name(cls, v, info):
        """Use collection_name as table_name if not specified."""
        if v is None and 'collection_name' in info.data:
            return info.data['collection_name']
        return v
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PgVectorConfig':
        """Create PgVectorConfig from a dictionary."""
        return cls(**config_dict)



Config = Union[
    ChromaConfig,
    FaissConfig,
    QdrantConfig,
    PineconeConfig,
    MilvusConfig,
    WeaviateConfig,
    PgVectorConfig
]



def create_config(provider: str, **kwargs) -> Config:
    """
    Factory function to create the appropriate config based on provider name.
    
    Args:
        provider: Name of the vector database provider
        **kwargs: Configuration parameters
        
    Returns:
        Provider-specific configuration object
    """
    provider_map = {
        'chroma': ChromaConfig,
        'faiss': FaissConfig,
        'qdrant': QdrantConfig,
        'pinecone': PineconeConfig,
        'milvus': MilvusConfig,
        'weaviate': WeaviateConfig,
        'pgvector': PgVectorConfig,
    }
    
    config_class = provider_map.get(provider.lower())
    if not config_class:
        raise ValueError(f"Unknown provider: {provider}")
    
    return config_class(**kwargs)