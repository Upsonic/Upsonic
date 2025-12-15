"""
Milvus Vector Database Provider

A comprehensive, high-level implementation supporting:
- Dense and sparse vectors for hybrid search
- Flexible metadata and field indexing
- Async-first operations
- Advanced filtering and search capabilities
- Compatible with Milvus 2.6+ API
"""

import json
import uuid
from hashlib import md5
from typing import Any, Dict, List, Optional, Union, Literal

try:
    from pymilvus import (
        AsyncMilvusClient,
        MilvusClient,
        DataType,
        AnnSearchRequest,
        RRFRanker,
        WeightedRanker,
    )
    _MILVUS_AVAILABLE = True
except ImportError:
    AsyncMilvusClient = None  # type: ignore
    MilvusClient = None  # type: ignore
    DataType = None  # type: ignore
    AnnSearchRequest = None  # type: ignore
    RRFRanker = None  # type: ignore
    WeightedRanker = None  # type: ignore
    _MILVUS_AVAILABLE = False

from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.vectordb.config import MilvusConfig, Mode
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.utils.logging_config import get_logger
from upsonic.utils.printing import info_log, error_log

logger = get_logger(__name__)


# Distance metric mapping: Framework -> Milvus
DISTANCE_METRIC_MAP = {
    'Cosine': 'COSINE',
    'Euclidean': 'L2',
    'DotProduct': 'IP',
}


class MilvusProvider(BaseVectorDBProvider):
    """
    Milvus vector database provider with comprehensive feature support.
    
    Features:
    - Dense and sparse vector support for hybrid search
    - Flexible metadata management with custom fields
    - Configurable field indexing for optimized filtering
    - Multiple ranking strategies (RRF and Weighted)
    - Async-first operations for high performance
    - Auto-generation of content IDs
    - Batch processing support
    """

    def __init__(self, config: Union[MilvusConfig, Dict[str, Any]]):
        """
        Initialize Milvus provider.
        
        Args:
            config: MilvusConfig object or dict that will be converted to MilvusConfig
        """
        if not _MILVUS_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pymilvus",
                install_command='pip install "upsonic[rag]"',
                feature_name="Milvus vector database provider"
            )
        
        # Convert dict to MilvusConfig if necessary
        if isinstance(config, dict):
            config = MilvusConfig.from_dict(config)
        
        super().__init__(config)
        self._config: MilvusConfig = config  # Type hint for IDE support
        
        # Provider metadata
        self.provider_name = config.provider_name or f"MilvusProvider_{config.collection_name}"
        self.provider_description = config.provider_description
        self.provider_id = config.provider_id or self._generate_provider_id()
        
        # Client instances (lazy initialization)
        self._sync_client: Optional[MilvusClient] = None
        self._async_client: Optional[AsyncMilvusClient] = None
        
        # Connection state
        self._is_connected: bool = False
        
        # Metric type
        self._metric_type = self._get_metric_type()
        
        info_log(
            f"Initialized MilvusProvider for collection '{self._config.collection_name}' "
            f"(sparse vectors: {self._config.use_sparse_vectors})",
            context="MilvusProvider"
        )

    # ============================================================================
    # Client Management
    # ============================================================================

    @property
    def _client(self) -> MilvusClient:
        """Get or create synchronous Milvus client."""
        if self._sync_client is None:
            info_log("Creating synchronous Milvus client", context="MilvusProvider")
            
            # Build connection parameters
            conn_params = self._build_connection_params()
            self._sync_client = MilvusClient(**conn_params)
            
        return self._sync_client
    
    @_client.setter
    def _client(self, value: Optional[MilvusClient]):
        """Allow setting _client (for base class compatibility)."""
        self._sync_client = value

    @property
    def _aclient(self) -> AsyncMilvusClient:
        """Get or create asynchronous Milvus client."""
        if self._async_client is None:
            info_log("Creating asynchronous Milvus client", context="MilvusProvider")
            
            # Build connection parameters
            conn_params = self._build_connection_params()
            self._async_client = AsyncMilvusClient(**conn_params)
            
        return self._async_client

    def _build_connection_params(self) -> Dict[str, Any]:
        """Build connection parameters from config."""
        conn = self._config.connection
        params = {}
        
        if conn.mode == Mode.EMBEDDED:
            # Embedded mode (Milvus Lite)
            params['uri'] = conn.db_path or './milvus.db'
        elif conn.mode == Mode.CLOUD:
            # Cloud mode (Zilliz Cloud)
            if conn.url:
                params['uri'] = conn.url
            else:
                params['uri'] = f"https://{conn.host}:{conn.port or 19530}"
            
            if conn.api_key:
                params['token'] = conn.api_key.get_secret_value()
        else:
            # Local/server mode
            if conn.url:
                params['uri'] = conn.url
            elif conn.host:
                protocol = 'https' if conn.use_tls else 'http'
                params['uri'] = f"{protocol}://{conn.host}:{conn.port or 19530}"
            
            if conn.api_key:
                params['token'] = conn.api_key.get_secret_value()
        
        # Add timeout if specified
        if conn.timeout:
            params['timeout'] = conn.timeout
        
        return params

    def _get_metric_type(self) -> str:
        """Convert framework distance metric to Milvus metric type."""
        return DISTANCE_METRIC_MAP.get(self._config.distance_metric.value, 'COSINE')
    
    def _generate_provider_id(self) -> str:
        """Generates a unique provider ID based on connection details and collection."""
        conn = self._config.connection
        identifier_parts = [
            conn.host or conn.url or "embedded",
            str(conn.port) if conn.port else "",
            self._config.collection_name
        ]
        identifier = "#".join(filter(None, identifier_parts))
        
        return md5(identifier.encode()).hexdigest()[:16]

    # ============================================================================
    # Connection Management (Async-First)
    # ============================================================================

    async def connect(self) -> None:
        """Establish connection to Milvus."""
        try:
            # Initialize async client (this validates connection)
            _ = self._aclient
            self._is_connected = True
            info_log(f"Connected to Milvus at {self._config.connection.host or 'embedded'}", context="MilvusProvider")
        except Exception as e:
            self._is_connected = False
            error_log(f"Failed to connect to Milvus: {e}", context="MilvusProvider")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Milvus."""
        try:
            # Close clients if they exist
            if self._async_client:
                await self._async_client.close()
                self._async_client = None
            
            if self._sync_client:
                self._sync_client.close()
                self._sync_client = None
            
            self._is_connected = False
            info_log("Disconnected from Milvus", context="MilvusProvider")
        except Exception as e:
            error_log(f"Error during disconnect: {e}", context="MilvusProvider")

    async def is_ready(self) -> bool:
        """Check if Milvus is ready and responsive."""
        try:
            # Try to list collections as a health check
            collections = await self._aclient.list_collections()
            return True
        except Exception as e:
            logger.debug(f"Milvus health check failed: {e}")
            return False

    # ============================================================================
    # Collection Management (Async-First)
    # ============================================================================

    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        try:
            return self._client.has_collection(self._config.collection_name)
        except Exception as e:
            logger.debug(f"Error checking collection existence: {e}")
            return False

    async def create_collection(self) -> None:
        """
        Create collection with schema based on configuration.
        
        Supports:
        - Dense-only vectors
        - Sparse-only vectors
        - Hybrid (dense + sparse) vectors
        - Scalar field indexing
        - Partition keys
        """
        if await self.collection_exists():
            if self._config.recreate_if_exists:
                info_log(f"Collection '{self._config.collection_name}' exists. Recreating...", context="MilvusProvider")
                await self.delete_collection()
            else:
                info_log(f"Collection '{self._config.collection_name}' already exists. Skipping creation.", context="MilvusProvider")
                return
        
        info_log(f"Creating collection '{self._config.collection_name}'...", context="MilvusProvider")
        
        # Create schema based on whether sparse vectors are enabled
        if self._config.use_sparse_vectors:
            await self._create_hybrid_collection()
        else:
            await self._create_dense_collection()
        
        info_log(f"Collection '{self._config.collection_name}' created successfully.", context="MilvusProvider")

    async def _create_dense_collection(self) -> None:
        """Create collection with dense vectors only."""
        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,  # Allow dynamic metadata fields
        )
        
        # Parse indexed_fields configuration
        indexed_fields_config = self._parse_indexed_fields()
        
        # Primary key: content_id
        schema.add_field(
            field_name="content_id",
            datatype=DataType.VARCHAR,
            max_length=256,
            is_primary=True
        )
        
        # Core fields with configurable types
        # document_name
        doc_name_config = indexed_fields_config.get("document_name", {"type": "keyword"})
        schema.add_field(
            field_name="document_name",
            datatype=self._get_milvus_datatype(doc_name_config.get("type", "keyword")),
            max_length=1024 if doc_name_config.get("type", "keyword") in ["text", "keyword"] else None
        )
        
        # document_id
        doc_id_config = indexed_fields_config.get("document_id", {"type": "keyword"})
        schema.add_field(
            field_name="document_id",
            datatype=self._get_milvus_datatype(doc_id_config.get("type", "keyword")),
            max_length=256 if doc_id_config.get("type", "keyword") in ["text", "keyword"] else None
        )
        
        # content
        content_config = indexed_fields_config.get("content", {"type": "text"})
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        
        # metadata (always VARCHAR for JSON)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)
        
        # Dense vector
        schema.add_field(
            field_name=self._config.dense_vector_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=self._config.vector_size
        )
        
        # Prepare index parameters
        index_params = self._client.prepare_index_params()
        
        # Add vector index
        vector_index_params = self._build_vector_index_params()
        index_params.add_index(
            field_name=self._config.dense_vector_field,
            index_name="dense_vector_index",
            **vector_index_params
        )
        
        # Add scalar indexes if specified (not supported in embedded mode)
        if indexed_fields_config and self._config.connection.mode != Mode.EMBEDDED:
            for field_name, field_config in indexed_fields_config.items():
                if field_name in ['document_name', 'document_id', 'content_id']:
                    logger.info(f"Creating scalar index for field: {field_name} (type: {field_config.get('type', 'keyword')})")
                    try:
                        # Determine index type based on field type
                        field_type = field_config.get("type", "keyword")
                        index_type = self._get_milvus_index_type(field_type)
                        
                        index_params.add_index(
                            field_name=field_name,
                            index_type=index_type,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add scalar index for {field_name}: {e}")
        
        # Create collection
        await self._aclient.create_collection(
            collection_name=self._config.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self._config.consistency_level,
        )

    async def _create_hybrid_collection(self) -> None:
        """Create collection with both dense and sparse vectors."""
        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        
        # Parse indexed_fields configuration
        indexed_fields_config = self._parse_indexed_fields()
        
        # Primary key: content_id
        schema.add_field(
            field_name="content_id",
            datatype=DataType.VARCHAR,
            max_length=256,
            is_primary=True
        )
        
        # Core fields with configurable types
        # document_name
        doc_name_config = indexed_fields_config.get("document_name", {"type": "keyword"})
        schema.add_field(
            field_name="document_name",
            datatype=self._get_milvus_datatype(doc_name_config.get("type", "keyword")),
            max_length=1024 if doc_name_config.get("type", "keyword") in ["text", "keyword"] else None
        )
        
        # document_id
        doc_id_config = indexed_fields_config.get("document_id", {"type": "keyword"})
        schema.add_field(
            field_name="document_id",
            datatype=self._get_milvus_datatype(doc_id_config.get("type", "keyword")),
            max_length=256 if doc_id_config.get("type", "keyword") in ["text", "keyword"] else None
        )
        
        # content
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        
        # metadata (always VARCHAR for JSON)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)
        
        # Dense vector
        schema.add_field(
            field_name=self._config.dense_vector_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=self._config.vector_size
        )
        
        # Sparse vector
        schema.add_field(
            field_name=self._config.sparse_vector_field,
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        
        # Prepare index parameters
        index_params = self._client.prepare_index_params()
        
        # Dense vector index
        vector_index_params = self._build_vector_index_params()
        index_params.add_index(
            field_name=self._config.dense_vector_field,
            index_name="dense_vector_index",
            **vector_index_params
        )
        
        # Sparse vector index
        index_params.add_index(
            field_name=self._config.sparse_vector_field,
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",  # Inner product for sparse vectors
            params={"drop_ratio_build": 0.2}
        )
        
        # Add scalar indexes if specified (not supported in embedded mode)
        if indexed_fields_config and self._config.connection.mode != Mode.EMBEDDED:
            for field_name, field_config in indexed_fields_config.items():
                if field_name in ['document_name', 'document_id', 'content_id']:
                    logger.info(f"Creating scalar index for field: {field_name} (type: {field_config.get('type', 'keyword')})")
                    try:
                        # Determine index type based on field type
                        field_type = field_config.get("type", "keyword")
                        index_type = self._get_milvus_index_type(field_type)
                        
                        index_params.add_index(
                            field_name=field_name,
                            index_type=index_type,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add scalar index for {field_name}: {e}")
        
        # Create collection
        await self._aclient.create_collection(
            collection_name=self._config.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self._config.consistency_level,
        )

    def _build_vector_index_params(self) -> Dict[str, Any]:
        """Build vector index parameters from config."""
        index_config = self._config.index
        
        # If custom index_params provided, use them
        if self._config.index_params:
            return self._config.index_params
        
        # Check if we're in embedded mode (Milvus Lite)
        is_embedded = self._config.connection.mode == Mode.EMBEDDED
        
        # Build based on index type
        params = {
            "metric_type": self._metric_type,
        }
        
        if index_config.type == 'HNSW':
            # Milvus Lite doesn't support HNSW, fall back to IVF_FLAT
            if is_embedded:
                logger.warning(
                    "HNSW index not supported in embedded mode (Milvus Lite). "
                    "Falling back to IVF_FLAT index."
                )
                params["index_type"] = "IVF_FLAT"
                # Use reasonable defaults for IVF_FLAT
                nlist = min(1024, max(64, index_config.m * 4))  # Convert M to nlist
                params["params"] = {"nlist": nlist}
            else:
                params["index_type"] = "HNSW"
                params["params"] = {
                    "M": index_config.m,
                    "efConstruction": index_config.ef_construction,
                }
        elif index_config.type == 'IVF_FLAT':
            params["index_type"] = "IVF_FLAT"
            params["params"] = {
                "nlist": index_config.nlist,
            }
        elif index_config.type == 'FLAT':
            params["index_type"] = "FLAT"
            params["params"] = {}
        
        return params
    
    def _parse_indexed_fields(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse indexed_fields into a standardized format.
        
        Supports two formats:
        1. Simple: ["document_name", "document_id"]
        2. Advanced: [{"field": "document_name", "type": "keyword"}, {"field": "age", "type": "integer"}]
        
        Returns:
            Dict mapping field_name to config: {"field_name": {"indexed": True, "type": "keyword"}}
        """
        if not self._config.indexed_fields:
            return {}
        
        result = {}
        for item in self._config.indexed_fields:
            if isinstance(item, str):
                # Simple format: just field name
                result[item] = {"indexed": True, "type": "keyword"}
            elif isinstance(item, dict):
                # Advanced format: {"field": "name", "type": "keyword"}
                field_name = item.get("field")
                if field_name:
                    result[field_name] = {
                        "indexed": True,
                        "type": item.get("type", "keyword")
                    }
        
        return result
    
    def _get_milvus_datatype(self, field_type: str) -> DataType:
        """
        Convert field type string to Milvus DataType.
        
        Args:
            field_type: One of 'text', 'keyword', 'integer', 'float', 'boolean'
        
        Returns:
            Milvus DataType enum value
        """
        type_map = {
            'text': DataType.VARCHAR,
            'keyword': DataType.VARCHAR,
            'integer': DataType.INT64,
            'int': DataType.INT64,
            'int8': DataType.INT8,
            'int16': DataType.INT16,
            'int32': DataType.INT32,
            'int64': DataType.INT64,
            'float': DataType.FLOAT,
            'double': DataType.DOUBLE,
            'boolean': DataType.BOOL,
            'bool': DataType.BOOL,
        }
        return type_map.get(field_type.lower(), DataType.VARCHAR)
    
    def _get_milvus_index_type(self, field_type: str) -> str:
        """
        Get appropriate Milvus index type for field type.
        
        Args:
            field_type: Field type string
        
        Returns:
            Milvus index type string
        """
        # Milvus supports different scalar index types
        if field_type.lower() in ['text', 'keyword']:
            return "TRIE"  # Trie index for strings
        elif field_type.lower() in ['integer', 'int', 'int64', 'int8', 'int16', 'int32']:
            return "STL_SORT"  # Sorted index for integers
        elif field_type.lower() in ['float', 'double']:
            return "STL_SORT"  # Sorted index for floats
        elif field_type.lower() in ['boolean', 'bool']:
            return "INVERTED"  # Inverted index for booleans
        else:
            return "TRIE"  # Default to TRIE for unknown types

    async def delete_collection(self) -> None:
        """Delete the collection."""
        if not await self.collection_exists():
            info_log(f"Collection '{self._config.collection_name}' does not exist.", context="MilvusProvider")
            return
        
        info_log(f"Deleting collection '{self._config.collection_name}'...", context="MilvusProvider")
        await self._aclient.drop_collection(self._config.collection_name)
        info_log(f"Collection '{self._config.collection_name}' deleted.", context="MilvusProvider")

    # ============================================================================
    # Data Operations (Async-First)
    # ============================================================================

    async def upsert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """
        Upsert data into Milvus.
        
        Args:
            vectors: Dense vector embeddings
            payloads: Metadata payloads (must contain 'content' field)
            ids: List of IDs (used as content_id if provided, otherwise auto-generated)
            chunks: Optional text chunks (used as 'content' if provided)
            sparse_vectors: Optional sparse vectors for hybrid search
            **kwargs: Additional options (metadata, etc.)
        """
        if not vectors or not payloads:
            raise ValueError("vectors and payloads cannot be empty")
        
        if len(vectors) != len(payloads):
            raise ValueError(f"vectors ({len(vectors)}) and payloads ({len(payloads)}) must have same length")
        
        # Validate sparse vectors if provided
        if sparse_vectors:
            if not self._config.use_sparse_vectors:
                raise ValueError("sparse_vectors provided but use_sparse_vectors is False in config")
            if len(sparse_vectors) != len(vectors):
                raise ValueError(f"sparse_vectors ({len(sparse_vectors)}) must match vectors ({len(vectors)})")
        
        info_log(f"Upserting {len(vectors)} records into '{self._config.collection_name}'", context="MilvusProvider")
        
        # Get additional metadata from kwargs
        additional_metadata = kwargs.get('metadata', {})
        
        # Prepare data for upsert
        data = []
        for i in range(len(vectors)):
            payload = payloads[i]
            
            # Extract core fields
            content = chunks[i] if chunks else payload.get('content', '')
            if not content:
                raise ValueError(f"Record {i} missing required 'content' field")
            
            # Generate or use provided content_id
            if ids and i < len(ids):
                content_id = str(ids[i])
            elif self._config.auto_generate_content_id:
                content_id = self._generate_content_id(content)
            else:
                content_id = payload.get('content_id', self._generate_content_id(content))
            
            document_name = payload.get('document_name', '')
            document_id = payload.get('document_id', '')
            
            # Merge metadata: default_metadata + payload metadata + additional metadata
            metadata = {}
            if self._config.default_metadata:
                metadata.update(self._config.default_metadata)
            if 'metadata' in payload:
                metadata.update(payload['metadata'])
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Build record
            record = {
                "content_id": content_id,
                "document_name": document_name,
                "document_id": document_id,
                "content": content,
                "metadata": json.dumps(metadata),  # Store as JSON string
                self._config.dense_vector_field: vectors[i],
            }
            
            # Add sparse vector if provided
            if sparse_vectors and i < len(sparse_vectors):
                record[self._config.sparse_vector_field] = sparse_vectors[i]
            
            data.append(record)
        
        # Batch upsert
        batch_size = self._config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            await self._aclient.upsert(
                collection_name=self._config.collection_name,
                data=batch,
            )
        
        info_log(f"Upserted {len(data)} records successfully.", context="MilvusProvider")

    def _generate_content_id(self, content: str) -> str:
        """Generate a unique content ID from content."""
        # Use MD5 hash of content as ID
        return md5(content.encode('utf-8')).hexdigest()

    async def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Delete records by IDs.
        
        Args:
            ids: List of content_ids to delete
        """
        if not ids:
            return
        
        info_log(f"Deleting {len(ids)} records from '{self._config.collection_name}'", context="MilvusProvider")
        
        # Convert IDs to strings
        str_ids = [str(id) for id in ids]
        
        # Delete by IDs
        await self._aclient.delete(
            collection_name=self._config.collection_name,
            ids=str_ids,
        )
        
        info_log(f"Deleted {len(ids)} records.", context="MilvusProvider")

    async def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Fetch records by IDs.
        
        Args:
            ids: List of content_ids to fetch
            
        Returns:
            List of VectorSearchResult objects
        """
        if not ids:
            return []
        
        str_ids = [str(id) for id in ids]
        
        # Fetch records
        results = await self._aclient.get(
            collection_name=self._config.collection_name,
            ids=str_ids,
        )
        
        # Convert to VectorSearchResult
        search_results = []
        for result in results:
            search_results.append(self._convert_to_search_result(result))
        
        return search_results

    # ============================================================================
    # Search Operations (Async-First)
    # ============================================================================

    async def search(
        self,
        top_k: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Master search method that dispatches to appropriate search function.
        
        Args:
            top_k: Number of results to return
            query_vector: Dense vector for vector search
            query_text: Text for full-text search (requires sparse vectors)
            filter: Metadata filter
            alpha: Hybrid search weighting (0.0 = sparse only, 1.0 = dense only)
            fusion_method: Ranking method for hybrid search ('rrf' or 'weighted')
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of VectorSearchResult objects
        """
        # Determine search type
        has_vector = query_vector is not None
        has_text = query_text is not None
        
        # Use config defaults if not provided
        top_k = top_k if top_k is not None else self._config.default_top_k
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        
        # Dispatch to appropriate search method
        if has_vector and has_text:
            # Hybrid search
            if not self._config.hybrid_search_enabled:
                raise ValueError("Hybrid search is disabled in config")
            return await self.hybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                alpha=alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        elif has_vector:
            # Dense vector search
            if not self._config.dense_search_enabled:
                raise ValueError("Dense search is disabled in config")
            return await self.dense_search(
                query_vector=query_vector,
                top_k=top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        elif has_text:
            # Full-text search (requires sparse vectors)
            if not self._config.full_text_search_enabled:
                raise ValueError("Full-text search is disabled in config")
            return await self.full_text_search(
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        else:
            raise ValueError("Either query_vector or query_text must be provided")

    async def dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Perform dense vector search.
        
        Args:
            query_vector: Dense query vector
            top_k: Number of results
            filter: Metadata filter
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of VectorSearchResult objects
        """
        info_log(f"Performing dense search (top_k={top_k})", context="MilvusProvider")
        
        # Use config default if not provided
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        
        # Build search params
        search_params = self._build_search_params()
        
        # Build filter expression
        filter_expr = self._build_filter_expression(filter) if filter else None
        
        # Perform search
        results = await self._aclient.search(
            collection_name=self._config.collection_name,
            data=[query_vector],
            anns_field=self._config.dense_vector_field,
            limit=top_k,
            output_fields=["*"],
            search_params=search_params,
            filter=filter_expr,
        )
        
        # Convert results
        search_results = []
        for hits in results:
            for hit in hits:
                result = self._convert_to_search_result(hit, final_similarity_threshold)
                if result:
                    search_results.append(result)
        
        info_log(f"Found {len(search_results)} results", context="MilvusProvider")
        return search_results

    async def full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Perform full-text search using sparse vectors.
        
        Note: sparse_vector must be provided in kwargs as this provider
        does not generate embeddings internally.
        
        Args:
            query_text: Query text (metadata only, sparse vector must be in kwargs)
            top_k: Number of results
            filter: Metadata filter
            similarity_threshold: Minimum similarity score
            **kwargs: Must contain 'sparse_vector' key
            
        Returns:
            List of VectorSearchResult objects
        """
        if not self._config.use_sparse_vectors:
            raise ValueError("Full-text search requires use_sparse_vectors=True in config")
        
        # Sparse vector must be provided externally
        sparse_vector = kwargs.get('sparse_vector')
        if sparse_vector is None:
            raise ValueError("sparse_vector must be provided in kwargs for full-text search")
        
        info_log(f"Performing full-text search (top_k={top_k})", context="MilvusProvider")
        
        # Use config default if not provided
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        
        # Build search params for sparse vectors
        search_params = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
        
        # Build filter expression
        filter_expr = self._build_filter_expression(filter) if filter else None
        
        # Perform search
        results = await self._aclient.search(
            collection_name=self._config.collection_name,
            data=[sparse_vector],
            anns_field=self._config.sparse_vector_field,
            limit=top_k,
            output_fields=["*"],
            search_params=search_params,
            filter=filter_expr,
        )
        
        # Convert results
        search_results = []
        for hits in results:
            for hit in hits:
                result = self._convert_to_search_result(hit, final_similarity_threshold)
                if result:
                    search_results.append(result)
        
        info_log(f"Found {len(search_results)} results", context="MilvusProvider")
        return search_results

    async def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Note: sparse_vector must be provided in kwargs.
        
        Args:
            query_vector: Dense query vector
            query_text: Query text (metadata only)
            top_k: Number of results
            filter: Metadata filter
            alpha: Weighting for dense vs sparse (0.0=sparse, 1.0=dense)
            fusion_method: 'rrf' or 'weighted'
            similarity_threshold: Minimum similarity score
            **kwargs: Must contain 'sparse_vector' key
            
        Returns:
            List of VectorSearchResult objects ranked by hybrid score
        """
        if not self._config.use_sparse_vectors:
            raise ValueError("Hybrid search requires use_sparse_vectors=True in config")
        
        # Sparse vector must be provided externally
        sparse_vector = kwargs.get('sparse_vector')
        if sparse_vector is None:
            raise ValueError("sparse_vector must be provided in kwargs for hybrid search")
        
        info_log(f"Performing hybrid search (top_k={top_k})", context="MilvusProvider")
        
        # Use config defaults if not provided
        alpha = alpha if alpha is not None else self._config.default_hybrid_alpha
        fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        
        # Build search params
        dense_search_params = self._build_search_params()
        sparse_search_params = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
        
        # Build filter expression
        filter_expr = self._build_filter_expression(filter) if filter else None
        
        # Create search requests for both dense and sparse
        dense_request = AnnSearchRequest(
            data=[query_vector],
            anns_field=self._config.dense_vector_field,
            param=dense_search_params,
            limit=top_k * 2,  # Fetch more for better reranking
        )
        
        sparse_request = AnnSearchRequest(
            data=[sparse_vector],
            anns_field=self._config.sparse_vector_field,
            param=sparse_search_params,
            limit=top_k * 2,
        )
        
        # Create ranker
        if fusion_method == 'rrf':
            ranker = RRFRanker(self._config.rrf_k)
        else:  # weighted
            # Alpha: weight for dense (1-alpha: weight for sparse)
            ranker = WeightedRanker(alpha, 1 - alpha)
        
        # Perform hybrid search
        results = await self._aclient.hybrid_search(
            collection_name=self._config.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=top_k,
            output_fields=["*"],
            filter=filter_expr,
        )
        
        # Convert results
        search_results = []
        for hits in results:
            for hit in hits:
                result = self._convert_to_search_result(hit, final_similarity_threshold)
                if result:
                    search_results.append(result)
        
        info_log(f"Found {len(search_results)} hybrid results", context="MilvusProvider")
        return search_results

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _build_search_params(self) -> Dict[str, Any]:
        """Build search parameters from config."""
        if self._config.search_params:
            return self._config.search_params
        
        # Build based on index type
        index_config = self._config.index
        params = {"metric_type": self._metric_type}
        
        if index_config.type == 'HNSW':
            ef_search = index_config.ef_search or max(index_config.ef_construction, 100)
            params["params"] = {"ef": ef_search}
        elif index_config.type == 'IVF_FLAT':
            nprobe = index_config.nprobe or min(index_config.nlist, 10)
            params["params"] = {"nprobe": nprobe}
        else:  # FLAT
            params["params"] = {}
        
        return params

    def _build_filter_expression(self, filter: Dict[str, Any]) -> str:
        """
        Build Milvus filter expression from filter dict.
        
        Supports:
        - Equality: {"document_id": "123"}
        - AND conditions: {"document_id": "123", "document_name": "test"}
        - Nested metadata: {"metadata.key": "value"} -> searches in JSON field
        """
        if not filter:
            return ""
        
        expressions = []
        
        for key, value in filter.items():
            if key in ['document_name', 'document_id', 'content_id', 'content']:
                # Direct field access
                if isinstance(value, str):
                    expressions.append(f'{key} == "{value}"')
                elif isinstance(value, (int, float)):
                    expressions.append(f'{key} == {value}')
                elif isinstance(value, bool):
                    expressions.append(f'{key} == {str(value).lower()}')
            else:
                # Metadata field (stored as JSON)
                # For simplicity, we search in the JSON string
                # Note: Milvus supports JSON field queries, but requires JSON field type
                if isinstance(value, str):
                    expressions.append(f'json_contains(metadata, "{{\\"{key}\\": \\"{value}\\"}}")')
        
        return " and ".join(expressions) if expressions else ""

    def _convert_to_search_result(
        self,
        hit: Dict[str, Any],
        similarity_threshold: Optional[float] = None
    ) -> Optional[VectorSearchResult]:
        """Convert Milvus hit to VectorSearchResult."""
        # Extract fields
        entity = hit.get('entity', hit)  # Support both formats
        
        # Get score/distance
        score = hit.get('distance', 0.0)
        
        # Apply similarity threshold if set
        if similarity_threshold is not None and score < similarity_threshold:
            return None
        
        # Parse metadata
        metadata_str = entity.get('metadata', '{}')
        try:
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        except json.JSONDecodeError:
            metadata = {}
        
        # Get vector (may be dense or sparse)
        vector = entity.get(self._config.dense_vector_field)
        
        return VectorSearchResult(
            id=entity.get('content_id', ''),
            score=score,
            payload={
                'content': entity.get('content', ''),
                'document_name': entity.get('document_name', ''),
                'document_id': entity.get('document_id', ''),
                'content_id': entity.get('content_id', ''),
                'metadata': metadata,
            },
            vector=vector,
        )



    async def delete_by_filter(self, filter: Dict[str, Any]) -> None:
        """
        Delete records matching a filter.
        
        Args:
            filter: Filter dict to match records for deletion
        """
        filter_expr = self._build_filter_expression(filter)
        if not filter_expr:
            raise ValueError("Invalid filter expression")
        
        info_log(f"Deleting records matching filter: {filter_expr}", context="MilvusProvider")
        
        await self._aclient.delete(
            collection_name=self._config.collection_name,
            filter=filter_expr,
        )

    async def delete_single_id(self, id: Union[str, int]) -> bool:
        """
        Delete a single record by its ID (convenience method).
        
        Note: For base class compatibility, use delete() with a list of ids.
        
        Args:
            id: The content_id to delete
            
        Returns:
            bool: True if record was deleted, False if it didn't exist
        """
        try:
            if not await self.id_exists(str(id)):
                info_log(f"Record with ID '{id}' does not exist.", context="MilvusProvider")
                return False
            
            await self._aclient.delete(
                collection_name=self._config.collection_name,
                ids=[str(id)],
            )
            info_log(f"Deleted record with ID '{id}'", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error deleting record with ID {id}: {e}", context="MilvusProvider")
            return False

    def delete_by_document_name(self, document_name: str) -> bool:
        """Delete records by document name (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_name(document_name))
    
    async def async_delete_by_document_name(self, document_name: str) -> bool:
        """
        Delete records by document name (async).
        
        Args:
            document_name: The document_name to match for deletion
            
        Returns:
            bool: True if records were deleted, False otherwise
        """
        try:
            if not await self.async_document_name_exists(document_name):
                info_log(f"No records with document_name '{document_name}' found.", context="MilvusProvider")
                return False
            
            filter_expr = f'document_name == "{document_name}"'
            await self._aclient.delete(
                collection_name=self._config.collection_name,
                filter=filter_expr,
            )
            info_log(f"Deleted records with document_name '{document_name}'", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error deleting records with document_name {document_name}: {e}", context="MilvusProvider")
            return False

    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete records by document ID (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_id(document_id))
    
    async def async_delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete records by document ID.
        
        Args:
            document_id: The document_id to match for deletion
            
        Returns:
            bool: True if records were deleted, False otherwise
        """
        try:
            filter_expr = f'document_id == "{document_id}"'
            await self._aclient.delete(
                collection_name=self._config.collection_name,
                filter=filter_expr,
            )
            info_log(f"Deleted records with document_id '{document_id}'", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error deleting records with document_id {document_id}: {e}", context="MilvusProvider")
            return False
    
    def delete_by_content_id(self, content_id: str) -> bool:
        """Delete records by content_id (sync)."""
        return self._run_async_from_sync(self.async_delete_by_content_id(content_id))
    
    async def async_delete_by_content_id(self, content_id: str) -> bool:
        """Delete records by content_id (async)."""
        try:
            filter_expr = f'content_id == "{content_id}"'
            await self._aclient.delete(
                collection_name=self._config.collection_name,
                filter=filter_expr,
            )
            info_log(f"Deleted records with content_id '{content_id}'", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error deleting records with content_id {content_id}: {e}", context="MilvusProvider")
            return False

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete records by metadata filter (sync)."""
        return self._run_async_from_sync(self.async_delete_by_metadata(metadata))
    
    async def async_delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Delete records by metadata filter (async).
        
        Args:
            metadata: Metadata dict to match for deletion
            
        Returns:
            bool: True if records were deleted, False otherwise
        """
        try:
            filter_expr = self._build_filter_expression(metadata)
            if not filter_expr:
                error_log("Invalid metadata filter", context="MilvusProvider")
                return False
            
            await self._aclient.delete(
                collection_name=self._config.collection_name,
                filter=filter_expr,
            )
            info_log(f"Deleted records matching metadata: {metadata}", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error deleting records with metadata {metadata}: {e}", context="MilvusProvider")
            return False

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific content_id (sync)."""
        return self._run_async_from_sync(self.async_update_metadata(content_id, metadata))
    
    async def async_update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific content_id (async).
        
        This operation fetches the record, merges the metadata, and upserts it back.
        
        Args:
            content_id: The content_id to update
            metadata: New metadata to merge with existing metadata
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Fetch existing record
            results = await self._aclient.get(
                collection_name=self._config.collection_name,
                ids=[content_id],
            )
            
            if not results:
                info_log(f"No record found with content_id: {content_id}", context="MilvusProvider")
                return False
            
            record = results[0]
            
            # Parse existing metadata
            existing_metadata_str = record.get('metadata', '{}')
            try:
                existing_metadata = json.loads(existing_metadata_str) if isinstance(existing_metadata_str, str) else existing_metadata_str
            except json.JSONDecodeError:
                existing_metadata = {}
            
            # Merge metadata
            updated_metadata = existing_metadata.copy()
            updated_metadata.update(metadata)
            
            # Update record
            updated_record = {
                "content_id": content_id,
                "document_name": record.get('document_name', ''),
                "document_id": record.get('document_id', ''),
                "content": record.get('content', ''),
                "metadata": json.dumps(updated_metadata),
                self._config.dense_vector_field: record.get(self._config.dense_vector_field),
            }
            
            # Add sparse vector if exists
            if self._config.use_sparse_vectors and self._config.sparse_vector_field in record:
                updated_record[self._config.sparse_vector_field] = record.get(self._config.sparse_vector_field)
            
            # Upsert
            await self._aclient.upsert(
                collection_name=self._config.collection_name,
                data=[updated_record],
            )
            
            info_log(f"Updated metadata for content_id: {content_id}", context="MilvusProvider")
            return True
            
        except Exception as e:
            error_log(f"Error updating metadata for content_id '{content_id}': {e}", context="MilvusProvider")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dict with collection stats including row_count
        """
        stats = await self._aclient.get_collection_stats(self._config.collection_name)
        return stats

    async def count(self) -> int:
        """
        Get total number of records in collection.
        
        Returns:
            int: Number of records
        """
        stats = await self.get_collection_stats()
        return stats.get('row_count', 0)


    def get_count(self) -> int:
        """
        Synchronous version of count().
        
        Returns:
            int: Number of records in collection
        """
        return self._run_async_from_sync(self.count())

    async def id_exists(self, id: str) -> bool:
        """
        Check if a record with the given content_id exists.
        
        Args:
            id: The content_id to check
            
        Returns:
            bool: True if record exists, False otherwise
        """
        try:
            results = await self._aclient.get(
                collection_name=self._config.collection_name,
                ids=[str(id)],
            )
            return len(results) > 0
        except Exception as e:
            logger.debug(f"Error checking ID existence: {e}")
            return False

    def document_name_exists(self, document_name: str) -> bool:
        """Check if any records with the given document_name exist (sync)."""
        return self._run_async_from_sync(self.async_document_name_exists(document_name))
    
    async def async_document_name_exists(self, document_name: str) -> bool:
        """
        Check if any records with the given document_name exist (async).
        
        Args:
            document_name: The document_name to check
            
        Returns:
            bool: True if records exist, False otherwise
        """
        try:
            filter_expr = f'document_name == "{document_name}"'
            results = await self._aclient.query(
                collection_name=self._config.collection_name,
                filter=filter_expr,
                limit=1,
            )
            return len(results) > 0
        except Exception as e:
            logger.debug(f"Error checking document_name existence: {e}")
            return False

    def document_id_exists(self, document_id: str) -> bool:
        """Check if any records with the given document_id exist (sync)."""
        return self._run_async_from_sync(self.async_document_id_exists(document_id))
    
    async def async_document_id_exists(self, document_id: str) -> bool:
        """
        Check if any records with the given document_id exist.
        
        Args:
            document_id: The document_id to check
            
        Returns:
            bool: True if records exist, False otherwise
        """
        try:
            filter_expr = f'document_id == "{document_id}"'
            results = await self._aclient.query(
                collection_name=self._config.collection_name,
                filter=filter_expr,
                limit=1,
            )
            return len(results) > 0
        except Exception as e:
            logger.debug(f"Error checking document_id existence: {e}")
            return False
    
    def content_id_exists(self, content_id: str) -> bool:
        """Check if any records with the given content_id exist (sync)."""
        return self._run_async_from_sync(self.async_content_id_exists(content_id))
    
    async def async_content_id_exists(self, content_id: str) -> bool:
        """
        Check if any records with the given content_id exist (async).
        
        Args:
            content_id: The content_id to check
            
        Returns:
            bool: True if records exist, False otherwise
        """
        try:
            filter_expr = f'content_id == "{content_id}"'
            results = await self._aclient.query(
                collection_name=self._config.collection_name,
                filter=filter_expr,
                limit=1,
            )
            return len(results) > 0
        except Exception as e:
            logger.debug(f"Error checking content_id existence: {e}")
            return False
    
    def optimize(self) -> bool:
        """Optimize the vector database (sync). Milvus automatically optimizes."""
        return True
    
    async def async_optimize(self) -> bool:
        """Optimize the vector database (async). Milvus automatically optimizes."""
        return True
    
    def get_supported_search_types(self) -> List[str]:
        """Get the supported search types for Milvus (sync)."""
        supported = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported
    
    async def async_get_supported_search_types(self) -> List[str]:
        """Get the supported search types for Milvus (async)."""
        return self.get_supported_search_types()

    async def drop(self) -> bool:
        """
        Drop (delete) the collection.
        
        Returns:
            bool: True if collection was dropped, False otherwise
        """
        try:
            if not await self.collection_exists():
                info_log(f"Collection '{self._config.collection_name}' does not exist.", context="MilvusProvider")
                return False
            
            await self._aclient.drop_collection(self._config.collection_name)
            info_log(f"Dropped collection '{self._config.collection_name}'", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error dropping collection: {e}", context="MilvusProvider")
            return False

    async def query(
        self,
        filter: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query records by filter without vector search.
        
        Args:
            filter: Metadata filter
            output_fields: Fields to return (None = all fields)
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching records
        """
        filter_expr = self._build_filter_expression(filter) if filter else ""
        
        results = await self._aclient.query(
            collection_name=self._config.collection_name,
            filter=filter_expr or None,
            output_fields=output_fields or ["*"],
            limit=limit,
            offset=offset,
        )
        
        return results

    # ============================================================================
    # Synchronous Convenience Methods
    # ============================================================================

    def upsert_sync(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """
        Synchronous upsert operation (for convenience).
        
        Note: Prefer async methods for better performance.
        """
        self._run_async_from_sync(self.upsert(vectors, payloads, ids, chunks, sparse_vectors, **kwargs))

    def search_sync(
        self,
        top_k: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Synchronous search operation (for convenience).
        
        Note: Prefer async methods for better performance.
        """
        return self._run_async_from_sync(
            self.search(top_k, query_vector, query_text, filter, alpha, fusion_method, similarity_threshold, **kwargs)
        )

    def delete_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Synchronous delete operation."""
        self._run_async_from_sync(self.delete(ids, **kwargs))

    def fetch_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Synchronous fetch operation."""
        return self._run_async_from_sync(self.fetch(ids, **kwargs))
    
    def fetch_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Synchronous fetch_by_id operation (alias for fetch_sync)."""
        return self._run_async_from_sync(self.fetch(ids, **kwargs))

    def delete_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Synchronous delete_by_id operation (alias for delete_sync)."""
        return self._run_async_from_sync(self.delete(ids, **kwargs))

    def sync_delete_by_document_name(self, name: str) -> bool:
        """Synchronous delete_by_document_name operation."""
        return self._run_async_from_sync(self.async_delete_by_document_name(name))

    def sync_delete_by_document_id(self, document_id: str) -> bool:
        """Synchronous delete_by_document_id operation."""
        return self._run_async_from_sync(self.async_delete_by_document_id(document_id))

    def sync_delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Synchronous delete_by_metadata operation."""
        return self._run_async_from_sync(self.async_delete_by_metadata(metadata))

    def sync_update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """Synchronous update_metadata operation."""
        return self._run_async_from_sync(self.async_update_metadata(content_id, metadata))

    def sync_query(
        self,
        filter: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Synchronous query operation."""
        return self._run_async_from_sync(self.query(filter, output_fields, limit, offset))
    
    # Additional sync methods for complete coverage
    
    def connect_sync(self) -> None:
        """Synchronous connect operation."""
        self._run_async_from_sync(self.connect())
    
    def disconnect_sync(self) -> None:
        """Synchronous disconnect operation."""
        self._run_async_from_sync(self.disconnect())
    
    def is_ready_sync(self) -> bool:
        """Synchronous is_ready operation."""
        return self._run_async_from_sync(self.is_ready())
    
    def collection_exists_sync(self) -> bool:
        """Synchronous collection_exists operation."""
        return self._run_async_from_sync(self.collection_exists())
    
    def create_collection_sync(self) -> None:
        """Synchronous create_collection operation."""
        self._run_async_from_sync(self.create_collection())
    
    def delete_collection_sync(self) -> None:
        """Synchronous delete_collection operation."""
        self._run_async_from_sync(self.delete_collection())
    
    def dense_search_sync(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Synchronous dense_search operation."""
        return self._run_async_from_sync(self.dense_search(query_vector, top_k, filter, similarity_threshold, **kwargs))
    
    def full_text_search_sync(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Synchronous full_text_search operation."""
        return self._run_async_from_sync(self.full_text_search(query_text, top_k, filter, similarity_threshold, **kwargs))
    
    def hybrid_search_sync(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Synchronous hybrid_search operation."""
        return self._run_async_from_sync(
            self.hybrid_search(query_vector, query_text, top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs)
        )
    
    def sync_delete_by_filter(self, filter: Dict[str, Any]) -> None:
        """Synchronous delete_by_filter operation."""
        self._run_async_from_sync(self.delete_by_filter(filter))
    
    def sync_get_collection_stats(self) -> Dict[str, Any]:
        """Synchronous get_collection_stats operation."""
        return self._run_async_from_sync(self.get_collection_stats())
    
    def sync_count(self) -> int:
        """Synchronous count operation."""
        return self._run_async_from_sync(self.count())
    
    def sync_id_exists(self, id: str) -> bool:
        """Synchronous id_exists operation."""
        return self._run_async_from_sync(self.id_exists(id))
    
    def sync_document_name_exists(self, name: str) -> bool:
        """Synchronous document_name_exists operation."""
        return self._run_async_from_sync(self.async_document_name_exists(name))
    
    def sync_document_id_exists(self, document_id: str) -> bool:
        """Synchronous document_id_exists operation."""
        return self._run_async_from_sync(self.async_document_id_exists(document_id))
    
    def sync_drop(self) -> bool:
        """Synchronous drop operation."""
        return self._run_async_from_sync(self.drop())

