from __future__ import annotations

import uuid
import hashlib
from typing import Any, Dict, List, Optional, Union, Literal, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse

try:
    from qdrant_client import AsyncQdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse
    _QDRANT_AVAILABLE = True
except ImportError:
    AsyncQdrantClient = None  # type: ignore
    models = None  # type: ignore
    UnexpectedResponse = None  # type: ignore
    _QDRANT_AVAILABLE = False


from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.utils.printing import info_log, debug_log, warning_log

from upsonic.vectordb.config import (
    QdrantConfig,
    Mode,
    DistanceMetric,
    HNSWIndexConfig,
    IVFIndexConfig,
    FlatIndexConfig
)

from upsonic.utils.package.exception import(
    VectorDBConnectionError, 
    ConfigurationError, 
    CollectionDoesNotExistError,
    VectorDBError,
    SearchError,
    UpsertError
)

from upsonic.schemas.vector_schemas import VectorSearchResult


class QdrantProvider(BaseVectorDBProvider):
    """
    A comprehensive, async-first vector database provider for Qdrant.
    
    This implementation provides:
    - Full async/await support using AsyncQdrantClient
    - Flexible payload schema with document_name, document_id, content_id, metadata, content
    - Dynamic field indexing based on configuration
    - Sparse vector support for hybrid search
    - Advanced filtering and search capabilities
    - Embedder and reranker integration for end-to-end workflows
    - Seamless integration with the framework's configuration system
    """

    def __init__(
        self,
        config: Union[QdrantConfig, Dict[str, Any]],
        reranker: Optional[Any] = None
    ):
        """
        Initializes the QdrantProvider with configuration, embedder, and reranker.
        
        Args:
            config: Either a QdrantConfig instance or a dictionary of configuration parameters
            reranker: Optional reranker instance for refining search results
        """
        if not _QDRANT_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="qdrant-client",
                install_command='pip install "upsonic[rag]"',
                feature_name="Qdrant vector database provider"
            )

        # Handle dict configuration
        if isinstance(config, dict):
            config = QdrantConfig.from_dict(config)
        
        if not isinstance(config, QdrantConfig):
            raise ConfigurationError("config must be either a QdrantConfig instance or a dictionary")
        
        if isinstance(config.index, IVFIndexConfig):
            raise ConfigurationError(
                "Qdrant provider does not support the 'IVF_FLAT' index_type. "
                "Please use 'HNSW' or 'FLAT'."
            )
        
        super().__init__(config)
        self._config: QdrantConfig = config
        self._client: Optional[AsyncQdrantClient] = None
        
        # Integration components
        self.reranker = reranker
        
        # Provider metadata
        self.provider_name = config.provider_name or f"QdrantProvider_{config.collection_name}"
        self.provider_description = config.provider_description
        self.provider_id = config.provider_id or self._generate_provider_id()
    
    # ============================================================================
    # Provider Metadata
    # ============================================================================
    
    def _generate_provider_id(self) -> str:
        """Generates a unique provider ID based on connection details and collection."""
        conn = self._config.connection
        identifier_parts = [
            conn.host or conn.url or conn.location or "local",
            str(conn.port) if conn.port else "",
            self._config.collection_name
        ]
        identifier = "#".join(filter(None, identifier_parts))
        
        import hashlib
        return hashlib.md5(identifier.encode()).hexdigest()[:16]
    
    # ============================================================================
    # ID Normalization
    # ============================================================================
    
    def _normalize_id(self, id_value: Union[str, int]) -> Union[str, int]:
        """
        Normalizes an ID to a format accepted by Qdrant (UUID string or integer).
        
        Args:
            id_value: The ID to normalize (can be string or int)
            
        Returns:
            Either a valid UUID string or an integer
        """
        if isinstance(id_value, int):
            return id_value
        
        try:
            uuid_obj = uuid.UUID(str(id_value))
            return str(uuid_obj)
        except ValueError:
            # Not a valid UUID, convert to deterministic integer using hash
            hash_obj = hashlib.md5(str(id_value).encode())
            return int.from_bytes(hash_obj.digest()[:8], byteorder='big', signed=False)
    
    def _generate_content_id(self, content: str) -> str:
        """
        Generates a unique content_id from content using MD5 hash.
        
        Args:
            content: The content string
            
        Returns:
            A hex string representing the content hash
        """
        cleaned_content = content.replace("\x00", "\ufffd")
        return hashlib.md5(cleaned_content.encode()).hexdigest()
    
    # ============================================================================
    # Connection Management
    # ============================================================================
    
    async def connect(self) -> None:
        """
        Establishes an async connection to the Qdrant vector database.
        
        Uses all available connection parameters including:
        - Standard: host, port, api_key
        - Advanced: grpc_port, prefer_grpc, https, prefix, timeout
        - Alternative: url, location
        
        Raises:
            VectorDBConnectionError: If the connection fails for any reason.
        """
        if self._is_connected and self._client is not None:
            info_log("Already connected to Qdrant.", context="QdrantVectorDB")
            return

        conn = self._config.connection
        try:
            # Handle special location strings (e.g., ":memory:")
            if conn.location:
                self._client = AsyncQdrantClient(location=conn.location)
            
            # Handle different modes
            elif conn.mode == Mode.IN_MEMORY:
                self._client = AsyncQdrantClient(":memory:")
            
            elif conn.mode == Mode.EMBEDDED:
                if not conn.db_path:
                    raise ConfigurationError("'db_path' must be set for embedded mode.")
                self._client = AsyncQdrantClient(path=conn.db_path)
            
            elif conn.mode == Mode.LOCAL:
                # Determine grpc_port
                grpc_port = conn.grpc_port
                if grpc_port is None and conn.port:
                    grpc_port = conn.port + 1
                elif grpc_port is None:
                    grpc_port = 6334
                
                client_kwargs = {
                    "host": conn.host or "localhost",
                    "port": conn.port or 6333,
                    "grpc_port": grpc_port,
                    "prefer_grpc": conn.prefer_grpc,
                }
                
                # Add optional parameters if specified
                if conn.https is not None:
                    client_kwargs["https"] = conn.https
                if conn.prefix:
                    client_kwargs["prefix"] = conn.prefix
                if conn.timeout is not None:
                    client_kwargs["timeout"] = int(conn.timeout) if conn.timeout else None
                
                self._client = AsyncQdrantClient(**client_kwargs)
            
            elif conn.mode == Mode.CLOUD:
                # Use full URL if provided, otherwise construct from host
                target_url = conn.url or conn.host
                if target_url and ":6333" in target_url:
                    target_url = target_url.replace(":6333", "")
                
                client_kwargs = {
                    "url": target_url,
                    "api_key": conn.api_key.get_secret_value() if conn.api_key else None,
                }
                
                # Add optional parameters
                if conn.prefer_grpc:
                    client_kwargs["prefer_grpc"] = conn.prefer_grpc
                if conn.prefix:
                    client_kwargs["prefix"] = conn.prefix
                if conn.timeout is not None:
                    client_kwargs["timeout"] = int(conn.timeout) if conn.timeout else None
                
                self._client = AsyncQdrantClient(**client_kwargs)
            
            else:
                raise ConfigurationError(f"Unsupported mode for Qdrant: {conn.mode.value}")

            self._is_connected = True
            info_log(f"Successfully connected to Qdrant (async) - {self.provider_name}", context="QdrantVectorDB")

        except Exception as e:
            self._client = None
            self._is_connected = False
            raise VectorDBConnectionError(f"Failed to connect to Qdrant: {e}") from e
    
    async def disconnect(self) -> None:
        """
        Gracefully terminates the async connection to Qdrant.
        Alias for close() for framework compatibility.
        """
        await self.close()
    
    def disconnect_sync(self) -> None:
        """
        Gracefully terminates the connection to the vector database (sync).
        """
        self._run_async_from_sync(self.disconnect())
    
    async def close(self) -> None:
        """
        Close the async Qdrant client connection properly.
        This is the recommended method for cleanup.
        """
        if self._client:
            try:
                await self._client.close()
                from upsonic.utils.printing import success_log
                success_log("Successfully closed Qdrant connection.", "QdrantProvider")
            except Exception as e:
                from upsonic.utils.printing import error_log
                error_log(f"Error closing Qdrant connection: {e}", "QdrantProvider")
            finally:
                self._client = None
                self._is_connected = False
    
    async def is_ready(self) -> bool:
        """
        Performs a health check to ensure the Qdrant instance is responsive.
        """
        if not self._is_connected or not self._client:
            return False
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False
    
    def connect_sync(self) -> None:
        """
        Establishes a connection to the vector database (sync).
        """
        self._run_async_from_sync(self.connect())
    
    def is_ready_sync(self) -> bool:
        """
        Performs a health check to ensure the database is responsive (sync).
        """
        return self._run_async_from_sync(self.is_ready())
    
    # ============================================================================
    # Collection Management
    # ============================================================================
    
    async def create_collection(self) -> None:
        """
        Creates the collection in Qdrant with full configuration support.
        
        Supports:
        - Dense vectors only
        - Dense + Sparse vectors (for hybrid search)
        - Named vectors when sparse vectors are enabled
        - Dynamic payload indexing based on indexed_fields config
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Qdrant to create a collection.")

        collection_name = self._config.collection_name
        
        try:
            if self._config.recreate_if_exists and await self.collection_exists():
                info_log(f"Collection '{collection_name}' exists and `recreate_if_exists` is True. Deleting...", context="QdrantVectorDB")
                await self.delete_collection()
            
            # Map distance metrics
            distance_map = {
                DistanceMetric.COSINE: models.Distance.COSINE,
                DistanceMetric.EUCLIDEAN: models.Distance.EUCLID,
                DistanceMetric.DOT_PRODUCT: models.Distance.DOT,
            }
            
            # Configure vectors (named if sparse enabled, otherwise simple)
            if self._config.use_sparse_vectors:
                vectors_config = {
                    self._config.dense_vector_name: models.VectorParams(
                        size=self._config.vector_size,
                        distance=distance_map[self._config.distance_metric]
                    )
                }
                sparse_vectors_config = {
                    self._config.sparse_vector_name: models.SparseVectorParams()
                }
            else:
                vectors_config = models.VectorParams(
                    size=self._config.vector_size,
                    distance=distance_map[self._config.distance_metric]
                )
                sparse_vectors_config = None
            
            # Configure HNSW parameters
            hnsw_config = None
            index_cfg = self._config.index
            if isinstance(index_cfg, HNSWIndexConfig):
                hnsw_config = models.HnswConfigDiff(
                    m=index_cfg.m,
                    ef_construct=index_cfg.ef_construction
                )
            
            # Configure quantization
            quantization_config = None
            if self._config.quantization_config:
                quant_cfg = self._config.quantization_config
                if quant_cfg.get('type') == 'scalar':
                    quantization_config = models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True
                        )
                    )

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
                shard_number=self._config.shard_number,
                replication_factor=self._config.replication_factor,
                on_disk_payload=self._config.on_disk_payload
            )
            
            info_log(f"Successfully created collection '{collection_name}'.", context="QdrantVectorDB")
            
            # Create indexes for specified fields
            await self._create_field_indexes(collection_name)

        except Exception as e:
            raise VectorDBError(f"Failed to create collection '{collection_name}': {e}") from e
    
    def create_collection_sync(self) -> None:
        """
        Creates the collection in the database according to the full config (sync).
        """
        self._run_async_from_sync(self.create_collection())
    
    async def delete_collection(self) -> None:
        """
        Permanently deletes the collection specified in the config.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Qdrant to delete a collection.")

        collection_name = self._config.collection_name
        try:
            result = await self._client.delete_collection(collection_name=collection_name)
            if isinstance(result, bool):
                if not result and not await self.collection_exists():
                    raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.")
            else:
                if hasattr(result, 'result') and not result.result:
                    if not await self.collection_exists():
                        raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.")
                    
            info_log(f"Successfully deleted collection '{collection_name}'.", context="QdrantVectorDB")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.") from e
            raise VectorDBError(f"API error while deleting collection '{collection_name}': {e}") from e
        except Exception as e:
            raise VectorDBError(f"An unexpected error occurred while deleting collection: {e}") from e
    
    def delete_collection_sync(self) -> None:
        """
        Permanently deletes the collection specified in `self._config.collection_name` (sync).
        """
        self._run_async_from_sync(self.delete_collection())
    
    async def collection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Qdrant to check for a collection.")
            
        collection_name = self._config.collection_name
        try:
            await self._client.get_collection(collection_name=collection_name)
            return True
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return False
            raise VectorDBError(f"API error while checking for collection '{collection_name}': {e}") from e
        except Exception:
            return False
    
    def collection_exists_sync(self) -> bool:
        """
        Checks if the collection specified in the config already exists (sync).
        """
        return self._run_async_from_sync(self.collection_exists())
    
    # ============================================================================
    # Payload & Index Management
    # ============================================================================
    
    def _build_payload(
        self,
        content: str,
        document_name: Optional[str] = None,
        document_id: Optional[str] = None,
        content_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Builds a standardized payload structure.
        
        Args:
            content: The text content (required)
            document_name: Optional document name
            document_id: Optional document ID
            content_id: Optional content ID (generated if not provided)
            metadata: Optional custom metadata dict
            **kwargs: Additional fields to include in payload
            
        Returns:
            A dictionary with the structured payload
        """
        # Generate content_id if not provided
        if content_id is None:
            content_id = self._generate_content_id(content)
        
        # Start with base payload
        payload = {
            "content": content,
            "content_id": content_id,
        }
        
        # Add optional fields
        if document_name:
            payload["document_name"] = document_name
        if document_id:
            payload["document_id"] = document_id
        
        # Merge metadata
        combined_metadata = {}
        if self._config.default_metadata:
            combined_metadata.update(self._config.default_metadata)
        if metadata:
            combined_metadata.update(metadata)
        
        if combined_metadata:
            payload["metadata"] = combined_metadata
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        return payload
    
    async def _create_field_indexes(self, collection_name: str) -> None:
        """
        Creates indexes for fields specified in config.
        
        Supports two modes:
        1. Advanced: payload_field_configs (explicit types and params)
        2. Simple: indexed_fields (auto-determines types)
        
        Standard indexable fields:
        - content (text)
        - document_name (keyword)
        - document_id (keyword)
        - content_id (keyword)
        - metadata.* (various types)
        """
        # Priority 1: Use advanced payload_field_configs if provided
        if self._config.payload_field_configs:
            debug_log(f"Creating indexes from payload_field_configs: {len(self._config.payload_field_configs)} fields", context="QdrantVectorDB")
            
            for field_config in self._config.payload_field_configs:
                if not field_config.indexed:
                    continue  # Skip fields that are not marked for indexing
                
                try:
                    # Map field_type to Qdrant schema
                    field_schema = self._get_qdrant_schema(field_config.field_type, field_config.params)
                    
                    debug_log(f"Creating index for field '{field_config.field_name}' (type: {field_config.field_type})...", context="QdrantVectorDB")
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_config.field_name,
                        field_schema=field_schema,
                        wait=True
                    )
                except Exception as e:
                    warning_log(f"Failed to create index for field '{field_config.field_name}': {e}", context="QdrantVectorDB")
            
            info_log("Field indexes created successfully from payload_field_configs.", context="QdrantVectorDB")
            return
        
        # Priority 2: Use simple indexed_fields (auto-determine types)
        if self._config.indexed_fields:
            debug_log(f"Creating indexes for fields: {self._config.indexed_fields}", context="QdrantVectorDB")
            
            # Default field type mapping
            field_type_map = {
                'content': models.TextIndexParams(type='text'),
                'document_name': models.KeywordIndexParams(type='keyword'),
                'document_id': models.KeywordIndexParams(type='keyword'),
                'content_id': models.KeywordIndexParams(type='keyword'),
            }
            
            for field_name in self._config.indexed_fields:
                try:
                    # Determine schema type
                    if field_name in field_type_map:
                        field_schema = field_type_map[field_name]
                    elif field_name.startswith('metadata.'):
                        # Default to keyword for metadata fields
                        field_schema = models.KeywordIndexParams(type='keyword')
                    else:
                        # Default to keyword for unknown fields
                        field_schema = models.KeywordIndexParams(type='keyword')
                    
                    debug_log(f"Creating index for field '{field_name}'...", context="QdrantVectorDB")
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True
                    )
                except Exception as e:
                    warning_log(f"Failed to create index for field '{field_name}': {e}", context="QdrantVectorDB")
            
            info_log("Field indexes created successfully from indexed_fields.", context="QdrantVectorDB")
            return
        
        # No indexing configured
        debug_log("No indexed_fields or payload_field_configs specified, skipping index creation.", context="QdrantVectorDB")
    
    def _get_qdrant_schema(self, field_type: str, params: Optional[Dict[str, Any]] = None):
        """
        Convert field_type string to Qdrant schema object.
        
        Args:
            field_type: One of 'text', 'keyword', 'integer', 'float', 'boolean', 'geo'
            params: Optional custom parameters for the index
            
        Returns:
            Qdrant schema object
        """
        if field_type == 'text':
            return models.TextIndexParams(type='text', **params) if params else models.TextIndexParams(type='text')
        elif field_type == 'keyword':
            return models.KeywordIndexParams(type='keyword', **params) if params else models.KeywordIndexParams(type='keyword')
        elif field_type == 'integer':
            return models.IntegerIndexParams(type='integer', **params) if params else models.IntegerIndexParams(type='integer')
        elif field_type == 'float':
            return models.FloatIndexParams(type='float', **params) if params else models.FloatIndexParams(type='float')
        elif field_type == 'boolean':
            return models.BoolIndexParams(type='bool', **params) if params else models.BoolIndexParams(type='bool')
        elif field_type == 'geo':
            return models.GeoIndexParams(type='geo', **params) if params else models.GeoIndexParams(type='geo')
        else:
            # Default to keyword
            return models.KeywordIndexParams(type='keyword')
    
    # ============================================================================
    # Data Operations
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
        Adds or updates data in the Qdrant collection.
        
        This method supports:
        - Dense vectors only
        - Dense + Sparse vectors for hybrid search
        - Structured payloads with document_name, document_id, content_id, metadata, content
        - Custom metadata via payloads or config
        
        Args:
            vectors: List of dense vector embeddings
            payloads: List of payload dicts (can include document_name, document_id, content, metadata, etc.)
            ids: List of unique identifiers
            chunks: Optional list of text chunks (deprecated, use 'content' in payloads)
            sparse_vectors: Optional list of sparse vectors for hybrid search
                           Each should be {'indices': [...], 'values': [...]}
            **kwargs: Additional options
            
        Raises:
            UpsertError: If the data ingestion fails
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to upsert data.")
        
        if not (len(vectors) == len(payloads) == len(ids)):
            raise ValueError("The lengths of vectors, payloads, and ids lists must be identical.")
        
        if chunks is not None and len(chunks) != len(vectors):
            raise ValueError("The length of the chunks list must be identical to the other lists.")
        
        if sparse_vectors is not None:
            if not self._config.use_sparse_vectors:
                raise ConfigurationError("Sparse vectors provided but use_sparse_vectors is False in config")
            if len(sparse_vectors) != len(vectors):
                raise ValueError("The length of sparse_vectors must match the other lists.")
        
        points = []
        
        # TODO: HANDLE SPARSE VECTORS IN THE LOOP CORRECTLY OR OUTSIDE OF THE LOOP! (DIMENSIONS HAS TO MATCH?)
        for i, (point_id, vector, payload) in enumerate(zip(ids, vectors, payloads)):
            # Build structured payload
            content = payload.get('content', chunks[i] if chunks else '')
            if not content:
                raise ValueError(f"Content is required for point at index {i}")
            
            structured_payload = self._build_payload(
                content=content,
                document_name=payload.get('document_name'),
                document_id=payload.get('document_id'),
                content_id=payload.get('content_id'),
                metadata=payload.get('metadata'),
                **{k: v for k, v in payload.items() if k not in ['content', 'document_name', 'document_id', 'content_id', 'metadata']}
            )
            
            # Normalize ID
            normalized_id = self._normalize_id(point_id)
            
            # Build vector structure
            if self._config.use_sparse_vectors:
                # Named vectors for hybrid search
                vector_data = {self._config.dense_vector_name: vector}
                if sparse_vectors and i < len(sparse_vectors):
                    sparse_vec = sparse_vectors[i]
                    vector_data[self._config.sparse_vector_name] = models.SparseVector(
                        indices=sparse_vec.get('indices', []),
                        values=sparse_vec.get('values', [])
                    )
            else:
                # Simple vector
                vector_data = vector
            
            points.append(
                models.PointStruct(
                    id=normalized_id,
                    vector=vector_data,
                    payload=structured_payload
                )
            )
        
        # Upsert with consistency settings
        wait_for_result = self._config.write_consistency_factor > 1
        
        try:
            await self._client.upsert(
                collection_name=self._config.collection_name,
                points=points,
                wait=wait_for_result,
            )
            debug_log(f"Successfully upserted {len(points)} points.", context="QdrantVectorDB")
        except Exception as e:
            raise UpsertError(f"Failed to upsert data into collection '{self._config.collection_name}': {e}") from e
    
    def upsert_sync(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]] = None, sparse_vectors: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        """
        Adds new data or updates existing data in the collection (sync).
        """
        self._run_async_from_sync(self.upsert(vectors, payloads, ids, chunks, sparse_vectors, **kwargs))
    
    async def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to delete data.")

        if not ids:
            return

        normalized_ids = [self._normalize_id(id_val) for id_val in ids]
        wait_for_result = self._config.write_consistency_factor > 1

        try:
            await self._client.delete(
                collection_name=self._config.collection_name,
                points_selector=models.PointIdsList(points=normalized_ids),
                wait=wait_for_result
            )
            debug_log(f"Successfully deleted {len(normalized_ids)} points.", context="QdrantVectorDB")
        except Exception as e:
            raise VectorDBError(f"Failed to delete points from collection '{self._config.collection_name}': {e}") from e
    
    def delete_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers (sync).
        """
        self._run_async_from_sync(self.delete(ids, **kwargs))
    
    async def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their unique IDs.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to fetch data.")

        if not ids:
            return []

        normalized_ids = [self._normalize_id(id_val) for id_val in ids]

        try:
            retrieved_records: List[models.Record] = await self._client.retrieve(
                collection_name=self._config.collection_name,
                ids=normalized_ids,
                with_payload=True,
                with_vectors=True
            )

            search_results = []
            for record in retrieved_records:
                # Extract vector (handle named vectors)
                vector = None
                if isinstance(record.vector, dict):
                    vector = record.vector.get(self._config.dense_vector_name)
                else:
                    vector = record.vector
                
                search_results.append(VectorSearchResult(
                    id=record.id,
                    score=1.0,
                    payload=record.payload,
                    vector=vector,
                    text=record.payload.get("content", "") if record.payload else ""
                ))
            
            return search_results

        except Exception as e:
            raise VectorDBError(f"Failed to fetch points from collection '{self._config.collection_name}': {e}") from e
    
    def fetch_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs (sync).
        """
        return self._run_async_from_sync(self.fetch(ids, **kwargs))
    
    # ============================================================================
    # Utility & Management Methods
    # ============================================================================
    
    async def get_count(self) -> int:
        """
        Get the total number of points/documents in the collection.
        
        Returns:
            int: Total count of documents
            
        Raises:
            VectorDBError: If the count operation fails
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to get count.")
        
        try:
            count_result = await self._client.count(
                collection_name=self._config.collection_name,
                exact=True
            )
            return count_result.count
        except Exception as e:
            raise VectorDBError(f"Failed to get count from collection '{self._config.collection_name}': {e}") from e
    
    async def id_exists(self, id: Union[str, int]) -> bool:
        """
        Check if a point with the given ID exists in the collection.
        
        Args:
            id: The ID to check
            
        Returns:
            bool: True if the point exists, False otherwise
        """
        if not self._is_connected or not self._client:
            return False
        
        try:
            normalized_id = self._normalize_id(id)
            points = await self._client.retrieve(
                collection_name=self._config.collection_name,
                ids=[normalized_id],
                with_payload=False,
                with_vectors=False
            )
            return len(points) > 0
        except Exception as e:
            debug_log(f"Error checking if point {id} exists: {e}", context="QdrantVectorDB")
            return False
    
    def content_id_exists(self, content_id: str) -> bool:
        """Check if any points with the given content_id exist in the collection (sync)."""
        return self._run_async_from_sync(self.async_content_id_exists(content_id))
    
    async def async_content_id_exists(self, content_id: str) -> bool:
        """
        Check if any points with the given content_id exist in the collection (async).
        
        Args:
            content_id: The content_id to check
            
        Returns:
            bool: True if points with the content_id exist, False otherwise
        """
        if not self._is_connected or not self._client:
            return False
        
        try:
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="content_id", match=models.MatchValue(value=content_id))]
            )
            count_result = await self._client.count(
                collection_name=self._config.collection_name,
                count_filter=filter_condition,
                exact=True
            )
            return count_result.count > 0
        except Exception as e:
            debug_log(f"Error checking if content_id {content_id} exists: {e}", context="QdrantVectorDB")
            return False
    
    def document_name_exists(self, document_name: str) -> bool:
        """Check if a document with the given name exists in the collection (sync)."""
        return self._run_async_from_sync(self.async_document_name_exists(document_name))
    
    async def async_document_name_exists(self, document_name: str) -> bool:
        """
        Check if a document with the given name exists in the collection (async).
        
        Args:
            document_name: The document name to check
            
        Returns:
            bool: True if a document with the given name exists, False otherwise
        """
        if not self._is_connected or not self._client:
            return False
        
        try:
            scroll_result = await self._client.scroll(
                collection_name=self._config.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="document_name", match=models.MatchValue(value=document_name))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            return len(scroll_result[0]) > 0
        except Exception as e:
            debug_log(f"Error checking if document_name {document_name} exists: {e}", context="QdrantVectorDB")
            return False
    
    def document_id_exists(self, document_id: str) -> bool:
        """Check if a document with the given document_id exists in the collection (sync)."""
        return self._run_async_from_sync(self.async_document_id_exists(document_id))
    
    async def async_document_id_exists(self, document_id: str) -> bool:
        """
        Check if a document with the given document_id exists in the collection (async).
        
        Args:
            document_id: The document ID to check
            
        Returns:
            bool: True if a document with the given document_id exists, False otherwise
        """
        if not self._is_connected or not self._client:
            return False
        
        try:
            scroll_result = await self._client.scroll(
                collection_name=self._config.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            return len(scroll_result[0]) > 0
        except Exception as e:
            debug_log(f"Error checking if document_id {document_id} exists: {e}", context="QdrantVectorDB")
            return False
    
    async def content_exists(self, content: str) -> bool:
        """
        Check if content already exists in the collection (by content hash).
        
        Args:
            content: The content text to check
            
        Returns:
            bool: True if the content exists, False otherwise
        """
        content_id = self._generate_content_id(content)
        return await self.async_content_id_exists(content_id)
    
    async def delete_by_id(self, id: Union[str, int]) -> bool:
        """
        Delete a single point by its ID.
        
        Args:
            id: The ID of the point to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self._is_connected or not self._client:
            warning_log("Not connected to Qdrant", context="QdrantVectorDB")
            return False
        
        try:
            # Check if point exists before deletion
            if not await self.id_exists(id):
                warning_log(f"Point with ID {id} does not exist", context="QdrantVectorDB")
                return True
            
            normalized_id = self._normalize_id(id)
            await self._client.delete(
                collection_name=self._config.collection_name,
                points_selector=models.PointIdsList(points=[normalized_id]),
                wait=True
            )
            debug_log(f"Successfully deleted point with ID {id}", context="QdrantVectorDB")
            return True
        except Exception as e:
            warning_log(f"Error deleting point with ID {id}: {e}", context="QdrantVectorDB")
            return False
    
    def delete_by_document_name(self, document_name: str) -> bool:
        """Delete all points that have the specified document_name in their payload (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_name(document_name))
    
    async def async_delete_by_document_name(self, document_name: str) -> bool:
        """
        Delete all points that have the specified document_name in their payload (async).
        
        Args:
            document_name: The document name to match for deletion
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self._is_connected or not self._client:
            warning_log("Not connected to Qdrant", context="QdrantVectorDB")
            return False
        
        try:
            info_log(f"Attempting to delete all points with document_name: {document_name}", context="QdrantVectorDB")
            
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="document_name", match=models.MatchValue(value=document_name))]
            )
            
            # Count how many points will be deleted
            count_result = await self._client.count(
                collection_name=self._config.collection_name,
                count_filter=filter_condition,
                exact=True
            )
            
            if count_result.count == 0:
                warning_log(f"No points found with document_name: {document_name}", context="QdrantVectorDB")
                return True
            
            info_log(f"Found {count_result.count} points to delete with document_name: {document_name}", context="QdrantVectorDB")
            
            # Delete all points matching the filter
            result = await self._client.delete(
                collection_name=self._config.collection_name,
                points_selector=filter_condition,
                wait=True
            )
            
            # Check if the deletion was successful
            if result.status == models.UpdateStatus.COMPLETED:
                info_log(f"Successfully deleted {count_result.count} points with document_name: {document_name}", context="QdrantVectorDB")
                return True
            else:
                warning_log(f"Deletion failed for document_name {document_name}. Status: {result.status}", context="QdrantVectorDB")
                return False
        except Exception as e:
            warning_log(f"Error deleting points with document_name {document_name}: {e}", context="QdrantVectorDB")
            return False
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all points that have the specified document_id in their payload (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_id(document_id))
    
    async def async_delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all points that have the specified document_id in their payload (async).
        
        Args:
            document_id: The document ID to match for deletion
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self._is_connected or not self._client:
            warning_log("Not connected to Qdrant", context="QdrantVectorDB")
            return False
        
        try:
            info_log(f"Attempting to delete all points with document_id: {document_id}", context="QdrantVectorDB")
            
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
            
            # Count how many points will be deleted
            count_result = await self._client.count(
                collection_name=self._config.collection_name,
                count_filter=filter_condition,
                exact=True
            )
            
            if count_result.count == 0:
                warning_log(f"No points found with document_id: {document_id}", context="QdrantVectorDB")
                return True
            
            info_log(f"Found {count_result.count} points to delete with document_id: {document_id}", context="QdrantVectorDB")
            
            # Delete all points matching the filter
            result = await self._client.delete(
                collection_name=self._config.collection_name,
                points_selector=filter_condition,
                wait=True
            )
            
            # Check if the deletion was successful
            if result.status == models.UpdateStatus.COMPLETED:
                info_log(f"Successfully deleted {count_result.count} points with document_id: {document_id}", context="QdrantVectorDB")
                return True
            else:
                warning_log(f"Deletion failed for document_id {document_id}. Status: {result.status}", context="QdrantVectorDB")
                return False
        except Exception as e:
            warning_log(f"Error deleting points with document_id {document_id}: {e}", context="QdrantVectorDB")
            return False
    
    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete all points where the given metadata matches (sync)."""
        return self._run_async_from_sync(self.async_delete_by_metadata(metadata))
    
    async def async_delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Delete all points where the given metadata matches (async).
        
        Args:
            metadata: Dictionary of metadata key-value pairs to match
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self._is_connected or not self._client:
            warning_log("Not connected to Qdrant", context="QdrantVectorDB")
            return False
        
        try:
            info_log(f"Attempting to delete all points with metadata: {metadata}", context="QdrantVectorDB")
            
            # Create filter conditions for each metadata key-value pair
            filter_conditions = []
            for key, value in metadata.items():
                filter_conditions.append(
                    models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
                )
            
            filter_condition = models.Filter(must=filter_conditions)
            
            # Count how many points will be deleted
            count_result = await self._client.count(
                collection_name=self._config.collection_name,
                count_filter=filter_condition,
                exact=True
            )
            
            if count_result.count == 0:
                warning_log(f"No points found with metadata: {metadata}", context="QdrantVectorDB")
                return True
            
            info_log(f"Found {count_result.count} points to delete with metadata: {metadata}", context="QdrantVectorDB")
            
            # Delete all points matching the filter
            result = await self._client.delete(
                collection_name=self._config.collection_name,
                points_selector=filter_condition,
                wait=True
            )
            
            if result.status == models.UpdateStatus.COMPLETED:
                info_log(f"Successfully deleted {count_result.count} points with metadata: {metadata}", context="QdrantVectorDB")
                return True
            else:
                warning_log(f"Deletion failed for metadata {metadata}. Status: {result.status}", context="QdrantVectorDB")
                return False
        except Exception as e:
            warning_log(f"Error deleting points with metadata {metadata}: {e}", context="QdrantVectorDB")
            return False
    
    def delete_by_content_id(self, content_id: str) -> bool:
        """Delete all points that have the specified content_id in their payload (sync)."""
        return self._run_async_from_sync(self.async_delete_by_content_id(content_id))
    
    async def async_delete_by_content_id(self, content_id: str) -> bool:
        """
        Delete all points that have the specified content_id in their payload (async).
        
        Args:
            content_id: The content_id to match for deletion
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self._is_connected or not self._client:
            warning_log("Not connected to Qdrant", context="QdrantVectorDB")
            return False
        
        try:
            info_log(f"Attempting to delete all points with content_id: {content_id}", context="QdrantVectorDB")
            
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="content_id", match=models.MatchValue(value=content_id))]
            )
            
            count_result = await self._client.count(
                collection_name=self._config.collection_name,
                count_filter=filter_condition,
                exact=True
            )
            
            if count_result.count == 0:
                warning_log(f"No points found with content_id: {content_id}", context="QdrantVectorDB")
                return True
            
            info_log(f"Found {count_result.count} points to delete with content_id: {content_id}", context="QdrantVectorDB")
            
            result = await self._client.delete(
                collection_name=self._config.collection_name,
                points_selector=filter_condition,
                wait=True
            )
            
            if result.status == models.UpdateStatus.COMPLETED:
                info_log(f"Successfully deleted {count_result.count} points with content_id: {content_id}", context="QdrantVectorDB")
                return True
            else:
                warning_log(f"Deletion failed for content_id {content_id}. Status: {result.status}", context="QdrantVectorDB")
                return False
        except Exception as e:
            warning_log(f"Error deleting points with content_id {content_id}: {e}", context="QdrantVectorDB")
            return False
    
    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Updates the metadata for a specific content ID (sync).
        
        Args:
            content_id: The content ID to update.
            metadata: The metadata to update/merge.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        return self._run_async_from_sync(self.async_update_metadata(content_id, metadata))
    
    async def async_update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Updates the metadata for a specific content ID (async).
        
        Args:
            content_id: The content ID to update.
            metadata: The metadata to update/merge.
            
        Returns:
            True if the update was successful, False otherwise.
            
        Raises:
            VectorDBError: If the update operation fails critically
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to update metadata.")
        
        try:
            # Create filter for content_id
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="content_id", match=models.MatchValue(value=content_id))]
            )
            
            # Scroll to get all points with the given content_id
            search_result = await self._client.scroll(
                collection_name=self._config.collection_name,
                scroll_filter=filter_condition,
                limit=10000,  # Get all matching points
                with_payload=True,
                with_vectors=False
            )
            
            if not search_result[0]:
                warning_log(f"No documents found with content_id: {content_id}", context="QdrantVectorDB")
                return False
            
            points = search_result[0]
            updated_count = 0
            
            # Update metadata for each point
            for point in points:
                point_id = point.id
                current_payload = point.payload or {}
                
                # Merge existing metadata with new metadata
                updated_payload = current_payload.copy()
                
                # Update the metadata field
                if "metadata" in updated_payload:
                    if isinstance(updated_payload["metadata"], dict):
                        updated_payload["metadata"].update(metadata)
                    else:
                        updated_payload["metadata"] = metadata
                else:
                    updated_payload["metadata"] = metadata
                
                # Set the updated payload
                await self._client.set_payload(
                    collection_name=self._config.collection_name,
                    payload=updated_payload,
                    points=[point_id],
                    wait=True
                )
                updated_count += 1
            
            info_log(f"Updated metadata for {updated_count} documents with content_id: {content_id}", context="QdrantVectorDB")
            return True
            
        except Exception as e:
            raise VectorDBError(f"Error updating metadata for content_id '{content_id}': {e}") from e
    
    def optimize(self) -> bool:
        """Trigger optimization of the Qdrant collection (sync)."""
        return self._run_async_from_sync(self.async_optimize())
    
    async def async_optimize(self) -> bool:
        """
        Trigger optimization of the Qdrant collection (async).
        
        This operation optimizes indexes and improves search performance.
        Useful to call periodically or after bulk operations.
        
        Note: This is a no-op for in-memory mode.
        
        Returns:
            True if optimization was successful, False otherwise
        """
        if not self._is_connected or not self._client:
            warning_log("Not connected to Qdrant", context="QdrantVectorDB")
            return False
        
        try:
            # Qdrant doesn't have explicit optimize, but we can trigger indexing
            info_log(f"Optimization requested for collection '{self._config.collection_name}'", context="QdrantVectorDB")
            
            # In Qdrant, optimization happens automatically
            # We can optionally trigger a collection info refresh
            await self._client.get_collection(collection_name=self._config.collection_name)
            
            debug_log("Collection optimization acknowledged", context="QdrantVectorDB")
            return True
        except Exception as e:
            warning_log(f"Error during optimization: {e}", context="QdrantVectorDB")
            return False
    
    def get_supported_search_types(self) -> List[str]:
        """
        Get the list of supported search types for this provider (sync).
        
        Returns:
            List of search type strings: ['dense', 'full_text', 'hybrid']
        """
        supported = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported
    
    async def async_get_supported_search_types(self) -> List[str]:
        """
        Get the list of supported search types for this provider (async).
        
        Returns:
            List of search type strings: ['dense', 'full_text', 'hybrid']
        """
        return self.get_supported_search_types()
    
    # ============================================================================
    # Search Operations
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
        Master search method that dispatches to the appropriate search type.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to perform a search.")

        effective_top_k = top_k if top_k is not None else self._config.default_top_k or 10

        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None

        if is_dense:
            if self._config.dense_search_enabled is False:
                raise ConfigurationError("Dense search is disabled by the current configuration.")
            return await self.dense_search(query_vector, effective_top_k, filter, similarity_threshold, **kwargs)
        
        elif is_hybrid:
            if self._config.hybrid_search_enabled is False:
                raise ConfigurationError("Hybrid search is disabled by the current configuration.")
            return await self.hybrid_search(query_vector, query_text, effective_top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs)

        elif is_full_text:
            if self._config.full_text_search_enabled is False:
                raise ConfigurationError("Full-text search is disabled by the current configuration.")
            return await self.full_text_search(query_text, effective_top_k, filter, similarity_threshold, **kwargs)
        
        else:
            raise SearchError("Invalid search query: You must provide a 'query_vector' and/or 'query_text'.")
    
    def search_sync(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        A master search method that dispatches to the appropriate specialized
        search function based on the provided arguments and the provider's
        configured capabilities (sync).
        """
        return self._run_async_from_sync(self.search(top_k, query_vector, query_text, filter, alpha, fusion_method, similarity_threshold, **kwargs))
    
    async def dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Performs pure vector similarity search.
        """
        try:
            final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
            
            search_params = models.SearchParams(
                hnsw_ef=kwargs.get('ef_search', getattr(self._config.index, 'ef_search', None) or 128),
                exact=False,
            )

            qdrant_filter = self._build_qdrant_filter(filter) if filter else None

            # Use query_points for consistency
            if self._config.use_sparse_vectors:
                query_response = await self._client.query_points(
                    collection_name=self._config.collection_name,
                    query=query_vector,
                    using=self._config.dense_vector_name,
                    query_filter=qdrant_filter,
                    search_params=search_params,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=True
                )
            else:
                query_response = await self._client.query_points(
                    collection_name=self._config.collection_name,
                    query=query_vector,
                    query_filter=qdrant_filter,
                    search_params=search_params,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=True
                )

            filtered_results = []
            
            for point in query_response.points:
                should_include = self._check_similarity_threshold(point.score, final_similarity_threshold)
                
                if should_include:
                    # Extract vector (handle named vectors)
                    vector = None
                    if isinstance(point.vector, dict):
                        vector = point.vector.get(self._config.dense_vector_name)
                    else:
                        vector = point.vector
                    
                    filtered_results.append(VectorSearchResult(
                        id=point.id,
                        score=point.score,
                        payload=point.payload,
                        vector=vector,
                        text=point.payload.get("content", "") if point.payload else ""
                    ))

            # Apply reranking if configured
            if self.reranker and kwargs.get('apply_reranking', True):
                filtered_results = self._apply_reranking(filtered_results, str(query_vector))
            
            return filtered_results
        except Exception as e:
            raise SearchError(f"An error occurred during dense search: {e}") from e
    
    def dense_search_sync(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a pure vector similarity search (sync).
        """
        return self._run_async_from_sync(self.dense_search(query_vector, top_k, filter, similarity_threshold, **kwargs))
    
    async def full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Performs full-text search using Qdrant's text indexing.
        
        For IN_MEMORY mode, falls back to client-side search.
        For other modes, uses server-side text indexing.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to perform a full-text search.")

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        target_text_field = kwargs.get("text_search_field", "content")

        if self._config.connection.mode == Mode.IN_MEMORY:
            # Client-side full-text search for in-memory mode
            return await self._client_side_full_text_search(query_text, top_k, filter, final_similarity_threshold, target_text_field, **kwargs)
        else:
            # Server-side full-text search
            return await self._server_side_full_text_search(query_text, top_k, filter, final_similarity_threshold, target_text_field, **kwargs)
    
    async def _client_side_full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]],
        similarity_threshold: float,
        target_text_field: str,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Client-side full-text search implementation."""
        try:
            records = await self._client.scroll(
                collection_name=self._config.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            
            query_terms = query_text.lower().split()
            matching_records = []
            
            for record in records[0]:
                if target_text_field in record.payload:
                    text_content = record.payload[target_text_field]
                    text_lower = text_content.lower()
                    text_words = text_lower.split()
                    
                    if any(term in text_lower for term in query_terms):
                        term_count = sum(text_lower.count(term) for term in query_terms)
                        doc_length = len(text_words)
                        
                        if doc_length > 0 and term_count > 0:
                            matched_terms = sum(1 for term in query_terms if term in text_lower)
                            match_ratio = matched_terms / len(query_terms)
                            
                            import math
                            term_density = term_count / doc_length
                            tf_score = math.log(1 + term_density * 5) / math.log(6)
                            
                            relevance_score = (tf_score * 0.7 + match_ratio * 0.3)
                            relevance_score = min(1.0, max(0.0, relevance_score))
                        else:
                            relevance_score = 0.0
                        
                        if relevance_score >= similarity_threshold:
                            # Extract vector (handle named vectors)
                            vector = None
                            if isinstance(record.vector, dict):
                                vector = record.vector.get(self._config.dense_vector_name)
                            else:
                                vector = record.vector
                            
                            matching_records.append(VectorSearchResult(
                                id=record.id,
                                score=relevance_score,
                                payload=record.payload,
                                vector=vector,
                                text=record.payload.get("content", "")
                            ))
            
            matching_records.sort(key=lambda x: x.score, reverse=True)
            matching_records = matching_records[:top_k]
            
            # Apply reranking if configured
            if self.reranker and kwargs.get('apply_reranking', True):
                matching_records = self._apply_reranking(matching_records, query_text)
            
            return matching_records
            
        except Exception as e:
            raise SearchError(f"An error occurred during client-side full-text search: {e}") from e
    
    async def _server_side_full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]],
        similarity_threshold: float,
        target_text_field: str,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Server-side full-text search implementation."""
        try:
            # Create text index if needed
            try:
                await self._client.create_payload_index(
                    collection_name=self._config.collection_name,
                    field_name=target_text_field,
                    field_schema=models.TextIndexParams(type="text"),
                    wait=True
                )
            except Exception:
                pass  # Index might already exist

            text_condition = models.FieldCondition(
                key=target_text_field, 
                match=models.MatchText(text=query_text)
            )

            if filter:
                metadata_filter = self._build_qdrant_filter(filter)
                metadata_filter.must.append(text_condition)
                final_filter = metadata_filter
            else:
                final_filter = models.Filter(must=[text_condition])
            
            records = await self._client.scroll(
                collection_name=self._config.collection_name,
                scroll_filter=final_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=True,
            )
            
            query_terms = query_text.lower().split()
            scored_results = []
            
            for r in records[0]:
                text_content = r.payload.get(target_text_field, "")
                if not text_content:
                    continue
                
                text_lower = text_content.lower()
                text_words = text_lower.split()
                
                term_count = sum(text_lower.count(term) for term in query_terms)
                doc_length = len(text_words)
                
                if doc_length > 0 and term_count > 0:
                    matched_terms = sum(1 for term in query_terms if term in text_lower)
                    match_ratio = matched_terms / len(query_terms)
                    
                    import math
                    term_density = term_count / doc_length
                    tf_score = math.log(1 + term_density * 5) / math.log(6)
                    
                    score = (tf_score * 0.7 + match_ratio * 0.3)
                    score = min(1.0, max(0.0, score))
                else:
                    score = 0.0
                
                if score >= similarity_threshold:
                    scored_results.append((score, r))
            
            scored_results.sort(key=lambda x: x[0], reverse=True)
            scored_results = scored_results[:top_k]
            
            filtered_results = []
            for score, r in scored_results:
                # Extract vector (handle named vectors)
                vector = None
                if isinstance(r.vector, dict):
                    vector = r.vector.get(self._config.dense_vector_name)
                else:
                    vector = r.vector
                
                filtered_results.append(VectorSearchResult(
                    id=r.id,
                    score=score,
                    payload=r.payload,
                    vector=vector,
                    text=r.payload.get("content", "")
                ))
            
            # Apply reranking if configured
            if self.reranker and kwargs.get('apply_reranking', True):
                filtered_results = self._apply_reranking(filtered_results, query_text)
            
            return filtered_results
        except Exception as e:
            raise SearchError(f"An error occurred during server-side full-text search: {e}") from e
    
    def full_text_search_sync(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a full-text search if the provider supports it (sync).
        """
        return self._run_async_from_sync(self.full_text_search(query_text, top_k, filter, similarity_threshold, **kwargs))
    
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
        Combines dense and full-text search results using fusion.
        
        If sparse vectors are enabled and provided in kwargs, uses native Qdrant hybrid search.
        Otherwise, performs manual fusion of dense + full-text search results.
        """
        effective_alpha = alpha if alpha is not None else self._config.default_hybrid_alpha or 0.5
        effective_fusion = fusion_method if fusion_method is not None else self._config.default_fusion_method or 'weighted'

        # Check if we should use native sparse vector hybrid search
        sparse_query_vector = kwargs.get('sparse_query_vector')
        
        if self._config.use_sparse_vectors and sparse_query_vector:
            return await self._native_hybrid_search(
                query_vector, sparse_query_vector, top_k, filter, effective_fusion, similarity_threshold, **kwargs
            )
        else:
            # Manual fusion of dense + full-text
            dense_results = await self.dense_search(query_vector, top_k, filter, similarity_threshold, **kwargs)
            ft_results = await self.full_text_search(query_text, top_k, filter, similarity_threshold, **kwargs)

            if effective_fusion == 'weighted':
                fused_results = self._fuse_weighted(dense_results, ft_results, effective_alpha)
            elif effective_fusion == 'rrf':
                fused_results = self._fuse_rrf(dense_results, ft_results)
            else:
                raise ConfigurationError(f"Unsupported fusion method: '{effective_fusion}'")

            fused_results.sort(key=lambda x: x.score, reverse=True)
            fused_results = fused_results[:top_k]
            
            # Apply reranking if configured
            if self.reranker and kwargs.get('apply_reranking', True):
                fused_results = self._apply_reranking(fused_results, query_text)
            
            return fused_results
    
    def hybrid_search_sync(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Combines dense and sparse/keyword search results (sync).
        """
        return self._run_async_from_sync(self.hybrid_search(query_vector, query_text, top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs))
    
    async def _native_hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        top_k: int,
        filter: Optional[Dict[str, Any]],
        fusion_method: str,
        similarity_threshold: Optional[float],
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Performs native Qdrant hybrid search using named dense and sparse vectors.
        """
        try:
            qdrant_filter = self._build_qdrant_filter(filter) if filter else None
            
            # Map fusion method
            fusion_map = {
                'rrf': models.Fusion.RRF,
                'dbsf': models.Fusion.DBSF
            }
            
            final_fusion = fusion_method if fusion_method is not None else self._config.default_fusion_method or 'rrf'

            query_response = await self._client.query_points(
                collection_name=self._config.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector.get('indices', []),
                            values=sparse_vector.get('values', [])
                        ),
                        using=self._config.sparse_vector_name,
                        limit=top_k,
                    ),
                    models.Prefetch(
                        query=dense_vector,
                        using=self._config.dense_vector_name,
                        limit=top_k,
                    ),
                ],
                query=models.FusionQuery(fusion=fusion_map.get(final_fusion, models.Fusion.RRF)),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=True
            )
            
            results = []
            for point in query_response.points:
                # Extract dense vector
                vector = None
                if isinstance(point.vector, dict):
                    vector = point.vector.get(self._config.dense_vector_name)
                else:
                    vector = point.vector
                
                results.append(VectorSearchResult(
                    id=point.id,
                    score=point.score,
                    payload=point.payload,
                    vector=vector,
                    text=point.payload.get("content", "") if point.payload else ""
                ))
            
            return results
            
        except Exception as e:
            raise SearchError(f"An error occurred during native hybrid search: {e}") from e
    
    # ============================================================================
    # Reranking
    # ============================================================================
    # TODO: HANDLE RERANKING!!!
    def _apply_reranking(
        self,
        results: List[VectorSearchResult],
        query: str
    ) -> List[VectorSearchResult]:
        """
        Applies reranking to search results if a reranker is configured.
        
        Args:
            results: Initial search results
            query: The original query text
            
        Returns:
            Reranked results if reranker is available, otherwise original results
        """
        if not self.reranker or not results:
            return results
        
        try:
            # Check if reranker has a rerank method
            if hasattr(self.reranker, 'rerank'):
                # Convert to format expected by reranker
                documents = []
                for result in results:
                    # Create document-like object
                    doc = {
                        'id': result.id,
                        'content': result.text or result.payload.get('content', ''),
                        'payload': result.payload,
                        'score': result.score
                    }
                    documents.append(doc)
                
                # Apply reranking
                reranked_docs = self.reranker.rerank(query=query, documents=documents)
                
                # Convert back to VectorSearchResult
                reranked_results = []
                for doc in reranked_docs:
                    if isinstance(doc, dict):
                        reranked_results.append(VectorSearchResult(
                            id=doc.get('id'),
                            score=doc.get('score', 0.0),
                            payload=doc.get('payload'),
                            vector=None,  # Vector not needed after reranking
                            text=doc.get('content', '')
                        ))
                    else:
                        # Handle object-based reranker response
                        original_result = next((r for r in results if r.id == getattr(doc, 'id', None)), None)
                        if original_result:
                            reranked_results.append(VectorSearchResult(
                                id=original_result.id,
                                score=getattr(doc, 'score', original_result.score),
                                payload=original_result.payload,
                                vector=original_result.vector,
                                text=original_result.text
                            ))
                
                debug_log(f"Reranked {len(results)} results to {len(reranked_results)}", context="QdrantVectorDB")
                return reranked_results
            
            return results
            
        except Exception as e:
            warning_log(f"Reranking failed: {e}. Returning original results.", context="QdrantVectorDB")
            return results
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    def _check_similarity_threshold(self, score: float, threshold: float) -> bool:
        """Check if score meets similarity threshold based on distance metric."""
        if self._config.distance_metric == DistanceMetric.COSINE:
            return score >= threshold
        elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            return score >= threshold
        elif self._config.distance_metric == DistanceMetric.EUCLIDEAN:
            max_distance = 1.0 / threshold if threshold > 0 else float('inf')
            return score <= max_distance
        return False
    
    def _fuse_weighted(
        self,
        list1: List[VectorSearchResult],
        list2: List[VectorSearchResult],
        alpha: float
    ) -> List[VectorSearchResult]:
        """Combines two result lists using weighted scores."""
        all_docs: Dict[Union[str, int], VectorSearchResult] = {res.id: res for res in list1}
        all_docs.update({res.id: res for res in list2})
        
        new_scores: Dict[Union[str, int], float] = defaultdict(float)

        for res in list1:
            new_scores[res.id] += res.score * alpha

        for res in list2:
            new_scores[res.id] += res.score * (1 - alpha)
        
        final_results = []
        for doc_id, fused_score in new_scores.items():
            original_doc = all_docs[doc_id]
            final_results.append(VectorSearchResult(
                id=original_doc.id,
                payload=original_doc.payload,
                vector=original_doc.vector,
                score=fused_score,
                text=original_doc.payload.get("content", "") if original_doc.payload else ""
            ))
        
        return final_results

    def _fuse_rrf(
        self,
        list1: List[VectorSearchResult],
        list2: List[VectorSearchResult],
        k: int = 60
    ) -> List[VectorSearchResult]:
        """Combines two result lists using Reciprocal Rank Fusion."""
        all_docs: Dict[Union[str, int], VectorSearchResult] = {}
        ranked_scores: Dict[Union[str, int], float] = defaultdict(float)

        for rank, res in enumerate(list1):
            if res.id not in all_docs:
                all_docs[res.id] = res
            ranked_scores[res.id] += 1.0 / (k + rank + 1)
        
        for rank, res in enumerate(list2):
            if res.id not in all_docs:
                all_docs[res.id] = res
            ranked_scores[res.id] += 1.0 / (k + rank + 1)
        
        final_results = []
        for doc_id, fused_score in ranked_scores.items():
            original_doc = all_docs[doc_id]
            final_results.append(VectorSearchResult(
                id=original_doc.id,
                payload=original_doc.payload,
                vector=original_doc.vector,
                score=fused_score,
                text=original_doc.payload.get("content", "") if original_doc.payload else ""
            ))
            
        return final_results

    def _build_qdrant_filter(self, filter_dict: Dict[str, Any]) -> models.Filter:
        """
        Translates MongoDB-style filter dict into Qdrant Filter.
        
        Supports:
        - Direct key-value: {'document_id': 'abc'} -> match
        - Range operators: {'metadata.age': {'$gte': 18}}
        - In operator: {'document_name': {'$in': ['a', 'b']}}
        """
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$gte":
                        conditions.append(models.FieldCondition(key=key, range=models.Range(gte=op_value)))
                    elif op == "$lte":
                        conditions.append(models.FieldCondition(key=key, range=models.Range(lte=op_value)))
                    elif op == "$gt":
                        conditions.append(models.FieldCondition(key=key, range=models.Range(gt=op_value)))
                    elif op == "$lt":
                        conditions.append(models.FieldCondition(key=key, range=models.Range(lt=op_value)))
                    elif op == "$in":
                        conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=op_value)))
                    elif op == "$eq":
                        conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=op_value)))
            else:
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        
        return models.Filter(must=conditions)
    
