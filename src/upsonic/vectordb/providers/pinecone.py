import time
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union, Literal, Generator

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec, PodSpec
    from pinecone.exceptions import PineconeApiException as ApiException, NotFoundException
    _PINECONE_AVAILABLE = True
except ImportError:
    pinecone = None
    Pinecone = None
    ServerlessSpec = None
    PodSpec = None
    ApiException = None
    NotFoundException = None
    _PINECONE_AVAILABLE = False

try:
    from pinecone_text.sparse import BM25Encoder
    _BM25_AVAILABLE = True
except ImportError:
    BM25Encoder = None
    _BM25_AVAILABLE = False


from upsonic.vectordb.base import BaseVectorDBProvider

from upsonic.vectordb.config import (
    PineconeConfig,
    DistanceMetric
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
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)


class PineconeProvider(BaseVectorDBProvider):
    """
    High-level, comprehensive, async-first vector database provider for Pinecone.
    
    This provider uses a SINGLE INDEX approach for hybrid search (following Pinecone best practices),
    where both dense and sparse vectors are stored in the same index.
    
    Key Features:
    - Single index for hybrid dense+sparse vectors
    - Comprehensive metadata handling (document_name, document_id, content_id, content, metadata)
    - Dynamic indexing configuration
    - Hybrid vector scaling (alpha-weighted combination)
    - Filter building and search optimization
    - Batch processing for efficient operations
    - Full async/await support
    - Support for ServerlessSpec, PodSpec, and dict specs
    - **Automatic score normalization to [0, 1] range** for all metrics
    
    Score Normalization:
    - **dotproduct**: Clips scores to [0, 1] (handles floating-point precision errors)
    - **cosine**: Clips scores to [0, 1] (already normalized by Pinecone)
    - **euclidean**: Converts distance to similarity using 1/(1+distance)
    
    Important Notes:
    - For **dotproduct metric**, vectors MUST be L2-normalized (unit vectors) for proper 0-1 scores
    - For **hybrid search**, dotproduct metric is REQUIRED and auto-configured
    - Sparse vectors are auto-generated using BM25Encoder if not provided
    """

    _DISTANCE_METRIC_MAP = {
        DistanceMetric.COSINE: "cosine",
        DistanceMetric.EUCLIDEAN: "euclidean",
        DistanceMetric.DOT_PRODUCT: "dotproduct",
    }


    def __init__(self, config: Union[PineconeConfig, Dict[str, Any]]):
        """
        Initializes the Pinecone provider with comprehensive configuration support.
        
        Args:
            config: Either a PineconeConfig object or a dictionary with configuration parameters
        """
        if not _PINECONE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pinecone",
                install_command='pip install "upsonic[rag]"',
                feature_name="Pinecone vector database provider"
            )

        # Convert dict to PineconeConfig if needed
        if isinstance(config, dict):
            config = PineconeConfig.from_dict(config)
        
        super().__init__(config)
        
        # Provider metadata
        self.provider_name = config.provider_name or f"PineconeProvider_{config.collection_name}"
        self.provider_description = config.provider_description
        self.provider_id = config.provider_id or self._generate_provider_id()
        

        self._index: Optional[object] = None
        self._client: Optional[object] = None
        self._is_connected = False
        
        # Initialize BM25 sparse encoder if hybrid search is enabled
        self._sparse_encoder: Optional[object] = None
        if self._config.hybrid_search_enabled or self._config.use_sparse_vectors:
            if _BM25_AVAILABLE:
                self._sparse_encoder = BM25Encoder().default()
                logger.info("BM25 sparse encoder initialized for hybrid search.")
            else:
                logger.warning("pinecone-text not available. Install with: pip install pinecone-text")
        
        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """
        Comprehensive validation of the Config object against Pinecone constraints.
        """
        logger.debug("Performing comprehensive Pinecone configuration validation...")

        # Pinecone requires API key
        if not self._config.api_key:
            raise ConfigurationError("Configuration Error: 'api_key' is mandatory for Pinecone.")
        
        # Validate spec or environment is provided
        if not self._config.spec and not self._config.environment:
            raise ConfigurationError("Either 'spec' or 'environment' must be provided.")
        
        # Validate hybrid search configuration
        if self._config.hybrid_search_enabled and self._config.metric != 'dotproduct':
            logger.warning("Hybrid search works best with dotproduct metric.")
        
        logger.info("Pinecone configuration validated successfully.")

    def _sanitize_collection_name(self, collection_name: str) -> str:
        """
        Convert collection name to Pinecone-compatible format.
        Pinecone requires names to consist of lowercase alphanumeric characters or hyphens only.
        """
        import re
        sanitized = re.sub(r'[^a-z0-9-]', '-', collection_name.lower())
        sanitized = re.sub(r'-+', '-', sanitized)
        sanitized = sanitized.strip('-')
        if not sanitized:
            sanitized = "default-collection"
        return sanitized

    def _generate_content_id(self) -> str:
        """Generate a unique content ID using UUID4."""
        return str(uuid.uuid4())
    
    def _generate_provider_id(self) -> str:
        """Generates a unique provider ID based on collection and environment."""
        identifier_parts = [
            self._config.collection_name,
            self._config.environment or self._config.host or "cloud",
        ]
        identifier = "#".join(filter(None, identifier_parts))
        
        import hashlib
        return hashlib.md5(identifier.encode()).hexdigest()[:16]

    def _build_metadata(
        self,
        payload: Dict[str, Any],
        chunk: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive metadata for Pinecone upsert.
        
        Metadata structure:
        - document_name (if exists)
        - document_id (if exists)
        - content_id (main ID, auto-generated if not exists)
        - content (text content)
        - metadata fields (flattened with metadata_ prefix)
        - user_metadata (highest priority)
        
        Args:
            payload: Base payload containing document information
            chunk: Text content/chunk
            user_metadata: Additional user-provided metadata
        
        Returns:
            Comprehensive metadata dictionary
        """
        metadata = {}
        
        # Add default metadata from config
        if self._config.default_metadata:
            metadata.update(self._config.default_metadata)
        
        # Extract standard fields from payload
        if 'document_name' in payload:
            metadata['document_name'] = payload['document_name']
        
        if 'document_id' in payload:
            metadata['document_id'] = payload['document_id']
        
        # Handle content_id (main ID)
        if 'content_id' in payload:
            metadata['content_id'] = payload['content_id']
        elif self._config.auto_generate_content_id:
            metadata['content_id'] = self._generate_content_id()
        
        # Add content (text)
        if chunk:
            metadata['content'] = chunk
        elif 'content' in payload:
            metadata['content'] = payload['content']
        
        # Handle nested metadata - flatten with prefix
        if 'metadata' in payload and isinstance(payload['metadata'], dict):
            for key, value in payload['metadata'].items():
                metadata[f'metadata_{key}'] = value
        
        # Add any remaining payload fields
        for key, value in payload.items():
            if key not in ['document_name', 'document_id', 'content_id', 'content', 'metadata']:
                metadata[key] = value
        
        # Apply user metadata (highest priority)
        if user_metadata:
            metadata.update(user_metadata)
        
        return metadata

    def _build_spec(self) -> Union[ServerlessSpec, PodSpec]:
        """
        Build Pinecone spec from configuration.
        
        Returns:
            ServerlessSpec or PodSpec instance
        """
        # If spec is already provided and is not a dict, return it
        if self._config.spec and not isinstance(self._config.spec, dict):
            return self._config.spec
        
        # If spec is a dict, it should be a ServerlessSpec or PodSpec dict
        if self._config.spec and isinstance(self._config.spec, dict):
            spec_dict = self._config.spec
            # Try to determine if it's serverless or pod based on keys
            if 'cloud' in spec_dict and 'region' in spec_dict:
                return ServerlessSpec(**spec_dict)
            elif 'environment' in spec_dict:
                return PodSpec(**spec_dict)
            else:
                # Default to serverless
                return ServerlessSpec(**spec_dict)
        
        # Build from environment (backward compatibility)
        if self._config.environment:
            if '-' in self._config.environment:
                parts = self._config.environment.split('-', 1)
                cloud = parts[0]
                region = parts[1]
            else:
                cloud = 'aws'
                region = 'us-east-1'
            return ServerlessSpec(cloud=cloud, region=region)
        
        # Build PodSpec if pod settings are provided
        if self._config.pod_type:
            pod_spec_params = {
                "environment": self._config.environment or "us-east-1-aws",
                "pod_type": self._config.pod_type
            }
            if self._config.pods:
                pod_spec_params["pods"] = self._config.pods
            if self._config.replicas:
                pod_spec_params["replicas"] = self._config.replicas
            if self._config.shards:
                pod_spec_params["shards"] = self._config.shards
            return PodSpec(**pod_spec_params)
        
        # Default to serverless
        return ServerlessSpec(cloud='aws', region='us-east-1')

    async def connect(self) -> None:
        """
        Establishes connection to Pinecone service asynchronously.
        """
        if self._is_connected:
            logger.debug("Already connected to Pinecone.")
            return
        
        logger.debug("Connecting to Pinecone...")
        
        def _connect():
            try:

                client_params = {
                    "api_key": self._config.api_key.get_secret_value()
                }
                
                if self._config.host:
                    client_params["host"] = self._config.host
                
                if self._config.additional_headers:
                    client_params["additional_headers"] = self._config.additional_headers
                
                if self._config.pool_threads:
                    client_params["pool_threads"] = self._config.pool_threads
                
                if self._config.index_api:
                    client_params["index_api"] = self._config.index_api
                
                self._client = Pinecone(**client_params)
                
                # Check for existing index
                existing_indexes = self._client.list_indexes().names()
                safe_collection_name = self._sanitize_collection_name(self._config.collection_name)
                
                if safe_collection_name in existing_indexes:
                    self._index = self._client.Index(safe_collection_name)
                    logger.debug(f"Connected to existing index: {safe_collection_name}")
                else:
                    logger.debug(f"Index {safe_collection_name} does not exist yet")
                
                self._is_connected = True
                logger.info("Successfully connected to Pinecone.")
            except ApiException as e:
                raise VectorDBConnectionError(f"Failed to connect to Pinecone: {e}") from e
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _connect)

    def connect_sync(self) -> None:
        """Establishes connection to Pinecone service (sync)."""
        return self._run_async_from_sync(self.connect())

    async def disconnect(self) -> None:
        """Performs logical disconnection from Pinecone service."""
        logger.debug("Disconnecting from Pinecone...")
        self._client = None
        self._index = None
        self._is_connected = False
        logger.info("Disconnected from Pinecone.")

    def disconnect_sync(self) -> None:
        """Performs logical disconnection from Pinecone service (sync)."""
        return self._run_async_from_sync(self.disconnect())

    async def is_ready(self) -> bool:
        """Health check - verifies connection to Pinecone service."""
        if not self._is_connected or not self._client:
            return False
        
        def _check():
            try:
                self._client.list_indexes()
                return True
            except Exception as e:
                logger.error(f"Readiness check failed: {e}")
                return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _check)

    def is_ready_sync(self) -> bool:
        """Health check - verifies connection to Pinecone service (sync)."""
        return self._run_async_from_sync(self.is_ready())

    async def collection_exists(self) -> bool:
        """Checks if the collection (index) exists in Pinecone."""
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to check collection existence.")
        
        def _exists():
            try:
                existing_indexes = self._client.list_indexes().names()
                safe_collection_name = self._sanitize_collection_name(self._config.collection_name)
                return safe_collection_name in existing_indexes
            except ApiException as e:
                raise VectorDBError(f"Failed to check collection existence: {e}") from e
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _exists)

    def collection_exists_sync(self) -> bool:
        """Checks if the collection (index) exists in Pinecone (sync)."""
        return self._run_async_from_sync(self.collection_exists())

    async def delete_collection(self) -> None:
        """Permanently deletes the collection from Pinecone."""
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to delete collection.")
        
        collection_name = self._config.collection_name
        safe_collection_name = self._sanitize_collection_name(collection_name)
        logger.warning(f"Deleting collection: '{safe_collection_name}'")
        
        def _delete():
            try:
                if safe_collection_name in self._client.list_indexes().names():
                    logger.info(f"Deleting index '{safe_collection_name}'...")
                    self._client.delete_index(safe_collection_name)
                    self._wait_for_deletion_sync(safe_collection_name)
                    logger.info(f"Successfully deleted index '{safe_collection_name}'.")
                else:
                    logger.info(f"Index '{safe_collection_name}' does not exist.")
            except ApiException as e:
                raise VectorDBError(f"Failed to delete collection: {e}") from e
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _delete)

    def delete_collection_sync(self) -> None:
        """Permanently deletes the collection from Pinecone (sync)."""
        return self._run_async_from_sync(self.delete_collection())

    async def create_collection(self) -> None:
        """Creates new collection in Pinecone according to configuration."""
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to create collection.")

        safe_collection_name = self._sanitize_collection_name(self._config.collection_name)
        
        def _create():
            # Check if exists
            if safe_collection_name in self._client.list_indexes().names():
                if self._config.recreate_if_exists:
                    logger.info(f"Index '{safe_collection_name}' exists. Recreating...")
                    self._client.delete_index(safe_collection_name)
                    self._wait_for_deletion_sync(safe_collection_name)
                else:
                    logger.info(f"Index '{safe_collection_name}' already exists.")
                    self._index = self._client.Index(safe_collection_name)
                    return

            logger.info(f"Creating index '{safe_collection_name}'...")
            try:
                spec = self._build_spec()
                
                # Build create params
                create_params = {
                    "name": safe_collection_name,
                    "dimension": self._config.vector_size,
                    "metric": self._config.metric,
                    "spec": spec
                }
                
                if self._config.timeout:
                    create_params["timeout"] = self._config.timeout
                
                # Add metadata_config for indexed fields (PodSpec supports this)
                if isinstance(spec, PodSpec) and self._config.indexed_fields:
                    create_params["metadata_config"] = {
                        "indexed": self._config.indexed_fields
                    }
                
                self._client.create_index(**create_params)
                self._wait_for_index_ready_sync(safe_collection_name)
                self._index = self._client.Index(safe_collection_name)
                logger.info(f"Successfully created index '{safe_collection_name}'.")
            except ApiException as e:
                raise VectorDBError(f"Failed to create index: {e}") from e
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _create)

    def create_collection_sync(self) -> None:
        """Creates new collection in Pinecone according to configuration (sync)."""
        return self._run_async_from_sync(self.create_collection())

    def _wait_for_index_ready_sync(self, index_name: str) -> None:
        """Wait for index to be ready (synchronous)."""
        wait_timeout = 600
        start_time = time.time()
        
        while True:
            try:
                status = self._client.describe_index(index_name)
                if status['status']['ready']:
                    break
            except ApiException:
                pass
            
            if time.time() - start_time > wait_timeout:
                raise VectorDBError(f"Timeout waiting for index '{index_name}' to be ready.")
            logger.debug(f"Waiting for index '{index_name}' to be ready...")
            time.sleep(10)

    def _wait_for_deletion_sync(self, index_name: str) -> None:
        """Wait for index deletion (synchronous)."""
        wait_timeout = 300
        start_time = time.time()
        
        while index_name in self._client.list_indexes().names():
            if time.time() - start_time > wait_timeout:
                raise VectorDBError(f"Timeout waiting for index '{index_name}' to be deleted.")
            logger.debug(f"Waiting for index '{index_name}' to be deleted...")
            time.sleep(5)

    def _generate_batches(
        self,
        ids: List[Union[str, int]],
        payloads: List[Dict[str, Any]],
        vectors: Optional[List[List[float]]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        chunks: Optional[List[str]] = None,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Generator:
        """
        Generate batches for efficient upserting.
        
        Sparse_values are included in the same vector dict.
        If sparse_vectors are not provided but use_hybrid_search is enabled,
        automatically generate them using BM25Encoder.
        """
        batch_size = self._config.batch_size
        batch = []
        
        for i in range(len(ids)):
            # Build comprehensive metadata
            metadata = self._build_metadata(
                payloads[i],
                chunks[i] if chunks else None,
                user_metadata
            )
            
            record_dict = {
                "id": str(ids[i]),
                "metadata": metadata
            }
            
            # Add dense vectors
            if vectors:
                import numpy as np
                record_dict["values"] = np.array(vectors[i], dtype=np.float32).tolist()
            
            # Add sparse vectors in same dict
            if self._config.hybrid_search_enabled:
                if sparse_vectors:
                    # Use provided sparse vectors
                    record_dict["sparse_values"] = sparse_vectors[i]
                elif self._sparse_encoder and chunks and chunks[i]:
                    # Auto-generate sparse vectors from text content using BM25
                    try:
                        record_dict["sparse_values"] = self._sparse_encoder.encode_documents(chunks[i])
                    except Exception as e:
                        logger.warning(f"Failed to generate sparse vectors for record {ids[i]}: {e}")
            
            batch.append(record_dict)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch

    def _hybrid_scale(
        self,
        dense: List[float],
        sparse: Dict[str, Any],
        alpha: float
    ) -> tuple:
        """
        Hybrid vector scaling using convex combination.
        
        alpha * dense + (1 - alpha) * sparse
        
        Args:
            dense: Dense vector values
            sparse: Sparse vector dict with 'indices' and 'values'
            alpha: Weight (1.0 = dense only, 0.0 = sparse only)
        
        Returns:
            Tuple of (scaled_dense, scaled_sparse)
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        # Scale sparse and dense vectors
        hsparse = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]]
        }
        hdense = [v * alpha for v in dense]
        
        return hdense, hsparse

    async def upsert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Adds or updates data in the collection with comprehensive metadata support.
        
        Upserts to a single index with both dense and sparse vectors.
        
        Args:
            vectors: Dense vector embeddings
            payloads: Metadata payloads (document_name, document_id, content_id, content, metadata)
            ids: Unique identifiers
            chunks: Text content/chunks
            sparse_vectors: Sparse vector representations for hybrid search
            user_metadata: Additional user-provided metadata
            **kwargs: Additional Pinecone-specific options (namespace, batch_size, show_progress)
        """
        if not self._is_connected or not self._index:
            raise VectorDBConnectionError("Not connected to Pinecone.")

        logger.debug("Initiating upsert with comprehensive metadata...")
        
        # Validate
        if not vectors:
            raise UpsertError("Vectors must be provided.")
        
        if len(ids) != len(payloads) != len(vectors):
            raise UpsertError("ids, payloads, and vectors must have the same length.")
        
        if chunks and len(chunks) != len(ids):
            raise UpsertError("chunks must have the same length as ids.")
        
        if sparse_vectors and len(sparse_vectors) != len(ids):
            raise UpsertError("sparse_vectors must have the same length as ids.")
        
        logger.info(f"Upserting {len(ids)} records...")
        
        # Extract kwargs
        namespace = kwargs.get("namespace", self._config.namespace or "")
        batch_size = kwargs.get("batch_size", self._config.batch_size)
        show_progress = kwargs.get("show_progress", self._config.show_progress)

        def _upsert():
            try:
                # Generate batches
                batches = self._generate_batches(
                    ids, payloads, vectors, sparse_vectors, chunks, user_metadata
                )
                
                for batch in batches:
                    # Upsert to single index
                    self._index.upsert(
                        vectors=batch,
                        namespace=namespace,
                        batch_size=batch_size,
                        show_progress=show_progress
                    )
                
                logger.info(f"Successfully upserted {len(ids)} records.")
            except ApiException as e:
                raise UpsertError(f"Upsert failed: {e}") from e
            except Exception as e:
                raise UpsertError(f"General upsert error: {e}") from e
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _upsert)

    def upsert_sync(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Adds or updates data in the collection with comprehensive metadata support (sync)."""
        return self._run_async_from_sync(self.upsert(vectors, payloads, ids, chunks, sparse_vectors, user_metadata, **kwargs))

    async def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Removes data by IDs from the index."""
        if not self._is_connected or not self._index:
            raise VectorDBConnectionError("Not connected to Pinecone.")

        str_ids = [str(i) for i in ids]
        namespace = kwargs.get("namespace", self._config.namespace or "")
        
        def _delete():
            try:
                self._index.delete(ids=str_ids, namespace=namespace)
                logger.info(f"Deleted {len(str_ids)} records.")
            except ApiException as e:
                raise VectorDBError(f"Delete failed: {e}") from e
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _delete)

    def delete_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Removes data by IDs from the index (sync)."""
        return self._run_async_from_sync(self.delete(ids, **kwargs))
    
    def delete_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Alias for delete_sync() - provided for backward compatibility."""
        return self.delete_sync(ids, **kwargs)

    async def delete_by_filter(self, filter: Dict[str, Any], **kwargs) -> bool:
        """Delete records matching a metadata filter."""
        if not self._is_connected or not self._index:
            raise VectorDBConnectionError("Not connected to Pinecone.")
        
        namespace = kwargs.get("namespace", self._config.namespace or "")
        
        def _delete():
            try:
                self._index.delete(filter=filter, namespace=namespace)
                logger.info(f"Deleted records matching filter.")
                return True
            except ApiException as e:
                logger.warning(f"Filter delete failed: {e}")
                return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _delete)

    def delete_by_content_id(self, content_id: str) -> bool:
        """Delete records by content_id metadata field (sync)."""
        return self._run_async_from_sync(self.async_delete_by_content_id(content_id))
    
    async def async_delete_by_content_id(self, content_id: str) -> bool:
        """Delete records by content_id metadata field (async)."""
        return await self.delete_by_filter({"content_id": {"$eq": content_id}})

    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete records by document_id metadata field (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_id(document_id))
    
    async def async_delete_by_document_id(self, document_id: str) -> bool:
        """Delete records by document_id metadata field (async)."""
        return await self.delete_by_filter({"document_id": {"$eq": document_id}})

    def delete_by_document_name(self, document_name: str) -> bool:
        """Delete records by document_name metadata field (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_name(document_name))
    
    async def async_delete_by_document_name(self, document_name: str) -> bool:
        """Delete records by document_name metadata field (async)."""
        return await self.delete_by_filter({"document_name": {"$eq": document_name}})
    
    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete records by matching all metadata fields (sync)."""
        return self._run_async_from_sync(self.async_delete_by_metadata(metadata))
    
    async def async_delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete records by matching all metadata fields (async)."""
        # Build filter with all metadata fields
        filter_dict = {key: {"$eq": value} for key, value in metadata.items()}
        return await self.delete_by_filter(filter_dict)

    async def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Fetch records by IDs."""
        if not self._is_connected or not self._index:
            raise VectorDBConnectionError("Not connected to Pinecone.")
        
        namespace = kwargs.get("namespace", self._config.namespace or "")
        
        def _fetch():
            try:
                response = self._index.fetch(ids=[str(i) for i in ids], namespace=namespace)
                return self._parse_fetch_response(response)
            except ApiException as e:
                logger.warning(f"Fetch failed: {e}")
                return []
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _fetch)

    def fetch_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Fetch records by IDs (sync)."""
        return self._run_async_from_sync(self.fetch(ids, **kwargs))
    
    def fetch_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Alias for fetch_sync() - provided for backward compatibility."""
        return self.fetch_sync(ids, **kwargs)

    def _parse_fetch_response(self, response) -> List[VectorSearchResult]:
        """Parse fetch response into VectorSearchResult objects."""
        results = []
        fetched_vectors = response.vectors if hasattr(response, 'vectors') else response.get('vectors', {})
        
        for record_id, vector_data in fetched_vectors.items():
            metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else vector_data.get('metadata', {})
            values = vector_data.values if hasattr(vector_data, 'values') else vector_data.get('values')
            
            results.append(
                VectorSearchResult(
                    id=record_id,
                    score=1.0,
                    payload=metadata,
                    vector=values,
                    text=metadata.get('content') if metadata else None
                )
            )
        return results

    def _parse_query_response(self, response) -> List[VectorSearchResult]:
        """Parse query response into VectorSearchResult objects with normalized scores."""
        matches = response.matches if hasattr(response, 'matches') else response.get('matches', [])
        
        results = []
        for match in matches:
            if hasattr(match, 'id'):
                match_id = match.id
                raw_score = match.score
                metadata = match.metadata if hasattr(match, 'metadata') else {}
                values = match.values if hasattr(match, 'values') else None
            else:
                match_id = match['id']
                raw_score = match['score']
                metadata = match.get('metadata', {})
                values = match.get('values')
            
            # Normalize score to [0, 1] range
            # For dotproduct with L2-normalized vectors: score should be in [0,1] but can have floating-point errors
            # For cosine: score is already in [0,1] range
            # For euclidean: score can be any positive value
            normalized_score = self._normalize_score(raw_score)
            
            results.append(
                VectorSearchResult(
                    id=match_id,
                    score=normalized_score,
                    payload=metadata,
                    vector=values,
                    text=metadata.get('content') if metadata else None
                )
            )
        return results
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize score to [0, 1] range based on the distance metric.
        
        Args:
            score: Raw score from Pinecone API
            
        Returns:
            Normalized score in [0, 1] range
        """
        metric = self._config.metric
        
        if metric == 'dotproduct':
            # Dotproduct with L2-normalized vectors should be in [0,1]
            # but can have floating-point precision errors (e.g., 1.0001)
            # Clip to ensure strict [0, 1] range
            return min(1.0, max(0.0, score))
        
        elif metric == 'cosine':
            # Cosine similarity is already in [0, 1] range
            # but clip for safety
            return min(1.0, max(0.0, score))
        
        elif metric == 'euclidean':
            # Euclidean distance: smaller is better, can be any positive value
            # Convert to similarity: 1 / (1 + distance)
            # This gives values in (0, 1] range
            if score < 0:
                score = 0  # Should not happen, but handle edge case
            return 1.0 / (1.0 + score)
        
        else:
            # Default: clip to [0, 1]
            return min(1.0, max(0.0, score))

    async def dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Performs dense vector similarity search."""
        if not self._index:
            raise VectorDBConnectionError("Index not available.")
        
        final_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        namespace = kwargs.get("namespace", self._config.namespace or "")
        include_values = kwargs.get("include_values", True)
        
        def _search():
            try:
                response = self._index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=filter,
                    namespace=namespace,
                    include_metadata=True,
                    include_values=include_values
                )
                results = self._parse_query_response(response)
                filtered = [r for r in results if r.score >= final_threshold]
                logger.debug(f"Dense search: {len(results)} results, {len(filtered)} after threshold.")
                return filtered
            except ApiException as e:
                raise SearchError(f"Dense search failed: {e}") from e
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

    def dense_search_sync(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Performs dense vector similarity search (sync)."""
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
        Performs sparse-vector-based lexical search.
        
        Automatically generates sparse vectors using BM25Encoder if not provided.
        """
        if not self._index:
            raise VectorDBConnectionError("Index not available.")
        
        if not self._config.use_sparse_vectors:
            raise ConfigurationError("Full-text search requires use_sparse_vectors to be enabled.")
        
        final_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        namespace = kwargs.get("namespace", self._config.namespace or "")
        
        def _search():
            try:
                # Generate sparse vector if not provided
                sparse_vector = kwargs.get("sparse_vector")
                if not sparse_vector:
                    if self._sparse_encoder:
                        # Use BM25Encoder to generate sparse vector from query text
                        sparse_vector = self._sparse_encoder.encode_queries(query_text)
                    else:
                        # Fallback to Pinecone inference API
                        result = self._client.inference.embed(
                            model=self._config.sparse_encoder_model,
                            inputs=query_text,
                            parameters={"input_type": "query", "truncate": "END"}
                        )
                        sparse_vector = {
                            'indices': result[0]['sparse_indices'],
                            'values': result[0]['sparse_values']
                        }
                
                # CRITICAL: Pinecone requires a dense vector even for sparse-only queries
                # when the index was created with dense vectors. 
                # Use a normalized random vector (not all zeros) to avoid interference
                # The sparse vector will dominate the search results
                import numpy as np
                np.random.seed(42)  # Deterministic for testing
                dummy_vector = np.random.rand(self._config.vector_size).astype(np.float32)
                dummy_vector = (dummy_vector / np.linalg.norm(dummy_vector)).tolist()
                
                response = self._index.query(
                    vector=dummy_vector,  # Normalized random dense vector (required by Pinecone API, but sparse dominates)
                    sparse_vector=sparse_vector,  # Actual sparse vector for search (this dominates)
                    top_k=top_k,
                    filter=filter,
                    namespace=namespace,
                    include_metadata=True,
                    include_values=True
                )
                results = self._parse_query_response(response)
                filtered = [r for r in results if r.score >= final_threshold]
                # Sort by score descending (highest relevance first)
                filtered.sort(key=lambda x: x.score, reverse=True)
                return filtered
            except ApiException as e:
                raise SearchError(f"Full-text search failed: {e}") from e
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

    def full_text_search_sync(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Performs sparse-vector-based lexical search (sync)."""
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
        Performs hybrid search combining dense and sparse vectors in SINGLE INDEX.
        
        Scales vectors with alpha and queries single index.
        
        Args:
            query_vector: Dense vector for semantic search
            query_text: Text for sparse/lexical search
            top_k: Number of results
            filter: Metadata filter
            alpha: Weight (1.0 = dense only, 0.0 = sparse only)
            fusion_method: 'rrf' or 'weighted' (for backward compatibility, uses alpha scaling)
            similarity_threshold: Minimum score threshold
        """
        if not self._index:
            raise VectorDBConnectionError("Index not available.")
        
        if not self._config.hybrid_search_enabled:
            raise ConfigurationError("Hybrid search requires hybrid_search_enabled to be enabled.")
        
        alpha = alpha if alpha is not None else (self._config.default_hybrid_alpha or 0.5)
        final_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        namespace = kwargs.get("namespace", self._config.namespace or "")
        include_values = kwargs.get("include_values", True)
        
        def _search():
            try:
                # Generate sparse vector if not provided
                sparse_vector = kwargs.get("sparse_vector")
                if not sparse_vector:
                    if self._sparse_encoder:
                        # Use BM25Encoder to generate sparse vector from query text
                        sparse_vector = self._sparse_encoder.encode_queries(query_text)
                    else:
                        # Fallback to Pinecone inference API
                        result = self._client.inference.embed(
                            model=self._config.sparse_encoder_model,
                            inputs=query_text,
                            parameters={"input_type": "query", "truncate": "END"}
                        )
                        sparse_vector = {
                            'indices': result[0]['sparse_indices'],
                            'values': result[0]['sparse_values']
                        }
                
                # Scale vectors
                hdense, hsparse = self._hybrid_scale(query_vector, sparse_vector, alpha)
                
                # Query single index with both scaled vectors
                response = self._index.query(
                    vector=hdense,
                    sparse_vector=hsparse,
                    top_k=top_k,
                    filter=filter,
                    namespace=namespace,
                    include_metadata=True,
                    include_values=include_values
                )
                
                results = self._parse_query_response(response)
                
                # Apply reranker if configured
                if self._config.reranker:
                    # Note: reranker integration would require additional setup
                    logger.debug("Reranker configured but not yet integrated")
                
                filtered = [r for r in results if r.score >= final_threshold]
                return filtered[:top_k]
            except ApiException as e:
                raise SearchError(f"Hybrid search failed: {e}") from e
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

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
        """Performs hybrid search combining dense and sparse vectors in SINGLE INDEX (sync)."""
        return self._run_async_from_sync(self.hybrid_search(query_vector, query_text, top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs))

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
        Master search method that dispatches to appropriate search type.
        """
        is_dense = query_vector is not None
        is_sparse = query_text is not None
        
        final_top_k = top_k if top_k is not None else self._config.default_top_k
        if final_top_k is None:
            raise ConfigurationError("'top_k' must be provided.")

        if is_dense and is_sparse and self._config.hybrid_search_enabled:
            logger.debug("Dispatching to HYBRID search.")
            return await self.hybrid_search(query_vector, query_text, final_top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs)
        
        elif is_dense:
            logger.debug("Dispatching to DENSE search.")
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled.")
            return await self.dense_search(query_vector, final_top_k, filter, similarity_threshold, **kwargs)
        
        elif is_sparse:
            logger.debug("Dispatching to FULL-TEXT search.")
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled.")
            return await self.full_text_search(query_text, final_top_k, filter, similarity_threshold, **kwargs)
        
        else:
            raise SearchError("Search requires 'query_vector' or 'query_text'.")

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
        """Master search method that dispatches to appropriate search type (sync)."""
        return self._run_async_from_sync(self.search(top_k, query_vector, query_text, filter, alpha, fusion_method, similarity_threshold, **kwargs))

    async def id_exists(self, id: str) -> bool:
        """Check if a record with the given ID exists."""
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to Pinecone.")
        
        if not self._index:
            logger.warning("Index not set, attempting to connect to it...")
            safe_collection_name = self._sanitize_collection_name(self._config.collection_name)
            if safe_collection_name in self._client.list_indexes().names():
                self._index = self._client.Index(safe_collection_name)
            else:
                raise VectorDBConnectionError("Index does not exist.")
        
        namespace = self._config.namespace or ""
        
        def _check():
            try:
                response = self._index.fetch(ids=[id], namespace=namespace)
                if hasattr(response, 'vectors'):
                    return len(response.vectors) > 0
                return len(response.get('vectors', {})) > 0
            except Exception as e:
                logger.warning(f"Error checking ID existence: {e}")
                return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _check)

    def content_id_exists(self, content_id: str) -> bool:
        """Check if a record with the given content_id exists (sync)."""
        return self._run_async_from_sync(self.async_content_id_exists(content_id))
    
    async def async_content_id_exists(self, content_id: str) -> bool:
        """
        Check if a record with the given content_id exists (async).
        
        Uses fetch method from Pinecone API
        Assumes content_id is used as the record ID.
        
        Args:
            content_id: The content_id to check (used as ID)
            
        Returns:
            bool: True if the record exists, False otherwise
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to Pinecone.")
        
        if not self._index:
            logger.warning("Index not set, attempting to connect to it...")
            safe_collection_name = self._sanitize_collection_name(self._config.collection_name)
            if safe_collection_name in self._client.list_indexes().names():
                self._index = self._client.Index(safe_collection_name)
            else:
                raise VectorDBConnectionError("Index does not exist.")
        
        namespace = self._config.namespace or ""
        
        def _check():
            try:
                response = self._index.fetch(ids=[content_id], namespace=namespace)
                return len(response.vectors) > 0
            except Exception as e:
                logger.warning(f"Error checking if content_id {content_id} exists: {e}")
                return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _check)
    
    def document_id_exists(self, document_id: str) -> bool:
        """Check if any records with the given document_id metadata exist (sync)."""
        return self._run_async_from_sync(self.async_document_id_exists(document_id))
    
    async def async_document_id_exists(self, document_id: str) -> bool:
        """Check if any records with the given document_id metadata exist (async)."""
        # Pinecone doesn't have a direct way to check metadata existence without querying
        # So we'll use a minimal query to check if records with this metadata exist
        if not self._is_connected or not self._index:
            return False
        
        try:
            # Query with filter for document_id
            def _check():
                try:
                    result = self._index.query(
                        vector=[0.0] * self._config.vector_size,
                        filter={"document_id": {"$eq": document_id}},
                        top_k=1,
                        include_metadata=False,
                        namespace=self._config.namespace or ""
                    )
                    return len(result.matches) > 0
                except Exception:
                    return False
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _check)
        except Exception:
            return False
    
    def document_name_exists(self, document_name: str) -> bool:
        """Check if any records with the given document_name metadata exist (sync)."""
        return self._run_async_from_sync(self.async_document_name_exists(document_name))
    
    async def async_document_name_exists(self, document_name: str) -> bool:
        """Check if any records with the given document_name metadata exist (async)."""
        if not self._is_connected or not self._index:
            return False
        
        try:
            # Query with filter for document_name
            def _check():
                try:
                    result = self._index.query(
                        vector=[0.0] * self._config.vector_size,
                        filter={"document_name": {"$eq": document_name}},
                        top_k=1,
                        include_metadata=False,
                        namespace=self._config.namespace or ""
                    )
                    return len(result.matches) > 0
                except Exception:
                    return False
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _check)
        except Exception:
            return False

    async def get_count(self) -> int:
        """Get total count of records in the index."""
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to Pinecone.")
        
        if not self._index:
            logger.warning("Index not set, attempting to connect to it...")
            safe_collection_name = self._sanitize_collection_name(self._config.collection_name)
            if safe_collection_name in self._client.list_indexes().names():
                self._index = self._client.Index(safe_collection_name)
            else:
                raise VectorDBConnectionError("Index does not exist.")
        
        def _count():
            try:
                stats = self._index.describe_index_stats()
                return stats.total_vector_count if hasattr(stats, 'total_vector_count') else stats.get('total_vector_count', 0)
            except Exception as e:
                logger.warning(f"Error getting count: {e}")
                return 0
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _count)

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for records with the given content_id (sync)."""
        return self._run_async_from_sync(self.async_update_metadata(content_id, metadata))
    
    async def async_update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for records with the given content_id (async).
        
        Args:
            content_id: The content_id to update
            metadata: New metadata to merge
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to Pinecone.")
        
        if not self._index:
            logger.warning("Index not set, attempting to connect to it...")
            safe_collection_name = self._sanitize_collection_name(self._config.collection_name)
            if safe_collection_name in self._client.list_indexes().names():
                self._index = self._client.Index(safe_collection_name)
            else:
                raise VectorDBConnectionError("Index does not exist.")
        
        namespace = self._config.namespace or ""
        
        def _update():
            try:
                # Pinecone requires a vector parameter even when filtering
                # Use a dummy vector to query with filter
                if not self._config.vector_size:
                    logger.warning("Cannot update metadata without vector_size configured")
                    return
                
                dummy_vector = [0.0] * self._config.vector_size
                
                # Query for vectors with the given content_id - must include vector values!
                query_response = self._index.query(
                    vector=dummy_vector,  # REQUIRED: Must provide vector for query
                    filter={"content_id": {"$eq": content_id}},
                    top_k=10000,  # Get all matching vectors
                    include_metadata=True,
                    include_values=True,  # CRITICAL: Must include values for update
                    namespace=namespace,
                )
                
                matches = query_response.matches if hasattr(query_response, 'matches') else query_response.get('matches', [])
                if not matches:
                    logger.debug(f"No documents found with content_id: {content_id}")
                    return
                
                # Update each matching vector individually
                update_data = []  # Track successfully updated IDs
                for match in matches:
                    vector_id = match.id if hasattr(match, 'id') else match['id']
                    current_metadata = match.metadata if hasattr(match, 'metadata') else match.get('metadata', {})
                    if current_metadata is None:
                        current_metadata = {}
                    
                    # Merge existing metadata with new metadata (directly, no nesting)
                    updated_metadata = current_metadata.copy()
                    updated_metadata.update(metadata)
                    
                    # Update each vector individually (Pinecone API requirement)
                    try:
                        self._index.update(
                            id=vector_id,
                            set_metadata=updated_metadata,
                            namespace=namespace
                        )
                        update_data.append(vector_id)
                    except Exception as e:
                        logger.warning(f"Failed to update {vector_id}: {e}")
                
                logger.info(f"Updated metadata for {len(update_data)} documents with content_id: {content_id}")
                return True
            except Exception as e:
                logger.error(f"Error updating metadata for content_id '{content_id}': {e}")
                return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _update)
    
    def optimize(self) -> bool:
        """Optimize the vector database (sync). Pinecone doesn't require explicit optimization."""
        return True
    
    async def async_optimize(self) -> bool:
        """Optimize the vector database (async). Pinecone doesn't require explicit optimization."""
        return True
    
    def get_supported_search_types(self) -> List[str]:
        """Get the supported search types for Pinecone (sync)."""
        supported = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported
    
    async def async_get_supported_search_types(self) -> List[str]:
        """Get the supported search types for Pinecone (async)."""
        return self.get_supported_search_types()
