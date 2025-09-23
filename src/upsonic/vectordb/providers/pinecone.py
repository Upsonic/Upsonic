import logging
import time
from typing import Any, Dict, List, Optional, Union, Literal, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    from pinecone.exceptions import PineconeApiException as ApiException, NotFoundException
except ImportError:
    pinecone = None
    ApiException = None
    NotFoundException = None

from upsonic.vectordb.base import BaseVectorDBProvider

from upsonic.vectordb.config import (
    Config,
    Mode,
    ProviderName,
    IndexType,
    IVFTuningConfig,
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

logger = logging.getLogger(__name__)


class PineconeProvider(BaseVectorDBProvider):
    """
    Vector database provider for the Pinecone service.

    This provider translates the abstract configuration of the framework into
    the concrete operational parameters of Pinecone's cloud-native platform.
    It handles connection, rigorous configuration validation, and the lifecycle
    management for Pinecone indexes, ensuring that the framework's capabilities
    are correctly and efficiently mapped to Pinecone's features.
    """

    _DISTANCE_METRIC_MAP = {
        DistanceMetric.COSINE: "cosine",
        DistanceMetric.EUCLIDEAN: "euclidean",
        DistanceMetric.DOT_PRODUCT: "dotproduct",
    }


    def __init__(self, config: Config):
        """
        Initializes the Pinecone provider and performs rigorous, Pinecone-specific
        configuration validation as the first step.

        This constructor internalizes the framework's configuration and immediately
        triggers a comprehensive validation routine. This ensures that the
        provider instance is created in a valid, consistent, and ready-to-connect state,
        preventing runtime errors due to misconfiguration.
        """
        if pinecone is None:
            raise ImportError(
                "The 'pinecone' package is required to use PineconeProvider. "
                "Please install it via pip: 'pip install pinecone'"
            )

        super().__init__(config)
        
        self._validate_configuration()
        
        self._config = config
        self._dense_index: Optional[object] = None
        self._sparse_index: Optional[object] = None
        self._client: Optional[object] = None
        self._is_connected = False
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """
        A private helper method that executes a comprehensive validation of the
        Config object against all known Pinecone constraints and logical
        framework consistencies.

        This validation is multi-layered:
        1.  **Provider-Specific Constraints:** Checks parameters that must be set
            in a specific way for Pinecone (e.g., mode must be 'cloud').
        2.  **Cross-Configuration Consistency:** Ensures that the user's declared
            intentions in the `SearchConfig` are supported by the physical index
            schema defined in the `IndexingConfig`. This is the most critical
            part of the validation logic.
        """
        logger.debug("Executing Phase 1: Performing deep Pinecone configuration validation...")

        
        if self._config.core.provider_name != ProviderName.PINECONE:
            raise ConfigurationError(f"Fatal: Attempted to initialize PineconeProvider with a configuration for '{self._config.core.provider_name.value}'.")
        
        if not self._config.core.api_key:
            raise ConfigurationError("Configuration Error: 'api_key' is mandatory for connecting to the Pinecone service.")
        
        if isinstance(self._config.indexing.index_config, IVFTuningConfig):
            raise ConfigurationError("Configuration Error: Pinecone does not support IVF-based indexes. Please use 'HNSW' or 'FLAT' index types.")
        
        if self._config.indexing.quantization is not None:
            raise ConfigurationError("Configuration Error: Pinecone manages quantization internally and does not support user-configurable quantization.")

        logger.debug("Layer 1 validation passed: Basic Pinecone constraints are met.")

        
        search_cfg = self._config.search
        indexing_cfg = self._config.indexing

        if search_cfg.hybrid_search_enabled:
            is_hybrid_ready = indexing_cfg.create_dense_index and indexing_cfg.create_sparse_index
            if not is_hybrid_ready:
                raise ConfigurationError(
                    "Configuration Conflict: Hybrid search is enabled in SearchConfig, but the IndexingConfig is not configured to create BOTH a dense and a sparse index. "
                    "To enable hybrid search, you must set 'create_dense_index: True' and 'create_sparse_index: True'."
                )

        if search_cfg.dense_search_enabled and not indexing_cfg.create_dense_index:
            raise ConfigurationError(
                "Configuration Conflict: Dense search is enabled in SearchConfig, but the IndexingConfig is not configured to create a dense index ('create_dense_index: False'). "
                "You must enable the dense index to perform dense searches."
            )

        if search_cfg.full_text_search_enabled and not indexing_cfg.create_sparse_index:
            raise ConfigurationError(
                "Configuration Conflict: Full-text search is enabled in SearchConfig, but the IndexingConfig is not configured to create a sparse index ('create_sparse_index: False'). "
                "You must enable the sparse index to perform full-text searches."
            )

        logger.debug("Layer 2 validation passed: Search and Indexing configurations are consistent.")
        
        logger.info("Phase 1 Succeeded: Pinecone configuration has been successfully and comprehensively validated.")

    def _sanitize_collection_name(self, collection_name: str) -> str:
        """
        Convert collection name to Pinecone-compatible format.
        Pinecone requires names to consist of lowercase alphanumeric characters or hyphens only.
        """
        import re
        # Convert to lowercase and replace underscores and other invalid characters with hyphens
        sanitized = re.sub(r'[^a-z0-9-]', '-', collection_name.lower())
        # Remove multiple consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        # Remove leading/trailing hyphens
        sanitized = sanitized.strip('-')
        # Ensure it's not empty
        if not sanitized:
            sanitized = "default-collection"
        return sanitized



    def connect(self) -> None:
        if self._is_connected:
            logger.debug("Connection to Pinecone already established.")
            return
        logger.debug("Attempting to connect to Pinecone...")
        try:
            self._client = Pinecone(
                api_key=self._config.core.api_key.get_secret_value()
            )
            
            existing_indexes = self._client.list_indexes().names()
            
            if self._config.indexing.create_dense_index:
                safe_collection_name = self._sanitize_collection_name(self._config.core.collection_name)
                dense_name = f"{safe_collection_name}-dense"
                if dense_name in existing_indexes:
                    self._dense_index = self._client.Index(dense_name)
                    logger.debug(f"Connected to existing dense index: {dense_name}")
                else:
                    logger.debug(f"Dense index {dense_name} does not exist yet")
            
            if self._config.indexing.create_sparse_index:
                safe_collection_name = self._sanitize_collection_name(self._config.core.collection_name)
                sparse_name = f"{safe_collection_name}-sparse"
                if sparse_name in existing_indexes:
                    self._sparse_index = self._client.Index(sparse_name)
                    logger.debug(f"Connected to existing sparse index: {sparse_name}")
                else:
                    logger.debug(f"Sparse index {sparse_name} does not exist yet")
                
            self._is_connected = True
            logger.info("Successfully connected to Pinecone.")
        except ApiException as e:
            raise Exception(f"Failed to connect to Pinecone: {e}") from e

    def disconnect(self) -> None:
        """
        Performs a logical disconnection from the Pinecone service.
        The client library manages connection pools, so this simply clears local state.
        """
        logger.debug("Disconnecting from Pinecone...")
        self._client = None
        self._dense_index = None
        self._sparse_index = None
        self._is_connected = False
        logger.info("Logically disconnected from Pinecone. Cleared local client state.")

    def is_ready(self) -> bool:
        """
        Performs a health check on the configured Pinecone indexes.
        Returns True if all required indexes exist and are in a ready state.
        """
        if not self._is_connected or not self._client:
            return False
        
        try:
            if self._config.indexing.create_dense_index:
                safe_collection_name = self._sanitize_collection_name(self._config.core.collection_name)
                dense_name = f"{safe_collection_name}-dense"
                try:
                    status = self._client.describe_index(dense_name)
                    if not status['status']['ready']:
                        logger.debug(f"Dense index '{dense_name}' is not ready.")
                        return False
                except ApiException:
                    logger.debug(f"Dense index '{dense_name}' does not exist or is not accessible.")
                    return False
                    
            if self._config.indexing.create_sparse_index:
                safe_collection_name = self._sanitize_collection_name(self._config.core.collection_name)
                sparse_name = f"{safe_collection_name}-sparse"
                try:
                    status = self._client.describe_index(sparse_name)
                    if not status['status']['ready']:
                        logger.debug(f"Sparse index '{sparse_name}' is not ready.")
                        return False
                except ApiException:
                    logger.debug(f"Sparse index '{sparse_name}' does not exist or is not accessible.")
                    return False
                    
            logger.debug("All configured Pinecone indexes are ready.")
            return True
            
        except ApiException as e:
            logger.error(f"API error during readiness check: {e}")
            return False

    def collection_exists(self) -> bool:
        """
        Checks if the collection (index) specified in the config already exists in Pinecone.
        """
        if not self._is_connected or not self._client:
            raise Exception("Must be connected to Pinecone to check if a collection exists.")
        try:
            existing_indexes = self._client.list_indexes().names()
            safe_collection_name = self._sanitize_collection_name(self._config.core.collection_name)
            dense_exists = f"{safe_collection_name}-dense" in existing_indexes
            sparse_exists = f"{safe_collection_name}-sparse" in existing_indexes
            
            if self._config.indexing.create_dense_index and self._config.indexing.create_sparse_index:
                return dense_exists and sparse_exists
            elif self._config.indexing.create_dense_index:
                return dense_exists
            elif self._config.indexing.create_sparse_index:
                return sparse_exists
            return False
        except ApiException as e:
            raise Exception(f"Failed to list Pinecone indexes: {e}") from e

    def delete_collection(self) -> None:
        """
        Permanently deletes the collections specified in the config from Pinecone.
        This method will wait until indexes are no longer listed before returning.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Pinecone to delete collections.")
            
        collection_name = self._config.core.collection_name
        safe_collection_name = self._sanitize_collection_name(collection_name)
        logger.warning(f"Attempting to permanently delete collections for: '{collection_name}' (sanitized: '{safe_collection_name}')")
        
        try:
            if self._config.indexing.create_dense_index:
                dense_name = f"{safe_collection_name}-dense"
                if dense_name in self._client.list_indexes().names():
                    logger.info(f"Deleting dense index '{dense_name}'...")
                    self._client.delete_index(dense_name)
                    self._wait_for_deletion(dense_name)
                    logger.info(f"Successfully deleted dense index '{dense_name}'.")
                else:
                    logger.info(f"Dense index '{dense_name}' did not exist, so no deletion was necessary.")
            
            if self._config.indexing.create_sparse_index:
                sparse_name = f"{safe_collection_name}-sparse"
                if sparse_name in self._client.list_indexes().names():
                    logger.info(f"Deleting sparse index '{sparse_name}'...")
                    self._client.delete_index(sparse_name)
                    self._wait_for_deletion(sparse_name)
                    logger.info(f"Successfully deleted sparse index '{sparse_name}'.")
                else:
                    logger.info(f"Sparse index '{sparse_name}' did not exist, so no deletion was necessary.")
                    
        except ApiException as e:
            raise VectorDBError(f"Failed to delete Pinecone collections for '{collection_name}': {e}") from e

    def create_collection(self) -> None:
        """
        Creates a new collection (index) in Pinecone according to the configuration.
        This method handles `recreate_if_exists` logic and waits for the index to be fully ready.
        The physical index created is a unified container, capable of storing dense, sparse,
        or hybrid vectors based on the logic enforced in the upsert phase.
        """
        if not self._is_connected or not self._client:
            raise Exception("Must be connected to Pinecone to create a collection.")

        if self._config.indexing.create_dense_index:
            self._create_dense_index()
            
        if self._config.indexing.create_sparse_index:
            self._create_sparse_index()


    def _create_dense_index(self) -> None:
        # Convert collection name to Pinecone-compatible format (lowercase, alphanumeric and hyphens only)
        safe_collection_name = self._sanitize_collection_name(self._config.core.collection_name)
        dense_name = f"{safe_collection_name}-dense"
        
        if dense_name in self._client.list_indexes().names():
            if self._config.core.recreate_if_exists:
                self._client.delete_index(dense_name)
                self._wait_for_deletion(dense_name)
            else:
                logger.info(f"Dense index '{dense_name}' already exists. Skipping creation.")
                return

        metadata_config = self._build_metadata_schema()
        
        logger.info(f"Creating dense index '{dense_name}'...")
        try:
            create_params = {
                "name": dense_name,
                "vector_type": "dense",
                "dimension": self._config.core.vector_size,
                "metric": self._DISTANCE_METRIC_MAP[self._config.core.distance_metric],
                "spec": ServerlessSpec(
                    cloud=self._config.core.cloud,
                    region=self._config.core.region
                ),
                "deletion_protection": "disabled"
            }
            
            if metadata_config:
                create_params["metadata_config"] = metadata_config
            
            self._client.create_index(**create_params)
            
            self._wait_for_index_ready(dense_name)
            self._dense_index = self._client.Index(dense_name)
            logger.info(f"Successfully created dense index '{dense_name}'.")
        except ApiException as e:
            raise Exception(f"Failed to create dense index '{dense_name}': {e}") from e

    def _create_sparse_index(self) -> None:
        # Convert collection name to Pinecone-compatible format (lowercase, alphanumeric and hyphens only)
        safe_collection_name = self._sanitize_collection_name(self._config.core.collection_name)
        sparse_name = f"{safe_collection_name}-sparse"
        
        if sparse_name in self._client.list_indexes().names():
            if self._config.core.recreate_if_exists:
                logger.info(f"Sparse index '{sparse_name}' exists and 'recreate_if_exists' is True. Deleting it first.")
                self._client.delete_index(sparse_name)
                self._wait_for_deletion(sparse_name)
            else:
                logger.info(f"Sparse index '{sparse_name}' already exists. Skipping creation.")
                self._sparse_index = self._client.Index(sparse_name)
                return

        metadata_config = self._build_metadata_schema()
        
        logger.info(f"Creating sparse index '{sparse_name}'...")
        try:
            create_params = {
                "name": sparse_name,
                "vector_type": "sparse",
                "metric": "dotproduct",  # Sparse indexes must use dotproduct
                "spec": ServerlessSpec(
                    cloud=self._config.core.cloud,
                    region=self._config.core.region
                ),
                "deletion_protection": "disabled"
            }
            
            if metadata_config:
                create_params["metadata_config"] = metadata_config
            
            self._client.create_index(**create_params)
            
            self._wait_for_index_ready(sparse_name)
            self._sparse_index = self._client.Index(sparse_name)
            logger.info(f"Successfully created sparse index '{sparse_name}'.")
        except ApiException as e:
            raise Exception(f"Failed to create sparse index '{sparse_name}': {e}") from e

    def _build_metadata_schema(self) -> Dict[str, Any]:
        if not self._config.indexing.payload_indexes:
            return None
        fields = {}
        for payload_index in self._config.indexing.payload_indexes:
            fields[payload_index.field_name] = {"filterable": True}
        return {"fields": fields}

    def _wait_for_index_ready(self, index_name: str) -> None:
        """Wait for index to be ready."""
        import time
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
                raise Exception(f"Timeout waiting for index '{index_name}' to become ready.")
            logger.debug(f"Waiting for index '{index_name}' to become ready...")
            time.sleep(10)


    def _wait_for_deletion(self, index_name: str) -> None:
        """Wait for index deletion to complete."""
        import time
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
    ) -> Generator:
        """
        A private helper to format and chunk data for efficient upserting.
        It constructs dictionaries in the precise format required by the Pinecone
        client, intelligently including dense or sparse values based on what is provided.
        """
        batch_size = self._config.data_management.batch_size
        batch = []
        for i in range(len(ids)):
            metadata = {}
            payload = payloads[i]
            
            for key, value in payload.items():
                if key == 'metadata' and isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        metadata[f"{key}_{nested_key}"] = nested_value
                else:
                    metadata[key] = value
            
            if chunks and chunks[i]:
                metadata["chunk"] = chunks[i]
            
            record_dict = {
                "id": str(ids[i]),
                "metadata": metadata
            }
            if vectors:
                # Convert to float32 for Pinecone compatibility
                import numpy as np
                record_dict["values"] = np.array(vectors[i], dtype=np.float32).tolist()
            if sparse_vectors:
                record_dict["sparse_values"] = sparse_vectors[i]
            
            batch.append(record_dict)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch

    def upsert(
        self, 
        vectors: List[List[float]], 
        payloads: List[Dict[str, Any]], 
        ids: List[Union[str, int]], 
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None, 
        **kwargs
    ) -> None:
        """
        Adds new data or updates existing data in the collection.
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to Pinecone. Please call connect() first.")

        logger.debug("Initiating upsert. Validating data against collection configuration...")
        
        cfg = self._config.indexing
        is_dense_provided = vectors is not None
        is_sparse_provided = sparse_vectors is not None
        
        if cfg.create_dense_index and cfg.create_sparse_index:
            # In hybrid mode, we can accept either dense-only or both types
            if not is_dense_provided and not is_sparse_provided:
                raise UpsertError("In Hybrid mode, at least one of 'vectors' or 'sparse_vectors' must be provided.")
            if is_dense_provided and is_sparse_provided:
                # Both provided - validate lengths match
                if not (len(ids) == len(payloads) == len(vectors) == len(sparse_vectors)):
                    raise UpsertError("In Hybrid mode, all lists must have the same length.")
            elif is_dense_provided:
                # Only dense provided - this is allowed for dense-only upserts
                if not (len(ids) == len(payloads) == len(vectors)):
                    raise UpsertError("In Hybrid mode with dense-only data, ids, payloads, and vectors must have the same length.")
            elif is_sparse_provided:
                # Only sparse provided - this is allowed for sparse-only upserts
                if not (len(ids) == len(payloads) == len(sparse_vectors)):
                    raise UpsertError("In Hybrid mode with sparse-only data, ids, payloads, and sparse_vectors must have the same length.")
        elif cfg.create_dense_index:
            if not is_dense_provided:
                raise UpsertError("In Dense-Only mode, 'vectors' must be provided.")
            if is_sparse_provided:
                raise UpsertError("Configuration mismatch: sparse_vectors provided but collection is dense-only.")
        elif cfg.create_sparse_index:
            if not is_sparse_provided:
                raise UpsertError("In Sparse-Only mode, 'sparse_vectors' must be provided.")
            if is_dense_provided:
                raise UpsertError("Configuration mismatch: vectors provided but collection is sparse-only.")
        
        logger.info(f"Starting upsert of {len(ids)} records...")
        namespace = self._config.advanced.namespace or ""

        try:
            if cfg.create_dense_index and self._dense_index and vectors:
                dense_batches = self._generate_batches(ids, payloads, vectors, None, chunks)
                for batch in dense_batches:
                    self._dense_index.upsert(vectors=batch, namespace=namespace, **kwargs)
                logger.info(f"Successfully upserted {len(ids)} records to dense index.")
            
            if cfg.create_sparse_index and self._sparse_index and sparse_vectors:
                sparse_batches = self._generate_batches(ids, payloads, None, sparse_vectors, chunks)
                for batch in sparse_batches:
                    self._sparse_index.upsert(vectors=batch, namespace=namespace, **kwargs)
                logger.info(f"Successfully upserted {len(ids)} records to sparse index.")
                
        except ApiException as e:
            raise UpsertError(f"An error occurred during Pinecone upsert operation: {e}") from e
        except Exception as e:
            raise UpsertError(f"A general error occurred during the upsert process: {e}") from e

    def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Removes data from collections by their unique identifiers."""
        if not self._is_connected:
            raise Exception("Not connected to Pinecone.")

        str_ids = [str(i) for i in ids]
        namespace = self._config.advanced.namespace or ""
        
        if self._dense_index:
            try:
                self._dense_index.delete(ids=str_ids, namespace=namespace, **kwargs)
                logger.info(f"Deleted {len(str_ids)} records from dense index.")
            except ApiException as e:
                logger.error(f"Failed to delete from dense index: {e}")
                
        if self._sparse_index:
            try:
                self._sparse_index.delete(ids=str_ids, namespace=namespace, **kwargs)
                logger.info(f"Deleted {len(str_ids)} records from sparse index.")
            except ApiException as e:
                logger.error(f"Failed to delete from sparse index: {e}")
    
    def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Fetch records from appropriate indexes."""
        results = []
        namespace = self._config.advanced.namespace or ""
        
        if self._dense_index:
            try:
                dense_response = self._dense_index.fetch(ids=[str(i) for i in ids], namespace=namespace, **kwargs)
                results.extend(self._parse_fetch_response(dense_response))
            except ApiException as e:
                logger.warning(f"Dense fetch failed: {e}")
                
        if self._sparse_index and not results:
            try:
                sparse_response = self._sparse_index.fetch(ids=[str(i) for i in ids], namespace=namespace, **kwargs)
                results.extend(self._parse_fetch_response(sparse_response))
            except ApiException as e:
                logger.warning(f"Sparse fetch failed: {e}")
                
        return results
    
    def _parse_fetch_response(self, response) -> List[VectorSearchResult]:
        """Parse fetch response into VectorSearchResult objects."""
        from upsonic.schemas.vector_schemas import VectorSearchResult
        
        results = []
        if hasattr(response, 'vectors'):
            fetched_vectors = response.vectors
        else:
            fetched_vectors = response.get('vectors', {})
        
        for record_id, vector_data in fetched_vectors.items():
            metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else vector_data.get('metadata', {})
            values = vector_data.values if hasattr(vector_data, 'values') else vector_data.get('values')
            
            results.append(
                VectorSearchResult(
                    id=record_id,
                    score=1.0,
                    payload=metadata,
                    vector=values,
                    text=metadata.get('chunk') if metadata else None
                )
            )
        return results

    def _parse_query_response(self, response) -> List[VectorSearchResult]:
        """A private helper to transform Pinecone's query response into a standardized list of VectorSearchResult objects."""
        if hasattr(response, 'matches'):
            matches = response.matches
        else:
            matches = response.get('matches', [])
        
        results = []
        for match in matches:
            if hasattr(match, 'id'):
                match_id = match.id
                score = match.score
                metadata = match.metadata if hasattr(match, 'metadata') else {}
                values = match.values if hasattr(match, 'values') else None
            else:
                match_id = match['id']
                score = match['score']
                metadata = match.get('metadata', {})
                values = match.get('values')
            
            results.append(
                VectorSearchResult(
                    id=match_id,
                    score=score,
                    payload=metadata,
                    vector=values,
                    text=metadata.get('chunk') if metadata else None
                )
            )
        return results

    def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """Performs a pure vector similarity search."""
        if not self._dense_index: raise VectorDBConnectionError("Not connected to a Pinecone index.")
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)
        
        namespace = self._config.advanced.namespace or ""
        try:
            response = self._dense_index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True,
                include_values=True,
                **kwargs
            )
            results = self._parse_query_response(response)
            
            # For Pinecone with cosine similarity, scores are typically in range [0, 1]
            # Higher scores indicate better similarity
            # Apply similarity threshold correctly based on distance metric
            filtered_results = []
            for result in results:
                should_include = False
                
                if self._config.core.distance_metric == DistanceMetric.COSINE:
                    # Cosine similarity: 1.0 = identical, 0.0 = orthogonal, higher is better
                    should_include = result.score >= final_similarity_threshold
                elif self._config.core.distance_metric == DistanceMetric.DOT_PRODUCT:
                    # Dot product: higher is better (can be negative)
                    should_include = result.score >= final_similarity_threshold
                elif self._config.core.distance_metric == DistanceMetric.EUCLIDEAN:
                    # Euclidean distance: lower is better, but Pinecone returns similarity-like scores
                    # For euclidean, Pinecone might normalize to similarity format
                    should_include = result.score >= final_similarity_threshold
                else:
                    # Default case
                    should_include = result.score >= final_similarity_threshold
                
                if should_include:
                    filtered_results.append(result)
            
            logger.debug(f"Dense search found {len(results)} results, {len(filtered_results)} after similarity threshold {final_similarity_threshold}")
            return filtered_results
        except ApiException as e:
            raise SearchError(f"Dense search operation failed in Pinecone: {e}") from e

    def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """Performs a sparse-vector-only lexical search."""
        if not self._sparse_index: raise VectorDBConnectionError("Not connected to a Pinecone index.")
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)
        
        sparse_vector = kwargs.pop("sparse_vector", None)
        if not sparse_vector:
            result = self._client.inference.embed(
                model="pinecone-sparse-english-v0",
                inputs=query_text,
                parameters={"input_type": "query", "truncate": "END"}
            )
            sparse_vector = {'indices': result[0]['sparse_indices'], 'values': result[0]['sparse_values']}
        
        namespace = self._config.advanced.namespace or ""
        try:
            response = self._sparse_index.query(
                sparse_vector=sparse_vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True,
                include_values=True,
                **kwargs
            )
            results = self._parse_query_response(response)
            
            filtered_results = [result for result in results if result.score >= final_similarity_threshold]
            return filtered_results
        except ApiException as e:
            raise SearchError(f"Full-text (sparse) search operation failed in Pinecone: {e}") from e

    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs advanced hybrid search combining dense and sparse vectors with multiple fusion strategies.
        
        This implementation follows Pinecone's recommended approach for separate dense and sparse indexes,
        combining results with sophisticated ranking and deduplication strategies.
        """
        if not self._dense_index or not self._sparse_index:
            raise VectorDBConnectionError("Both dense and sparse indexes required for hybrid search.")
        
        namespace = self._config.advanced.namespace or ""
        
        if alpha is None:
            alpha = 0.5
        
        if fusion_method is None:
            fusion_method = 'weighted'
        
        try:
            dense_response = self._dense_index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True,
                include_values=True,
                **kwargs
            )
            dense_results = self._parse_query_response(dense_response)
            
            sparse_vector = kwargs.pop("sparse_vector", None)
            if not sparse_vector:
                result = self._client.inference.embed(
                    model="pinecone-sparse-english-v0",
                    inputs=query_text,
                    parameters={"input_type": "query", "truncate": "END"}
                )
                sparse_vector = {'indices': result[0]['sparse_indices'], 'values': result[0]['sparse_values']}
            
            sparse_response = self._sparse_index.query(
                sparse_vector=sparse_vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True,
                include_values=True,
                **kwargs
            )
            sparse_results = self._parse_query_response(sparse_response)
            
            final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)
            
            combined_results = self._merge_hybrid_results(
                dense_results, sparse_results, alpha, fusion_method, top_k
            )
            
            filtered_results = [result for result in combined_results if result.score >= final_similarity_threshold]
            
            return filtered_results[:top_k]
            
        except ApiException as e:
            raise SearchError(f"Hybrid search operation failed in Pinecone: {e}") from e

    def _merge_hybrid_results(
        self, 
        dense_results: List[VectorSearchResult], 
        sparse_results: List[VectorSearchResult], 
        alpha: float, 
        fusion_method: str,
        top_k: int
    ) -> List[VectorSearchResult]:
        """
        Advanced result fusion implementing multiple strategies for combining dense and sparse search results.
        """
        dense_lookup = {result.id: result for result in dense_results}
        sparse_lookup = {result.id: result for result in sparse_results}
        
        all_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())
        
        merged_results = []
        
        for doc_id in all_ids:
            dense_result = dense_lookup.get(doc_id)
            sparse_result = sparse_lookup.get(doc_id)
            
            if fusion_method == 'weighted':
                final_score = self._weighted_fusion(dense_result, sparse_result, alpha)
            elif fusion_method == 'rrf':
                final_score = self._reciprocal_rank_fusion(
                    dense_result, sparse_result, dense_results, sparse_results
                )
            else:
                final_score = self._weighted_fusion(dense_result, sparse_result, alpha)
            
            base_result = dense_result if dense_result else sparse_result
            
            merged_results.append(VectorSearchResult(
                id=doc_id,
                score=final_score,
                payload=base_result.payload,
                vector=base_result.vector
            ))
        
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results

    def _weighted_fusion(
        self, 
        dense_result: Optional[VectorSearchResult], 
        sparse_result: Optional[VectorSearchResult], 
        alpha: float
    ) -> float:
        """
        Implements weighted score fusion: alpha * dense_score + (1 - alpha) * sparse_score
        """
        dense_score = dense_result.score if dense_result else 0.0
        sparse_score = sparse_result.score if sparse_result else 0.0
        
        dense_normalized = self._normalize_score(dense_score, 'dense')
        sparse_normalized = self._normalize_score(sparse_score, 'sparse')
        
        return alpha * dense_normalized + (1 - alpha) * sparse_normalized

    def _reciprocal_rank_fusion(
        self, 
        dense_result: Optional[VectorSearchResult], 
        sparse_result: Optional[VectorSearchResult],
        dense_results: List[VectorSearchResult],
        sparse_results: List[VectorSearchResult],
        k: int = 60
    ) -> float:
        """
        Implements Reciprocal Rank Fusion (RRF) scoring.
        RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        rrf_score = 0.0
        
        if dense_result:
            dense_rank = next((i for i, r in enumerate(dense_results) if r.id == dense_result.id), len(dense_results))
            rrf_score += 1 / (k + dense_rank + 1)
        
        if sparse_result:
            sparse_rank = next((i for i, r in enumerate(sparse_results) if r.id == sparse_result.id), len(sparse_results))
            rrf_score += 1 / (k + sparse_rank + 1)
        
        return rrf_score

    def _normalize_score(self, score: float, search_type: str) -> float:
        """
        Normalize scores to [0, 1] range for fair combination.
        Dense scores (cosine similarity) are typically in [0, 1] or [-1, 1]
        Sparse scores (dot product) can vary widely
        """
        if search_type == 'dense':
            return (score + 1) / 2 if score < 0 else score
        elif search_type == 'sparse':
            import math
            return 1 / (1 + math.exp(-score / 10))
        
        return score

    def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        A master search method that validates the user's intent against the
        collection's configuration and dispatches to the appropriate specialized
        search function.
        """
        is_dense_intent = query_vector is not None
        is_sparse_intent = query_text is not None
        
        fusion_method = fusion_method if fusion_method is not None else self._config.search.default_fusion_method or 'weighted'

        final_top_k = top_k if top_k is not None else self._config.search.default_top_k
        if final_top_k is None:
            raise ConfigurationError("A 'top_k' value must be provided either in the search call or as 'default_top_k' in the configuration.")

        final_filter = filter if filter is not None else self._config.search.filter
        
        if is_dense_intent and is_sparse_intent:
            logger.debug("Dispatching to HYBRID search.")
            if not self._config.search.hybrid_search_enabled:
                raise ConfigurationError("Hybrid search was requested, but it is disabled in the provider's SearchConfig.")
            if not (self._dense_index and self._sparse_index):
                raise VectorDBConnectionError("Hybrid search requires both dense and sparse indexes to be available.")
            return self.hybrid_search(query_vector, query_text, final_top_k, final_filter, alpha, fusion_method, similarity_threshold, **kwargs)
        

        elif is_dense_intent:
            logger.debug("Dispatching to DENSE search.")
            if not self._config.search.dense_search_enabled:
                raise ConfigurationError("Dense search was requested, but it is disabled in the provider's SearchConfig.")
            if not self._dense_index:
                raise VectorDBConnectionError("Dense search requires a dense index to be available.")
            return self.dense_search(query_vector, final_top_k, final_filter, similarity_threshold, **kwargs)
            

        elif is_sparse_intent:
            logger.debug("Dispatching to FULL-TEXT search.")
            if not self._config.search.full_text_search_enabled:
                raise ConfigurationError("Full-text search was requested, but it is disabled in the provider's SearchConfig.")
            if not self._sparse_index:
                raise VectorDBConnectionError("Full-text search requires a sparse index to be available.")
            return self.full_text_search(query_text, final_top_k, final_filter, similarity_threshold, **kwargs)
            
        else:
            raise SearchError("Search requires at least one of 'query_vector' or 'query_text' to be provided.")