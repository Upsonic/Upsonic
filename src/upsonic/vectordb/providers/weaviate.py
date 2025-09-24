import uuid 
from typing import Any, Dict, List, Optional, Union, Literal

import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import (
    WeaviateConnectionError,
    UnexpectedStatusCodeError,
)
from weaviate.util import generate_uuid5
from weaviate.classes.query import HybridFusion

from upsonic.vectordb.config import (
    Config, 
    Mode, 
    DistanceMetric, 
    WriteConsistency,
    HNSWTuningConfig
)
# Also import from relative path to handle import path differences
from ..config import HNSWTuningConfig as RelativeHNSWTuningConfig
from upsonic.vectordb.base import BaseVectorDBProvider

from upsonic.utils.package.exception import(
    VectorDBConnectionError, 
    ConfigurationError, 
    CollectionDoesNotExistError,
    VectorDBError,
    SearchError,
    UpsertError
)

from upsonic.schemas.vector_schemas import VectorSearchResult

class WeaviateProvider(BaseVectorDBProvider):
    """
    An implementation of the BaseVectorDBProvider for the Weaviate vector database.
    
    This provider translates the abstract framework configurations and calls into
    concrete operations using the Weaviate Python Client v4. It handles connection
    management, schema creation, data operations, and search functionalities
    as defined by the framework's contract.
    """

    def __init__(self, config: Config):
        """
        Initializes the WeaviateProvider with a complete configuration.
        
        This step validates that the provider is configured for Weaviate and stores
        the configuration. No actual connection is established until `connect()` is called.

        Args:
            config: A validated and immutable Config object.

        Raises:
            ConfigurationError: If the provided config is not for the 'weaviate' provider.
        """
        if config.core.provider_name.value != 'weaviate':
            raise ConfigurationError(
                f"Attempted to initialize WeaviateProvider with a configuration for "
                f"'{config.core.provider_name.value}'."
            )
        
        super().__init__(config)
        self._client: Optional[weaviate.WeaviateClient] = None
        print(f"WeaviateProvider initialized for collection '{self._config.core.collection_name}' in '{self._config.core.mode.value}' mode.")


    def connect(self) -> None:
        """
        Establishes a connection to the Weaviate vector database instance.

        This method interprets the `CoreConfig` to determine the connection mode
        (cloud, local, embedded) and uses the appropriate Weaviate client constructor.
        It then verifies the connection is live and ready for operations.

        Raises:
            VectorDBConnectionError: If the connection fails for any reason, such as
                                     authentication errors, network issues, or a
                                     non-responsive Weaviate instance.
        """

        if self._is_connected and self._client:
            print("Already connected to Weaviate.")
            return

        print(f"Attempting to connect to Weaviate in '{self._config.core.mode.value}' mode...")
        
        try:
            if self._config.core.mode == Mode.CLOUD:
                if not self._config.core.host or not self._config.core.api_key:
                    raise ConfigurationError("Cloud mode requires 'host' (cluster URL) and 'api_key'.")
                
                auth_credentials = weaviate.auth.AuthApiKey(self._config.core.api_key.get_secret_value())
                additional_config = wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=60, query=30, insert=30),
                    startup_period=30
                )
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self._config.core.host,
                    auth_credentials=auth_credentials,
                    additional_config=additional_config,
                    skip_init_checks=True  # Skip gRPC health checks for network issues
                )

            elif self._config.core.mode == Mode.LOCAL:
                if not self._config.core.host or not self._config.core.port:
                     raise ConfigurationError("Local mode requires 'host' and 'port'.")

                self._client = weaviate.connect_to_local(
                    host=self._config.core.host,
                    port=self._config.core.port
                )

            elif self._config.core.mode == Mode.EMBEDDED or self._config.core.mode == Mode.IN_MEMORY:
                persistence_path = self._config.core.db_path if self._config.core.mode == Mode.EMBEDDED else None
                
                self._client = weaviate.connect_to_embedded(
                    persistence_data_path=persistence_path
                )
            
            else:
                raise ConfigurationError(f"Unsupported Weaviate mode: {self._config.core.mode.value}")

            if not self._client.is_ready():
                raise WeaviateConnectionError("Health check failed after connection attempt.")

            self._is_connected = True
            print("Successfully connected to Weaviate and health check passed.")

        except WeaviateConnectionError as e:
            self._client = None
            self._is_connected = False
            raise VectorDBConnectionError(f"Failed to connect to Weaviate: {e}")
        except Exception as e:
            self._client = None
            self._is_connected = False
            raise VectorDBConnectionError(f"An unexpected error occurred during connection: {e}")

    def disconnect(self) -> None:
        """
        Gracefully terminates the connection to the Weaviate database.
        
        This method is idempotent; calling it on an already disconnected
        provider will not raise an error.
        """
        if self._client and self._is_connected:
            try:
                self._client.close()
                self._is_connected = False
                self._client = None
                print("Successfully disconnected from Weaviate.")
            except Exception as e:
                self._is_connected = False
                self._client = None
                print(f"An error occurred during disconnection, but status is now 'disconnected'. Error: {e}")
        else:
            print("Already disconnected. No action taken.")

    def is_ready(self) -> bool:
        """
        Performs a health check to ensure the Weaviate instance is responsive.

        Returns:
            True if the client is connected and the database is responsive, False otherwise.
        """
        if not self._client or not self._is_connected:
            return False
        
        try:
            return self._client.is_ready()
        except WeaviateConnectionError:
            self._is_connected = False
            return False


    def create_collection(self) -> None:
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Weaviate before creating a collection.")

        collection_name = self._config.core.collection_name

        if self.collection_exists():
            if self._config.core.recreate_if_exists:
                print(f"Collection '{collection_name}' already exists. Deleting and recreating as requested.")
                self.delete_collection()
            else:
                print(f"Collection '{collection_name}' already exists and 'recreate_if_exists' is False. No action taken.")
                return

        try:
            distance_map = {
                DistanceMetric.COSINE: wvc.config.VectorDistances.COSINE,
                DistanceMetric.DOT_PRODUCT: wvc.config.VectorDistances.DOT,
                DistanceMetric.EUCLIDEAN: wvc.config.VectorDistances.L2_SQUARED,
            }
            pq_config = wvc.config.Configure.pq(training_limit=100000) if self._config.indexing.quantization else None
            
            index_conf = self._config.indexing.index_config
            if isinstance(index_conf, (HNSWTuningConfig, RelativeHNSWTuningConfig)):
                vector_index_config = wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=distance_map[self._config.core.distance_metric],
                    max_connections=index_conf.m,
                    ef_construction=index_conf.ef_construction,
                    quantizer=pq_config
                )
            else:
                raise ConfigurationError(f"Weaviate provider only supports HNSW index type, but '{index_conf.index_type.value}' was configured.")
            
            properties = []
            if self._config.indexing.payload_indexes:
                datatype_map = {'keyword': wvc.config.DataType.TEXT, 'text': wvc.config.DataType.TEXT, 'integer': wvc.config.DataType.INT, 'float': wvc.config.DataType.NUMBER, 'boolean': wvc.config.DataType.BOOL, 'geo': wvc.config.DataType.GEO_COORDINATES}
                tokenization_map = {'keyword': wvc.config.Tokenization.WORD, 'text': wvc.config.Tokenization.WHITESPACE}
                for prop_config in self._config.indexing.payload_indexes:
                    properties.append(wvc.config.Property(name=prop_config.field_name, data_type=datatype_map[prop_config.field_schema_type], tokenization=tokenization_map.get(prop_config.field_schema_type)))
            properties.append(
                wvc.config.Property(
                    name="chunk",
                    data_type=wvc.config.DataType.TEXT,
                    tokenization=wvc.config.Tokenization.WHITESPACE
                )
            )
            sharding_config = wvc.config.Configure.sharding(desired_count=self._config.advanced.num_shards) if self._config.advanced.num_shards is not None else None
            replication_config = wvc.config.Configure.replication(factor=self._config.advanced.replication_factor) if self._config.advanced.replication_factor is not None else None
            multi_tenancy_config = wvc.config.Configure.multi_tenancy(enabled=True) if self._config.advanced.namespace is not None else None
            self._client.collections.create(
                name=collection_name,
                    vector_config=wvc.config.Configure.Vectors.self_provided(vector_index_config=vector_index_config),
                properties=properties if properties else None,
                sharding_config=sharding_config,
                replication_config=replication_config,
                multi_tenancy_config=multi_tenancy_config
            )
            print(f"Successfully created collection '{collection_name}'.")

            if self._config.advanced.namespace:
                print(f"Multi-tenancy is enabled. Creating tenant: '{self._config.advanced.namespace}'...")
                collection = self._client.collections.get(collection_name)
                collection.tenants.create(
                    tenants=[weaviate.collections.classes.tenants.Tenant(name=self._config.advanced.namespace)]
                )
                print("Tenant created successfully.")

        except UnexpectedStatusCodeError as e:
            raise VectorDBError(f"Failed to create collection '{collection_name}' in Weaviate. Status: {e.status_code}. Message: {e.message}")
        except Exception as e:
            raise VectorDBError(f"An unexpected error occurred during collection creation: {e}")

    def delete_collection(self) -> None:
        """
        Permanently deletes the collection specified in the config from Weaviate.
        This operation is irreversible and is a core part of lifecycle management.

        Raises:
            VectorDBConnectionError: If not connected to the database.
            CollectionDoesNotExistError: If the collection to be deleted does not exist,
                                         providing a clear, actionable error.
            VectorDBError: For other unexpected API or operational errors.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Weaviate before deleting a collection.")
        
        collection_name = self._config.core.collection_name
        
        try:
            self._client.collections.delete(collection_name)
            print(f"Successfully deleted collection '{collection_name}'.")
        except UnexpectedStatusCodeError as e:
            if e.status_code == 404: 
                 raise CollectionDoesNotExistError(f"Collection '{collection_name}' could not be deleted because it does not exist.")
            else:
                 raise VectorDBError(f"API error while deleting collection '{collection_name}': {e.message}")
        except Exception as e:
            raise VectorDBError(f"An unexpected error occurred during collection deletion: {e}")

    def collection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists in Weaviate.
        This is a critical guard method to prevent accidental overwrites or errors.

        Returns:
            True if the collection exists, False otherwise.
        
        Raises:
            VectorDBConnectionError: If not connected to the database.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Weaviate to check for a collection's existence.")
        
        return self._client.collections.exists(self._config.core.collection_name)

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]] = None, **kwargs) -> None:
        """
        Adds new data or updates existing data in the collection using Weaviate's
        high-performance batching system.

        Args:
            vectors: A list of vector embeddings.
            payloads: A list of corresponding metadata objects (properties).
            ids: A list of unique identifiers. These will be deterministically
                 converted to UUIDs for idempotency.
            chunks: A list of text chunks
            **kwargs: Provider-specific options are ignored in this implementation but
                      could be used for overriding batch settings in the future.

        Raises:
            UpsertError: If the data ingestion fails for any reason.
            VectorDBConnectionError: If not connected to the database.
        """
        if not (len(vectors) == len(payloads) == len(ids)):
            raise UpsertError("The lengths of vectors, payloads, and ids lists must be identical.")
        if not vectors:
            print("Upsert called with empty lists. No action taken.")
            return
        if chunks is not None:
            if len(chunks) != len(payloads):
                raise UpsertError("The lengths of chunks and payloads lists must be identical.")
            for i in range(len(payloads)):
                payloads[i]["chunk"] = chunks[i]

        collection_obj = self._get_collection()

        consistency_map = {
            WriteConsistency.STRONG: wvc.config.ConsistencyLevel.ALL,
            WriteConsistency.EVENTUAL: wvc.config.ConsistencyLevel.ONE,
        }
        consistency_level = consistency_map[self._config.data_management.write_consistency]

        try:
            print(f"Starting upsert of {len(vectors)} objects with batch size {self._config.data_management.batch_size}...")
            collection_with_consistency = collection_obj.with_consistency_level(consistency_level)
            with collection_with_consistency.batch.fixed_size(
                batch_size=self._config.data_management.batch_size,
            ) as batch:
                for i in range(len(vectors)):
                    try:
                        object_uuid = uuid.UUID(str(ids[i]))
                    except ValueError:
                        object_uuid = generate_uuid5(identifier=ids[i], namespace=self._config.core.collection_name)

                    batch.add_object(
                        properties=payloads[i],
                        vector=vectors[i],
                        uuid=object_uuid
                    )
            
            print(f"Successfully upserted {len(vectors)} objects.")

        except Exception as e:
            failed_objects = batch.failed_objects # WILL BE IMPLEMENTED LATER!!!
            raise UpsertError(f"Failed to upsert data to Weaviate collection '{self._config.core.collection_name}': {e}")


    def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers.

        Args:
            ids: A list of specific IDs to remove.
            **kwargs: Ignored.

        Raises:
            VectorDBError: If the deletion fails.
        """
        if not ids:
            print("Delete called with an empty list of IDs. No action taken.")
            return
        
        collection_obj = self._get_collection()

        uuids_to_delete = []
        for item_id in ids:
            try:
                uuids_to_delete.append(uuid.UUID(str(item_id)))
            except ValueError:
                uuids_to_delete.append(generate_uuid5(identifier=item_id, namespace=self._config.core.collection_name))

        try:

            delete_filter = wvc.query.Filter.by_id().contains_any(uuids_to_delete)
            
            result = collection_obj.data.delete_many(where=delete_filter)
            
            if result.failed > 0:
                 raise VectorDBError(f"Deletion partially failed. Successful: {result.successful}, Failed: {result.failed}. Check Weaviate logs for details.")

            print(f"Successfully processed deletion request for {len(ids)} IDs. Matched and deleted: {result.successful}.")

        except Exception as e:
            raise VectorDBError(f"An error occurred during deletion: {e}")


    def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs.

        Args:
            ids: A list of IDs to retrieve the full records for.
            **kwargs: Ignored.

        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        if not ids:
            return []
            
        collection_obj = self._get_collection()

        uuids_to_fetch = []
        for item_id in ids:
            try:
                uuids_to_fetch.append(uuid.UUID(str(item_id)))
            except ValueError:
                uuids_to_fetch.append(generate_uuid5(identifier=item_id, namespace=self._config.core.collection_name))

        try:
            fetch_filter = wvc.query.Filter.by_id().contains_any(uuids_to_fetch)

            response = collection_obj.query.fetch_objects(
                limit=len(ids),
                filters=fetch_filter,
                include_vector=True
            )
            
            results = []
            for obj in response.objects:
                results.append(VectorSearchResult(
                    id=str(obj.uuid),
                    score=1.0, 
                    payload=obj.properties,
                    vector=obj.vector.get('default') if obj.vector else None,
                    text=obj.properties["chunk"]
                ))
            return results
        except Exception as e:
            error_message = str(e).lower()
            if "could not find class" in error_message and "in schema" in error_message:
                raise CollectionDoesNotExistError(f"Collection '{self._config.core.collection_name}' does not exist in Weaviate.")
            else:
                raise VectorDBError(f"An error occurred while fetching objects: {e}")


    def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        A master search method that dispatches to the appropriate specialized
        search function based on the provided arguments.

        This is the primary entry point for all search queries.

        Raises:
            ConfigurationError: If the requested search is not possible with the
                                provided arguments (e.g., asking for hybrid search
                                without both text and vector).
            SearchError: If any underlying search operation fails.
        """
        filter = filter if filter is not None else self._config.search.filter
        final_top_k = top_k if top_k is not None else self._config.search.default_top_k or 10

        fusion_method = fusion_method if fusion_method is not None else self._config.search.default_fusion_method or 'weighted'

        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None


        if is_dense:
            if self._config.search.dense_search_enabled is False:
                raise ConfigurationError("Dense search is disabled by the current configuration.")
            return self.dense_search(query_vector=query_vector, top_k=final_top_k, filter=filter, similarity_threshold=similarity_threshold, **kwargs)
        
        elif is_full_text:
            if self._config.search.full_text_search_enabled is False:
                raise ConfigurationError("Full-text search is disabled by the current configuration.")
            return self.full_text_search(query_text=query_text, top_k=final_top_k, filter=filter, similarity_threshold=similarity_threshold, **kwargs)

        elif is_hybrid:
            if self._config.search.hybrid_search_enabled is False:
                raise ConfigurationError("Hybrid search is disabled by the current configuration.")
            final_alpha = alpha if alpha is not None else self._config.search.default_hybrid_alpha or 0.5
            return self.hybrid_search(
                query_vector=query_vector, 
                query_text=query_text, 
                top_k=final_top_k, 
                filter=filter, 
                alpha=final_alpha, 
                fusion_method=fusion_method, 
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        else:
            raise ConfigurationError("Search requires at least one of 'query_vector' or 'query_text'.")


    def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """ 
        Performs a pure vector similarity search using Weaviate's `near_vector` query.

        Args:
            query_vector: The vector embedding to search for.
            top_k: The number of top results to return.
            filter: An optional metadata filter dictionary to apply.
            **kwargs: Can include `score_threshold` for filtering by certainty.

        Returns:
            A list of the most similar results, translated into the standard VectorSearchResult format.
        """
        collection_obj = self._get_collection()

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)

        try:
            weaviate_filter = self._translate_filter(filter) if filter else None
            
            score_threshold = kwargs.get('score_threshold')

            response = collection_obj.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=weaviate_filter,
                certainty=score_threshold,
                return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True), 
                include_vector=True
            )

            results = []
            for obj in response.objects:
                certainty = obj.metadata.certainty if obj.metadata and obj.metadata.certainty is not None else None
                distance = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else None
                                
                if certainty is not None:
                    score = certainty
                elif distance is not None:
                    score = 1.0 - distance
                else:
                    score = 0.0

                if score >= final_similarity_threshold:
                    results.append(VectorSearchResult(
                        id=str(obj.uuid),
                        score=score,
                        payload=obj.properties,
                        vector=obj.vector.get('default') if obj.vector else None,
                        text=obj.properties["chunk"]
                    ))
            
            return results

        except Exception as e:
            raise SearchError(f"An error occurred during dense search: {e}")

    def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a full-text (keyword) search using Weaviate's BM25 algorithm.

        Args:
            query_text: The text string to search for.
            top_k: The number of top results to return.
            filter: An optional metadata filter to apply before the search.
            **kwargs: Ignored in this implementation.

        Returns:
            A list of matching results, ordered by BM25 relevance score.
        """
        collection_obj = self._get_collection()

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)

        try:
            weaviate_filter = self._translate_filter(filter) if filter else None

            response = collection_obj.query.bm25(
                query=query_text,
                limit=top_k,
                filters=weaviate_filter,
                return_metadata=wvc.query.MetadataQuery(score=True),
                include_vector=True
            )

            results = []
            for obj in response.objects:
                score = obj.metadata.score if obj.metadata and obj.metadata.score is not None else 0.0

                if score >= final_similarity_threshold:
                    results.append(VectorSearchResult(
                        id=str(obj.uuid),
                        score=score,
                        payload=obj.properties,
                        vector=obj.vector.get('default') if obj.vector else None,
                        text=obj.properties["chunk"]
                    ))
            
            return results

        except Exception as e:
            raise SearchError(f"An error occurred during full-text search: {e}")


    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Combines dense and sparse search results using Weaviate's native hybrid query.

        Args:
            query_vector: The dense vector for the semantic part of the search.
            query_text: The raw text for the keyword/sparse part of the search.
            top_k: The number of final results to return.
            filter: An optional metadata filter.
            alpha: The weight for combining scores (0.0 = pure keyword, 1.0 = pure vector).
            fusion_method: NOTE: Weaviate's native hybrid endpoint uses its own fusion
                         algorithm. This parameter is ignored in favor of Weaviate's
                         default behavior, which is controlled by `alpha`.

        Returns:
            A list of VectorSearchResult objects, ordered by the combined hybrid score.
        """
        collection_obj = self._get_collection()
        
        final_alpha = alpha if alpha is not None else self._config.search.default_hybrid_alpha or 0.5

        if not (0.0 <= final_alpha <= 1.0):
            raise ConfigurationError(f"Hybrid search alpha must be between 0.0 and 1.0, but got {final_alpha}.")

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)

        fusion_type = None
        if fusion_method is not None:
            if fusion_method == "rrf":
                fusion_type = HybridFusion.RANKED
            elif fusion_method == "weighted":
                fusion_type = HybridFusion.RELATIVE_SCORE
            else:
                raise ConfigurationError(f"Unsupported fusion_method '{fusion_method}'. Use 'rrf' or 'weighted'.")
            

        try:
            weaviate_filter = self._translate_filter(filter) if filter else None
            
            response = collection_obj.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=final_alpha,
                limit=top_k,
                filters=weaviate_filter,
                fusion_type=fusion_type,
                return_metadata=wvc.query.MetadataQuery(score=True),
                include_vector=True
            )

            results = []
            for obj in response.objects:
                score = obj.metadata.score if obj.metadata and obj.metadata.score is not None else 0.0
                
                if score >= final_similarity_threshold:
                    results.append(VectorSearchResult(
                        id=str(obj.uuid),
                        score=score,
                        payload=obj.properties,
                        vector=obj.vector.get('default') if obj.vector else None,
                        text=obj.properties["chunk"]
                    ))

            return results
            
        except Exception as e:
            raise SearchError(f"An error occurred during hybrid search: {e}")
    
    def _get_collection(self) -> weaviate.collections.Collection:
        """
        Private helper to get the collection object, applying tenancy if configured.
        This centralizes collection retrieval and ensures all data/query operations
        are correctly scoped.
        """
        if not self._client or not self._is_connected:
            raise VectorDBConnectionError("Client is not connected.")
        
        try:
            collection = self._client.collections.get(self._config.core.collection_name)
        except UnexpectedStatusCodeError as e:
            if e.status_code == 404:
                raise CollectionDoesNotExistError(f"Collection '{self._config.core.collection_name}' does not exist in Weaviate.")
            raise VectorDBError(f"Failed to retrieve collection: {e.message}")
        
        if self._config.advanced.namespace:
            return collection.with_tenant(self._config.advanced.namespace)
    
        return collection
    
    def _translate_filter(self, filter_dict: Dict[str, Any]) -> wvc.query.Filter:
        """
        Recursively translates a framework-standard filter dictionary into a
        Weaviate Filter object. This is the core of provider-agnostic filtering.

        Args:
            filter_dict: A dictionary representing the filter logic.

        Returns:
            A Weaviate Filter object ready to be used in a query.
            
        Raises:
            SearchError: If an unknown operator or invalid filter structure is provided.
        """
        logical_ops = {
            "and": wvc.query.Filter.all_of,
            "or": wvc.query.Filter.any_of,
        }

        comparison_ops = {
            "$eq": lambda p, v: p.equal(v),
            "$ne": lambda p, v: p.not_equal(v),
            "$gt": lambda p, v: p.greater_than(v),
            "$gte": lambda p, v: p.greater_or_equal(v),
            "$lt": lambda p, v: p.less_than(v),
            "$lte": lambda p, v: p.less_or_equal(v),
            "$in": lambda p, v: p.contains_any(v),
        }

        filters = []
        for key, value in filter_dict.items():
            if key in logical_ops:
                sub_filters = [self._translate_filter(sub_filter) for sub_filter in value]
                return logical_ops[key](sub_filters)
            
            prop_filter = wvc.query.Filter.by_property(key)
            if isinstance(value, dict):
                if len(value) != 1:
                    raise SearchError(f"Field filter for '{key}' must have exactly one operator.")
                
                op, val = list(value.items())[0]
                if op in comparison_ops:
                    filters.append(comparison_ops[op](prop_filter, val))
                else:
                    raise SearchError(f"Unsupported filter operator '{op}' for field '{key}'.")
            else:
                filters.append(prop_filter.equal(value))

        if not filters:
            raise SearchError("Filter dictionary cannot be empty.")
        
        return wvc.query.Filter.all_of(filters) if len(filters) > 1 else filters[0]