from typing import Any, Dict, List, Optional, Union, Literal
from collections import defaultdict

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    raise ImportError(
        "The 'qdrant-client' package is required to use the Qdrant provider. "
        "Please install it with 'pip install qdrant-client'."
    )

from upsonic.vectordb.base import BaseVectorDBProvider

from upsonic.vectordb.config import (
    Config,
    Mode,
    IndexType,
    DistanceMetric,
    HNSWTuningConfig,
    IVFTuningConfig,
    FlatTuningConfig,
    WriteConsistency,
    PayloadIndexConfig
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
    An advanced vector database provider for Qdrant, fully integrated with the
    framework's configuration and operational contract.

    This class translates the abstract settings from a `Config` object into
    concrete Qdrant-specific API calls, managing the entire lifecycle of a
    collection from connection and creation to deletion.
    """

    DISTANCE_MAP = {
        DistanceMetric.COSINE: models.Distance.COSINE,
        DistanceMetric.EUCLIDEAN: models.Distance.EUCLID,
        DistanceMetric.DOT_PRODUCT: models.Distance.DOT,
    }

    def __init__(self, config: Config):
        """
        Initializes the QdrantProvider.

        Validates that the provided configuration is compatible with Qdrant's
        capabilities before proceeding.
        """
        super().__init__(config)
        if isinstance(self._config.indexing.index_config, IVFTuningConfig):
            raise ConfigurationError(
                "Qdrant provider does not support the 'IVF_FLAT' index_type. "
                "Please use 'HNSW' or 'FLAT'."
            )
        self._client: Optional[QdrantClient] = None

    def connect(self) -> None:
        """
        Establishes a connection to the Qdrant vector database.

        This method uses the connection parameters from `self._config.core`
        to initialize the QdrantClient based on the specified operational `mode`.

        Raises:
            VectorDBConnectionError: If the connection fails for any reason.
        """
        if self._is_connected:
            print("Already connected to Qdrant.")
            return

        core_cfg = self._config.core
        try:
            if core_cfg.mode == Mode.IN_MEMORY:
                self._client = QdrantClient(":memory:")
            elif core_cfg.mode == Mode.EMBEDDED:
                if not core_cfg.db_path:
                    raise ConfigurationError("'db_path' must be set for embedded mode.")
                self._client = QdrantClient(path=core_cfg.db_path)
            elif core_cfg.mode == Mode.LOCAL:
                self._client = QdrantClient(
                    host=core_cfg.host or "localhost",
                    port=core_cfg.port or 6333,
                    grpc_port=core_cfg.port + 1540 if core_cfg.port else 6334,
                )
            elif core_cfg.mode == Mode.CLOUD:
                # Remove port from URL for cloud connections
                url = core_cfg.host
                if url and ":6333" in url:
                    url = url.replace(":6333", "")
                self._client = QdrantClient(
                    url=url,
                    api_key=core_cfg.api_key.get_secret_value() if core_cfg.api_key else None
                )
            else:
                raise ConfigurationError(f"Unsupported mode for Qdrant: {core_cfg.mode.value}")

            self._is_connected = True
            print("Successfully connected to Qdrant.")

        except Exception as e:
            self._client = None
            self._is_connected = False
            raise VectorDBConnectionError(f"Failed to connect to Qdrant: {e}") from e

    def disconnect(self) -> None:
        """
        Gracefully terminates the connection to the Qdrant database.
        """
        if self._client:
            try:
                if hasattr(self._client, 'aclose'):
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(self._client.aclose())
                        else:
                            loop.run_until_complete(self._client.aclose())
                    except:
                        pass
                elif hasattr(self._client, 'close'):
                    self._client.close()
                
                print("Successfully disconnected from Qdrant.")
            except Exception as e:
                print(f"Error during Qdrant disconnection: {e}")
            finally:
                self._client = None
                self._is_connected = False
    
    async def disconnect_async(self) -> None:
        """
        Async version of disconnect for proper async cleanup.
        """
        if self._client:
            try:
                if hasattr(self._client, 'aclose'):
                    await self._client.aclose()
                elif hasattr(self._client, 'close'):
                    self._client.close()
                
                print("Successfully disconnected from Qdrant.")
            except Exception as e:
                print(f"Error during Qdrant disconnection: {e}")
            finally:
                self._client = None
                self._is_connected = False

    def is_ready(self) -> bool:
        """
        Performs a health check to ensure the Qdrant instance is responsive.
        """
        if not self._is_connected or not self._client:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def create_collection(self) -> None:
        """
        Creates the collection in Qdrant according to the full framework config.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Qdrant to create a collection.")

        collection_name = self._config.core.collection_name
        
        try:
            if self._config.core.recreate_if_exists and self.collection_exists():
                print(f"Collection '{collection_name}' exists and `recreate_if_exists` is True. Deleting...")
                self.delete_collection()
            
            vectors_config = models.VectorParams(
                size=self._config.core.vector_size,
                distance=self.DISTANCE_MAP[self._config.core.distance_metric]
            )

            hnsw_config = None
            quantization_config = None
            index_cfg = self._config.indexing.index_config

            if isinstance(index_cfg, HNSWTuningConfig):
                hnsw_config = models.HnswConfigDiff(
                    m=index_cfg.m,
                    ef_construct=index_cfg.ef_construction
                )
            elif isinstance(index_cfg, FlatTuningConfig):
                hnsw_config = None 
            
            if self._config.indexing.quantization:
                quant_cfg = self._config.indexing.quantization
                if quant_cfg.quantization_type == 'Scalar':
                    quantization_config = models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True
                        )
                    )

            sharding_cfg = self._config.advanced
            
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
                shard_number=sharding_cfg.num_shards,
                replication_factor=sharding_cfg.replication_factor,
            )
            print(f"Successfully created or configured collection '{collection_name}'.")
            if self._config.indexing.payload_indexes:
                self._create_payload_indexes(collection_name)

        except Exception as e:
            raise VectorDBError(f"Failed to create collection '{collection_name}': {e}") from e

    def delete_collection(self) -> None:
        """
        Permanently deletes the collection specified in the config.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Qdrant to delete a collection.")

        collection_name = self._config.core.collection_name
        try:
            operation_info = self._client.delete_collection(collection_name=collection_name)
            if type(operation_info) is not bool:
                if not operation_info.result:
                    if not self.collection_exists():
                        raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.")
                    else:
                        raise VectorDBError(f"Failed to delete collection '{collection_name}' but it still exists.")
            else:
                if not operation_info:
                    if not self.collection_exists():
                        raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.")
                    else:
                        raise VectorDBError(f"Failed to delete collection '{collection_name}' but it still exists.")
                    
            print(f"Successfully deleted collection '{collection_name}'.")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.") from e
            raise VectorDBError(f"API error while deleting collection '{collection_name}': {e}") from e
        except Exception as e:
            raise VectorDBError(f"An unexpected error occurred while deleting collection: {e}") from e

    def collection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to Qdrant to check for a collection.")
            
        collection_name = self._config.core.collection_name
        try:
            self._client.get_collection(collection_name=collection_name)
            return True
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return False
            raise VectorDBError(f"API error while checking for collection '{collection_name}': {e}") from e
        except Exception:
            return False

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]], **kwargs) -> None:
        """
        Adds new data or updates existing data in the Qdrant collection.

        This method transforms the framework-native data lists into Qdrant's
        `PointStruct` format and uses the high-performance `upsert` API.
        It respects the write consistency and batching parameters defined in the config.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to upsert data.")
        
        if not (len(vectors) == len(payloads) == len(ids)):
            raise ValueError("The lengths of vectors, payloads, and ids lists must be identical.")
        
        if chunks is not None and len(chunks) != len(vectors):
            raise ValueError("The length of the chunks list must be identical to the other lists.")

        points = []

        for i, (point_id, vector, payload) in enumerate(zip(ids, vectors, payloads)):
            if chunks:
                payload['chunk'] = chunks[i]
            
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        wait_for_result = self._config.data_management.write_consistency == WriteConsistency.STRONG

        try:
            self._client.upsert(
                collection_name=self._config.core.collection_name,
                points=points,
                wait=wait_for_result,
            )
        except Exception as e:
            raise UpsertError(f"Failed to upsert data into collection '{self._config.core.collection_name}': {e}") from e

    def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to delete data.")

        if not ids:
            return

        wait_for_result = self._config.data_management.write_consistency == WriteConsistency.STRONG

        try:
            self._client.delete(
                collection_name=self._config.core.collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=wait_for_result
            )
        except Exception as e:
            raise VectorDBError(f"Failed to delete points from collection '{self._config.core.collection_name}': {e}") from e

    def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their unique IDs.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to fetch data.")

        if not ids:
            return []

        try:
            retrieved_records: List[models.Record] = self._client.retrieve(
                collection_name=self._config.core.collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=True
            )

            search_results = [
                VectorSearchResult(
                    id=record.id,
                    score=1.0,
                    payload=record.payload,
                    vector=record.vector,
                    text=record.payload.get("chunk", "")
                )
                for record in retrieved_records
            ]
            return search_results

        except Exception as e:
            raise VectorDBError(f"Failed to fetch points from collection '{self._config.core.collection_name}': {e}") from e
        
    def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to perform a search.")

        filter = filter if filter is not None else self._config.search.filter
        effective_top_k = top_k if top_k is not None else self._config.search.default_top_k or 10

        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None

        if is_dense:
            if self._config.search.dense_search_enabled is False:
                raise ConfigurationError("Dense search is disabled by the current configuration.")
            return self.dense_search(query_vector, effective_top_k, filter, similarity_threshold, **kwargs)
        
        elif is_hybrid:
            if self._config.search.hybrid_search_enabled is False:
                raise ConfigurationError("Hybrid search is disabled by the current configuration.")
            return self.hybrid_search(query_vector, query_text, effective_top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs)

        elif is_full_text:
            if self._config.search.full_text_search_enabled is False:
                raise ConfigurationError("Full-text search is disabled by the current configuration.")
            return self.full_text_search(query_text, effective_top_k, filter, similarity_threshold, **kwargs)
        
        else:
            raise SearchError("Invalid search query: You must provide a 'query_vector' and/or 'query_text'.")

    def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        try:
            final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.search.default_similarity_threshold or 0.5
            
            search_params = models.SearchParams(
                hnsw_ef=kwargs.get('ef_search', self._config.search.default_ef_search),
                exact=False,
            )

            qdrant_filter = self._build_qdrant_filter(filter) if filter else None

            query_response = self._client.query_points(
                collection_name=self._config.core.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                search_params=search_params,
                limit=top_k,
                with_payload=True,
                with_vectors=True
            )

            filtered_results = []
            
            for point in query_response.points:
                # For Qdrant, scores need to be interpreted based on distance metric
                # Cosine: higher is better (0 to 1), Euclidean: lower is better, Dot: higher is better
                should_include = False
                
                if self._config.core.distance_metric == DistanceMetric.COSINE:
                    # Cosine similarity: 1.0 = identical, 0.0 = orthogonal, higher is better
                    should_include = point.score >= final_similarity_threshold
                elif self._config.core.distance_metric == DistanceMetric.DOT_PRODUCT:
                    # Dot product: higher is better (can be negative)
                    should_include = point.score >= final_similarity_threshold
                elif self._config.core.distance_metric == DistanceMetric.EUCLIDEAN:
                    # Euclidean distance: lower is better, so we need to invert the logic
                    # Convert threshold to distance: smaller distance = higher similarity
                    max_distance = 1.0 / final_similarity_threshold if final_similarity_threshold > 0 else float('inf')
                    should_include = point.score <= max_distance
                
                if should_include:
                    filtered_results.append(VectorSearchResult(
                        id=point.id,
                        score=point.score,
                        payload=point.payload,
                        vector=point.vector,
                        text=point.payload.get("chunk", "")
                    ))
            
            return filtered_results
        except Exception as e:
            raise SearchError(f"An error occurred during dense search: {e}") from e
    
    def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a full-text search using Qdrant's payload filtering.

        For this to be performant, a payload index should be created on the
        target text field in Qdrant beforehand.
        """
        if not self._is_connected or not self._client:
            raise VectorDBConnectionError("Must be connected to perform a full-text search.")

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.search.default_similarity_threshold or 0.5

        target_text_field = kwargs.get("text_search_field", "chunk")

        if self._config.core.mode == Mode.IN_MEMORY:
            try:
                records = self._client.scroll(
                    collection_name=self._config.core.collection_name,
                    limit=10000,  # Get all records
                    with_payload=True,
                    with_vectors=True,
                )
                
                query_words = query_text.lower().split()
                matching_records = []
                
                for record in records[0]:
                    if target_text_field in record.payload:
                        chunk_text = record.payload[target_text_field].lower()
                        if any(word in chunk_text for word in query_words):
                            word_matches = sum(1 for word in query_words if word in chunk_text)
                            relevance_score = word_matches / len(query_words)
                            
                            if relevance_score >= final_similarity_threshold:
                                matching_records.append(VectorSearchResult(
                                    id=record.id,
                                    score=relevance_score,
                                    payload=record.payload,
                                    vector=record.vector,
                                    text=record.payload.get("chunk", "")
                                ))
                
                matching_records.sort(key=lambda x: x.score, reverse=True)
                return matching_records[:top_k]
                
            except Exception as e:
                raise SearchError(f"An error occurred during full-text search: {e}") from e
        else:
            try:
                self._client.create_payload_index(
                    collection_name=self._config.core.collection_name,
                    field_name=target_text_field,
                    field_schema=models.TextIndexParams(type="text"),
                    wait=True
                )
            except Exception:
                pass

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
            
            try:
                records = self._client.scroll(
                    collection_name=self._config.core.collection_name,
                    scroll_filter=final_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=True,
                )
                filtered_results = []
                for r in records[0]:
                    score = 1.0  # Default score for full-text search
                    if score >= final_similarity_threshold:
                        filtered_results.append(VectorSearchResult(
                            id=r.id,
                            score=score,
                            payload=r.payload,
                            vector=r.vector,
                            text=r.payload.get("chunk", "")
                        ))
                return filtered_results
            except Exception as e:
                raise SearchError(f"An error occurred during full-text search: {e}") from e

    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Combines dense and full-text search results using a specified fusion method.
        """
        effective_alpha = alpha if alpha is not None else self._config.search.default_hybrid_alpha or 0.5
        effective_fusion = fusion_method if fusion_method is not None else self._config.search.default_fusion_method or 'weighted'

        dense_results = self.dense_search(query_vector, top_k, filter, similarity_threshold, **kwargs)
        ft_results = self.full_text_search(query_text, top_k, filter, similarity_threshold, **kwargs)

        if effective_fusion == 'weighted':
            fused_results = self._fuse_weighted(dense_results, ft_results, effective_alpha)
        elif effective_fusion == 'rrf':
            fused_results = self._fuse_rrf(dense_results, ft_results)
        else:
            raise ConfigurationError(f"Unsupported fusion method: '{effective_fusion}'")

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:top_k]

    def _fuse_weighted(self, list1: List[VectorSearchResult], list2: List[VectorSearchResult], alpha: float) -> List[VectorSearchResult]:
        """Combines two result lists using a weighted score."""
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
                text=original_doc.payload.get("chunk", "")
            ))
        
        return final_results

    def _fuse_rrf(self, list1: List[VectorSearchResult], list2: List[VectorSearchResult], k: int = 60) -> List[VectorSearchResult]:
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
                text=original_doc.payload.get("chunk", "")
            ))
            
        return final_results


    def _build_qdrant_filter(self, filter_dict: Dict[str, Any]) -> models.Filter:
        """
        A helper function to translate the framework's MongoDB-style filter
        dictionary into a Qdrant `models.Filter` object.
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
    
    def _create_payload_indexes(self, collection_name: str) -> None:
        """
        Private helper method to create payload indexes on a collection.
        """
        print("Applying payload indexes...")
        schema_map = {
            'text': models.TextIndexParams,
            'keyword': models.KeywordIndexParams,
            'integer': models.IntegerIndexParams,
            'float': models.FloatIndexParams,
            'boolean': models.BoolIndexParams,
            'geo': models.GeoIndexParams
        }

        for index_config in self._config.indexing.payload_indexes:
            field_name = index_config.field_name
            schema_type = index_config.field_schema_type
            params = index_config.params or {}

            if schema_type not in schema_map:
                print(f"Warning: Unknown payload index schema type '{schema_type}'. Skipping field '{field_name}'.")
                continue

            try:
                if schema_type == 'text':
                    qdrant_schema = models.TextIndexParams(type='text')
                elif schema_type == 'keyword':
                    qdrant_schema = models.KeywordIndexParams(type='keyword')
                elif schema_type == 'integer':
                    qdrant_schema = models.IntegerIndexParams(type='integer')
                elif schema_type == 'float':
                    qdrant_schema = models.FloatIndexParams(type='float')
                elif schema_type == 'boolean':
                    qdrant_schema = models.BoolIndexParams(type='bool')
                elif schema_type == 'geo':
                    qdrant_schema = models.GeoIndexParams(type='geo')
                else:
                    qdrant_schema = schema_map[schema_type](**params)
                
                print(f"  - Creating index for field '{field_name}' of type '{schema_type}'...")
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=qdrant_schema,
                    wait=True
                )

            except Exception as e:
                raise VectorDBError(
                    f"Failed to create payload index for field '{field_name}' "
                    f"in collection '{collection_name}': {e}"
                ) from e
        print("Payload indexes applied successfully.")