from typing import Any, Dict, List, Optional, Union, Literal

import chromadb
from chromadb.api.client import Client as ChromaClientAPI
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.errors import NotFoundError

from upsonic.vectordb.base import BaseVectorDBProvider

from upsonic.vectordb.config import (
    Config, 
    IndexTuningConfig,
    HNSWTuningConfig,
    IndexType, 
    Mode, 
    ProviderName, 
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


class ChromaProvider(BaseVectorDBProvider):
    """
    A concrete implementation of the BaseVectorDBProvider contract for ChromaDB.

    This class acts as a translator, mapping the framework's abstract commands
    and configuration into specific API calls for the `chromadb` client library.
    It handles connection logic, validation, and lifecycle management for all
    of Chroma's operational modes.
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        self._validate_config()
        self._collection_instance: Optional[ChromaCollection] = None
        
    def _validate_config(self) -> None:
        print("Performing Chroma-specific configuration validation...")
        if self._config.core.provider_name is not ProviderName.CHROMA:
            raise ConfigurationError(f"Configuration Mismatch: ChromaProvider received a config for '{self._config.core.provider_name.value}'.")
        if self._config.indexing.index_config.index_type not in [IndexType.HNSW, IndexType.FLAT]:
            raise ConfigurationError(f"Unsupported Index: ChromaProvider does not support index type '{self._config.indexing.index_config.index_type.value}'. Please use HNSW or FLAT.")
        if self._config.core.mode == Mode.EMBEDDED and self._config.core.db_path is None:
            raise ConfigurationError("Missing Path: 'db_path' must be set in CoreConfig for EMBEDDED (PersistentClient) mode.")
        print("Chroma configuration validated successfully.")


    def connect(self) -> None:
        if self._is_connected: return
        print(f"Connecting to ChromaDB in '{self._config.core.mode.value}' mode...")
        try:
            client_instance: ChromaClientAPI
            if self._config.core.mode == Mode.IN_MEMORY: client_instance = chromadb.Client()
            elif self._config.core.mode == Mode.EMBEDDED: client_instance = chromadb.PersistentClient(path=self._config.core.db_path)
            elif self._config.core.mode == Mode.LOCAL:
                if not self._config.core.host or not self._config.core.port: raise ConfigurationError("Host and port must be specified for LOCAL mode.")
                client_instance = chromadb.HttpClient(host=self._config.core.host, port=self._config.core.port)
            elif self._config.core.mode == Mode.CLOUD:
                if not self._config.core.api_key: raise ConfigurationError("api_key must be specified for CLOUD mode.")
                
                # Prepare CloudClient kwargs
                cloud_kwargs = {
                    "api_key": self._config.core.api_key.get_secret_value()
                }
                
                # Add tenant and database if provided in advanced config namespace
                if hasattr(self._config.advanced, 'namespace') and self._config.advanced.namespace:
                    # Parse namespace for tenant and database
                    if ',' in self._config.advanced.namespace:
                        parts = self._config.advanced.namespace.split(',')
                        for part in parts:
                            part = part.strip()
                            if 'tenant=' in part:
                                tenant = part.split('tenant=')[1].strip()
                                cloud_kwargs["tenant"] = tenant
                            elif 'database=' in part:
                                database = part.split('database=')[1].strip()
                                cloud_kwargs["database"] = database
                
                # Use CloudClient for Chroma Cloud connections
                try:
                    client_instance = chromadb.CloudClient(**cloud_kwargs)
                except (AttributeError, ImportError, TypeError) as e:
                    # Fallback to HttpClient if CloudClient is not available
                    if not self._config.core.host:
                        raise ConfigurationError("CloudClient not available and no host specified for fallback HttpClient.")
                    
                    headers = {"Authorization": f"Bearer {self._config.core.api_key.get_secret_value()}"}
                    fallback_kwargs = {
                        "host": self._config.core.host, 
                        "headers": headers,
                        "ssl": self._config.core.use_tls
                    }
                    client_instance = chromadb.HttpClient(**fallback_kwargs)
            else: raise ConfigurationError(f"Unsupported mode for ChromaProvider: {self._config.core.mode}")
            
            client_instance.heartbeat()
            self._client = client_instance
            self._is_connected = True
            print("✅ ChromaDB connection successful and verified.")
        except Exception as e: raise VectorDBConnectionError(f"Failed to connect to ChromaDB: {e}") from e

    def disconnect(self) -> None:
        if not self._is_connected or not self._client: return
        print("Disconnecting from ChromaDB...")
        try: 
            self._client.reset()
        except Exception:
            pass
        finally:
            self._client, self._is_connected, self._collection_instance = None, False, None
            print("ChromaDB client session has been reset.")

    def is_ready(self) -> bool:
        if not self._is_connected or not self._client: return False
        try:
            self._client.heartbeat()
            return True
        except Exception: return False

    def create_collection(self) -> None:
        if not self.is_ready(): raise VectorDBConnectionError("Cannot create collection: Provider is not connected or ready.")
        collection_name = self._config.core.collection_name
        try:
            if self._config.core.recreate_if_exists and self.collection_exists():
                print(f"Configuration specifies 'recreate_if_exists'. Deleting existing collection '{collection_name}'...")
                self.delete_collection()
            chroma_metadata = self._translate_config_to_chroma_metadata()
            print(f"Creating or retrieving collection '{collection_name}'...")
            self._collection_instance = self._client.get_or_create_collection(name=collection_name, metadata=chroma_metadata)
            print(f"✅ Successfully prepared collection '{collection_name}'.")
        except Exception as e: raise VectorDBError(f"Failed to create or get collection '{collection_name}': {e}") from e
            
    def _translate_config_to_chroma_metadata(self) -> dict:
        distance_map = {DistanceMetric.COSINE: "cosine", DistanceMetric.EUCLIDEAN: "l2", DistanceMetric.DOT_PRODUCT: "ip"}
        metadata = {"hnsw:space": distance_map[self._config.core.distance_metric]}
        if isinstance(self._config.indexing.index_config, HNSWTuningConfig):
            metadata["hnsw:M"] = self._config.indexing.index_config.m
            metadata["hnsw:construction_ef"] = self._config.indexing.index_config.ef_construction
        return metadata

    def delete_collection(self) -> None:
        if not self.is_ready(): raise VectorDBConnectionError("Cannot delete collection: Provider is not connected or ready.")
        collection_name = self._config.core.collection_name
        print(f"Attempting to delete collection '{collection_name}'...")
        try:
            self._client.delete_collection(name=collection_name)
            self._collection_instance = None
            print(f"Collection '{collection_name}' deleted successfully.")
        except (ValueError, chromadb.errors.NotFoundError) as e: 
            raise CollectionDoesNotExistError(f"Cannot delete collection '{collection_name}' because it does not exist.") from e
        except Exception as e: 
            raise VectorDBError(f"An unexpected error occurred while deleting collection '{collection_name}': {e}") from e

    def collection_exists(self) -> bool:
        if not self.is_ready(): raise VectorDBConnectionError("Cannot check for collection: Provider is not connected or ready.")
        try:
            collection = self._client.get_collection(name=self._config.core.collection_name)
            if self._collection_instance is None:
                self._collection_instance = collection
            return True
        except NotFoundError: return False
        except Exception as e: raise VectorDBConnectionError(f"Failed to check collection existence due to a server error: {e}") from e

    def _get_active_collection(self) -> ChromaCollection:
        """A helper to ensure the collection instance is available."""
        if self._collection_instance is None:
            raise VectorDBError("Collection is not initialized. Please call 'create_collection' before performing data operations.")
        return self._collection_instance

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]] = None, sparse_vectors: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        """
        Adds or updates records in the ChromaDB collection.

        Args:
            vectors: A list of vector embeddings.
            payloads: A list of corresponding metadata objects.
            ids: A list of unique identifiers for each vector-payload pair.
            chunks: A list of chunks

        Raises:
            UpsertError: If the data ingestion operation fails.
            VectorDBError: If the collection is not initialized.
        """
        collection = self._get_active_collection()
        print(f"Upserting {len(ids)} records into collection '{collection.name}'...")
        try:
            upsert_params = {
                "embeddings": vectors,
                "metadatas": payloads,
                "ids": [str(i) for i in ids]
            }
            if chunks is not None:
                if len(chunks) != len(ids):
                    raise UpsertError(f"Number of documents ({len(chunks)}) must match number of IDs ({len(ids)})")
                upsert_params["documents"] = chunks
            collection.upsert(**upsert_params)
            print(f"Successfully upserted {len(ids)} records.")
        except Exception as e:
            raise UpsertError(f"Failed to upsert data into collection '{collection.name}': {e}") from e

    def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes records from the collection by their unique identifiers.

        Args:
            ids: A list of specific IDs to remove.

        Raises:
            VectorDBError: If the deletion fails or the collection is not initialized.
        """
        collection = self._get_active_collection()
        print(f"Deleting {len(ids)} records from collection '{collection.name}'...")
        try:
            collection.delete(ids=[str(i) for i in ids])
            print(f"Successfully deleted {len(ids)} records.")
        except Exception as e:
            raise VectorDBError(f"Failed to delete records from collection '{collection.name}': {e}") from e

    def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records from the collection by their IDs.

        Args:
            ids: A list of IDs for which to retrieve the full records.

        Returns:
            A list of VectorSearchResult objects containing the fetched data. The
            order of results is guaranteed to match the order of the input IDs.

        Raises:
            VectorDBError: If fetching fails or the collection is not initialized.
        """
        collection = self._get_active_collection()
        print(f"Fetching {len(ids)} records from collection '{collection.name}'...")
        try:
            results = collection.get(
                ids=[str(i) for i in ids],
                include=["metadatas", "embeddings", "documents"]
            )
            
            results_map = {
                results['ids'][i]: {
                    "payload": results['metadatas'][i],
                    "vector": results['embeddings'][i] if results['embeddings'] is not None else None,
                    "text": results['documents'][i] if results['documents'] is not None else None
                }
                for i in range(len(results['ids']))
            }

            final_results = []
            for an_id in ids:
                str_id = str(an_id)
                if str_id in results_map:
                    final_results.append(
                        VectorSearchResult(
                            id=str_id,
                            score=1.0,
                            payload=results_map[str_id]["payload"],
                            vector=results_map[str_id]["vector"],
                            text=results_map[str_id]["text"]
                        )
                    )
            
            print(f"Successfully fetched {len(final_results)} records.")
            return final_results
            
        except Exception as e:
            raise VectorDBError(f"Failed to fetch records from collection '{collection.name}': {e}") from e
        

    def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:

        filter = filter if filter is not None else self._config.search.filter
        final_top_k = top_k if top_k is not None else self._config.search.default_top_k

        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None

        if is_hybrid:
            if not self._config.search.hybrid_search_enabled: raise ConfigurationError("Hybrid search is not enabled for this provider.")
            return self.hybrid_search(query_vector=query_vector, query_text=query_text, top_k=final_top_k, filter=filter, alpha=alpha, fusion_method=fusion_method, similarity_threshold=similarity_threshold, **kwargs)
        elif is_dense:
            if not self._config.search.dense_search_enabled: raise ConfigurationError("Dense search is not enabled for this provider.")
            return self.dense_search(query_vector=query_vector, top_k=final_top_k, filter=filter, similarity_threshold=similarity_threshold, **kwargs)
        elif is_full_text:
            if not self._config.search.full_text_search_enabled: raise ConfigurationError("Full-text search is not enabled for this provider.")
            return self.full_text_search(query_text=query_text, top_k=final_top_k, filter=filter, similarity_threshold=similarity_threshold, **kwargs)
        else:
            raise ValueError("Invalid search arguments: Please provide 'query_vector', 'query_text', or both.")

    def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        if not self._config.search.dense_search_enabled:
            raise ConfigurationError("Dense search is not enabled for this provider.")
        collection = self._get_active_collection()
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.search.default_similarity_threshold or 0.5
        
        try:
            results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=filter, include=["metadatas", "distances", "embeddings", "documents"])
            ids, distances, metadatas = results['ids'][0], results['distances'][0], results['metadatas'][0]
            vectors = results['embeddings'][0] if results['embeddings'] else [None] * len(ids)
            chunks = results['documents'][0] if results['documents'] else [None] * len(ids)
            max_dist = max(distances) if distances else 1.0
            
            filtered_results = []
            for i in range(len(ids)):
                if self._config.core.distance_metric == DistanceMetric.COSINE:
                    score = 1 - distances[i]
                elif self._config.core.distance_metric == DistanceMetric.EUCLIDEAN:
                    score = min(1.0, max(0.0, 1 - distances[i] / max_dist if max_dist > 0 else 1.0))
                elif self._config.core.distance_metric == DistanceMetric.DOT_PRODUCT:
                    score = distances[i]
                else:
                    score = 1 - distances[i]
                
                if score >= final_similarity_threshold:
                    filtered_results.append(VectorSearchResult(id=ids[i], score=score, payload=metadatas[i], vector=vectors[i], text=chunks[i]))
            
            return filtered_results
        except Exception as e: raise SearchError(f"An error occurred during dense search: {e}") from e

    def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        if not self._config.search.full_text_search_enabled:
            raise ConfigurationError("Full-text search is not enabled for this provider.")
        collection = self._get_active_collection()
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.search.default_similarity_threshold or 0.5
        
        where_document_filter = {"$contains": query_text}
        final_where = {"$and": [filter, where_document_filter]} if filter else where_document_filter
        try:
            results = collection.get(where=final_where, limit=top_k, include=["metadatas", "embeddings", "documents"])
            
            filtered_results = []
            for i in range(len(results['ids'])):
                score = 0.5  # Default score for full-text search
                if score >= final_similarity_threshold:
                    filtered_results.append(VectorSearchResult(
                        id=results['ids'][i], 
                        score=score, 
                        payload=results['metadatas'][i], 
                        vector=results['embeddings'][i],
                        text=results['documents'][i] if results['documents'] else None
                    ))
            
            return filtered_results
        except Exception as e: raise SearchError(f"An error occurred during full-text search: {e}") from e

    def _reciprocal_rank_fusion(self, results_lists: List[List[VectorSearchResult]], k: int = 60) -> dict:
        fused_scores = {}
        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = str(doc.id)
                if doc_id not in fused_scores: fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank + 1)
        return fused_scores
    
    def _weighted_fusion(self, dense_results: List[VectorSearchResult], ft_results: List[VectorSearchResult], alpha: float) -> dict:
        fused_scores = {}
        for doc in dense_results:
            fused_scores[str(doc.id)] = doc.score * alpha
        for doc in ft_results:
            doc_id = str(doc.id)
            if doc_id not in fused_scores: fused_scores[doc_id] = 0
            fused_scores[doc_id] += doc.score * (1 - alpha)
        return fused_scores

    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        if not self._config.search.hybrid_search_enabled:
            raise ConfigurationError("Hybrid search is not enabled for this provider.")
        final_alpha = alpha if alpha is not None else (self._config.search.default_hybrid_alpha or 0.5)
        fusion_method = fusion_method if fusion_method is not None else (self._config.search.default_fusion_method or 'weighted')
        
        try:
            candidate_k = max(top_k * 2, 20)
            dense_results = self.dense_search(query_vector, candidate_k, filter, similarity_threshold, **kwargs)
            ft_results = self.full_text_search(query_text, candidate_k, filter, similarity_threshold, **kwargs)
            
            fused_scores: dict
            if fusion_method == 'rrf':
                fused_scores = self._reciprocal_rank_fusion([dense_results, ft_results])
            elif fusion_method == 'weighted':
                fused_scores = self._weighted_fusion(dense_results, ft_results, final_alpha)
            else:
                raise ValueError(f"Unknown fusion_method: {fusion_method}")

            reranked_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)[:top_k]
            if not reranked_ids: return []

            final_results = self.fetch(ids=reranked_ids)
            updated_results = []
            for result in final_results:
                updated_result = VectorSearchResult(
                    id=result.id,
                    score=fused_scores.get(str(result.id), 0.0),
                    payload=result.payload,
                    vector=result.vector,
                    text=result.text
                )
                updated_results.append(updated_result)
            updated_results.sort(key=lambda x: x.score, reverse=True)
            return updated_results
        except Exception as e: raise SearchError(f"An error occurred during hybrid search: {e}") from e