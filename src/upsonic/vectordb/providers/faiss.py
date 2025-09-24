import faiss
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal, Callable

from upsonic.vectordb.config import (
    Config,
    Mode,
    DistanceMetric,
    IndexType,
    HNSWTuningConfig,
    IVFTuningConfig,
    FlatTuningConfig,
    QuantizationConfig
)
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

class FaissProvider(BaseVectorDBProvider):
    """
    An implementation of the BaseVectorDBProvider for the FAISS library.

    This provider behaves as a self-contained, file-based vector database. It manages
    a FAISS index, its associated metadata, and ID mappings directly on the local
    filesystem. 'Connecting' hydrates the state into memory, and 'disconnecting'
    persists the state back to disk.

    **Concurrency Warning:** This implementation is NOT thread-safe or process-safe.
    Concurrent write operations can lead to state corruption. It is designed for
    single-threaded access patterns, such as in local applications or batch processing.
    """

    def __init__(self, config: Config):
        if config.core.provider_name.value != 'faiss': raise ConfigurationError("...")
        super().__init__(config)
        self._index: Optional[faiss.Index] = None
        self._metadata_store: Dict[Union[str, int], Dict[str, Any]] = {}
        self._user_id_to_faiss_id: Dict[Union[str, int], int] = {}
        self._faiss_id_to_user_id: Dict[int, Union[str, int]] = {}
        self._base_db_path: Optional[Path] = Path(self._config.core.db_path) if self._config.core.db_path else None
        self._normalize_vectors = (self._config.core.distance_metric == DistanceMetric.COSINE)

    def connect(self) -> None:
        if self._is_connected: return
        db_path = self._active_db_path
        try:
            if db_path:
                db_path.mkdir(parents=True, exist_ok=True)
                index_file = db_path / "index.faiss"
                metadata_file = db_path / "metadata.json"
                id_map_file = db_path / "id_map.json"
                if index_file.exists(): self._index = faiss.read_index(str(index_file))
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f: self._metadata_store = json.load(f)
                if id_map_file.exists():
                    with open(id_map_file, 'r') as f:
                        maps = json.load(f)
                        self._user_id_to_faiss_id = maps.get("user_to_faiss", {})
                        self._faiss_id_to_user_id = {int(k): v for k, v in maps.get("faiss_to_user", {}).items()}
            self._is_connected = True
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to hydrate FAISS state from disk: {e}")

    def disconnect(self) -> None:
        if not self._is_connected: return
        db_path = self._active_db_path
        if not db_path:
            print("Running in 'in_memory' mode. Clearing state without persisting.")
        else:
            try:
                db_path.mkdir(parents=True, exist_ok=True)
                if self._index: faiss.write_index(self._index, str(db_path / "index.faiss"))
                with open(db_path / "metadata.json", 'w') as f: json.dump(self._metadata_store, f)
                with open(db_path / "id_map.json", 'w') as f:
                    json.dump({"user_to_faiss": self._user_id_to_faiss_id, "faiss_to_user": self._faiss_id_to_user_id}, f)
            except Exception as e:
                print(f"[ERROR] Failed to persist FAISS state to disk: {e}")
        self._index = None; self._metadata_store = {}; self._user_id_to_faiss_id = {}; self._faiss_id_to_user_id = {}
        self._is_connected = False

    def is_ready(self) -> bool:
        return self._is_connected and self._index is not None

    def create_collection(self) -> None:
        """
        Creates the collection by building a FAISS index in memory based on the
        provider's configuration. This method is the designated "index factory".
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Must be connected before creating a collection.")
        if self._index is not None:
            print("Collection (FAISS index) already exists in memory.")
            return

        if self.collection_exists():
            if self._config.core.recreate_if_exists:
                print("Deleting existing collection on disk to recreate.")
                self.delete_collection()
            else:
                print("Collection path exists but index not loaded. Proceeding to create new index.")

        if self._active_db_path:
            self._active_db_path.mkdir(parents=True, exist_ok=True)
        
        d = self._config.core.vector_size
        index_conf = self._config.indexing.index_config
        
        factory_parts = []
        
        if isinstance(index_conf, IVFTuningConfig):
            factory_parts.append(f"IVF{index_conf.nlist}")
        elif isinstance(index_conf, HNSWTuningConfig):
            factory_parts.append(f"HNSW{index_conf.m}")

        quant_conf = self._config.indexing.quantization
        if quant_conf:
            if quant_conf.quantization_type == 'Product':
                m = d // 4 
                factory_parts.append(f"PQ{m}")
            elif quant_conf.quantization_type == 'Scalar':
                factory_parts.append(f"SQ{quant_conf.bits}")
        
        if isinstance(index_conf, IVFTuningConfig):
            factory_parts.append("Flat")

        if factory_parts:
            factory_string = ",".join(factory_parts)
        else:
            factory_string = "Flat"

        metric_map = {
            DistanceMetric.EUCLIDEAN: faiss.METRIC_L2,
            DistanceMetric.DOT_PRODUCT: faiss.METRIC_INNER_PRODUCT,
            DistanceMetric.COSINE: faiss.METRIC_INNER_PRODUCT
        }
        metric_type = metric_map[self._config.core.distance_metric]

        try:
            print(f"Creating FAISS index with factory string: '{factory_string}' and dimension: {d}")
            self._index = faiss.index_factory(d, factory_string, metric_type)
            print("FAISS index created successfully.")
        except Exception as e:
            raise VectorDBError(f"Failed to create FAISS index with factory string '{factory_string}': {e}")


    def delete_collection(self) -> None:
        """
        Permanently deletes the collection from the filesystem.
        """
        if not self._active_db_path:
            print("Cannot delete collection in 'in_memory' mode.")
            return

        if self.collection_exists():
            try:
                shutil.rmtree(self._active_db_path)
                print(f"Successfully deleted collection directory: '{self._active_db_path}'")
            except OSError as e:
                raise VectorDBError(f"Error deleting collection directory '{self._active_db_path}': {e}")
        else:
            print("Collection directory does not exist. No action taken.")

        self._index = None
        self._metadata_store.clear()
        self._user_id_to_faiss_id.clear()
        self._faiss_id_to_user_id.clear()

    def collection_exists(self) -> bool:
        """
        Checks if the collection (directory) exists on the filesystem.
        """
        if not self._active_db_path:
            return self._index is not None
        
        return self._active_db_path.is_dir() and any(self._active_db_path.iterdir())

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]] = None, sparse_vectors: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        """
        Adds new data or updates existing data by performing a delete-then-add operation.
        This method synchronizes the FAISS index, metadata store, and ID maps.

        Raises:
            UpsertError: If the data ingestion fails.
            VectorDBError: If the index is not ready.
        """
        if not self.is_ready():
            raise VectorDBError("FAISS index is not created. Please call 'create_collection' first.")
        if not (len(vectors) == len(payloads) == len(ids)):
            raise UpsertError("The lengths of vectors, payloads, and ids lists must be identical.")
        if not vectors:
            return

        # Handle chunks parameter - add to payloads if provided
        if chunks is not None:
            if len(chunks) != len(payloads):
                raise UpsertError("The lengths of chunks and payloads lists must be identical.")
            for i in range(len(payloads)):
                payloads[i]["chunk"] = chunks[i]

        vectors_np = np.array(vectors, dtype=np.float32)
        if self._normalize_vectors:
            faiss.normalize_L2(vectors_np)

        ids_to_update = [user_id for user_id in ids if user_id in self._user_id_to_faiss_id]
        if ids_to_update:
            print(f"Updating {len(ids_to_update)} existing IDs by deleting old entries first.")
            self.delete(ids_to_update)

        if not self._index.is_trained:
            print(f"FAISS index is not trained. Training on {len(vectors_np)} vectors...")
            self._index.train(vectors_np)
            print("Training complete.")

        try:
            start_faiss_id = self._index.ntotal
            self._index.add(vectors_np)
            
            for i, user_id in enumerate(ids):
                faiss_id = start_faiss_id + i
                self._user_id_to_faiss_id[user_id] = faiss_id
                self._faiss_id_to_user_id[faiss_id] = user_id
                self._metadata_store[user_id] = payloads[i]
            
            print(f"Successfully upserted {len(vectors)} vectors. Index total: {self._index.ntotal}")

        except Exception as e:
            raise UpsertError(f"An error occurred during FAISS add operation: {e}")


    def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Deletes data using a "tombstone" strategy. It removes entries from the
        metadata and ID maps, but the vector remains in the FAISS index until the
        next full rebuild. Search results will ignore these "ghost" vectors.
        """
        if not self.is_ready():
            return

        deleted_count = 0
        for user_id in ids:
            if user_id in self._user_id_to_faiss_id:
                faiss_id = self._user_id_to_faiss_id[user_id]
                
                del self._user_id_to_faiss_id[user_id]
                del self._faiss_id_to_user_id[faiss_id]
                if user_id in self._metadata_store:
                    del self._metadata_store[user_id]
                
                deleted_count += 1
        
        if deleted_count > 0:
            print(f"Successfully marked {deleted_count} IDs for deletion (tombstoned).")

    
    def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their user-provided IDs.

        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        if not self.is_ready():
            return []
        
        results = []
        for user_id in ids:
            if user_id in self._user_id_to_faiss_id:
                try:
                    faiss_id = self._user_id_to_faiss_id[user_id]
                    payload = self._metadata_store.get(user_id)
                    
                    vector = self._index.reconstruct(faiss_id).tolist()
                    
                    text = payload.get("chunk", "") if payload else ""
                    results.append(VectorSearchResult(
                        id=user_id,
                        score=1.0,
                        payload=payload,
                        vector=vector,
                        text=text
                    ))
                except Exception as e:
                    print(f"Could not fetch data for ID '{user_id}': {e}")
        return results

    def _translate_filter(self, filter_dict: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
        """
        Recursively translates a filter dictionary into a Python function that
        returns True for a matching payload, False otherwise.
        """
        def build_checker(key, value):
            if isinstance(value, dict):
                op, val = list(value.items())[0]
                if op == "$in": return lambda payload: payload.get(key) in val
                if op == "$gte": return lambda payload: payload.get(key, float('-inf')) >= val
                if op == "$lte": return lambda payload: payload.get(key, float('inf')) <= val
                if op == "$gt": return lambda payload: payload.get(key, float('-inf')) > val
                if op == "$lt": return lambda payload: payload.get(key, float('inf')) < val
                if op == "$ne": return lambda payload: payload.get(key) != val
            return lambda payload: payload.get(key) == value

        if "and" in filter_dict:
            checkers = [self._translate_filter(sub_filter) for sub_filter in filter_dict["and"]]
            return lambda payload: all(checker(payload) for checker in checkers)
        if "or" in filter_dict:
            checkers = [self._translate_filter(sub_filter) for sub_filter in filter_dict["or"]]
            return lambda payload: any(checker(payload) for checker in checkers)
        
        checkers = [build_checker(k, v) for k, v in filter_dict.items()]
        return lambda payload: all(checker(payload) for checker in checkers)

    def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        final_top_k = top_k if top_k is not None else self._config.search.default_top_k or 10
        has_vector = query_vector is not None and len(query_vector) > 0
        has_text = query_text is not None and query_text.strip()

        if has_vector and has_text:
            return self.hybrid_search(query_vector, query_text, final_top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs)
        elif has_vector:
            return self.dense_search(query_vector, final_top_k, filter, similarity_threshold, **kwargs)
        elif has_text:
            return self.full_text_search(query_text, final_top_k, filter, similarity_threshold, **kwargs)
        else:
            raise ConfigurationError("Search requires at least one of 'query_vector' or 'query_text'.")

    def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        if not self.is_ready(): raise SearchError("FAISS index is not ready for search.")
        if self._index.ntotal == 0: return []

        filter_func = self._translate_filter(filter) if filter else None
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.search.default_similarity_threshold or 0.5
        
        candidate_multiplier = 10
        candidate_k = top_k * candidate_multiplier if filter_func else top_k
        candidate_k = min(candidate_k, self._index.ntotal)

        query_np = np.array([query_vector], dtype=np.float32)
        if self._normalize_vectors:
            faiss.normalize_L2(query_np)

        try:
            distances, faiss_ids = self._index.search(query_np, candidate_k)
        except Exception as e:
            raise SearchError(f"An error occurred during FAISS search: {e}")

        results: List[VectorSearchResult] = []
        for dist, faiss_id in zip(distances[0], faiss_ids[0]):
            if len(results) >= top_k: break
            if faiss_id == -1: continue

            user_id = self._faiss_id_to_user_id.get(faiss_id)
            if not user_id: continue

            payload = self._metadata_store.get(user_id)
            if filter_func and not filter_func(payload or {}):
                continue

            if self._config.core.distance_metric == DistanceMetric.EUCLIDEAN:
                score = 1 / (1 + dist)
            elif self._config.core.distance_metric == DistanceMetric.COSINE:
                # For cosine with L2-normalized vectors, FAISS returns inner product 
                # which equals cosine similarity. Clamp to [0, 1] range.
                score = max(0.0, min(1.0, float(dist)))
            elif self._config.core.distance_metric == DistanceMetric.DOT_PRODUCT:
                score = float(dist)  # For dot product, distance is already similarity-like
            else:
                # Default to cosine-like conversion for unknown metrics
                score = max(0.0, min(1.0, float(dist)))

            if score >= final_similarity_threshold:
                text = payload.get("chunk", "") if payload else ""
                results.append(VectorSearchResult(id=user_id, score=score, payload=payload, text=text))
        
        return results

    def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        raise NotImplementedError("FAISS is a dense-vector-only library and does not support full-text search.")

    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        print("[WARNING] FAISS provider received a hybrid search request. It will ignore the text query and alpha, performing a dense search instead.")
        return self.dense_search(query_vector=query_vector, top_k=top_k, filter=filter, similarity_threshold=similarity_threshold, **kwargs)
    

    @property
    def _active_db_path(self) -> Optional[Path]:
        """Private helper to get the active db path, applying tenancy if configured."""
        if not self._base_db_path:
            return None
        if self._config.advanced.namespace:
            return self._base_db_path / self._config.advanced.namespace
        return self._base_db_path