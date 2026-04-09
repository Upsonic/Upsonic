from __future__ import annotations

import hashlib
import json
import shutil
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal, Callable, Set, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    import faiss
    import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

from upsonic.vectordb.config import (
    FaissConfig,
    DistanceMetric,
    HNSWIndexConfig,
    IVFIndexConfig,
)
from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.utils.printing import info_log, debug_log, warning_log

from upsonic.utils.package.exception import(
    VectorDBConnectionError, 
    ConfigurationError, 
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

    Standard properties stored per chunk:
    - chunk_id (str) — unique per-chunk identifier (UUID)
    - chunk_content_hash (str) — MD5 of chunk text content for deduplication
    - content (str) — the chunk text
    - metadata (dict) — all non-standard data
    - document_name (str) — human-readable source name (optional)
    - document_id (str) — parent document identifier (optional)
    - doc_content_hash (str) — MD5 of parent document content for change detection (optional)

    **Concurrency Warning:** This implementation is NOT thread-safe or process-safe.
    Concurrent write operations can lead to state corruption. It is designed for
    single-threaded access patterns, such as in local applications or batch processing.
    """

    _STANDARD_FIELDS: frozenset = frozenset({
        'chunk_id', 'document_id', 'doc_content_hash',
        'chunk_content_hash', 'document_name', 'content', 'metadata',
        'knowledge_base_id',
    })

    def __init__(self, config: Union[FaissConfig, Dict[str, Any]]):
        if not _FAISS_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="faiss-cpu",
                install_command='pip install "upsonic[faiss]"',
                feature_name="FAISS vector database provider"
            )

        if not _NUMPY_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="numpy",
                install_command='pip install "upsonic[faiss]"',
                feature_name="FAISS vector database provider"
            )

        if isinstance(config, dict):
            config = FaissConfig.from_dict(config)

        super().__init__(config)
        
        self._index: Optional[Any] = None
        self._metadata_store: Dict[str, Dict[str, Any]] = {}  # chunk_id -> full payload
        self._chunk_id_to_faiss_id: Dict[str, int] = {}  # chunk_id -> FAISS internal ID
        self._faiss_id_to_chunk_id: Dict[int, str] = {}  # FAISS internal ID -> chunk_id
        
        # field_name -> field_value -> Set[chunk_id]
        self._field_indexes: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        self._base_db_path: Optional[Path] = Path(self._config.db_path) if self._config.db_path else None
        self._normalize_vectors: bool = self._config.normalize_vectors

    # ============================================================================
    # Connection Management
    # ============================================================================

    async def aconnect(self) -> None:
        """Establishes a connection to the vector database (async)."""
        if self._is_connected:
            return
        
        db_path = self._active_db_path
        try:
            if db_path:
                db_path.mkdir(parents=True, exist_ok=True)
                index_file = db_path / "index.faiss"
                metadata_file = db_path / "metadata.json"
                id_map_file = db_path / "id_map.json"
                indexes_file = db_path / "field_indexes.json"
                
                if index_file.exists():
                    self._index = faiss.read_index(str(index_file))
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self._metadata_store = json.load(f)
                
                if id_map_file.exists():
                    with open(id_map_file, 'r') as f:
                        maps = json.load(f)
                        self._chunk_id_to_faiss_id = maps.get("chunk_id_to_faiss", {})
                        self._faiss_id_to_chunk_id = {int(k): v for k, v in maps.get("faiss_to_chunk_id", {}).items()}
                
                if indexes_file.exists():
                    with open(indexes_file, 'r') as f:
                        indexes_data = json.load(f)
                        for field_name, value_map in indexes_data.items():
                            self._field_indexes[field_name] = {
                                k: set(v) for k, v in value_map.items()
                            }
            
            self._is_connected = True
            info_log(f"Connected to FAISS collection '{self._config.collection_name}'", context="FaissVectorDB")
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to hydrate FAISS state from disk: {e}")

    async def adisconnect(self) -> None:
        """Gracefully terminates the connection to the vector database (async)."""
        if not self._is_connected:
            return
        
        db_path = self._active_db_path
        if not db_path:
            debug_log("Running in 'in_memory' mode. Clearing state without persisting.", context="FaissVectorDB")
        else:
            try:
                db_path.mkdir(parents=True, exist_ok=True)
                if self._index:
                    faiss.write_index(self._index, str(db_path / "index.faiss"))
                
                with open(db_path / "metadata.json", 'w') as f:
                    json.dump(self._metadata_store, f)
                
                with open(db_path / "id_map.json", 'w') as f:
                    json.dump({
                        "chunk_id_to_faiss": self._chunk_id_to_faiss_id,
                        "faiss_to_chunk_id": self._faiss_id_to_chunk_id
                    }, f)
                
                indexes_data = {
                    field_name: {
                        str(k): list(v) for k, v in value_map.items()
                    }
                    for field_name, value_map in self._field_indexes.items()
                }
                with open(db_path / "field_indexes.json", 'w') as f:
                    json.dump(indexes_data, f)
                
                info_log("FAISS state persisted to disk.", context="FaissVectorDB")
            except Exception as e:
                warning_log(f"Failed to persist FAISS state to disk: {e}", context="FaissVectorDB")
        
        self._index = None
        self._metadata_store.clear()
        self._chunk_id_to_faiss_id.clear()
        self._faiss_id_to_chunk_id.clear()
        self._field_indexes.clear()
        self._is_connected = False

    async def ais_ready(self) -> bool:
        """Performs a health check to ensure the database is responsive (async)."""
        return self._is_connected and self._index is not None

    # ============================================================================
    # Collection Management
    # ============================================================================

    async def acreate_collection(self) -> None:
        """
        Creates the collection by building a FAISS index in memory based on the
        provider's configuration.
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Must be connected before creating a collection.")
        
        if self._index is not None:
            if self._config.recreate_if_exists:
                info_log("Collection exists in memory. Recreating due to recreate_if_exists=True.", context="FaissVectorDB")
                await self.adelete_collection()
            else:
                info_log("Collection (FAISS index) already exists in memory.", context="FaissVectorDB")
                return

        if await self.acollection_exists():
            if self._config.recreate_if_exists:
                info_log("Deleting existing collection on disk to recreate.", context="FaissVectorDB")
                await self.adelete_collection()
            else:
                info_log("Collection path exists but index not loaded. Proceeding to create new index.", context="FaissVectorDB")

        if self._active_db_path:
            self._active_db_path.mkdir(parents=True, exist_ok=True)
        
        d: int = self._config.vector_size
        index_conf = self._config.index
        
        factory_parts: List[str] = []
        
        if isinstance(index_conf, IVFIndexConfig):
            factory_parts.append(f"IVF{index_conf.nlist}")
        elif isinstance(index_conf, HNSWIndexConfig):
            factory_parts.append(f"HNSW{index_conf.m}")

        if self._config.quantization_type:
            if self._config.quantization_type == 'product':
                m = d // 4 
                factory_parts.append(f"PQ{m}")
            elif self._config.quantization_type == 'scalar':
                factory_parts.append(f"SQ{self._config.quantization_bits}")
        
        if isinstance(index_conf, IVFIndexConfig):
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
        metric_type = metric_map[self._config.distance_metric]

        try:
            debug_log(f"Creating FAISS index with factory string: '{factory_string}' and dimension: {d}", context="FaissVectorDB")
            self._index = faiss.index_factory(d, factory_string, metric_type)
            info_log("FAISS index created successfully.", context="FaissVectorDB")
        except Exception as e:
            raise VectorDBError(f"Failed to create FAISS index with factory string '{factory_string}': {e}")

    async def adelete_collection(self) -> None:
        """Permanently deletes the collection specified in config (async)."""
        if not self._active_db_path:
            debug_log("Cannot delete collection in 'in_memory' mode.", context="FaissVectorDB")
            self._index = None
            self._metadata_store.clear()
            self._chunk_id_to_faiss_id.clear()
            self._faiss_id_to_chunk_id.clear()
            self._field_indexes.clear()
            return

        if await self.acollection_exists():
            try:
                shutil.rmtree(self._active_db_path)
                info_log(f"Successfully deleted collection directory: '{self._active_db_path}'", context="FaissVectorDB")
            except OSError as e:
                raise VectorDBError(f"Error deleting collection directory '{self._active_db_path}': {e}")
        else:
            debug_log("Collection directory does not exist. No action taken.", context="FaissVectorDB")

        self._index = None
        self._metadata_store.clear()
        self._chunk_id_to_faiss_id.clear()
        self._faiss_id_to_chunk_id.clear()
        self._field_indexes.clear()

    async def acollection_exists(self) -> bool:
        """Checks if the collection specified in the config already exists (async)."""
        if not self._active_db_path:
            return self._index is not None
        
        return self._active_db_path.is_dir() and any(self._active_db_path.iterdir())

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _generate_provider_id(self) -> str:
        """Generates a unique provider ID based on db_path and collection."""
        identifier_parts = [
            self._config.db_path or "in_memory",
            self._config.collection_name
        ]
        identifier = "#".join(filter(None, identifier_parts))
        return hashlib.md5(identifier.encode()).hexdigest()[:16]

    def _build_payload(
        self,
        content: str,
        chunk_id: str,
        chunk_content_hash: str = "",
        document_id: str = "",
        doc_content_hash: str = "",
        document_name: str = "",
        knowledge_base_id: str = "",
        extra_payload: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        _warned_standard_keys: Optional[Set[str]] = None,
        apply_defaults: bool = True,
    ) -> Dict[str, Any]:
        """
        Builds a standardized payload with the seven standard properties.

        Primary (chunk-focused) identifiers are always set. Document-level
        identifiers are optional and default to empty strings.

        Args:
            content: The chunk text (required).
            chunk_id: The unique chunk identifier.
            chunk_content_hash: MD5 hash of the chunk's text content.
            document_id: The parent document identifier.
            doc_content_hash: MD5 hash of the parent document content.
            document_name: Human-readable source name.
            extra_payload: Caller-provided payload dict (non-standard keys -> metadata).
            extra_metadata: Additional metadata dict.

        Returns:
            A properly structured payload dict.
        """
        payload: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "chunk_content_hash": chunk_content_hash,
            "document_id": document_id,
            "doc_content_hash": doc_content_hash,
            "document_name": document_name,
            "content": content,
            "knowledge_base_id": knowledge_base_id,
        }

        combined_metadata: Dict[str, Any] = {}

        if apply_defaults and self._config.default_metadata:
            combined_metadata.update(self._config.default_metadata)

        if extra_payload:
            if 'metadata' in extra_payload and isinstance(extra_payload['metadata'], dict):
                combined_metadata.update(extra_payload['metadata'])
            for key, value in extra_payload.items():
                if key in self._STANDARD_FIELDS:
                    if key == 'metadata':
                        continue
                    if _warned_standard_keys is not None and key not in _warned_standard_keys:
                        debug_log(
                            f"Standard field '{key}' found in payload dict and will be ignored. "
                            f"Standard fields must be passed via dedicated parameters "
                            f"(ids, document_ids, document_names, doc_content_hashes, "
                            f"chunk_content_hashes, knowledge_base_ids).",
                            context="FaissVectorDB",
                        )
                        _warned_standard_keys.add(key)
                else:
                    combined_metadata[key] = value

        if extra_metadata:
            combined_metadata.update(extra_metadata)

        payload["metadata"] = combined_metadata

        return payload

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
        
        result: Dict[str, Dict[str, Any]] = {}
        for item in self._config.indexed_fields:
            if isinstance(item, str):
                result[item] = {"indexed": True, "type": "keyword"}
            elif isinstance(item, dict):
                field_name = item.get("field")
                if field_name:
                    result[field_name] = {
                        "indexed": True,
                        "type": item.get("type", "keyword")
                    }
        
        return result
    
    def _get_field_names_from_config(self) -> List[str]:
        """Extract field names from indexed_fields configuration."""
        indexed_fields_config = self._parse_indexed_fields()
        return list(indexed_fields_config.keys())
    
    def _update_field_indexes(self, chunk_id: str, payload: Dict[str, Any], operation: str = 'add') -> None:
        """
        Updates field indexes for fast lookups.

        Args:
            chunk_id: The chunk ID to index.
            payload: The payload containing field values.
            operation: 'add' or 'remove'.
        """
        if not self._config.indexed_fields:
            return
        
        field_names = self._get_field_names_from_config()
        
        for field_name in field_names:
            if '.' in field_name:
                parts = field_name.split('.')
                value: Any = payload
                try:
                    for part in parts:
                        value = value[part]
                except (KeyError, TypeError):
                    continue
            else:
                value = payload.get(field_name)
            
            if value is None:
                continue
            
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            
            if operation == 'add':
                self._field_indexes[field_name][value].add(chunk_id)
            elif operation == 'remove':
                if value in self._field_indexes[field_name]:
                    self._field_indexes[field_name][value].discard(chunk_id)
                    if not self._field_indexes[field_name][value]:
                        del self._field_indexes[field_name][value]

    def _build_filter_function(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Callable[[Dict[str, Any]], bool]]:
        """
        Builds a filter function from a filter dictionary.

        Supports:
        - Direct field matching: {"field": "value"}
        - Operators: {"field": {"$in": [...]}, {"$gte": value}, etc.}
        - Logical operators: {"and": [...]}, {"or": [...]}
        - Nested metadata: {"metadata.key": "value"}
        """
        if not filter_dict:
            return None

        def build_checker(key: str, value: Any) -> Callable[[Dict[str, Any]], bool]:
            if key not in self._STANDARD_FIELDS and '.' not in key:
                key = f"metadata.{key}"
            if isinstance(value, dict):
                if "$in" in value:
                    op_value = value["$in"]
                    return lambda payload: self._get_nested_value(payload, key) in op_value
                elif "$gte" in value:
                    op_value = value["$gte"]
                    return lambda payload: self._get_nested_value(payload, key, float('-inf')) >= op_value
                elif "$lte" in value:
                    op_value = value["$lte"]
                    return lambda payload: self._get_nested_value(payload, key, float('inf')) <= op_value
                elif "$gt" in value:
                    op_value = value["$gt"]
                    return lambda payload: self._get_nested_value(payload, key, float('-inf')) > op_value
                elif "$lt" in value:
                    op_value = value["$lt"]
                    return lambda payload: self._get_nested_value(payload, key, float('inf')) < op_value
                elif "$ne" in value:
                    op_value = value["$ne"]
                    return lambda payload: self._get_nested_value(payload, key) != op_value
                else:
                    return lambda payload: self._get_nested_value(payload, key) == value
            
            return lambda payload: self._get_nested_value(payload, key) == value

        if "and" in filter_dict:
            checkers = [self._build_filter_function(sub_filter) for sub_filter in filter_dict["and"]]
            checkers = [c for c in checkers if c is not None]
            if not checkers:
                return None
            return lambda payload: all(checker(payload) for checker in checkers)
        
        if "or" in filter_dict:
            checkers = [self._build_filter_function(sub_filter) for sub_filter in filter_dict["or"]]
            checkers = [c for c in checkers if c is not None]
            if not checkers:
                return None
            return lambda payload: any(checker(payload) for checker in checkers)
        
        checkers = [build_checker(k, v) for k, v in filter_dict.items()]
        if not checkers:
            return None
        
        return lambda payload: all(checker(payload) for checker in checkers)

    def _get_nested_value(self, payload: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get a nested value from payload, supporting dot notation."""
        if '.' not in key:
            return payload.get(key, default)
        
        parts = key.split('.')
        value: Any = payload
        try:
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                    if value is None:
                        return default
                else:
                    return default
            return value
        except (KeyError, TypeError, AttributeError):
            return default

    @property
    def _active_db_path(self) -> Optional[Path]:
        """Private helper to get the active db path."""
        if not self._base_db_path:
            return None
        return self._base_db_path

    def _batch_find_existing_by_hashes(
        self,
        chunk_hashes: List[str],
    ) -> Dict[str, List[str]]:
        """
        Batch lookup: finds which chunk_content_hash values already exist
        and returns the chunk_ids that own them.

        Args:
            chunk_hashes: The chunk_content_hash values to check.

        Returns:
            Mapping of chunk_content_hash -> list of existing chunk_ids.
        """
        if not chunk_hashes:
            return {}

        unique_hashes: Set[str] = set(chunk_hashes)
        existing: Dict[str, List[str]] = defaultdict(list)

        for stored_chunk_id, stored_payload in self._metadata_store.items():
            stored_hash: str = stored_payload.get("chunk_content_hash", "")
            if stored_hash in unique_hashes:
                existing[stored_hash].append(stored_chunk_id)

        return dict(existing)

    # ============================================================================
    # Data Operations
    # ============================================================================

    async def aupsert(
        self,
        vectors: Optional[List[List[float]]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        chunks: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        document_names: Optional[List[str]] = None,
        doc_content_hashes: Optional[List[str]] = None,
        chunk_content_hashes: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        knowledge_base_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Adds or updates data in the FAISS collection.

        Uses chunk_content_hash for content-based deduplication via a batch
        pre-check. If identical chunk content already exists under a different
        chunk_id, the old entry is deleted first to avoid duplicates.
        """
        if not await self.ais_ready():
            warning_log("FAISS index is not created. Please call 'create_collection' first.", context="FaissVectorDB")
            return

        if (vectors is None or len(vectors) == 0) and (sparse_vectors is None or len(sparse_vectors) == 0):
            info_log("Nothing to upsert: no dense or sparse vectors provided.", context="FaissVectorDB")
            return

        n: int = len(vectors) if vectors else len(sparse_vectors)  # type: ignore[arg-type]

        for _arr_name, _arr in (
            ("vectors", vectors),
            ("payloads", payloads),
            ("ids", ids),
            ("chunks", chunks),
            ("document_ids", document_ids),
            ("document_names", document_names),
            ("doc_content_hashes", doc_content_hashes),
            ("chunk_content_hashes", chunk_content_hashes),
            ("sparse_vectors", sparse_vectors),
            ("knowledge_base_ids", knowledge_base_ids),
        ):
            if _arr is not None and len(_arr) != n:
                raise UpsertError(
                    f"Length mismatch for {_arr_name}: expected {n}, got {len(_arr)}"
                )

        extra_metadata: Dict[str, Any] = metadata or {}
        _warned_standard_keys: Set[str] = set()

        contents: List[str] = []
        for i in range(n):
            payload: Dict[str, Any] = payloads[i] if payloads and i < len(payloads) else {}
            content: str = chunks[i] if chunks and i < len(chunks) else ""

            contents.append(content)

        record_ids: List[str] = []
        for i in range(n):
            record_ids.append(str(ids[i]) if ids and i < len(ids) else str(_uuid.uuid4()))

        computed_hashes: List[str] = []
        for i in range(n):
            if chunk_content_hashes and i < len(chunk_content_hashes) and chunk_content_hashes[i]:
                computed_hashes.append(chunk_content_hashes[i])
            else:
                computed_hashes.append(hashlib.md5(contents[i].encode("utf-8")).hexdigest())

        existing_hash_map: Dict[str, List[str]] = self._batch_find_existing_by_hashes(computed_hashes)

        stale_chunk_ids: List[str] = []
        chunk_ids_to_replace: List[str] = []

        for i in range(n):
            chunk_id_str: str = record_ids[i]
            chunk_hash: str = computed_hashes[i]

            old_ids: List[str] = existing_hash_map.get(chunk_hash, [])
            for old_id in old_ids:
                if old_id != chunk_id_str:
                    stale_chunk_ids.append(old_id)

            if chunk_id_str in self._chunk_id_to_faiss_id:
                chunk_ids_to_replace.append(chunk_id_str)

        ids_to_delete: List[str] = list(set(stale_chunk_ids + chunk_ids_to_replace))
        if ids_to_delete:
            debug_log(
                f"Deleting {len(ids_to_delete)} entries (stale duplicates + replacements) before upsert.",
                context="FaissVectorDB",
            )
            await self.adelete(ids_to_delete)

        standardized_payloads: List[Dict[str, Any]] = []
        for i in range(n):
            payload: Dict[str, Any] = payloads[i] if payloads and i < len(payloads) else {}
            content: str = contents[i]
            chunk_id_str = record_ids[i]
            doc_id: str = document_ids[i] if document_ids and i < len(document_ids) else ""
            doc_hash: str = doc_content_hashes[i] if doc_content_hashes and i < len(doc_content_hashes) else ""
            chunk_hash = computed_hashes[i]
            doc_name: str = document_names[i] if document_names else ""
            kb_id: str = knowledge_base_ids[i] if knowledge_base_ids else ""

            final_payload = self._build_payload(
                content=content,
                chunk_id=chunk_id_str,
                chunk_content_hash=chunk_hash,
                document_id=doc_id,
                doc_content_hash=doc_hash,
                document_name=doc_name,
                knowledge_base_id=kb_id,
                extra_payload=payload,
                extra_metadata=extra_metadata,
                _warned_standard_keys=_warned_standard_keys,
            )
            standardized_payloads.append(final_payload)

        vectors_np = np.array(vectors, dtype=np.float32)
        if self._normalize_vectors:
            faiss.normalize_L2(vectors_np)

        if not self._index.is_trained:
            debug_log(f"FAISS index is not trained. Training on {len(vectors_np)} vectors...", context="FaissVectorDB")
            self._index.train(vectors_np)
            info_log("Training complete.", context="FaissVectorDB")

        try:
            start_faiss_id: int = self._index.ntotal
            self._index.add(vectors_np)
            
            for i, final_payload in enumerate(standardized_payloads):
                faiss_id: int = start_faiss_id + i
                chunk_id_str = record_ids[i]
                self._chunk_id_to_faiss_id[chunk_id_str] = faiss_id
                self._faiss_id_to_chunk_id[faiss_id] = chunk_id_str
                self._metadata_store[chunk_id_str] = final_payload
                self._update_field_indexes(chunk_id_str, final_payload, operation='add')
            
            replaced_count: int = sum(1 for h in computed_hashes if h in existing_hash_map)
            new_count: int = len(standardized_payloads) - replaced_count
            info_log(
                f"Successfully upserted {n} vectors "
                f"({new_count} new, {replaced_count} replaced by content hash). "
                f"Index total: {self._index.ntotal}",
                context="FaissVectorDB",
            )

        except Exception as e:
            raise UpsertError(f"An error occurred during FAISS add operation: {e}")

    async def adelete(self, ids: List[Union[str, int]]) -> None:
        """
        Removes data from the collection by their chunk_ids (async).

        Pre-checks existence for each ID before attempting deletion.
        Non-existent IDs are skipped with a debug log, making this
        method idempotent and retry-safe.

        Args:
            ids: A list of chunk_ids to remove.
        """
        if not await self.ais_ready():
            return

        if not ids:
            debug_log("Delete called with an empty list of IDs. No action taken.", context="FaissVectorDB")
            return

        deleted_count: int = 0
        for chunk_id in ids:
            chunk_id_str: str = str(chunk_id)
            if chunk_id_str in self._chunk_id_to_faiss_id:
                faiss_id: int = self._chunk_id_to_faiss_id[chunk_id_str]

                if chunk_id_str in self._metadata_store:
                    self._update_field_indexes(chunk_id_str, self._metadata_store[chunk_id_str], operation='remove')

                del self._chunk_id_to_faiss_id[chunk_id_str]
                del self._faiss_id_to_chunk_id[faiss_id]
                if chunk_id_str in self._metadata_store:
                    del self._metadata_store[chunk_id_str]

                deleted_count += 1
            else:
                debug_log(
                    f"Chunk with ID '{chunk_id_str}' does not exist, skipping deletion.",
                    context="FaissVectorDB",
                )

        if deleted_count > 0:
            info_log(
                f"Successfully processed deletion. "
                f"Existed: {deleted_count}, Deleted: {deleted_count}.",
                context="FaissVectorDB",
            )
        else:
            info_log("No matching entries found to delete. No action taken.", context="FaissVectorDB")

    async def afetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their chunk_ids (async).

        Args:
            ids: A list of chunk_ids to retrieve.

        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        if not await self.ais_ready():
            return []
        
        results: List[VectorSearchResult] = []
        for chunk_id in ids:
            chunk_id_str: str = str(chunk_id)
            if chunk_id_str in self._chunk_id_to_faiss_id:
                try:
                    faiss_id: int = self._chunk_id_to_faiss_id[chunk_id_str]
                    payload: Optional[Dict[str, Any]] = self._metadata_store.get(chunk_id_str)
                    
                    vector: List[float] = self._index.reconstruct(faiss_id).tolist()
                    
                    text: str = payload.get("content", "") if payload else ""
                    results.append(VectorSearchResult(
                        id=chunk_id_str,
                        score=1.0,
                        payload=payload,
                        vector=vector,
                        text=text
                    ))
                except Exception as e:
                    debug_log(f"Could not fetch data for chunk_id '{chunk_id_str}': {e}", context="FaissVectorDB")
        return results

    # ============================================================================
    # Search Operations
    # ============================================================================

    async def asearch(
        self,
        top_k: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """
        A master search method that dispatches to the appropriate specialized
        search function based on the provided arguments (async).
        """
        final_top_k: int = top_k if top_k is not None else self._config.default_top_k or 10
        has_vector: bool = query_vector is not None and len(query_vector) > 0
        has_text: bool = query_text is not None and bool(query_text.strip())

        final_alpha: float = alpha if alpha is not None else self._config.default_hybrid_alpha
        final_fusion_method: str = fusion_method if fusion_method is not None else self._config.default_fusion_method

        if has_vector and has_text:
            if not self._config.hybrid_search_enabled:
                raise ConfigurationError("Hybrid search is disabled in the configuration.")
            return await self.ahybrid_search(query_vector, query_text, final_top_k, filter, final_alpha, final_fusion_method, similarity_threshold, apply_reranking=apply_reranking, sparse_query_vector=sparse_query_vector)
        elif has_vector:
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled in the configuration.")
            return await self.adense_search(query_vector, final_top_k, filter, similarity_threshold, apply_reranking=apply_reranking)
        elif has_text:
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled in the configuration.")
            return await self.afull_text_search(query_text, final_top_k, filter, similarity_threshold, apply_reranking=apply_reranking, sparse_query_vector=sparse_query_vector)
        else:
            raise ConfigurationError("Search requires at least one of 'query_vector' or 'query_text'.")

    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        """Performs a pure vector similarity search (async)."""
        _ = apply_reranking
        if not await self.ais_ready():
            raise SearchError("FAISS index is not ready for search.")
        if self._index.ntotal == 0:
            return []

        filter_func = self._build_filter_function(filter)
        
        final_similarity_threshold: Optional[float] = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        
        candidate_chunk_ids: Optional[Set[str]] = None
        if filter_func and self._config.indexed_fields:
            candidate_chunk_ids = self._get_candidate_ids_from_filter(filter)
        
        if candidate_chunk_ids is not None and len(candidate_chunk_ids) == 0:
            return []
        
        candidate_multiplier: int = 10
        candidate_k: int = top_k * candidate_multiplier if filter_func else top_k
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
            if len(results) >= top_k:
                break
            if faiss_id == -1:
                continue

            chunk_id: Optional[str] = self._faiss_id_to_chunk_id.get(faiss_id)
            if not chunk_id:
                continue

            if candidate_chunk_ids is not None and chunk_id not in candidate_chunk_ids:
                continue

            payload: Optional[Dict[str, Any]] = self._metadata_store.get(chunk_id)
            if filter_func and not filter_func(payload or {}):
                continue

            if self._config.distance_metric == DistanceMetric.EUCLIDEAN:
                score: float = 1 / (1 + dist)
            elif self._config.distance_metric == DistanceMetric.COSINE:
                score = max(0.0, min(1.0, float(dist)))
            elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
                dist_float: float = float(dist)
                if dist_float <= -1.0:
                    score = 0.0
                elif dist_float >= 1.0:
                    score = 1.0
                else:
                    score = (dist_float + 1.0) / 2.0
            else:
                score = max(0.0, min(1.0, float(dist)))

            if final_similarity_threshold is None or score >= final_similarity_threshold:
                text: str = payload.get("content", "") if payload else ""
                try:
                    vector: Optional[List[float]] = self._index.reconstruct(int(faiss_id)).tolist()
                except Exception as e:
                    debug_log(f"Failed to reconstruct vector for faiss_id {faiss_id}: {e}", context="FaissVectorDB")
                    vector = None
                results.append(VectorSearchResult(
                    id=chunk_id,
                    score=score,
                    payload=payload,
                    vector=vector,
                    text=text
                ))
        
        return results

    def _get_candidate_ids_from_filter(self, filter_dict: Dict[str, Any]) -> Optional[Set[str]]:
        """
        Uses field indexes to get candidate chunk_ids that match the filter.
        Returns None if indexes can't be used for this filter.
        """
        if not self._config.indexed_fields:
            return None
        
        indexed_field_names: List[str] = self._get_field_names_from_config()
        if not indexed_field_names:
            return None
        
        candidate_sets: List[Set[str]] = []
        
        def extract_indexable_conditions(filt: Dict[str, Any], conditions: List[tuple]) -> None:
            if "and" in filt:
                for sub_filter in filt["and"]:
                    extract_indexable_conditions(sub_filter, conditions)
            elif "or" in filt:
                return
            else:
                for key, value in filt.items():
                    if key in indexed_field_names:
                        if isinstance(value, dict):
                            if "$in" in value:
                                conditions.append((key, value["$in"]))
                            elif "$ne" not in value and "$gt" not in value and "$lt" not in value and "$gte" not in value and "$lte" not in value:
                                conditions.append((key, value))
                        else:
                            conditions.append((key, value))
        
        conditions: List[tuple] = []
        extract_indexable_conditions(filter_dict, conditions)
        
        if not conditions:
            return None
        
        for field_name, value in conditions:
            if field_name in self._field_indexes:
                if isinstance(value, list):
                    candidate_set: Set[str] = set()
                    for v in value:
                        if v in self._field_indexes[field_name]:
                            candidate_set.update(self._field_indexes[field_name][v])
                    if candidate_set:
                        candidate_sets.append(candidate_set)
                else:
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, sort_keys=True)
                    if value in self._field_indexes[field_name]:
                        candidate_sets.append(self._field_indexes[field_name][value])
        
        if not candidate_sets:
            return None
        
        if len(candidate_sets) == 1:
            return candidate_sets[0]
        else:
            result: Set[str] = candidate_sets[0]
            for s in candidate_sets[1:]:
                result = result & s
            return result

    async def afull_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Performs a full-text search if the provider supports it (async)."""
        _ = (query_text, top_k, filter, similarity_threshold, apply_reranking, sparse_query_vector)
        raise NotImplementedError("FAISS is a dense-vector-only library and does not support full-text search.")

    async def ahybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Combines dense and sparse/keyword search results (async)."""
        _ = (query_text, alpha, fusion_method, sparse_query_vector)
        warning_log("FAISS provider received a hybrid search request. It will ignore the text query and alpha, performing a dense search instead.", context="FaissVectorDB")
        return await self.adense_search(query_vector=query_vector, top_k=top_k, filter=filter, similarity_threshold=similarity_threshold, apply_reranking=apply_reranking)

    # ============================================================================
    # Generic Field Helpers
    # ============================================================================

    async def afield_exists(self, field_name: str, field_value: Any) -> bool:
        """
        Generic check if any entry with the given field value exists.

        Args:
            field_name: The payload field name to filter on.
            field_value: The value to match.

        Returns:
            True if at least one matching entry exists, False otherwise.
        """
        if not await self.ais_ready():
            return False

        for payload in self._metadata_store.values():
            if payload.get(field_name) == field_value:
                return True
        return False

    async def adelete_by_field(self, field_name: str, field_value: Any) -> bool:
        """
        Generic deletion of all entries matching a payload field value.

        Includes a pre-existence check. Returns True if no matches found (idempotent).

        Args:
            field_name: The payload field name to filter on.
            field_value: The value to match for deletion.

        Returns:
            True if deletion was successful or no matches found, False on failure.
        """
        if not await self.ais_ready():
            return False

        chunk_ids_to_delete: List[str] = [
            chunk_id for chunk_id, payload in self._metadata_store.items()
            if payload.get(field_name) == field_value
        ]

        if not chunk_ids_to_delete:
            warning_log(f"No entries found with {field_name}: {field_value}", context="FaissVectorDB")
            return True

        info_log(f"Found {len(chunk_ids_to_delete)} entries to delete with {field_name}: {field_value}", context="FaissVectorDB")
        await self.adelete(chunk_ids_to_delete)
        return True

    # ============================================================================
    # Existence Check Methods
    # ============================================================================

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        """Checks if a chunk ID exists in the vector database (async)."""
        return await self.afield_exists("chunk_id", chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        """Checks if a document content hash exists in the vector database (async)."""
        return await self.afield_exists("doc_content_hash", doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        """Checks if a chunk content hash exists in the vector database (async)."""
        return await self.afield_exists("chunk_content_hash", chunk_content_hash)

    async def adocument_id_exists(self, document_id: str) -> bool:
        """Checks if a document ID exists in the vector database (async)."""
        return await self.afield_exists("document_id", document_id)

    async def adocument_name_exists(self, document_name: str) -> bool:
        """Checks if a document name exists in the vector database (async)."""
        return await self.afield_exists("document_name", document_name)

    # ============================================================================
    # Delete-by-Field Methods
    # ============================================================================

    async def adelete_by_document_name(self, document_name: str) -> bool:
        """Removes data from the collection by their document name (async)."""
        return await self.adelete_by_field("document_name", document_name)

    async def adelete_by_document_id(self, document_id: str) -> bool:
        """Removes data from the collection by their document ID (async)."""
        return await self.adelete_by_field("document_id", document_id)

    async def adelete_by_chunk_id(self, chunk_id: str) -> bool:
        """Removes data from the collection by their chunk ID (async)."""
        return await self.adelete_by_field("chunk_id", chunk_id)

    async def adelete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        """Removes data from the collection by their document content hash (async)."""
        return await self.adelete_by_field("doc_content_hash", doc_content_hash)

    async def adelete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        """Removes data from the collection by their chunk content hash (async)."""
        return await self.adelete_by_field("chunk_content_hash", chunk_content_hash)

    async def adelete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Removes data from the collection by their metadata (async).

        Returns True if no matches found (idempotent no-op).

        Args:
            metadata: Dictionary of metadata key-value pairs to match.

        Returns:
            True if deletion was successful or no matches found, False on failure.
        """
        if not await self.ais_ready():
            return False

        filter_func = self._build_filter_function(metadata)
        if not filter_func:
            return False

        chunk_ids_to_delete: List[str] = [
            chunk_id for chunk_id, payload in self._metadata_store.items()
            if filter_func(payload)
        ]

        if not chunk_ids_to_delete:
            warning_log(f"No entries found matching metadata: {metadata}", context="FaissVectorDB")
            return True

        await self.adelete(chunk_ids_to_delete)
        return True

    # ============================================================================
    # Metadata & Optimization
    # ============================================================================

    async def aupdate_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """Updates the metadata for a specific chunk ID (async)."""
        if not await self.ais_ready():
            return False
        
        chunk_id_str: str = str(chunk_id)
        if chunk_id_str not in self._metadata_store:
            return False
        
        payload: Dict[str, Any] = self._metadata_store[chunk_id_str]
        
        if 'metadata' not in payload:
            payload['metadata'] = {}
        
        payload['metadata'].update(metadata)
        
        self._update_field_indexes(chunk_id_str, payload, operation='remove')
        self._update_field_indexes(chunk_id_str, payload, operation='add')
        
        return True

    async def aoptimize(self) -> bool:
        """Optimizes the vector database (async)."""
        return True

    async def aget_supported_search_types(self) -> List[str]:
        """Gets the supported search types for the vector database (async)."""
        return ['dense']
