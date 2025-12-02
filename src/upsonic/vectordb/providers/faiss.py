from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal, Callable, Set, TYPE_CHECKING
from hashlib import md5
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
    FlatIndexConfig
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

    **Key Features:**
    - Uses content_id as the main identifier (auto-generated if not provided)
    - Supports comprehensive metadata structure: document_name, document_id, content_id, metadata, content
    - Custom indexing for fast field lookups (FAISS doesn't support native field indexing)
    - Advanced filtering with configurable filter building logic
    - User metadata merging from config or method parameters

    **Concurrency Warning:** This implementation is NOT thread-safe or process-safe.
    Concurrent write operations can lead to state corruption. It is designed for
    single-threaded access patterns, such as in local applications or batch processing.
    """

    def __init__(self, config: Union[FaissConfig, Dict[str, Any]]):
        """
        Initialize the FAISS provider.

        Args:
            config: Either a FaissConfig object or a dictionary that will be converted to FaissConfig.
        """
        if not _FAISS_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="faiss-cpu",
                install_command='pip install "upsonic[rag]"',
                feature_name="FAISS vector database provider"
            )

        if not _NUMPY_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="numpy",
                install_command='pip install "upsonic[embeddings]"',
                feature_name="FAISS vector database provider"
            )

        # Convert dict to FaissConfig if needed
        if isinstance(config, dict):
            config = FaissConfig.from_dict(config)

        # Config validation is handled by pydantic models
        super().__init__(config)
        
        # FAISS index and core mappings
        self._index: Optional[faiss.Index] = None
        self._metadata_store: Dict[str, Dict[str, Any]] = {}  # content_id -> full payload
        self._content_id_to_faiss_id: Dict[str, int] = {}  # content_id -> FAISS internal ID
        self._faiss_id_to_content_id: Dict[int, str] = {}  # FAISS internal ID -> content_id
        
        # Custom field indexes for fast lookups (FAISS doesn't support native field indexing)
        # Structure: field_name -> field_value -> Set[content_id]
        self._field_indexes: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Path management
        self._base_db_path: Optional[Path] = Path(self._config.db_path) if self._config.db_path else None
        self._normalize_vectors = self._config.normalize_vectors

    # ============================================================================
    # Connection Management
    # ============================================================================

    async def connect(self) -> None:
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
                        self._content_id_to_faiss_id = maps.get("content_id_to_faiss", {})
                        self._faiss_id_to_content_id = {int(k): v for k, v in maps.get("faiss_to_content_id", {}).items()}
                
                if indexes_file.exists():
                    with open(indexes_file, 'r') as f:
                        indexes_data = json.load(f)
                        # Reconstruct field indexes
                        for field_name, value_map in indexes_data.items():
                            self._field_indexes[field_name] = {
                                k: set(v) for k, v in value_map.items()
                            }
            
            self._is_connected = True
            info_log(f"Connected to FAISS collection '{self._config.collection_name}'", context="FaissVectorDB")
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to hydrate FAISS state from disk: {e}")

    def connect_sync(self) -> None:
        """Establishes a connection to the vector database (sync)."""
        self._run_async_from_sync(self.connect())

    async def disconnect(self) -> None:
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
                        "content_id_to_faiss": self._content_id_to_faiss_id,
                        "faiss_to_content_id": self._faiss_id_to_content_id
                    }, f)
                
                # Persist field indexes
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
        self._content_id_to_faiss_id.clear()
        self._faiss_id_to_content_id.clear()
        self._field_indexes.clear()
        self._is_connected = False

    def disconnect_sync(self) -> None:
        """Gracefully terminates the connection to the vector database (sync)."""
        self._run_async_from_sync(self.disconnect())

    async def is_ready(self) -> bool:
        """Performs a health check to ensure the database is responsive (async)."""
        return self._is_connected and self._index is not None

    def is_ready_sync(self) -> bool:
        """Performs a health check to ensure the database is responsive (sync)."""
        return self._run_async_from_sync(self.is_ready())

    # ============================================================================
    # Collection Management
    # ============================================================================

    async def create_collection(self) -> None:
        """
        Creates the collection by building a FAISS index in memory based on the
        provider's configuration.
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Must be connected before creating a collection.")
        
        # Check if index already exists in memory
        if self._index is not None:
            if self._config.recreate_if_exists:
                info_log("Collection exists in memory. Recreating due to recreate_if_exists=True.", context="FaissVectorDB")
                await self.delete_collection()
            else:
                info_log("Collection (FAISS index) already exists in memory.", context="FaissVectorDB")
                return

        # Check if collection exists on disk
        if await self.collection_exists():
            if self._config.recreate_if_exists:
                info_log("Deleting existing collection on disk to recreate.", context="FaissVectorDB")
                await self.delete_collection()
            else:
                info_log("Collection path exists but index not loaded. Proceeding to create new index.", context="FaissVectorDB")

        if self._active_db_path:
            self._active_db_path.mkdir(parents=True, exist_ok=True)
        
        d = self._config.vector_size
        index_conf = self._config.index
        
        factory_parts = []
        
        if isinstance(index_conf, IVFIndexConfig):
            factory_parts.append(f"IVF{index_conf.nlist}")
        elif isinstance(index_conf, HNSWIndexConfig):
            factory_parts.append(f"HNSW{index_conf.m}")

        # Handle quantization if specified
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

    def create_collection_sync(self) -> None:
        """Creates the collection in the database according to the full config (sync)."""
        self._run_async_from_sync(self.create_collection())

    async def delete_collection(self) -> None:
        """Permanently deletes the collection specified in config (async)."""
        if not self._active_db_path:
            debug_log("Cannot delete collection in 'in_memory' mode.", context="FaissVectorDB")
            self._index = None
            self._metadata_store.clear()
            self._content_id_to_faiss_id.clear()
            self._faiss_id_to_content_id.clear()
            self._field_indexes.clear()
            return

        if await self.collection_exists():
            try:
                shutil.rmtree(self._active_db_path)
                info_log(f"Successfully deleted collection directory: '{self._active_db_path}'", context="FaissVectorDB")
            except OSError as e:
                raise VectorDBError(f"Error deleting collection directory '{self._active_db_path}': {e}")
        else:
            debug_log("Collection directory does not exist. No action taken.", context="FaissVectorDB")

        self._index = None
        self._metadata_store.clear()
        self._content_id_to_faiss_id.clear()
        self._faiss_id_to_content_id.clear()
        self._field_indexes.clear()

    def delete_collection_sync(self) -> None:
        """Permanently deletes the collection specified in config (sync)."""
        self._run_async_from_sync(self.delete_collection())

    async def collection_exists(self) -> bool:
        """Checks if the collection specified in the config already exists (async)."""
        if not self._active_db_path:
            return self._index is not None
        
        return self._active_db_path.is_dir() and any(self._active_db_path.iterdir())

    def collection_exists_sync(self) -> bool:
        """Checks if the collection specified in the config already exists (sync)."""
        return self._run_async_from_sync(self.collection_exists())

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
        return md5(identifier.encode()).hexdigest()[:16]

    def _generate_content_id(self, content: str) -> str:
        """Generate a unique content ID from content."""
        return md5(content.encode('utf-8')).hexdigest()

    def _build_payload(
        self,
        payload: Dict[str, Any],
        content: str,
        user_id: Union[str, int],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds a standardized payload structure with all required fields.

        Args:
            payload: Original payload from user
            content: Text content (required)
            user_id: User-provided ID (may be different from content_id)
            additional_metadata: Additional metadata to merge (from method parameter)

        Returns:
            Standardized payload dictionary
        """
        # Extract or generate content_id (main ID)
        content_id = payload.get('content_id')
        if not content_id:
            if self._config.auto_generate_content_id:
                content_id = self._generate_content_id(content)
            else:
                content_id = str(user_id)  # Fallback to user_id
        
        # Extract optional fields
        document_name = payload.get('document_name')
        document_id = payload.get('document_id')
        
        # Build metadata dict
        metadata = {}
        
        # 1. Start with default_metadata from config
        if self._config.default_metadata:
            metadata.update(self._config.default_metadata)
        
        # 2. Add metadata from payload if exists
        if 'metadata' in payload and isinstance(payload['metadata'], dict):
            metadata.update(payload['metadata'])
        
        # 3. Add additional_metadata from method parameter (highest priority)
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Build final payload
        final_payload = {
            'content_id': content_id,
            'content': content,
        }
        
        # Add optional fields
        if document_name:
            final_payload['document_name'] = document_name
        if document_id:
            final_payload['document_id'] = document_id
        if metadata:
            final_payload['metadata'] = metadata
        
        return final_payload

    def _parse_indexed_fields(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse indexed_fields into a standardized format.
        
        Supports two formats:
        1. Simple: ["document_name", "document_id"]
        2. Advanced: [{"field": "document_name", "type": "keyword"}, {"field": "age", "type": "integer"}]
        
        Note: FAISS doesn't support native field types. Types are parsed for consistency but not used.
        
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
    
    def _get_field_names_from_config(self) -> List[str]:
        """
        Extract field names from indexed_fields configuration.
        
        Returns:
            List of field names to index
        """
        indexed_fields_config = self._parse_indexed_fields()
        return list(indexed_fields_config.keys())
    
    def _update_field_indexes(self, content_id: str, payload: Dict[str, Any], operation: str = 'add') -> None:
        """
        Updates field indexes for fast lookups.
        
        Note: FAISS uses Python data structures for indexing. Field types from configuration
        are not used since FAISS doesn't support native typed indexes.

        Args:
            content_id: The content ID to index
            payload: The payload containing field values
            operation: 'add' or 'remove'
        """
        if not self._config.indexed_fields:
            return
        
        # Get field names (supports both simple and advanced format)
        field_names = self._get_field_names_from_config()
        
        for field_name in field_names:
            # Handle nested fields (e.g., 'metadata.key')
            if '.' in field_name:
                parts = field_name.split('.')
                value = payload
                try:
                    for part in parts:
                        value = value[part]
                except (KeyError, TypeError):
                    continue
            else:
                value = payload.get(field_name)
            
            if value is None:
                continue
            
            # Convert value to hashable type for indexing
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            
            if operation == 'add':
                self._field_indexes[field_name][value].add(content_id)
            elif operation == 'remove':
                if value in self._field_indexes[field_name]:
                    self._field_indexes[field_name][value].discard(content_id)
                    # Clean up empty sets
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

        Args:
            filter_dict: Filter dictionary

        Returns:
            Filter function or None
        """
        if not filter_dict:
            return None

        def build_checker(key: str, value: Any) -> Callable[[Dict[str, Any]], bool]:
            """Build a checker function for a single field-value pair."""
            if isinstance(value, dict):
                # Handle operators
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
                    # Nested dict comparison
                    return lambda payload: self._get_nested_value(payload, key) == value
            
            # Direct value matching
            return lambda payload: self._get_nested_value(payload, key) == value

        # Handle logical operators
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
        
        # Build checkers for all field-value pairs
        checkers = [build_checker(k, v) for k, v in filter_dict.items()]
        if not checkers:
            return None
        
        return lambda payload: all(checker(payload) for checker in checkers)

    def _get_nested_value(self, payload: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get a nested value from payload, supporting dot notation."""
        if '.' not in key:
            return payload.get(key, default)
        
        parts = key.split('.')
        value = payload
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
        Adds new data or updates existing data in the collection (async).

        This method:
        1. Builds standardized payloads with document_name, document_id, content_id, metadata, content
        2. Uses content_id as the main identifier
        3. Merges user metadata from config or method parameter
        4. Updates field indexes for fast lookups

        Args:
            vectors: A list of dense vector embeddings
            payloads: A list of corresponding metadata objects
            ids: A list of unique identifiers (may differ from content_id)
            chunks: Optional list of text content (required if not in payloads)
            sparse_vectors: Not supported by FAISS, will be ignored
            **kwargs: Additional options including 'metadata' dict for merging
        """
        if not await self.is_ready():
            raise VectorDBError("FAISS index is not created. Please call 'create_collection' first.")
        
        if not (len(vectors) == len(payloads) == len(ids)):
            raise UpsertError("The lengths of vectors, payloads, and ids lists must be identical.")
        
        if not vectors:
            return

        # Get additional metadata from kwargs
        additional_metadata = kwargs.get('metadata', {})

        # Validate and extract content
        contents = []
        for i, payload in enumerate(payloads):
            if chunks and i < len(chunks):
                content = chunks[i]
            elif 'content' in payload:
                content = payload['content']
            elif 'text' in payload:
                content = payload['text']
            else:
                raise UpsertError(f"Record {i} missing required 'content' field. Provide 'chunks' parameter or include 'content'/'text' in payload.")
            
            if not isinstance(content, str) or not content.strip():
                raise UpsertError(f"Record {i} has invalid content. Content must be a non-empty string.")
            
            contents.append(content)

        # Build standardized payloads
        standardized_payloads = []
        content_ids_to_update = []
        
        for i, (payload, content, user_id) in enumerate(zip(payloads, contents, ids)):
            final_payload = self._build_payload(payload, content, user_id, additional_metadata)
            content_id = final_payload['content_id']
            standardized_payloads.append(final_payload)
            
            # Check if content_id already exists (update scenario)
            if content_id in self._content_id_to_faiss_id:
                content_ids_to_update.append(content_id)

        # Delete existing entries for updates
        if content_ids_to_update:
            debug_log(f"Updating {len(content_ids_to_update)} existing content_ids by deleting old entries first.", context="FaissVectorDB")
            await self.delete(content_ids_to_update)

        # Prepare vectors
        vectors_np = np.array(vectors, dtype=np.float32)
        if self._normalize_vectors:
            faiss.normalize_L2(vectors_np)

        # Train index if needed
        if not self._index.is_trained:
            debug_log(f"FAISS index is not trained. Training on {len(vectors_np)} vectors...", context="FaissVectorDB")
            self._index.train(vectors_np)
            info_log("Training complete.", context="FaissVectorDB")

        try:
            start_faiss_id = self._index.ntotal
            self._index.add(vectors_np)
            
            # Store metadata and update indexes
            for i, (final_payload, content_id) in enumerate(zip(standardized_payloads, [p['content_id'] for p in standardized_payloads])):
                faiss_id = start_faiss_id + i
                self._content_id_to_faiss_id[content_id] = faiss_id
                self._faiss_id_to_content_id[faiss_id] = content_id
                self._metadata_store[content_id] = final_payload
                
                # Update field indexes
                self._update_field_indexes(content_id, final_payload, operation='add')
            
            info_log(f"Successfully upserted {len(vectors)} vectors. Index total: {self._index.ntotal}", context="FaissVectorDB")

        except Exception as e:
            raise UpsertError(f"An error occurred during FAISS add operation: {e}")

    def upsert_sync(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Adds new data or updates existing data in the collection (sync)."""
        self._run_async_from_sync(self.upsert(vectors, payloads, ids, chunks, sparse_vectors, **kwargs))

    async def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their content_ids (async).

        Args:
            ids: A list of content_ids to remove
            **kwargs: Provider-specific options
        """
        if not await self.is_ready():
            return

        deleted_count = 0
        for content_id in ids:
            content_id_str = str(content_id)
            if content_id_str in self._content_id_to_faiss_id:
                faiss_id = self._content_id_to_faiss_id[content_id_str]
                
                # Remove from indexes
                if content_id_str in self._metadata_store:
                    self._update_field_indexes(content_id_str, self._metadata_store[content_id_str], operation='remove')
                
                del self._content_id_to_faiss_id[content_id_str]
                del self._faiss_id_to_content_id[faiss_id]
                if content_id_str in self._metadata_store:
                    del self._metadata_store[content_id_str]
                
                deleted_count += 1
        
        if deleted_count > 0:
            info_log(f"Successfully deleted {deleted_count} content_ids (tombstoned).", context="FaissVectorDB")

    def delete_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Removes data from the collection by their content_ids (sync)."""
        self._run_async_from_sync(self.delete(ids, **kwargs))

    async def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their content_ids (async).

        Args:
            ids: A list of content_ids to retrieve
            **kwargs: Provider-specific options

        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        if not await self.is_ready():
            return []
        
        results = []
        for content_id in ids:
            content_id_str = str(content_id)
            if content_id_str in self._content_id_to_faiss_id:
                try:
                    faiss_id = self._content_id_to_faiss_id[content_id_str]
                    payload = self._metadata_store.get(content_id_str)
                    
                    vector = self._index.reconstruct(faiss_id).tolist()
                    
                    text = payload.get("content", "") if payload else ""
                    results.append(VectorSearchResult(
                        id=content_id_str,
                        score=1.0,
                        payload=payload,
                        vector=vector,
                        text=text
                    ))
                except Exception as e:
                    debug_log(f"Could not fetch data for content_id '{content_id_str}': {e}", context="FaissVectorDB")
        return results

    def fetch_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Retrieves full records (payload and vector) by their content_ids (sync)."""
        return self._run_async_from_sync(self.fetch(ids, **kwargs))

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
        A master search method that dispatches to the appropriate specialized
        search function based on the provided arguments (async).
        """
        final_top_k = top_k if top_k is not None else self._config.default_top_k or 10
        has_vector = query_vector is not None and len(query_vector) > 0
        has_text = query_text is not None and query_text.strip()

        # Use defaults from config if not provided
        final_alpha = alpha if alpha is not None else self._config.default_hybrid_alpha
        final_fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method

        if has_vector and has_text:
            if not self._config.hybrid_search_enabled:
                raise ConfigurationError("Hybrid search is disabled in the configuration.")
            return await self.hybrid_search(query_vector, query_text, final_top_k, filter, final_alpha, final_fusion_method, similarity_threshold, **kwargs)
        elif has_vector:
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled in the configuration.")
            return await self.dense_search(query_vector, final_top_k, filter, similarity_threshold, **kwargs)
        elif has_text:
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled in the configuration.")
            return await self.full_text_search(query_text, final_top_k, filter, similarity_threshold, **kwargs)
        else:
            raise ConfigurationError("Search requires at least one of 'query_vector' or 'query_text'.")

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
        """A master search method that dispatches to the appropriate specialized search function (sync)."""
        return self._run_async_from_sync(self.search(top_k, query_vector, query_text, filter, alpha, fusion_method, similarity_threshold, **kwargs))

    async def dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Performs a pure vector similarity search (async)."""
        if not await self.is_ready():
            raise SearchError("FAISS index is not ready for search.")
        if self._index.ntotal == 0:
            return []

        filter_func = self._build_filter_function(filter)
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        
        # Use field indexes to pre-filter if possible
        candidate_content_ids = None
        if filter_func and self._config.indexed_fields:
            # Try to use indexes for fast pre-filtering
            candidate_content_ids = self._get_candidate_ids_from_filter(filter)
        
        # If we have candidate IDs, we can optimize search
        if candidate_content_ids is not None and len(candidate_content_ids) == 0:
            return []  # No candidates match filter
        
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
            if len(results) >= top_k:
                break
            if faiss_id == -1:
                continue

            content_id = self._faiss_id_to_content_id.get(faiss_id)
            if not content_id:
                continue

            # Pre-filter using candidate IDs if available
            if candidate_content_ids is not None and content_id not in candidate_content_ids:
                continue

            payload = self._metadata_store.get(content_id)
            if filter_func and not filter_func(payload or {}):
                continue

            # Convert distance to similarity score (0-1 range)
            if self._config.distance_metric == DistanceMetric.EUCLIDEAN:
                # Euclidean distance: convert to similarity (0-1)
                score = 1 / (1 + dist)
            elif self._config.distance_metric == DistanceMetric.COSINE:
                # Cosine similarity: FAISS returns inner product for normalized vectors
                # Clamp to [0, 1] range
                score = max(0.0, min(1.0, float(dist)))
            elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
                # Dot product: normalize to [0, 1] range
                # For normalized vectors, dot product is in [-1, 1]
                # For unnormalized vectors, we need to handle larger values
                # Use sigmoid-like transformation for better normalization
                dist_float = float(dist)
                if dist_float <= -1.0:
                    score = 0.0
                elif dist_float >= 1.0:
                    score = 1.0
                else:
                    # Normalize from [-1, 1] to [0, 1]
                    score = (dist_float + 1.0) / 2.0
            else:
                # Default: clamp to [0, 1]
                score = max(0.0, min(1.0, float(dist)))

            if final_similarity_threshold is None or score >= final_similarity_threshold:
                text = payload.get("content", "") if payload else ""
                results.append(VectorSearchResult(
                    id=content_id,
                    score=score,
                    payload=payload,
                    text=text
                ))
        
        return results

    def dense_search_sync(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Performs a pure vector similarity search (sync)."""
        return self._run_async_from_sync(self.dense_search(query_vector, top_k, filter, similarity_threshold, **kwargs))

    def _get_candidate_ids_from_filter(self, filter_dict: Dict[str, Any]) -> Optional[Set[str]]:
        """
        Uses field indexes to get candidate content_ids that match the filter.
        Returns None if indexes can't be used for this filter.
        """
        if not self._config.indexed_fields:
            return None
        
        # Get indexed field names (supports both simple and advanced format)
        indexed_field_names = self._get_field_names_from_config()
        if not indexed_field_names:
            return None
        
        # Try to extract simple field-value pairs that can use indexes
        candidate_sets = []
        
        def extract_indexable_conditions(filt: Dict[str, Any], conditions: List[tuple]) -> None:
            """Recursively extract indexable conditions."""
            if "and" in filt:
                for sub_filter in filt["and"]:
                    extract_indexable_conditions(sub_filter, conditions)
            elif "or" in filt:
                # For OR, we can't easily use indexes
                return
            else:
                for key, value in filt.items():
                    if key in indexed_field_names:
                        if isinstance(value, dict):
                            if "$in" in value:
                                conditions.append((key, value["$in"]))
                            elif "$ne" not in value and "$gt" not in value and "$lt" not in value and "$gte" not in value and "$lte" not in value:
                                # Simple equality
                                conditions.append((key, value))
                        else:
                            conditions.append((key, value))
        
        conditions = []
        extract_indexable_conditions(filter_dict, conditions)
        
        if not conditions:
            return None
        
        # Get candidate sets from indexes
        for field_name, value in conditions:
            if field_name in self._field_indexes:
                if isinstance(value, list):  # $in operator
                    candidate_set = set()
                    for v in value:
                        if v in self._field_indexes[field_name]:
                            candidate_set.update(self._field_indexes[field_name][v])
                    if candidate_set:
                        candidate_sets.append(candidate_set)
                else:
                    # Convert value to hashable type
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, sort_keys=True)
                    if value in self._field_indexes[field_name]:
                        candidate_sets.append(self._field_indexes[field_name][value])
        
        if not candidate_sets:
            return None
        
        # Intersect all candidate sets (AND logic)
        if len(candidate_sets) == 1:
            return candidate_sets[0]
        else:
            result = candidate_sets[0]
            for s in candidate_sets[1:]:
                result = result & s
            return result

    async def full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Performs a full-text search if the provider supports it (async)."""
        raise NotImplementedError("FAISS is a dense-vector-only library and does not support full-text search.")

    def full_text_search_sync(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Performs a full-text search if the provider supports it (sync)."""
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
        """Combines dense and sparse/keyword search results (async)."""
        warning_log("FAISS provider received a hybrid search request. It will ignore the text query and alpha, performing a dense search instead.", context="FaissVectorDB")
        return await self.dense_search(query_vector=query_vector, top_k=top_k, filter=filter, similarity_threshold=similarity_threshold, **kwargs)

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
        """Combines dense and sparse/keyword search results (sync)."""
        return self._run_async_from_sync(self.hybrid_search(query_vector, query_text, top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs))

    # ============================================================================
    # Document/Content Management Methods
    # ============================================================================

    def document_id_exists(self, document_id: str) -> bool:
        """Checks if a document ID exists in the vector database (sync)."""
        if not self.is_ready_sync():
            return False
        
        for payload in self._metadata_store.values():
            if payload.get('document_id') == document_id:
                return True
        return False

    async def async_document_id_exists(self, document_id: str) -> bool:
        """Checks if a document ID exists in the vector database (async)."""
        return self.document_id_exists(document_id)

    def document_name_exists(self, document_name: str) -> bool:
        """Checks if a document name exists in the vector database (sync)."""
        if not self.is_ready_sync():
            return False
        
        for payload in self._metadata_store.values():
            if payload.get('document_name') == document_name:
                return True
        return False

    async def async_document_name_exists(self, document_name: str) -> bool:
        """Checks if a document name exists in the vector database (async)."""
        return self.document_name_exists(document_name)

    def content_id_exists(self, content_id: str) -> bool:
        """Checks if a content ID exists in the vector database (sync)."""
        return self.is_ready_sync() and str(content_id) in self._content_id_to_faiss_id

    async def async_content_id_exists(self, content_id: str) -> bool:
        """Checks if a content ID exists in the vector database (async)."""
        return self.content_id_exists(content_id)

    def optimize(self) -> bool:
        """Optimizes the vector database (sync)."""
        return True

    async def async_optimize(self) -> bool:
        """Optimizes the vector database (async)."""
        return self.optimize()

    def delete_by_document_name(self, document_name: str) -> bool:
        """Removes data from the collection by their document name (sync)."""
        if not self.is_ready_sync():
            return False
        
        content_ids_to_delete = []
        for content_id, payload in self._metadata_store.items():
            if payload.get('document_name') == document_name:
                content_ids_to_delete.append(content_id)
        
        if content_ids_to_delete:
            self.delete_sync(content_ids_to_delete)
            return True
        return False

    async def async_delete_by_document_name(self, document_name: str) -> bool:
        """Removes data from the collection by their document name (async)."""
        return self.delete_by_document_name(document_name)

    def delete_by_document_id(self, document_id: str) -> bool:
        """Removes data from the collection by their document ID (sync)."""
        if not self.is_ready_sync():
            return False
        
        content_ids_to_delete = []
        for content_id, payload in self._metadata_store.items():
            if payload.get('document_id') == document_id:
                content_ids_to_delete.append(content_id)
        
        if content_ids_to_delete:
            self.delete_sync(content_ids_to_delete)
            return True
        return False

    async def async_delete_by_document_id(self, document_id: str) -> bool:
        """Removes data from the collection by their document ID (async)."""
        return self.delete_by_document_id(document_id)

    def delete_by_content_id(self, content_id: str) -> bool:
        """Removes data from the collection by their content ID (sync)."""
        if not self.is_ready_sync():
            return False
        
        if str(content_id) in self._content_id_to_faiss_id:
            self.delete_sync([content_id])
            return True
        return False

    async def async_delete_by_content_id(self, content_id: str) -> bool:
        """Removes data from the collection by their content ID (async)."""
        return self.delete_by_content_id(content_id)

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Removes data from the collection by their metadata (sync)."""
        if not self.is_ready_sync():
            return False
        
        filter_func = self._build_filter_function(metadata)
        if not filter_func:
            return False
        
        content_ids_to_delete = []
        for content_id, payload in self._metadata_store.items():
            if filter_func(payload):
                content_ids_to_delete.append(content_id)
        
        if content_ids_to_delete:
            self.delete_sync(content_ids_to_delete)
            return True
        return False

    async def async_delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Removes data from the collection by their metadata (async)."""
        return self.delete_by_metadata(metadata)

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """Updates the metadata for a specific content ID (sync)."""
        if not self.is_ready_sync():
            return False
        
        content_id_str = str(content_id)
        if content_id_str not in self._metadata_store:
            return False
        
        payload = self._metadata_store[content_id_str]
        
        # Update metadata field
        if 'metadata' not in payload:
            payload['metadata'] = {}
        
        # Merge new metadata
        payload['metadata'].update(metadata)
        
        # Update field indexes
        self._update_field_indexes(content_id_str, payload, operation='remove')
        self._update_field_indexes(content_id_str, payload, operation='add')
        
        return True

    async def async_update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """Updates the metadata for a specific content ID (async)."""
        return self.update_metadata(content_id, metadata)

    def get_supported_search_types(self) -> List[str]:
        """Gets the supported search types for the vector database (sync)."""
        return ['dense']  # FAISS only supports dense vector search

    async def async_get_supported_search_types(self) -> List[str]:
        """Gets the supported search types for the vector database (async)."""
        return self.get_supported_search_types()
