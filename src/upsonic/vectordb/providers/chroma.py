from __future__ import annotations

import asyncio
import json
import math
import uuid as _uuid
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb
    from chromadb.errors import NotFoundError

try:
    import chromadb
    from chromadb.errors import NotFoundError
    _CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore
    NotFoundError = None  # type: ignore
    _CHROMADB_AVAILABLE = False


from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.utils.printing import info_log, debug_log

from upsonic.vectordb.config import (
    ChromaConfig,
    Mode,
    DistanceMetric,
    HNSWIndexConfig,
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
    A comprehensive, async-first implementation of BaseVectorDBProvider for ChromaDB.

    Standard metadata fields stored per record:
    - chunk_id (str) — unique per-chunk identifier
    - chunk_content_hash (str) — MD5 of chunk text content for deduplication
    - document_id (str) — parent document identifier
    - doc_content_hash (str) — MD5 of parent document content for change detection
    - document_name (str) — human-readable source name
    - metadata (str) — JSON-serialised dict of all non-standard data

    This provider offers:
    - Full async/sync support with context manager lifecycle
    - Content-based deduplication via chunk_content_hash
    - Dense vector search (ChromaDB auto-indexes all metadata)
    - Full-text search using document content filtering
    - Hybrid search combining dense vectors and full-text
    - Generic field_exists / delete_by_field helpers
    - Rich filtering with ChromaDB operators ($eq, $ne, $in, $gt, etc.)

    Note: ChromaDB does NOT support sparse vectors. Hybrid search combines
    dense vector similarity with document-content keyword matching.

    Attributes:
        reranker: Optional reranker instance for re-ranking search results.
    """
    
    def __init__(
        self, 
        config: Union[ChromaConfig, Dict[str, Any]],
        reranker: Optional[Any] = None
    ):
        """
        Initialize ChromaProvider with config or dict.
        
        Args:
            config: ChromaConfig object or dict containing configuration
            reranker: Optional reranker for search result re-ranking (to be implemented)
        """
        if not _CHROMADB_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="chromadb",
                install_command='pip install "upsonic[chroma]"',
                feature_name="ChromaDB vector database provider"
            )
        
        # Handle dict config
        if isinstance(config, dict):
            config = ChromaConfig.from_dict(config)
        
        super().__init__(config)
        self._config: ChromaConfig = config  # Type hint for better IDE support
        self._validate_config()
        self._collection_instance: Optional[Any] = None
        
        # Provider utilities
        self.reranker = reranker

    def _validate_config(self) -> None:
        """Validate Chroma-specific configuration."""
        debug_log("Performing Chroma-specific configuration validation...", context="ChromaVectorDB")
        
        if self._config.connection.mode == Mode.EMBEDDED and self._config.connection.db_path is None:
            raise ConfigurationError("Missing Path: 'db_path' must be set for EMBEDDED (PersistentClient) mode.")
        
        info_log("Chroma configuration validated successfully.", context="ChromaVectorDB")
    
    def _generate_provider_id(self) -> str:
        """Generates a unique provider ID based on connection details and collection."""
        conn = self._config.connection
        identifier_parts = [
            conn.host or conn.url or conn.location or "local",
            str(conn.port) if conn.port else "",
            self._config.collection_name
        ]
        identifier = "#".join(filter(None, identifier_parts))
        
        return md5(identifier.encode()).hexdigest()[:16]

    # ============================================================================
    # Client Lifecycle Management
    # ============================================================================

    async def aget_client(self) -> Any:
        """
        Gets or creates the ChromaDB client, ensuring it is connected and ready.

        Follows a singleton pattern: reuses the existing client if available,
        creates a new one if needed, verifies readiness via heartbeat, and
        auto-recovers if the existing client is unresponsive.

        Returns:
            A connected and ready ChromaDB client instance.

        Raises:
            VectorDBConnectionError: If the client cannot be created or is not ready.
        """
        if self.client is None:
            debug_log(
                f"Creating client for '{self._config.connection.mode.value}' mode...",
                context="ChromaVectorDB",
            )
            self.client = await asyncio.to_thread(self._create_client)

        try:
            await asyncio.to_thread(self.client.heartbeat)
        except Exception:
            debug_log("Existing ChromaDB client unresponsive, recreating...", context="ChromaVectorDB")
            try:
                self.client = await asyncio.to_thread(self._create_client)
                await asyncio.to_thread(self.client.heartbeat)
            except Exception as e:
                self.client = None
                self._is_connected = False
                self._collection_instance = None
                raise VectorDBConnectionError(
                    f"ChromaDB client not ready after recreation: {e}"
                ) from e

        self._is_connected = True
        return self.client

    def get_client(self) -> Any:
        """
        Gets or creates the ChromaDB client (sync wrapper).

        Returns:
            A connected and ready ChromaDB client instance.

        Raises:
            VectorDBConnectionError: If the client cannot be created or is not ready.
        """
        return self._run_async_from_sync(self.aget_client())

    async def ais_client_connected(self) -> bool:
        """
        Checks whether the ChromaDB client exists and is responsive.
        Does not create or reconnect — purely a read-only status check.

        Returns:
            True if the client is connected and responsive, False otherwise.
        """
        if self.client is None:
            return False

        try:
            await asyncio.to_thread(self.client.heartbeat)
            return True
        except Exception:
            return False

    def is_client_connected(self) -> bool:
        """
        Checks whether the ChromaDB client exists and is responsive (sync).

        Returns:
            True if the client is connected and responsive, False otherwise.
        """
        return self._run_async_from_sync(self.ais_client_connected())

    # ============================================================================
    # Connection Lifecycle
    # ============================================================================

    async def aconnect(self) -> None:
        """
        Establishes an async connection to ChromaDB.

        Delegates to aget_client() which handles client creation, heartbeat
        verification, and auto-recovery. This method is idempotent.

        Raises:
            VectorDBConnectionError: If the connection fails for any reason.
        """
        if await self.ais_client_connected():
            info_log("Already connected to ChromaDB.", context="ChromaVectorDB")
            return

        debug_log(
            f"Attempting to connect to ChromaDB in '{self._config.connection.mode.value}' mode...",
            context="ChromaVectorDB",
        )

        try:
            await self.aget_client()
            info_log(
                "Successfully connected to ChromaDB and health check passed.",
                context="ChromaVectorDB",
            )
        except (VectorDBConnectionError, ConfigurationError):
            self.client = None
            self._is_connected = False
            self._collection_instance = None
            raise
        except Exception as e:
            self.client = None
            self._is_connected = False
            self._collection_instance = None
            raise VectorDBConnectionError(
                f"An unexpected error occurred during connection: {e}"
            ) from e

    def _create_client(self) -> Any:
        """Create ChromaDB client based on configuration."""
        client_instance: Any
        
        if self._config.connection.mode == Mode.IN_MEMORY:
            client_instance = chromadb.Client()
            
        elif self._config.connection.mode == Mode.EMBEDDED:
            client_instance = chromadb.PersistentClient(path=self._config.connection.db_path)
            
        elif self._config.connection.mode == Mode.LOCAL:
            if not self._config.connection.host or not self._config.connection.port:
                raise ConfigurationError("Host and port must be specified for LOCAL mode.")
            client_instance = chromadb.HttpClient(
                host=self._config.connection.host, 
                port=self._config.connection.port
            )
            
        elif self._config.connection.mode == Mode.CLOUD:
            if not self._config.connection.api_key:
                raise ConfigurationError("api_key must be specified for CLOUD mode.")
            
            # Prepare CloudClient kwargs
            cloud_kwargs = {
                "api_key": self._config.connection.api_key.get_secret_value()
            }
            
            # Add tenant and database if provided
            if self._config.tenant:
                cloud_kwargs["tenant"] = self._config.tenant
            if self._config.database:
                cloud_kwargs["database"] = self._config.database
            
            # Use CloudClient for Chroma Cloud connections
            try:
                client_instance = chromadb.CloudClient(**cloud_kwargs)
            except (AttributeError, ImportError, TypeError):
                # Fallback to HttpClient if CloudClient is not available
                if not self._config.connection.host:
                    raise ConfigurationError("CloudClient not available and no host specified for fallback HttpClient.")
                
                headers = {"Authorization": f"Bearer {self._config.connection.api_key.get_secret_value()}"}
                fallback_kwargs = {
                    "host": self._config.connection.host, 
                    "headers": headers,
                    "ssl": self._config.connection.use_tls
                }
                client_instance = chromadb.HttpClient(**fallback_kwargs)
        else:
            raise ConfigurationError(f"Unsupported mode for ChromaProvider: {self._config.connection.mode}")
        
        return client_instance

    async def adisconnect(self) -> None:
        """Gracefully disconnect from ChromaDB."""
        if not self._is_connected or not self.client:
            return
        
        debug_log("Disconnecting from ChromaDB...", context="ChromaVectorDB")
        
        try:
            # Add timeout to prevent hanging on reset
            await asyncio.wait_for(asyncio.to_thread(self.client.reset), timeout=5.0)
        except asyncio.TimeoutError:
            debug_log("ChromaDB reset timed out, forcing cleanup...", context="ChromaVectorDB")
        except Exception:
            pass
        finally:
            self.client = None
            self._is_connected = False
            self._collection_instance = None
            info_log("ChromaDB client session has been reset.", context="ChromaVectorDB")

    async def ais_ready(self) -> bool:
        """Check if the database is ready and responsive."""
        if not self._is_connected or not self.client:
            return False
        
        try:
            await asyncio.to_thread(self.client.heartbeat)
            return True
        except Exception:
            return False

    # ============================================================================
    # Collection Management (Async)
    # ============================================================================

    async def acreate_collection(self) -> None:
        """Create or retrieve the collection with proper configuration."""
        client = await self.aget_client()

        collection_name = self._config.collection_name

        try:
            if self._config.recreate_if_exists and await self.acollection_exists():
                info_log(
                    f"Configuration specifies 'recreate_if_exists'. "
                    f"Deleting existing collection '{collection_name}'...",
                    context="ChromaVectorDB",
                )
                await self.adelete_collection()

            chroma_metadata = self._translate_config_to_chroma_metadata()

            debug_log(f"Creating or retrieving collection '{collection_name}'...", context="ChromaVectorDB")

            self._collection_instance = await asyncio.to_thread(
                client.get_or_create_collection,
                name=collection_name,
                metadata=chroma_metadata,
            )

            info_log(f"Successfully prepared collection '{collection_name}'.", context="ChromaVectorDB")

        except Exception as e:
            raise VectorDBError(f"Failed to create or get collection '{collection_name}': {e}") from e

    def _translate_config_to_chroma_metadata(self) -> dict:
        """Translate framework config to ChromaDB metadata."""
        distance_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.DOT_PRODUCT: "ip"
        }
        
        metadata = {"hnsw:space": distance_map[self._config.distance_metric]}
        
        # Add HNSW-specific parameters
        if isinstance(self._config.index, HNSWIndexConfig):
            metadata["hnsw:M"] = self._config.index.m
            metadata["hnsw:construction_ef"] = self._config.index.ef_construction
        
        return metadata

    async def adelete_collection(self) -> None:
        """Delete the collection permanently."""
        client = await self.aget_client()

        collection_name = self._config.collection_name
        debug_log(f"Attempting to delete collection '{collection_name}'...", context="ChromaVectorDB")

        try:
            await asyncio.to_thread(client.delete_collection, name=collection_name)
            self._collection_instance = None
            info_log(f"Collection '{collection_name}' deleted successfully.", context="ChromaVectorDB")

        except (ValueError, chromadb.errors.NotFoundError) as e:
            raise CollectionDoesNotExistError(
                f"Cannot delete collection '{collection_name}' because it does not exist."
            ) from e
        except Exception as e:
            raise VectorDBError(
                f"An unexpected error occurred while deleting collection '{collection_name}': {e}"
            ) from e

    async def acollection_exists(self) -> bool:
        """Check if the collection exists."""
        client = await self.aget_client()

        try:
            collection = await asyncio.to_thread(
                client.get_collection,
                name=self._config.collection_name,
            )

            if self._collection_instance is None:
                self._collection_instance = collection

            return True

        except NotFoundError:
            return False
        except Exception as e:
            raise VectorDBConnectionError(
                f"Failed to check collection existence due to a server error: {e}"
            ) from e

    async def _get_active_collection(self) -> Any:
        """
        Ensure the collection instance is available.

        Uses aget_client() to guarantee the client is connected and ready
        before retrieving the collection reference.

        Returns:
            A ChromaDB Collection object.

        Raises:
            VectorDBConnectionError: If the client is not connected or not ready.
            VectorDBError: If the collection is not initialized.
        """
        client = await self.aget_client()

        if self._collection_instance is None:
            try:
                self._collection_instance = await asyncio.to_thread(
                    client.get_collection,
                    name=self._config.collection_name,
                )
            except Exception:
                raise VectorDBError(
                    "Collection is not initialized. Please call 'create_collection' before performing data operations."
                )

        return self._collection_instance


    _STANDARD_FIELDS: frozenset = frozenset({
        'document_name', 'document_id', 'chunk_id',
        'metadata', 'content', 'doc_content_hash', 'chunk_content_hash',
        'knowledge_base_id',
    })

    def _prepare_metadata(
        self,
        payload: Dict[str, Any],
        chunk_id: str,
        chunk_content_hash: str,
        document_id: str = "",
        doc_content_hash: str = "",
        document_name: str = "",
        knowledge_base_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        _warned_standard_keys: Optional[set] = None,
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Build the ChromaDB metadata dict (flat primitives only) with strict
        payload-contract compliance.

        Standard fields are taken ONLY from dedicated parameters. Non-standard
        keys from ``payload`` and the nested ``payload["metadata"]`` dict are
        merged into a single ``metadata`` JSON-serialized Chroma key.
        """
        combined_user_metadata: Dict[str, Any] = {}

        if self._config.default_metadata:
            combined_user_metadata.update(self._config.default_metadata)

        if payload:
            nested = payload.get("metadata")
            if isinstance(nested, dict):
                combined_user_metadata.update(nested)

            for key, value in payload.items():
                if key in self._STANDARD_FIELDS:
                    if key == "metadata":
                        continue
                    if _warned_standard_keys is not None and key not in _warned_standard_keys:
                        debug_log(
                            f"Standard field '{key}' found in payload dict and will be ignored. "
                            f"Standard fields must be passed via dedicated parameters "
                            f"(ids, document_ids, document_names, doc_content_hashes, "
                            f"chunk_content_hashes, knowledge_base_ids).",
                            context="ChromaVectorDB",
                        )
                        _warned_standard_keys.add(key)
                    continue
                combined_user_metadata[key] = value

        if extra_metadata:
            combined_user_metadata.update(extra_metadata)

        chroma_metadata: Dict[str, Union[str, int, float, bool]] = {
            "chunk_id": chunk_id or "",
            "chunk_content_hash": chunk_content_hash or "",
            "document_id": document_id or "",
            "doc_content_hash": doc_content_hash or "",
            "document_name": document_name or "",
            "knowledge_base_id": knowledge_base_id or "",
            "metadata": json.dumps(combined_user_metadata),
        }

        return chroma_metadata

    def _hydrate_payload(
        self,
        chroma_metadata: Optional[Dict[str, Any]],
        chroma_document: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build a contract-compliant VectorSearchResult.payload from Chroma's
        flat metadata dict + documents field.
        """
        chroma_metadata = chroma_metadata or {}
        raw_metadata = chroma_metadata.get("metadata", "{}")
        if isinstance(raw_metadata, dict):
            parsed_metadata = raw_metadata
        elif isinstance(raw_metadata, str) and raw_metadata:
            try:
                parsed = json.loads(raw_metadata)
                parsed_metadata = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed_metadata = {}
        else:
            parsed_metadata = {}

        return {
            "chunk_id": chroma_metadata.get("chunk_id", "") or "",
            "document_id": chroma_metadata.get("document_id", "") or "",
            "document_name": chroma_metadata.get("document_name", "") or "",
            "content": chroma_document or "",
            "doc_content_hash": chroma_metadata.get("doc_content_hash", "") or "",
            "chunk_content_hash": chroma_metadata.get("chunk_content_hash", "") or "",
            "knowledge_base_id": chroma_metadata.get("knowledge_base_id", "") or "",
            "metadata": parsed_metadata,
        }

    # ============================================================================
    # Filter Building
    # ============================================================================

    def _convert_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Convert simple filters to ChromaDB's format.
        
        ChromaDB requires:
        - Single condition: {"key": "value"} or {"key": {"$operator": value}}
        - Multiple conditions: {"$and": [{"key1": value1}, {"key2": value2}]}
        
        Args:
            filters: Filter dictionary to convert
            
        Returns:
            ChromaDB-compatible filter dictionary
        """
        if not filters:
            return None
        
        # If filters already use logical operators at top level, return as is
        if any(key.startswith("$") for key in filters.keys()):
            return filters
        
        # Convert simple key-value pairs
        conditions = []
        for key, value in filters.items():
            # If the value is already a dict with operators, add as is
            if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                conditions.append({key: value})
            # Convert lists to $in operator
            elif isinstance(value, (list, tuple)):
                conditions.append({key: {"$in": list(value)}})
            # Keep simple values as is (ChromaDB interprets as equality)
            else:
                conditions.append({key: value})
        
        # If only one condition, return it directly
        if len(conditions) == 1:
            return conditions[0]
        
        # If multiple conditions, wrap in $and
        return {"$and": conditions}

    def _split_filter(
        self,
        filters: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Split a filter dict into (native_chroma_filter, python_post_filter).

        Native filter: standard fields (excluding 'metadata' and 'content') that
        Chroma can match natively at the flat top level of its metadata.

        Post filter: non-standard user fields that live inside the JSON-serialized
        ``metadata`` Chroma key and must be matched in Python after fetching.
        """
        if not filters:
            return None, {}

        native_standard_keys = self._STANDARD_FIELDS - {'metadata', 'content'}

        # Logical operators at top level: collect all leaf keys; if any are
        # non-standard, fall back to a pure-Python post-filter for the whole tree.
        if any(k.startswith("$") for k in filters.keys()):
            def _leaf_keys(f: Any) -> set:
                keys: set = set()
                if isinstance(f, dict):
                    for k, v in f.items():
                        if k.startswith("$"):
                            if isinstance(v, list):
                                for item in v:
                                    keys |= _leaf_keys(item)
                            elif isinstance(v, dict):
                                keys |= _leaf_keys(v)
                        else:
                            keys.add(k)
                return keys

            all_keys = _leaf_keys(filters)
            if all_keys and all_keys <= native_standard_keys:
                return filters, {}
            return None, filters

        native_part: Dict[str, Any] = {}
        post_part: Dict[str, Any] = {}
        for key, value in filters.items():
            if key in native_standard_keys:
                native_part[key] = value
            else:
                post_part[key] = value

        native_chroma = self._convert_filters(native_part) if native_part else None
        return native_chroma, post_part

    def _matches_post_filter(
        self,
        payload: Dict[str, Any],
        post_filter: Dict[str, Any],
    ) -> bool:
        """
        Check if a hydrated payload matches a Python-side post-filter
        (non-standard fields stored inside the user metadata blob).
        """
        if not post_filter:
            return True

        user_meta = payload.get("metadata") if isinstance(payload, dict) else None
        if not isinstance(user_meta, dict):
            return False

        for key, expected in post_filter.items():
            if key.startswith("$"):
                if key == "$and":
                    if not all(self._matches_post_filter(payload, sub) for sub in expected):
                        return False
                elif key == "$or":
                    if not any(self._matches_post_filter(payload, sub) for sub in expected):
                        return False
                else:
                    return False
                continue

            actual = user_meta.get(key)
            if isinstance(expected, dict):
                for op, val in expected.items():
                    if op == "$eq":
                        if actual != val:
                            return False
                    elif op == "$ne":
                        if actual == val:
                            return False
                    elif op == "$in":
                        if actual not in val:
                            return False
                    elif op == "$nin":
                        if actual in val:
                            return False
                    elif op == "$gt":
                        if not (actual is not None and actual > val):
                            return False
                    elif op == "$gte":
                        if not (actual is not None and actual >= val):
                            return False
                    elif op == "$lt":
                        if not (actual is not None and actual < val):
                            return False
                    elif op == "$lte":
                        if not (actual is not None and actual <= val):
                            return False
                    else:
                        return False
            else:
                if actual != expected:
                    return False

        return True


    # ============================================================================
    # Data Operations (Async)
    # ============================================================================

    async def _batch_find_existing_by_hashes(
        self,
        chunk_content_hashes: List[str],
    ) -> Dict[str, List[str]]:
        """
        Batch lookup of existing records by chunk_content_hash.

        Uses ChromaDB ``$in`` filter for a single network roundtrip.

        Args:
            chunk_content_hashes: List of chunk content hashes to search for.

        Returns:
            Mapping of chunk_content_hash -> list of matching ChromaDB IDs.
        """
        if not chunk_content_hashes:
            return {}

        collection = await self._get_active_collection()
        unique_hashes: List[str] = list(set(chunk_content_hashes))

        try:
            where_filter: Dict[str, Any] = {"chunk_content_hash": {"$in": unique_hashes}}
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                include=["metadatas"],
            )

            hash_map: Dict[str, List[str]] = {}
            result_ids: List[str] = result.get("ids", [])
            result_metadatas: List[Dict[str, Any]] = result.get("metadatas", [])

            for idx, record_id in enumerate(result_ids):
                meta: Dict[str, Any] = result_metadatas[idx] if idx < len(result_metadatas) else {}
                h: str = meta.get("chunk_content_hash", "")
                if h:
                    hash_map.setdefault(h, []).append(record_id)

            return hash_map

        except Exception as e:
            debug_log(f"Batch hash lookup failed: {e}", context="ChromaVectorDB")
            return {}

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
        Adds new data or updates existing data in the collection.

        Uses chunk_content_hash for content-based deduplication: stale records
        (same content hash, different ID) are deleted before upserting.
        """
        if (vectors is None or len(vectors) == 0) and (sparse_vectors is None or len(sparse_vectors) == 0):
            info_log("Nothing to upsert: no dense or sparse vectors provided.", context="ChromaVectorDB")
            return

        n: int = len(vectors) if vectors else len(sparse_vectors)  # type: ignore[arg-type]

        # Strict per-item array length validation (Invariant #1)
        _length_checks = {
            "vectors": vectors,
            "payloads": payloads,
            "ids": ids,
            "chunks": chunks,
            "document_ids": document_ids,
            "document_names": document_names,
            "doc_content_hashes": doc_content_hashes,
            "chunk_content_hashes": chunk_content_hashes,
            "sparse_vectors": sparse_vectors,
            "knowledge_base_ids": knowledge_base_ids,
        }
        for _name, _arr in _length_checks.items():
            if _arr is not None and len(_arr) != n:
                raise UpsertError(
                    f"Length mismatch in aupsert: '{_name}' has {len(_arr)} items, expected {n}."
                )

        if sparse_vectors is not None:
            debug_log(
                "Sparse vectors are not supported by ChromaDB and will be ignored.",
                context="ChromaVectorDB",
            )

        collection = await self._get_active_collection()
        extra_metadata: Dict[str, Any] = metadata or {}
        _warned_standard_keys: set = set()

        computed_hashes: List[str] = []
        for i in range(n):
            chunk_text: str = chunks[i] if chunks and i < len(chunks) else ""
            if chunk_content_hashes and i < len(chunk_content_hashes):
                computed_hashes.append(chunk_content_hashes[i])
            else:
                computed_hashes.append(md5(chunk_text.encode("utf-8")).hexdigest())

        existing_hash_map: Dict[str, List[str]] = await self._batch_find_existing_by_hashes(
            computed_hashes
        )

        stale_ids: List[str] = []
        upsert_embeddings: List[List[float]] = []
        upsert_metadatas: List[Dict[str, Union[str, int, float, bool]]] = []
        upsert_ids: List[str] = []
        upsert_documents: List[str] = []

        for i in range(n):
            chunk_id_str: str = str(ids[i]) if ids and i < len(ids) else str(_uuid.uuid4())
            payload: Dict[str, Any] = payloads[i] if payloads and i < len(payloads) else {}
            doc_id: str = document_ids[i] if document_ids and i < len(document_ids) else ""
            doc_name: str = document_names[i] if document_names and i < len(document_names) else ""
            doc_hash: str = doc_content_hashes[i] if doc_content_hashes and i < len(doc_content_hashes) else ""
            chunk_hash: str = computed_hashes[i]
            content: str = chunks[i] if chunks and i < len(chunks) else ""

            old_ids: List[str] = existing_hash_map.get(chunk_hash, [])
            for old_id in old_ids:
                if old_id != chunk_id_str:
                    stale_ids.append(old_id)

            kbi: Optional[str] = knowledge_base_ids[i] if knowledge_base_ids else None
            prepared_metadata = self._prepare_metadata(
                payload=payload,
                chunk_id=chunk_id_str,
                chunk_content_hash=chunk_hash,
                document_id=doc_id,
                doc_content_hash=doc_hash,
                document_name=doc_name,
                knowledge_base_id=kbi,
                extra_metadata=extra_metadata,
                _warned_standard_keys=_warned_standard_keys,
            )

            if vectors and i < len(vectors):
                upsert_embeddings.append(vectors[i])
            upsert_metadatas.append(prepared_metadata)
            upsert_ids.append(chunk_id_str)

            cleaned_content: str = content.replace("\x00", "\ufffd")
            upsert_documents.append(cleaned_content)

        if stale_ids:
            try:
                unique_stale: List[str] = list(set(stale_ids))
                await asyncio.to_thread(collection.delete, ids=unique_stale)
                debug_log(
                    f"Deleted {len(unique_stale)} stale records with duplicate chunk_content_hash.",
                    context="ChromaVectorDB",
                )
            except Exception as e:
                debug_log(
                    f"Failed to delete stale duplicate records: {e}",
                    context="ChromaVectorDB",
                )

        try:
            await asyncio.to_thread(
                collection.upsert,
                embeddings=upsert_embeddings,
                metadatas=upsert_metadatas,
                ids=upsert_ids,
                documents=upsert_documents,
            )

            replaced_count: int = sum(1 for h in computed_hashes if h in existing_hash_map)
            new_count: int = len(upsert_ids) - replaced_count
            info_log(
                f"Successfully upserted {len(upsert_ids)} records "
                f"({new_count} new, {replaced_count} replaced by content hash).",
                context="ChromaVectorDB",
            )

        except Exception as e:
            raise UpsertError(
                f"Failed to upsert data into collection '{collection.name}': {e}"
            ) from e

    async def adelete(self, ids: List[Union[str, int]]) -> None:
        """
        Remove records from the collection by their unique identifiers.

        Pre-checks existence for each ID before attempting deletion.
        Non-existent IDs are skipped with a debug log, making this
        method idempotent and retry-safe.

        Args:
            ids: List of specific IDs to remove.

        Raises:
            VectorDBError: If the deletion fails or the collection is not initialized.
        """
        if not ids:
            debug_log(
                "Delete called with an empty list of IDs. No action taken.",
                context="ChromaVectorDB",
            )
            return

        collection = await self._get_active_collection()
        str_ids: List[str] = [str(i) for i in ids]

        try:
            existing_result = await asyncio.to_thread(
                collection.get,
                ids=str_ids,
            )
            existing_ids: List[str] = existing_result.get("ids", [])
            existing_set: set = set(existing_ids)

            for sid in str_ids:
                if sid not in existing_set:
                    debug_log(
                        f"Record with ID '{sid}' does not exist, skipping deletion.",
                        context="ChromaVectorDB",
                    )

            if not existing_ids:
                info_log(
                    "No matching records found to delete. No action taken.",
                    context="ChromaVectorDB",
                )
                return

            await asyncio.to_thread(collection.delete, ids=existing_ids)

            info_log(
                f"Successfully processed deletion request for {len(str_ids)} IDs. "
                f"Existed: {len(existing_ids)}, Deleted: {len(existing_ids)}.",
                context="ChromaVectorDB",
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete records from collection '{collection.name}': {e}"
            ) from e

    async def afield_exists(self, field_name: str, field_value: Any) -> bool:
        """
        Generic check if any record with the given metadata field value exists.

        Args:
            field_name: The metadata field name to filter on.
            field_value: The value to match.

        Returns:
            True if at least one matching record exists, False otherwise.
        """
        try:
            collection = await self._get_active_collection()
            where_filter = self._convert_filters({field_name: field_value})
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                limit=1,
            )
            return len(result.get("ids", [])) > 0
        except Exception as e:
            debug_log(f"Error checking if {field_name}='{field_value}' exists: {e}", context="ChromaVectorDB")
            return False

    async def adelete_by_field(self, field_name: str, field_value: Any) -> bool:
        """
        Generic deletion of all records matching a metadata field value.

        Includes a pre-existence check. Returns True if no matches found (idempotent).

        Args:
            field_name: The metadata field name to filter on.
            field_value: The value to match for deletion.

        Returns:
            True if deletion was successful or no matches found, False on failure.
        """
        try:
            collection = await self._get_active_collection()
            where_filter = self._convert_filters({field_name: field_value})

            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
            )

            ids_to_delete: List[str] = result.get("ids", [])

            if not ids_to_delete:
                debug_log(
                    f"No records with {field_name}='{field_value}' found. No action taken.",
                    context="ChromaVectorDB",
                )
                return True

            await asyncio.to_thread(collection.delete, ids=ids_to_delete)

            info_log(
                f"Deleted {len(ids_to_delete)} records with {field_name}='{field_value}'.",
                context="ChromaVectorDB",
            )
            return True

        except Exception as e:
            debug_log(
                f"Error deleting records by {field_name}='{field_value}': {e}",
                context="ChromaVectorDB",
            )
            return False

    # --- Document-name deletion ---

    async def adelete_by_document_name(self, document_name: str) -> bool:
        """Delete all records with the given document_name."""
        return await self.adelete_by_field("document_name", document_name)

    # --- Document-ID deletion ---

    async def adelete_by_document_id(self, document_id: str) -> bool:
        """Delete all records with the given document_id."""
        return await self.adelete_by_field("document_id", document_id)

    # --- Chunk-ID deletion ---

    async def adelete_by_chunk_id(self, chunk_id: str) -> bool:
        """Delete all records with the given chunk_id."""
        return await self.adelete_by_field("chunk_id", chunk_id)

    # --- Doc-content-hash deletion ---

    async def adelete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        """Delete all records with the given doc_content_hash."""
        return await self.adelete_by_field("doc_content_hash", doc_content_hash)

    # --- Chunk-content-hash deletion ---

    async def adelete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        """Delete all records with the given chunk_content_hash."""
        return await self.adelete_by_field("chunk_content_hash", chunk_content_hash)

    # --- Metadata deletion ---

    async def adelete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete all records matching the given metadata."""
        collection = await self._get_active_collection()
        try:
            native_filter, post_filter = self._split_filter(metadata)
            result = await asyncio.to_thread(
                collection.get,
                where=native_filter,
                include=["metadatas", "documents"],
            )
            ids_candidates: List[str] = result.get("ids", [])
            if not ids_candidates:
                return False
            if post_filter:
                metadatas = result.get("metadatas") or []
                documents = result.get("documents") or []
                ids_to_delete: List[str] = []
                for i, rid in enumerate(ids_candidates):
                    md = metadatas[i] if i < len(metadatas) else None
                    doc = documents[i] if i < len(documents) else None
                    payload = self._hydrate_payload(md, doc)
                    if self._matches_post_filter(payload, post_filter):
                        ids_to_delete.append(rid)
            else:
                ids_to_delete = ids_candidates
            if not ids_to_delete:
                return False
            await asyncio.to_thread(collection.delete, ids=ids_to_delete)
            return True
        except Exception as e:
            debug_log(f"Error deleting by metadata: {e}", context="ChromaVectorDB")
            return False

    async def afetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        """
        Retrieve full records from the collection by their IDs.
        
        Args:
            ids: List of IDs for which to retrieve the full records
            
        Returns:
            List of VectorSearchResult objects containing the fetched data.
            The order of results matches the order of the input IDs.
            
        Raises:
            VectorDBError: If fetching fails or the collection is not initialized
        """
        collection = await self._get_active_collection()
        debug_log(f"Fetching {len(ids)} records from collection '{collection.name}'...", context="ChromaVectorDB")
        
        try:
            results = await asyncio.to_thread(
                collection.get,
                ids=[str(i) for i in ids],
                include=["metadatas", "embeddings", "documents"]
            )
            
            # Build results map for efficient lookup
            results_map = {
                results['ids'][i]: {
                    "payload": self._hydrate_payload(
                        results['metadatas'][i] if results.get('metadatas') is not None else None,
                        results['documents'][i] if results.get('documents') is not None else None,
                    ),
                    "vector": results['embeddings'][i] if results['embeddings'] is not None else None,
                    "text": results['documents'][i] if results['documents'] is not None else None
                }
                for i in range(len(results['ids']))
            }
            
            # Build final results in the order of input IDs
            final_results = []
            for an_id in ids:
                str_id = str(an_id)
                if str_id in results_map:
                    final_results.append(
                        VectorSearchResult(
                            id=str_id,
                            score=1.0,  # Fetch doesn't have scores
                            payload=results_map[str_id]["payload"],
                            vector=results_map[str_id]["vector"],
                            text=results_map[str_id]["text"]
                        )
                    )
            
            info_log(f"Successfully fetched {len(final_results)} records.", context="ChromaVectorDB")
            return final_results
            
        except Exception as e:
            raise VectorDBError(f"Failed to fetch records from collection '{collection.name}': {e}") from e

    # ============================================================================
    # Existence Checks
    # ============================================================================

    async def aid_exists(self, id: Union[str, int]) -> bool:
        """Check if a record with the given ChromaDB ID exists."""
        collection = await self._get_active_collection()
        try:
            result = await asyncio.to_thread(collection.get, ids=[str(id)])
            return len(result.get("ids", [])) > 0
        except Exception as e:
            debug_log(f"Error checking if ID '{id}' exists: {e}", context="ChromaVectorDB")
            return False

    async def adocument_name_exists(self, document_name: str) -> bool:
        """Check if any record with the given document_name exists."""
        return await self.afield_exists("document_name", document_name)

    async def adocument_id_exists(self, document_id: str) -> bool:
        """Check if any record with the given document_id exists."""
        return await self.afield_exists("document_id", document_id)

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        """Check if any record with the given chunk_id exists."""
        return await self.afield_exists("chunk_id", chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        """Check if any record with the given doc_content_hash exists."""
        return await self.afield_exists("doc_content_hash", doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        """Check if any record with the given chunk_content_hash exists."""
        return await self.afield_exists("chunk_content_hash", chunk_content_hash)

    # ============================================================================
    # Metadata Management
    # ============================================================================

    async def aupdate_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata JSON field for the record with the given chunk_id.

        Merges the provided metadata into the existing ``metadata`` JSON field
        of the matching record.  Since chunk_id is unique, at most one record
        is updated.

        Args:
            chunk_id: The chunk ID to update.
            metadata: Metadata fields to merge into the existing metadata.

        Returns:
            True if update was successful, False otherwise.

        Raises:
            VectorDBError: If the update operation fails critically.
        """
        if not metadata:
            raise ValueError("'metadata' must be provided")

        collection = await self._get_active_collection()

        try:
            where_filter = self._convert_filters({"chunk_id": chunk_id})
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                include=["metadatas"],
                limit=1,
            )

            target_ids: List[str] = result.get("ids", [])
            current_metadatas: List[Dict[str, Any]] = result.get("metadatas", [])

            if not target_ids:
                debug_log(f"No record found with chunk_id '{chunk_id}'.", context="ChromaVectorDB")
                return False

            current_meta: Dict[str, Any] = dict(current_metadatas[0]) if current_metadatas else {}

            existing_metadata: Dict[str, Any] = {}
            metadata_raw: Any = current_meta.get("metadata", "{}")
            if isinstance(metadata_raw, str):
                try:
                    existing_metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    existing_metadata = {}
            elif isinstance(metadata_raw, dict):
                existing_metadata = metadata_raw

            existing_metadata.update(metadata)
            current_meta["metadata"] = json.dumps(existing_metadata) if existing_metadata else "{}"

            await asyncio.to_thread(
                collection.update,
                ids=[target_ids[0]],
                metadatas=[current_meta],
            )

            info_log(f"Updated metadata for chunk_id '{chunk_id}'.", context="ChromaVectorDB")
            return True

        except Exception as e:
            raise VectorDBError(f"Failed to update metadata for chunk_id '{chunk_id}': {e}") from e

    async def aget_count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Get the count of documents in the collection.
        
        Args:
            filter: Optional filter to count specific documents
            
        Returns:
            Number of documents
        """
        collection = await self._get_active_collection()
        
        try:
            if filter:
                native_filter, post_filter = self._split_filter(filter)
                result = await asyncio.to_thread(
                    collection.get,
                    where=native_filter,
                    include=["metadatas", "documents"],
                )
                ids = result.get("ids", [])
                if not post_filter:
                    return len(ids)
                metadatas = result.get("metadatas") or []
                documents = result.get("documents") or []
                count = 0
                for i in range(len(ids)):
                    md = metadatas[i] if i < len(metadatas) else None
                    doc = documents[i] if i < len(documents) else None
                    payload = self._hydrate_payload(md, doc)
                    if self._matches_post_filter(payload, post_filter):
                        count += 1
                return count
            else:
                return await asyncio.to_thread(collection.count)
        except Exception as e:
            debug_log(f"Error getting count: {e}", context="ChromaVectorDB")
            return 0

    def get_count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        return self._run_async_from_sync(self.aget_count(filter))
    
    # ============================================================================
    # Optimization
    # ============================================================================
    
    async def aoptimize(self) -> bool:
        """Optimize the vector database. ChromaDB doesn't require explicit optimization."""
        return True
    
    # ============================================================================
    # Search Type Support
    # ============================================================================
    
    async def aget_supported_search_types(self) -> List[str]:
        """Get the supported search types for ChromaDB."""
        supported: List[str] = []
        if self._config.dense_search_enabled:
            supported.append("dense")
        if self._config.full_text_search_enabled:
            supported.append("full_text")
        if self._config.hybrid_search_enabled:
            supported.append("hybrid")
        return supported

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
        Master search method that dispatches to appropriate search type.
        
        Args:
            top_k: Number of results to return
            query_vector: Dense vector for semantic search
            query_text: Text for full-text search
            filter: Metadata filter
            alpha: Hybrid search weighting (0=text only, 1=vector only)
            fusion_method: Fusion algorithm ('rrf' or 'weighted')
            similarity_threshold: Minimum similarity score
            apply_reranking: Whether to apply reranking to results
            sparse_query_vector: Optional sparse vector for hybrid search
            
        Returns:
            List of VectorSearchResult objects
            
        Raises:
            ConfigurationError: If requested search type is disabled
            SearchError: If search fails
        """
        final_top_k = top_k if top_k is not None else self._config.default_top_k
        
        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None
        
        if is_hybrid:
            if not self._config.hybrid_search_enabled:
                raise ConfigurationError("Hybrid search is disabled.")
            return await self.ahybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=final_top_k,
                filter=filter,
                alpha=alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )
        elif is_dense:
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled.")
            return await self.adense_search(
                query_vector=query_vector,
                top_k=final_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
            )
        elif is_full_text:
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled.")
            return await self.afull_text_search(
                query_text=query_text,
                top_k=final_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )
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
        """
        Perform pure vector similarity search.
        
        Args:
            query_vector: Dense vector embedding
            top_k: Number of results
            filter: Metadata filter
            similarity_threshold: Minimum similarity score
            apply_reranking: Whether to apply reranking to results
            
        Returns:
            List of VectorSearchResult objects sorted by similarity
            
        Raises:
            SearchError: If search fails
        """
        _ = apply_reranking  # accepted for API parity; not applied in dense path
        collection = await self._get_active_collection()

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5

        try:
            native_filter, post_filter = self._split_filter(filter)
            fetch_n = top_k * 5 if post_filter else top_k

            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_vector],
                n_results=fetch_n,
                where=native_filter,
                include=["metadatas", "distances", "embeddings", "documents"]
            )
            
            ids = results['ids'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            vectors = results['embeddings'][0] if results['embeddings'] else [None] * len(ids)
            chunks = results['documents'][0] if results['documents'] else [None] * len(ids)
            
            # Convert distances to scores
            max_dist = max(distances) if distances else 1.0
            
            filtered_results = []
            for i in range(len(ids)):
                # Score calculation based on distance metric
                if self._config.distance_metric == DistanceMetric.COSINE:
                    score = 1 - distances[i]
                elif self._config.distance_metric == DistanceMetric.EUCLIDEAN:
                    score = min(1.0, max(0.0, 1 - distances[i] / max_dist if max_dist > 0 else 1.0))
                elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
                    score = distances[i]
                else:
                    score = 1 - distances[i]
                
                if score >= final_similarity_threshold:
                    payload = self._hydrate_payload(metadatas[i], chunks[i])
                    if post_filter and not self._matches_post_filter(payload, post_filter):
                        continue
                    filtered_results.append(
                        VectorSearchResult(
                            id=ids[i],
                            score=score,
                            payload=payload,
                            vector=vectors[i],
                            text=chunks[i]
                        )
                    )
                    if len(filtered_results) >= top_k:
                        break

            return filtered_results
            
        except Exception as e:
            raise SearchError(f"An error occurred during dense search: {e}") from e

    async def afull_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """
        Perform full-text search using ChromaDB's document filtering.
        
        Args:
            query_text: Text query
            top_k: Number of results
            filter: Metadata filter
            similarity_threshold: Minimum relevance score
            apply_reranking: Whether to apply reranking to results
            sparse_query_vector: Optional sparse vector (unused in ChromaDB)
            
        Returns:
            List of VectorSearchResult objects sorted by relevance
            
        Raises:
            SearchError: If search fails
        """
        _ = (apply_reranking, sparse_query_vector)  # accepted for API parity; ChromaDB uses BM25-like scoring with no sparse vectors
        collection = await self._get_active_collection()

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5

        where_document_filter = {"$contains": query_text}
        native_filter, post_filter = self._split_filter(filter)

        try:
            fetch_limit = top_k * 5 if post_filter else top_k * 2
            # Use where_document parameter for document filtering in Chroma
            results = await asyncio.to_thread(
                collection.get,
                where=native_filter,
                where_document=where_document_filter,
                limit=fetch_limit,  # Get more to account for filtering
                include=["metadatas", "embeddings", "documents"]
            )
            
            # Calculate relevance scores using BM25-inspired algorithm
            query_terms = query_text.lower().split()
            scored_results = []
            
            for i in range(len(results['ids'])):
                document_text = results['documents'][i] if results['documents'] else ""
                if not document_text:
                    continue
                if post_filter:
                    md_i = results['metadatas'][i] if results.get('metadatas') else None
                    payload_i = self._hydrate_payload(md_i, document_text)
                    if not self._matches_post_filter(payload_i, post_filter):
                        continue
                
                doc_lower = document_text.lower()
                doc_words = doc_lower.split()
                
                # Calculate term frequency
                term_count = 0
                for term in query_terms:
                    term_count += doc_lower.count(term)
                
                # Calculate BM25-inspired relevance score
                doc_length = len(doc_words)
                if doc_length > 0 and term_count > 0:
                    # Match ratio: fraction of query terms that appear in document
                    matched_terms = sum(1 for term in query_terms if term in doc_lower)
                    match_ratio = matched_terms / len(query_terms)
                    
                    # Term density with logarithmic scaling
                    term_density = term_count / doc_length
                    tf_score = math.log(1 + term_density * 5) / math.log(6)  # Normalize to ~0-1 range
                    
                    # Final score: 70% TF density + 30% match coverage
                    score = (tf_score * 0.7 + match_ratio * 0.3)
                    score = min(1.0, max(0.0, score))
                else:
                    score = 0.0
                
                if score >= final_similarity_threshold:
                    scored_results.append((score, i))
            
            # Sort by score and return top_k
            scored_results.sort(key=lambda x: x[0], reverse=True)
            scored_results = scored_results[:top_k]
            
            filtered_results = []
            for score, i in scored_results:
                # Safely extract vector and text
                vector = results['embeddings'][i] if results.get('embeddings') is not None else None
                text = results['documents'][i] if results.get('documents') is not None else None
                
                filtered_results.append(
                    VectorSearchResult(
                        id=results['ids'][i],
                        score=score,
                        payload=self._hydrate_payload(results['metadatas'][i], text),
                        vector=vector,
                        text=text
                    )
                )
            
            return filtered_results
            
        except Exception as e:
            raise SearchError(f"An error occurred during full-text search: {e}") from e

    def _reciprocal_rank_fusion(self, results_lists: List[List[VectorSearchResult]], k: int = 60) -> dict:
        """
        Fuse multiple result lists using Reciprocal Rank Fusion (RRF).
        
        Args:
            results_lists: List of search result lists to fuse
            k: RRF constant (default 60)
            
        Returns:
            Dictionary mapping document IDs to fused scores
        """
        fused_scores = {}
        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = str(doc.id)
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank + 1)
        return fused_scores
    
    def _weighted_fusion(self, dense_results: List[VectorSearchResult], ft_results: List[VectorSearchResult], alpha: float) -> dict:
        """
        Fuse dense and full-text results using weighted scoring.
        
        Args:
            dense_results: Results from dense vector search
            ft_results: Results from full-text search
            alpha: Weight for dense results (0-1), full-text gets (1-alpha)
            
        Returns:
            Dictionary mapping document IDs to fused scores
        """
        fused_scores = {}
        
        # Add dense results with weight alpha
        for doc in dense_results:
            fused_scores[str(doc.id)] = doc.score * alpha
        
        # Add full-text results with weight (1 - alpha)
        for doc in ft_results:
            doc_id = str(doc.id)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += doc.score * (1 - alpha)
        
        return fused_scores

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
        """
        Perform hybrid search combining dense vector and full-text search.
        
        Args:
            query_vector: Dense vector for semantic search
            query_text: Text for keyword search
            top_k: Number of final results
            filter: Metadata filter
            alpha: Weighting factor (0=text only, 1=vector only)
            fusion_method: Algorithm for fusing results ('rrf' or 'weighted')
            similarity_threshold: Minimum similarity threshold
            apply_reranking: Whether to apply reranking to results
            sparse_query_vector: Optional sparse vector (unused in ChromaDB)
            
        Returns:
            List of VectorSearchResult objects with hybrid scores
            
        Raises:
            SearchError: If search fails
        """
        final_alpha = alpha if alpha is not None else self._config.default_hybrid_alpha or 0.5
        final_fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method or 'weighted'
        
        try:
            candidate_k = max(top_k * 2, 20)
            
            dense_results, ft_results = await asyncio.gather(
                self.adense_search(query_vector, candidate_k, filter, similarity_threshold, apply_reranking=apply_reranking),
                self.afull_text_search(query_text, candidate_k, filter, similarity_threshold, apply_reranking=apply_reranking, sparse_query_vector=sparse_query_vector)
            )
            
            # Fuse results
            fused_scores: dict
            if final_fusion_method == 'rrf':
                fused_scores = self._reciprocal_rank_fusion([dense_results, ft_results])
            elif final_fusion_method == 'weighted':
                fused_scores = self._weighted_fusion(dense_results, ft_results, final_alpha)
            else:
                raise ConfigurationError(f"Unknown fusion_method: {final_fusion_method}")
            
            # Get top_k document IDs
            reranked_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)[:top_k]
            
            if not reranked_ids:
                return []
            
            # Fetch full documents
            final_results = await self.afetch(ids=reranked_ids)
            
            # Update scores with fused scores
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
            
            # Sort by fused score
            updated_results.sort(key=lambda x: x.score, reverse=True)
            
            return updated_results
            
        except Exception as e:
            raise SearchError(f"An error occurred during hybrid search: {e}") from e


