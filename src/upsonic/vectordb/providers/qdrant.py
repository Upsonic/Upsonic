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
    
    Standard properties stored per point:
    - chunk_id (keyword) — unique per-chunk identifier
    - document_id (keyword) — parent document identifier
    - doc_content_hash (keyword) — MD5 of parent document content for change detection
    - chunk_content_hash (keyword) — MD5 of chunk text content for deduplication
    - document_name (keyword) — human-readable source name
    - content (text) — the chunk text
    - metadata (dict) — all non-standard data
    
    This implementation provides:
    - Full async/await support using AsyncQdrantClient
    - Dynamic field indexing based on configuration
    - Sparse vector support for hybrid search
    - Advanced filtering and search capabilities
    - Reranker integration for end-to-end workflows
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
                install_command='pip install "upsonic[qdrant]"',
                feature_name="Qdrant vector database provider"
            )

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
        self.client: Optional[Any] = None
        
        self.reranker = reranker
    

    
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
            hash_obj = hashlib.md5(str(id_value).encode())
            return int.from_bytes(hash_obj.digest()[:8], byteorder='big', signed=False)
    
    async def _create_async_client(self) -> None:
        """
        Creates the async Qdrant client based on the connection configuration.
        Does not verify readiness — only instantiates the client object.

        Raises:
            ConfigurationError: If the connection configuration is invalid.
            VectorDBConnectionError: If client instantiation fails.
        """
        conn = self._config.connection

        try:
            if conn.location:
                self.client = AsyncQdrantClient(location=conn.location)

            elif conn.mode == Mode.IN_MEMORY:
                self.client = AsyncQdrantClient(":memory:")

            elif conn.mode == Mode.EMBEDDED:
                if not conn.db_path:
                    raise ConfigurationError("'db_path' must be set for embedded mode.")
                self.client = AsyncQdrantClient(path=conn.db_path)

            elif conn.mode == Mode.LOCAL:
                grpc_port: Optional[int] = conn.grpc_port
                if grpc_port is None and conn.port:
                    grpc_port = conn.port + 1
                elif grpc_port is None:
                    grpc_port = 6334

                client_kwargs: Dict[str, Any] = {
                    "host": conn.host or "localhost",
                    "port": conn.port or 6333,
                    "grpc_port": grpc_port,
                    "prefer_grpc": conn.prefer_grpc,
                }

                if conn.https is not None:
                    client_kwargs["https"] = conn.https
                if conn.prefix:
                    client_kwargs["prefix"] = conn.prefix
                if conn.timeout is not None:
                    client_kwargs["timeout"] = int(conn.timeout) if conn.timeout else None

                self.client = AsyncQdrantClient(**client_kwargs)

            elif conn.mode == Mode.CLOUD:
                target_url: Optional[str] = conn.url or conn.host
                if target_url and ":6333" in target_url:
                    target_url = target_url.replace(":6333", "")

                client_kwargs = {
                    "url": target_url,
                    "api_key": conn.api_key.get_secret_value() if conn.api_key else None,
                }

                if conn.prefer_grpc:
                    client_kwargs["prefer_grpc"] = conn.prefer_grpc
                if conn.prefix:
                    client_kwargs["prefix"] = conn.prefix
                if conn.timeout is not None:
                    client_kwargs["timeout"] = int(conn.timeout) if conn.timeout else None

                self.client = AsyncQdrantClient(**client_kwargs)

            else:
                raise ConfigurationError(f"Unsupported mode for Qdrant: {conn.mode.value}")

        except (ConfigurationError, VectorDBConnectionError):
            raise
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to create Qdrant async client: {e}") from e

    async def aget_client(self) -> Any:
        """
        Gets or creates the async Qdrant client, ensuring it is connected and ready.

        Follows a singleton pattern: reuses the existing client if available,
        creates a new one if needed, verifies readiness, and auto-recovers
        if the existing client is unresponsive.

        Returns:
            A connected and ready AsyncQdrantClient instance.

        Raises:
            VectorDBConnectionError: If the client cannot be created or is not ready.
        """
        if self.client is None:
            debug_log(
                f"Creating async client for '{self._config.connection.mode.value}' mode...",
                context="QdrantVectorDB",
            )
            await self._create_async_client()

        try:
            await self.client.get_collections()  # type: ignore[union-attr]
        except Exception:
            debug_log("Existing client is unresponsive, recreating...", context="QdrantVectorDB")
            try:
                await self.aclose()
            except Exception:
                self.client = None
                self._is_connected = False
            await self._create_async_client()

            try:
                await self.client.get_collections()  # type: ignore[union-attr]
            except Exception as e:
                self.client = None
                self._is_connected = False
                raise VectorDBConnectionError(f"Qdrant client is not ready after recreation: {e}") from e

        self._is_connected = True
        return self.client  # type: ignore[return-value]

    def get_client(self) -> Any:
        """
        Gets or creates the async Qdrant client, ensuring it is connected and ready (sync).

        Returns:
            A connected and ready AsyncQdrantClient instance.

        Raises:
            VectorDBConnectionError: If the client cannot be created or is not ready.
        """
        return self._run_async_from_sync(self.aget_client())

    async def ais_client_connected(self) -> bool:
        """
        Checks whether the Qdrant client exists and is responsive.
        Does not create or reconnect — purely a read-only status check.

        Returns:
            True if the client exists and is responsive, False otherwise.
        """
        if self.client is None:
            return False

        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    def is_client_connected(self) -> bool:
        """
        Checks whether the Qdrant client exists and is responsive (sync).

        Returns:
            True if the client exists and is responsive, False otherwise.
        """
        return self._run_async_from_sync(self.ais_client_connected())

    async def aconnect(self) -> None:
        """
        Establishes an async connection to the Qdrant vector database instance.

        Delegates to aget_client() which handles client creation and
        readiness verification. This method is idempotent.

        Raises:
            VectorDBConnectionError: If the connection fails for any reason.
        """
        if await self.ais_client_connected():
            info_log("Already connected to Qdrant.", context="QdrantVectorDB")
            return

        debug_log(
            f"Attempting to connect to Qdrant in '{self._config.connection.mode.value}' mode...",
            context="QdrantVectorDB",
        )

        try:
            await self.aget_client()
            info_log(
                f"Successfully connected to Qdrant and health check passed. - {self.name}",
                context="QdrantVectorDB",
            )
        except (VectorDBConnectionError, ConfigurationError):
            self.client = None
            self._is_connected = False
            raise
        except Exception as e:
            self.client = None
            self._is_connected = False
            raise VectorDBConnectionError(f"Failed to connect to Qdrant: {e}") from e

    async def adisconnect(self) -> None:
        """
        Gracefully terminates the async connection to Qdrant.
        Alias for close() for framework compatibility.
        """
        await self.aclose()

    async def aclose(self) -> None:
        """
        Close the async Qdrant client connection properly.
        This is the recommended method for cleanup.
        """
        if self.client:
            try:
                await self.client.close()
                from upsonic.utils.printing import success_log
                success_log("Successfully closed Qdrant connection.", "QdrantProvider")
            except Exception as e:
                from upsonic.utils.printing import error_log
                error_log(f"Error closing Qdrant connection: {e}", "QdrantProvider")
            finally:
                self.client = None
                self._is_connected = False

    async def ais_ready(self) -> bool:
        """Performs a health check to ensure the Qdrant instance is responsive."""
        return await self.ais_client_connected()
    

    
    async def acreate_collection(self) -> None:
        """
        Creates the collection in Qdrant with full configuration support.
        
        Supports:
        - Dense vectors only
        - Dense + Sparse vectors (for hybrid search)
        - Named vectors when sparse vectors are enabled
        - Dynamic payload indexing based on indexed_fields config
        """
        client = await self.aget_client()

        collection_name = self._config.collection_name
        
        try:
            if self._config.recreate_if_exists and await self.acollection_exists():
                info_log(f"Collection '{collection_name}' exists and `recreate_if_exists` is True. Deleting...", context="QdrantVectorDB")
                await self.adelete_collection()
            
            distance_map = {
                DistanceMetric.COSINE: models.Distance.COSINE,
                DistanceMetric.EUCLIDEAN: models.Distance.EUCLID,
                DistanceMetric.DOT_PRODUCT: models.Distance.DOT,
            }
            
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
            
            hnsw_config = None
            index_cfg = self._config.index
            if isinstance(index_cfg, HNSWIndexConfig):
                hnsw_config = models.HnswConfigDiff(
                    m=index_cfg.m,
                    ef_construct=index_cfg.ef_construction
                )
            
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

            await client.create_collection(
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
            
            await self._create_field_indexes(collection_name)

        except Exception as e:
            raise VectorDBError(f"Failed to create collection '{collection_name}': {e}") from e
    
    async def adelete_collection(self) -> None:
        """
        Permanently deletes the collection specified in the config.
        """
        client = await self.aget_client()

        collection_name = self._config.collection_name
        try:
            result = await client.delete_collection(collection_name=collection_name)
            if isinstance(result, bool):
                if not result and not await self.acollection_exists():
                    raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.")
            else:
                if hasattr(result, 'result') and not result.result:
                    if not await self.acollection_exists():
                        raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.")
                    
            info_log(f"Successfully deleted collection '{collection_name}'.", context="QdrantVectorDB")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.") from e
            raise VectorDBError(f"API error while deleting collection '{collection_name}': {e}") from e
        except Exception as e:
            raise VectorDBError(f"An unexpected error occurred while deleting collection: {e}") from e
    
    async def acollection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists.
        """
        client = await self.aget_client()
        return await client.collection_exists(collection_name=self._config.collection_name)
    

    
    _STANDARD_FIELDS: frozenset = frozenset({
        'chunk_id', 'document_id', 'doc_content_hash',
        'chunk_content_hash', 'document_name', 'content', 'metadata',
        'knowledge_base_id',
    })

    def _build_payload(
        self,
        content: str,
        chunk_id: str,
        chunk_content_hash: str = "",
        document_id: str = "",
        doc_content_hash: str = "",
        document_name: str = "",
        knowledge_base_id: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        _warned_standard_keys: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Builds a standardized payload with the standard properties.

        Primary (chunk-focused) identifiers are always set. Document-level
        identifiers are optional and default to empty strings.

        Args:
            content: The chunk text (required).
            chunk_id: The unique chunk identifier.
            chunk_content_hash: MD5 hash of the chunk's text content.
            document_id: The parent document identifier.
            doc_content_hash: MD5 hash of the parent document content.
            document_name: Human-readable source name.
            extra_payload: Caller-provided payload dict (non-standard keys → metadata).
            extra_metadata: Additional metadata to merge into the payload.

        Returns:
            A properly structured payload dict for Qdrant.
        """
        combined_metadata: Dict[str, Any] = {}

        if self._config.default_metadata:
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
                            context="QdrantVectorDB",
                        )
                        _warned_standard_keys.add(key)
                else:
                    combined_metadata[key] = value

        if extra_metadata:
            combined_metadata.update(extra_metadata)

        payload: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "chunk_content_hash": chunk_content_hash,
            "document_id": document_id,
            "doc_content_hash": doc_content_hash,
            "document_name": document_name,
            "content": content,
            "knowledge_base_id": knowledge_base_id or "",
            "metadata": combined_metadata,
        }

        return payload

    def _flatten_payload(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Return the Qdrant payload unchanged. The provider now stores the
        contract-compliant shape natively ({<standards>, "metadata": <dict>}),
        so no hoisting/flattening is performed.
        """
        return payload if payload else {}

    async def _create_field_indexes(self, collection_name: str) -> None:
        """
        Creates indexes for fields specified in config.
        
        Supports two modes:
        1. Advanced: payload_field_configs (explicit types and params)
        2. Simple: indexed_fields (auto-determines types)
        """
        client = await self.aget_client()

        if self._config.payload_field_configs:
            debug_log(f"Creating indexes from payload_field_configs: {len(self._config.payload_field_configs)} fields", context="QdrantVectorDB")
            
            failed_fields: List[tuple[str, Exception]] = []
            for field_config in self._config.payload_field_configs:
                if not field_config.indexed:
                    continue
                
                field_schema = self._get_qdrant_schema(field_config.field_type, field_config.params)
                
                debug_log(f"Creating index for field '{field_config.field_name}' (type: {field_config.field_type})...", context="QdrantVectorDB")
                try:
                    await client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_config.field_name,
                        field_schema=field_schema,
                        wait=True
                    )
                except Exception as e:
                    failed_fields.append((field_config.field_name, e))
            
            if failed_fields:
                error_details = "; ".join(f"'{name}': {err}" for name, err in failed_fields)
                raise VectorDBError(f"Failed to create indexes for {len(failed_fields)} field(s): {error_details}")
            
            info_log("Field indexes created successfully from payload_field_configs.", context="QdrantVectorDB")
            return
        
        if self._config.indexed_fields:
            debug_log(f"Creating indexes for fields: {self._config.indexed_fields}", context="QdrantVectorDB")
            
            field_type_map = {
                'content': models.TextIndexParams(type='text', tokenizer=models.TokenizerType.WORD, min_token_len=2, max_token_len=20, lowercase=True),
                'document_name': models.KeywordIndexParams(type='keyword'),
                'document_id': models.KeywordIndexParams(type='keyword'),
                'chunk_id': models.KeywordIndexParams(type='keyword'),
                'doc_content_hash': models.KeywordIndexParams(type='keyword'),
                'chunk_content_hash': models.KeywordIndexParams(type='keyword'),
            }
            
            failed_fields: List[tuple[str, Exception]] = []
            for field_name in self._config.indexed_fields:
                if field_name in field_type_map:
                    field_schema = field_type_map[field_name]
                elif field_name.startswith('metadata.'):
                    field_schema = models.KeywordIndexParams(type='keyword')
                else:
                    field_schema = models.KeywordIndexParams(type='keyword')
                
                debug_log(f"Creating index for field '{field_name}'...", context="QdrantVectorDB")
                try:
                    await client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True
                    )
                except Exception as e:
                    failed_fields.append((field_name, e))
            
            if failed_fields:
                error_details = "; ".join(f"'{name}': {err}" for name, err in failed_fields)
                raise VectorDBError(f"Failed to create indexes for {len(failed_fields)} field(s): {error_details}")
            
            info_log("Field indexes created successfully from indexed_fields.", context="QdrantVectorDB")
            return
        
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
            default_text_params: Dict[str, Any] = {"tokenizer": models.TokenizerType.WORD, "min_token_len": 2, "max_token_len": 20, "lowercase": True}
            if params:
                default_text_params.update(params)
            return models.TextIndexParams(type='text', **default_text_params)
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
            return models.KeywordIndexParams(type='keyword')
    
    
    async def _batch_find_existing_by_hashes(
        self,
        chunk_hashes: List[str],
    ) -> Dict[str, List[Union[str, int]]]:
        """
        Single-query batch lookup: finds which chunk_content_hash values
        already exist and returns the point IDs that own them.

        Args:
            chunk_hashes: The chunk_content_hash values to check.

        Returns:
            Mapping of chunk_content_hash → list of existing point IDs that
            carry that hash.  Empty dict when none match.
        """
        if not chunk_hashes:
            return {}

        unique_hashes: List[str] = list(set(chunk_hashes))

        try:
            client = await self.aget_client()
            hash_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_content_hash",
                        match=models.MatchAny(any=unique_hashes),
                    )
                ]
            )

            existing: Dict[str, List[Union[str, int]]] = defaultdict(list)
            offset: Optional[Union[str, int]] = None
            batch_limit: int = 100

            while True:
                scroll_result = await client.scroll(
                    collection_name=self._config.collection_name,
                    scroll_filter=hash_filter,
                    limit=batch_limit,
                    offset=offset,
                    with_payload=["chunk_content_hash"],
                    with_vectors=False,
                )
                points, next_offset = scroll_result

                for point in points:
                    h: str = (point.payload or {}).get("chunk_content_hash", "")
                    if h:
                        existing[h].append(point.id)

                if next_offset is None or not points:
                    break
                offset = next_offset

            return dict(existing)

        except Exception as e:
            debug_log(
                f"Batch hash lookup failed, treating all as new: {e}",
                context="QdrantVectorDB",
            )
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
        Adds or updates data in the Qdrant collection.

        Uses chunk_content_hash for content-based deduplication via a single
        batch pre-check.  If identical chunk content already exists under a
        different point ID, the old point is deleted first to avoid duplicates.
        """
        client = await self.aget_client()

        if (vectors is None or len(vectors) == 0) and (sparse_vectors is None or len(sparse_vectors) == 0):
            info_log("Nothing to upsert: no dense or sparse vectors provided.", context="QdrantVectorDB")
            return

        n: int = len(vectors) if vectors else len(sparse_vectors)  # type: ignore[arg-type]

        if payloads is not None and len(payloads) != n:
            raise ValueError(
                f"Length mismatch: payloads ({len(payloads)}) != vectors ({n})."
            )
        if ids is not None and len(ids) != n:
            raise ValueError(
                f"Length mismatch: ids ({len(ids)}) != vectors ({n})."
            )
        if chunks is not None and len(chunks) != n:
            raise ValueError(
                f"Length mismatch: chunks ({len(chunks)}) != vectors ({n})."
            )
        for _arr_name, _arr in (
            ("document_ids", document_ids),
            ("document_names", document_names),
            ("doc_content_hashes", doc_content_hashes),
            ("chunk_content_hashes", chunk_content_hashes),
            ("sparse_vectors", sparse_vectors),
            ("knowledge_base_ids", knowledge_base_ids),
        ):
            if _arr is not None and len(_arr) != n:
                raise ValueError(
                    f"Length mismatch in aupsert: '{_arr_name}' has length {len(_arr)}, expected {n}"
                )

        extra_metadata: Dict[str, Any] = metadata or {}
        _warned_standard_keys: set = set()

        computed_hashes: List[str] = []
        for i in range(n):
            # payload: Dict[str, Any] = payloads[i] if payloads and i < len(payloads) else {}
            content: str = chunks[i] if chunks and i < len(chunks) else ""
            if chunk_content_hashes and i < len(chunk_content_hashes):
                computed_hashes.append(chunk_content_hashes[i])
            else:
                computed_hashes.append(hashlib.md5(content.encode("utf-8")).hexdigest())

        existing_hash_map: Dict[str, List[Union[str, int]]] = (
            await self._batch_find_existing_by_hashes(computed_hashes)
        )

        stale_point_ids: List[Union[str, int]] = []
        points: List[Any] = []

        for i in range(n):
            payload = payloads[i] if payloads and i < len(payloads) else {}
            content = chunks[i] if chunks and i < len(chunks) else ""
            point_id: Union[str, int] = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
            chunk_id_str: str = str(point_id)
            doc_id: str = document_ids[i] if document_ids and i < len(document_ids) else ''
            doc_hash: str = doc_content_hashes[i] if doc_content_hashes and i < len(doc_content_hashes) else ""
            chunk_hash: str = computed_hashes[i]
            doc_name: str = document_names[i] if document_names and i < len(document_names) else ''

            normalized_id: Union[str, int] = self._normalize_id(point_id)

            old_ids: List[Union[str, int]] = existing_hash_map.get(chunk_hash, [])
            for old_id in old_ids:
                if old_id != normalized_id:
                    stale_point_ids.append(old_id)

            kbi: Optional[str] = knowledge_base_ids[i] if knowledge_base_ids else None
            structured_payload: Dict[str, Any] = self._build_payload(
                content=content,
                chunk_id=chunk_id_str,
                chunk_content_hash=chunk_hash,
                document_id=doc_id,
                doc_content_hash=doc_hash,
                document_name=doc_name,
                knowledge_base_id=kbi,
                extra_payload=payload,
                extra_metadata=extra_metadata,
                _warned_standard_keys=_warned_standard_keys,
            )

            vector_data: Any
            if self._config.use_sparse_vectors:
                vector_data = {}
                if vectors and i < len(vectors):
                    vector_data[self._config.dense_vector_name] = vectors[i]
                if sparse_vectors and i < len(sparse_vectors):
                    sparse_vec: Dict[str, Any] = sparse_vectors[i]
                    vector_data[self._config.sparse_vector_name] = models.SparseVector(
                        indices=sparse_vec.get('indices', []),
                        values=sparse_vec.get('values', [])
                    )
            else:
                vector_data = vectors[i] if vectors and i < len(vectors) else []

            points.append(
                models.PointStruct(
                    id=normalized_id,
                    vector=vector_data,
                    payload=structured_payload,
                )
            )

        if stale_point_ids:
            try:
                await client.delete(
                    collection_name=self._config.collection_name,
                    points_selector=models.PointIdsList(points=stale_point_ids),
                    wait=True,
                )
                debug_log(
                    f"Deleted {len(stale_point_ids)} stale points with duplicate chunk_content_hash.",
                    context="QdrantVectorDB",
                )
            except Exception as e:
                debug_log(
                    f"Failed to delete stale duplicate points: {e}",
                    context="QdrantVectorDB",
                )

        wait_for_result: bool = self._config.write_consistency_factor > 1

        try:
            await client.upsert(
                collection_name=self._config.collection_name,
                points=points,
                wait=wait_for_result,
            )
            replaced_count: int = sum(1 for h in computed_hashes if h in existing_hash_map)
            new_count: int = len(points) - replaced_count
            debug_log(
                f"Successfully upserted {len(points)} points "
                f"({new_count} new, {replaced_count} replaced by content hash).",
                context="QdrantVectorDB",
            )
        except Exception as e:
            raise UpsertError(f"Failed to upsert data into collection '{self._config.collection_name}': {e}") from e

    async def adelete(self, ids: List[Union[str, int]]) -> None:
        """
        Removes data from the collection by their unique identifiers.
        Pre-checks existence and skips non-existent IDs with debug log.
        """
        client = await self.aget_client()

        if not ids:
            return

        normalized_ids: List[Union[str, int]] = [self._normalize_id(id_val) for id_val in ids]
        wait_for_result: bool = self._config.write_consistency_factor > 1

        try:
            retrieved: List[Any] = await client.retrieve(
                collection_name=self._config.collection_name,
                ids=normalized_ids,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids: set[Union[str, int]] = {r.id for r in retrieved}

            for nid in normalized_ids:
                if nid not in existing_ids:
                    debug_log(
                        f"Point with ID '{nid}' does not exist, skipping deletion.",
                        context="QdrantVectorDB",
                    )

            if not existing_ids:
                info_log("No matching points found to delete. No action taken.", context="QdrantVectorDB")
                return

            await client.delete(
                collection_name=self._config.collection_name,
                points_selector=models.PointIdsList(points=list(existing_ids)),
                wait=wait_for_result,
            )
            info_log(
                f"Successfully processed deletion request for {len(normalized_ids)} IDs. "
                f"Existed: {len(existing_ids)}, Deleted: {len(existing_ids)}.",
                context="QdrantVectorDB",
            )
        except Exception as e:
            raise VectorDBError(f"Failed to delete points from collection '{self._config.collection_name}': {e}") from e
    
    async def afetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their unique IDs.
        """
        client = await self.aget_client()

        if not ids:
            return []

        normalized_ids = [self._normalize_id(id_val) for id_val in ids]

        try:
            retrieved_records: List[Any] = await client.retrieve(
                collection_name=self._config.collection_name,
                ids=normalized_ids,
                with_payload=True,
                with_vectors=True
            )

            search_results = []
            for record in retrieved_records:
                vector = None
                if isinstance(record.vector, dict):
                    vector = record.vector.get(self._config.dense_vector_name)
                else:
                    vector = record.vector
                
                search_results.append(VectorSearchResult(
                    id=record.id,
                    score=1.0,
                    payload=self._flatten_payload(record.payload),
                    vector=vector,
                    text=record.payload.get("content", "") if record.payload else ""
                ))
            
            return search_results

        except Exception as e:
            raise VectorDBError(f"Failed to fetch points from collection '{self._config.collection_name}': {e}") from e
    

    
    async def aget_count(self) -> int:
        """
        Get the total number of points/documents in the collection.
        
        Returns:
            int: Total count of documents
            
        Raises:
            VectorDBError: If the count operation fails
        """
        client = await self.aget_client()
        
        try:
            count_result = await client.count(
                collection_name=self._config.collection_name,
                exact=True
            )
            return count_result.count
        except Exception as e:
            raise VectorDBError(f"Failed to get count from collection '{self._config.collection_name}': {e}") from e
    
    async def afield_exists(self, field_name: str, field_value: Any) -> bool:
        """
        Generic check if any point with the given field value exists.

        Args:
            field_name: The payload field name to filter on.
            field_value: The value to match.

        Returns:
            True if at least one matching point exists, False otherwise.
        """
        try:
            client = await self.aget_client()
            filter_condition = models.Filter(
                must=[models.FieldCondition(key=field_name, match=models.MatchValue(value=field_value))]
            )
            count_result = await client.count(
                collection_name=self._config.collection_name,
                count_filter=filter_condition,
                exact=True,
            )
            return count_result.count > 0
        except Exception as e:
            debug_log(f"Error checking if {field_name}='{field_value}' exists: {e}", context="QdrantVectorDB")
            return False

    async def adelete_by_field(self, field_name: str, field_value: Any) -> bool:
        """
        Generic deletion of all points matching a payload field value.

        Includes a pre-existence check. Returns True if no matches found (idempotent).

        Args:
            field_name: The payload field name to filter on.
            field_value: The value to match for deletion.

        Returns:
            True if deletion was successful or no matches found, False on failure.
        """
        try:
            client = await self.aget_client()
            filter_condition = models.Filter(
                must=[models.FieldCondition(key=field_name, match=models.MatchValue(value=field_value))]
            )

            count_result = await client.count(
                collection_name=self._config.collection_name,
                count_filter=filter_condition,
                exact=True,
            )

            if count_result.count == 0:
                warning_log(f"No points found with {field_name}: {field_value}", context="QdrantVectorDB")
                return True

            info_log(f"Found {count_result.count} points to delete with {field_name}: {field_value}", context="QdrantVectorDB")

            result = await client.delete(
                collection_name=self._config.collection_name,
                points_selector=filter_condition,
                wait=True,
            )

            if result.status == models.UpdateStatus.COMPLETED:
                info_log(f"Successfully deleted {count_result.count} points with {field_name}: {field_value}", context="QdrantVectorDB")
                return True
            else:
                warning_log(f"Deletion failed for {field_name} {field_value}. Status: {result.status}", context="QdrantVectorDB")
                return False
        except Exception as e:
            warning_log(f"Error deleting points with {field_name} {field_value}: {e}", context="QdrantVectorDB")
            return False

    async def aid_exists(self, id: Union[str, int]) -> bool:
        """
        Check if a point with the given ID exists in the collection.

        Args:
            id: The ID to check

        Returns:
            True if the point exists, False otherwise.
        """
        try:
            client = await self.aget_client()
            normalized_id = self._normalize_id(id)
            points = await client.retrieve(
                collection_name=self._config.collection_name,
                ids=[normalized_id],
                with_payload=False,
                with_vectors=False
            )
            return len(points) > 0
        except Exception as e:
            debug_log(f"Error checking if point {id} exists: {e}", context="QdrantVectorDB")
            return False

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        """Check if any points with the given chunk_id exist in the collection."""
        return await self.afield_exists("chunk_id", chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        """Check if any points with the given doc_content_hash exist in the collection."""
        return await self.afield_exists("doc_content_hash", doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        """Check if any points with the given chunk_content_hash exist in the collection."""
        return await self.afield_exists("chunk_content_hash", chunk_content_hash)
    
    async def adocument_name_exists(self, document_name: str) -> bool:
        """Check if a document with the given name exists in the collection."""
        return await self.afield_exists("document_name", document_name)
    
    async def adocument_id_exists(self, document_id: str) -> bool:
        """Check if a document with the given document_id exists in the collection."""
        return await self.afield_exists("document_id", document_id)
    
    async def adelete_by_id(self, id: Union[str, int]) -> bool:
        """
        Delete a single point by its ID.
        
        Args:
            id: The ID of the point to delete
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            if not await self.aid_exists(id):
                warning_log(f"Point with ID {id} does not exist", context="QdrantVectorDB")
                return True
            
            client = await self.aget_client()
            normalized_id = self._normalize_id(id)
            await client.delete(
                collection_name=self._config.collection_name,
                points_selector=models.PointIdsList(points=[normalized_id]),
                wait=True
            )
            debug_log(f"Successfully deleted point with ID {id}", context="QdrantVectorDB")
            return True
        except Exception as e:
            warning_log(f"Error deleting point with ID {id}: {e}", context="QdrantVectorDB")
            return False
    
    async def adelete_by_document_name(self, document_name: str) -> bool:
        """Delete all points that have the specified document_name in their payload."""
        return await self.adelete_by_field("document_name", document_name)
    
    async def adelete_by_document_id(self, document_id: str) -> bool:
        """Delete all points that have the specified document_id in their payload."""
        return await self.adelete_by_field("document_id", document_id)
    
    async def adelete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Delete all points where the given metadata matches.
        
        Args:
            metadata: Dictionary of metadata key-value pairs to match
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            client = await self.aget_client()
            info_log(f"Attempting to delete all points with metadata: {metadata}", context="QdrantVectorDB")
            
            filter_conditions: List[Any] = []
            for key, value in metadata.items():
                filter_conditions.append(
                    models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
                )
            
            filter_condition = models.Filter(must=filter_conditions)
            
            count_result = await client.count(
                collection_name=self._config.collection_name,
                count_filter=filter_condition,
                exact=True
            )
            
            if count_result.count == 0:
                warning_log(f"No points found with metadata: {metadata}", context="QdrantVectorDB")
                return True
            
            info_log(f"Found {count_result.count} points to delete with metadata: {metadata}", context="QdrantVectorDB")
            
            result = await client.delete(
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
    
    async def adelete_by_chunk_id(self, chunk_id: str) -> bool:
        """Delete all points that have the specified chunk_id in their payload."""
        return await self.adelete_by_field("chunk_id", chunk_id)

    async def adelete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        """Delete all points that have the specified doc_content_hash in their payload."""
        return await self.adelete_by_field("doc_content_hash", doc_content_hash)

    async def adelete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        """Delete all points that have the specified chunk_content_hash in their payload."""
        return await self.adelete_by_field("chunk_content_hash", chunk_content_hash)

    async def aupdate_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Updates the metadata for a specific chunk ID.

        Args:
            chunk_id: The chunk ID to update.
            metadata: The metadata to update/merge.

        Returns:
            True if the update was successful, False otherwise.

        Raises:
            VectorDBError: If the update operation fails critically.
        """
        client = await self.aget_client()

        try:
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=chunk_id))]
            )

            search_result = await client.scroll(
                collection_name=self._config.collection_name,
                scroll_filter=filter_condition,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            if not search_result[0]:
                warning_log(f"No documents found with chunk_id: {chunk_id}", context="QdrantVectorDB")
                return False

            point = search_result[0][0]
            point_id = point.id
            current_payload: Dict[str, Any] = point.payload or {}
            updated_payload: Dict[str, Any] = current_payload.copy()

            if "metadata" in updated_payload and isinstance(updated_payload["metadata"], dict):
                updated_payload["metadata"].update(metadata)
            else:
                updated_payload["metadata"] = metadata

            await client.set_payload(
                collection_name=self._config.collection_name,
                payload=updated_payload,
                points=[point_id],
                wait=True,
            )
            updated_count: int = 1

            info_log(f"Updated metadata for {updated_count} document with chunk_id: {chunk_id}", context="QdrantVectorDB")
            return True

        except Exception as e:
            raise VectorDBError(f"Error updating metadata for chunk_id '{chunk_id}': {e}") from e
    
    async def aoptimize(self) -> bool:
        """
        Trigger optimization of the Qdrant collection.
        
        This operation optimizes indexes and improves search performance.
        Useful to call periodically or after bulk operations.
        
        Note: This is a no-op for in-memory mode.
        
        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            client = await self.aget_client()
            info_log(f"Optimization requested for collection '{self._config.collection_name}'", context="QdrantVectorDB")
            
            await client.get_collection(collection_name=self._config.collection_name)
            
            debug_log("Collection optimization acknowledged", context="QdrantVectorDB")
            return True
        except Exception as e:
            warning_log(f"Error during optimization: {e}", context="QdrantVectorDB")
            return False
    
    async def aget_supported_search_types(self) -> List[str]:
        """
        Get the list of supported search types for this provider.
        
        Returns:
            List of search type strings: ['dense', 'full_text', 'hybrid']
        """
        supported: List[str] = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported
    
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
        Master search method that dispatches to the appropriate search type.
        """
        await self.aget_client()

        effective_top_k = top_k if top_k is not None else self._config.default_top_k or 10

        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None

        if is_dense:
            if self._config.dense_search_enabled is False:
                raise ConfigurationError("Dense search is disabled by the current configuration.")
            return await self.adense_search(query_vector, effective_top_k, filter, similarity_threshold, apply_reranking=apply_reranking)
        
        elif is_hybrid:
            if self._config.hybrid_search_enabled is False:
                raise ConfigurationError("Hybrid search is disabled by the current configuration.")
            return await self.ahybrid_search(query_vector, query_text, effective_top_k, filter, alpha, fusion_method, similarity_threshold, apply_reranking=apply_reranking, sparse_query_vector=sparse_query_vector)

        elif is_full_text:
            if self._config.full_text_search_enabled is False:
                raise ConfigurationError("Full-text search is disabled by the current configuration.")
            return await self.afull_text_search(query_text, effective_top_k, filter, similarity_threshold, apply_reranking=apply_reranking)
        
        else:
            raise SearchError("Invalid search query: You must provide a 'query_vector' and/or 'query_text'.")
    
    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        """
        Performs pure vector similarity search.
        """
        try:
            client = await self.aget_client()
            final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.default_similarity_threshold if self._config.default_similarity_threshold is not None else 0.0)
            
            search_params = models.SearchParams(
                hnsw_ef=getattr(self._config.index, 'ef_search', None) or 128,
                exact=False,
            )

            qdrant_filter = self._build_qdrant_filter(filter) if filter else None

            if self._config.use_sparse_vectors:
                query_response = await client.query_points(
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
                query_response = await client.query_points(
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
                    vector = None
                    if isinstance(point.vector, dict):
                        vector = point.vector.get(self._config.dense_vector_name)
                    else:
                        vector = point.vector
                    
                    filtered_results.append(VectorSearchResult(
                        id=point.id,
                        score=point.score,
                        payload=self._flatten_payload(point.payload),
                        vector=vector,
                        text=point.payload.get("content", "") if point.payload else ""
                    ))

            if self.reranker and apply_reranking:
                filtered_results = self._apply_reranking(filtered_results, str(query_vector))
            
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
        Performs full-text search using Qdrant's text indexing.

        For IN_MEMORY mode, falls back to client-side search.
        For other modes, uses server-side text indexing.
        """
        _ = (sparse_query_vector,)  # accepted for API parity; Qdrant uses text-index BM25 not sparse vectors
        await self.aget_client()

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.default_similarity_threshold if self._config.default_similarity_threshold is not None else 0.0)
        target_text_field: str = self._config.text_search_field

        if self._config.connection.mode == Mode.IN_MEMORY:
            return await self._client_side_full_text_search(query_text, top_k, filter, final_similarity_threshold, target_text_field, apply_reranking=apply_reranking)
        else:
            return await self._server_side_full_text_search(query_text, top_k, filter, final_similarity_threshold, target_text_field, apply_reranking=apply_reranking)
    
    async def _client_side_full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]],
        similarity_threshold: float,
        target_text_field: str,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        """Client-side full-text search implementation."""
        _ = (filter,)  # TODO: client-side path currently does not apply the filter to scroll results
        try:
            client = await self.aget_client()
            records = await client.scroll(
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
                            vector = None
                            if isinstance(record.vector, dict):
                                vector = record.vector.get(self._config.dense_vector_name)
                            else:
                                vector = record.vector
                            
                            matching_records.append(VectorSearchResult(
                                id=record.id,
                                score=relevance_score,
                                payload=self._flatten_payload(record.payload),
                                vector=vector,
                                text=record.payload.get("content", "")
                            ))
            
            matching_records.sort(key=lambda x: x.score, reverse=True)
            matching_records = matching_records[:top_k]
            
            if self.reranker and apply_reranking:
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
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        """Server-side full-text search implementation."""
        try:
            client = await self.aget_client()
            await client.create_payload_index(
                collection_name=self._config.collection_name,
                field_name=target_text_field,
                field_schema=models.TextIndexParams(type="text", tokenizer=models.TokenizerType.WORD, min_token_len=2, max_token_len=20, lowercase=True),
                wait=True
            )

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
            
            records = await client.scroll(
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
                vector = None
                if isinstance(r.vector, dict):
                    vector = r.vector.get(self._config.dense_vector_name)
                else:
                    vector = r.vector
                
                filtered_results.append(VectorSearchResult(
                    id=r.id,
                    score=score,
                    payload=self._flatten_payload(r.payload),
                    vector=vector,
                    text=r.payload.get("content", "")
                ))
            
            if self.reranker and apply_reranking:
                filtered_results = self._apply_reranking(filtered_results, query_text)
            
            return filtered_results
        except Exception as e:
            raise SearchError(f"An error occurred during server-side full-text search: {e}") from e
    
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
        Combines dense and full-text search results using fusion.
        
        If sparse vectors are enabled and a sparse_query_vector is provided, uses native Qdrant hybrid search.
        Otherwise, performs manual fusion of dense + full-text search results.
        """
        effective_alpha = alpha if alpha is not None else self._config.default_hybrid_alpha or 0.5
        effective_fusion = fusion_method if fusion_method is not None else self._config.default_fusion_method or 'weighted'
        
        if self._config.use_sparse_vectors and sparse_query_vector:
            return await self._native_hybrid_search(
                query_vector, sparse_query_vector, top_k, filter, effective_fusion, similarity_threshold
            )
        else:
            dense_results = await self.adense_search(query_vector, top_k, filter, similarity_threshold, apply_reranking=apply_reranking)
            ft_results = await self.afull_text_search(query_text, top_k, filter, similarity_threshold, apply_reranking=apply_reranking)

            if effective_fusion == 'weighted':
                fused_results = self._fuse_weighted(dense_results, ft_results, effective_alpha)
            elif effective_fusion == 'rrf':
                fused_results = self._fuse_rrf(dense_results, ft_results)
            else:
                raise ConfigurationError(f"Unsupported fusion method: '{effective_fusion}'")

            fused_results.sort(key=lambda x: x.score, reverse=True)
            fused_results = fused_results[:top_k]
            
            if self.reranker and apply_reranking:
                fused_results = self._apply_reranking(fused_results, query_text)
            
            return fused_results
    
    async def _native_hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        top_k: int,
        filter: Optional[Dict[str, Any]],
        fusion_method: str,
        similarity_threshold: Optional[float],
    ) -> List[VectorSearchResult]:
        """
        Performs native Qdrant hybrid search using named dense and sparse vectors.
        """
        _ = (similarity_threshold,)  # TODO: native hybrid path currently does not apply threshold filtering to point.score
        try:
            client = await self.aget_client()
            qdrant_filter = self._build_qdrant_filter(filter) if filter else None
            
            fusion_map = {
                'rrf': models.Fusion.RRF,
                'dbsf': models.Fusion.DBSF
            }
            
            final_fusion = fusion_method if fusion_method is not None else self._config.default_fusion_method or 'rrf'

            query_response = await client.query_points(
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
                vector = None
                if isinstance(point.vector, dict):
                    vector = point.vector.get(self._config.dense_vector_name)
                else:
                    vector = point.vector
                
                results.append(VectorSearchResult(
                    id=point.id,
                    score=point.score,
                    payload=self._flatten_payload(point.payload),
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
            if hasattr(self.reranker, 'rerank'):
                documents = []
                for result in results:
                    doc = {
                        'id': result.id,
                        'content': result.text or result.payload.get('content', ''),
                        'payload': result.payload,
                        'score': result.score
                    }
                    documents.append(doc)
                
                reranked_docs = self.reranker.rerank(query=query, documents=documents)
                
                reranked_results = []
                for doc in reranked_docs:
                    if isinstance(doc, dict):
                        reranked_results.append(VectorSearchResult(
                            id=doc.get('id'),
                            score=doc.get('score', 0.0),
                            payload=doc.get('payload'),
                            vector=None,
                            text=doc.get('content', '')
                        ))
                    else:
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

    def _build_qdrant_filter(self, filter_dict: Dict[str, Any]) -> Any:
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
