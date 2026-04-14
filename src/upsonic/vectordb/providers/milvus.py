"""
Milvus Vector Database Provider

A comprehensive, high-level implementation supporting:
- Dense and sparse vectors for hybrid search
- Flexible metadata and field indexing
- Async-first operations with centralized client lifecycle
- Content-based deduplication via chunk_content_hash
- Two-layer dedup architecture (document-level + chunk-level)
- Compatible with Milvus 2.6+ API
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid as _uuid
from typing import Any, Dict, List, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from pymilvus import AsyncMilvusClient as _AsyncMilvusClient

try:
    from pymilvus import (
        AsyncMilvusClient,
        DataType,
        AnnSearchRequest,
        RRFRanker,
        WeightedRanker,
    )
    _MILVUS_AVAILABLE = True
except (ImportError, AttributeError):
    AsyncMilvusClient = None  # type: ignore
    DataType = None  # type: ignore
    AnnSearchRequest = None  # type: ignore
    RRFRanker = None  # type: ignore
    WeightedRanker = None  # type: ignore
    _MILVUS_AVAILABLE = False

from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.vectordb.config import MilvusConfig, Mode
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.utils.logging_config import get_logger
from upsonic.utils.printing import info_log, error_log, debug_log

logger = get_logger(__name__)


DISTANCE_METRIC_MAP: Dict[str, str] = {
    'Cosine': 'COSINE',
    'Euclidean': 'L2',
    'DotProduct': 'IP',
}


class MilvusProvider(BaseVectorDBProvider):
    """
    Milvus vector database provider with comprehensive feature support.

    Standard properties stored per record:
    - chunk_id (VARCHAR, primary key) — unique per-chunk identifier
    - chunk_content_hash (VARCHAR) — MD5 of chunk text for deduplication
    - content (VARCHAR) — the chunk text
    - metadata (VARCHAR, JSON) — all non-standard data
    - document_name (VARCHAR) — human-readable source name
    - document_id (VARCHAR) — parent document identifier
    - doc_content_hash (VARCHAR) — MD5 of parent document for change detection

    Features:
    - Centralized client lifecycle with auto-reconnect
    - Content-based deduplication via chunk_content_hash
    - Dense and sparse vector support for hybrid search
    - Flexible metadata management with custom fields
    - Configurable field indexing for optimized filtering
    - Multiple ranking strategies (RRF and Weighted)
    - Async-first operations for high performance
    - Batch processing support
    """

    _STANDARD_FIELDS: frozenset = frozenset({
        'chunk_id', 'document_id', 'doc_content_hash',
        'chunk_content_hash', 'document_name', 'content', 'metadata',
        'knowledge_base_id',
    })

    def __init__(self, config: Union[MilvusConfig, Dict[str, Any]]) -> None:
        if not _MILVUS_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pymilvus",
                install_command='pip install "upsonic[milvus]"',
                feature_name="Milvus vector database provider"
            )

        if isinstance(config, dict):
            config = MilvusConfig.from_dict(config)

        super().__init__(config)
        self._config: MilvusConfig = config

        self._is_connected: bool = False
        self._metric_type: str = self._get_metric_type()

        info_log(
            f"Initialized MilvusProvider for collection '{self._config.collection_name}' "
            f"(sparse vectors: {self._config.use_sparse_vectors})",
            context="MilvusProvider"
        )

    # ============================================================================
    # Client Lifecycle Management
    # ============================================================================

    async def _create_async_client(self) -> None:
        """
        Instantiates the AsyncMilvusClient based on connection config.
        Does NOT verify readiness — only creates the client object.
        """
        conn_params: Dict[str, Any] = self._build_connection_params()
        self.client = AsyncMilvusClient(**conn_params)

    async def aget_client(self) -> "_AsyncMilvusClient":
        """
        Singleton entry point for the async client.
        Creates client if None, verifies connectivity, auto-recovers if unresponsive.
        """
        if self.client is None:
            info_log("Creating async Milvus client...", context="MilvusProvider")
            await self._create_async_client()

        try:
            await self.client.list_collections()  # type: ignore[union-attr]
        except Exception:
            info_log("Existing client unresponsive, recreating...", context="MilvusProvider")
            try:
                if self.client:
                    await asyncio.wait_for(self.client.close(), timeout=2.0)
            except Exception:
                pass
            self.client = None
            self._is_connected = False
            await self._create_async_client()

            try:
                await self.client.list_collections()  # type: ignore[union-attr]
            except Exception as e:
                self.client = None
                self._is_connected = False
                raise ConnectionError(f"Milvus client not ready after recreation: {e}") from e

        self._is_connected = True
        return self.client  # type: ignore[return-value]

    def get_client(self) -> "_AsyncMilvusClient":
        return self._run_async_from_sync(self.aget_client())

    async def ais_client_connected(self) -> bool:
        if self.client is None:
            return False
        try:
            await self.client.list_collections()
            return True
        except Exception:
            return False

    def is_client_connected(self) -> bool:
        return self._run_async_from_sync(self.ais_client_connected())

    def _build_connection_params(self) -> Dict[str, Any]:
        conn = self._config.connection
        params: Dict[str, Any] = {}

        if conn.mode == Mode.EMBEDDED:
            params['uri'] = conn.db_path or './milvus.db'
        elif conn.mode == Mode.CLOUD:
            if conn.url:
                params['uri'] = conn.url
            else:
                params['uri'] = f"https://{conn.host}:{conn.port or 19530}"
            if conn.api_key:
                params['token'] = conn.api_key.get_secret_value()
        else:
            if conn.url:
                params['uri'] = conn.url
            elif conn.host:
                protocol = 'https' if conn.use_tls else 'http'
                params['uri'] = f"{protocol}://{conn.host}:{conn.port or 19530}"
            if conn.api_key:
                params['token'] = conn.api_key.get_secret_value()

        if conn.timeout:
            params['timeout'] = conn.timeout

        return params

    def _get_metric_type(self) -> str:
        return DISTANCE_METRIC_MAP.get(self._config.distance_metric.value, 'COSINE')

    def _generate_provider_id(self) -> str:
        conn = self._config.connection
        identifier_parts = [
            conn.host or conn.url or "embedded",
            str(conn.port) if conn.port else "",
            self._config.collection_name
        ]
        identifier = "#".join(filter(None, identifier_parts))
        return hashlib.md5(identifier.encode()).hexdigest()[:16]

    # ============================================================================
    # Connection Management
    # ============================================================================

    async def aconnect(self) -> None:
        """Establish connection to Milvus. Idempotent — no-op if already connected."""
        if await self.ais_client_connected():
            info_log("Already connected to Milvus.", context="MilvusProvider")
            return
        try:
            await self.aget_client()
            info_log(
                f"Connected to Milvus at {self._config.connection.host or 'embedded'}",
                context="MilvusProvider",
            )
        except Exception as e:
            self.client = None
            self._is_connected = False
            error_log(f"Failed to connect to Milvus: {e}", context="MilvusProvider")
            raise

    async def adisconnect(self) -> None:
        """Gracefully close the Milvus connection."""
        if self.client:
            try:
                await asyncio.wait_for(self.client.close(), timeout=2.0)
            except asyncio.TimeoutError:
                error_log("Timeout closing async client, forcing cleanup", context="MilvusProvider")
            except Exception as e:
                error_log(f"Error closing async client: {e}", context="MilvusProvider")
            finally:
                self.client = None

        await asyncio.sleep(0.1)
        self._is_connected = False
        info_log("Disconnected from Milvus", context="MilvusProvider")

    async def ais_ready(self) -> bool:
        return await self.ais_client_connected()

    # ============================================================================
    # Collection Management
    # ============================================================================

    async def acollection_exists(self) -> bool:
        client = await self.aget_client()
        try:
            return await client.has_collection(self._config.collection_name)
        except Exception as e:
            logger.debug(f"Error checking collection existence: {e}")
            return False

    async def acreate_collection(self) -> None:
        if await self.acollection_exists():
            if self._config.recreate_if_exists:
                info_log(f"Collection '{self._config.collection_name}' exists. Recreating...", context="MilvusProvider")
                await self.adelete_collection()
            else:
                info_log(f"Collection '{self._config.collection_name}' already exists. Skipping creation.", context="MilvusProvider")
                return

        info_log(f"Creating collection '{self._config.collection_name}'...", context="MilvusProvider")

        if self._config.use_sparse_vectors:
            await self._create_hybrid_collection()
        else:
            await self._create_dense_collection()

        info_log(f"Collection '{self._config.collection_name}' created successfully.", context="MilvusProvider")

    async def _create_dense_collection(self) -> None:
        # NOTE: `knowledge_base_id` is a first-class schema field (VARCHAR, max_length=256)
        # with an INVERTED scalar index (non-EMBEDDED only; MilvusLite rejects scalar
        # indexes + `nullable`, so Embedded mode stores an empty string for missing values
        # and relies on a full scan for filter expressions — still faster than a dynamic
        # JSON field). Users upgrading from older Upsonic versions
        # whose collections pre-date this field must set `recreate_if_exists=True`
        # in their MilvusConfig to pick up the new schema, otherwise filter queries
        # on `knowledge_base_id` will fall back to the slow dynamic-field path.
        client = await self.aget_client()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        indexed_fields_config = self._parse_indexed_fields()

        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field(field_name="chunk_content_hash", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="doc_content_hash", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="knowledge_base_id", datatype=DataType.VARCHAR, max_length=256)

        doc_name_config = indexed_fields_config.get("document_name", {"type": "keyword"})
        schema.add_field(
            field_name="document_name",
            datatype=self._get_milvus_datatype(doc_name_config.get("type", "keyword")),
            max_length=1024 if doc_name_config.get("type", "keyword") in ["text", "keyword"] else None,
        )

        doc_id_config = indexed_fields_config.get("document_id", {"type": "keyword"})
        schema.add_field(
            field_name="document_id",
            datatype=self._get_milvus_datatype(doc_id_config.get("type", "keyword")),
            max_length=256 if doc_id_config.get("type", "keyword") in ["text", "keyword"] else None,
        )

        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name=self._config.dense_vector_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=self._config.vector_size,
        )

        index_params = client.prepare_index_params()
        vector_index_params = self._build_vector_index_params()
        index_params.add_index(
            field_name=self._config.dense_vector_field,
            index_name="dense_vector_index",
            **vector_index_params,
        )

        if self._config.connection.mode != Mode.EMBEDDED:
            try:
                index_params.add_index(
                    field_name="knowledge_base_id",
                    index_type="INVERTED",
                    index_name="knowledge_base_id_idx",
                )
            except Exception as e:
                logger.warning(f"Failed to add scalar index for knowledge_base_id: {e}")
        if indexed_fields_config and self._config.connection.mode != Mode.EMBEDDED:
            self._add_scalar_indexes(index_params, indexed_fields_config)

        await client.create_collection(
            collection_name=self._config.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self._config.consistency_level,
        )

    async def _create_hybrid_collection(self) -> None:
        # See `_create_dense_collection` for notes on `knowledge_base_id` schema promotion
        # and the `recreate_if_exists=True` migration requirement.
        client = await self.aget_client()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        indexed_fields_config = self._parse_indexed_fields()

        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field(field_name="chunk_content_hash", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="doc_content_hash", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="knowledge_base_id", datatype=DataType.VARCHAR, max_length=256)

        doc_name_config = indexed_fields_config.get("document_name", {"type": "keyword"})
        schema.add_field(
            field_name="document_name",
            datatype=self._get_milvus_datatype(doc_name_config.get("type", "keyword")),
            max_length=1024 if doc_name_config.get("type", "keyword") in ["text", "keyword"] else None,
        )

        doc_id_config = indexed_fields_config.get("document_id", {"type": "keyword"})
        schema.add_field(
            field_name="document_id",
            datatype=self._get_milvus_datatype(doc_id_config.get("type", "keyword")),
            max_length=256 if doc_id_config.get("type", "keyword") in ["text", "keyword"] else None,
        )

        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)

        schema.add_field(
            field_name=self._config.dense_vector_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=self._config.vector_size,
        )
        schema.add_field(
            field_name=self._config.sparse_vector_field,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        index_params = client.prepare_index_params()
        vector_index_params = self._build_vector_index_params()
        index_params.add_index(
            field_name=self._config.dense_vector_field,
            index_name="dense_vector_index",
            **vector_index_params,
        )
        index_params.add_index(
            field_name=self._config.sparse_vector_field,
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )

        if self._config.connection.mode != Mode.EMBEDDED:
            try:
                index_params.add_index(
                    field_name="knowledge_base_id",
                    index_type="INVERTED",
                    index_name="knowledge_base_id_idx",
                )
            except Exception as e:
                logger.warning(f"Failed to add scalar index for knowledge_base_id: {e}")
        if indexed_fields_config and self._config.connection.mode != Mode.EMBEDDED:
            self._add_scalar_indexes(index_params, indexed_fields_config)

        await client.create_collection(
            collection_name=self._config.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self._config.consistency_level,
        )

    def _add_scalar_indexes(self, index_params: Any, indexed_fields_config: Dict[str, Dict[str, Any]]) -> None:
        indexable_fields = self._STANDARD_FIELDS - {'content', 'metadata'}
        for field_name, field_config in indexed_fields_config.items():
            if field_name in indexable_fields:
                logger.info(f"Creating scalar index for field: {field_name} (type: {field_config.get('type', 'keyword')})")
                try:
                    field_type = field_config.get("type", "keyword")
                    index_type = self._get_milvus_index_type(field_type)
                    index_params.add_index(field_name=field_name, index_type=index_type)
                except Exception as e:
                    logger.warning(f"Failed to add scalar index for {field_name}: {e}")

    async def adelete_collection(self) -> None:
        client = await self.aget_client()
        if not await self.acollection_exists():
            info_log(f"Collection '{self._config.collection_name}' does not exist.", context="MilvusProvider")
            return
        info_log(f"Deleting collection '{self._config.collection_name}'...", context="MilvusProvider")
        await client.drop_collection(self._config.collection_name)
        info_log(f"Collection '{self._config.collection_name}' deleted.", context="MilvusProvider")

    def _build_vector_index_params(self) -> Dict[str, Any]:
        index_config = self._config.index

        if self._config.index_params:
            return self._config.index_params

        is_embedded = self._config.connection.mode == Mode.EMBEDDED
        params: Dict[str, Any] = {"metric_type": self._metric_type}

        if index_config.type == 'HNSW':
            if is_embedded:
                logger.warning(
                    "HNSW index not supported in embedded mode (Milvus Lite). "
                    "Falling back to IVF_FLAT index."
                )
                params["index_type"] = "IVF_FLAT"
                nlist = min(1024, max(64, index_config.m * 4))
                params["params"] = {"nlist": nlist}
            else:
                params["index_type"] = "HNSW"
                params["params"] = {"M": index_config.m, "efConstruction": index_config.ef_construction}
        elif index_config.type == 'IVF_FLAT':
            params["index_type"] = "IVF_FLAT"
            params["params"] = {"nlist": index_config.nlist}
        elif index_config.type == 'FLAT':
            params["index_type"] = "FLAT"
            params["params"] = {}

        return params

    def _parse_indexed_fields(self) -> Dict[str, Dict[str, Any]]:
        if not self._config.indexed_fields:
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for item in self._config.indexed_fields:
            if isinstance(item, str):
                result[item] = {"indexed": True, "type": "keyword"}
            elif isinstance(item, dict):
                field_name = item.get("field")
                if field_name:
                    result[field_name] = {"indexed": True, "type": item.get("type", "keyword")}
        return result

    def _get_milvus_datatype(self, field_type: str):
        type_map = {
            'text': DataType.VARCHAR, 'keyword': DataType.VARCHAR,
            'integer': DataType.INT64, 'int': DataType.INT64,
            'int8': DataType.INT8, 'int16': DataType.INT16,
            'int32': DataType.INT32, 'int64': DataType.INT64,
            'float': DataType.FLOAT, 'double': DataType.DOUBLE,
            'boolean': DataType.BOOL, 'bool': DataType.BOOL,
        }
        return type_map.get(field_type.lower(), DataType.VARCHAR)

    def _get_milvus_index_type(self, field_type: str) -> str:
        if field_type.lower() in ['text', 'keyword']:
            return "TRIE"
        elif field_type.lower() in ['integer', 'int', 'int64', 'int8', 'int16', 'int32']:
            return "STL_SORT"
        elif field_type.lower() in ['float', 'double']:
            return "STL_SORT"
        elif field_type.lower() in ['boolean', 'bool']:
            return "INVERTED"
        return "TRIE"

    # ============================================================================
    # Payload Builder
    # ============================================================================

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
        apply_defaults: bool = True,
    ) -> Dict[str, Any]:
        """
        Builds a standardized record dict with all structural fields as explicit
        parameters. Non-standard keys from extra_payload are folded into metadata.
        Standard keys in extra_payload are ignored (with a warning).

        When ``apply_defaults`` is True (default, used by inserts/upserts), the
        configured ``default_metadata`` is merged into the row. Update paths
        must pass ``apply_defaults=False`` so that config defaults do not
        silently overwrite caller-provided values that were set on the
        original upsert.
        """
        combined_metadata: Dict[str, Any] = {}

        if apply_defaults and self._config.default_metadata:
            combined_metadata.update(self._config.default_metadata)

        if extra_payload:
            for key in list(extra_payload.keys()):
                if key in self._STANDARD_FIELDS:
                    if _warned_standard_keys is not None and key not in _warned_standard_keys:
                        debug_log(
                            f"Standard field '{key}' found in payload dict and will be ignored. "
                            f"Standard fields must be passed via dedicated parameters "
                            f"(ids, document_ids, document_names, doc_content_hashes, "
                            f"chunk_content_hashes, knowledge_base_ids).",
                            context="MilvusVectorDB",
                        )
                        _warned_standard_keys.add(key)
                else:
                    combined_metadata[key] = extra_payload[key]

        if extra_metadata:
            combined_metadata.update(extra_metadata)

        return {
            "chunk_id": chunk_id,
            "chunk_content_hash": chunk_content_hash,
            "document_id": document_id,
            "doc_content_hash": doc_content_hash,
            "document_name": document_name,
            "content": content,
            "knowledge_base_id": knowledge_base_id or "",
            "metadata": json.dumps(combined_metadata),
        }

    # ============================================================================
    # Data Operations
    # ============================================================================

    async def _batch_find_existing_by_hashes(
        self,
        chunk_hashes: List[str],
    ) -> Dict[str, List[str]]:
        """
        Single-query batch lookup: finds which chunk_content_hash values
        already exist and returns the chunk_ids that own them.
        """
        if not chunk_hashes:
            return {}

        unique_hashes: List[str] = list(set(chunk_hashes))
        try:
            client = await self.aget_client()
            quoted = ', '.join(f'"{h}"' for h in unique_hashes)
            filter_expr = f'chunk_content_hash in [{quoted}]'

            results = await client.query(
                collection_name=self._config.collection_name,
                filter=filter_expr,
                output_fields=["chunk_id", "chunk_content_hash"],
                limit=len(unique_hashes) * 10,
            )

            existing: Dict[str, List[str]] = {}
            for record in results:
                h = record.get("chunk_content_hash", "")
                cid = record.get("chunk_id", "")
                if h:
                    existing.setdefault(h, []).append(cid)

            return existing
        except Exception as e:
            logger.debug(f"Batch hash lookup failed, treating all as new: {e}")
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
        Upsert data into Milvus with content-based deduplication.

        Uses chunk_content_hash for dedup via a batch pre-check.
        Milvus native upsert handles primary-key (chunk_id) replacement.
        """
        client = await self.aget_client()

        if (vectors is None or len(vectors) == 0) and (sparse_vectors is None or len(sparse_vectors) == 0):
            info_log("Nothing to upsert: no dense or sparse vectors provided.", context="MilvusProvider")
            return

        n: int = len(vectors) if vectors else len(sparse_vectors)  # type: ignore[arg-type]

        # Validate per-item array lengths
        for _arr_name, _arr in (
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
                raise ValueError(
                    f"Length mismatch in aupsert: '{_arr_name}' has length {len(_arr)}, expected {n}"
                )

        info_log(f"Upserting {n} records into '{self._config.collection_name}'", context="MilvusProvider")
        extra_metadata: Dict[str, Any] = metadata or {}
        _warned_standard_keys: set = set()

        computed_hashes: List[str] = []
        for i in range(n):
            payload: Dict[str, Any] = payloads[i] if payloads and i < len(payloads) else {}
            content_text: str = chunks[i] if chunks and i < len(chunks) else ""
            if chunk_content_hashes and i < len(chunk_content_hashes) and chunk_content_hashes[i]:
                computed_hashes.append(chunk_content_hashes[i])
            else:
                computed_hashes.append(hashlib.md5(content_text.encode("utf-8")).hexdigest())

        existing_hash_map: Dict[str, List[str]] = await self._batch_find_existing_by_hashes(computed_hashes)

        stale_chunk_ids: List[str] = []

        data: List[Dict[str, Any]] = []
        for i in range(n):
            payload = payloads[i] if payloads and i < len(payloads) else {}
            content: str = chunks[i] if chunks and i < len(chunks) else ""

            chunk_id_str: str = str(ids[i]) if ids and i < len(ids) else str(_uuid.uuid4())
            doc_id: str = document_ids[i] if document_ids and i < len(document_ids) else ""
            doc_hash: str = doc_content_hashes[i] if doc_content_hashes and i < len(doc_content_hashes) else ""
            chunk_hash: str = computed_hashes[i]
            # Standard fields come ONLY from dedicated parameters, never from payload dicts.
            doc_name: str = document_names[i] if document_names and i < len(document_names) else ""

            old_ids: List[str] = existing_hash_map.get(chunk_hash, [])
            for old_id in old_ids:
                if old_id != chunk_id_str:
                    stale_chunk_ids.append(old_id)

            kbi: Optional[str] = knowledge_base_ids[i] if knowledge_base_ids else None
            record = self._build_payload(
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
            if vectors and i < len(vectors):
                record[self._config.dense_vector_field] = vectors[i]

            if sparse_vectors and i < len(sparse_vectors):
                record[self._config.sparse_vector_field] = sparse_vectors[i]

            data.append(record)

        if stale_chunk_ids:
            try:
                await client.delete(
                    collection_name=self._config.collection_name,
                    ids=list(set(stale_chunk_ids)),
                )
                logger.debug(f"Deleted {len(set(stale_chunk_ids))} stale records with duplicate chunk_content_hash.")
            except Exception as e:
                logger.debug(f"Failed to delete stale duplicate records: {e}")

        batch_size: int = self._config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            await client.upsert(
                collection_name=self._config.collection_name,
                data=batch,
            )

        replaced_count: int = sum(1 for h in computed_hashes if h in existing_hash_map)
        new_count: int = len(data) - replaced_count
        info_log(
            f"Upserted {len(data)} records ({new_count} new, {replaced_count} replaced by content hash).",
            context="MilvusProvider",
        )

    async def adelete(self, ids: List[Union[str, int]]) -> None:
        if not ids:
            return
        client = await self.aget_client()
        info_log(f"Deleting {len(ids)} records from '{self._config.collection_name}'", context="MilvusProvider")
        str_ids: List[str] = [str(id_val) for id_val in ids]
        # Idempotent — delete by PK directly. Avoids read-after-write consistency
        # issues on Zilliz Serverless where a query-based existence check can
        # return empty results for rows that were just upserted.
        await client.delete(collection_name=self._config.collection_name, ids=str_ids)
        info_log(
            f"Delete issued for {len(str_ids)} IDs in '{self._config.collection_name}'.",
            context="MilvusProvider",
        )

    async def afetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        if not ids:
            return []
        client = await self.aget_client()
        str_ids = [str(id_val) for id_val in ids]
        results = await client.get(collection_name=self._config.collection_name, ids=str_ids)
        return [self._convert_to_search_result(r) for r in results if r]

    # ============================================================================
    # Generic Field Helpers
    # ============================================================================

    async def afield_exists(self, field_name: str, field_value: Any) -> bool:
        """Generic check: does any record with field_name == field_value exist?"""
        try:
            client = await self.aget_client()
            filter_expr = f'{field_name} == "{field_value}"'
            results = await client.query(
                collection_name=self._config.collection_name,
                filter=filter_expr,
                limit=1,
            )
            return len(results) > 0
        except Exception as e:
            logger.debug(f"Error checking if {field_name}='{field_value}' exists: {e}")
            return False

    async def adelete_by_field(self, field_name: str, field_value: Any) -> bool:
        """Generic deletion of all records matching a field value. Idempotent."""
        try:
            client = await self.aget_client()
            filter_expr = f'{field_name} == "{field_value}"'
            await client.delete(
                collection_name=self._config.collection_name,
                filter=filter_expr,
            )
            info_log(f"Deleted records with {field_name}='{field_value}'", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error deleting records with {field_name}='{field_value}': {e}", context="MilvusProvider")
            return False

    # ============================================================================
    # Existence Check Methods (delegates to afield_exists)
    # ============================================================================

    async def aid_exists(self, id: str) -> bool:
        """Check if a record with the given primary key (chunk_id) exists."""
        try:
            client = await self.aget_client()
            results = await client.get(collection_name=self._config.collection_name, ids=[str(id)])
            return len(results) > 0
        except Exception as e:
            logger.debug(f"Error checking ID existence: {e}")
            return False

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        return await self.afield_exists("chunk_id", chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        return await self.afield_exists("doc_content_hash", doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        return await self.afield_exists("chunk_content_hash", chunk_content_hash)

    async def adocument_name_exists(self, document_name: str) -> bool:
        return await self.afield_exists("document_name", document_name)

    async def adocument_id_exists(self, document_id: str) -> bool:
        return await self.afield_exists("document_id", document_id)

    # ============================================================================
    # Delete Methods (delegates to adelete_by_field)
    # ============================================================================

    async def adelete_by_document_name(self, document_name: str) -> bool:
        return await self.adelete_by_field("document_name", document_name)

    async def adelete_by_document_id(self, document_id: str) -> bool:
        return await self.adelete_by_field("document_id", document_id)

    async def adelete_by_chunk_id(self, chunk_id: str) -> bool:
        return await self.adelete_by_field("chunk_id", chunk_id)

    async def adelete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        return await self.adelete_by_field("doc_content_hash", doc_content_hash)

    async def adelete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        return await self.adelete_by_field("chunk_content_hash", chunk_content_hash)

    async def adelete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        try:
            client = await self.aget_client()
            filter_expr = self._build_filter_expression(metadata)
            if not filter_expr:
                error_log("Invalid metadata filter", context="MilvusProvider")
                return False
            await client.delete(collection_name=self._config.collection_name, filter=filter_expr)
            info_log(f"Deleted records matching metadata: {metadata}", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error deleting records with metadata {metadata}: {e}", context="MilvusProvider")
            return False

    async def adelete_by_filter(self, filter: Dict[str, Any]) -> None:
        client = await self.aget_client()
        filter_expr = self._build_filter_expression(filter)
        if not filter_expr:
            raise ValueError("Invalid filter expression")
        info_log(f"Deleting records matching filter: {filter_expr}", context="MilvusProvider")
        await client.delete(collection_name=self._config.collection_name, filter=filter_expr)

    async def adelete_single_id(self, id: Union[str, int]) -> bool:
        try:
            if not await self.aid_exists(str(id)):
                info_log(f"Record with ID '{id}' does not exist.", context="MilvusProvider")
                return False
            client = await self.aget_client()
            await client.delete(collection_name=self._config.collection_name, ids=[str(id)])
            info_log(f"Deleted record with ID '{id}'", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error deleting record with ID {id}: {e}", context="MilvusProvider")
            return False

    # ============================================================================
    # Metadata Update
    # ============================================================================

    async def aupdate_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific chunk_id.
        Fetches the record, merges metadata, upserts back.
        """
        try:
            client = await self.aget_client()
            results = await client.query(
                collection_name=self._config.collection_name,
                filter=f'chunk_id == "{chunk_id}"',
                output_fields=["*"],
                limit=1,
            )

            if not results:
                info_log(f"No record found with chunk_id: {chunk_id}", context="MilvusProvider")
                return False

            record = results[0]

            existing_metadata_str = record.get('metadata', '{}')
            try:
                existing_metadata = json.loads(existing_metadata_str) if isinstance(existing_metadata_str, str) else existing_metadata_str
            except json.JSONDecodeError:
                existing_metadata = {}

            updated_metadata = existing_metadata.copy()
            updated_metadata.update(metadata)

            updated_record = self._build_payload(
                content=record.get('content', ''),
                chunk_id=chunk_id,
                chunk_content_hash=record.get('chunk_content_hash', ''),
                document_id=record.get('document_id', ''),
                doc_content_hash=record.get('doc_content_hash', ''),
                document_name=record.get('document_name', ''),
                knowledge_base_id=record.get('knowledge_base_id', '') or None,
                extra_metadata=updated_metadata,
                _warned_standard_keys=set(),
                apply_defaults=False,
            )
            updated_record[self._config.dense_vector_field] = record.get(self._config.dense_vector_field)

            if self._config.use_sparse_vectors and self._config.sparse_vector_field in record:
                updated_record[self._config.sparse_vector_field] = record.get(self._config.sparse_vector_field)

            await client.upsert(collection_name=self._config.collection_name, data=[updated_record])
            info_log(f"Updated metadata for chunk_id: {chunk_id}", context="MilvusProvider")
            return True

        except Exception as e:
            error_log(f"Error updating metadata for chunk_id '{chunk_id}': {e}", context="MilvusProvider")
            return False

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
        await self.aget_client()

        top_k = top_k if top_k is not None else self._config.default_top_k
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold

        has_vector = query_vector is not None
        has_text = query_text is not None

        if has_vector and has_text:
            if not self._config.hybrid_search_enabled:
                raise ValueError("Hybrid search is disabled in config")
            return await self.ahybrid_search(query_vector, query_text, top_k, filter, alpha, fusion_method, similarity_threshold, apply_reranking=apply_reranking, sparse_query_vector=sparse_query_vector)
        elif has_vector:
            if not self._config.dense_search_enabled:
                raise ValueError("Dense search is disabled in config")
            return await self.adense_search(query_vector, top_k, filter, similarity_threshold, apply_reranking=apply_reranking)
        elif has_text:
            if not self._config.full_text_search_enabled:
                raise ValueError("Full-text search is disabled in config")
            return await self.afull_text_search(query_text, top_k, filter, similarity_threshold, apply_reranking=apply_reranking, sparse_query_vector=sparse_query_vector)
        else:
            raise ValueError("Either query_vector or query_text must be provided")

    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        del apply_reranking  # accepted for API parity; not applied in dense path
        client = await self.aget_client()
        info_log(f"Performing dense search (top_k={top_k})", context="MilvusProvider")

        final_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        search_params = self._build_search_params()
        filter_expr = self._build_filter_expression(filter) if filter else None

        results = await client.search(
            collection_name=self._config.collection_name,
            data=[query_vector],
            anns_field=self._config.dense_vector_field,
            limit=top_k,
            output_fields=["*"],
            search_params=search_params,
            filter=filter_expr,
        )

        search_results: List[VectorSearchResult] = []
        for hits in results:
            for hit in hits:
                result = self._convert_to_search_result(hit, final_threshold)
                if result:
                    search_results.append(result)

        info_log(f"Found {len(search_results)} results", context="MilvusProvider")
        return search_results

    async def afull_text_search(
        self,
        query_text: Optional[str] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        _ = (query_text, apply_reranking)  # accepted for API parity; sparse_query_vector is authoritative
        if not self._config.use_sparse_vectors:
            raise ValueError("Full-text search requires use_sparse_vectors=True in config")

        top_k = top_k or self._config.default_top_k

        if sparse_query_vector is None:
            raise ValueError("sparse_query_vector must be provided for full-text search")

        client = await self.aget_client()
        info_log(f"Performing full-text search (top_k={top_k})", context="MilvusProvider")

        final_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        search_params: Dict[str, Any] = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
        filter_expr = self._build_filter_expression(filter) if filter else None

        results = await client.search(
            collection_name=self._config.collection_name,
            data=[sparse_query_vector],
            anns_field=self._config.sparse_vector_field,
            limit=top_k,
            output_fields=["*"],
            search_params=search_params,
            filter=filter_expr,
        )

        search_results: List[VectorSearchResult] = []
        for hits in results:
            for hit in hits:
                result = self._convert_to_search_result(hit, final_threshold)
                if result:
                    search_results.append(result)

        info_log(f"Found {len(search_results)} results", context="MilvusProvider")
        return search_results

    async def ahybrid_search(
        self,
        query_vector: List[float],
        query_text: Optional[str] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        _ = (query_text, apply_reranking)  # accepted for API parity; sparse_query_vector is authoritative
        if not self._config.use_sparse_vectors:
            raise ValueError("Hybrid search requires use_sparse_vectors=True in config")

        top_k = top_k or self._config.default_top_k

        if sparse_query_vector is None:
            raise ValueError("sparse_query_vector must be provided for hybrid search")

        client = await self.aget_client()
        info_log(f"Performing hybrid search (top_k={top_k})", context="MilvusProvider")

        alpha = alpha if alpha is not None else self._config.default_hybrid_alpha
        fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method
        final_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold

        dense_search_params = self._build_search_params()
        sparse_search_params: Dict[str, Any] = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
        filter_expr = self._build_filter_expression(filter) if filter else None

        dense_request = AnnSearchRequest(
            data=[query_vector],
            anns_field=self._config.dense_vector_field,
            param=dense_search_params,
            limit=top_k * 2,
        )

        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field=self._config.sparse_vector_field,
            param=sparse_search_params,
            limit=top_k * 2,
        )

        if fusion_method == 'rrf':
            ranker = RRFRanker(self._config.rrf_k)
        else:
            ranker = WeightedRanker(alpha, 1 - alpha)

        results = await client.hybrid_search(
            collection_name=self._config.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=top_k,
            output_fields=["*"],
            filter=filter_expr,
        )

        search_results: List[VectorSearchResult] = []
        for hits in results:
            for hit in hits:
                result = self._convert_to_search_result(hit, final_threshold)
                if result:
                    search_results.append(result)

        info_log(f"Found {len(search_results)} hybrid results", context="MilvusProvider")
        return search_results

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _build_search_params(self) -> Dict[str, Any]:
        if self._config.search_params:
            return self._config.search_params

        index_config = self._config.index
        params: Dict[str, Any] = {"metric_type": self._metric_type}

        if index_config.type == 'HNSW':
            ef_search = index_config.ef_search or max(index_config.ef_construction, 100)
            params["params"] = {"ef": ef_search}
        elif index_config.type == 'IVF_FLAT':
            nprobe = index_config.nprobe or min(index_config.nlist, 10)
            params["params"] = {"nprobe": nprobe}
        else:
            params["params"] = {}

        return params

    def _build_filter_expression(self, filter: Dict[str, Any]) -> str:
        if not filter:
            return ""

        standard_fields = self._STANDARD_FIELDS - {'metadata'}
        expressions: List[str] = []

        for key, value in filter.items():
            # Support dotted paths like "metadata.category"
            lookup_key = key[len("metadata."):] if key.startswith("metadata.") else key
            is_standard = lookup_key in standard_fields and not key.startswith("metadata.")
            if is_standard:
                if isinstance(value, bool):
                    expressions.append(f'{lookup_key} == {str(value).lower()}')
                elif isinstance(value, str):
                    expressions.append(f'{lookup_key} == "{value}"')
                elif isinstance(value, (int, float)):
                    expressions.append(f'{lookup_key} == {value}')
            else:
                # User metadata stored as JSON string in 'metadata' field.
                # Use LIKE matching against the JSON substring (Milvus Lite/Zilliz
                # json_contains on string-serialized JSON is inconsistent).
                if isinstance(value, str):
                    needle = f'"{lookup_key}": "{value}"'
                    expressions.append(f'metadata like "%{needle}%"')
                elif isinstance(value, bool):
                    needle = f'"{lookup_key}": {str(value).lower()}'
                    expressions.append(f'metadata like "%{needle}%"')
                elif isinstance(value, (int, float)):
                    needle = f'"{lookup_key}": {value}'
                    expressions.append(f'metadata like "%{needle}%"')

        return " and ".join(expressions) if expressions else ""

    def _convert_to_search_result(
        self,
        hit: Dict[str, Any],
        similarity_threshold: Optional[float] = None,
    ) -> Optional[VectorSearchResult]:
        entity = hit.get('entity', hit)
        score: float = hit.get('distance', 0.0)

        if similarity_threshold is not None and score < similarity_threshold:
            return None

        metadata_str = entity.get('metadata', '{}')
        try:
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        except json.JSONDecodeError:
            metadata = {}

        vector = entity.get(self._config.dense_vector_field)

        return VectorSearchResult(
            id=entity.get('chunk_id', ''),
            score=score,
            payload={
                'content': entity.get('content', ''),
                'document_name': entity.get('document_name', ''),
                'document_id': entity.get('document_id', ''),
                'chunk_id': entity.get('chunk_id', ''),
                'chunk_content_hash': entity.get('chunk_content_hash', ''),
                'doc_content_hash': entity.get('doc_content_hash', ''),
                'knowledge_base_id': entity.get('knowledge_base_id', ''),
                'metadata': metadata,
            },
            vector=vector,
            text=entity.get('content', ''),
        )

    # ============================================================================
    # Stats / Optimization / Misc
    # ============================================================================

    async def aget_collection_stats(self) -> Dict[str, Any]:
        client = await self.aget_client()
        return await client.get_collection_stats(self._config.collection_name)

    async def acount(self) -> int:
        stats = await self.aget_collection_stats()
        return stats.get('row_count', 0)

    async def aget_count(self) -> int:
        return await self.acount()

    async def aoptimize(self) -> bool:
        return True

    async def aget_supported_search_types(self) -> List[str]:
        supported: List[str] = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported

    async def adrop(self) -> bool:
        try:
            if not await self.acollection_exists():
                info_log(f"Collection '{self._config.collection_name}' does not exist.", context="MilvusProvider")
                return False
            client = await self.aget_client()
            await client.drop_collection(self._config.collection_name)
            info_log(f"Dropped collection '{self._config.collection_name}'", context="MilvusProvider")
            return True
        except Exception as e:
            error_log(f"Error dropping collection: {e}", context="MilvusProvider")
            return False

    async def query(
        self,
        filter: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        client = await self.aget_client()
        filter_expr = self._build_filter_expression(filter) if filter else ""
        results = await client.query(
            collection_name=self._config.collection_name,
            filter=filter_expr or None,
            output_fields=output_fields or ["*"],
            limit=limit,
            offset=offset,
        )
        return results
