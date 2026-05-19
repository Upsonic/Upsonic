"""
Valkey Search Provider Implementation

A comprehensive, async-first vector database provider for Valkey with the
valkey-search module (v1.2+). Uses valkey-glide as the client library.

Requires Valkey 9.1+ with the search module loaded.

Standard properties stored per hash key ({key_prefix}{chunk_id}):
- vector (VECTOR, float32 blob) — the embedding vector
- content (TEXT) — the chunk text, full-text indexed
- chunk_id (TAG) — unique per-chunk identifier
- document_id (TAG) — parent document identifier
- document_name (TAG) — human-readable source name
- doc_content_hash (TAG) — MD5 of parent document content for change detection
- chunk_content_hash (TAG) — MD5 of chunk text content for deduplication
- knowledge_base_id (TAG) — KnowledgeBase isolation
- metadata (TEXT) — JSON-serialized non-standard metadata

This implementation provides:
- Full async/await support using valkey-glide (GlideClient)
- HNSW and FLAT vector index support
- Dense (KNN), full-text, and hybrid search
- Content-based deduplication via chunk_content_hash
- Batch upsert operations via pipelining
- Cluster mode support via GlideClusterClient
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import uuid as _uuid
from typing import Any, Dict, List, Optional, Union, Literal, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from glide import (
        GlideClient,
        GlideClusterClient,
        GlideClientConfiguration,
        GlideClusterClientConfiguration,
        NodeAddress,
        FtCreateOptions,
        FtSearchOptions,
        FtSearchLimit,
        ReturnField,
        VectorField,
        VectorFieldAttributesHnsw,
        VectorFieldAttributesFlat,
        VectorAlgorithm,
        DistanceMetricType,
        VectorType,
        TextField,
        TagField,
    )

try:
    from glide import (
        GlideClient,
        GlideClusterClient,
        GlideClientConfiguration,
        GlideClusterClientConfiguration,
        NodeAddress,
        FtCreateOptions,
        FtSearchOptions,
        FtSearchLimit,
        ReturnField,
        VectorField,
        VectorFieldAttributesHnsw,
        VectorFieldAttributesFlat,
        VectorAlgorithm,
        DistanceMetricType,
        VectorType,
        TextField,
        TagField,
        ft,
    )
    _VALKEY_GLIDE_AVAILABLE = True
except ImportError:
    _VALKEY_GLIDE_AVAILABLE = False

from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.vectordb.config import (
    ValkeyConfig,
    Mode,
    DistanceMetric,
    HNSWIndexConfig,
    FlatIndexConfig,
)
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.utils.logging_config import get_logger
from upsonic.utils.printing import info_log, debug_log
from upsonic.utils.package.exception import (
    VectorDBConnectionError,
    VectorDBError,
    CollectionDoesNotExistError,
    ConfigurationError,
    SearchError,
    UpsertError,
)

logger = get_logger(__name__)


class ValkeyProvider(BaseVectorDBProvider):
    """
    Valkey Search vector database provider using valkey-glide.

    Async-native implementation using GLIDE's built-in FT.* module API.
    Stores documents as Valkey Hash keys with vector, content, and metadata fields.

    Supports:
    - HNSW and FLAT vector indexes
    - Dense (KNN) vector similarity search
    - Full-text search via Valkey Search text fields
    - Hybrid search combining dense + full-text via RRF fusion
    - Content-based deduplication
    - Cluster mode
    """

    _STANDARD_FIELDS: frozenset = frozenset({
        'chunk_id', 'document_id', 'doc_content_hash',
        'chunk_content_hash', 'document_name', 'content', 'metadata',
        'knowledge_base_id',
    })

    def __init__(
        self,
        config: Union[ValkeyConfig, Dict[str, Any]],
        reranker: Optional[Any] = None,
    ) -> None:
        if not _VALKEY_GLIDE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="valkey-glide",
                install_command='pip install "upsonic[valkey]"',
                feature_name="Valkey Search vector database provider",
            )

        if isinstance(config, dict):
            config = ValkeyConfig.from_dict(config)

        if not isinstance(config, ValkeyConfig):
            raise ConfigurationError(
                "config must be either a ValkeyConfig instance or a dictionary"
            )

        super().__init__(config)
        self._config: ValkeyConfig = cast(ValkeyConfig, self._config)
        self.client: Optional[Any] = None
        self.reranker: Optional[Any] = reranker

        self._index_name: str = self._config.collection_name
        self._key_prefix: str = self._config.key_prefix

        logger.info(
            f"Initialized ValkeyProvider for index '{self._index_name}' "
            f"(key_prefix: '{self._key_prefix}')"
        )

    # ========================================================================
    # Helpers
    # ========================================================================

    def _get_distance_metric(self) -> "DistanceMetricType":
        """Map framework DistanceMetric to GLIDE DistanceMetricType."""
        metric_map = {
            DistanceMetric.COSINE: DistanceMetricType.COSINE,
            DistanceMetric.EUCLIDEAN: DistanceMetricType.L2,
            DistanceMetric.DOT_PRODUCT: DistanceMetricType.IP,
        }
        return metric_map[self._config.distance_metric]

    def _vector_to_bytes(self, vector: List[float]) -> bytes:
        """Convert a list of floats to a float32 binary blob for Valkey."""
        return struct.pack(f"<{len(vector)}f", *vector)

    def _bytes_to_vector(self, data: bytes) -> List[float]:
        """Convert a float32 binary blob back to a list of floats."""
        count = len(data) // 4
        return list(struct.unpack(f"<{count}f", data))

    def _make_key(self, chunk_id: str) -> str:
        """Build the full hash key for a chunk."""
        return f"{self._key_prefix}{chunk_id}"

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert a distance score to a similarity score in [0, 1].

        For COSINE: Valkey returns 1 - cosine_similarity, range [0, 2].
        For DOT_PRODUCT (IP): Valkey returns 1 - inner_product. Can be negative
            for vectors with negative components; clamped to [0, 1].
        For L2: Lower distance = more similar; uses inverse transform.
        """
        if self._config.distance_metric == DistanceMetric.COSINE:
            return max(0.0, min(1.0, 1.0 - distance))
        elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            return max(0.0, min(1.0, 1.0 - distance))
        else:
            return 1.0 / (1.0 + distance)

    # ========================================================================
    # Connection Lifecycle
    # ========================================================================

    async def aconnect(self) -> None:
        """Connect to Valkey using GLIDE client."""
        if self._is_connected and self.client is not None:
            logger.info("Already connected to Valkey.")
            return

        try:
            conn = self._config.connection
            addresses = []

            if conn.url:
                # Parse URL for host/port
                from urllib.parse import urlparse
                parsed = urlparse(conn.url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 6379
                addresses.append(NodeAddress(host, port))
            elif conn.host and conn.port is not None:
                addresses.append(NodeAddress(conn.host, conn.port))
            elif conn.mode == Mode.IN_MEMORY:
                # IN_MEMORY mode: connect to localhost default
                addresses.append(NodeAddress("localhost", 6379))
            else:
                addresses.append(NodeAddress("localhost", 6379))

            kwargs: Dict[str, Any] = {"addresses": addresses}
            if self._config.request_timeout is not None:
                kwargs["request_timeout"] = self._config.request_timeout

            if self._config.cluster_mode:
                client_config = GlideClusterClientConfiguration(**kwargs)
                self.client = await GlideClusterClient.create(client_config)
            else:
                client_config = GlideClientConfiguration(**kwargs)
                self.client = await GlideClient.create(client_config)

            self._is_connected = True
            logger.info("Successfully connected to Valkey")

        except Exception as e:
            self.client = None
            self._is_connected = False
            raise VectorDBConnectionError(
                f"Failed to connect to Valkey: {e}"
            ) from e

    async def adisconnect(self) -> None:
        """Disconnect from Valkey."""
        try:
            if self.client is not None:
                await self.client.close()
                self.client = None
            self._is_connected = False
            logger.info("Disconnected from Valkey")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            raise VectorDBError(f"Disconnect failed: {e}") from e

    async def ais_ready(self) -> bool:
        """Check if the Valkey connection is alive."""
        if not self._is_connected or self.client is None:
            return False
        try:
            result = await self.client.ping()
            return result == b"PONG" or result == "PONG"
        except Exception:
            return False

    # ========================================================================
    # Collection Lifecycle (FT.CREATE / FT.DROPINDEX)
    # ========================================================================

    async def acreate_collection(self) -> None:
        """Create the FT index over hash keys with the configured schema."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        exists = await self.acollection_exists()
        if exists:
            if self._config.recreate_if_exists:
                logger.info(f"Index '{self._index_name}' exists, recreating...")
                await self.adelete_collection()
            else:
                logger.info(f"Index '{self._index_name}' already exists")
                return

        try:
            # Build schema fields
            schema: List[Any] = []

            # Vector field
            dim = self._config.vector_size
            metric = self._get_distance_metric()

            if isinstance(self._config.index, HNSWIndexConfig):
                hnsw_cfg = self._config.index
                attrs = VectorFieldAttributesHnsw(
                    dimensions=dim,
                    distance_metric=metric,
                    type=VectorType.FLOAT32,
                    number_of_edges=hnsw_cfg.m,
                    vectors_examined_on_construction=hnsw_cfg.ef_construction,
                    vectors_examined_on_runtime=self._config.ef_runtime,
                )
                schema.append(VectorField(
                    name=self._config.vector_field_name,
                    algorithm=VectorAlgorithm.HNSW,
                    attributes=attrs,
                ))
            else:
                attrs_flat = VectorFieldAttributesFlat(
                    dimensions=dim,
                    distance_metric=metric,
                    type=VectorType.FLOAT32,
                )
                schema.append(VectorField(
                    name=self._config.vector_field_name,
                    algorithm=VectorAlgorithm.FLAT,
                    attributes=attrs_flat,
                ))

            # Text field for full-text search
            schema.append(TextField(name=self._config.text_field_name))

            # Tag fields for filtering and existence checks
            for tag_name in [
                "chunk_id", "document_id", "document_name",
                "doc_content_hash", "chunk_content_hash", "knowledge_base_id",
            ]:
                schema.append(TagField(name=tag_name, separator="|"))

            # Create options with prefix
            options = FtCreateOptions(prefixes=[self._key_prefix])

            await ft.create(self.client, self._index_name, schema, options)
            logger.info(f"Successfully created index '{self._index_name}'")

        except Exception as e:
            if "Index already exists" in str(e):
                logger.info(f"Index '{self._index_name}' already exists")
                return
            logger.error(f"Failed to create index: {e}")
            raise VectorDBError(f"Index creation failed: {e}") from e

    async def adelete_collection(self) -> None:
        """Drop the FT index. Does NOT delete the underlying hash keys."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        try:
            await ft.dropindex(self.client, self._index_name)
            logger.info(f"Successfully dropped index '{self._index_name}'")
        except Exception as e:
            if "Unknown index" in str(e) or "not found" in str(e).lower():
                raise CollectionDoesNotExistError(
                    f"Index '{self._index_name}' does not exist"
                ) from e
            raise VectorDBError(f"Failed to drop index: {e}") from e

    async def acollection_exists(self) -> bool:
        """Check if the FT index exists."""
        if self.client is None:
            return False
        try:
            indexes = await ft.list(self.client)
            names = {
                i.decode() if isinstance(i, bytes) else str(i)
                for i in (indexes or [])
            }
            return self._index_name in names
        except Exception:
            return False

    # ========================================================================
    # Data Operations
    # ========================================================================

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
        """Upsert vectors and metadata as Valkey Hash keys."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        if vectors is None or len(vectors) == 0:
            info_log("Nothing to upsert: no vectors provided.", context="ValkeyDB")
            return

        n = len(vectors)

        # Validate lengths
        for name, arr in (
            ("payloads", payloads),
            ("ids", ids),
            ("chunks", chunks),
            ("document_ids", document_ids),
            ("document_names", document_names),
            ("doc_content_hashes", doc_content_hashes),
            ("chunk_content_hashes", chunk_content_hashes),
            ("knowledge_base_ids", knowledge_base_ids),
        ):
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"Length mismatch: '{name}' has {len(arr)} items, expected {n}"
                )

        try:
            records: List[Dict[str, Any]] = []
            keys: List[str] = []
            for i in range(n):
                content = chunks[i] if chunks and i < len(chunks) else ""
                chunk_id = str(ids[i]) if ids and i < len(ids) else str(_uuid.uuid4())
                chunk_hash = (
                    chunk_content_hashes[i]
                    if chunk_content_hashes and i < len(chunk_content_hashes)
                    else hashlib.md5(content.encode("utf-8")).hexdigest()
                )
                doc_id = document_ids[i] if document_ids and i < len(document_ids) else ""
                doc_name = document_names[i] if document_names and i < len(document_names) else ""
                doc_hash = doc_content_hashes[i] if doc_content_hashes and i < len(doc_content_hashes) else ""
                kbi = knowledge_base_ids[i] if knowledge_base_ids and i < len(knowledge_base_ids) else ""

                # Build metadata from payload
                payload = payloads[i] if payloads and i < len(payloads) else {}
                combined_metadata: Dict[str, Any] = {}
                nested_meta = payload.get("metadata")
                if isinstance(nested_meta, dict):
                    combined_metadata.update(nested_meta)
                for key, value in payload.items():
                    if key in self._STANDARD_FIELDS or key == "metadata":
                        continue
                    combined_metadata[key] = value
                if metadata:
                    combined_metadata.update(metadata)

                # Build hash fields
                vector_bytes = self._vector_to_bytes(vectors[i])
                hash_key = self._make_key(chunk_id)

                field_map: Dict[str, Any] = {
                    self._config.vector_field_name: vector_bytes,
                    self._config.text_field_name: content,
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "doc_content_hash": doc_hash,
                    "chunk_content_hash": chunk_hash,
                    "knowledge_base_id": kbi,
                    "metadata": json.dumps(combined_metadata) if combined_metadata else "{}",
                }

                records.append(field_map)
                keys.append(hash_key)

            # Batch upsert using asyncio.gather in chunks of batch_size
            batch_size = self._config.batch_size
            for batch_start in range(0, len(records), batch_size):
                batch_end = min(batch_start + batch_size, len(records))
                coros = [
                    self.client.hset(keys[i], records[i])
                    for i in range(batch_start, batch_end)
                ]
                await asyncio.gather(*coros)

            logger.info(f"Successfully upserted {n} records")

        except UpsertError:
            raise
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise UpsertError(f"Failed to upsert data: {e}") from e

    async def adelete(self, ids: List[Union[str, int]]) -> None:
        """Delete records by their chunk IDs (deletes the hash keys)."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        if not ids:
            return

        try:
            keys = [self._make_key(str(i)) for i in ids]
            await self.client.unlink(keys)
            logger.info(f"Deleted {len(keys)} records")
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise VectorDBError(f"Failed to delete records: {e}") from e

    async def afetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        """Fetch records by their chunk IDs."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        if not ids:
            return []

        results: List[VectorSearchResult] = []
        try:
            for record_id in ids:
                key = self._make_key(str(record_id))
                data = await self.client.hgetall(key)
                if not data:
                    continue

                # Decode the hash fields
                fields = self._decode_hash_fields(data)
                vector = None
                vector_bytes = fields.get(self._config.vector_field_name)
                if isinstance(vector_bytes, bytes):
                    vector = self._bytes_to_vector(vector_bytes)

                results.append(VectorSearchResult(
                    id=fields.get("chunk_id", str(record_id)),
                    score=1.0,
                    payload=self._fields_to_payload(fields),
                    vector=vector,
                    text=fields.get(self._config.text_field_name, ""),
                ))

            return results

        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            raise VectorDBError(f"Failed to fetch records: {e}") from e

    def _decode_hash_fields(self, data: Any) -> Dict[str, Any]:
        """Decode hash field data from GLIDE response to a string-keyed dict."""
        decoded: Dict[str, Any] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                key_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                # Keep vector field as bytes for later conversion
                if key_str == self._config.vector_field_name:
                    decoded[key_str] = v if isinstance(v, bytes) else v
                else:
                    decoded[key_str] = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        return decoded

    def _fields_to_payload(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Convert decoded hash fields to the standard payload format."""
        meta_str = fields.get("metadata", "{}")
        try:
            meta = json.loads(meta_str) if isinstance(meta_str, str) else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        return {
            "chunk_id": fields.get("chunk_id", ""),
            "document_id": fields.get("document_id", ""),
            "document_name": fields.get("document_name", ""),
            "content": fields.get(self._config.text_field_name, ""),
            "doc_content_hash": fields.get("doc_content_hash", ""),
            "chunk_content_hash": fields.get("chunk_content_hash", ""),
            "knowledge_base_id": fields.get("knowledge_base_id", ""),
            "metadata": meta,
        }

    # ========================================================================
    # Search Operations
    # ========================================================================

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
        """Route to the appropriate search method based on inputs."""
        top_k = top_k if top_k is not None else self._config.default_top_k
        similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else self._config.default_similarity_threshold
        )

        has_vector = query_vector is not None
        has_text = query_text is not None and query_text.strip() != ""

        if has_vector and has_text:
            if not self._config.hybrid_search_enabled:
                raise ConfigurationError("Hybrid search is disabled in configuration")
            return await self.ahybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                alpha=alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )
        elif has_vector:
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled in configuration")
            return await self.adense_search(
                query_vector=query_vector,
                top_k=top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
            )
        elif has_text:
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled in configuration")
            return await self.afull_text_search(
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )
        else:
            raise ConfigurationError(
                "Must provide either query_vector, query_text, or both"
            )

    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        """KNN vector similarity search using FT.SEARCH."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        try:
            # Build filter expression
            filter_expr = self._build_filter_expression(filter) if filter else "*"

            # KNN query: filter=>[KNN top_k @vector_field $query_vec]
            query = (
                f"({filter_expr})=>[KNN {top_k} "
                f"@{self._config.vector_field_name} $query_vec]"
            )

            vector_bytes = self._vector_to_bytes(query_vector)
            options = FtSearchOptions(
                params={"query_vec": vector_bytes},
                limit=FtSearchLimit(offset=0, count=top_k),
            )

            raw_results = await ft.search(
                self.client, self._index_name, query, options
            )

            results = self._parse_search_results(raw_results, similarity_threshold)
            logger.debug(f"Dense search returned {len(results)} results")
            return results

        except Exception as e:
            if "no such index" in str(e).lower() or "unknown index" in str(e).lower():
                raise CollectionDoesNotExistError(
                    f"Index '{self._index_name}' does not exist"
                ) from e
            logger.error(f"Dense search failed: {e}")
            raise SearchError(f"Dense search failed: {e}") from e

    async def afull_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Full-text search using FT.SEARCH with text query."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        try:
            # Escape special characters in query text
            escaped_text = self._escape_query_text(query_text)

            # Build query with optional filter
            if filter:
                filter_expr = self._build_filter_expression(filter)
                query = f"({filter_expr}) (@{self._config.text_field_name}:{escaped_text})"
            else:
                query = f"@{self._config.text_field_name}:{escaped_text}"

            options = FtSearchOptions(
                limit=FtSearchLimit(offset=0, count=top_k),
            )

            raw_results = await ft.search(
                self.client, self._index_name, query, options
            )

            results = self._parse_search_results(raw_results, similarity_threshold)
            logger.debug(f"Full-text search returned {len(results)} results")
            return results

        except Exception as e:
            if "no such index" in str(e).lower() or "unknown index" in str(e).lower():
                raise CollectionDoesNotExistError(
                    f"Index '{self._index_name}' does not exist"
                ) from e
            logger.error(f"Full-text search failed: {e}")
            raise SearchError(f"Full-text search failed: {e}") from e

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
        """Hybrid search combining dense + full-text via RRF fusion."""
        fusion_method = fusion_method or self._config.default_fusion_method

        # Execute both searches with expanded top_k
        expanded_k = top_k * 2

        vector_results = await self.adense_search(
            query_vector=query_vector,
            top_k=expanded_k,
            filter=filter,
        )

        text_results = await self.afull_text_search(
            query_text=query_text,
            top_k=expanded_k,
            filter=filter,
        )

        # RRF fusion
        k = self._config.rrf_k
        vector_ranks: Dict[Any, int] = {r.id: i + 1 for i, r in enumerate(vector_results)}
        text_ranks: Dict[Any, int] = {r.id: i + 1 for i, r in enumerate(text_results)}

        rrf_scores: Dict[Any, float] = {}
        all_ids = set(vector_ranks.keys()) | set(text_ranks.keys())

        for doc_id in all_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += 1.0 / (k + vector_ranks[doc_id])
            if doc_id in text_ranks:
                score += 1.0 / (k + text_ranks[doc_id])
            rrf_scores[doc_id] = score

        # Normalize scores to [0, 1]
        if rrf_scores:
            max_score = max(rrf_scores.values())
            min_score = min(rrf_scores.values())
            score_range = max_score - min_score
            if score_range > 0:
                rrf_scores = {
                    did: (s - min_score) / score_range
                    for did, s in rrf_scores.items()
                }
            else:
                rrf_scores = {did: 1.0 for did in rrf_scores}

        # Sort by score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build result map
        result_map: Dict[Any, VectorSearchResult] = {}
        for r in vector_results + text_results:
            if r.id not in result_map:
                result_map[r.id] = r

        results: List[VectorSearchResult] = []
        for doc_id in sorted_ids[:top_k]:
            result = result_map[doc_id]
            results.append(VectorSearchResult(
                id=result.id,
                score=rrf_scores[doc_id],
                payload=result.payload,
                vector=result.vector,
                text=result.text,
            ))

        if similarity_threshold is not None:
            results = [r for r in results if r.score >= similarity_threshold]

        logger.debug(f"Hybrid search (RRF) returned {len(results)} results")
        return results

    # ========================================================================
    # Search Helpers
    # ========================================================================

    # Allowed TAG field names for filter expressions (security: prevents query injection)
    _ALLOWED_FILTER_FIELDS = frozenset({
        'chunk_id', 'document_id', 'document_name',
        'doc_content_hash', 'chunk_content_hash', 'knowledge_base_id',
    })

    def _build_filter_expression(self, filter: Dict[str, Any]) -> str:
        """Build a Valkey Search filter expression from a dict."""
        clauses: List[str] = []
        for key, value in filter.items():
            if key.startswith("metadata."):
                # Metadata fields are stored as JSON; can't filter directly
                continue
            if key not in self._ALLOWED_FILTER_FIELDS:
                raise ValueError(f"Invalid filter field: {key!r}")
            # TAG field filter: @field:{value}
            # Escape all Valkey Search special characters in TAG values
            escaped_val = self._escape_tag_value(str(value))
            clauses.append(f"@{key}:{{{escaped_val}}}")

        if not clauses:
            return "*"
        return " ".join(clauses)

    @staticmethod
    def _escape_tag_value(value: str) -> str:
        """Escape special characters in a TAG field value for Valkey Search.

        With separator='|' on TAG fields, most punctuation is safe.
        Only Valkey Search query syntax characters need escaping.
        """
        # Characters that have special meaning in FT.SEARCH query syntax
        special_chars = r'{}[]\\"|'
        escaped = value
        for char in special_chars:
            escaped = escaped.replace(char, f"\\{char}")
        return escaped

    def _escape_query_text(self, text: str) -> str:
        """Escape special characters for Valkey Search query syntax."""
        # Valkey Search special chars that need escaping
        special_chars = r',.<>{}[]"\'`:;!@#$%^&*()-+=~'
        escaped = text
        for char in special_chars:
            escaped = escaped.replace(char, f"\\{char}")
        # Wrap individual words for OR matching
        words = escaped.split()
        if len(words) > 1:
            return " | ".join(words)
        return escaped if escaped else "*"

    def _parse_search_results(
        self,
        raw_results: Any,
        similarity_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """Parse FT.SEARCH response into VectorSearchResult list."""
        results: List[VectorSearchResult] = []

        if not raw_results or not isinstance(raw_results, list):
            return results

        # FT.SEARCH returns: [total_count, {key: {field: value, ...}}, ...]
        # In GLIDE format: [count, mapping_of_key_to_fields]
        if len(raw_results) < 2:
            return results

        total_count = raw_results[0] if isinstance(raw_results[0], int) else 0
        if total_count == 0:
            return results

        # The second element is a mapping of key -> fields
        docs_map = raw_results[1] if len(raw_results) > 1 else {}

        if not isinstance(docs_map, dict):
            return results

        for key, fields_data in docs_map.items():
            key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)

            # Decode fields
            fields: Dict[str, Any] = {}
            if isinstance(fields_data, dict):
                for fk, fv in fields_data.items():
                    fk_str = fk.decode("utf-8") if isinstance(fk, bytes) else str(fk)
                    if fk_str == self._config.vector_field_name:
                        fields[fk_str] = fv  # Keep as bytes
                    else:
                        fields[fk_str] = fv.decode("utf-8") if isinstance(fv, bytes) else str(fv)

            # Extract score from __vector_score or __score field
            score_str = fields.pop("__vector_score", None) or fields.pop("__score", None)
            if score_str is not None:
                try:
                    distance = float(score_str)
                    score = self._distance_to_similarity(distance)
                except (ValueError, TypeError):
                    score = 0.0
            else:
                score = 1.0  # Full-text results without explicit score

            if similarity_threshold is not None and score < similarity_threshold:
                continue

            # Extract vector if present
            vector = None
            vector_data = fields.get(self._config.vector_field_name)
            if isinstance(vector_data, bytes):
                vector = self._bytes_to_vector(vector_data)

            chunk_id = fields.get("chunk_id", "")
            if not chunk_id:
                # Extract from key: remove prefix
                chunk_id = key_str.removeprefix(self._key_prefix)

            results.append(VectorSearchResult(
                id=chunk_id,
                score=score,
                payload=self._fields_to_payload(fields),
                vector=vector,
                text=fields.get(self._config.text_field_name, ""),
            ))

        return results

    # ========================================================================
    # Existence Checks
    # ========================================================================

    async def _field_exists(self, field_name: str, field_value: str) -> bool:
        """Check if any record with the given TAG field value exists."""
        if self.client is None:
            return False
        try:
            escaped_val = self._escape_tag_value(field_value)
            query = f"@{field_name}:{{{escaped_val}}}"
            options = FtSearchOptions(limit=FtSearchLimit(offset=0, count=0))
            raw = await ft.search(self.client, self._index_name, query, options)
            if raw and isinstance(raw, list) and len(raw) > 0:
                count = raw[0] if isinstance(raw[0], int) else 0
                return count > 0
            return False
        except Exception:
            return False

    async def adocument_id_exists(self, document_id: str) -> bool:
        return await self._field_exists("document_id", document_id)

    async def adocument_name_exists(self, document_name: str) -> bool:
        return await self._field_exists("document_name", document_name)

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        return await self._field_exists("chunk_id", chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        return await self._field_exists("doc_content_hash", doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        return await self._field_exists("chunk_content_hash", chunk_content_hash)

    # ========================================================================
    # Delete by Field
    # ========================================================================

    async def _delete_by_field(self, field_name: str, field_value: str) -> bool:
        """Find and delete all records matching a TAG field value."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        try:
            escaped_val = self._escape_tag_value(field_value)
            query = f"@{field_name}:{{{escaped_val}}}"
            # Fetch matching keys (up to 10000)
            options = FtSearchOptions(limit=FtSearchLimit(offset=0, count=10000))
            raw = await ft.search(self.client, self._index_name, query, options)

            if not raw or not isinstance(raw, list) or len(raw) < 2:
                return True  # Nothing to delete

            count = raw[0] if isinstance(raw[0], int) else 0
            if count == 0:
                return True

            docs_map = raw[1] if len(raw) > 1 else {}
            if isinstance(docs_map, dict):
                keys_to_delete = [
                    key.decode("utf-8") if isinstance(key, bytes) else str(key)
                    for key in docs_map.keys()
                ]
                if keys_to_delete:
                    await self.client.unlink(keys_to_delete)

            logger.info(f"Deleted records with {field_name}='{field_value}'")
            return True

        except Exception as e:
            logger.error(f"Delete by {field_name} failed: {e}")
            return False

    async def adelete_by_document_name(self, document_name: str) -> bool:
        return await self._delete_by_field("document_name", document_name)

    async def adelete_by_document_id(self, document_id: str) -> bool:
        return await self._delete_by_field("document_id", document_id)

    async def adelete_by_chunk_id(self, chunk_id: str) -> bool:
        return await self._delete_by_field("chunk_id", chunk_id)

    async def adelete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        return await self._delete_by_field("doc_content_hash", doc_content_hash)

    async def adelete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        return await self._delete_by_field("chunk_content_hash", chunk_content_hash)

    async def adelete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete by metadata — uses TAG field filters where possible."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        try:
            filter_expr = self._build_filter_expression(metadata)
            if filter_expr == "*":
                logger.warning("Cannot delete by metadata: no filterable fields")
                return False

            options = FtSearchOptions(limit=FtSearchLimit(offset=0, count=10000))
            raw = await ft.search(self.client, self._index_name, filter_expr, options)

            if not raw or not isinstance(raw, list) or len(raw) < 2:
                return True

            docs_map = raw[1] if len(raw) > 1 else {}
            if isinstance(docs_map, dict):
                keys_to_delete = [
                    key.decode("utf-8") if isinstance(key, bytes) else str(key)
                    for key in docs_map.keys()
                ]
                if keys_to_delete:
                    await self.client.unlink(keys_to_delete)

            return True

        except Exception as e:
            logger.error(f"Delete by metadata failed: {e}")
            return False

    # ========================================================================
    # Update Metadata
    # ========================================================================

    async def aupdate_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a record identified by chunk_id."""
        if self.client is None:
            raise VectorDBConnectionError("Not connected to Valkey")

        try:
            key = self._make_key(chunk_id)
            # Read existing metadata
            existing = await self.client.hget(key, "metadata")
            if existing is None:
                logger.warning(f"Record with chunk_id '{chunk_id}' not found")
                return False

            existing_str = existing.decode("utf-8") if isinstance(existing, bytes) else str(existing)
            try:
                current_meta = json.loads(existing_str)
            except (json.JSONDecodeError, TypeError):
                current_meta = {}

            # Merge
            current_meta.update(metadata)
            await self.client.hset(key, {"metadata": json.dumps(current_meta)})
            logger.info(f"Updated metadata for chunk_id '{chunk_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False

    # ========================================================================
    # Optimize
    # ========================================================================

    async def aoptimize(self) -> bool:
        """Optimization is handled automatically by Valkey Search. No-op."""
        logger.info("Valkey Search handles index optimization automatically.")
        return True

    # ========================================================================
    # Supported Search Types
    # ========================================================================

    async def aget_supported_search_types(self) -> List[str]:
        supported: List[str] = []
        if self._config.dense_search_enabled:
            supported.append("dense")
        if self._config.full_text_search_enabled:
            supported.append("full_text")
        if self._config.hybrid_search_enabled:
            supported.append("hybrid")
        return supported

    # ========================================================================
    # Repr
    # ========================================================================

    def __repr__(self) -> str:
        return (
            f"ValkeyProvider(index='{self._index_name}', "
            f"vector_size={self._config.vector_size}, "
            f"metric={self._config.distance_metric.value}, "
            f"algorithm={'HNSW' if isinstance(self._config.index, HNSWIndexConfig) else 'FLAT'})"
        )
