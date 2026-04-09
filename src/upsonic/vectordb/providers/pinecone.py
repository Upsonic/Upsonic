import time
import asyncio
import hashlib
import json
import uuid as _uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, Generator

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec, PodSpec
    from pinecone.exceptions import PineconeApiException as ApiException, NotFoundException
    _PINECONE_AVAILABLE = True
except ImportError:
    pinecone = None
    Pinecone = None
    ServerlessSpec = None
    PodSpec = None
    ApiException = None
    NotFoundException = None
    _PINECONE_AVAILABLE = False

try:
    from pinecone_text.sparse import BM25Encoder
    _BM25_AVAILABLE = True
except ImportError:
    BM25Encoder = None
    _BM25_AVAILABLE = False


from upsonic.vectordb.base import BaseVectorDBProvider

from upsonic.vectordb.config import (
    PineconeConfig,
    DistanceMetric
)

from upsonic.utils.package.exception import(
    VectorDBConnectionError,
    ConfigurationError,
    VectorDBError,
    SearchError,
    UpsertError
)

from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.utils.logging_config import get_logger
from upsonic.utils.printing import info_log, debug_log

logger = get_logger(__name__)


class PineconeProvider(BaseVectorDBProvider):
    """
    High-level, comprehensive, async-first vector database provider for Pinecone.

    This provider uses a SINGLE INDEX approach for hybrid search (following Pinecone best practices),
    where both dense and sparse vectors are stored in the same index.

    Key Features:
    - Single index for hybrid dense+sparse vectors
    - Chunk-primary metadata: chunk_id, chunk_content_hash, content, document_name, document_id, doc_content_hash
    - Two-layer deduplication: document-level (KnowledgeBase) + chunk-level (vector DB)
    - Dynamic indexing configuration
    - Hybrid vector scaling (alpha-weighted combination)
    - Batch processing for efficient operations
    - Full async/await support with auto-reconnecting client lifecycle
    - Support for ServerlessSpec, PodSpec, and dict specs
    - Automatic score normalization to [0, 1] range for all metrics

    Score Normalization:
    - **dotproduct**: Clips scores to [0, 1] (handles floating-point precision errors)
    - **cosine**: Clips scores to [0, 1] (already normalized by Pinecone)
    - **euclidean**: Converts distance to similarity using 1/(1+distance)

    Important Notes:
    - For **dotproduct metric**, vectors MUST be L2-normalized (unit vectors) for proper 0-1 scores
    - For **hybrid search**, dotproduct metric is REQUIRED and auto-configured
    - Sparse vectors are auto-generated using BM25Encoder if not provided
    """

    _DISTANCE_METRIC_MAP: Dict[DistanceMetric, str] = {
        DistanceMetric.COSINE: "cosine",
        DistanceMetric.EUCLIDEAN: "euclidean",
        DistanceMetric.DOT_PRODUCT: "dotproduct",
    }

    _STANDARD_FIELDS: frozenset = frozenset({
        'chunk_id', 'document_id', 'doc_content_hash',
        'chunk_content_hash', 'document_name', 'content', 'metadata',
        'knowledge_base_id',
    })


    def __init__(self, config: Union[PineconeConfig, Dict[str, Any]]):
        if not _PINECONE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pinecone",
                install_command='pip install "upsonic[pinecone]"',
                feature_name="Pinecone vector database provider"
            )

        if isinstance(config, dict):
            config = PineconeConfig.from_dict(config)
        
        super().__init__(config)
        
        self._index: Optional[object] = None
        self.client: Optional[object] = None
        self._is_connected = False
        
        self._sparse_encoder: Optional[object] = None
        if self._config.hybrid_search_enabled or self._config.use_sparse_vectors:
            if _BM25_AVAILABLE:
                self._sparse_encoder = BM25Encoder().default()
                logger.info("BM25 sparse encoder initialized for hybrid search.")
            else:
                logger.warning("pinecone-text not available. Install with: pip install pinecone-text")
        
        self._validate_configuration()

    # ========================================================================
    # Private Helpers
    # ========================================================================

    def _validate_configuration(self) -> None:
        logger.debug("Performing comprehensive Pinecone configuration validation...")

        if not self._config.api_key:
            raise ConfigurationError("Configuration Error: 'api_key' is mandatory for Pinecone.")
        
        if not self._config.spec and not self._config.environment:
            raise ConfigurationError("Either 'spec' or 'environment' must be provided.")
        
        if self._config.hybrid_search_enabled and self._config.metric != 'dotproduct':
            logger.warning("Hybrid search works best with dotproduct metric.")
        
        logger.info("Pinecone configuration validated successfully.")

    def _sanitize_collection_name(self, collection_name: str) -> str:
        import re
        sanitized = re.sub(r'[^a-z0-9-]', '-', collection_name.lower())
        sanitized = re.sub(r'-+', '-', sanitized)
        sanitized = sanitized.strip('-')
        if not sanitized:
            sanitized = "default-collection"
        return sanitized

    def _generate_provider_id(self) -> str:
        identifier_parts: List[Optional[str]] = [
            self._config.collection_name,
            self._config.environment or self._config.host or "cloud",
        ]
        identifier: str = "#".join(filter(None, identifier_parts))
        return hashlib.md5(identifier.encode()).hexdigest()[:16]

    def _build_metadata(
        self,
        payload: Dict[str, Any],
        content: str,
        chunk_id: str,
        chunk_content_hash: str,
        document_id: str = "",
        document_name: str = "",
        doc_content_hash: str = "",
        knowledge_base_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        _warned_standard_keys: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Build the Pinecone metadata dict (flat primitives only) with strict
        payload-contract compliance.

        Standard fields are taken ONLY from dedicated parameters. Non-standard
        keys from ``payload`` and the nested ``payload["metadata"]`` dict are
        merged into a single ``metadata`` JSON-serialized Pinecone key. Content
        is stored in the flat ``content`` metadata key (Pinecone has no native
        document field).
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
                            context="PineconeVectorDB",
                        )
                        _warned_standard_keys.add(key)
                    continue
                combined_user_metadata[key] = value

        if extra_metadata:
            combined_user_metadata.update(extra_metadata)

        pinecone_metadata: Dict[str, Any] = {
            "chunk_id": chunk_id or "",
            "chunk_content_hash": chunk_content_hash or "",
            "content": content or "",
            "document_id": document_id or "",
            "doc_content_hash": doc_content_hash or "",
            "document_name": document_name or "",
            "knowledge_base_id": knowledge_base_id or "",
            "metadata": json.dumps(combined_user_metadata),
        }

        return pinecone_metadata

    def _hydrate_payload(self, pinecone_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a contract-compliant VectorSearchResult.payload from Pinecone's
        flat metadata dict.
        """
        pinecone_metadata = pinecone_metadata or {}
        raw_metadata = pinecone_metadata.get("metadata", "{}")
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
            "chunk_id": pinecone_metadata.get("chunk_id", "") or "",
            "document_id": pinecone_metadata.get("document_id", "") or "",
            "document_name": pinecone_metadata.get("document_name", "") or "",
            "content": pinecone_metadata.get("content", "") or "",
            "doc_content_hash": pinecone_metadata.get("doc_content_hash", "") or "",
            "chunk_content_hash": pinecone_metadata.get("chunk_content_hash", "") or "",
            "knowledge_base_id": pinecone_metadata.get("knowledge_base_id", "") or "",
            "metadata": parsed_metadata,
        }

    def _split_filter(
        self,
        filters: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Split a filter dict into (native_pinecone_filter, python_post_filter).

        Native filter: standard fields (excluding 'metadata' and 'content') that
        Pinecone can match natively at the flat top level of its metadata.

        Post filter: non-standard user fields that live inside the JSON-serialized
        ``metadata`` key and must be matched in Python after fetching.
        """
        if not filters:
            return None, {}

        native_standard_keys = self._STANDARD_FIELDS - {'metadata', 'content'}

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

        return (native_part or None), post_part

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

    def _build_spec(self) -> Union["ServerlessSpec", "PodSpec"]:
        if self._config.spec and not isinstance(self._config.spec, dict):
            return self._config.spec
        
        if self._config.spec and isinstance(self._config.spec, dict):
            spec_dict = self._config.spec
            if 'cloud' in spec_dict and 'region' in spec_dict:
                return ServerlessSpec(**spec_dict)
            elif 'environment' in spec_dict:
                return PodSpec(**spec_dict)
            else:
                return ServerlessSpec(**spec_dict)
        
        if self._config.environment:
            if '-' in self._config.environment:
                parts = self._config.environment.split('-', 1)
                cloud = parts[0]
                region = parts[1]
            else:
                cloud = 'aws'
                region = 'us-east-1'
            return ServerlessSpec(cloud=cloud, region=region)
        
        if self._config.pod_type:
            pod_spec_params: Dict[str, Any] = {
                "environment": self._config.environment or "us-east-1-aws",
                "pod_type": self._config.pod_type
            }
            if self._config.pods:
                pod_spec_params["pods"] = self._config.pods
            if self._config.replicas:
                pod_spec_params["replicas"] = self._config.replicas
            if self._config.shards:
                pod_spec_params["shards"] = self._config.shards
            return PodSpec(**pod_spec_params)
        
        return ServerlessSpec(cloud='aws', region='us-east-1')

    def _create_client(self) -> "Pinecone":
        try:
            client_params: Dict[str, Any] = {
                "api_key": self._config.api_key.get_secret_value()
            }
            if self._config.host:
                client_params["host"] = self._config.host
            if self._config.additional_headers:
                client_params["additional_headers"] = self._config.additional_headers
            if self._config.pool_threads:
                client_params["pool_threads"] = self._config.pool_threads
            if self._config.index_api:
                client_params["index_api"] = self._config.index_api

            return Pinecone(**client_params)
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to create Pinecone client: {e}") from e

    def get_client(self) -> "Pinecone":
        """
        Singleton entry-point for the Pinecone client.

        Creates the client if ``None``, verifies it can reach the Pinecone API
        via ``list_indexes()``, and auto-recovers on failure.
        """
        if self.client is None:
            logger.debug("Creating Pinecone client...")
            self.client = self._create_client()

        try:
            self.client.list_indexes()
        except Exception:
            logger.debug("Existing Pinecone client unresponsive, recreating...")
            self.client = self._create_client()
            try:
                self.client.list_indexes()
            except Exception as e:
                self.client = None
                self._is_connected = False
                raise VectorDBConnectionError(
                    f"Pinecone client not ready after recreation: {e}"
                ) from e

        self._is_connected = True
        return self.client

    def is_client_connected(self) -> bool:
        """Read-only status check.  Does NOT create or reconnect."""
        if self.client is None:
            return False
        try:
            self.client.list_indexes()
            return True
        except Exception:
            return False

    async def ais_client_connected(self) -> bool:
        """Async wrapper for ``is_client_connected()``."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.is_client_connected)

    async def aget_client(self) -> "Pinecone":
        """
        Gets or creates the Pinecone client asynchronously, ensuring it is
        connected and ready.

        Follows a singleton pattern: reuses the existing client if available,
        creates a new one if needed, verifies it can reach the Pinecone API
        via ``list_indexes()``, and auto-recovers on failure.

        Returns:
            A connected and ready Pinecone client instance.

        Raises:
            VectorDBConnectionError: If the client cannot be created or is not ready.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_client)

    # ========================================================================
    # Connection Lifecycle
    # ========================================================================

    async def aconnect(self) -> None:
        """
        Establishes an async connection to the Pinecone service.

        Delegates to aget_client() which handles client creation and
        readiness verification. This method is idempotent.

        Raises:
            VectorDBConnectionError: If the connection fails for any reason.
        """
        if await self.ais_client_connected():
            logger.info("Already connected to Pinecone.")
            return

        logger.debug("Attempting to connect to Pinecone...")

        try:
            client = await self.aget_client()

            safe_collection_name = self._sanitize_collection_name(self._config.collection_name)

            def _attach_index() -> None:
                existing_indexes = client.list_indexes().names()
                if safe_collection_name in existing_indexes:
                    self._index = client.Index(safe_collection_name)
                    logger.debug(f"Connected to existing index: {safe_collection_name}")
                else:
                    logger.debug(f"Index {safe_collection_name} does not exist yet")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _attach_index)

            logger.info("Successfully connected to Pinecone.")
        except (VectorDBConnectionError, ConfigurationError):
            self.client = None
            self._is_connected = False
            raise
        except Exception as e:
            self.client = None
            self._is_connected = False
            raise VectorDBConnectionError(
                f"An unexpected error occurred during connection: {e}"
            ) from e

    async def adisconnect(self) -> None:
        logger.debug("Disconnecting from Pinecone...")
        self.client = None
        self._index = None
        self._is_connected = False
        logger.info("Disconnected from Pinecone.")

    async def ais_ready(self) -> bool:
        return await self.ais_client_connected()

    # ========================================================================
    # Collection Lifecycle
    # ========================================================================

    async def acollection_exists(self) -> bool:
        client = await self.aget_client()

        def _exists() -> bool:
            try:
                existing_indexes = client.list_indexes().names()
                safe_collection_name = self._sanitize_collection_name(self._config.collection_name)
                return safe_collection_name in existing_indexes
            except ApiException as e:
                raise VectorDBError(f"Failed to check collection existence: {e}") from e

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _exists)

    async def adelete_collection(self) -> None:
        client = await self.aget_client()

        collection_name = self._config.collection_name
        safe_collection_name = self._sanitize_collection_name(collection_name)
        logger.warning(f"Deleting collection: '{safe_collection_name}'")

        def _delete() -> None:
            try:
                if safe_collection_name in client.list_indexes().names():
                    logger.info(f"Deleting index '{safe_collection_name}'...")
                    client.delete_index(safe_collection_name)
                    self._wait_for_deletion_sync(safe_collection_name)
                    logger.info(f"Successfully deleted index '{safe_collection_name}'.")
                else:
                    logger.info(f"Index '{safe_collection_name}' does not exist.")
            except ApiException as e:
                raise VectorDBError(f"Failed to delete collection: {e}") from e

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _delete)

    async def acreate_collection(self) -> None:
        client = await self.aget_client()

        safe_collection_name = self._sanitize_collection_name(self._config.collection_name)

        def _create() -> None:
            if safe_collection_name in client.list_indexes().names():
                if self._config.recreate_if_exists:
                    logger.info(f"Index '{safe_collection_name}' exists. Recreating...")
                    client.delete_index(safe_collection_name)
                    self._wait_for_deletion_sync(safe_collection_name)
                else:
                    logger.info(f"Index '{safe_collection_name}' already exists.")
                    self._index = client.Index(safe_collection_name)
                    return

            logger.info(f"Creating index '{safe_collection_name}'...")
            try:
                spec = self._build_spec()

                create_params: Dict[str, Any] = {
                    "name": safe_collection_name,
                    "dimension": self._config.vector_size,
                    "metric": self._config.metric,
                    "spec": spec,
                }

                if self._config.timeout:
                    create_params["timeout"] = self._config.timeout

                if isinstance(spec, PodSpec) and self._config.indexed_fields:
                    create_params["metadata_config"] = {
                        "indexed": self._config.indexed_fields
                    }

                client.create_index(**create_params)
                self._wait_for_index_ready_sync(safe_collection_name)
                self._index = client.Index(safe_collection_name)
                logger.info(f"Successfully created index '{safe_collection_name}'.")
            except ApiException as e:
                raise VectorDBError(f"Failed to create index: {e}") from e

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _create)

    def _wait_for_index_ready_sync(self, index_name: str) -> None:
        wait_timeout: int = 600
        start_time: float = time.time()

        while True:
            try:
                status = self.client.describe_index(index_name)
                if status['status']['ready']:
                    break
            except ApiException:
                pass

            if time.time() - start_time > wait_timeout:
                raise VectorDBError(f"Timeout waiting for index '{index_name}' to be ready.")
            logger.debug(f"Waiting for index '{index_name}' to be ready...")
            time.sleep(10)

    def _wait_for_deletion_sync(self, index_name: str) -> None:
        wait_timeout: int = 300
        start_time: float = time.time()

        while index_name in self.client.list_indexes().names():
            if time.time() - start_time > wait_timeout:
                raise VectorDBError(f"Timeout waiting for index '{index_name}' to be deleted.")
            logger.debug(f"Waiting for index '{index_name}' to be deleted...")
            time.sleep(5)

    # ========================================================================
    # Data Operations
    # ========================================================================

    def _generate_batches(
        self,
        records: List[Dict[str, Any]],
        batch_size: int,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        batch: List[Dict[str, Any]] = []
        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _batch_find_existing_by_hashes(
        self,
        chunk_hashes: List[str],
    ) -> Dict[str, List[str]]:
        if not chunk_hashes or not self._index:
            return {}

        unique_hashes: List[str] = list(set(chunk_hashes))
        dummy_vector: List[float] = [0.0] * self._config.vector_size

        try:
            result = self._index.query(
                vector=dummy_vector,
                filter={"chunk_content_hash": {"$in": unique_hashes}},
                top_k=10000,
                include_metadata=True,
                include_values=False,
                namespace=self._config.namespace or "",
            )
            matches = result.matches if hasattr(result, "matches") else result.get("matches", [])

            existing: Dict[str, List[str]] = {}
            for match in matches:
                match_id: str = match.id if hasattr(match, "id") else match["id"]
                meta: Dict[str, Any] = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})
                h: str = meta.get("chunk_content_hash", "")
                if h:
                    existing.setdefault(h, []).append(match_id)
            return existing
        except Exception as e:
            logger.debug(f"Batch hash lookup failed, treating all as new: {e}")
            return {}

    def _hybrid_scale(
        self,
        dense: List[float],
        sparse: Dict[str, Any],
        alpha: float
    ) -> tuple:
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        hsparse: Dict[str, Any] = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]]
        }
        hdense: List[float] = [v * alpha for v in dense]
        
        return hdense, hsparse

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
        await self.aget_client()
        if not self._index:
            raise VectorDBConnectionError("Index not available. Call 'create_collection' first.")

        if (vectors is None or len(vectors) == 0) and (sparse_vectors is None or len(sparse_vectors) == 0):
            info_log("Nothing to upsert: no dense or sparse vectors provided.", context="PineconeVectorDB")
            return

        n: int = len(vectors) if vectors else len(sparse_vectors)  # type: ignore[arg-type]

        # Length validation for all provided per-item arrays
        for name, arr in (
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
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"Length mismatch: '{name}' has {len(arr)} items, expected {n}"
                )

        logger.info(f"Upserting {n} records...")

        namespace: str = self._config.namespace or ""
        batch_size: int = self._config.batch_size
        show_progress: bool = self._config.show_progress
        extra_metadata: Dict[str, Any] = metadata or {}

        _warned_standard_keys: set = set()

        def _upsert() -> None:
            try:
                computed_hashes: List[str] = []
                for i in range(n):
                    payload: Dict[str, Any] = payloads[i] if payloads and i < len(payloads) else {}
                    content_text: str = chunks[i] if chunks and i < len(chunks) else ""
                    if chunk_content_hashes and i < len(chunk_content_hashes) and chunk_content_hashes[i]:
                        computed_hashes.append(chunk_content_hashes[i])
                    else:
                        computed_hashes.append(
                            hashlib.md5(content_text.encode("utf-8")).hexdigest()
                        )

                existing_hash_map: Dict[str, List[str]] = self._batch_find_existing_by_hashes(
                    computed_hashes
                )

                stale_ids: List[str] = []
                for i in range(n):
                    new_id: str = str(ids[i]) if ids and i < len(ids) else str(_uuid.uuid4())
                    old_ids: List[str] = existing_hash_map.get(computed_hashes[i], [])
                    for old_id in old_ids:
                        if old_id != new_id:
                            stale_ids.append(old_id)

                if stale_ids:
                    for start in range(0, len(stale_ids), 1000):
                        batch_ids: List[str] = stale_ids[start : start + 1000]
                        try:
                            self._index.delete(ids=batch_ids, namespace=namespace)
                            logger.debug(f"Deleted {len(batch_ids)} stale duplicate(s).")
                        except Exception as e:
                            logger.warning(f"Failed to delete stale duplicates: {e}")

                records: List[Dict[str, Any]] = []
                for i in range(n):
                    payload = payloads[i] if payloads and i < len(payloads) else {}
                    content_text = chunks[i] if chunks and i < len(chunks) else ""
                    chunk_id_str: str = str(ids[i]) if ids and i < len(ids) else str(_uuid.uuid4())
                    doc_id: str = document_ids[i] if document_ids and i < len(document_ids) else ""
                    doc_name: str = document_names[i] if document_names and i < len(document_names) else ""
                    doc_hash: str = doc_content_hashes[i] if doc_content_hashes and i < len(doc_content_hashes) else ""
                    chunk_hash: str = computed_hashes[i]
                    kbi: Optional[str] = knowledge_base_ids[i] if knowledge_base_ids else None

                    metadata: Dict[str, Any] = self._build_metadata(
                        payload=payload,
                        content=content_text,
                        chunk_id=chunk_id_str,
                        chunk_content_hash=chunk_hash,
                        document_id=doc_id,
                        document_name=doc_name,
                        doc_content_hash=doc_hash,
                        knowledge_base_id=kbi,
                        extra_metadata=extra_metadata or None,
                        _warned_standard_keys=_warned_standard_keys,
                    )

                    record: Dict[str, Any] = {
                        "id": chunk_id_str,
                        "metadata": metadata,
                    }

                    if vectors and i < len(vectors):
                        import numpy as np
                        record["values"] = np.array(vectors[i], dtype=np.float32).tolist()

                    if self._config.hybrid_search_enabled:
                        if sparse_vectors and i < len(sparse_vectors):
                            record["sparse_values"] = sparse_vectors[i]
                        elif self._sparse_encoder and content_text:
                            try:
                                record["sparse_values"] = self._sparse_encoder.encode_documents(content_text)
                            except Exception as e:
                                logger.warning(f"Failed to generate sparse vectors for record {chunk_id_str}: {e}")

                    records.append(record)

                for batch in self._generate_batches(records, batch_size):
                    self._index.upsert(
                        vectors=batch,
                        namespace=namespace,
                        batch_size=batch_size,
                        show_progress=show_progress,
                    )

                logger.info(f"Successfully upserted {n} records.")
            except (UpsertError, VectorDBConnectionError):
                raise
            except ApiException as e:
                raise UpsertError(f"Upsert failed: {e}") from e
            except Exception as e:
                raise UpsertError(f"General upsert error: {e}") from e

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _upsert)

    async def adelete(self, ids: List[Union[str, int]]) -> None:
        await self.aget_client()
        if not self._index:
            raise VectorDBConnectionError("Index not available.")

        if not ids:
            return

        str_ids: List[str] = [str(i) for i in ids]
        namespace: str = self._config.namespace or ""

        def _delete() -> None:
            try:
                response = self._index.fetch(ids=str_ids, namespace=namespace)
                fetched_vectors = (
                    response.vectors
                    if hasattr(response, "vectors")
                    else response.get("vectors", {})
                )
                existing_ids: List[str] = list(fetched_vectors.keys())

                for sid in str_ids:
                    if sid not in fetched_vectors:
                        logger.debug(
                            f"Record with ID '{sid}' does not exist, skipping deletion."
                        )

                if not existing_ids:
                    logger.info(
                        "No matching records found to delete. No action taken."
                    )
                    return

                self._index.delete(ids=existing_ids, namespace=namespace)
                logger.info(
                    f"Successfully processed deletion for {len(str_ids)} IDs. "
                    f"Existed: {len(existing_ids)}, Deleted: {len(existing_ids)}."
                )
            except ApiException as e:
                raise VectorDBError(f"Delete failed: {e}") from e

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _delete)

    async def adelete_by_filter(self, filter: Dict[str, Any]) -> bool:
        await self.aget_client()
        if not self._index:
            raise VectorDBConnectionError("Index not available.")

        namespace: str = self._config.namespace or ""
        native_filter, post_filter = self._split_filter(filter)
        dummy_vector: List[float] = [0.0] * self._config.vector_size

        def _delete() -> bool:
            try:
                if not post_filter:
                    self._index.delete(filter=native_filter, namespace=namespace)
                    logger.info("Deleted records matching filter.")
                    return True

                # Post-filter path: query candidates, filter in Python, delete by ID.
                response = self._index.query(
                    vector=dummy_vector,
                    filter=native_filter,
                    top_k=10000,
                    include_metadata=True,
                    include_values=False,
                    namespace=namespace,
                )
                results = self._parse_query_response(response)
                matching_ids = [
                    str(r.id) for r in results
                    if self._matches_post_filter(r.payload, post_filter)
                ]
                if matching_ids:
                    for start in range(0, len(matching_ids), 1000):
                        batch_ids = matching_ids[start : start + 1000]
                        self._index.delete(ids=batch_ids, namespace=namespace)
                logger.info(f"Deleted {len(matching_ids)} records matching filter (post-filter).")
                return True
            except ApiException as e:
                logger.warning(f"Filter delete failed: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _delete)

    # ------------------------------------------------------------------
    # Generic field helpers
    # ------------------------------------------------------------------

    async def afield_exists(self, field_name: str, field_value: Any) -> bool:
        try:
            await self.aget_client()
        except VectorDBConnectionError:
            return False
        if not self._index:
            return False

        dummy_vector: List[float] = [0.0] * self._config.vector_size
        namespace: str = self._config.namespace or ""

        def _check() -> bool:
            try:
                result = self._index.query(
                    vector=dummy_vector,
                    filter={field_name: {"$eq": field_value}},
                    top_k=1,
                    include_metadata=False,
                    namespace=namespace,
                )
                matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
                return len(matches) > 0
            except Exception as e:
                logger.debug(f"Error checking if {field_name}='{field_value}' exists: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _check)

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
        if not await self.afield_exists(field_name, field_value):
            logger.debug(
                f"No records with {field_name}='{field_value}' found. No action taken."
            )
            return True

        return await self.adelete_by_filter({field_name: {"$eq": field_value}})

    # ------------------------------------------------------------------
    # Existence checks (abstract implementations)
    # ------------------------------------------------------------------

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        return await self.afield_exists("chunk_id", chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        return await self.afield_exists("doc_content_hash", doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        return await self.afield_exists("chunk_content_hash", chunk_content_hash)

    async def adocument_id_exists(self, document_id: str) -> bool:
        return await self.afield_exists("document_id", document_id)

    async def adocument_name_exists(self, document_name: str) -> bool:
        return await self.afield_exists("document_name", document_name)

    # ------------------------------------------------------------------
    # Delete by field (abstract implementations)
    # ------------------------------------------------------------------

    async def adelete_by_chunk_id(self, chunk_id: str) -> bool:
        return await self.adelete_by_field("chunk_id", chunk_id)

    async def adelete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        return await self.adelete_by_field("doc_content_hash", doc_content_hash)

    async def adelete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        return await self.adelete_by_field("chunk_content_hash", chunk_content_hash)

    async def adelete_by_document_id(self, document_id: str) -> bool:
        return await self.adelete_by_field("document_id", document_id)

    async def adelete_by_document_name(self, document_name: str) -> bool:
        return await self.adelete_by_field("document_name", document_name)

    async def adelete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        filter_dict: Dict[str, Any] = {key: {"$eq": value} for key, value in metadata.items()}
        return await self.adelete_by_filter(filter_dict)

    # ========================================================================
    # Fetch
    # ========================================================================

    async def afetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        await self.aget_client()
        if not self._index:
            raise VectorDBConnectionError("Index not available.")
        
        namespace: str = self._config.namespace or ""
        
        def _fetch() -> List[VectorSearchResult]:
            try:
                response = self._index.fetch(ids=[str(i) for i in ids], namespace=namespace)
                return self._parse_fetch_response(response)
            except ApiException as e:
                logger.warning(f"Fetch failed: {e}")
                return []
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _fetch)

    def _parse_fetch_response(self, response: Any) -> List[VectorSearchResult]:
        results: List[VectorSearchResult] = []
        fetched_vectors = response.vectors if hasattr(response, 'vectors') else response.get('vectors', {})
        
        for record_id, vector_data in fetched_vectors.items():
            metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else vector_data.get('metadata', {})
            values = vector_data.values if hasattr(vector_data, 'values') else vector_data.get('values')
            
            hydrated = self._hydrate_payload(metadata)
            results.append(
                VectorSearchResult(
                    id=record_id,
                    score=1.0,
                    payload=hydrated,
                    vector=values,
                    text=hydrated.get('content') or None,
                )
            )
        return results

    def _parse_query_response(self, response: Any) -> List[VectorSearchResult]:
        matches = response.matches if hasattr(response, 'matches') else response.get('matches', [])
        
        results: List[VectorSearchResult] = []
        for match in matches:
            if hasattr(match, 'id'):
                match_id = match.id
                raw_score = match.score
                metadata = match.metadata if hasattr(match, 'metadata') else {}
                values = match.values if hasattr(match, 'values') else None
            else:
                match_id = match['id']
                raw_score = match['score']
                metadata = match.get('metadata', {})
                values = match.get('values')
            
            normalized_score: float = self._normalize_score(raw_score)
            hydrated = self._hydrate_payload(metadata)

            results.append(
                VectorSearchResult(
                    id=match_id,
                    score=normalized_score,
                    payload=hydrated,
                    vector=values,
                    text=hydrated.get('content') or None,
                )
            )
        return results
    
    def _normalize_score(self, score: float) -> float:
        metric: str = self._config.metric
        
        if metric == 'dotproduct':
            return min(1.0, max(0.0, score))
        
        elif metric == 'cosine':
            return min(1.0, max(0.0, score))
        
        elif metric == 'euclidean':
            if score < 0:
                score = 0
            return 1.0 / (1.0 + score)
        
        else:
            return min(1.0, max(0.0, score))

    # ========================================================================
    # Search Operations
    # ========================================================================

    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        _ = apply_reranking  # accepted for API parity; not applied in dense path
        await self.aget_client()
        if not self._index:
            raise VectorDBConnectionError("Index not available.")

        final_threshold: float = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        namespace: str = self._config.namespace or ""
        include_values: bool = self._config.include_values

        native_filter, post_filter = self._split_filter(filter)
        query_top_k = top_k * 5 if post_filter else top_k

        def _search() -> List[VectorSearchResult]:
            try:
                response = self._index.query(
                    vector=query_vector,
                    top_k=query_top_k,
                    filter=native_filter,
                    namespace=namespace,
                    include_metadata=True,
                    include_values=include_values
                )
                results = self._parse_query_response(response)
                if post_filter:
                    results = [r for r in results if self._matches_post_filter(r.payload, post_filter)]
                filtered = [r for r in results if r.score >= final_threshold]
                filtered = filtered[:top_k]
                logger.debug(f"Dense search: {len(results)} results, {len(filtered)} after threshold.")
                return filtered
            except ApiException as e:
                raise SearchError(f"Dense search failed: {e}") from e
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

    async def afull_text_search(
        self,
        query_text: Optional[str] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        _ = apply_reranking  # accepted for API parity; not applied in full-text path
        await self.aget_client()
        if not self._index:
            raise VectorDBConnectionError("Index not available.")

        if not self._config.use_sparse_vectors:
            raise ConfigurationError("Full-text search requires use_sparse_vectors to be enabled.")
        
        top_k = top_k or self._config.default_top_k
        final_threshold: float = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        namespace: str = self._config.namespace or ""

        native_filter, post_filter = self._split_filter(filter)
        query_top_k = top_k * 5 if post_filter else top_k

        def _search() -> List[VectorSearchResult]:
            try:
                sparse_vector: Optional[Dict[str, Any]] = sparse_query_vector
                if not sparse_vector:
                    if self._sparse_encoder:
                        sparse_vector = self._sparse_encoder.encode_queries(query_text)
                    else:
                        result = self.client.inference.embed(
                            model=self._config.sparse_encoder_model,
                            inputs=query_text,
                            parameters={"input_type": "query", "truncate": "END"}
                        )
                        sparse_vector = {
                            'indices': result[0]['sparse_indices'],
                            'values': result[0]['sparse_values']
                        }

                import numpy as np
                np.random.seed(42)
                dummy_vector = np.random.rand(self._config.vector_size).astype(np.float32)
                dummy_vector = (dummy_vector / np.linalg.norm(dummy_vector)).tolist()

                response = self._index.query(
                    vector=dummy_vector,
                    sparse_vector=sparse_vector,
                    top_k=query_top_k,
                    filter=native_filter,
                    namespace=namespace,
                    include_metadata=True,
                    include_values=True
                )
                results = self._parse_query_response(response)
                if post_filter:
                    results = [r for r in results if self._matches_post_filter(r.payload, post_filter)]
                filtered = [r for r in results if r.score >= final_threshold]
                filtered.sort(key=lambda x: x.score, reverse=True)
                return filtered[:top_k]
            except ApiException as e:
                raise SearchError(f"Full-text search failed: {e}") from e
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

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
        _ = (fusion_method, apply_reranking)  # accepted for API parity; Pinecone uses alpha-weighted fusion
        await self.aget_client()
        if not self._index:
            raise VectorDBConnectionError("Index not available.")

        if not self._config.hybrid_search_enabled:
            raise ConfigurationError("Hybrid search requires hybrid_search_enabled to be enabled.")
        
        top_k = top_k or self._config.default_top_k
        alpha = alpha if alpha is not None else (self._config.default_hybrid_alpha or 0.5)
        final_threshold: float = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        namespace: str = self._config.namespace or ""
        include_values: bool = self._config.include_values

        native_filter, post_filter = self._split_filter(filter)
        query_top_k = top_k * 5 if post_filter else top_k

        def _search() -> List[VectorSearchResult]:
            try:
                sparse_vector: Optional[Dict[str, Any]] = sparse_query_vector
                if not sparse_vector:
                    if self._sparse_encoder:
                        sparse_vector = self._sparse_encoder.encode_queries(query_text)
                    else:
                        result = self.client.inference.embed(
                            model=self._config.sparse_encoder_model,
                            inputs=query_text,
                            parameters={"input_type": "query", "truncate": "END"}
                        )
                        sparse_vector = {
                            'indices': result[0]['sparse_indices'],
                            'values': result[0]['sparse_values']
                        }

                hdense, hsparse = self._hybrid_scale(query_vector, sparse_vector, alpha)

                response = self._index.query(
                    vector=hdense,
                    sparse_vector=hsparse,
                    top_k=query_top_k,
                    filter=native_filter,
                    namespace=namespace,
                    include_metadata=True,
                    include_values=include_values
                )

                results = self._parse_query_response(response)
                if post_filter:
                    results = [r for r in results if self._matches_post_filter(r.payload, post_filter)]

                if self._config.reranker:
                    logger.debug("Reranker configured but not yet integrated")

                filtered = [r for r in results if r.score >= final_threshold]
                return filtered[:top_k]
            except ApiException as e:
                raise SearchError(f"Hybrid search failed: {e}") from e
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

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
        is_dense: bool = query_vector is not None
        is_sparse: bool = query_text is not None
        
        final_top_k: Optional[int] = top_k if top_k is not None else self._config.default_top_k
        if final_top_k is None:
            raise ConfigurationError("'top_k' must be provided.")

        if is_dense and is_sparse and self._config.hybrid_search_enabled:
            logger.debug("Dispatching to HYBRID search.")
            return await self.ahybrid_search(
                query_vector, query_text, final_top_k, filter, alpha,
                fusion_method, similarity_threshold, apply_reranking, sparse_query_vector,
            )
        
        elif is_dense:
            logger.debug("Dispatching to DENSE search.")
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled.")
            return await self.adense_search(
                query_vector, final_top_k, filter, similarity_threshold, apply_reranking,
            )
        
        elif is_sparse:
            logger.debug("Dispatching to FULL-TEXT search.")
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled.")
            return await self.afull_text_search(
                query_text, final_top_k, filter, similarity_threshold, apply_reranking, sparse_query_vector,
            )
        
        else:
            raise SearchError("Search requires 'query_vector' or 'query_text'.")

    # ========================================================================
    # Provider-specific async helpers
    # ========================================================================

    async def aid_exists(self, id: str) -> bool:
        client = await self.aget_client()

        if not self._index:
            logger.warning("Index not set, attempting to connect to it...")
            safe_collection_name = self._sanitize_collection_name(self._config.collection_name)

            def _attach() -> None:
                if safe_collection_name in client.list_indexes().names():
                    self._index = client.Index(safe_collection_name)
                else:
                    raise VectorDBConnectionError("Index does not exist.")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _attach)

        namespace: str = self._config.namespace or ""

        def _check() -> bool:
            try:
                response = self._index.fetch(ids=[id], namespace=namespace)
                if hasattr(response, 'vectors'):
                    return len(response.vectors) > 0
                return len(response.get('vectors', {})) > 0
            except Exception as e:
                logger.warning(f"Error checking ID existence: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _check)

    async def aget_count(self) -> int:
        client = await self.aget_client()

        if not self._index:
            logger.warning("Index not set, attempting to connect to it...")
            safe_collection_name = self._sanitize_collection_name(self._config.collection_name)

            def _attach() -> None:
                if safe_collection_name in client.list_indexes().names():
                    self._index = client.Index(safe_collection_name)
                else:
                    raise VectorDBConnectionError("Index does not exist.")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _attach)

        def _count() -> int:
            try:
                stats = self._index.describe_index_stats()
                return stats.total_vector_count if hasattr(stats, 'total_vector_count') else stats.get('total_vector_count', 0)
            except Exception as e:
                logger.warning(f"Error getting count: {e}")
                return 0

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _count)

    # ========================================================================
    # Update Metadata
    # ========================================================================

    async def aupdate_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a chunk. Standard fields are updated at flat top level;
        non-standard user fields are merged into the JSON-serialized ``metadata``
        Pinecone key (read-modify-write).
        """
        await self.aget_client()
        if not self._index:
            raise VectorDBConnectionError("Index not available.")

        namespace: str = self._config.namespace or ""

        def _update() -> bool:
            try:
                # Fetch existing record to merge user metadata
                response = self._index.fetch(ids=[chunk_id], namespace=namespace)
                fetched = response.vectors if hasattr(response, "vectors") else response.get("vectors", {})
                existing_pinecone_meta: Dict[str, Any] = {}
                if chunk_id in fetched:
                    vec = fetched[chunk_id]
                    existing_pinecone_meta = (
                        vec.metadata if hasattr(vec, "metadata") else vec.get("metadata", {})
                    ) or {}

                # Parse existing user metadata blob
                raw_existing = existing_pinecone_meta.get("metadata", "{}")
                if isinstance(raw_existing, str) and raw_existing:
                    try:
                        existing_user_meta = json.loads(raw_existing)
                        if not isinstance(existing_user_meta, dict):
                            existing_user_meta = {}
                    except (json.JSONDecodeError, TypeError, ValueError):
                        existing_user_meta = {}
                elif isinstance(raw_existing, dict):
                    existing_user_meta = raw_existing
                else:
                    existing_user_meta = {}

                set_meta: Dict[str, Any] = {}
                user_updates: Dict[str, Any] = {}
                for key, value in (metadata or {}).items():
                    if key == "metadata":
                        if isinstance(value, dict):
                            user_updates.update(value)
                        continue
                    if key in self._STANDARD_FIELDS:
                        set_meta[key] = value
                    else:
                        user_updates[key] = value

                if user_updates:
                    existing_user_meta.update(user_updates)
                    set_meta["metadata"] = json.dumps(existing_user_meta)

                if not set_meta:
                    return True

                self._index.update(
                    id=chunk_id,
                    set_metadata=set_meta,
                    namespace=namespace,
                )
                logger.info(f"Updated metadata for chunk_id: {chunk_id}")
                return True
            except Exception as e:
                logger.error(f"Error updating metadata for chunk_id '{chunk_id}': {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _update)

    # ========================================================================
    # Optimize & Supported Search Types
    # ========================================================================

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
