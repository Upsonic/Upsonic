from __future__ import annotations

import asyncio
import hashlib
import os
import uuid as _uuid
from typing import Any, Dict, List, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from supermemory import AsyncSupermemory

try:
    from supermemory import AsyncSupermemory
    _SUPERMEMORY_AVAILABLE = True
except ImportError:
    AsyncSupermemory = None  # type: ignore[assignment,misc]
    _SUPERMEMORY_AVAILABLE = False


from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.utils.printing import info_log, debug_log, warning_log, error_log

from upsonic.vectordb.config import SuperMemoryConfig

from upsonic.utils.package.exception import (
    VectorDBConnectionError,
    VectorDBError,
    SearchError,
    UpsertError,
)

from upsonic.schemas.vector_schemas import VectorSearchResult


class SuperMemoryProvider(BaseVectorDBProvider):
    """
    A vector database provider that wraps the SuperMemory managed memory API.

    Standard properties stored per document (as metadata):
    - chunk_id (string) — unique per-chunk identifier
    - document_id (string) — parent document identifier
    - doc_content_hash (string) — MD5 of parent document content for change detection
    - chunk_content_hash (string) — MD5 of chunk text content for deduplication
    - document_name (string) — human-readable source name
    - content (string) — the chunk text (stored as document content, not metadata)
    - metadata (dict) — all non-standard data folded into a JSON metadata key
    """

    _STANDARD_FIELDS: frozenset = frozenset({
        'chunk_id', 'document_id', 'doc_content_hash',
        'chunk_content_hash', 'document_name', 'content', 'metadata',
        'knowledge_base_id',
    })

    def __init__(
        self,
        config: Union[SuperMemoryConfig, Dict[str, Any]],
    ) -> None:
        if not _SUPERMEMORY_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="supermemory",
                install_command='pip install "upsonic[supermemory]"',
                feature_name="SuperMemory vector database provider",
            )

        if isinstance(config, dict):
            config = SuperMemoryConfig.from_dict(config)

        super().__init__(config)
        self._config: SuperMemoryConfig = config

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    def _generate_provider_id(self) -> str:
        identifier: str = f"supermemory#{self._config.container_tag or self._config.collection_name}"
        return hashlib.md5(identifier.encode()).hexdigest()[:16]

    def _resolve_api_key(self) -> str:
        if self._config.api_key is not None:
            return self._config.api_key.get_secret_value()
        env_key: Optional[str] = os.environ.get("SUPERMEMORY_API_KEY")
        if env_key:
            return env_key
        raise VectorDBConnectionError(
            "SuperMemory API key not provided. Set it via SuperMemoryConfig.api_key "
            "or the SUPERMEMORY_API_KEY environment variable."
        )

    @property
    def _container_tag(self) -> str:
        return self._config.container_tag or self._config.collection_name

    # ------------------------------------------------------------------
    # Client lifecycle management
    # ------------------------------------------------------------------

    async def _create_async_client(self) -> None:
        """
        Instantiates the AsyncSupermemory client from configuration.
        Does NOT verify readiness — only creates the client object.

        Raises:
            VectorDBConnectionError: If client instantiation fails.
        """
        try:
            api_key: str = self._resolve_api_key()
            self.client = AsyncSupermemory(
                api_key=api_key,
                max_retries=self._config.max_retries,
                timeout=self._config.timeout,
            )
        except VectorDBConnectionError:
            raise
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to create SuperMemory async client: {e}") from e

    async def aget_client(self) -> "AsyncSupermemory":
        """
        Singleton entry point. Creates client if None, verifies readiness,
        and auto-recovers if the existing client is unresponsive.

        Returns:
            A connected and ready AsyncSupermemory client instance.

        Raises:
            VectorDBConnectionError: If the client cannot be created or verified.
        """
        if self.client is None:
            debug_log("Creating SuperMemory async client...", context="SuperMemoryVectorDB")
            await self._create_async_client()

        try:
            await self.client.search.execute(q="ping", limit=1)  # type: ignore[union-attr]
        except Exception:
            debug_log("Existing SuperMemory client is unresponsive, recreating...", context="SuperMemoryVectorDB")
            try:
                if self.client is not None:
                    await self.client.close()
            except Exception:
                pass
            self.client = None
            self._is_connected = False

            await self._create_async_client()

            try:
                await self.client.search.execute(q="ping", limit=1)  # type: ignore[union-attr]
            except Exception as e:
                self.client = None
                self._is_connected = False
                raise VectorDBConnectionError(
                    f"SuperMemory client is not ready after recreation: {e}"
                ) from e

        self._is_connected = True
        return self.client  # type: ignore[return-value]

    def get_client(self) -> "AsyncSupermemory":
        return self._run_async_from_sync(self.aget_client())

    async def ais_client_connected(self) -> bool:
        """
        Read-only check whether the client exists and is responsive.
        Does NOT create or reconnect.
        """
        if self.client is None:
            return False
        try:
            await self.client.search.execute(q="ping", limit=1)
            return True
        except Exception:
            return False

    def is_client_connected(self) -> bool:
        return self._run_async_from_sync(self.ais_client_connected())

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def aconnect(self) -> None:
        if await self.ais_client_connected():
            info_log("Already connected to SuperMemory.", context="SuperMemoryVectorDB")
            return

        debug_log("Connecting to SuperMemory API...", context="SuperMemoryVectorDB")
        await self.aget_client()
        info_log("SuperMemory connection established.", context="SuperMemoryVectorDB")

    async def adisconnect(self) -> None:
        if self.client is None and not self._is_connected:
            return

        debug_log("Disconnecting from SuperMemory...", context="SuperMemoryVectorDB")

        try:
            if self.client is not None:
                await self.client.close()
        except Exception:
            pass
        finally:
            self.client = None
            self._is_connected = False
            info_log("SuperMemory client session closed.", context="SuperMemoryVectorDB")

    async def ais_ready(self) -> bool:
        if self.client is None:
            return False
        try:
            await self.client.search.execute(q="ping", limit=1)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    async def acreate_collection(self) -> None:
        debug_log(
            f"create_collection called (no-op for SuperMemory, container_tag='{self._container_tag}')",
            context="SuperMemoryVectorDB",
        )

    async def adelete_collection(self) -> None:
        client: AsyncSupermemory = await self.aget_client()

        debug_log(
            f"Deleting all documents for container_tag='{self._container_tag}'",
            context="SuperMemoryVectorDB",
        )

        try:
            await client.documents.delete_bulk(
                container_tags=[self._container_tag],
            )
            info_log("SuperMemory collection deleted.", context="SuperMemoryVectorDB")
        except Exception as e:
            raise VectorDBError(f"Failed to delete SuperMemory collection: {e}") from e

    async def acollection_exists(self) -> bool:
        if self.client is None:
            return False

        try:
            client: AsyncSupermemory = await self.aget_client()
            results = await client.search.execute(
                q="*",
                container_tags=[self._container_tag],
                limit=1,
            )
            return bool(results.results)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

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
        _apply_defaults: bool = True,
    ) -> Dict[str, Any]:
        """
        Builds a SuperMemory metadata dict.

        Standard fields are stored as top-level flat metadata keys (SuperMemory
        requires flat primitive K/V). Non-standard user fields are ALSO stored
        at the flat top level (SuperMemory filter API matches on flat keys);
        they are re-nested under ``payload["metadata"]`` on read via
        ``_hydrate_payload``.
        """
        user_meta: Dict[str, Any] = {}

        if _apply_defaults and self._config.default_metadata:
            user_meta.update(self._config.default_metadata)

        if extra_payload:
            nested = extra_payload.get("metadata")
            if isinstance(nested, dict):
                user_meta.update(nested)

            for key, value in extra_payload.items():
                if key in self._STANDARD_FIELDS:
                    if key == "metadata":
                        continue
                    if _warned_standard_keys is not None and key not in _warned_standard_keys:
                        debug_log(
                            f"Standard field '{key}' found in payload dict and will be ignored. "
                            f"Standard fields must be passed via dedicated parameters "
                            f"(ids, document_ids, document_names, doc_content_hashes, "
                            f"chunk_content_hashes, knowledge_base_ids).",
                            context="SuperMemoryVectorDB",
                        )
                        _warned_standard_keys.add(key)
                    continue
                user_meta[key] = value

        if extra_metadata:
            user_meta.update(extra_metadata)

        metadata: Dict[str, Any] = {
            "chunk_id": chunk_id or "",
            "chunk_content_hash": chunk_content_hash or "",
            "document_id": document_id or "",
            "doc_content_hash": doc_content_hash or "",
            "document_name": document_name or "",
            "knowledge_base_id": knowledge_base_id or "",
        }

        for k, v in user_meta.items():
            if k in self._STANDARD_FIELDS:
                continue
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v
            elif v is not None:
                metadata[k] = str(v)

        return metadata

    def _hydrate_payload(self, raw_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a contract-compliant VectorSearchResult.payload from SuperMemory's
        flat metadata dict. Standard fields go to top level; non-standards are
        nested under ``payload["metadata"]`` as a Python dict.
        """
        raw_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        user_meta: Dict[str, Any] = {
            k: v for k, v in raw_metadata.items()
            if k not in self._STANDARD_FIELDS
        }
        return {
            "chunk_id": raw_metadata.get("chunk_id", "") or "",
            "document_id": raw_metadata.get("document_id", "") or "",
            "document_name": raw_metadata.get("document_name", "") or "",
            "content": raw_metadata.get("content", "") or "",
            "doc_content_hash": raw_metadata.get("doc_content_hash", "") or "",
            "chunk_content_hash": raw_metadata.get("chunk_content_hash", "") or "",
            "knowledge_base_id": raw_metadata.get("knowledge_base_id", "") or "",
            "metadata": user_meta,
        }

    # ------------------------------------------------------------------
    # Upsert with content-based deduplication
    # ------------------------------------------------------------------

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
        client: AsyncSupermemory = await self.aget_client()

        if chunks is None:
            raise UpsertError("aupsert requires 'chunks' for SuperMemory (text-based ingestion).")

        if (vectors is None or len(vectors) == 0) and (sparse_vectors is None or len(sparse_vectors) == 0):
            info_log("Nothing to upsert: no dense or sparse vectors provided.", context="SuperMemoryVectorDB")
            return

        n: int = len(vectors) if vectors else len(sparse_vectors)  # type: ignore[arg-type]

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
                raise UpsertError(
                    f"Length mismatch: '{name}' has {len(arr)} items, expected {n}"
                )

        _warned_standard_keys: set = set()

        debug_log(
            f"Upserting {n} chunks to SuperMemory (container_tag='{self._container_tag}')",
            context="SuperMemoryVectorDB",
        )

        extra_metadata: Dict[str, Any] = metadata or {}

        computed_hashes: List[str] = []
        record_ids: List[str] = []
        for i in range(n):
            record_ids.append(str(ids[i]) if ids and i < len(ids) else str(_uuid.uuid4()))
            chunk_text: str = chunks[i] if chunks and i < len(chunks) else ""
            if chunk_content_hashes and i < len(chunk_content_hashes) and chunk_content_hashes[i]:
                computed_hashes.append(chunk_content_hashes[i])
            else:
                computed_hashes.append(hashlib.md5(chunk_text.encode("utf-8")).hexdigest())

        stale_ids: List[str] = await self._batch_find_stale_by_hashes(
            client, computed_hashes, record_ids
        )

        if stale_ids:
            try:
                await client.documents.delete_bulk(ids=stale_ids)
                debug_log(
                    f"Deleted {len(stale_ids)} stale documents with duplicate chunk_content_hash.",
                    context="SuperMemoryVectorDB",
                )
            except Exception as e:
                debug_log(
                    f"Failed to delete stale duplicate documents: {e}",
                    context="SuperMemoryVectorDB",
                )

        valid_docs: List[Dict[str, Any]] = []

        for idx in range(n):
            chunk_text = chunks[idx] if chunks and idx < len(chunks) else ""
            if not chunk_text or not chunk_text.strip():
                debug_log(f"Skipping empty chunk at index {idx}", context="SuperMemoryVectorDB")
                continue

            chunk_id: str = record_ids[idx]
            payload: Dict[str, Any] = payloads[idx] if payloads and idx < len(payloads) else {}
            doc_id: str = document_ids[idx] if document_ids and idx < len(document_ids) else ""
            doc_hash: str = doc_content_hashes[idx] if doc_content_hashes and idx < len(doc_content_hashes) else ""
            chunk_hash: str = computed_hashes[idx]
            doc_name: str = document_names[idx] if document_names and idx < len(document_names) else ""
            kbi: Optional[str] = knowledge_base_ids[idx] if knowledge_base_ids and idx < len(knowledge_base_ids) else None

            metadata: Dict[str, Any] = self._build_payload(
                content=chunk_text,
                chunk_id=str(chunk_id),
                chunk_content_hash=chunk_hash,
                document_id=doc_id,
                doc_content_hash=doc_hash,
                document_name=doc_name,
                knowledge_base_id=kbi,
                extra_payload=payload,
                extra_metadata=extra_metadata,
                _warned_standard_keys=_warned_standard_keys,
            )

            valid_docs.append({
                "content": chunk_text,
                "custom_id": str(chunk_id),
                "metadata": metadata,
            })

        if not valid_docs:
            info_log("No valid chunks to upsert (all empty).", context="SuperMemoryVectorDB")
            return

        batch_size: int = self._config.batch_size
        errors: List[str] = []
        total_succeeded: int = 0

        for batch_start in range(0, len(valid_docs), batch_size):
            batch_docs: List[Dict[str, Any]] = valid_docs[batch_start:batch_start + batch_size]

            try:
                await client.documents.batch_add(
                    documents=batch_docs,  # type: ignore[arg-type]
                    container_tag=self._container_tag,
                )
                total_succeeded += len(batch_docs)
            except Exception as e:
                error_msg: str = f"Batch upsert failed (items {batch_start}-{batch_start + len(batch_docs)}): {e}"
                error_log(error_msg, context="SuperMemoryVectorDB")
                errors.append(error_msg)

            if self._config.batch_delay > 0 and batch_start + batch_size < len(valid_docs):
                await asyncio.sleep(self._config.batch_delay)

        if errors and total_succeeded == 0:
            raise UpsertError(f"All batch upserts failed. First error: {errors[0]}")

        info_log(
            f"Upserted {total_succeeded}/{len(valid_docs)} chunks successfully.",
            context="SuperMemoryVectorDB",
        )

        # SuperMemory indexes asynchronously; wait for content to become searchable
        if total_succeeded > 0 and self._config.index_delay > 0:
            info_log(
                f"Waiting {self._config.index_delay}s for SuperMemory indexing to complete...",
                context="SuperMemoryVectorDB",
            )
            await asyncio.sleep(self._config.index_delay)

    async def _batch_find_stale_by_hashes(
        self,
        client: "AsyncSupermemory",
        chunk_hashes: List[str],
        new_chunk_ids: List[str],
    ) -> List[str]:
        """
        Finds SuperMemory document IDs that have a matching chunk_content_hash
        but a different chunk_id (i.e. stale duplicates that should be replaced).

        Args:
            client: Connected SuperMemory client.
            chunk_hashes: List of chunk content hashes to check.
            new_chunk_ids: Corresponding chunk_ids for the new points.

        Returns:
            List of SuperMemory document IDs to delete.
        """
        unique_hashes: set[str] = set(chunk_hashes)
        stale_ids: List[str] = []
        new_chunk_id_set: set[str] = set(new_chunk_ids)

        for h in unique_hashes:
            try:
                results = await client.search.execute(
                    q="*",
                    container_tags=[self._container_tag],
                    limit=100,
                    filters={"AND": [{"key": "chunk_content_hash", "value": h}]},
                )
                for item in results.results:
                    item_id: str = getattr(item, "id", "")
                    item_metadata: Optional[Dict[str, Any]] = getattr(item, "metadata", None)
                    existing_chunk_id: str = ""
                    if isinstance(item_metadata, dict):
                        existing_chunk_id = str(item_metadata.get("chunk_id", ""))

                    if existing_chunk_id and existing_chunk_id not in new_chunk_id_set and item_id:
                        stale_ids.append(item_id)
            except Exception:
                pass

        return stale_ids

    # ------------------------------------------------------------------
    # Delete / Fetch
    # ------------------------------------------------------------------

    async def adelete(self, ids: List[Union[str, int]]) -> None:
        client: AsyncSupermemory = await self.aget_client()

        if not ids:
            return

        debug_log(f"Deleting {len(ids)} documents from SuperMemory", context="SuperMemoryVectorDB")

        existed: int = 0
        deleted: int = 0
        for doc_id in ids:
            str_id: str = str(doc_id)
            try:
                await client.documents.get(str_id)
                existed += 1
            except Exception:
                debug_log(
                    f"Document with ID '{str_id}' does not exist, skipping deletion.",
                    context="SuperMemoryVectorDB",
                )
                continue
            try:
                await client.documents.delete(str_id)
                deleted += 1
            except Exception as e:
                warning_log(f"Failed to delete document '{str_id}': {e}", context="SuperMemoryVectorDB")

        if existed == 0:
            info_log("No matching documents found to delete. No action taken.", context="SuperMemoryVectorDB")
        else:
            info_log(
                f"Successfully processed deletion request for {len(ids)} IDs. "
                f"Existed: {existed}, Deleted: {deleted}.",
                context="SuperMemoryVectorDB",
            )

    async def afetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        client: AsyncSupermemory = await self.aget_client()

        results: List[VectorSearchResult] = []

        for doc_id in ids:
            try:
                doc = await client.documents.get(str(doc_id))
                raw_metadata: Optional[Dict[str, Any]] = getattr(doc, "metadata", None)
                text_content = getattr(doc, "content", None)
                payload = self._hydrate_payload(raw_metadata)
                if text_content and not payload.get("content"):
                    payload["content"] = text_content
                results.append(VectorSearchResult(
                    id=str(doc_id),
                    score=1.0,
                    payload=payload,
                    vector=None,
                    text=text_content,
                ))
            except Exception as e:
                warning_log(f"Failed to fetch document '{doc_id}': {e}", context="SuperMemoryVectorDB")

        return results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def asearch(
        self,
        top_k: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal["rrf", "weighted"]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        if query_text:
            return await self.ahybrid_search(
                query_vector=query_vector or [],
                query_text=query_text,
                top_k=top_k or self._config.default_top_k,
                filter=filter,
                alpha=alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )

        if query_vector is not None:
            return await self.adense_search(
                query_vector=query_vector,
                top_k=top_k or self._config.default_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
            )

        raise SearchError("Either query_text or query_vector must be provided for search.")

    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        warning_log(
            "SuperMemory does not support raw vector search. "
            "Use hybrid or full-text search by providing query_text. "
            "Returning empty results.",
            context="SuperMemoryVectorDB",
        )
        return []

    async def afull_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        return await self._execute_search(
            query_text=query_text,
            top_k=top_k,
            search_mode="memories",
            filter=filter,
            similarity_threshold=similarity_threshold,
        )

    async def ahybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal["rrf", "weighted"]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        return await self._execute_search(
            query_text=query_text,
            top_k=top_k,
            search_mode=self._config.search_mode,
            filter=filter,
            similarity_threshold=similarity_threshold,
        )

    async def _execute_search(
        self,
        query_text: str,
        top_k: int,
        search_mode: Literal["hybrid", "memories", "documents"],
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        client: AsyncSupermemory = await self.aget_client()

        threshold: float = similarity_threshold or self._config.threshold

        debug_log(
            f"SuperMemory search: q='{query_text[:80]}...', mode={search_mode}, "
            f"top_k={top_k}, threshold={threshold}",
            context="SuperMemoryVectorDB",
        )

        try:
            search_kwargs: Dict[str, Any] = {
                "q": query_text,
                "container_tag": self._container_tag,
                "search_mode": search_mode,
                "limit": top_k,
                "threshold": threshold,
                "rerank": self._config.rerank,
            }

            if filter:
                sm_filters: Dict[str, Any] = self._convert_filters(filter)
                if sm_filters:
                    search_kwargs["filters"] = sm_filters

            response = await client.search.memories(**search_kwargs)

            results: List[VectorSearchResult] = []
            for item in response.results:
                text_content: Optional[str] = getattr(item, "memory", None) or getattr(item, "chunk", None)
                score: float = getattr(item, "similarity", 0.0)
                item_id: str = getattr(item, "id", "")
                raw_md: Optional[Dict[str, Any]] = getattr(item, "metadata", None)
                hydrated = self._hydrate_payload(raw_md)
                if text_content and not hydrated.get("content"):
                    hydrated["content"] = text_content

                results.append(VectorSearchResult(
                    id=item_id,
                    score=score,
                    payload=hydrated,
                    vector=None,
                    text=text_content,
                ))

            info_log(f"SuperMemory returned {len(results)} results.", context="SuperMemoryVectorDB")
            return results

        except Exception as e:
            raise SearchError(f"SuperMemory search failed: {e}") from e

    # ------------------------------------------------------------------
    # Generic field helpers
    # ------------------------------------------------------------------

    async def afield_exists(self, field_name: str, field_value: Any) -> bool:
        """
        Generic check if any document with the given metadata field value exists.

        Args:
            field_name: The metadata field name to filter on.
            field_value: The value to match.

        Returns:
            True if at least one matching document exists, False otherwise.
        """
        try:
            client: AsyncSupermemory = await self.aget_client()
            results = await client.search.execute(
                q="*",
                container_tags=[self._container_tag],
                limit=1,
                filters={"AND": [{"key": field_name, "value": str(field_value)}]},
            )
            return bool(results.results)
        except Exception:
            return False

    async def adelete_by_field(self, field_name: str, field_value: Any) -> bool:
        """
        Generic deletion of all documents matching a metadata field value.

        Includes a pre-search to find matching document IDs, then bulk deletes.
        Returns True if no matches found (idempotent).

        Args:
            field_name: The metadata field name to filter on.
            field_value: The value to match for deletion.

        Returns:
            True if deletion was successful or no matches found, False on failure.
        """
        try:
            client: AsyncSupermemory = await self.aget_client()
            results = await client.search.execute(
                q="*",
                container_tags=[self._container_tag],
                limit=100,
                filters={"AND": [{"key": field_name, "value": str(field_value)}]},
            )
            if results.results:
                ids_to_delete: List[str] = [r.id for r in results.results if r.id]
                if ids_to_delete:
                    await client.documents.delete_bulk(ids=ids_to_delete)
            return True
        except Exception as e:
            warning_log(f"adelete_by_field({field_name}={field_value}) failed: {e}", context="SuperMemoryVectorDB")
            return False

    # ------------------------------------------------------------------
    # Existence checks — delegates to afield_exists
    # ------------------------------------------------------------------

    async def adocument_id_exists(self, document_id: str) -> bool:
        return await self.afield_exists("document_id", document_id)

    async def adocument_name_exists(self, document_name: str) -> bool:
        return await self.afield_exists("document_name", document_name)

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        return await self.afield_exists("chunk_id", chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        return await self.afield_exists("doc_content_hash", doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        return await self.afield_exists("chunk_content_hash", chunk_content_hash)

    # ------------------------------------------------------------------
    # Deletion by field — delegates to adelete_by_field
    # ------------------------------------------------------------------

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
            client: AsyncSupermemory = await self.aget_client()
            sm_filters: Dict[str, Any] = self._convert_filters(metadata)
            results = await client.search.execute(
                q="*",
                container_tags=[self._container_tag],
                limit=100,
                filters=sm_filters,
            )
            if results.results:
                ids_to_delete: List[str] = [r.id for r in results.results if r.id]
                if ids_to_delete:
                    await client.documents.delete_bulk(ids=ids_to_delete)
            return True
        except Exception as e:
            warning_log(f"adelete_by_metadata failed: {e}", context="SuperMemoryVectorDB")
            return False

    # ------------------------------------------------------------------
    # Update metadata
    # ------------------------------------------------------------------

    async def aupdate_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        try:
            client: AsyncSupermemory = await self.aget_client()
            results = await client.search.execute(
                q="*",
                container_tags=[self._container_tag],
                limit=1,
                filters={"AND": [{"key": "chunk_id", "value": chunk_id}]},
            )
            if not results.results:
                warning_log(f"No document found for chunk_id='{chunk_id}'", context="SuperMemoryVectorDB")
                return False

            first = results.results[0]
            sm_doc_id: str = getattr(first, "id", "")
            if not sm_doc_id:
                return False

            existing_md: Optional[Dict[str, Any]] = getattr(first, "metadata", None)
            merged: Dict[str, Any] = dict(existing_md) if isinstance(existing_md, dict) else {}
            # Apply user update, but DO NOT allow standard fields to be overwritten
            # and do NOT reapply default_metadata (would silently revert user overrides).
            for k, v in (metadata or {}).items():
                if k in self._STANDARD_FIELDS:
                    continue
                merged[k] = v

            await client.documents.update(
                sm_doc_id,
                metadata=merged,
            )
            return True
        except Exception as e:
            warning_log(f"aupdate_metadata failed for chunk_id='{chunk_id}': {e}", context="SuperMemoryVectorDB")
            return False

    # ------------------------------------------------------------------
    # Optimize / Supported search types
    # ------------------------------------------------------------------

    async def aoptimize(self) -> bool:
        return True

    async def aget_supported_search_types(self) -> List[str]:
        types: List[str] = []
        if self._config.full_text_search_enabled:
            types.append("full_text")
        if self._config.hybrid_search_enabled:
            types.append("hybrid")
        return types

    # ------------------------------------------------------------------
    # Filter conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        if "AND" in filters or "OR" in filters:
            return filters

        conditions: List[Dict[str, Any]] = []
        for key, value in filters.items():
            conditions.append({
                "key": key,
                "value": str(value) if not isinstance(value, str) else value,
            })

        return {"AND": conditions}
