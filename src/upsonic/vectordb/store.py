from __future__ import annotations

import asyncio
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.vectordb.base import BaseVectorDBProvider
    from upsonic.embeddings.base import EmbeddingProvider
    from upsonic.schemas.vector_schemas import VectorSearchResult


class VectorStore:
    """
    High-level convenience wrapper for standalone VectorDB usage.

    Provides simple add/search/delete operations without the full
    KnowledgeBase machinery (no loaders, splitters, source management).
    Handles embedding, deduplication, and collection lifecycle automatically.

    Example::

        store = VectorStore(
            vectordb=QdrantProvider(config=qdrant_config),
            embedding_provider=OpenAIEmbedding(),
        )
        await store.aconnect()
        doc_id = await store.aadd("The quick brown fox...")
        results = await store.asearch("brown fox")
        await store.adelete(doc_id)
        await store.adisconnect()
    """

    def __init__(
        self,
        vectordb: "BaseVectorDBProvider",
        embedding_provider: Optional["EmbeddingProvider"] = None,
    ) -> None:
        self.vectordb: "BaseVectorDBProvider" = vectordb
        self.embedding_provider: Optional["EmbeddingProvider"] = embedding_provider
        self._is_connected: bool = False



    def _run_async_from_sync(self, awaitable: Any) -> Any:
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                def _run() -> Any:
                    loop = asyncio.new_event_loop()
                    return loop.run_until_complete(awaitable)
                return executor.submit(_run).result()
        except RuntimeError:
            return asyncio.new_event_loop().run_until_complete(awaitable)


    async def aconnect(self) -> None:
        """Connect the underlying VectorDB provider."""
        if not self._is_connected:
            await self.vectordb.aconnect()
            self._is_connected = True

    def connect(self) -> None:
        """Synchronous wrapper for aconnect."""
        self._run_async_from_sync(self.aconnect())

    async def adisconnect(self) -> None:
        """Disconnect the underlying VectorDB provider."""
        if self._is_connected:
            await self.vectordb.adisconnect()
            self._is_connected = False

    def disconnect(self) -> None:
        """Synchronous wrapper for adisconnect."""
        self._run_async_from_sync(self.adisconnect())



    async def aadd(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> str:
        """
        Embed a single text and upsert it into the VectorDB.

        Args:
            text: The text content to store.
            metadata: Optional metadata to associate with this entry.
            document_id: Optional explicit document ID. Defaults to a content-based hash.

        Returns:
            The chunk_id assigned to the stored entry.
        """
        await self._ensure_connected()
        await self._ensure_collection()

        chunk_content_hash: str = hashlib.md5(text.encode("utf-8")).hexdigest()

        if await self.vectordb.achunk_content_hash_exists(chunk_content_hash):
            return chunk_content_hash

        vector: List[float] = await self._embed_single(text)
        chunk_id: str = chunk_content_hash
        doc_id: str = document_id or chunk_content_hash

        await self.vectordb.aupsert(
            vectors=[vector],
            payloads=[metadata or {}],
            ids=[chunk_id],
            chunks=[text],
            document_ids=[doc_id],
            doc_content_hashes=[chunk_content_hash],
            chunk_content_hashes=[chunk_content_hash],
        )
        return chunk_id

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper for aadd."""
        return self._run_async_from_sync(self.aadd(text, metadata, document_id))

    async def aadd_many(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Embed and upsert a batch of texts.

        Args:
            texts: List of text strings to store.
            metadatas: Optional per-text metadata dicts.
            document_ids: Optional per-text document IDs.

        Returns:
            List of chunk_ids for stored entries (skipped duplicates still return their IDs).
        """
        await self._ensure_connected()
        await self._ensure_collection()

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts.")
        if document_ids and len(document_ids) != len(texts):
            raise ValueError("Length of document_ids must match length of texts.")

        texts_to_embed: List[str] = []
        payloads_to_upsert: List[Dict[str, Any]] = []
        ids_to_upsert: List[str] = []
        doc_ids_to_upsert: List[str] = []
        doc_hashes_to_upsert: List[str] = []
        chunk_hashes_to_upsert: List[str] = []

        result_ids: List[str] = []

        for idx, text in enumerate(texts):
            chunk_hash: str = hashlib.md5(text.encode("utf-8")).hexdigest()
            result_ids.append(chunk_hash)

            if await self.vectordb.achunk_content_hash_exists(chunk_hash):
                continue

            texts_to_embed.append(text)
            payloads_to_upsert.append((metadatas[idx] if metadatas else {}))
            ids_to_upsert.append(chunk_hash)
            doc_id: str = (document_ids[idx] if document_ids else chunk_hash)
            doc_ids_to_upsert.append(doc_id)
            doc_hashes_to_upsert.append(chunk_hash)
            chunk_hashes_to_upsert.append(chunk_hash)

        if texts_to_embed:
            vectors: List[List[float]] = await self._embed_many(texts_to_embed)

            await self.vectordb.aupsert(
                vectors=vectors,
                payloads=payloads_to_upsert,
                ids=ids_to_upsert,
                chunks=texts_to_embed,
                document_ids=doc_ids_to_upsert,
                doc_content_hashes=doc_hashes_to_upsert,
                chunk_content_hashes=chunk_hashes_to_upsert,
            )

        return result_ids

    def add_many(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Synchronous wrapper for aadd_many."""
        return self._run_async_from_sync(self.aadd_many(texts, metadatas, document_ids))



    async def asearch(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List["VectorSearchResult"]:
        """
        Embed a query and search the VectorDB.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            filter: Optional metadata filter dict.

        Returns:
            List of VectorSearchResult objects.
        """
        await self._ensure_connected()

        query_vector: List[float] = await self._embed_single(query)

        return await self.vectordb.asearch(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            filter=filter,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List["VectorSearchResult"]:
        """Synchronous wrapper for asearch."""
        return self._run_async_from_sync(self.asearch(query, top_k, filter))


    async def adelete(self, ids: Union[str, List[str]]) -> bool:
        """
        Delete entries by chunk ID(s).

        Args:
            ids: Single chunk ID string or list of chunk IDs.

        Returns:
            True if deletion was successful.
        """
        await self._ensure_connected()
        id_list: List[str] = [ids] if isinstance(ids, str) else ids
        await self.vectordb.adelete(id_list)
        return True

    def delete(self, ids: Union[str, List[str]]) -> bool:
        """Synchronous wrapper for adelete."""
        return self._run_async_from_sync(self.adelete(ids))

    async def adelete_by_filter(self, filter: Dict[str, Any]) -> bool:
        """
        Delete entries matching a metadata filter.

        Args:
            filter: Metadata filter dict.

        Returns:
            True if deletion was successful.
        """
        await self._ensure_connected()
        return await self.vectordb.adelete_by_metadata(filter)

    def delete_by_filter(self, filter: Dict[str, Any]) -> bool:
        """Synchronous wrapper for adelete_by_filter."""
        return self._run_async_from_sync(self.adelete_by_filter(filter))


    async def __aenter__(self) -> "VectorStore":
        await self.aconnect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.adisconnect()

    def __enter__(self) -> "VectorStore":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.disconnect()



    async def _ensure_connected(self) -> None:
        if not self._is_connected:
            await self.aconnect()

    async def _ensure_collection(self) -> None:
        if not await self.vectordb.acollection_exists():
            await self.vectordb.acreate_collection()

    async def _embed_single(self, text: str) -> List[float]:
        if self.embedding_provider is None:
            vector_size: int = getattr(self.vectordb._config, "vector_size", 0) or 1
            return [0.0] * vector_size
        return await self.embedding_provider.embed_query(text)

    async def _embed_many(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_provider is None:
            vector_size: int = getattr(self.vectordb._config, "vector_size", 0) or 1
            return [[0.0] * vector_size for _ in texts]
        return await self.embedding_provider.embed_texts(texts)
