from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Literal, Awaitable, TypeVar
from types import TracebackType
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor

from upsonic.vectordb.config import BaseVectorDBConfig
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.utils.printing import info_log

T = TypeVar('T')


class BaseVectorDBProvider(ABC):
    """
    Abstract base class defining the contract for vector database providers.

    Async-first design: all abstract methods are async (prefixed with 'a').
    Sync wrappers are provided in this base class via _run_async_from_sync.
    Providers only need to implement the async abstract methods.
    """

    _sync_loop: Optional[asyncio.AbstractEventLoop] = None
    _sync_loop_thread: Optional[Any] = None

    def __init__(self, config: Union[BaseVectorDBConfig, Dict[str, Any]]):
        self._config = BaseVectorDBConfig(**config) if isinstance(config, dict) else config
        self.client: Any = None
        self._is_connected: bool = False
        self._instance_sync_loop: Optional[asyncio.AbstractEventLoop] = None

        self.name: str = self._config.provider_name or f"{self.__class__.__name__}_{self._config.collection_name}"
        self.description: Optional[str] = self._config.provider_description
        self.id: str = self._config.provider_id or self._generate_provider_id()
        info_log(f"Initializing {self.__class__.__name__} for collection '{self._config.collection_name}'.", context="BaseVectorDBProvider")

    # ========================================================================
    # Context Managers
    # ========================================================================

    async def __aenter__(self) -> "BaseVectorDBProvider":
        await self.aconnect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.adisconnect()

    def __enter__(self) -> "BaseVectorDBProvider":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.disconnect()

    # ========================================================================
    # Sync Helpers
    # ========================================================================

    def _generate_provider_id(self) -> str:
        identifier: str = f"{self.__class__.__name__}#{self._config.collection_name}"
        return hashlib.md5(identifier.encode()).hexdigest()[:16]

    def _get_sync_loop(self) -> asyncio.AbstractEventLoop:
        if self._instance_sync_loop is None or self._instance_sync_loop.is_closed():
            self._instance_sync_loop = asyncio.new_event_loop()
        return self._instance_sync_loop

    def _run_async_from_sync(self, awaitable: Awaitable[T]) -> T:
        """
        Executes an awaitable from a synchronous method using a persistent event loop.
        When there's already a running event loop, runs in a separate thread.
        """
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                def run_in_thread() -> T:
                    thread_loop = self._get_sync_loop()
                    return thread_loop.run_until_complete(awaitable)
                future = executor.submit(run_in_thread)
                return future.result()
        except RuntimeError:
            loop = self._get_sync_loop()
            return loop.run_until_complete(awaitable)

    # ========================================================================
    # Connection Lifecycle
    # ========================================================================

    @abstractmethod
    async def aconnect(self) -> None:
        raise NotImplementedError

    def connect(self) -> None:
        return self._run_async_from_sync(self.aconnect())

    @abstractmethod
    async def adisconnect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        return self._run_async_from_sync(self.adisconnect())

    @abstractmethod
    async def ais_ready(self) -> bool:
        raise NotImplementedError

    def is_ready(self) -> bool:
        return self._run_async_from_sync(self.ais_ready())

    # ========================================================================
    # Collection Lifecycle
    # ========================================================================

    @abstractmethod
    async def acreate_collection(self) -> None:
        raise NotImplementedError

    def create_collection(self) -> None:
        return self._run_async_from_sync(self.acreate_collection())

    @abstractmethod
    async def adelete_collection(self) -> None:
        raise NotImplementedError

    def delete_collection(self) -> None:
        return self._run_async_from_sync(self.adelete_collection())

    @abstractmethod
    async def acollection_exists(self) -> bool:
        raise NotImplementedError

    def collection_exists(self) -> bool:
        return self._run_async_from_sync(self.acollection_exists())

    # ========================================================================
    # Data Operations
    # ========================================================================

    @abstractmethod
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
        raise NotImplementedError

    def upsert(
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
        return self._run_async_from_sync(
            self.aupsert(vectors, payloads, ids, chunks, document_ids, document_names, doc_content_hashes, chunk_content_hashes, sparse_vectors, metadata, knowledge_base_ids)
        )

    @abstractmethod
    async def adelete(self, ids: List[Union[str, int]]) -> None:
        raise NotImplementedError

    def delete(self, ids: List[Union[str, int]]) -> None:
        return self._run_async_from_sync(self.adelete(ids))

    async def adelete_by_id(self, ids: List[Union[str, int]]) -> None:
        return await self.adelete(ids)

    def delete_by_id(self, ids: List[Union[str, int]]) -> None:
        return self.delete(ids)

    @abstractmethod
    async def afetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        raise NotImplementedError

    def fetch(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        return self._run_async_from_sync(self.afetch(ids))

    async def afetch_by_id(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        return await self.afetch(ids)

    def fetch_by_id(self, ids: List[Union[str, int]]) -> List[VectorSearchResult]:
        return self.fetch(ids)

    # ========================================================================
    # Search Operations
    # ========================================================================

    @abstractmethod
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
        raise NotImplementedError

    def search(
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
        return self._run_async_from_sync(
            self.asearch(top_k, query_vector, query_text, filter, alpha, fusion_method, similarity_threshold, apply_reranking, sparse_query_vector)
        )

    @abstractmethod
    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        raise NotImplementedError

    def dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        return self._run_async_from_sync(
            self.adense_search(query_vector, top_k, filter, similarity_threshold, apply_reranking)
        )

    @abstractmethod
    async def afull_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        raise NotImplementedError

    def full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        return self._run_async_from_sync(
            self.afull_text_search(query_text, top_k, filter, similarity_threshold, apply_reranking, sparse_query_vector)
        )

    @abstractmethod
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
        raise NotImplementedError

    def hybrid_search(
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
        return self._run_async_from_sync(
            self.ahybrid_search(query_vector, query_text, top_k, filter, alpha, fusion_method, similarity_threshold, apply_reranking, sparse_query_vector)
        )

    # ========================================================================
    # Existence Checks
    # ========================================================================

    @abstractmethod
    async def adocument_id_exists(self, document_id: str) -> bool:
        raise NotImplementedError

    def document_id_exists(self, document_id: str) -> bool:
        return self._run_async_from_sync(self.adocument_id_exists(document_id))

    @abstractmethod
    async def adocument_name_exists(self, document_name: str) -> bool:
        raise NotImplementedError

    def document_name_exists(self, document_name: str) -> bool:
        return self._run_async_from_sync(self.adocument_name_exists(document_name))

    @abstractmethod
    async def achunk_id_exists(self, chunk_id: str) -> bool:
        raise NotImplementedError

    def chunk_id_exists(self, chunk_id: str) -> bool:
        return self._run_async_from_sync(self.achunk_id_exists(chunk_id))

    @abstractmethod
    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        raise NotImplementedError

    def doc_content_hash_exists(self, doc_content_hash: str) -> bool:
        return self._run_async_from_sync(self.adoc_content_hash_exists(doc_content_hash))

    @abstractmethod
    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        raise NotImplementedError

    def chunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        return self._run_async_from_sync(self.achunk_content_hash_exists(chunk_content_hash))

    # ========================================================================
    # Delete by Field
    # ========================================================================

    @abstractmethod
    async def adelete_by_document_name(self, document_name: str) -> bool:
        raise NotImplementedError

    def delete_by_document_name(self, document_name: str) -> bool:
        return self._run_async_from_sync(self.adelete_by_document_name(document_name))

    @abstractmethod
    async def adelete_by_document_id(self, document_id: str) -> bool:
        raise NotImplementedError

    def delete_by_document_id(self, document_id: str) -> bool:
        return self._run_async_from_sync(self.adelete_by_document_id(document_id))

    @abstractmethod
    async def adelete_by_chunk_id(self, chunk_id: str) -> bool:
        raise NotImplementedError

    def delete_by_chunk_id(self, chunk_id: str) -> bool:
        return self._run_async_from_sync(self.adelete_by_chunk_id(chunk_id))

    @abstractmethod
    async def adelete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        raise NotImplementedError

    def delete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        return self._run_async_from_sync(self.adelete_by_doc_content_hash(doc_content_hash))

    @abstractmethod
    async def adelete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        raise NotImplementedError

    def delete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        return self._run_async_from_sync(self.adelete_by_chunk_content_hash(chunk_content_hash))

    @abstractmethod
    async def adelete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        return self._run_async_from_sync(self.adelete_by_metadata(metadata))

    # ========================================================================
    # Update Metadata
    # ========================================================================

    @abstractmethod
    async def aupdate_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def update_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        return self._run_async_from_sync(self.aupdate_metadata(chunk_id, metadata))

    # ========================================================================
    # Optimize
    # ========================================================================

    @abstractmethod
    async def aoptimize(self) -> bool:
        raise NotImplementedError

    def optimize(self) -> bool:
        return self._run_async_from_sync(self.aoptimize())

    # ========================================================================
    # Supported Search Types
    # ========================================================================

    @abstractmethod
    async def aget_supported_search_types(self) -> List[str]:
        raise NotImplementedError

    def get_supported_search_types(self) -> List[str]:
        return self._run_async_from_sync(self.aget_supported_search_types())
