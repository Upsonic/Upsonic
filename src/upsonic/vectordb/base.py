from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Literal, Awaitable, TypeVar
import asyncio
from concurrent.futures import ThreadPoolExecutor

from upsonic.vectordb.config import BaseVectorDBConfig
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.utils.printing import info_log

T = TypeVar('T')


class BaseVectorDBProvider(ABC):
    """
    An abstract base class that defines the operational contract for any
    vector database provider within the framework.

    This class establishes a standardized, **async-first** interface for all essential 
    vector database operations, from lifecycle management to data manipulation and
    complex querying. Concrete implementations (e.g., ChromaProvider,
    QdrantProvider) must inherit from this class and implement all its
    abstract methods as async methods.

    The provider is initialized with a validated, immutable configuration object
    that inherits from BaseVectorDBConfig, which serves as the single source of
    truth for its entire configuration.
    
    **Design Philosophy**: All methods are async to ensure:
    - Maximum performance with I/O-bound operations
    - Seamless integration with modern async frameworks
    - Consistency across all provider implementations
    """

    def __init__(self, config: Union[BaseVectorDBConfig, Dict[str, Any]]):
        """
        Initializes the provider with a complete configuration.

        Args:
            config: A validated and immutable configuration object containing all
                    necessary parameters for the provider's operation.
        """
        self._config = BaseVectorDBConfig(**config) if isinstance(config, dict) else config
        self._client: Any = None
        self._async_client: Any = None
        self._is_connected: bool = False

        self.name = self._config.provider_name
        self.description = self._config.provider_description
        self.id = self._config.provider_id or self._generate_provider_id()
        info_log(f"Initializing {self.__class__.__name__} for collection '{self._config.collection_name}'.", context="BaseVectorDBProvider")
    
    def _run_async_from_sync(self, awaitable: Awaitable[T]) -> T:
        """
        Executes an awaitable from a synchronous method, managing the event loop intelligently.
        
        This helper method handles the case where there's already a running event loop
        (e.g., in Jupyter notebooks or async frameworks) by running the coroutine in a
        separate thread with a new event loop.
        
        Args:
            awaitable: The coroutine or other awaitable object to run.
            
        Returns:
            The result of the awaitable.
        """
        try:
            loop = asyncio.get_running_loop()
            # There's a running event loop, need to run in a separate thread
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, awaitable)
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(awaitable)

    @abstractmethod
    async def connect(self) -> None:
        """
        Establishes a connection to the vector database (async).
        
        This method uses the connection parameters from `self._config`
        to initialize the database client and verify the connection.

        Raises:
            VectorDBConnectionError: If the connection fails.
        """
        raise NotImplementedError
    
    def connect_sync(self) -> None:
        """
        Establishes a connection to the vector database (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully terminates the connection to the vector database (async).
        """
        raise NotImplementedError

    def disconnect_sync(self) -> None:
        """
        Gracefully terminates the connection to the vector database (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def is_ready(self) -> bool:
        """
        Performs a health check to ensure the database is responsive (async).

        Returns:
            True if the database is connected and responsive, False otherwise.
        """
        raise NotImplementedError

    def is_ready_sync(self) -> bool:
        """
        Performs a health check to ensure the database is responsive (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def create_collection(self) -> None:
        """
        Creates the collection in the database according to the full config (async).

        This method reads all necessary parameters from `self._config`—including
        vector size, distance metric, and provider-specific settings—to
        create and configure the collection. It should handle the
        `recreate_if_exists` logic.

        Raises:
            VectorDBConnectionError: If not connected to the database.
            VectorDBError: If the collection creation fails for other reasons.
        """
        raise NotImplementedError

    def create_collection_sync(self) -> None:
        """
        Creates the collection in the database according to the full config (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_collection(self) -> None:
        """
        Permanently deletes the collection specified in `self._config.collection_name` (async).

        Raises:
            VectorDBConnectionError: If not connected to the database.
            CollectionDoesNotExistError: If the collection to be deleted does not exist.
        """
        raise NotImplementedError

    def delete_collection_sync(self) -> None:
        """
        Permanently deletes the collection specified in `self._config.collection_name` (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def collection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists (async).

        Returns:
            True if the collection exists, False otherwise.
        """
        raise NotImplementedError

    def collection_exists_sync(self) -> bool:
        """
        Checks if the collection specified in the config already exists (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, 
                vectors: List[List[float]], 
                payloads: List[Dict[str, Any]], 
                ids: List[Union[str, int]],
                chunks: Optional[List[str]] = None,
                sparse_vectors: Optional[List[Dict[str, Any]]] = None,  
                **kwargs) -> None:
        """
        Adds new data or updates existing data in the collection (async).

        This method is designed to be flexible, handling dense-only, sparse-only,
        or hybrid (dense + sparse) data based on the provided arguments.
        Implementations must validate that the provided data aligns with the
        collection's configured capabilities (e.g., rejecting sparse vectors if
        the index is dense-only).

        Args:
            vectors: A list of dense vector embeddings.
            payloads: A list of corresponding metadata objects.
            ids: A list of unique identifiers for each record.
            sparse_vectors: An optional list of sparse vector representations.
                            Each sparse vector should be a dict, e.g.,
                            {'indices': [...], 'values': [...]}.
            **kwargs: Provider-specific options.

        Raises:
            UpsertError: If the data ingestion fails or if the provided data
                         is inconsistent with the collection's configuration.
        """
        raise NotImplementedError

    def upsert_sync(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]] = None, sparse_vectors: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        """
        Adds new data or updates existing data in the collection (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers (async).

        Args:
            ids: A list of specific IDs to remove.
            **kwargs: Provider-specific options.
        
        Raises:
            VectorDBError: If the deletion fails.
        """
        raise NotImplementedError
    
    def delete_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers (sync).

        Args:
            ids: A list of specific IDs to remove.
            **kwargs: Provider-specific options.
        
        Raises:
            VectorDBError: If the deletion fails.
        """
        raise NotImplementedError
    
    # Backward compatibility aliases
    async def delete_by_id(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Alias for delete() - provided for backward compatibility."""
        return await self.delete(ids, **kwargs)
    
    def delete_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Alias for delete_sync() - provided for backward compatibility."""
        return self.delete_sync(ids, **kwargs)

    @abstractmethod
    async def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs (async).

        Args:
            ids: A list of IDs to retrieve the full records for.
            **kwargs: Provider-specific options.

        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        raise NotImplementedError

    def fetch_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs (sync).
        """
        raise NotImplementedError
    
    # Backward compatibility aliases
    async def fetch_by_id(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Alias for fetch() - provided for backward compatibility."""
        return await self.fetch(ids, **kwargs)
    
    def fetch_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Alias for fetch_sync() - provided for backward compatibility."""
        return self.fetch_sync(ids, **kwargs)

    @abstractmethod
    async def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        A master search method that dispatches to the appropriate specialized
        search function based on the provided arguments and the provider's
        configured capabilities (async).

        Args:
            top_k: The number of results to return. If None, falls back to default_top_k in config.
            query_vector: The vector for dense or hybrid search.
            query_text: The text for full-text or hybrid search.
            filter: An optional metadata filter.
            alpha: The weighting factor for hybrid search.
            fusion_method: The algorithm to use for hybrid search ('rrf' or 'weighted').
            similarity_threshold: The minimum similarity score for results. If None, falls back to default_similarity_threshold in config.

        Returns:
            A list of VectorSearchResult objects.

        Raises:
            ConfigurationError: If the requested search is disabled or the
                                wrong combination of arguments is provided.
            SearchError: If any underlying search operation fails.
        """
        raise NotImplementedError

    def search_sync(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        A master search method that dispatches to the appropriate specialized
        search function based on the provided arguments and the provider's
        configured capabilities (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """ 
        Performs a pure vector similarity search (async).

        Args:
            query_vector (List[float]): The vector embedding to search for.
            top_k (int): The number of top results to return.
            filter (Optional[Dict[str, Any]], optional): A metadata filter to apply. Defaults to None.
            similarity_threshold (Optional[float], optional): The minimum similarity score for results. Defaults to None.

        Returns:
            List[VectorSearchResult]: A list of the most similar results.
        """
        raise NotImplementedError

    def dense_search_sync(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a pure vector similarity search (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a full-text search if the provider supports it (async).

        Args:
            query_text (str): The text string to search for.
            top_k (int): The number of top results to return.
            filter (Optional[Dict[str, Any]], optional): A metadata filter to apply. Defaults to None.
            similarity_threshold (Optional[float], optional): The minimum similarity score for results. Defaults to None.

        Returns:
            List[VectorSearchResult]: A list of matching results.
        """
        raise NotImplementedError

    def full_text_search_sync(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a full-text search if the provider supports it (sync).
        """
        raise NotImplementedError

    @abstractmethod
    async def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Combines dense and sparse/keyword search results (async).

        This can be implemented by the provider in two ways:
        1. Natively, if the backend supports a true hybrid search endpoint.
        2. Manually, by calling dense_search and full_text_search and fusing the results.

        Args:
            query_vector: The dense vector for the semantic part of the search.
            query_text: The raw text for the keyword/sparse part of the search.
            top_k: The number of final results to return.
            filter: An optional metadata filter.
            alpha: The weight for combining scores.
            fusion_method: The algorithm to use for fusing results ('rrf' or 'weighted').
            similarity_threshold: The minimum similarity score for results. If None, falls back to default_similarity_threshold in config.

        Returns:
            A list of VectorSearchResult objects, ordered by the combined hybrid score.
        """
        raise NotImplementedError

    def hybrid_search_sync(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Combines dense and sparse/keyword search results (sync).
        """
        raise NotImplementedError

    @abstractmethod
    def document_id_exists(self, document_id: str) -> bool:
        """
        Checks if a document ID exists in the vector database (sync).

        Args:
            document_id: The document ID to check.

        Returns:
            True if the document ID exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_document_id_exists(self, document_id: str) -> bool:
        """
        Checks if a document ID exists in the vector database (async).

        Args:
            document_id: The document ID to check.

        Returns:
            True if the document ID exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def document_name_exists(self, document_name: str) -> bool:
        """
        Checks if a document name exists in the vector database (sync).

        Args:
            document_name: The document name to check.

        Returns:
            True if the document name exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_document_name_exists(self, document_name: str) -> bool:
        """
        Checks if a document name exists in the vector database (async).

        Args:
            document_name: The document name to check.

        Returns:
            True if the document name exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def content_id_exists(self, content_id: str) -> bool:
        """
        Checks if a content ID exists in the vector database (sync).

        Args:
            content_id: The content ID to check.

        Returns:
            True if the content ID exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_content_id_exists(self, content_id: str) -> bool:
        """
        Checks if a content ID exists in the vector database (async).

        Args:
            content_id: The content ID to check.

        Returns:
            True if the content ID exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def optimize(self) -> bool:
        """
        Optimizes the vector database (sync).

        Returns:
            True if the optimization was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_optimize(self) -> bool:
        """
        Optimizes the vector database (async).

        Returns:
            True if the optimization was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_by_document_name(self, document_name: str) -> bool:
        """
        Removes data from the collection by their document name (sync).

        Args:
            document_name: The document name to match for deletion.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_delete_by_document_name(self, document_name: str) -> bool:
        """
        Removes data from the collection by their document name (async).

        Args:
            document_name: The document name to match for deletion.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Removes data from the collection by their document ID (sync).

        Args:
            document_id: The document ID to match for deletion.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_delete_by_document_id(self, document_id: str) -> bool:
        """
        Removes data from the collection by their document ID (async).

        Args:
            document_id: The document ID to match for deletion.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_by_content_id(self, content_id: str) -> bool:
        """
        Removes data from the collection by their content ID (sync).

        Args:
            content_id: The content ID to match for deletion.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_delete_by_content_id(self, content_id: str) -> bool:
        """
        Removes data from the collection by their content ID (async).

        Args:
            content_id: The content ID to match for deletion.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Removes data from the collection by their metadata (sync).

        Args:
            metadata: The metadata to match for deletion.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Removes data from the collection by their metadata (async).

        Args:
            metadata: The metadata to match for deletion.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Updates the metadata for a specific content ID (sync).

        Args:
            content_id: The content ID to update.
            metadata: The metadata to update/merge.

        Returns:
            True if the update was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Updates the metadata for a specific content ID (async).

        Args:
            content_id: The content ID to update.
            metadata: The metadata to update/merge.

        Returns:
            True if the update was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_supported_search_types(self) -> List[str]:
        """
        Gets the supported search types for the vector database (sync).

        Returns:
            A list of supported search types.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_get_supported_search_types(self) -> List[str]:
        """
        Gets the supported search types for the vector database (async).

        Returns:
            A list of supported search types.
        """
        raise NotImplementedError