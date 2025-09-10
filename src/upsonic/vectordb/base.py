from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Literal

from upsonic.vectordb.config import Config 
from upsonic.schemas.vector_schemas import VectorSearchResult


class BaseVectorDBProvider(ABC):
    """
    An abstract base class that defines the operational contract for any
    vector database provider within the framework.

    This class establishes a standardized interface for all essential vector
    database operations, from lifecycle management to data manipulation and
    complex querying. Concrete implementations (e.g., ChromaProvider,
    QdrantProvider) must inherit from this class and implement all its
    abstract methods.

    The provider is initialized with a validated, immutable `Config` object,
    which serves as the single source of truth for its entire configuration.
    """

    def __init__(self, config: Config):
        """
        Initializes the provider with a complete configuration.

        Args:
            config: A validated and immutable Config object containing all
                    necessary parameters for the provider's operation.
        """
        self._config = config
        self._client: Any = None
        self._is_connected: bool = False
        print(f"Initializing {self.__class__.__name__} with provider '{self._config.core.provider_name.value}'.")

    @abstractmethod
    def connect(self) -> None:
        """
        Establishes a connection to the vector database.
        
        This method uses the connection parameters from `self._config.core`
        to initialize the database client and verify the connection.

        Raises:
            VectorDBConnectionError: If the connection fails.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Gracefully terminates the connection to the vector database.
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Performs a health check to ensure the database is responsive.

        Returns:
            True if the database is connected and responsive, False otherwise.
        """
        pass

    @abstractmethod
    def create_collection(self) -> None:
        """
        Creates the collection in the database according to the full config.

        This method reads all necessary parameters from `self._config`—including
        vector size, distance metric, indexing configuration, and sharding—to
        create and configure the collection. It should handle the
        `recreate_if_exists` logic from the core config.

        Raises:
            VectorDBConnectionError: If not connected to the database.
            VectorDBError: If the collection creation fails for other reasons.
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """
        Permanently deletes the collection specified in `self._config.core.collection_name`.

        Raises:
            VectorDBConnectionError: If not connected to the database.
            CollectionDoesNotExistError: If the collection to be deleted does not exist.
        """
        pass

    @abstractmethod
    def collection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists.

        Returns:
            True if the collection exists, False otherwise.
        """
        pass

    @abstractmethod
    def upsert(self, 
                vectors: List[List[float]], 
                payloads: List[Dict[str, Any]], 
                ids: List[Union[str, int]],
                chunks: Optional[List[str]] = None,
                sparse_vectors: Optional[List[Dict[str, Any]]] = None,  
                **kwargs) -> None:
        """
        Adds new data or updates existing data in the collection.

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
        pass

    @abstractmethod
    def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers.

        Args:
            ids: A list of specific IDs to remove.
            **kwargs: Provider-specific options.
        
        Raises:
            VectorDBError: If the deletion fails.
        """
        pass
    
    @abstractmethod
    def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs.

        Args:
            ids: A list of IDs to retrieve the full records for.
            **kwargs: Provider-specific options.

        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        pass

    @abstractmethod
    def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        A master search method that dispatches to the appropriate specialized
        search function based on the provided arguments and the provider's
        configured capabilities.

        Args:
            top_k: The number of results to return. If None, falls back to the default in the Config.
            query_vector: The vector for dense or hybrid search.
            query_text: The text for full-text or hybrid search.
            filter: An optional metadata filter.
            alpha: The weighting factor for hybrid search. If None, falls back to the default in the Config.
            fusion_method: The algorithm to use for hybrid search ('rrf' or 'weighted').
            similarity_threshold: The minimum similarity score for results. If None, falls back to the default in the Config.

        Returns:
            A list of VectorSearchResult objects.

        Raises:
            ConfigurationError: If the requested search is disabled or the
                                wrong combination of arguments is provided.
            SearchError: If any underlying search operation fails.
        """
        pass

    @abstractmethod
    def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """ 
        Performs a pure vector similarity search.

        Args:
            query_vector (List[float]): The vector embedding to search for.
            top_k (int): The number of top results to return.
            filter (Optional[Dict[str, Any]], optional): A metadata filter to apply. Defaults to None.
            similarity_threshold (Optional[float], optional): The minimum similarity score for results. Defaults to None.

        Returns:
            List[VectorSearchResult]: A list of the most similar results.
        """
        pass

    @abstractmethod
    def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a full-text search if the provider supports it.

        Args:
            query_text (str): The text string to search for.
            top_k (int): The number of top results to return.
            filter (Optional[Dict[str, Any]], optional): A metadata filter to apply. Defaults to None.
            similarity_threshold (Optional[float], optional): The minimum similarity score for results. Defaults to None.

        Returns:
            List[VectorSearchResult]: A list of matching results.
        """
        pass

    @abstractmethod
    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Combines dense and sparse/keyword search results.

        This can be implemented by the provider in two ways:
        1. Natively, if the backend supports a true hybrid search endpoint.
        2. Manually, by calling dense_search and full_text_search and fusing the results.

        Args:
            query_vector: The dense vector for the semantic part of the search.
            query_text: The raw text for the keyword/sparse part of the search.
            top_k: The number of final results to return.
            filter: An optional metadata filter.
            alpha: The weight for combining scores. If None, falls back to the default in the Config.
            fusion_method: The algorithm to use for fusing results ('rrf' or 'weighted').
            similarity_threshold: The minimum similarity score for results. If None, falls back to the default in the Config.

        Returns:
            A list of VectorSearchResult objects, ordered by the combined hybrid score.
        """
        pass
