from __future__ import annotations

import asyncio
import uuid
import json
import os
from hashlib import md5
from typing import Any, Dict, List, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import weaviate
    from weaviate import WeaviateAsyncClient
    import weaviate.classes as wvc
    from weaviate.exceptions import (
        WeaviateConnectionError,
        UnexpectedStatusCodeError,
    )
    from weaviate.util import generate_uuid5
    from weaviate.classes.query import HybridFusion, Rerank

try:
    import weaviate
    from weaviate import WeaviateAsyncClient
    import weaviate.classes as wvc
    from weaviate.exceptions import (
        WeaviateConnectionError,
        UnexpectedStatusCodeError,
    )
    from weaviate.util import generate_uuid5
    from weaviate.classes.query import HybridFusion, Rerank
    from weaviate.classes.init import Auth
    _WEAVIATE_AVAILABLE = True
except ImportError:
    weaviate = None  # type: ignore
    WeaviateAsyncClient = None  # type: ignore
    wvc = None  # type: ignore
    WeaviateConnectionError = None  # type: ignore
    UnexpectedStatusCodeError = None  # type: ignore
    generate_uuid5 = None  # type: ignore
    HybridFusion = None  # type: ignore
    Rerank = None  # type: ignore
    _WEAVIATE_AVAILABLE = False


from upsonic.vectordb.config import (
    WeaviateConfig,
    Mode, 
    DistanceMetric,
    HNSWIndexConfig,
    FlatIndexConfig
)
from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.utils.printing import info_log, debug_log

from upsonic.utils.package.exception import(
    VectorDBConnectionError, 
    ConfigurationError, 
    CollectionDoesNotExistError,
    VectorDBError,
    SearchError,
    UpsertError
)

from upsonic.schemas.vector_schemas import VectorSearchResult


class WeaviateProvider(BaseVectorDBProvider):
    """
    A comprehensive async-first implementation of BaseVectorDBProvider for Weaviate vector database.
    
    This provider offers a high-level, dynamic interface with support for:
    - Async operations for maximum performance
    - Dense vector indexing and search
    - Flexible metadata management with configurable indexing
    - Multiple connection modes (cloud, local, embedded, in-memory)
    - Advanced search capabilities:
      * Dense vector search (semantic)
      * Full-text BM25 search (keyword)
      * Hybrid search (combines dense vectors + BM25)
    - Comprehensive data lifecycle management
    
    Key Features:
    - Auto-generation of content_id if not provided
    - Configurable field indexing for fast filtering
    - Default metadata support
    - Multi-tenancy support via namespaces
    - Replication and sharding configuration
    - Optional generative AI and reranker modules
    - Proper error handling with custom exceptions
    
    Note: Weaviate does NOT support sparse vectors. Hybrid search combines
    dense vector similarity with BM25 keyword search (inverted index).
    """

    def __init__(self, config: Union[WeaviateConfig, Dict[str, Any]]):
        """
        Initializes the WeaviateProvider with a configuration.
        
        Args:
            config: Either a WeaviateConfig object or a dictionary that can be converted to one.
        
        Raises:
            ConfigurationError: If the configuration is invalid or Weaviate client is not installed.
        """
        if not _WEAVIATE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="weaviate-client",
                install_command='pip install "upsonic[rag]"',
                feature_name="Weaviate vector database provider"
            )
        
        # Convert dict to WeaviateConfig if needed
        if isinstance(config, dict):
            config = WeaviateConfig.from_dict(config)
        
        super().__init__(config)
        
        # Provider metadata
        self.provider_name = config.provider_name or f"WeaviateProvider_{config.collection_name}"
        self.provider_description = config.provider_description
        self.provider_id = config.provider_id or self._generate_provider_id()
        
        self._client: Optional[weaviate.WeaviateClient] = None
        self._async_client: Optional[WeaviateAsyncClient] = None
        info_log(
            f"WeaviateProvider initialized for collection '{self._config.collection_name}' "
            f"in '{self._config.connection.mode.value}' mode.",
            context="WeaviateVectorDB"
        )

    # ============================================================================
    # Provider Metadata
    # ============================================================================
    
    def _generate_provider_id(self) -> str:
        """Generates a unique provider ID based on connection details and collection."""
        conn = self._config.connection
        identifier_parts = [
            conn.host or conn.url or "local",
            str(conn.port) if conn.port else "",
            self._config.collection_name
        ]
        identifier = "#".join(filter(None, identifier_parts))
        
        return md5(identifier.encode()).hexdigest()[:16]
    
    # ============================================================================
    # Connection Management
    # ============================================================================

    def _build_api_headers(self) -> Dict[str, str]:
        """
        Build API headers for generative AI and reranker modules.
        
        Supports all Weaviate API-based model provider integrations:
        https://docs.weaviate.io/weaviate/model-providers
        
        Returns:
            A dictionary of headers with API keys for configured providers.
        """
        headers = {}
        
        # Provider to header key mapping (based on Weaviate documentation)
        provider_header_map = {
            # API-based providers
            'anthropic': 'X-Anthropic-Api-Key',
            'anyscale': 'X-Anyscale-Api-Key',
            'aws': 'X-AWS-Access-Key',  # Also needs X-AWS-Secret-Key
            'cohere': 'X-Cohere-Api-Key',
            'contextualai': 'X-Contextual-Api-Key',
            'databricks': 'X-Databricks-Token',
            'friendliai': 'X-FriendliAI-Api-Key',
            'google': 'X-Google-Api-Key',  # or X-Google-Vertex-Api-Key
            'huggingface': 'X-HuggingFace-Api-Key',
            'jinaai': 'X-JinaAI-Api-Key',
            'jina': 'X-JinaAI-Api-Key',  # Alias
            'mistral': 'X-Mistral-Api-Key',
            'nvidia': 'X-NVIDIA-Api-Key',
            'octoai': 'X-OctoAI-Api-Key',
            'openai': 'X-OpenAI-Api-Key',
            'azure': 'X-Azure-Api-Key',
            'azure_openai': 'X-Azure-Api-Key',
            'voyageai': 'X-VoyageAI-Api-Key',
            'voyage': 'X-VoyageAI-Api-Key',  # Alias
            'xai': 'X-xAI-Api-Key',
        }
        
        # Environment variable mapping (fallback if not in config)
        env_var_map = {
            'anthropic': 'ANTHROPIC_APIKEY',
            'anyscale': 'ANYSCALE_APIKEY',
            'aws': 'AWS_ACCESS_KEY',
            'cohere': 'COHERE_APIKEY',
            'contextualai': 'CONTEXTUAL_APIKEY',
            'databricks': 'DATABRICKS_TOKEN',
            'friendliai': 'FRIENDLIAI_APIKEY',
            'google': 'GOOGLE_APIKEY',
            'huggingface': 'HUGGINGFACE_APIKEY',
            'jinaai': 'JINAAI_APIKEY',
            'jina': 'JINAAI_APIKEY',
            'mistral': 'MISTRAL_APIKEY',
            'nvidia': 'NVIDIA_APIKEY',
            'octoai': 'OCTOAI_APIKEY',
            'openai': 'OPENAI_APIKEY',
            'azure': 'AZURE_APIKEY',
            'azure_openai': 'AZURE_APIKEY',
            'voyageai': 'VOYAGEAI_APIKEY',
            'voyage': 'VOYAGEAI_APIKEY',
            'xai': 'XAI_APIKEY',
        }
        
        # Determine which providers need API keys based on config
        providers_needed = set()
        
        if self._config.generative_config:
            provider = self._config.generative_config.get('provider', '').lower()
            if provider:
                providers_needed.add(provider)
        
        if self._config.reranker_config:
            provider = self._config.reranker_config.get('provider', '').lower()
            if provider:
                providers_needed.add(provider)
        
        # Build headers from config or environment variables
        for provider in providers_needed:
            if provider not in provider_header_map:
                continue
            
            header_key = provider_header_map[provider]
            api_key = None
            
            # First try to get from config
            if self._config.api_keys and provider in self._config.api_keys:
                api_key = self._config.api_keys[provider]
            
            # Fallback to environment variable
            if not api_key and provider in env_var_map:
                api_key = os.getenv(env_var_map[provider])
            
            if api_key:
                headers[header_key] = api_key
                debug_log(
                    f"Added API key header for provider '{provider}': {header_key}",
                    context="WeaviateVectorDB"
                )
        
        return headers

    async def connect(self) -> None:
        """
        Establishes an async connection to the Weaviate vector database instance.
        
        This method interprets the configuration to determine the connection mode
        (cloud, local, embedded) and uses the appropriate Weaviate async client constructor.
        
        Raises:
            VectorDBConnectionError: If the connection fails for any reason.
        """
        if self._is_connected and self._async_client:
            info_log("Already connected to Weaviate.", context="WeaviateVectorDB")
            return

        debug_log(
            f"Attempting to connect to Weaviate in '{self._config.connection.mode.value}' mode...",
            context="WeaviateVectorDB"
        )
        
        # Build API headers for generative/reranker modules
        # TODO: ADD GENERATIVE PARAMETER SETTING SUPPORT FOR SEARCH METHODS!
        additional_headers = self._build_api_headers()
        
        try:
            if self._config.connection.mode == Mode.CLOUD:
                if not self._config.connection.host or not self._config.connection.api_key:
                    raise ConfigurationError("Cloud mode requires 'host' (cluster URL) and 'api_key'.")
                
                auth_credentials = Auth.api_key(self._config.connection.api_key.get_secret_value())
                additional_config = wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=60, query=30, insert=30),
                    startup_period=30
                )
                self._async_client = weaviate.use_async_with_weaviate_cloud(
                    cluster_url=self._config.connection.host,
                    auth_credentials=auth_credentials,
                    headers=additional_headers if additional_headers else None,  # Pass API key headers
                    additional_config=additional_config,
                    skip_init_checks=True  # Skip gRPC health checks for network issues
                )

            elif self._config.connection.mode == Mode.LOCAL:
                if not self._config.connection.host or not self._config.connection.port:
                    raise ConfigurationError("Local mode requires 'host' and 'port'.")

                self._async_client = weaviate.use_async_with_local(
                    host=self._config.connection.host,
                    port=self._config.connection.port
                )

            elif self._config.connection.mode in (Mode.EMBEDDED, Mode.IN_MEMORY):
                persistence_path = (
                    self._config.connection.db_path 
                    if self._config.connection.mode == Mode.EMBEDDED 
                    else None
                )
                
                self._async_client = weaviate.use_async_with_embedded(
                    persistence_data_path=persistence_path
                )
            
            else:
                raise ConfigurationError(
                    f"Unsupported Weaviate mode: {self._config.connection.mode.value}"
                )

            # Connect and verify
            await self._async_client.connect()
            
            if not await self._async_client.is_ready():
                raise WeaviateConnectionError("Health check failed after connection attempt.")

            self._is_connected = True
            info_log(
                "Successfully connected to Weaviate and health check passed.",
                context="WeaviateVectorDB"
            )

        except WeaviateConnectionError as e:
            self._async_client = None
            self._is_connected = False
            raise VectorDBConnectionError(f"Failed to connect to Weaviate: {e}")
        except Exception as e:
            self._async_client = None
            self._is_connected = False
            raise VectorDBConnectionError(
                f"An unexpected error occurred during connection: {e}"
            )

    async def disconnect(self) -> None:
        """
        Gracefully terminates the connection to the Weaviate database.
        
        This method is idempotent; calling it on an already disconnected
        provider will not raise an error.
        """
        if self._async_client and self._is_connected:
            try:
                await self._async_client.close()
                self._is_connected = False
                self._async_client = None
                info_log("Successfully disconnected from Weaviate.", context="WeaviateVectorDB")
            except Exception as e:
                self._is_connected = False
                self._async_client = None
                debug_log(
                    f"An error occurred during disconnection, but status is now 'disconnected'. Error: {e}",
                    context="WeaviateVectorDB"
                )
        else:
            debug_log("Already disconnected. No action taken.", context="WeaviateVectorDB")

    def connect_sync(self) -> None:
        """Establishes a connection to the vector database (sync)."""
        return self._run_async_from_sync(self.connect())

    def disconnect_sync(self) -> None:
        """Gracefully terminates the connection to the vector database (sync)."""
        return self._run_async_from_sync(self.disconnect())

    async def is_ready(self) -> bool:
        """
        Performs a health check to ensure the Weaviate instance is responsive.
        
        Returns:
            True if the client is connected and the database is responsive, False otherwise.
        """
        if not self._async_client or not self._is_connected:
            return False
        
        try:
            return await self._async_client.is_ready()
        except WeaviateConnectionError:
            self._is_connected = False
            return False

    def is_ready_sync(self) -> bool:
        """Performs a health check to ensure the database is responsive (sync)."""
        return self._run_async_from_sync(self.is_ready())

    # ============================================================================
    # Collection Management
    # ============================================================================

    async def create_collection(self, **kwargs) -> None:
        """
        Creates the collection in Weaviate with comprehensive configuration.
        
        This method creates a collection with:
        - Proper vector configuration (dense and optionally sparse)
        - Metadata properties with optional indexing
        - Multi-tenancy support if configured
        - Replication, sharding, inverted index configuration
        - Optional generative and reranker modules
        
        All configuration can be provided via config or overridden via kwargs.
        
        Args:
            **kwargs: Override config parameters:
                - description: Collection description
                - inverted_index_config: Dict for inverted index config
                - multi_tenancy_config: Dict for multi-tenancy config
                - replication_config: Dict for replication config
                - sharding_config: Dict for sharding config
                - generative_config: Dict for generative AI config
                - reranker_config: Dict for reranker config
                - properties: List of additional properties
                - references: List of cross-references
        
        Raises:
            VectorDBConnectionError: If not connected to the database.
            VectorDBError: If the collection creation fails.
        """
        if not self._is_connected or not self._async_client:
            raise VectorDBConnectionError(
                "Must be connected to Weaviate before creating a collection."
            )

        collection_name = self._config.collection_name

        if await self.collection_exists():
            if self._config.recreate_if_exists:
                info_log(
                    f"Collection '{collection_name}' already exists. "
                    f"Deleting and recreating as requested.",
                    context="WeaviateVectorDB"
                )
                await self.delete_collection()
            else:
                info_log(
                    f"Collection '{collection_name}' already exists and "
                    f"'recreate_if_exists' is False. No action taken.",
                    context="WeaviateVectorDB"
                )
                return

        try:
            # Distance metric mapping
            distance_map = {
                DistanceMetric.COSINE: wvc.config.VectorDistances.COSINE,
                DistanceMetric.DOT_PRODUCT: wvc.config.VectorDistances.DOT,
                DistanceMetric.EUCLIDEAN: wvc.config.VectorDistances.L2_SQUARED,
            }
            
            # Build all configurations
            description = kwargs.get('description', self._config.description)
            vector_config = self._build_vector_config(distance_map)
            properties = self._build_properties_schema(
                additional_properties=kwargs.get('properties')
            )
            inverted_index_config = self._build_inverted_index_config(
                kwargs.get('inverted_index_config')
            )
            multi_tenancy_config = self._build_multi_tenancy_config(
                kwargs.get('multi_tenancy_config')
            )
            replication_config = self._build_replication_config(
                kwargs.get('replication_config')
            )
            sharding_config = self._build_sharding_config(
                kwargs.get('sharding_config')
            )
            generative_config = self._build_generative_config(
                kwargs.get('generative_config')
            )
            reranker_config = self._build_reranker_config(
                kwargs.get('reranker_config')
            )
            references = self._build_references(
                kwargs.get('references')
            )
            
            # Create collection with all configurations
            await self._async_client.collections.create(
                name=collection_name,
                description=description,
                vector_config=vector_config,
                properties=properties,
                references=references,
                inverted_index_config=inverted_index_config,
                multi_tenancy_config=multi_tenancy_config,
                replication_config=replication_config,
                sharding_config=sharding_config,
                generative_config=generative_config,
                reranker_config=reranker_config
            )
            
            info_log(
                f"Successfully created collection '{collection_name}'.",
                context="WeaviateVectorDB"
            )

            # Create tenant if namespace is configured and multi-tenancy is enabled
            if self._config.namespace and self._config.multi_tenancy_enabled:
                debug_log(
                    f"Creating tenant: '{self._config.namespace}'...",
                    context="WeaviateVectorDB"
                )
                collection = self._async_client.collections.get(collection_name)
                await collection.tenants.create(
                    tenants=[weaviate.collections.classes.tenants.Tenant(
                        name=self._config.namespace
                    )]
                )
                info_log("Tenant created successfully.", context="WeaviateVectorDB")

        except UnexpectedStatusCodeError as e:
            raise VectorDBError(
                f"Failed to create collection '{collection_name}' in Weaviate. "
                f"Status: {e.status_code}. Message: {e.message}"
            )
        except Exception as e:
            raise VectorDBError(
                f"An unexpected error occurred during collection creation: {e}"
            )

    def create_collection_sync(self) -> None:
        """Creates the collection in the database according to the full config (sync)."""
        return self._run_async_from_sync(self.create_collection())

    async def delete_collection(self) -> None:
        """
        Permanently deletes the collection specified in the config from Weaviate.
        
        Raises:
            VectorDBConnectionError: If not connected to the database.
            CollectionDoesNotExistError: If the collection does not exist.
            VectorDBError: For other unexpected API or operational errors.
        """
        if not self._is_connected or not self._async_client:
            raise VectorDBConnectionError(
                "Must be connected to Weaviate before deleting a collection."
            )
        
        collection_name = self._config.collection_name
        
        try:
            await self._async_client.collections.delete(collection_name)
            info_log(
                f"Successfully deleted collection '{collection_name}'.",
                context="WeaviateVectorDB"
            )
        except UnexpectedStatusCodeError as e:
            if e.status_code == 404: 
                raise CollectionDoesNotExistError(
                    f"Collection '{collection_name}' could not be deleted because it does not exist."
                )
            else:
                raise VectorDBError(
                    f"API error while deleting collection '{collection_name}': {e.message}"
                )
        except Exception as e:
            raise VectorDBError(
                f"An unexpected error occurred during collection deletion: {e}"
            )

    def delete_collection_sync(self) -> None:
        """Permanently deletes the collection specified in `self._config.collection_name` (sync)."""
        return self._run_async_from_sync(self.delete_collection())

    async def collection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists in Weaviate.
        
        Returns:
            True if the collection exists, False otherwise.
        
        Raises:
            VectorDBConnectionError: If not connected to the database.
        """
        if not self._is_connected or not self._async_client:
            raise VectorDBConnectionError(
                "Must be connected to Weaviate to check for a collection's existence."
            )
        
        return await self._async_client.collections.exists(self._config.collection_name)

    def collection_exists_sync(self) -> bool:
        """Checks if the collection specified in the config already exists (sync)."""
        return self._run_async_from_sync(self.collection_exists())

    # ============================================================================
    # Data Operations
    # ============================================================================

    async def upsert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """
        Adds new data or updates existing data in the collection using Weaviate's batching system.
        
        This method handles:
        - Dense vectors (required)
        - Metadata with document_name, document_id, content_id, metadata, content
        - Auto-generation of content_id if not provided
        - Merging with default_metadata from config
        
        Note: Weaviate does NOT support sparse vectors. The sparse_vectors parameter is
        ignored for compatibility with the base interface but should not be used.
        
        Args:
            vectors: A list of dense vector embeddings.
            payloads: A list of corresponding metadata objects.
            ids: A list of unique identifiers (will be used as document_id if not in payload).
            chunks: A list of text chunks (required - becomes 'content' field).
            sparse_vectors: IGNORED - Weaviate does not support sparse vectors.
            **kwargs: Provider-specific options:
                - metadata: Additional metadata to merge (Dict[str, Any])
        
        Raises:
            UpsertError: If the data ingestion fails.
            VectorDBConnectionError: If not connected to the database.
        """
        # Validation
        if not (len(vectors) == len(payloads) == len(ids)):
            raise UpsertError(
                "The lengths of vectors, payloads, and ids lists must be identical."
            )
        
        if not vectors:
            debug_log("Upsert called with empty lists. No action taken.", context="WeaviateVectorDB")
            return
        
        if chunks is None:
            raise UpsertError("chunks (content) is required and cannot be None.")
        
        if len(chunks) != len(payloads):
            raise UpsertError(
                "The lengths of chunks and payloads lists must be identical."
            )
        
        # Warn if sparse vectors provided
        if sparse_vectors is not None:
            debug_log(
                "Warning: Weaviate does not support sparse vectors. sparse_vectors parameter is ignored.",
                context="WeaviateVectorDB"
            )

        collection_obj = await self._get_collection()

        try:
            info_log(
                f"Starting upsert of {len(vectors)} objects...",
                context="WeaviateVectorDB"
            )
            
            # Get additional metadata from kwargs
            extra_metadata = kwargs.get('metadata', {})
            
            # Insert objects one by one (async Weaviate doesn't have batch context manager)
            for i in range(len(vectors)):
                # Process payload and generate properties
                properties = self._process_payload(
                    payload=payloads[i],
                    content=chunks[i],
                    document_id=str(ids[i]),
                    extra_metadata=extra_metadata
                )
                
                # Generate UUID
                try:
                    object_uuid = uuid.UUID(str(ids[i]))
                except ValueError:
                    # Generate deterministic UUID from the ID itself (not content_id)
                    # This ensures fetch can find the same objects using the same IDs
                    object_uuid = generate_uuid5(
                        identifier=str(ids[i]),
                        namespace=self._config.collection_name
                    )
                
                # Insert object with dense vector only
                await collection_obj.data.insert(
                    properties=properties,
                    vector=vectors[i],  # Single dense vector
                    uuid=object_uuid
                )
            
            info_log(
                f"Successfully upserted {len(vectors)} objects.",
                context="WeaviateVectorDB"
            )

        except Exception as e:
            raise UpsertError(
                f"Failed to upsert data to Weaviate collection '{self._config.collection_name}': {e}"
            )

    def upsert_sync(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]] = None, sparse_vectors: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        """Adds new data or updates existing data in the collection (sync)."""
        return self._run_async_from_sync(self.upsert(vectors, payloads, ids, chunks, sparse_vectors, **kwargs))

    async def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers.
        
        Args:
            ids: A list of specific IDs to remove.
            **kwargs: Ignored.
        
        Raises:
            VectorDBError: If the deletion fails.
        """
        if not ids:
            debug_log(
                "Delete called with an empty list of IDs. No action taken.",
                context="WeaviateVectorDB"
            )
            return
        
        collection_obj = await self._get_collection()

        uuids_to_delete = []
        for item_id in ids:
            try:
                uuids_to_delete.append(uuid.UUID(str(item_id)))
            except ValueError:
                uuids_to_delete.append(
                    generate_uuid5(
                        identifier=str(item_id),
                        namespace=self._config.collection_name
                    )
                )

        try:
            delete_filter = wvc.query.Filter.by_id().contains_any(uuids_to_delete)
            result = await collection_obj.data.delete_many(where=delete_filter)
            
            if result.failed > 0:
                raise VectorDBError(
                    f"Deletion partially failed. Successful: {result.successful}, "
                    f"Failed: {result.failed}. Check Weaviate logs for details."
                )

            info_log(
                f"Successfully processed deletion request for {len(ids)} IDs. "
                f"Matched and deleted: {result.successful}.",
                context="WeaviateVectorDB"
            )

        except Exception as e:
            raise VectorDBError(f"An error occurred during deletion: {e}")

    def delete_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Removes data from the collection by their unique identifiers (sync)."""
        return self._run_async_from_sync(self.delete(ids, **kwargs))

    def delete_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Alias for delete_sync() - provided for backward compatibility."""
        return self.delete_sync(ids, **kwargs)

    async def fetch(
        self,
        ids: List[Union[str, int]],
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs.
        
        Args:
            ids: A list of IDs to retrieve the full records for.
            **kwargs: Ignored.
        
        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        if not ids:
            return []
            
        collection_obj = await self._get_collection()

        uuids_to_fetch = []
        for item_id in ids:
            try:
                uuids_to_fetch.append(uuid.UUID(str(item_id)))
            except ValueError:
                uuids_to_fetch.append(
                    generate_uuid5(
                        identifier=str(item_id),
                        namespace=self._config.collection_name
                    )
                )

        try:
            fetch_filter = wvc.query.Filter.by_id().contains_any(uuids_to_fetch)

            response = await collection_obj.query.fetch_objects(
                limit=len(ids),
                filters=fetch_filter,
                include_vector=True
            )
            
            results = []
            for obj in response.objects:
                # Extract vector (handle both single and named vectors)
                vector = self._extract_vector(obj.vector)
                
                # Extract content from properties
                content = obj.properties.get("content", "")
                
                results.append(VectorSearchResult(
                    id=str(obj.uuid),
                    score=1.0,  # No score for direct fetch
                    payload=obj.properties,
                    vector=vector,
                    text=content
                ))
            
            return results
            
        except Exception as e:
            error_message = str(e).lower()
            if "could not find class" in error_message and "in schema" in error_message:
                raise CollectionDoesNotExistError(
                    f"Collection '{self._config.collection_name}' does not exist in Weaviate."
                )
            else:
                raise VectorDBError(f"An error occurred while fetching objects: {e}")

    def fetch_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Retrieves full records (payload and vector) by their IDs (sync)."""
        return self._run_async_from_sync(self.fetch(ids, **kwargs))

    def fetch_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Alias for fetch_sync() - provided for backward compatibility."""
        return self.fetch_sync(ids, **kwargs)

    # ============================================================================
    # Search Operations
    # ============================================================================

    async def search(
        self,
        top_k: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        A master search method that dispatches to the appropriate specialized search function.
        
        Args:
            top_k: The number of results to return.
            query_vector: The vector for dense or hybrid search.
            query_text: The text for full-text or hybrid search.
            filter: An optional metadata filter dictionary.
            alpha: The weighting factor for hybrid search (0.0 = pure keyword, 1.0 = pure vector).
            fusion_method: The algorithm to use for hybrid search ('rrf' or 'weighted').
            similarity_threshold: The minimum similarity score for results.
            **kwargs: Additional provider-specific options.
        
        Returns:
            A list of VectorSearchResult objects.
        
        Raises:
            ConfigurationError: If the requested search is not possible with provided arguments.
            SearchError: If any underlying search operation fails.
        """
        filter = filter if filter is not None else None
        final_top_k = top_k if top_k is not None else self._config.default_top_k or 10
        fusion_method = (
            fusion_method if fusion_method is not None 
            else self._config.default_fusion_method or 'weighted'
        )

        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None

        if is_dense:
            if self._config.dense_search_enabled is False:
                raise ConfigurationError(
                    "Dense search is disabled by the current configuration."
                )
            return await self.dense_search(
                query_vector=query_vector,
                top_k=final_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        
        elif is_full_text:
            if self._config.full_text_search_enabled is False:
                raise ConfigurationError(
                    "Full-text search is disabled by the current configuration."
                )
            return await self.full_text_search(
                query_text=query_text,
                top_k=final_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                **kwargs
            )

        elif is_hybrid:
            if self._config.hybrid_search_enabled is False:
                raise ConfigurationError(
                    "Hybrid search is disabled by the current configuration."
                )
            final_alpha = alpha if alpha is not None else self._config.default_hybrid_alpha or 0.5
            return await self.hybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=final_top_k,
                filter=filter,
                alpha=final_alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        else:
            raise ConfigurationError(
                "Search requires at least one of 'query_vector' or 'query_text'."
            )

    def search_sync(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """A master search method that dispatches to the appropriate specialized search function (sync)."""
        return self._run_async_from_sync(self.search(top_k, query_vector, query_text, filter, alpha, fusion_method, similarity_threshold, **kwargs))

    async def dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Performs a pure vector similarity search using Weaviate's `near_vector` query.
        
        Args:
            query_vector: The vector embedding to search for.
            top_k: The number of top results to return.
            filter: An optional metadata filter dictionary to apply.
            similarity_threshold: The minimum similarity score for results.
            **kwargs: Can include:
                - `score_threshold`: Filtering by certainty
                - `rerank`: Dict with {'property': str, 'query': str} for reranking
        
        Returns:
            A list of the most similar results as VectorSearchResult objects.
        """
        collection_obj = await self._get_collection()

        final_similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else self._config.default_similarity_threshold or 0.0
        )

        try:
            weaviate_filter = self._translate_filter(filter) if filter else None
            score_threshold = kwargs.get('score_threshold')
            rerank_config = kwargs.get('rerank')
            
            # Build rerank object if provided
            rerank_obj = None
            if rerank_config:
                rerank_obj = Rerank(
                    prop=rerank_config.get('property', 'content'),
                    query=rerank_config.get('query')
                )

            # Perform dense vector search (Weaviate uses single default vector)
            response = await collection_obj.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=weaviate_filter,
                certainty=score_threshold,
                rerank=rerank_obj,
                return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True),
                include_vector=True
            )

            results = []
            for obj in response.objects:
                certainty = (
                    obj.metadata.certainty
                    if obj.metadata and obj.metadata.certainty is not None
                    else None
                )
                distance = (
                    obj.metadata.distance
                    if obj.metadata and obj.metadata.distance is not None
                    else None
                )
                                
                # Calculate score
                if certainty is not None:
                    score = certainty
                elif distance is not None:
                    score = 1.0 - distance
                else:
                    score = 0.0

                if score >= final_similarity_threshold:
                    vector = self._extract_vector(obj.vector)
                    content = obj.properties.get("content", "")
                    
                    results.append(VectorSearchResult(
                        id=str(obj.uuid),
                        score=score,
                        payload=obj.properties,
                        vector=vector,
                        text=content
                    ))
            
            return results

        except Exception as e:
            raise SearchError(f"An error occurred during dense search: {e}")

    def dense_search_sync(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """Performs a pure vector similarity search (sync)."""
        return self._run_async_from_sync(self.dense_search(query_vector, top_k, filter, similarity_threshold, **kwargs))

    async def full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Performs a full-text (keyword) search using Weaviate's BM25 algorithm.
        
        Args:
            query_text: The text string to search for.
            top_k: The number of top results to return.
            filter: An optional metadata filter to apply before the search.
            similarity_threshold: The minimum similarity score for results.
            **kwargs: Can include:
                - `rerank`: Dict with {'property': str, 'query': str} for reranking
        
        Returns:
            A list of matching results, ordered by BM25 relevance score.
        """
        collection_obj = await self._get_collection()

        final_similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else self._config.default_similarity_threshold or 0.0
        )

        try:
            weaviate_filter = self._translate_filter(filter) if filter else None
            rerank_config = kwargs.get('rerank')
            
            # Build rerank object if provided
            rerank_obj = None
            if rerank_config:
                rerank_obj = Rerank(
                    prop=rerank_config.get('property', 'content'),
                    query=rerank_config.get('query', query_text)  # Default to query_text
                )

            response = await collection_obj.query.bm25(
                query=query_text,
                query_properties=["content"],  # Search in content field
                limit=top_k,
                filters=weaviate_filter,
                rerank=rerank_obj,
                return_metadata=wvc.query.MetadataQuery(score=True),
                include_vector=True
            )

            results = []
            for obj in response.objects:
                score = (
                    obj.metadata.score
                    if obj.metadata and obj.metadata.score is not None
                    else 0.0
                )

                if score >= final_similarity_threshold:
                    vector = self._extract_vector(obj.vector)
                    content = obj.properties.get("content", "")
                    
                    results.append(VectorSearchResult(
                        id=str(obj.uuid),
                        score=score,
                        payload=obj.properties,
                        vector=vector,
                        text=content
                    ))
            
            return results

        except Exception as e:
            raise SearchError(f"An error occurred during full-text search: {e}")

    def full_text_search_sync(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """Performs a full-text search if the provider supports it (sync)."""
        return self._run_async_from_sync(self.full_text_search(query_text, top_k, filter, similarity_threshold, **kwargs))

    async def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Combines dense and sparse search results using Weaviate's native hybrid query.
        
        Args:
            query_vector: The dense vector for the semantic part of the search.
            query_text: The raw text for the keyword/sparse part of the search.
            top_k: The number of final results to return.
            filter: An optional metadata filter.
            alpha: The weight for combining scores (0.0 = pure keyword, 1.0 = pure vector).
            fusion_method: The algorithm to use ('rrf' or 'weighted').
            similarity_threshold: The minimum similarity score for results.
            **kwargs: Can include:
                - `rerank`: Dict with {'property': str, 'query': str} for reranking
        
        Returns:
            A list of VectorSearchResult objects, ordered by the combined hybrid score.
        """
        collection_obj = await self._get_collection()
        
        final_alpha = alpha if alpha is not None else self._config.default_hybrid_alpha or 0.5

        if not (0.0 <= final_alpha <= 1.0):
            raise ConfigurationError(
                f"Hybrid search alpha must be between 0.0 and 1.0, but got {final_alpha}."
            )

        final_similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else self._config.default_similarity_threshold or 0.0
        )

        fusion_type = None
        if fusion_method is not None:
            if fusion_method == "rrf":
                fusion_type = HybridFusion.RANKED
            elif fusion_method == "weighted":
                fusion_type = HybridFusion.RELATIVE_SCORE
            else:
                raise ConfigurationError(
                    f"Unsupported fusion_method '{fusion_method}'. Use 'rrf' or 'weighted'."
                )

        try:
            weaviate_filter = self._translate_filter(filter) if filter else None
            rerank_config = kwargs.get('rerank')
            
            # Build rerank object if provided
            rerank_obj = None
            if rerank_config:
                rerank_obj = Rerank(
                    prop=rerank_config.get('property', 'content'),
                    query=rerank_config.get('query', query_text)  # Default to query_text
                )
            
            # Perform hybrid search (combines dense vector + BM25 keyword search)
            response = await collection_obj.query.hybrid(
                query=query_text,
                vector=query_vector,
                query_properties=["content"],  # BM25 searches in content field
                alpha=final_alpha,
                limit=top_k,
                filters=weaviate_filter,
                fusion_type=fusion_type,
                rerank=rerank_obj,
                return_metadata=wvc.query.MetadataQuery(score=True),
                include_vector=True
            )

            results = []
            for obj in response.objects:
                score = (
                    obj.metadata.score
                    if obj.metadata and obj.metadata.score is not None
                    else 0.0
                )
                
                if score >= final_similarity_threshold:
                    vector = self._extract_vector(obj.vector)
                    content = obj.properties.get("content", "")
                    
                    results.append(VectorSearchResult(
                        id=str(obj.uuid),
                        score=score,
                        payload=obj.properties,
                        vector=vector,
                        text=content
                    ))

            return results
            
        except Exception as e:
            raise SearchError(f"An error occurred during hybrid search: {e}")

    def hybrid_search_sync(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """Combines dense and sparse/keyword search results (sync)."""
        return self._run_async_from_sync(self.hybrid_search(query_vector, query_text, top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs))

    async def delete_by_field(
        self,
        field_name: str,
        field_value: Any
    ) -> bool:
        """
        Delete documents by a specific field value.
        
        Args:
            field_name: The name of the field to filter by.
            field_value: The value to match.
        
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            collection_obj = await self._get_collection()
            
            delete_filter = wvc.query.Filter.by_property(field_name).equal(field_value)
            result = await collection_obj.data.delete_many(where=delete_filter)
            
            info_log(
                f"Deleted {result.successful} documents with {field_name}='{field_value}' "
                f"from collection '{self._config.collection_name}'.",
                context="WeaviateVectorDB"
            )
            return True

        except Exception as e:
            debug_log(
                f"Error deleting documents by {field_name}='{field_value}': {e}",
                context="WeaviateVectorDB"
            )
            return False
    
    def delete_by_document_name(self, document_name: str) -> bool:
        """Delete documents by document_name (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_name(document_name))
    
    async def async_delete_by_document_name(self, document_name: str) -> bool:
        """Delete documents by document_name (async)."""
        return await self.delete_by_field("document_name", document_name)
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete documents by document_id (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_id(document_id))
    
    async def async_delete_by_document_id(self, document_id: str) -> bool:
        """Delete documents by document_id (async)."""
        return await self.delete_by_field("document_id", document_id)
    
    def delete_by_content_id(self, content_id: str) -> bool:
        """Delete documents by content_id (sync)."""
        return self._run_async_from_sync(self.async_delete_by_content_id(content_id))
    
    async def async_delete_by_content_id(self, content_id: str) -> bool:
        """Delete documents by content_id (async)."""
        return await self.delete_by_field("content_id", content_id)

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete documents by metadata filter (sync)."""
        return self._run_async_from_sync(self.async_delete_by_metadata(metadata))
    
    async def async_delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Delete documents by metadata filter (async).
        
        Args:
            metadata: Dictionary of metadata fields to match.
        
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            collection_obj = await self._get_collection()
            
            # Build filter from metadata
            filter_expr = self._translate_filter(metadata)
            if filter_expr is None:
                debug_log(
                    f"No valid filter could be built for metadata: {metadata}",
                    context="WeaviateVectorDB"
                )
                return False

            result = await collection_obj.data.delete_many(where=filter_expr)
            
            info_log(
                f"Deleted {result.successful} documents with metadata '{metadata}' "
                f"from collection '{self._config.collection_name}'.",
                context="WeaviateVectorDB"
            )
            return True

        except Exception as e:
            debug_log(
                f"Error deleting documents by metadata '{metadata}': {e}",
                context="WeaviateVectorDB"
            )
            return False
    
    def document_name_exists(self, document_name: str) -> bool:
        """Check if a document with the given document_name exists (sync)."""
        return self._run_async_from_sync(self.async_document_name_exists(document_name))
    
    async def async_document_name_exists(self, document_name: str) -> bool:
        """Check if a document with the given document_name exists (async)."""
        return await self.field_exists("document_name", document_name)
    
    def document_id_exists(self, document_id: str) -> bool:
        """Check if a document with the given document_id exists (sync)."""
        return self._run_async_from_sync(self.async_document_id_exists(document_id))
    
    async def async_document_id_exists(self, document_id: str) -> bool:
        """Check if a document with the given document_id exists (async)."""
        return await self.field_exists("document_id", document_id)
    
    def content_id_exists(self, content_id: str) -> bool:
        """Check if a document with the given content_id exists (sync)."""
        return self._run_async_from_sync(self.async_content_id_exists(content_id))
    
    async def async_content_id_exists(self, content_id: str) -> bool:
        """Check if a document with the given content_id exists (async)."""
        return await self.field_exists("content_id", content_id)

    async def field_exists(self, field_name: str, field_value: Any) -> bool:
        """
        Check if a document with the given field value exists.
        
        Args:
            field_name: The name of the field to check.
            field_value: The value to match.
        
        Returns:
            True if a document exists, False otherwise.
        """
        try:
            collection_obj = await self._get_collection()
            
            result = await collection_obj.query.fetch_objects(
                limit=1,
                filters=wvc.query.Filter.by_property(field_name).equal(field_value)
            )
            
            return len(result.objects) > 0
            
        except Exception as e:
            debug_log(
                f"Error checking if {field_name}='{field_value}' exists: {e}",
                context="WeaviateVectorDB"
            )
            return False

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """Update the metadata for documents with the given content_id (sync)."""
        return self._run_async_from_sync(self.async_update_metadata(content_id, metadata))
    
    async def async_update_metadata(
        self,
        content_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update the metadata for documents with the given content_id (async).
        
        Args:
            content_id: The content ID to update.
            metadata: The metadata to update/merge.
        
        Returns:
            True if update was successful, False otherwise.
        """
        try:
            collection_obj = await self._get_collection()

            # Query for objects with the given content_id
            query_result = await collection_obj.query.fetch_objects(
                filters=wvc.query.Filter.by_property("content_id").equal(content_id),
                limit=1000  # Get all matching objects
            )

            if not query_result.objects:
                debug_log(
                    f"No documents found with content_id: {content_id}",
                    context="WeaviateVectorDB"
                )
                return False

            # Update each matching object
            updated_count = 0
            for obj in query_result.objects:
                # Get current properties
                current_properties = obj.properties or {}

                # Merge existing metadata with new metadata
                updated_properties = current_properties.copy()

                # Handle nested metadata updates
                if "metadata" in updated_properties and isinstance(
                    updated_properties["metadata"], dict
                ):
                    updated_properties["metadata"].update(metadata)
                else:
                    if "metadata" not in updated_properties:
                        updated_properties["metadata"] = {}
                    updated_properties["metadata"].update(metadata)

                # Update the object
                await collection_obj.data.update(
                    uuid=obj.uuid,
                    properties=updated_properties
                )
                updated_count += 1

            info_log(
                f"Updated metadata for {updated_count} documents with content_id: {content_id}",
                context="WeaviateVectorDB"
            )
            return True

        except Exception as e:
            debug_log(
                f"Error updating metadata for content_id '{content_id}': {e}",
                context="WeaviateVectorDB"
            )
            return False
    
    def optimize(self) -> bool:
        """Optimize the vector database (sync). Weaviate doesn't require explicit optimization."""
        return True
    
    async def async_optimize(self) -> bool:
        """Optimize the vector database (async). Weaviate doesn't require explicit optimization."""
        return True
    
    def get_supported_search_types(self) -> List[str]:
        """Get the supported search types for Weaviate (sync)."""
        supported = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported
    
    async def async_get_supported_search_types(self) -> List[str]:
        """Get the supported search types for Weaviate (async)."""
        return self.get_supported_search_types()

    # ============================================================================
    # Private Helper Methods
    # ============================================================================

    async def _get_collection(self) -> weaviate.collections.Collection:
        """
        Private helper to get the collection object, applying tenancy if configured.
        
        Returns:
            A Weaviate collection object, properly scoped with tenant if applicable.
        
        Raises:
            VectorDBConnectionError: If not connected.
            CollectionDoesNotExistError: If the collection doesn't exist.
        """
        if not self._async_client or not self._is_connected:
            raise VectorDBConnectionError("Client is not connected.")
        
        try:
            collection = self._async_client.collections.get(self._config.collection_name)
        except UnexpectedStatusCodeError as e:
            if e.status_code == 404:
                raise CollectionDoesNotExistError(
                    f"Collection '{self._config.collection_name}' does not exist in Weaviate."
                )
            raise VectorDBError(f"Failed to retrieve collection: {e.message}")
        
        if self._config.namespace:
            return collection.with_tenant(self._config.namespace)
    
        return collection

    def _build_vector_index_config(self, distance_map: Dict) -> Any:
        """
        Build the vector index configuration based on config.
        
        Args:
            distance_map: Mapping of DistanceMetric to Weaviate distance metrics.
        
        Returns:
            A Weaviate vector index configuration object.
        """
        index_conf = self._config.index
        
        if isinstance(index_conf, HNSWIndexConfig):
            hnsw_params = {
                "distance_metric": distance_map[self._config.distance_metric],
                "max_connections": index_conf.m,
                "ef_construction": index_conf.ef_construction
            }
            if index_conf.ef_search is not None:
                hnsw_params["ef"] = index_conf.ef_search
            return wvc.config.Configure.VectorIndex.hnsw(**hnsw_params)
        elif isinstance(index_conf, FlatIndexConfig):
            return wvc.config.Configure.VectorIndex.flat(
                distance_metric=distance_map[self._config.distance_metric]
            )
        else:
            # Default to HNSW
            return wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=distance_map[self._config.distance_metric],
                max_connections=16,
                ef_construction=200
            )

    def _build_vector_config(self, distance_map: Dict) -> Any:
        """
        Build the vector configuration for self-provided dense vectors.
        
        Weaviate only supports dense vectors. Hybrid search combines dense vectors
        with BM25 keyword search (not sparse vectors).
        
        Args:
            distance_map: Mapping of DistanceMetric to Weaviate distance metrics.
        
        Returns:
            A Weaviate vector configuration object for self-provided vectors.
        """
        # Build the vector index config
        vector_index_config = self._build_vector_index_config(distance_map)
        
        # Single dense vector with self-provided vectors (we provide vectors ourselves)
        return wvc.config.Configure.Vectors.self_provided(
            vector_index_config=vector_index_config
        )

    def _build_properties_schema(
        self,
        additional_properties: Optional[List[Dict[str, Any]]] = None
    ) -> List[Any]:
        """
        Build the properties schema for the collection.
        
        This creates properties for:
        - document_name (TEXT, optionally indexed)
        - document_id (TEXT, optionally indexed)
        - content_id (TEXT, always indexed - main ID)
        - content (TEXT, tokenized for BM25, always searchable)
        - metadata (TEXT, JSON serialized, optionally indexed)
        
        indexed_fields can be:
        - Simple list: ["document_name", "document_id"]
        - Advanced list: [{"field": "document_name", "type": "keyword"}, {"field": "age", "type": "integer"}]
        
        Args:
            additional_properties: Additional custom properties from method kwargs.
        
        Returns:
            A list of Weaviate Property objects.
        """
        properties = []
        indexed_fields_config = self._parse_indexed_fields()
        
        # document_name
        field_config = indexed_fields_config.get("document_name", {})
        properties.append(wvc.config.Property(
            name="document_name",
            data_type=self._get_weaviate_datatype(field_config.get("type", "text")),
            tokenization=self._get_weaviate_tokenization(field_config.get("type", "text")),
            skip_vectorization=True,
            index_filterable=field_config.get("indexed", False),
            index_searchable=field_config.get("indexed", False)
        ))
        
        # document_id
        field_config = indexed_fields_config.get("document_id", {})
        properties.append(wvc.config.Property(
            name="document_id",
            data_type=self._get_weaviate_datatype(field_config.get("type", "text")),
            tokenization=self._get_weaviate_tokenization(field_config.get("type", "text")),
            skip_vectorization=True,
            index_filterable=field_config.get("indexed", False),
            index_searchable=field_config.get("indexed", False)
        ))
        
        # content_id (main ID - always indexed)
        properties.append(wvc.config.Property(
            name="content_id",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.WORD,
            skip_vectorization=True,
            index_filterable=True,  # Always index content_id
            index_searchable=True
        ))
        
        # content (required, tokenized for BM25, always searchable)
        field_config = indexed_fields_config.get("content", {})
        properties.append(wvc.config.Property(
            name="content",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.LOWERCASE,
            skip_vectorization=True,
            index_filterable=field_config.get("indexed", False),
            index_searchable=True  # Always searchable for BM25
        ))
        
        # metadata (JSON serialized) - always filterable and searchable for nested property queries
        properties.append(wvc.config.Property(
            name="metadata",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.WORD,
            skip_vectorization=True,
            index_filterable=True,  # Always True to enable filtering on nested JSON properties
            index_searchable=True   # Always True to enable searching on nested JSON properties
        ))
        
        # Add any custom properties from config
        if self._config.properties:
            properties.extend(self._parse_custom_properties(self._config.properties))
        
        # Add additional properties from kwargs
        if additional_properties:
            properties.extend(self._parse_custom_properties(additional_properties))
        
        return properties
    
    def _parse_indexed_fields(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse indexed_fields into a standardized format.
        
        Supports two formats:
        1. Simple: ["document_name", "document_id"]
        2. Advanced: [{"field": "document_name", "type": "keyword"}, {"field": "age", "type": "integer"}]
        
        Returns:
            Dict mapping field_name to config: {"field_name": {"indexed": True, "type": "keyword"}}
        """
        if not self._config.indexed_fields:
            return {}
        
        result = {}
        for item in self._config.indexed_fields:
            if isinstance(item, str):
                # Simple format: just field name
                result[item] = {"indexed": True, "type": "text"}
            elif isinstance(item, dict):
                # Advanced format: {"field": "name", "type": "keyword"}
                field_name = item.get("field")
                if field_name:
                    result[field_name] = {
                        "indexed": True,
                        "type": item.get("type", "text")
                    }
        
        return result
    
    def _get_weaviate_datatype(self, field_type: str) -> Any:
        """
        Convert field type string to Weaviate DataType.
        
        Args:
            field_type: One of 'text', 'keyword', 'integer', 'float', 'boolean', 'geo'
        
        Returns:
            Weaviate DataType enum value
        """
        datatype_map = {
            'keyword': wvc.config.DataType.TEXT,
            'text': wvc.config.DataType.TEXT,
            'integer': wvc.config.DataType.INT,
            'int': wvc.config.DataType.INT,
            'float': wvc.config.DataType.NUMBER,
            'number': wvc.config.DataType.NUMBER,
            'boolean': wvc.config.DataType.BOOL,
            'bool': wvc.config.DataType.BOOL,
            'geo': wvc.config.DataType.GEO_COORDINATES
        }
        return datatype_map.get(field_type.lower(), wvc.config.DataType.TEXT)
    
    def _get_weaviate_tokenization(self, field_type: str) -> Any:
        """
        Get appropriate tokenization for field type.
        
        Args:
            field_type: Field type string
        
        Returns:
            Weaviate Tokenization enum value or None for non-text types
        """
        # Only TEXT fields need tokenization
        if field_type.lower() in ['text', 'keyword']:
            if field_type.lower() == 'keyword':
                return wvc.config.Tokenization.WORD
            else:
                return wvc.config.Tokenization.WHITESPACE
        return None
    
    def _parse_custom_properties(self, props: List[Dict[str, Any]]) -> List[Any]:
        """Parse custom properties from dict format to Weaviate Property objects."""
        parsed_properties = []
        datatype_map = {
            'keyword': wvc.config.DataType.TEXT,
            'text': wvc.config.DataType.TEXT,
            'integer': wvc.config.DataType.INT,
            'float': wvc.config.DataType.NUMBER,
            'boolean': wvc.config.DataType.BOOL,
            'geo': wvc.config.DataType.GEO_COORDINATES
        }
        tokenization_map = {
            'keyword': wvc.config.Tokenization.WORD,
            'text': wvc.config.Tokenization.WHITESPACE
        }
        
        for prop in props:
            prop_name = prop.get('name')
            # Skip if it's a standard field
            if prop_name in ['document_name', 'document_id', 'content_id', 'content', 'metadata']:
                continue
            
            parsed_properties.append(wvc.config.Property(
                name=prop_name,
                data_type=datatype_map.get(
                    prop.get('dataType', 'text'),
                    wvc.config.DataType.TEXT
                ),
                tokenization=tokenization_map.get(
                    prop.get('dataType', 'text'),
                    wvc.config.Tokenization.WORD
                ),
                skip_vectorization=True,
                index_filterable=prop.get('indexed', False),
                index_searchable=prop.get('searchable', False)
            ))
        
        return parsed_properties
    
    def _build_inverted_index_config(self, override: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Build inverted index configuration for BM25 tuning."""
        config_dict = override or self._config.inverted_index_config
        if not config_dict:
            return None
        
        # Parse inverted index config
        # Example: {'bm25': {'k1': 1.2, 'b': 0.75}}
        if 'bm25' in config_dict:
            bm25_params = config_dict['bm25']
            return wvc.config.Configure.inverted_index(
                bm25_k1=bm25_params.get('k1', 1.2),
                bm25_b=bm25_params.get('b', 0.75)
            )
        
        return None
    
    def _build_multi_tenancy_config(self, override: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Build multi-tenancy configuration."""
        if override:
            enabled = override.get('enabled', False)
        else:
            enabled = self._config.multi_tenancy_enabled
        
        if not enabled:
            return None
        
        return wvc.config.Configure.multi_tenancy(enabled=True)
    
    def _build_replication_config(self, override: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Build replication configuration."""
        config_dict = override or self._config.replication_config
        if not config_dict:
            return None
        
        # Example: {'factor': 3, 'asyncEnabled': True}
        return wvc.config.Configure.replication(
            factor=config_dict.get('factor', 1),
            async_enabled=config_dict.get('asyncEnabled', False)
        )
    
    def _build_sharding_config(self, override: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Build sharding configuration."""
        config_dict = override or self._config.sharding_config
        if not config_dict:
            return None
        
        # Example: {'virtualPerPhysical': 128, 'desiredCount': 2}
        return wvc.config.Configure.sharding(
            virtual_per_physical=config_dict.get('virtualPerPhysical', 128),
            desired_count=config_dict.get('desiredCount', 1),
            desired_virtual_count=config_dict.get('desiredVirtualCount')
        )
    
    def _build_generative_config(self, override: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Build generative AI configuration (e.g., OpenAI, Cohere)."""
        config_dict = override or self._config.generative_config
        if not config_dict:
            return None
        
        # Dynamically import based on provider
        provider = config_dict.get('provider', '').lower()
        
        try:
            if provider == 'openai':
                return wvc.config.Configure.Generative.openai(
                    model=config_dict.get('model', 'gpt-3.5-turbo')
                )
            elif provider == 'cohere':
                return wvc.config.Configure.Generative.cohere(
                    model=config_dict.get('model')
                )
            elif provider == 'anthropic':
                return wvc.config.Configure.Generative.anthropic(
                    model=config_dict.get('model', 'claude-2')
                )
            # Add more providers as needed
        except AttributeError:
            # Provider not available in this Weaviate version
            debug_log(
                f"Generative provider '{provider}' not available.",
                context="WeaviateVectorDB"
            )
        
        return None
    
    def _build_reranker_config(self, override: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Build reranker configuration (e.g., Cohere, Transformers)."""
        config_dict = override or self._config.reranker_config
        if not config_dict:
            return None
        
        provider = config_dict.get('provider', '').lower()
        
        try:
            if provider == 'cohere':
                return wvc.config.Configure.Reranker.cohere(
                    model=config_dict.get('model')
                )
            elif provider == 'transformers':
                return wvc.config.Configure.Reranker.transformers()
            # Add more providers as needed
        except AttributeError:
            debug_log(
                f"Reranker provider '{provider}' not available.",
                context="WeaviateVectorDB"
            )
        
        return None
    
    def _build_references(self, override: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        """Build cross-references to other collections."""
        refs = override or self._config.references
        if not refs:
            return None
        
        # Parse reference configurations
        # Example: [{'name': 'hasAuthor', 'target': 'Author'}]
        parsed_refs = []
        for ref in refs:
            parsed_refs.append(
                wvc.config.ReferenceProperty(
                    name=ref.get('name'),
                    target_collection=ref.get('target')
                )
            )
        
        return parsed_refs if parsed_refs else None

    def _process_payload(
        self,
        payload: Dict[str, Any],
        content: str,
        document_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a payload into the standard Weaviate property format.
        
        This method:
        1. Extracts document_name, document_id, content_id from payload
        2. Generates content_id if not provided
        3. Merges default_metadata from config
        4. Merges extra_metadata from method call
        5. Serializes metadata to JSON string
        
        Args:
            payload: The original payload dict.
            content: The content/chunk text.
            document_id: The document ID (from ids parameter).
            extra_metadata: Additional metadata to merge.
        
        Returns:
            A properly formatted properties dict for Weaviate.
        """
        properties = {}
        
        # Extract or generate content_id (main ID)
        content_id = payload.get('content_id')
        if not content_id and self._config.auto_generate_content_id:
            # Generate from content hash
            content_id = md5(content.encode()).hexdigest()
        elif not content_id:
            # Fall back to document_id
            content_id = document_id
        
        properties['content_id'] = content_id
        
        # Extract document_name
        properties['document_name'] = payload.get('document_name', '')
        
        # Extract document_id (prefer from payload, fall back to parameter)
        properties['document_id'] = payload.get('document_id', document_id)
        
        # Set content (required)
        properties['content'] = content
        
        # Build metadata
        metadata = {}
        
        # Start with default_metadata from config
        if self._config.default_metadata:
            metadata.update(self._config.default_metadata)
        
        # Merge metadata from payload
        if 'metadata' in payload and isinstance(payload['metadata'], dict):
            metadata.update(payload['metadata'])
        
        # Merge extra_metadata from method call
        if extra_metadata:
            metadata.update(extra_metadata)
        
        # Also include any other fields from payload that aren't the standard ones
        standard_fields = {'document_name', 'document_id', 'content_id', 'metadata', 'content'}
        for key, value in payload.items():
            if key not in standard_fields:
                metadata[key] = value
        
        # Serialize metadata to JSON string
        properties['metadata'] = json.dumps(metadata) if metadata else "{}"
        
        return properties

    def _extract_vector(self, vector_obj: Any) -> Optional[List[float]]:
        """
        Extract the dense vector from Weaviate's vector object.
        
        Weaviate returns vectors either as a dict with 'default' key or as a list directly.
        
        Args:
            vector_obj: The vector object from Weaviate response.
        
        Returns:
            The dense vector as a list of floats, or None if not available.
        """
        if vector_obj is None:
            return None
        
        if isinstance(vector_obj, dict):
            # Vector stored in 'default' key (standard format)
            return vector_obj.get('default')
        else:
            # Direct vector list
            return vector_obj

    def _translate_filter(self, filter_dict: Dict[str, Any]) -> wvc.query.Filter:
        """
        Recursively translates a framework-standard filter dictionary into a Weaviate Filter object.
        
        Supports:
        - Logical operators: "and", "or"
        - Comparison operators: "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"
        - Direct field equality: {"field": "value"}
        
        Fields that are not standard properties (document_name, document_id, content_id, content)
        are automatically searched within the JSON-serialized metadata field.
        
        Args:
            filter_dict: A dictionary representing the filter logic.
        
        Returns:
            A Weaviate Filter object ready to be used in a query.
        
        Raises:
            SearchError: If an unknown operator or invalid filter structure is provided.
        """
        # Standard fields that exist as properties in the schema
        standard_fields = {'document_name', 'document_id', 'content_id', 'content', 'metadata'}
        
        logical_ops = {
            "and": wvc.query.Filter.all_of,
            "or": wvc.query.Filter.any_of,
        }

        comparison_ops = {
            "$eq": lambda p, v: p.equal(v),
            "$ne": lambda p, v: p.not_equal(v),
            "$gt": lambda p, v: p.greater_than(v),
            "$gte": lambda p, v: p.greater_or_equal(v),
            "$lt": lambda p, v: p.less_than(v),
            "$lte": lambda p, v: p.less_or_equal(v),
            "$in": lambda p, v: p.contains_any(v),
        }

        filters = []
        for key, value in filter_dict.items():
            if key in logical_ops:
                sub_filters = [self._translate_filter(sub_filter) for sub_filter in value]
                return logical_ops[key](sub_filters)
            
            # Determine if this is a standard field or nested in metadata
            if key in standard_fields:
                # Direct property filter
                prop_filter = wvc.query.Filter.by_property(key)
                if isinstance(value, dict):
                    if len(value) != 1:
                        raise SearchError(
                            f"Field filter for '{key}' must have exactly one operator."
                        )
                    
                    op, val = list(value.items())[0]
                    if op in comparison_ops:
                        filters.append(comparison_ops[op](prop_filter, val))
                    else:
                        raise SearchError(
                            f"Unsupported filter operator '{op}' for field '{key}'."
                        )
                else:
                    filters.append(prop_filter.equal(value))
            else:
                # Non-standard field - search in JSON metadata using 'like'
                # Build JSON pattern to match
                import json as json_module
                pattern = f'"{key}": "{value}"' if isinstance(value, str) else f'"{key}": {json_module.dumps(value)}'
                metadata_filter = wvc.query.Filter.by_property("metadata").like(f"*{pattern}*")
                filters.append(metadata_filter)

        if not filters:
            raise SearchError("Filter dictionary cannot be empty.")
        
        return wvc.query.Filter.all_of(filters) if len(filters) > 1 else filters[0]
