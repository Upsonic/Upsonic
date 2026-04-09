from __future__ import annotations

import uuid
import json
import os
from hashlib import md5
from typing import Any, Dict, List, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import weaviate
    import weaviate.classes as wvc
    from weaviate.exceptions import (
        WeaviateConnectionError,
        UnexpectedStatusCodeError,
    )
    from weaviate.util import generate_uuid5
    from weaviate.classes.query import HybridFusion, Rerank

try:
    import weaviate
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
    - Explicit chunk_id, chunk_content_hash, document_id, doc_content_hash property storage
    - Configurable field indexing for fast filtering
    - Default metadata support
    - Multi-tenancy support via namespaces
    - Replication and sharding configuration
    - Optional generative AI and reranker modules
    - Proper error handling with custom exceptions
    
    Note: Weaviate does NOT support sparse vectors. Hybrid search combines
    dense vector similarity with BM25 keyword search (inverted index).
    """

    _STANDARD_FIELDS: frozenset = frozenset({
        'document_name', 'document_id', 'chunk_id',
        'metadata', 'content', 'doc_content_hash', 'chunk_content_hash',
        'knowledge_base_id',
    })

    def __init__(self, config: Union[WeaviateConfig, Dict[str, Any]]):
        if not _WEAVIATE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="weaviate-client",
                install_command='pip install "upsonic[weaviate]"',
                feature_name="Weaviate vector database provider"
            )
        
        if isinstance(config, dict):
            config = WeaviateConfig.from_dict(config)
        
        super().__init__(config)
        
        info_log(
            f"WeaviateProvider initialized for collection '{self._config.collection_name}' "
            f"in '{self._config.connection.mode.value}' mode.",
            context="WeaviateVectorDB"
        )

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
    

    def _build_api_headers(self) -> Dict[str, str]:
        """
        Build API headers for generative AI and reranker modules.
        
        Supports all Weaviate API-based model provider integrations:
        https://docs.weaviate.io/weaviate/model-providers
        
        Returns:
            A dictionary of headers with API keys for configured providers.
        """
        headers: Dict[str, str] = {}
        
        provider_header_map = {
            'anthropic': 'X-Anthropic-Api-Key',
            'anyscale': 'X-Anyscale-Api-Key',
            'aws': 'X-AWS-Access-Key',
            'cohere': 'X-Cohere-Api-Key',
            'contextualai': 'X-Contextual-Api-Key',
            'databricks': 'X-Databricks-Token',
            'friendliai': 'X-FriendliAI-Api-Key',
            'google': 'X-Google-Api-Key',
            'huggingface': 'X-HuggingFace-Api-Key',
            'jinaai': 'X-JinaAI-Api-Key',
            'jina': 'X-JinaAI-Api-Key',
            'mistral': 'X-Mistral-Api-Key',
            'nvidia': 'X-NVIDIA-Api-Key',
            'octoai': 'X-OctoAI-Api-Key',
            'openai': 'X-OpenAI-Api-Key',
            'azure': 'X-Azure-Api-Key',
            'azure_openai': 'X-Azure-Api-Key',
            'voyageai': 'X-VoyageAI-Api-Key',
            'voyage': 'X-VoyageAI-Api-Key',
            'xai': 'X-xAI-Api-Key',
        }
        
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
        
        providers_needed: set[str] = set()
        
        if self._config.generative_config:
            provider = self._config.generative_config.get('provider', '').lower()
            if provider:
                providers_needed.add(provider)
        
        if self._config.reranker_config:
            provider = self._config.reranker_config.get('provider', '').lower()
            if provider:
                providers_needed.add(provider)
        
        for provider in providers_needed:
            if provider not in provider_header_map:
                continue
            
            header_key = provider_header_map[provider]
            api_key: Optional[str] = None
            
            if self._config.api_keys and provider in self._config.api_keys:
                api_key = self._config.api_keys[provider]
            
            if not api_key and provider in env_var_map:
                api_key = os.getenv(env_var_map[provider])
            
            if api_key:
                headers[header_key] = api_key
                debug_log(
                    f"Added API key header for provider '{provider}': {header_key}",
                    context="WeaviateVectorDB"
                )
        
        return headers



    async def _create_async_client(self) -> None:
        """
        Creates the async Weaviate client based on the connection configuration.
        Does not connect - only instantiates the client object.

        Raises:
            ConfigurationError: If the connection configuration is invalid.
            VectorDBConnectionError: If client instantiation fails.
        """
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
                self.client = weaviate.use_async_with_weaviate_cloud(
                    cluster_url=self._config.connection.host,
                    auth_credentials=auth_credentials,
                    headers=additional_headers if additional_headers else None,
                    additional_config=additional_config,
                    skip_init_checks=True
                )

            elif self._config.connection.mode == Mode.LOCAL:
                if not self._config.connection.host or not self._config.connection.port:
                    raise ConfigurationError("Local mode requires 'host' and 'port'.")

                self.client = weaviate.use_async_with_local(
                    host=self._config.connection.host,
                    port=self._config.connection.port
                )

            elif self._config.connection.mode in (Mode.EMBEDDED, Mode.IN_MEMORY):
                persistence_path = (
                    self._config.connection.db_path
                    if self._config.connection.mode == Mode.EMBEDDED
                    else None
                )

                self.client = weaviate.use_async_with_embedded(
                    persistence_data_path=persistence_path
                )

            else:
                raise ConfigurationError(
                    f"Unsupported Weaviate mode: {self._config.connection.mode.value}"
                )

        except (ConfigurationError, VectorDBConnectionError):
            raise
        except Exception as e:
            raise VectorDBConnectionError(
                f"Failed to create Weaviate async client: {e}"
            )

    async def aget_client(self) -> Any:
        """
        Gets or creates the async Weaviate client, ensuring it is connected and ready.

        Follows a singleton pattern: reuses the existing client if available,
        creates a new one if needed, reconnects if disconnected, and verifies readiness.

        Returns:
            A connected and ready WeaviateAsyncClient instance.

        Raises:
            VectorDBConnectionError: If the client cannot be created, connected, or is not ready.
        """
        if self.client is None:
            debug_log(
                f"Creating async client for '{self._config.connection.mode.value}' mode...",
                context="WeaviateVectorDB"
            )
            await self._create_async_client()

        if not self.client.is_connected():  # type: ignore[union-attr]
            try:
                await self.client.connect()  # type: ignore[union-attr]
            except Exception as e:
                self.client = None
                self._is_connected = False
                raise VectorDBConnectionError(f"Failed to connect to Weaviate: {e}")

        if not await self.client.is_ready():  # type: ignore[union-attr]
            self._is_connected = False
            raise VectorDBConnectionError("Weaviate async client is not ready after connection.")

        self._is_connected = True
        return self.client  # type: ignore[return-value]

    def get_client(self) -> Any:
        """
        Gets or creates the async Weaviate client, ensuring it is connected and ready (sync).

        Returns:
            A connected and ready WeaviateAsyncClient instance.

        Raises:
            VectorDBConnectionError: If the client cannot be created, connected, or is not ready.
        """
        return self._run_async_from_sync(self.aget_client())

    async def ais_client_connected(self) -> bool:
        """
        Checks whether the Weaviate client exists, is connected, and is ready.
        Does not create or reconnect - purely a read-only status check.

        Returns:
            True if the client is connected and ready, False otherwise.
        """
        if self.client is None:
            return False

        try:
            if not self.client.is_connected():
                return False
            return await self.client.is_ready()
        except Exception:
            return False

    def is_client_connected(self) -> bool:
        """
        Checks whether the Weaviate client exists, is connected, and is ready (sync).

        Returns:
            True if the client is connected and ready, False otherwise.
        """
        return self._run_async_from_sync(self.ais_client_connected())



    async def aconnect(self) -> None:
        """
        Establishes an async connection to the Weaviate vector database instance.

        Delegates to aget_client() which handles client creation, connection,
        and readiness verification. This method is idempotent.

        Raises:
            VectorDBConnectionError: If the connection fails for any reason.
        """
        if await self.ais_client_connected():
            info_log("Already connected to Weaviate.", context="WeaviateVectorDB")
            return

        debug_log(
            f"Attempting to connect to Weaviate in '{self._config.connection.mode.value}' mode...",
            context="WeaviateVectorDB"
        )

        try:
            await self.aget_client()
            info_log(
                "Successfully connected to Weaviate and health check passed.",
                context="WeaviateVectorDB"
            )
        except (VectorDBConnectionError, ConfigurationError):
            self.client = None
            self._is_connected = False
            raise
        except Exception as e:
            self.client = None
            self._is_connected = False
            raise VectorDBConnectionError(
                f"An unexpected error occurred during connection: {e}"
            )

    async def adisconnect(self) -> None:
        """
        Gracefully terminates the connection to the Weaviate database.
        
        This method is idempotent; calling it on an already disconnected
        provider will not raise an error.
        """
        if self.client and self._is_connected:
            try:
                await self.client.close()
                self._is_connected = False
                self.client = None
                info_log("Successfully disconnected from Weaviate.", context="WeaviateVectorDB")
            except Exception as e:
                self._is_connected = False
                self.client = None
                debug_log(
                    f"An error occurred during disconnection, but status is now 'disconnected'. Error: {e}",
                    context="WeaviateVectorDB"
                )
        else:
            debug_log("Already disconnected. No action taken.", context="WeaviateVectorDB")

    async def ais_ready(self) -> bool:
        """
        Performs a health check to ensure the Weaviate instance is responsive.
        
        Returns:
            True if the client is connected and the database is responsive, False otherwise.
        """
        if not self.client or not self._is_connected:
            return False
        
        try:
            return await self.client.is_ready()
        except WeaviateConnectionError:
            self._is_connected = False
            return False


    async def acreate_collection(self) -> None:
        """
        Creates the collection in Weaviate with comprehensive configuration.
        
        This method creates a collection with:
        - Proper vector configuration (dense and optionally sparse)
        - Metadata properties with optional indexing
        - Multi-tenancy support if configured
        - Replication, sharding, inverted index configuration
        - Optional generative and reranker modules
        
        Raises:
            VectorDBConnectionError: If not connected to the database.
            VectorDBError: If the collection creation fails.
        """
        client = await self.aget_client()

        collection_name = self._config.collection_name

        if await self.acollection_exists():
            if self._config.recreate_if_exists:
                info_log(
                    f"Collection '{collection_name}' already exists. "
                    f"Deleting and recreating as requested.",
                    context="WeaviateVectorDB"
                )
                await self.adelete_collection()
            else:
                info_log(
                    f"Collection '{collection_name}' already exists and "
                    f"'recreate_if_exists' is False. No action taken.",
                    context="WeaviateVectorDB"
                )
                return

        try:
            distance_map = {
                DistanceMetric.COSINE: wvc.config.VectorDistances.COSINE,
                DistanceMetric.DOT_PRODUCT: wvc.config.VectorDistances.DOT,
                DistanceMetric.EUCLIDEAN: wvc.config.VectorDistances.L2_SQUARED,
            }
            
            description = self._config.description
            vector_config = self._build_vector_config(distance_map)
            properties = self._build_properties_schema(
                additional_properties=self._config.properties
            )
            inverted_index_config = self._build_inverted_index_config(
                self._config.inverted_index_config
            )
            multi_tenancy_config = self._build_multi_tenancy_config(
                None
            )
            replication_config = self._build_replication_config(
                self._config.replication_config
            )
            sharding_config = self._build_sharding_config(
                self._config.sharding_config
            )
            generative_config = self._build_generative_config(
                self._config.generative_config
            )
            reranker_config = self._build_reranker_config(
                self._config.reranker_config
            )
            references = self._build_references(
                self._config.references
            )
            
            await client.collections.create(
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

            if self._config.namespace and self._config.multi_tenancy_enabled:
                debug_log(
                    f"Creating tenant: '{self._config.namespace}'...",
                    context="WeaviateVectorDB"
                )
                collection = client.collections.get(collection_name)
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

    async def adelete_collection(self) -> None:
        """
        Permanently deletes the collection specified in the config from Weaviate.

        Verifies both client readiness and collection existence before attempting deletion.

        Raises:
            VectorDBConnectionError: If the client cannot be connected or is not ready.
            CollectionDoesNotExistError: If the collection does not exist.
            VectorDBError: For other unexpected API or operational errors.
        """
        client = await self.aget_client()
        collection_name = self._config.collection_name

        if not await client.collections.exists(collection_name):
            raise CollectionDoesNotExistError(
                f"Collection '{collection_name}' could not be deleted because it does not exist."
            )

        try:
            await client.collections.delete(collection_name)
            info_log(
                f"Successfully deleted collection '{collection_name}'.",
                context="WeaviateVectorDB"
            )
        except UnexpectedStatusCodeError as e:
            raise VectorDBError(
                f"API error while deleting collection '{collection_name}': {e.message}"
            )
        except Exception as e:
            raise VectorDBError(
                f"An unexpected error occurred during collection deletion: {e}"
            )

    async def acollection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists in Weaviate.

        Uses aget_client() to ensure the client is connected and ready
        before performing the existence check.

        Returns:
            True if the collection exists, False otherwise.

        Raises:
            VectorDBConnectionError: If the client cannot be connected or is not ready.
        """
        client = await self.aget_client()
        return await client.collections.exists(self._config.collection_name)

    # ============================================================================
    # Data Operations
    # ============================================================================

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
        Adds new data or updates existing data in the collection.

        Uses chunk_content_hash to decide between replace (identical chunk
        content already exists) and insert (new content).
        """
        if (vectors is None or len(vectors) == 0) and (sparse_vectors is None or len(sparse_vectors) == 0):
            info_log("Nothing to upsert: no dense or sparse vectors provided.", context="WeaviateVectorDB")
            return

        n: int = len(vectors) if vectors else len(sparse_vectors)  # type: ignore[arg-type]

        _per_item = {
            "vectors": vectors,
            "payloads": payloads,
            "ids": ids,
            "chunks": chunks,
            "document_ids": document_ids,
            "document_names": document_names,
            "doc_content_hashes": doc_content_hashes,
            "chunk_content_hashes": chunk_content_hashes,
            "sparse_vectors": sparse_vectors,
            "knowledge_base_ids": knowledge_base_ids,
        }
        for _name, _arr in _per_item.items():
            if _arr is not None and len(_arr) != n:
                raise UpsertError(
                    f"Length mismatch in upsert: '{_name}' has length {len(_arr)}, expected {n}."
                )

        if sparse_vectors is not None:
            debug_log(
                "Warning: Weaviate does not support sparse vectors. sparse_vectors parameter is ignored.",
                context="WeaviateVectorDB"
            )

        collection_obj = await self._get_collection()

        try:
            info_log(
                f"Starting upsert of {n} objects...",
                context="WeaviateVectorDB"
            )

            extra_metadata: Dict[str, Any] = metadata or {}
            _warned_standard_keys: set = set()

            for i in range(n):
                payload: Dict[str, Any] = payloads[i] if payloads and i < len(payloads) else {}
                content: str = chunks[i] if chunks and i < len(chunks) else ""
                chunk_id_str: str = str(ids[i]) if ids and i < len(ids) else str(uuid.uuid4())
                doc_id: str = document_ids[i] if document_ids and i < len(document_ids) else ""
                doc_hash: str = doc_content_hashes[i] if doc_content_hashes and i < len(doc_content_hashes) else ""
                doc_name: str = document_names[i] if document_names and i < len(document_names) else ""
                kbi: Optional[str] = knowledge_base_ids[i] if knowledge_base_ids and i < len(knowledge_base_ids) else None

                chunk_hash: str
                if chunk_content_hashes and i < len(chunk_content_hashes):
                    chunk_hash = chunk_content_hashes[i]
                else:
                    chunk_hash = md5(content.encode("utf-8")).hexdigest()

                properties = self._process_payload(
                    payload=payload,
                    content=content,
                    chunk_id=chunk_id_str,
                    chunk_content_hash=chunk_hash,
                    document_id=doc_id,
                    doc_content_hash=doc_hash,
                    document_name=doc_name,
                    knowledge_base_id=kbi,
                    extra_metadata=extra_metadata,
                    _warned_standard_keys=_warned_standard_keys,
                )

                try:
                    object_uuid = uuid.UUID(chunk_id_str)
                except ValueError:
                    object_uuid = generate_uuid5(
                        identifier=chunk_id_str,
                        namespace=self._config.collection_name,
                    )

                vector_val: Optional[List[float]] = vectors[i] if vectors and i < len(vectors) else None

                if await self.afield_exists("chunk_content_hash", chunk_hash):
                    await collection_obj.data.replace(
                        uuid=object_uuid,
                        properties=properties,
                        vector=vector_val,
                    )
                    debug_log(
                        f"Replaced existing object with chunk_content_hash '{chunk_hash[:12]}...'.",
                        context="WeaviateVectorDB",
                    )
                else:
                    await collection_obj.data.insert(
                        properties=properties,
                        vector=vector_val,
                        uuid=object_uuid,
                    )

            info_log(
                f"Successfully upserted {n} objects.",
                context="WeaviateVectorDB",
            )

        except UpsertError:
            raise
        except Exception as e:
            raise UpsertError(
                f"Failed to upsert data to Weaviate collection '{self._config.collection_name}': {e}"
            )

    async def adelete(self, ids: List[Union[str, int]]) -> None:
        """
        Removes data from the collection by their unique identifiers.
        
        Args:
            ids: A list of specific IDs to remove.
        
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

        uuids_to_delete: List[uuid.UUID] = []
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
            existing_uuids: List[uuid.UUID] = []
            for uid in uuids_to_delete:
                if await collection_obj.data.exists(uid):
                    existing_uuids.append(uid)
                else:
                    debug_log(
                        f"Object with UUID '{uid}' does not exist, skipping deletion.",
                        context="WeaviateVectorDB"
                    )

            if not existing_uuids:
                info_log(
                    "No matching objects found to delete. No action taken.",
                    context="WeaviateVectorDB"
                )
                return

            delete_filter = wvc.query.Filter.by_id().contains_any(existing_uuids)
            result = await collection_obj.data.delete_many(where=delete_filter)

            if result.failed > 0:
                raise VectorDBError(
                    f"Deletion partially failed. Successful: {result.successful}, "
                    f"Failed: {result.failed}. Check Weaviate logs for details."
                )

            info_log(
                f"Successfully processed deletion request for {len(ids)} IDs. "
                f"Existed: {len(existing_uuids)}, Deleted: {result.successful}.",
                context="WeaviateVectorDB"
            )

        except VectorDBError:
            raise
        except Exception as e:
            raise VectorDBError(f"An error occurred during deletion: {e}")

    async def afetch(
        self,
        ids: List[Union[str, int]],
    ) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs.
        
        Args:
            ids: A list of IDs to retrieve the full records for.
        
        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        if not ids:
            return []
            
        collection_obj = await self._get_collection()

        uuids_to_fetch: List[uuid.UUID] = []
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
            
            results: List[VectorSearchResult] = []
            for obj in response.objects:
                vector = self._extract_vector(obj.vector)
                content = obj.properties.get("content", "")
                
                results.append(VectorSearchResult(
                    id=str(obj.uuid),
                    score=1.0,
                    payload=self._hydrate_payload(obj.properties),
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
            apply_reranking: Whether to apply reranking if a reranker is configured.
            sparse_query_vector: Sparse query vector (ignored by Weaviate).
        
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
            return await self.adense_search(
                query_vector=query_vector,
                top_k=final_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
            )
        
        elif is_full_text:
            if self._config.full_text_search_enabled is False:
                raise ConfigurationError(
                    "Full-text search is disabled by the current configuration."
                )
            return await self.afull_text_search(
                query_text=query_text,
                top_k=final_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )

        elif is_hybrid:
            if self._config.hybrid_search_enabled is False:
                raise ConfigurationError(
                    "Hybrid search is disabled by the current configuration."
                )
            final_alpha = alpha if alpha is not None else self._config.default_hybrid_alpha or 0.5
            return await self.ahybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=final_top_k,
                filter=filter,
                alpha=final_alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )
        else:
            raise ConfigurationError(
                "Search requires at least one of 'query_vector' or 'query_text'."
            )

    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        """
        Performs a pure vector similarity search using Weaviate's `near_vector` query.
        
        Args:
            query_vector: The vector embedding to search for.
            top_k: The number of top results to return.
            filter: An optional metadata filter dictionary to apply.
            similarity_threshold: The minimum similarity score for results.
            apply_reranking: Whether to apply reranking if a reranker is configured.
        
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

            rerank_obj = None
            if apply_reranking and self._config.reranker_config is not None:
                rerank_obj = Rerank(
                    prop=self._config.rerank_property,
                    query=None,
                )

            response = await collection_obj.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=weaviate_filter,
                certainty=similarity_threshold,
                rerank=rerank_obj,
                return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True),
                include_vector=True
            )

            results: List[VectorSearchResult] = []
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
                        payload=self._hydrate_payload(obj.properties),
                        vector=vector,
                        text=content
                    ))
            
            return results

        except Exception as e:
            raise SearchError(f"An error occurred during dense search: {e}")

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
        Performs a full-text (keyword) search using Weaviate's BM25 algorithm.
        
        Args:
            query_text: The text string to search for.
            top_k: The number of top results to return.
            filter: An optional metadata filter to apply before the search.
            similarity_threshold: The minimum similarity score for results.
            apply_reranking: Whether to apply reranking if a reranker is configured.
            sparse_query_vector: Sparse query vector (ignored by Weaviate).

        Returns:
            A list of matching results, ordered by BM25 relevance score.
        """
        _ = sparse_query_vector  # accepted for API parity; Weaviate doesn't use sparse vectors
        collection_obj = await self._get_collection()

        final_similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else self._config.default_similarity_threshold or 0.0
        )

        try:
            weaviate_filter = self._translate_filter(filter) if filter else None

            rerank_obj = None
            if apply_reranking and self._config.reranker_config is not None:
                rerank_obj = Rerank(
                    prop=self._config.rerank_property,
                    query=query_text,
                )

            response = await collection_obj.query.bm25(
                query=query_text,
                query_properties=["content"],
                limit=top_k,
                filters=weaviate_filter,
                rerank=rerank_obj,
                return_metadata=wvc.query.MetadataQuery(score=True),
                include_vector=True
            )

            results: List[VectorSearchResult] = []
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
                        payload=self._hydrate_payload(obj.properties),
                        vector=vector,
                        text=content
                    ))
            
            return results

        except Exception as e:
            raise SearchError(f"An error occurred during full-text search: {e}")

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
        Combines dense and sparse search results using Weaviate's native hybrid query.
        
        Args:
            query_vector: The dense vector for the semantic part of the search.
            query_text: The raw text for the keyword/sparse part of the search.
            top_k: The number of final results to return.
            filter: An optional metadata filter.
            alpha: The weight for combining scores (0.0 = pure keyword, 1.0 = pure vector).
            fusion_method: The algorithm to use ('rrf' or 'weighted').
            similarity_threshold: The minimum similarity score for results.
            apply_reranking: Whether to apply reranking if a reranker is configured.
            sparse_query_vector: Sparse query vector (ignored by Weaviate).

        Returns:
            A list of VectorSearchResult objects, ordered by the combined hybrid score.
        """
        _ = sparse_query_vector  # accepted for API parity; Weaviate uses BM25, not sparse vectors
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

            rerank_obj = None
            if apply_reranking and self._config.reranker_config is not None:
                rerank_obj = Rerank(
                    prop=self._config.rerank_property,
                    query=query_text,
                )
            
            response = await collection_obj.query.hybrid(
                query=query_text,
                vector=query_vector,
                query_properties=["content"],
                alpha=final_alpha,
                limit=top_k,
                filters=weaviate_filter,
                fusion_type=fusion_type,
                rerank=rerank_obj,
                return_metadata=wvc.query.MetadataQuery(score=True),
                include_vector=True
            )

            results: List[VectorSearchResult] = []
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
                        payload=self._hydrate_payload(obj.properties),
                        vector=vector,
                        text=content
                    ))

            return results
            
        except Exception as e:
            raise SearchError(f"An error occurred during hybrid search: {e}")

    async def adelete_by_field(
        self,
        field_name: str,
        field_value: Any
    ) -> bool:
        """
        Delete documents by a specific field value.

        Checks if any matching documents exist before attempting deletion.

        Args:
            field_name: The name of the field to filter by.
            field_value: The value to match.

        Returns:
            True if deletion was successful or no matching docs found, False on error.
        """
        try:
            if not await self.afield_exists(field_name, field_value):
                debug_log(
                    f"No documents with {field_name}='{field_value}' found. No action taken.",
                    context="WeaviateVectorDB"
                )
                return True

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
    
    async def adelete_by_document_name(self, document_name: str) -> bool:
        """Delete documents by document_name (async)."""
        return await self.adelete_by_field("document_name", document_name)
    
    async def adelete_by_document_id(self, document_id: str) -> bool:
        """Delete documents by document_id (async)."""
        return await self.adelete_by_field("document_id", document_id)

    async def adelete_by_chunk_id(self, chunk_id: str) -> bool:
        """Delete documents by chunk_id (async)."""
        return await self.adelete_by_field("chunk_id", chunk_id)

    async def adelete_by_doc_content_hash(self, doc_content_hash: str) -> bool:
        """Delete documents by doc_content_hash (async)."""
        return await self.adelete_by_field("doc_content_hash", doc_content_hash)

    async def adelete_by_chunk_content_hash(self, chunk_content_hash: str) -> bool:
        """Delete documents by chunk_content_hash (async)."""
        return await self.adelete_by_field("chunk_content_hash", chunk_content_hash)
    
    async def adelete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Delete documents by metadata filter (async).

        Checks if any matching documents exist before attempting deletion.

        Args:
            metadata: Dictionary of metadata fields to match.

        Returns:
            True if deletion was successful or no matching docs found, False on error.
        """
        try:
            collection_obj = await self._get_collection()

            filter_expr = self._translate_filter(metadata)
            if filter_expr is None:
                debug_log(
                    f"No valid filter could be built for metadata: {metadata}",
                    context="WeaviateVectorDB"
                )
                return False

            check_result = await collection_obj.query.fetch_objects(
                limit=1,
                filters=filter_expr
            )
            if not check_result.objects:
                debug_log(
                    f"No documents matching metadata '{metadata}' found. No action taken.",
                    context="WeaviateVectorDB"
                )
                return True

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
    
    async def adocument_name_exists(self, document_name: str) -> bool:
        """Check if a document with the given document_name exists (async)."""
        return await self.afield_exists("document_name", document_name)
    
    async def adocument_id_exists(self, document_id: str) -> bool:
        """Check if a document with the given document_id exists (async)."""
        return await self.afield_exists("document_id", document_id)

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        """Check if a document with the given chunk_id exists (async)."""
        return await self.afield_exists("chunk_id", chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        """Check if a document with the given doc_content_hash exists (async)."""
        return await self.afield_exists("doc_content_hash", doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        """Check if a chunk with the given chunk_content_hash exists (async)."""
        return await self.afield_exists("chunk_content_hash", chunk_content_hash)

    async def afield_exists(self, field_name: str, field_value: Any) -> bool:
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

    async def aupdate_metadata(
        self,
        chunk_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Update the metadata for the chunk with the given chunk_id (async).

        Args:
            chunk_id: The chunk ID to update.
            metadata: The metadata fields to update/merge.

        Returns:
            True if update was successful, False otherwise.
        """
        try:
            collection_obj = await self._get_collection()

            query_result = await collection_obj.query.fetch_objects(
                filters=wvc.query.Filter.by_property("chunk_id").equal(chunk_id),
                limit=1,
            )

            if not query_result.objects:
                debug_log(
                    f"No document found with chunk_id: {chunk_id}",
                    context="WeaviateVectorDB",
                )
                return False

            obj = query_result.objects[0]
            current_properties: Dict[str, Any] = obj.properties or {}
            updated_properties: Dict[str, Any] = current_properties.copy()

            existing_metadata: Dict[str, Any] = {}
            if "metadata" in updated_properties:
                metadata_str = updated_properties["metadata"]
                if isinstance(metadata_str, str):
                    try:
                        existing_metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        existing_metadata = {}
                elif isinstance(metadata_str, dict):
                    existing_metadata = metadata_str

            existing_metadata.update(metadata)
            updated_properties["metadata"] = json.dumps(existing_metadata) if existing_metadata else "{}"

            await collection_obj.data.update(
                uuid=obj.uuid,
                properties=updated_properties,
            )

            info_log(
                f"Updated metadata for chunk_id: {chunk_id}",
                context="WeaviateVectorDB",
            )
            return True

        except Exception as e:
            debug_log(
                f"Error updating metadata for chunk_id '{chunk_id}': {e}",
                context="WeaviateVectorDB",
            )
            return False
    
    async def aoptimize(self) -> bool:
        """Optimize the vector database (async). Weaviate doesn't require explicit optimization."""
        return True
    
    async def aget_supported_search_types(self) -> List[str]:
        """Get the supported search types for Weaviate (async)."""
        supported: List[str] = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported

    # ============================================================================
    # Private Helper Methods
    # ============================================================================

    async def _get_collection(self) -> Any:
        """
        Private helper to get the collection object, applying tenancy if configured.

        Uses aget_client() to ensure the client is connected and ready
        before retrieving the collection reference. The expensive
        ``collections.exists()`` check is deliberately omitted here because
        existence is already verified in lifecycle methods (acreate_collection,
        adelete_collection). ``collections.get()`` is a local operation in
        Weaviate v4 and does not make a network call, so it is safe to call
        without a prior existence check.

        Returns:
            A Weaviate collection object, properly scoped with tenant if applicable.

        Raises:
            VectorDBConnectionError: If the client is not connected or not ready.
            CollectionDoesNotExistError: If the collection doesn't exist.
        """
        client = await self.aget_client()

        collection_name = self._config.collection_name

        try:
            collection = client.collections.get(collection_name)
        except UnexpectedStatusCodeError as e:
            if e.status_code == 404:
                raise CollectionDoesNotExistError(
                    f"Collection '{collection_name}' does not exist in Weaviate."
                )
            raise VectorDBError(f"Failed to retrieve collection: {e.message}")

        if self._config.namespace:
            return collection.with_tenant(self._config.namespace)

        return collection

    def _build_vector_index_config(self, distance_map: Dict[DistanceMetric, Any]) -> Any:
        """
        Build the vector index configuration based on config.
        
        Args:
            distance_map: Mapping of DistanceMetric to Weaviate distance metrics.
        
        Returns:
            A Weaviate vector index configuration object.
        """
        index_conf = self._config.index
        
        if isinstance(index_conf, HNSWIndexConfig):
            hnsw_params: Dict[str, Any] = {
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
            return wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=distance_map[self._config.distance_metric],
                max_connections=16,
                ef_construction=200
            )

    def _build_vector_config(self, distance_map: Dict[DistanceMetric, Any]) -> Any:
        """
        Build the vector configuration for self-provided dense vectors.
        
        Weaviate only supports dense vectors. Hybrid search combines dense vectors
        with BM25 keyword search (not sparse vectors).
        
        Args:
            distance_map: Mapping of DistanceMetric to Weaviate distance metrics.
        
        Returns:
            A Weaviate vector configuration object for self-provided vectors.
        """
        vector_index_config = self._build_vector_index_config(distance_map)
        
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
        - chunk_id (TEXT, always indexed)
        - content (TEXT, tokenized for BM25, always searchable)
        - metadata (TEXT, JSON serialized, optionally indexed)
        
        indexed_fields can be:
        - Simple list: ["document_name", "document_id"]
        - Advanced list: [{"field": "document_name", "type": "keyword"}, {"field": "age", "type": "integer"}]
        
        Args:
            additional_properties: Additional custom properties from config.
        
        Returns:
            A list of Weaviate Property objects.
        """
        properties: List[Any] = []
        indexed_fields_config = self._parse_indexed_fields()
        
        field_config = indexed_fields_config.get("document_name", {})
        properties.append(wvc.config.Property(
            name="document_name",
            data_type=self._get_weaviate_datatype(field_config.get("type", "text")),
            tokenization=self._get_weaviate_tokenization(field_config.get("type", "text")),
            skip_vectorization=True,
            index_filterable=field_config.get("indexed", False),
            index_searchable=field_config.get("indexed", False)
        ))
        
        field_config = indexed_fields_config.get("document_id", {})
        properties.append(wvc.config.Property(
            name="document_id",
            data_type=self._get_weaviate_datatype(field_config.get("type", "text")),
            tokenization=self._get_weaviate_tokenization(field_config.get("type", "text")),
            skip_vectorization=True,
            index_filterable=field_config.get("indexed", False),
            index_searchable=field_config.get("indexed", False)
        ))
        
        properties.append(wvc.config.Property(
            name="chunk_id",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.WORD,
            skip_vectorization=True,
            index_filterable=True,
            index_searchable=True,
        ))
        
        field_config = indexed_fields_config.get("content", {})
        properties.append(wvc.config.Property(
            name="content",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.LOWERCASE,
            skip_vectorization=True,
            index_filterable=field_config.get("indexed", False),
            index_searchable=True
        ))
        
        properties.append(wvc.config.Property(
            name="metadata",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.WORD,
            skip_vectorization=True,
            index_filterable=True,
            index_searchable=True
        ))

        properties.append(wvc.config.Property(
            name="chunk_content_hash",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.WORD,
            skip_vectorization=True,
            index_filterable=True,
            index_searchable=True,
        ))

        field_config = indexed_fields_config.get("doc_content_hash", {})
        properties.append(wvc.config.Property(
            name="doc_content_hash",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.WORD,
            skip_vectorization=True,
            index_filterable=field_config.get("indexed", True),
            index_searchable=field_config.get("indexed", True),
        ))

        properties.append(wvc.config.Property(
            name="knowledge_base_id",
            data_type=wvc.config.DataType.TEXT,
            tokenization=wvc.config.Tokenization.WORD,
            skip_vectorization=True,
            index_filterable=True,
            index_searchable=False,
        ))

        if self._config.properties:
            properties.extend(self._parse_custom_properties(self._config.properties))
        
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
        
        result: Dict[str, Dict[str, Any]] = {}
        for item in self._config.indexed_fields:
            if isinstance(item, str):
                result[item] = {"indexed": True, "type": "text"}
            elif isinstance(item, dict):
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
        if field_type.lower() in ['text', 'keyword']:
            if field_type.lower() == 'keyword':
                return wvc.config.Tokenization.WORD
            else:
                return wvc.config.Tokenization.WHITESPACE
        return None
    
    def _parse_custom_properties(self, props: List[Dict[str, Any]]) -> List[Any]:
        """Parse custom properties from dict format to Weaviate Property objects."""
        parsed_properties: List[Any] = []
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
            if prop_name in self._STANDARD_FIELDS:
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
        
        return wvc.config.Configure.replication(
            factor=config_dict.get('factor', 1),
            async_enabled=config_dict.get('asyncEnabled', False)
        )
    
    def _build_sharding_config(self, override: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Build sharding configuration."""
        config_dict = override or self._config.sharding_config
        if not config_dict:
            return None
        
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
        except AttributeError:
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
        
        parsed_refs: List[Any] = []
        for ref in refs:
            parsed_refs.append(
                wvc.config.ReferenceProperty(
                    name=ref.get('name'),
                    target_collection=ref.get('target')
                )
            )
        
        return parsed_refs if parsed_refs else None

    def _hydrate_payload(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a contract-compliant VectorSearchResult.payload dict from a
        Weaviate ``obj.properties`` mapping.

        Standard fields are placed at the top level; non-standard keys that
        Weaviate returns as a JSON-serialized ``metadata`` property are parsed
        back to a dict and placed under ``payload["metadata"]``.
        """
        raw_metadata = properties.get("metadata", "{}") if properties else "{}"
        if isinstance(raw_metadata, dict):
            metadata_dict = raw_metadata
        elif isinstance(raw_metadata, str) and raw_metadata:
            try:
                parsed = json.loads(raw_metadata)
                metadata_dict = parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                metadata_dict = {}
        else:
            metadata_dict = {}

        return {
            "chunk_id": properties.get("chunk_id", "") if properties else "",
            "document_id": properties.get("document_id", "") if properties else "",
            "document_name": properties.get("document_name", "") if properties else "",
            "content": properties.get("content", "") if properties else "",
            "doc_content_hash": properties.get("doc_content_hash", "") if properties else "",
            "chunk_content_hash": properties.get("chunk_content_hash", "") if properties else "",
            "knowledge_base_id": properties.get("knowledge_base_id", "") if properties else "",
            "metadata": metadata_dict,
        }

    def _process_payload(
        self,
        payload: Dict[str, Any],
        content: str,
        chunk_id: str,
        chunk_content_hash: str,
        document_id: str = "",
        doc_content_hash: str = "",
        document_name: str = "",
        knowledge_base_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        _warned_standard_keys: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Build the Weaviate properties dict from explicit IDs and a metadata payload.

        Primary (chunk-focused) identifiers are always set.  Document-level
        identifiers are optional — they default to empty strings when the
        vector DB is used without KnowledgeBase.

        The *payload* dict is treated as pure metadata — any keys that are not
        recognised standard fields are swept into the serialised ``metadata``
        JSON property.

        Args:
            payload: Metadata dict (source, document_name, etc.).
            content: The chunk text.
            chunk_id: The unique chunk identifier (same value used as object UUID).
            chunk_content_hash: MD5 hash of the chunk's text content.
            document_id: The parent document's deterministic identifier.
            doc_content_hash: MD5 hash of the parent document's full content.
            extra_metadata: Additional metadata to merge from caller.

        Returns:
            A properly formatted properties dict for Weaviate.
        """
        properties: Dict[str, Any] = {}

        properties['chunk_id'] = chunk_id
        properties['chunk_content_hash'] = chunk_content_hash
        properties['content'] = content

        properties['document_id'] = document_id
        properties['doc_content_hash'] = doc_content_hash
        properties['document_name'] = document_name
        properties['knowledge_base_id'] = knowledge_base_id or ""

        metadata: Dict[str, Any] = {}

        if self._config.default_metadata:
            metadata.update(self._config.default_metadata)

        if payload:
            for key, value in payload.items():
                if key in self._STANDARD_FIELDS:
                    if _warned_standard_keys is not None and key not in _warned_standard_keys:
                        debug_log(
                            f"Standard field '{key}' found in payload dict and will be ignored. "
                            f"Standard fields must be passed via dedicated parameters "
                            f"(ids, document_ids, document_names, doc_content_hashes, "
                            f"chunk_content_hashes, knowledge_base_ids).",
                            context="WeaviateVectorDB",
                        )
                        _warned_standard_keys.add(key)
                    continue
                metadata[key] = value

        if extra_metadata:
            metadata.update(extra_metadata)

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
            return vector_obj.get('default')
        else:
            return vector_obj

    def _translate_filter(self, filter_dict: Dict[str, Any]) -> Any:
        """
        Recursively translates a framework-standard filter dictionary into a Weaviate Filter object.
        
        Supports:
        - Logical operators: "and", "or"
        - Comparison operators: "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"
        - Direct field equality: {"field": "value"}
        
        Fields that are not standard properties (document_name, document_id, chunk_id, content)
        are automatically searched within the JSON-serialized metadata field.
        
        Args:
            filter_dict: A dictionary representing the filter logic.
        
        Returns:
            A Weaviate Filter object ready to be used in a query.
        
        Raises:
            SearchError: If an unknown operator or invalid filter structure is provided.
        """
        # Use the canonical class-level frozenset so the filter routing stays in sync
        # whenever a new standard field is added. All 8 standards are first-class Weaviate
        # schema properties, so any of them can be filtered natively.
        standard_fields = self._STANDARD_FIELDS
        
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

        filters: List[Any] = []
        for key, value in filter_dict.items():
            if key in logical_ops:
                sub_filters = [self._translate_filter(sub_filter) for sub_filter in value]
                return logical_ops[key](sub_filters)
            
            if key in standard_fields:
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
                import json as json_module
                pattern = f'"{key}": "{value}"' if isinstance(value, str) else f'"{key}": {json_module.dumps(value)}'
                metadata_filter = wvc.query.Filter.by_property("metadata").like(f"*{pattern}*")
                filters.append(metadata_filter)

        if not filters:
            raise SearchError("Filter dictionary cannot be empty.")
        
        return wvc.query.Filter.all_of(filters) if len(filters) > 1 else filters[0]
