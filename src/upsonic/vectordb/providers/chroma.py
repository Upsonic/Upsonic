from __future__ import annotations

import asyncio
import json
import math
from hashlib import md5
from typing import Any, Dict, List, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb
    from chromadb.api.client import Client as ChromaClientAPI
    from chromadb.api.models.Collection import Collection as ChromaCollection
    from chromadb.errors import NotFoundError

try:
    import chromadb
    from chromadb.api.client import Client as ChromaClientAPI
    from chromadb.api.models.Collection import Collection as ChromaCollection
    from chromadb.errors import NotFoundError
    _CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore
    ChromaClientAPI = None  # type: ignore
    ChromaCollection = None  # type: ignore
    NotFoundError = None  # type: ignore
    _CHROMADB_AVAILABLE = False


from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.utils.printing import info_log, debug_log

from upsonic.vectordb.config import (
    ChromaConfig,
    Mode, 
    DistanceMetric,
    HNSWIndexConfig,
    FlatIndexConfig
)

from upsonic.utils.package.exception import(
    VectorDBConnectionError, 
    ConfigurationError, 
    CollectionDoesNotExistError,
    VectorDBError,
    SearchError,
    UpsertError
)

from upsonic.schemas.vector_schemas import VectorSearchResult


class ChromaProvider(BaseVectorDBProvider):
    """
    A high-level, comprehensive, async-first implementation of ChromaDB provider.
    
    This provider integrates best practices from multiple frameworks and provides:
    - Full async support for all operations
    - Flexible metadata handling with nested structure flattening
    - Advanced filtering capabilities
    - Content hash tracking for deduplication
    - Comprehensive data management methods
    - Optional reranker support for search result re-ranking
    
    Key Features:
    - document_name, document_id, content_id metadata tracking
    - Dense vector search (ChromaDB automatically indexes all metadata fields)
    - Full-text search using document content filtering
    - Hybrid search combining vector and text
    - Rich filtering with ChromaDB operators ($eq, $ne, $in, $gt, etc.)
    - Multiple deletion strategies (by ID, filter, document_name, content_id, etc.)
    - Metadata updates and existence checks
    
    Note: ChromaDB only supports dense vectors. Sparse vectors are not supported.
    
    Attributes:
        reranker: Optional reranker instance for re-ranking search results
    """
    
    def __init__(
        self, 
        config: Union[ChromaConfig, Dict[str, Any]],
        reranker: Optional[Any] = None
    ):
        """
        Initialize ChromaProvider with config or dict.
        
        Args:
            config: ChromaConfig object or dict containing configuration
            reranker: Optional reranker for search result re-ranking (to be implemented)
        """
        if not _CHROMADB_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="chromadb",
                install_command='pip install "upsonic[rag]"',
                feature_name="ChromaDB vector database provider"
            )
        
        # Handle dict config
        if isinstance(config, dict):
            config = ChromaConfig.from_dict(config)
        
        super().__init__(config)
        self._config: ChromaConfig = config  # Type hint for better IDE support
        self._validate_config()
        self._collection_instance: Optional[ChromaCollection] = None
        
        # Provider metadata
        self.provider_name = config.provider_name or f"ChromaProvider_{config.collection_name}"
        self.provider_description = config.provider_description
        self.provider_id = config.provider_id or self._generate_provider_id()
        
        # Provider utilities
        self.reranker = reranker
        
    def _validate_config(self) -> None:
        """Validate Chroma-specific configuration."""
        debug_log("Performing Chroma-specific configuration validation...", context="ChromaVectorDB")
        
        if self._config.connection.mode == Mode.EMBEDDED and self._config.connection.db_path is None:
            raise ConfigurationError("Missing Path: 'db_path' must be set for EMBEDDED (PersistentClient) mode.")
        
        info_log("Chroma configuration validated successfully.", context="ChromaVectorDB")
    
    def _generate_provider_id(self) -> str:
        """Generates a unique provider ID based on connection details and collection."""
        conn = self._config.connection
        identifier_parts = [
            conn.host or conn.url or conn.location or "local",
            str(conn.port) if conn.port else "",
            self._config.collection_name
        ]
        identifier = "#".join(filter(None, identifier_parts))
        
        return md5(identifier.encode()).hexdigest()[:16]

    # ============================================================================
    # Connection Management (Async)
    # ============================================================================

    async def connect(self) -> None:
        """Establish connection to ChromaDB asynchronously."""
        if self._is_connected:
            return
        
        debug_log(f"Connecting to ChromaDB in '{self._config.connection.mode.value}' mode...", context="ChromaVectorDB")
        
        try:
            # Run synchronous client creation in thread pool
            client_instance = await asyncio.to_thread(self._create_client)
            
            # Verify connection with heartbeat
            await asyncio.to_thread(client_instance.heartbeat)
            
            self._client = client_instance
            self._is_connected = True
            info_log("ChromaDB connection successful and verified.", context="ChromaVectorDB")
            
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to connect to ChromaDB: {e}") from e

    def _create_client(self) -> ChromaClientAPI:
        """Create ChromaDB client based on configuration."""
        client_instance: ChromaClientAPI
        
        if self._config.connection.mode == Mode.IN_MEMORY:
            client_instance = chromadb.Client()
            
        elif self._config.connection.mode == Mode.EMBEDDED:
            client_instance = chromadb.PersistentClient(path=self._config.connection.db_path)
            
        elif self._config.connection.mode == Mode.LOCAL:
            if not self._config.connection.host or not self._config.connection.port:
                raise ConfigurationError("Host and port must be specified for LOCAL mode.")
            client_instance = chromadb.HttpClient(
                host=self._config.connection.host, 
                port=self._config.connection.port
            )
            
        elif self._config.connection.mode == Mode.CLOUD:
            if not self._config.connection.api_key:
                raise ConfigurationError("api_key must be specified for CLOUD mode.")
            
            # Prepare CloudClient kwargs
            cloud_kwargs = {
                "api_key": self._config.connection.api_key.get_secret_value()
            }
            
            # Add tenant and database if provided
            if self._config.tenant:
                cloud_kwargs["tenant"] = self._config.tenant
            if self._config.database:
                cloud_kwargs["database"] = self._config.database
            
            # Use CloudClient for Chroma Cloud connections
            try:
                client_instance = chromadb.CloudClient(**cloud_kwargs)
            except (AttributeError, ImportError, TypeError) as e:
                # Fallback to HttpClient if CloudClient is not available
                if not self._config.connection.host:
                    raise ConfigurationError("CloudClient not available and no host specified for fallback HttpClient.")
                
                headers = {"Authorization": f"Bearer {self._config.connection.api_key.get_secret_value()}"}
                fallback_kwargs = {
                    "host": self._config.connection.host, 
                    "headers": headers,
                    "ssl": self._config.connection.use_tls
                }
                client_instance = chromadb.HttpClient(**fallback_kwargs)
        else:
            raise ConfigurationError(f"Unsupported mode for ChromaProvider: {self._config.connection.mode}")
        
        return client_instance

    def connect_sync(self) -> None:
        """Establish connection to ChromaDB synchronously."""
        return self._run_async_from_sync(self.connect())

    async def disconnect(self) -> None:
        """Gracefully disconnect from ChromaDB."""
        if not self._is_connected or not self._client:
            return
        
        debug_log("Disconnecting from ChromaDB...", context="ChromaVectorDB")
        
        try:
            await asyncio.to_thread(self._client.reset)
        except Exception:
            pass
        finally:
            self._client = None
            self._is_connected = False
            self._collection_instance = None
            info_log("ChromaDB client session has been reset.", context="ChromaVectorDB")

    def disconnect_sync(self) -> None:
        """Gracefully disconnect from ChromaDB synchronously."""
        return self._run_async_from_sync(self.disconnect())

    async def is_ready(self) -> bool:
        """Check if the database is ready and responsive."""
        if not self._is_connected or not self._client:
            return False
        
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception:
            return False

    def is_ready_sync(self) -> bool:
        """Check if the database is ready and responsive (sync)."""
        return self._run_async_from_sync(self.is_ready())

    # ============================================================================
    # Collection Management (Async)
    # ============================================================================

    async def create_collection(self) -> None:
        """Create or retrieve the collection with proper configuration."""
        if not await self.is_ready():
            raise VectorDBConnectionError("Cannot create collection: Provider is not connected or ready.")
        
        collection_name = self._config.collection_name
        
        try:
            # Handle recreate_if_exists
            if self._config.recreate_if_exists and await self.collection_exists():
                info_log(f"Configuration specifies 'recreate_if_exists'. Deleting existing collection '{collection_name}'...", context="ChromaVectorDB")
                await self.delete_collection()
            
            # Prepare collection metadata
            chroma_metadata = self._translate_config_to_chroma_metadata()
            
            debug_log(f"Creating or retrieving collection '{collection_name}'...", context="ChromaVectorDB")
            
            # Create or get collection
            self._collection_instance = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=collection_name,
                metadata=chroma_metadata
            )
            
            info_log(f"Successfully prepared collection '{collection_name}'.", context="ChromaVectorDB")
            
        except Exception as e:
            raise VectorDBError(f"Failed to create or get collection '{collection_name}': {e}") from e

    def create_collection_sync(self) -> None:
        """Create or retrieve the collection with proper configuration (sync)."""
        return self._run_async_from_sync(self.create_collection())
            
    def _translate_config_to_chroma_metadata(self) -> dict:
        """Translate framework config to ChromaDB metadata."""
        distance_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.DOT_PRODUCT: "ip"
        }
        
        metadata = {"hnsw:space": distance_map[self._config.distance_metric]}
        
        # Add HNSW-specific parameters
        if isinstance(self._config.index, HNSWIndexConfig):
            metadata["hnsw:M"] = self._config.index.m
            metadata["hnsw:construction_ef"] = self._config.index.ef_construction
        
        return metadata

    async def delete_collection(self) -> None:
        """Delete the collection permanently."""
        if not await self.is_ready():
            raise VectorDBConnectionError("Cannot delete collection: Provider is not connected or ready.")
        
        collection_name = self._config.collection_name
        debug_log(f"Attempting to delete collection '{collection_name}'...", context="ChromaVectorDB")
        
        try:
            await asyncio.to_thread(self._client.delete_collection, name=collection_name)
            self._collection_instance = None
            info_log(f"Collection '{collection_name}' deleted successfully.", context="ChromaVectorDB")
            
        except (ValueError, chromadb.errors.NotFoundError) as e:
            raise CollectionDoesNotExistError(f"Cannot delete collection '{collection_name}' because it does not exist.") from e
        except Exception as e:
            raise VectorDBError(f"An unexpected error occurred while deleting collection '{collection_name}': {e}") from e

    def delete_collection_sync(self) -> None:
        """Delete the collection permanently (sync)."""
        return self._run_async_from_sync(self.delete_collection())

    async def collection_exists(self) -> bool:
        """Check if the collection exists."""
        if not await self.is_ready():
            raise VectorDBConnectionError("Cannot check for collection: Provider is not connected or ready.")
        
        try:
            collection = await asyncio.to_thread(
                self._client.get_collection,
                name=self._config.collection_name
            )
            
            if self._collection_instance is None:
                self._collection_instance = collection
            
            return True
            
        except NotFoundError:
            return False
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to check collection existence due to a server error: {e}") from e

    def collection_exists_sync(self) -> bool:
        """Check if the collection exists (sync)."""
        return self._run_async_from_sync(self.collection_exists())

    async def _get_active_collection(self) -> ChromaCollection:
        """Ensure the collection instance is available."""
        if self._collection_instance is None:
            # Try to get the collection
            try:
                self._collection_instance = await asyncio.to_thread(
                    self._client.get_collection,
                    name=self._config.collection_name
                )
            except Exception:
                raise VectorDBError("Collection is not initialized. Please call 'create_collection' before performing data operations.")
        
        return self._collection_instance

    # ============================================================================
    # Metadata Handling
    # ============================================================================

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Flatten nested metadata to ChromaDB-compatible format.
        
        ChromaDB only supports primitive types in metadata. This method
        recursively flattens nested structures using dot notation and
        converts complex types to JSON strings.
        
        Args:
            metadata: Dictionary that may contain nested structures
            
        Returns:
            Flattened dictionary with only primitive values
        """
        flattened: Dict[str, Any] = {}
        
        def _flatten_recursive(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                if len(obj) == 0:
                    # Handle empty dictionaries by converting to JSON string
                    flattened[prefix] = json.dumps(obj)
                else:
                    for key, value in obj.items():
                        new_key = f"{prefix}.{key}" if prefix else key
                        _flatten_recursive(value, new_key)
            elif isinstance(obj, (list, tuple)):
                # Convert lists/tuples to JSON strings
                flattened[prefix] = json.dumps(obj)
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                if obj is not None:  # ChromaDB doesn't accept None values
                    flattened[prefix] = obj
            else:
                # Convert other complex types to JSON strings
                try:
                    flattened[prefix] = json.dumps(obj)
                except (TypeError, ValueError):
                    # If it can't be serialized, convert to string
                    flattened[prefix] = str(obj)
        
        _flatten_recursive(metadata)
        return flattened

    def _prepare_metadata(
        self, 
        payload: Dict[str, Any],
        document_name: Optional[str] = None,
        document_id: Optional[str] = None,
        content_id: Optional[str] = None,
        content_hash: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare metadata for upserting by merging all sources.
        
        Args:
            payload: Base payload/metadata
            document_name: Optional document name
            document_id: Optional document ID
            content_id: Content ID (main identifier)
            content_hash: Hash of content for deduplication
            additional_metadata: Additional metadata from method call
            
        Returns:
            Merged and flattened metadata
        """
        # Start with payload
        merged_metadata = dict(payload) if payload else {}
        
        # Add default metadata from config
        if self._config.default_metadata:
            for key, value in self._config.default_metadata.items():
                if key not in merged_metadata:
                    merged_metadata[key] = value
        
        # Add additional metadata from method call
        if additional_metadata:
            merged_metadata.update(additional_metadata)
        
        # Add tracking fields
        if document_name is not None:
            merged_metadata["document_name"] = document_name
        if document_id is not None:
            merged_metadata["document_id"] = document_id
        if content_id is not None:
            merged_metadata["content_id"] = content_id
        if content_hash is not None:
            merged_metadata["content_hash"] = content_hash
        
        # Flatten for ChromaDB compatibility
        return self._flatten_metadata(merged_metadata)

    def _generate_content_id(self, content: str) -> str:
        """Generate a unique content ID based on content hash."""
        return md5(content.encode()).hexdigest()

    # ============================================================================
    # Filter Building
    # ============================================================================

    def _convert_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Convert simple filters to ChromaDB's format.
        
        ChromaDB requires:
        - Single condition: {"key": "value"} or {"key": {"$operator": value}}
        - Multiple conditions: {"$and": [{"key1": value1}, {"key2": value2}]}
        
        Args:
            filters: Filter dictionary to convert
            
        Returns:
            ChromaDB-compatible filter dictionary
        """
        if not filters:
            return None
        
        # If filters already use logical operators at top level, return as is
        if any(key.startswith("$") for key in filters.keys()):
            return filters
        
        # Convert simple key-value pairs
        conditions = []
        for key, value in filters.items():
            # If the value is already a dict with operators, add as is
            if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                conditions.append({key: value})
            # Convert lists to $in operator
            elif isinstance(value, (list, tuple)):
                conditions.append({key: {"$in": list(value)}})
            # Keep simple values as is (ChromaDB interprets as equality)
            else:
                conditions.append({key: value})
        
        # If only one condition, return it directly
        if len(conditions) == 1:
            return conditions[0]
        
        # If multiple conditions, wrap in $and
        return {"$and": conditions}


    # ============================================================================
    # Data Operations (Async)
    # ============================================================================

    async def upsert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        document_names: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Add or update records in the ChromaDB collection.
        
        Args:
            vectors: List of dense vector embeddings
            payloads: List of corresponding metadata objects
            ids: List of unique identifiers
            chunks: List of text content (required for full-text search)
            sparse_vectors: Not supported by ChromaDB (ignored if provided)
            document_names: Optional list of document names
            document_ids: Optional list of document IDs
            content_ids: Optional list of content IDs (auto-generated if not provided)
            additional_metadata: Additional metadata to merge into all records
            **kwargs: Provider-specific options
            
        Raises:
            UpsertError: If the data ingestion operation fails
            VectorDBError: If the collection is not initialized
        """
        collection = await self._get_active_collection()
        debug_log(f"Upserting {len(ids)} records into collection '{collection.name}'...", context="ChromaVectorDB")
        
        # Validate inputs
        if len(vectors) != len(ids):
            raise UpsertError(f"Number of vectors ({len(vectors)}) must match number of IDs ({len(ids)})")
        if len(payloads) != len(ids):
            raise UpsertError(f"Number of payloads ({len(payloads)}) must match number of IDs ({len(ids)})")
        
        # Log warning if sparse vectors provided (not supported)
        if sparse_vectors is not None:
            debug_log("Sparse vectors are not supported by ChromaDB and will be ignored.", context="ChromaVectorDB")
        
        try:
            # Prepare data
            upsert_embeddings = []
            upsert_metadatas = []
            upsert_ids = []
            upsert_documents = []
            
            for i in range(len(ids)):
                # Get components
                vector = vectors[i]
                payload = payloads[i]
                doc_id = str(ids[i])
                content = chunks[i] if chunks and i < len(chunks) else None
                
                # Get tracking fields
                doc_name = document_names[i] if document_names and i < len(document_names) else None
                document_id = document_ids[i] if document_ids and i < len(document_ids) else None
                
                # Generate or get content_id
                if content_ids and i < len(content_ids):
                    content_id = content_ids[i]
                elif self._config.auto_generate_content_id and content:
                    content_id = self._generate_content_id(content)
                else:
                    content_id = doc_id  # Fallback to record ID
                
                # Generate content hash if content exists
                content_hash = None
                if content:
                    content_hash = md5(content.encode()).hexdigest()
                
                # Prepare metadata
                prepared_metadata = self._prepare_metadata(
                    payload=payload,
                    document_name=doc_name,
                    document_id=document_id,
                    content_id=content_id,
                    content_hash=content_hash,
                    additional_metadata=additional_metadata
                )
                
                upsert_embeddings.append(vector)
                upsert_metadatas.append(prepared_metadata)
                upsert_ids.append(doc_id)
                
                if content is not None:
                    # Clean content (remove null bytes)
                    cleaned_content = content.replace("\x00", "\ufffd")
                    upsert_documents.append(cleaned_content)
                else:
                    upsert_documents.append("")
            
            # Perform upsert
            upsert_params = {
                "embeddings": upsert_embeddings,
                "metadatas": upsert_metadatas,
                "ids": upsert_ids
            }
            
            if chunks is not None:
                upsert_params["documents"] = upsert_documents
            
            await asyncio.to_thread(collection.upsert, **upsert_params)
            
            info_log(f"Successfully upserted {len(ids)} records.", context="ChromaVectorDB")
            
        except Exception as e:
            raise UpsertError(f"Failed to upsert data into collection '{collection.name}': {e}") from e

    def upsert_sync(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Add or update records in the ChromaDB collection (sync)."""
        return self._run_async_from_sync(
            self.upsert(vectors, payloads, ids, chunks, sparse_vectors, **kwargs)
        )

    async def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Remove records from the collection by their unique identifiers.
        
        Args:
            ids: List of specific IDs to remove
            **kwargs: Provider-specific options
            
        Raises:
            VectorDBError: If the deletion fails or the collection is not initialized
        """
        collection = await self._get_active_collection()
        debug_log(f"Deleting {len(ids)} records from collection '{collection.name}'...", context="ChromaVectorDB")
        
        try:
            await asyncio.to_thread(
                collection.delete,
                ids=[str(i) for i in ids]
            )
            info_log(f"Successfully deleted {len(ids)} records.", context="ChromaVectorDB")
            
        except Exception as e:
            raise VectorDBError(f"Failed to delete records from collection '{collection.name}': {e}") from e

    def delete_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Remove records from the collection by their unique identifiers (sync)."""
        return self._run_async_from_sync(self.delete(ids, **kwargs))

    def delete_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Alias for delete_sync() - provided for backward compatibility."""
        return self.delete_sync(ids, **kwargs)

    async def delete_by_filter(self, filter: Dict[str, Any], **kwargs) -> bool:
        """
        Delete documents matching the given filter.
        
        Args:
            filter: Metadata filter to match documents
            **kwargs: Provider-specific options
            
        Returns:
            True if deletion was successful, False otherwise
            
        Raises:
            VectorDBError: If deletion fails
        """
        collection = await self._get_active_collection()
        
        try:
            # Convert filter to ChromaDB format
            where_filter = self._convert_filters(filter)
            
            # Get matching documents
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter
            )
            
            ids_to_delete = result.get("ids", [])
            
            if not ids_to_delete:
                debug_log(f"No documents found matching filter: {filter}", context="ChromaVectorDB")
                return False
            
            # Delete matching documents
            await asyncio.to_thread(collection.delete, ids=ids_to_delete)
            
            info_log(f"Deleted {len(ids_to_delete)} documents matching filter.", context="ChromaVectorDB")
            return True
            
        except Exception as e:
            raise VectorDBError(f"Failed to delete documents by filter: {e}") from e

    def delete_by_document_name(self, document_name: str) -> bool:
        """Delete all documents with the given document_name (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_name(document_name))
    
    async def async_delete_by_document_name(self, document_name: str) -> bool:
        """Delete all documents with the given document_name (async)."""
        return await self.delete_by_filter({"document_name": document_name})

    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all documents with the given document_id (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_id(document_id))
    
    async def async_delete_by_document_id(self, document_id: str) -> bool:
        """Delete all documents with the given document_id (async)."""
        return await self.delete_by_filter({"document_id": document_id})

    def delete_by_content_id(self, content_id: str) -> bool:
        """Delete all documents with the given content_id (sync)."""
        return self._run_async_from_sync(self.async_delete_by_content_id(content_id))
    
    async def async_delete_by_content_id(self, content_id: str) -> bool:
        """Delete all documents with the given content_id (async)."""
        return await self.delete_by_filter({"content_id": content_id})

    async def delete_by_content_hash(self, content_hash: str, **kwargs) -> bool:
        """Delete all documents with the given content_hash."""
        return await self.delete_by_filter({"content_hash": content_hash}, **kwargs)
    
    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete all documents matching the given metadata (sync)."""
        return self._run_async_from_sync(self.async_delete_by_metadata(metadata))
    
    async def async_delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete all documents matching the given metadata (async)."""
        return await self.delete_by_filter(metadata)

    async def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieve full records from the collection by their IDs.
        
        Args:
            ids: List of IDs for which to retrieve the full records
            **kwargs: Provider-specific options
            
        Returns:
            List of VectorSearchResult objects containing the fetched data.
            The order of results matches the order of the input IDs.
            
        Raises:
            VectorDBError: If fetching fails or the collection is not initialized
        """
        collection = await self._get_active_collection()
        debug_log(f"Fetching {len(ids)} records from collection '{collection.name}'...", context="ChromaVectorDB")
        
        try:
            results = await asyncio.to_thread(
                collection.get,
                ids=[str(i) for i in ids],
                include=["metadatas", "embeddings", "documents"]
            )
            
            # Build results map for efficient lookup
            results_map = {
                results['ids'][i]: {
                    "payload": results['metadatas'][i],
                    "vector": results['embeddings'][i] if results['embeddings'] is not None else None,
                    "text": results['documents'][i] if results['documents'] is not None else None
                }
                for i in range(len(results['ids']))
            }
            
            # Build final results in the order of input IDs
            final_results = []
            for an_id in ids:
                str_id = str(an_id)
                if str_id in results_map:
                    final_results.append(
                        VectorSearchResult(
                            id=str_id,
                            score=1.0,  # Fetch doesn't have scores
                            payload=results_map[str_id]["payload"],
                            vector=results_map[str_id]["vector"],
                            text=results_map[str_id]["text"]
                        )
                    )
            
            info_log(f"Successfully fetched {len(final_results)} records.", context="ChromaVectorDB")
            return final_results
            
        except Exception as e:
            raise VectorDBError(f"Failed to fetch records from collection '{collection.name}': {e}") from e

    def fetch_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Retrieve full records from the collection by their IDs (sync)."""
        return self._run_async_from_sync(self.fetch(ids, **kwargs))

    def fetch_by_id_sync(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """Alias for fetch_sync() - provided for backward compatibility."""
        return self.fetch_sync(ids, **kwargs)

    # ============================================================================
    # Existence Checks
    # ============================================================================

    async def id_exists(self, id: Union[str, int], **kwargs) -> bool:
        """Check if a document with the given ID exists."""
        collection = await self._get_active_collection()
        
        try:
            result = await asyncio.to_thread(collection.get, ids=[str(id)])
            found_ids = result.get("ids", [])
            return len(found_ids) > 0
        except Exception as e:
            debug_log(f"Error checking if ID '{id}' exists: {e}", context="ChromaVectorDB")
            return False

    def document_name_exists(self, document_name: str) -> bool:
        """Check if any document with the given document_name exists (sync)."""
        return self._run_async_from_sync(self.async_document_name_exists(document_name))
    
    async def async_document_name_exists(self, document_name: str) -> bool:
        """Check if any document with the given document_name exists (async)."""
        collection = await self._get_active_collection()
        
        try:
            where_filter = self._convert_filters({"document_name": document_name})
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                limit=1
            )
            return len(result.get("ids", [])) > 0
        except Exception as e:
            debug_log(f"Error checking if document_name '{document_name}' exists: {e}", context="ChromaVectorDB")
            return False
    
    def document_id_exists(self, document_id: str) -> bool:
        """Check if any document with the given document_id exists (sync)."""
        return self._run_async_from_sync(self.async_document_id_exists(document_id))
    
    async def async_document_id_exists(self, document_id: str) -> bool:
        """Check if any document with the given document_id exists (async)."""
        collection = await self._get_active_collection()
        
        try:
            where_filter = self._convert_filters({"document_id": document_id})
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                limit=1
            )
            return len(result.get("ids", [])) > 0
        except Exception as e:
            debug_log(f"Error checking if document_id '{document_id}' exists: {e}", context="ChromaVectorDB")
            return False

    def content_id_exists(self, content_id: str) -> bool:
        """Check if any document with the given content_id exists (sync)."""
        return self._run_async_from_sync(self.async_content_id_exists(content_id))
    
    async def async_content_id_exists(self, content_id: str) -> bool:
        """Check if any document with the given content_id exists (async)."""
        collection = await self._get_active_collection()
        
        try:
            where_filter = self._convert_filters({"content_id": content_id})
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                limit=1
            )
            return len(result.get("ids", [])) > 0
        except Exception as e:
            debug_log(f"Error checking if content_id '{content_id}' exists: {e}", context="ChromaVectorDB")
            return False

    async def content_hash_exists(self, content_hash: str, **kwargs) -> bool:
        """Check if any document with the given content_hash exists."""
        collection = await self._get_active_collection()
        
        try:
            where_filter = self._convert_filters({"content_hash": content_hash})
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                limit=1
            )
            return len(result.get("ids", [])) > 0
        except Exception as e:
            debug_log(f"Error checking if content_hash '{content_hash}' exists: {e}", context="ChromaVectorDB")
            return False

    # ============================================================================
    # Metadata Management
    # ============================================================================

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific content ID (sync)."""
        return self._run_async_from_sync(self.async_update_metadata(content_id, metadata))
    
    async def async_update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific content ID (async).
        
        Args:
            content_id: The content ID to update
            metadata: Metadata updates to apply
            
        Returns:
            True if update was successful, False otherwise
            
        Raises:
            VectorDBError: If update fails
        """
        if not metadata:
            raise ValueError("'metadata' must be provided")
        
        collection = await self._get_active_collection()
        
        try:
            # Find documents with matching content_id
            where_filter = self._convert_filters({"content_id": content_id})
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                include=["metadatas"]
            )
            
            target_ids = result.get("ids", [])
            current_metadatas = result.get("metadatas", [])
            
            if not target_ids:
                debug_log(f"No documents found with content_id '{content_id}' to update metadata.", context="ChromaVectorDB")
                return False
            
            # Flatten new metadata
            flattened_new_metadata = self._flatten_metadata(metadata)
            
            # Merge metadata for each document
            updated_metadatas = []
            for current_meta in current_metadatas:
                meta_dict = dict(current_meta) if current_meta else {}
                meta_dict.update(flattened_new_metadata)
                updated_metadatas.append(meta_dict)
            
            # Update in ChromaDB
            await asyncio.to_thread(
                collection.update,
                ids=target_ids,
                metadatas=updated_metadatas
            )
            
            info_log(f"Updated metadata for {len(target_ids)} documents with content_id '{content_id}'.", context="ChromaVectorDB")
            return True
            
        except Exception as e:
            raise VectorDBError(f"Failed to update metadata: {e}") from e

    async def get_count(self, filter: Optional[Dict[str, Any]] = None, **kwargs) -> int:
        """
        Get the count of documents in the collection.
        
        Args:
            filter: Optional filter to count specific documents
            **kwargs: Provider-specific options
            
        Returns:
            Number of documents
        """
        collection = await self._get_active_collection()
        
        try:
            if filter:
                where_filter = self._convert_filters(filter)
                result = await asyncio.to_thread(collection.get, where=where_filter)
                return len(result.get("ids", []))
            else:
                return await asyncio.to_thread(collection.count)
        except Exception as e:
            debug_log(f"Error getting count: {e}", context="ChromaVectorDB")
            return 0

    def get_count_sync(self, filter: Optional[Dict[str, Any]] = None, **kwargs) -> int:
        """Get the count of documents in the collection (sync)."""
        return self._run_async_from_sync(self.get_count(filter, **kwargs))
    
    # ============================================================================
    # Optimization
    # ============================================================================
    
    def optimize(self) -> bool:
        """Optimize the vector database (sync). ChromaDB doesn't require explicit optimization."""
        return True
    
    async def async_optimize(self) -> bool:
        """Optimize the vector database (async). ChromaDB doesn't require explicit optimization."""
        return True
    
    # ============================================================================
    # Search Type Support
    # ============================================================================
    
    def get_supported_search_types(self) -> List[str]:
        """Get the supported search types for ChromaDB (sync)."""
        supported = []
        if self._config.dense_search_enabled:
            supported.append("dense")
        if self._config.full_text_search_enabled:
            supported.append("full_text")
        if self._config.hybrid_search_enabled:
            supported.append("hybrid")
        return supported
    
    async def async_get_supported_search_types(self) -> List[str]:
        """Get the supported search types for ChromaDB (async)."""
        return self.get_supported_search_types()

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
        Master search method that dispatches to appropriate search type.
        
        Args:
            top_k: Number of results to return
            query_vector: Dense vector for semantic search
            query_text: Text for full-text search
            filter: Metadata filter
            alpha: Hybrid search weighting (0=text only, 1=vector only)
            fusion_method: Fusion algorithm ('rrf' or 'weighted')
            similarity_threshold: Minimum similarity score
            **kwargs: Provider-specific options
            
        Returns:
            List of VectorSearchResult objects
            
        Raises:
            ConfigurationError: If requested search type is disabled
            SearchError: If search fails
        """
        filter = self._convert_filters(filter)
        final_top_k = top_k if top_k is not None else self._config.default_top_k
        
        # Determine search type
        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None
        
        if is_hybrid:
            if not self._config.hybrid_search_enabled:
                raise ConfigurationError("Hybrid search is disabled.")
            return await self.hybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=final_top_k,
                filter=filter,
                alpha=alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        elif is_dense:
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled.")
            return await self.dense_search(
                query_vector=query_vector,
                top_k=final_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        elif is_full_text:
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled.")
            return await self.full_text_search(
                query_text=query_text,
                top_k=final_top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        else:
            raise ConfigurationError("Search requires at least one of 'query_vector' or 'query_text'.")

    def search_sync(
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
        """Master search method that dispatches to appropriate search type (sync)."""
        return self._run_async_from_sync(
            self.search(top_k, query_vector, query_text, filter, alpha, fusion_method, similarity_threshold, **kwargs)
        )

    async def dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Perform pure vector similarity search.
        
        Args:
            query_vector: Dense vector embedding
            top_k: Number of results
            filter: Metadata filter
            similarity_threshold: Minimum similarity score
            **kwargs: Provider-specific options
            
        Returns:
            List of VectorSearchResult objects sorted by similarity
            
        Raises:
            SearchError: If search fails
        """
        collection = await self._get_active_collection()
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        
        try:
            where_filter = self._convert_filters(filter)
            
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "distances", "embeddings", "documents"]
            )
            
            ids = results['ids'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            vectors = results['embeddings'][0] if results['embeddings'] else [None] * len(ids)
            chunks = results['documents'][0] if results['documents'] else [None] * len(ids)
            
            # Convert distances to scores
            max_dist = max(distances) if distances else 1.0
            
            filtered_results = []
            for i in range(len(ids)):
                # Score calculation based on distance metric
                if self._config.distance_metric == DistanceMetric.COSINE:
                    score = 1 - distances[i]
                elif self._config.distance_metric == DistanceMetric.EUCLIDEAN:
                    score = min(1.0, max(0.0, 1 - distances[i] / max_dist if max_dist > 0 else 1.0))
                elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
                    score = distances[i]
                else:
                    score = 1 - distances[i]
                
                if score >= final_similarity_threshold:
                    filtered_results.append(
                        VectorSearchResult(
                            id=ids[i],
                            score=score,
                            payload=metadatas[i],
                            vector=vectors[i],
                            text=chunks[i]
                        )
                    )
            
            return filtered_results
            
        except Exception as e:
            raise SearchError(f"An error occurred during dense search: {e}") from e

    def dense_search_sync(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Perform pure vector similarity search (sync)."""
        return self._run_async_from_sync(
            self.dense_search(query_vector, top_k, filter, similarity_threshold, **kwargs)
        )

    async def full_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Perform full-text search using ChromaDB's document filtering.
        
        Args:
            query_text: Text query
            top_k: Number of results
            filter: Metadata filter
            similarity_threshold: Minimum relevance score
            **kwargs: Provider-specific options
            
        Returns:
            List of VectorSearchResult objects sorted by relevance
            
        Raises:
            SearchError: If search fails
        """
        collection = await self._get_active_collection()
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold or 0.5
        
        where_document_filter = {"$contains": query_text}
        where_filter = self._convert_filters(filter)
        
        try:
            # Use where_document parameter for document filtering in Chroma
            results = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                where_document=where_document_filter,
                limit=top_k * 2,  # Get more to account for filtering
                include=["metadatas", "embeddings", "documents"]
            )
            
            # Calculate relevance scores using BM25-inspired algorithm
            query_terms = query_text.lower().split()
            scored_results = []
            
            for i in range(len(results['ids'])):
                document_text = results['documents'][i] if results['documents'] else ""
                if not document_text:
                    continue
                
                doc_lower = document_text.lower()
                doc_words = doc_lower.split()
                
                # Calculate term frequency
                term_count = 0
                for term in query_terms:
                    term_count += doc_lower.count(term)
                
                # Calculate BM25-inspired relevance score
                doc_length = len(doc_words)
                if doc_length > 0 and term_count > 0:
                    # Match ratio: fraction of query terms that appear in document
                    matched_terms = sum(1 for term in query_terms if term in doc_lower)
                    match_ratio = matched_terms / len(query_terms)
                    
                    # Term density with logarithmic scaling
                    term_density = term_count / doc_length
                    tf_score = math.log(1 + term_density * 5) / math.log(6)  # Normalize to ~0-1 range
                    
                    # Final score: 70% TF density + 30% match coverage
                    score = (tf_score * 0.7 + match_ratio * 0.3)
                    score = min(1.0, max(0.0, score))
                else:
                    score = 0.0
                
                if score >= final_similarity_threshold:
                    scored_results.append((score, i))
            
            # Sort by score and return top_k
            scored_results.sort(key=lambda x: x[0], reverse=True)
            scored_results = scored_results[:top_k]
            
            filtered_results = []
            for score, i in scored_results:
                # Safely extract vector and text
                vector = results['embeddings'][i] if results.get('embeddings') is not None else None
                text = results['documents'][i] if results.get('documents') is not None else None
                
                filtered_results.append(
                    VectorSearchResult(
                        id=results['ids'][i],
                        score=score,
                        payload=results['metadatas'][i],
                        vector=vector,
                        text=text
                    )
                )
            
            return filtered_results
            
        except Exception as e:
            raise SearchError(f"An error occurred during full-text search: {e}") from e

    def full_text_search_sync(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Perform full-text search using ChromaDB's document filtering (sync)."""
        return self._run_async_from_sync(
            self.full_text_search(query_text, top_k, filter, similarity_threshold, **kwargs)
        )

    def _reciprocal_rank_fusion(self, results_lists: List[List[VectorSearchResult]], k: int = 60) -> dict:
        """
        Fuse multiple result lists using Reciprocal Rank Fusion (RRF).
        
        Args:
            results_lists: List of search result lists to fuse
            k: RRF constant (default 60)
            
        Returns:
            Dictionary mapping document IDs to fused scores
        """
        fused_scores = {}
        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = str(doc.id)
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank + 1)
        return fused_scores
    
    def _weighted_fusion(self, dense_results: List[VectorSearchResult], ft_results: List[VectorSearchResult], alpha: float) -> dict:
        """
        Fuse dense and full-text results using weighted scoring.
        
        Args:
            dense_results: Results from dense vector search
            ft_results: Results from full-text search
            alpha: Weight for dense results (0-1), full-text gets (1-alpha)
            
        Returns:
            Dictionary mapping document IDs to fused scores
        """
        fused_scores = {}
        
        # Add dense results with weight alpha
        for doc in dense_results:
            fused_scores[str(doc.id)] = doc.score * alpha
        
        # Add full-text results with weight (1 - alpha)
        for doc in ft_results:
            doc_id = str(doc.id)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += doc.score * (1 - alpha)
        
        return fused_scores

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
        Perform hybrid search combining dense vector and full-text search.
        
        Args:
            query_vector: Dense vector for semantic search
            query_text: Text for keyword search
            top_k: Number of final results
            filter: Metadata filter
            alpha: Weighting factor (0=text only, 1=vector only)
            fusion_method: Algorithm for fusing results ('rrf' or 'weighted')
            similarity_threshold: Minimum similarity threshold
            **kwargs: Provider-specific options
            
        Returns:
            List of VectorSearchResult objects with hybrid scores
            
        Raises:
            SearchError: If search fails
        """
        final_alpha = alpha if alpha is not None else self._config.default_hybrid_alpha or 0.5
        final_fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method or 'weighted'
        
        try:
            # Get more candidates for better fusion results
            candidate_k = max(top_k * 2, 20)
            
            # Perform both searches in parallel
            dense_results, ft_results = await asyncio.gather(
                self.dense_search(query_vector, candidate_k, filter, similarity_threshold, **kwargs),
                self.full_text_search(query_text, candidate_k, filter, similarity_threshold, **kwargs)
            )
            
            # Fuse results
            fused_scores: dict
            if final_fusion_method == 'rrf':
                fused_scores = self._reciprocal_rank_fusion([dense_results, ft_results])
            elif final_fusion_method == 'weighted':
                fused_scores = self._weighted_fusion(dense_results, ft_results, final_alpha)
            else:
                raise ConfigurationError(f"Unknown fusion_method: {final_fusion_method}")
            
            # Get top_k document IDs
            reranked_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)[:top_k]
            
            if not reranked_ids:
                return []
            
            # Fetch full documents
            final_results = await self.fetch(ids=reranked_ids)
            
            # Update scores with fused scores
            updated_results = []
            for result in final_results:
                updated_result = VectorSearchResult(
                    id=result.id,
                    score=fused_scores.get(str(result.id), 0.0),
                    payload=result.payload,
                    vector=result.vector,
                    text=result.text
                )
                updated_results.append(updated_result)
            
            # Sort by fused score
            updated_results.sort(key=lambda x: x.score, reverse=True)
            
            return updated_results
            
        except Exception as e:
            raise SearchError(f"An error occurred during hybrid search: {e}") from e

    def hybrid_search_sync(
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
        """Perform hybrid search combining dense vector and full-text search (sync)."""
        return self._run_async_from_sync(
            self.hybrid_search(query_vector, query_text, top_k, filter, alpha, fusion_method, similarity_threshold, **kwargs)
        )
