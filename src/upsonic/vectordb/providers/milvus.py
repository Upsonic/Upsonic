from typing import Any, Dict, List, Optional, Union, Literal, Generator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Cannot infer schema from data.*")

from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    WeightedRanker,
    RRFRanker,
    AnnSearchRequest
)
from pymilvus.exceptions import MilvusException


from upsonic.vectordb.base import BaseVectorDBProvider

from upsonic.vectordb.config import (
    Config,
    Mode,
    DistanceMetric,
    IndexType,
    WriteConsistency,
    HNSWTuningConfig,
    IVFTuningConfig,
    FlatTuningConfig,
    PayloadIndexConfig,
    ConsistencyLevel
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


class MilvusProvider(BaseVectorDBProvider):
    """
    A vector database provider that integrates with Milvus.

    This class implements the BaseVectorDBProvider interface and translates the
    framework's abstract configuration and commands into specific operations
    for a Milvus database, whether it's running locally, embedded (Lite),
    or in the cloud (Zilliz).
    """

    _DISTANCE_MAP = {
        DistanceMetric.COSINE: "COSINE",
        DistanceMetric.EUCLIDEAN: "L2",
        DistanceMetric.DOT_PRODUCT: "IP"
    }

    _PAYLOAD_TYPE_MAP = {
        'keyword': DataType.VARCHAR,
        'text': DataType.VARCHAR,
        'integer': DataType.INT64,
        'float': DataType.FLOAT,
        'boolean': DataType.BOOL,
        'geo': DataType.UNKNOWN
    }

    _CONSISTENCY_MAP = {
        WriteConsistency.STRONG: ConsistencyLevel.STRONG,
        WriteConsistency.EVENTUAL: ConsistencyLevel.EVENTUALLY
    }


    def __init__(self, config: Config):
        """
        Initializes the MilvusProvider.

        Args:
            config: A validated Config object.

        Raises:
            ImportError: If the `pymilvus` package is not installed.
            ConfigurationError: If the provided config is incompatible with Milvus.
        """
        if Collection is None:
            raise ImportError(
                "The 'pymilvus' package is required to use MilvusProvider. "
                "Please install it with 'pip install pymilvus'."
            )
        
        super().__init__(config)
        self._validate_config()
        self._collection: Optional[Collection] = None


    def _validate_config(self) -> None:
        """
        Performs Milvus-specific validation on the provided configuration.
        This is a critical step to fail early if the config is invalid.
        """
        if self._config.core.mode == Mode.EMBEDDED and not self._config.core.db_path:
            raise ConfigurationError(
                "For 'embedded' mode with Milvus (Milvus Lite), a `db_path` "
                "must be specified in CoreConfig (e.g., './milvus.db')."
            )

        if self._config.core.mode == Mode.CLOUD and not self._config.core.host:
            raise ConfigurationError(
                "For 'cloud' mode with Milvus, a `host` must be specified in CoreConfig."
            )

        if self._config.core.distance_metric not in self._DISTANCE_MAP:
            raise ConfigurationError(
                f"Distance metric '{self._config.core.distance_metric}' is not "
                f"supported by the MilvusProvider. Supported metrics are: "
                f"{list(self._DISTANCE_MAP.keys())}"
            )
        if self._config.indexing.create_dense_index and self._config.core.vector_size is None:
             raise ConfigurationError(
                  "A `vector_size` must be specified in CoreConfig when 'create_dense_index' is True."
             )
    
        
        print("MilvusProvider: Configuration validated successfully.")

    def _sanitize_collection_name(self, collection_name: str) -> str:
        """
        Sanitizes collection name to conform to Milvus naming requirements.
        Collection names can only contain numbers, letters, and underscores.
        """
        import re
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', collection_name)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'col_' + sanitized
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'default_collection'
        return sanitized

    @property
    def _sanitized_collection_name(self) -> str:
        """Get the sanitized collection name."""
        return self._sanitize_collection_name(self._config.core.collection_name)

    def _get_connection_alias(self) -> str:
        """Generates a unique but deterministic connection alias for pymilvus."""
        db_path = self._config.core.db_path or "memory"
        db_name = db_path.split('/')[-1].replace('.db', '') if '/' in db_path else db_path
        unique_id = str(hash(db_path))[-8:]
        # Use sanitized collection name for connection alias
        return f"upsonic_milvus_{self._sanitized_collection_name}_{db_name}_{unique_id}"

    def _build_connection_params(self) -> Dict[str, Any]:
        """
        Translates the CoreConfig into a dictionary of connection arguments
        for the `pymilvus.connections.connect()` function.
        """
        mode = self._config.core.mode
        
        if mode in [Mode.EMBEDDED, Mode.IN_MEMORY]:
            db_path = self._config.core.db_path or ":memory:"
            if mode == Mode.EMBEDDED and db_path != ":memory:" and not db_path.endswith('.db'):
                db_path = db_path + '.db'
            return {"uri": db_path}

        elif mode == Mode.CLOUD:
            if not self._config.core.host:
                raise ConfigurationError(f"A `host` is required for '{mode.value}' mode.")
            
            params = {}
            
            if self._config.core.host.startswith('http'):
                params["uri"] = self._config.core.host
            else:
                protocol = "https" if self._config.core.use_tls else "http"
                port = self._config.core.port or (443 if self._config.core.use_tls else 19530)
                params["uri"] = f"{protocol}://{self._config.core.host}:{port}"
            
            if self._config.core.use_tls:
                params["secure"] = True
            
            if self._config.core.api_key:
                api_key_value = self._config.core.api_key.get_secret_value()
                
                if ":" in api_key_value:
                    if api_key_value.startswith("db_") or len(api_key_value.split(":")) == 2:
                        params["token"] = api_key_value
                    else:
                        params["token"] = api_key_value
                else:
                    params["token"] = api_key_value
            
            if hasattr(self._config, 'advanced') and hasattr(self._config.advanced, 'namespace') and self._config.advanced.namespace:
                params["db_name"] = self._config.advanced.namespace
            
            return params

        elif mode == Mode.LOCAL:
            if not self._config.core.host:
                raise ConfigurationError(f"A `host` is required for '{mode.value}' mode.")
            
            params = {
                "host": self._config.core.host,
                "port": str(self._config.core.port or 19530),
            }
            if self._config.core.api_key:
                api_key_value = self._config.core.api_key.get_secret_value()
                if ":" in api_key_value:
                    user, password = api_key_value.split(":", 1)
                    params["user"] = user
                    params["password"] = password
                else:
                    params["token"] = api_key_value

            if self._config.core.use_tls:
                params["secure"] = True

            return params
        
        else:
            raise ConfigurationError(f"Unsupported mode '{mode.value}' for MilvusProvider.")


    def _build_field_schema(self) -> List[FieldSchema]:
        """
        Constructs the list of `pymilvus.FieldSchema` objects that define the
        collection's structure, now dynamically based on CoreConfig and IndexingConfig flags.
        """
        fields = []

        pk_field = FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=65535
        )
        fields.append(pk_field)


        chunk_field = FieldSchema(
            name="chunk",
            dtype=DataType.VARCHAR,
            max_length=65535
        )
        fields.append(chunk_field)
        
        if self._config.indexing.create_dense_index:
            dense_vector_field = FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._config.core.vector_size
            )
            fields.append(dense_vector_field)
        
        if self._config.indexing.create_sparse_index:
            sparse_vector_field = FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR
            )
            fields.append(sparse_vector_field)
        
        if self._config.indexing.payload_indexes:
            reserved_names = ["id", "dense_vector", "sparse_vector", "chunk"]
            for payload_index in self._config.indexing.payload_indexes:
                field_name = payload_index.field_name
                if field_name in reserved_names:
                    warnings.warn(f"Payload field name '{field_name}' is reserved and will be ignored.")
                    continue 
                
                field_type = self._PAYLOAD_TYPE_MAP.get(payload_index.field_schema_type)
                if field_type == DataType.UNKNOWN:
                    warnings.warn(f"Payload field '{field_name}' has an unsupported schema type "
                                  f"'{payload_index.field_schema_type}' for Milvus and will be ignored.")
                    continue
                
                field_params = {"name": field_name, "dtype": field_type}
                if field_type == DataType.VARCHAR:
                    field_params["max_length"] = 65535

                fields.append(FieldSchema(**field_params))

        return fields

    def _build_index_params(self) -> Dict[str, Any]:
        """
        Translates the IndexingConfig into the `index_params` dictionary
        required by Milvus's `collection.create_index()` method.
        """
        index_cfg = self._config.indexing.index_config
        metric_type = self._DISTANCE_MAP[self._config.core.distance_metric]

        if self._config.core.mode == Mode.EMBEDDED:
            if isinstance(index_cfg, HNSWTuningConfig):
                return {
                    "metric_type": metric_type,
                    "index_type": "FLAT",
                    "params": {},
                }
            elif isinstance(index_cfg, IVFTuningConfig):
                return {
                    "metric_type": metric_type,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": index_cfg.nlist},
                }
            elif isinstance(index_cfg, FlatTuningConfig):
                return {
                    "metric_type": metric_type,
                    "index_type": "FLAT",
                    "params": {},
                }
        else:
            if isinstance(index_cfg, HNSWTuningConfig):
                return {
                    "metric_type": metric_type,
                    "index_type": "HNSW",
                    "params": {
                        "M": index_cfg.m,
                        "efConstruction": index_cfg.ef_construction,
                    },
                }
            elif isinstance(index_cfg, IVFTuningConfig):
                return {
                    "metric_type": metric_type,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": index_cfg.nlist},
                }
            elif isinstance(index_cfg, FlatTuningConfig):
                return {
                    "metric_type": metric_type,
                    "index_type": "FLAT",
                    "params": {},
                }
        
        raise ConfigurationError(f"Unsupported index type: {index_cfg.index_type}")


    def connect(self) -> None:
        """
        Establishes a connection to the Milvus database.

        This method uses the connection parameters from `self._config.core`
        to initialize the database client and verify the connection. It also
        pre-loads a handle to the collection if it already exists.

        Raises:
            VectorDBConnectionError: If the connection fails.
        """
        if self._is_connected:
            return

        alias = self._get_connection_alias()
        params = self._build_connection_params()
        
        try:
            print(f"MilvusProvider: Attempting to connect with alias '{alias}' and params: {params}")
            connections.connect(alias=alias, **params)
            self._is_connected = True
            print("MilvusProvider: Connection successful.")

            try:
                if utility.has_collection(self._sanitized_collection_name, using=alias):
                    self._collection = Collection(
                        name=self._sanitized_collection_name, 
                        using=alias
                    )
                    print(f"MilvusProvider: Attached to existing collection '{self._sanitized_collection_name}'.")
            except MilvusException as e:
                print(f"MilvusProvider: Could not check collection existence: {e}")

        except MilvusException as e:
            self._is_connected = False
            raise VectorDBConnectionError(f"Failed to connect to Milvus: {e}")

    def disconnect(self) -> None:
        """
        Gracefully terminates the connection to the Milvus database.
        Resets the internal state of the provider.
        """
        if not self._is_connected:
            return

        alias = self._get_connection_alias()
        try:
            connections.disconnect(alias)
            print(f"MilvusProvider: Disconnected from alias '{alias}'.")
        except MilvusException as e:
            print(f"Warning: Error during Milvus disconnection: {e}")
        finally:
            self._is_connected = False
            self._collection = None


    def is_ready(self) -> bool:
        """
        Performs a health check to ensure the database is responsive.

        Returns:
            True if the database is connected and responsive, False otherwise.
        """
        if not self._is_connected:
            return False

        alias = self._get_connection_alias()
        try:
            addr = connections.get_connection_addr(alias)
            return addr is not None
        except MilvusException:
            return False

    def collection_exists(self) -> bool:
        """
        Checks if the collection specified in the config already exists.

        Returns:
            True if the collection exists, False otherwise.
        
        Raises:
            VectorDBConnectionError: If not connected to the database.
        """
        if not self.is_ready():
            raise VectorDBConnectionError("Not connected to Milvus. Call connect() first.")
        
        try:
            return utility.has_collection(
                self._sanitized_collection_name, 
                using=self._get_connection_alias()
            )
        except MilvusException as e:
            raise VectorDBError(f"Failed to check if collection exists: {e}")

    def delete_collection(self) -> None:
        """
        Permanently deletes the collection specified in `self._config.core.collection_name`.

        Raises:
            VectorDBConnectionError: If not connected to the database.
            CollectionDoesNotExistError: If the collection to be deleted does not exist.
            VectorDBError: For other Milvus-related errors.
        """
        if not self.is_ready():
            raise VectorDBConnectionError("Not connected to Milvus. Call connect() first.")

        collection_name = self._sanitized_collection_name
        
        if not self.collection_exists():
             raise CollectionDoesNotExistError(f"Collection '{collection_name}' does not exist.")

        try:
            utility.drop_collection(collection_name, using=self._get_connection_alias())
            self._collection = None
            print(f"MilvusProvider: Successfully deleted collection '{collection_name}'.")
        except MilvusException as e:
            raise VectorDBError(f"Failed to delete collection '{collection_name}': {e}")


    def create_collection(self) -> None:
        """
        Creates the collection and all specified indexes based on the full config.

        This method now dynamically builds the schema and then proceeds to create
        up to three different types of indexes (dense, sparse, and full-text)
        based on the flags set in the IndexingConfig.
        """
        if not self.is_ready():
            raise VectorDBConnectionError("Not connected to Milvus. Call connect() first.")
        
        collection_name = self._sanitized_collection_name
        alias = self._get_connection_alias()

        if self.collection_exists():
            if self._config.core.recreate_if_exists:
                print(f"MilvusProvider: Collection '{collection_name}' exists and will be recreated.")
                self.delete_collection()
            else:
                print(f"MilvusProvider: Collection '{collection_name}' already exists. Skipping creation.")
                if self._collection is None:
                    self._collection = Collection(name=collection_name, using=alias)
                return

        try:
            fields = self._build_field_schema()
            schema = CollectionSchema(fields, description=f"Upsonic collection: {collection_name}", enable_dynamic_field=True)
            consistency_level = self._CONSISTENCY_MAP[self._config.data_management.write_consistency]
            
            self._collection = Collection(
                name=collection_name, schema=schema, using=alias, consistency_level=consistency_level
            )
            print(f"MilvusProvider: Collection '{collection_name}' created successfully with dynamic schema.")

            if self._config.indexing.create_dense_index:
                index_params = self._build_index_params()
                print(f"MilvusProvider: Creating dense vector index on field 'dense_vector' with params: {index_params}")
                self._collection.create_index(field_name="dense_vector", index_params=index_params)
                print("MilvusProvider: Dense vector index created successfully.")

            if self._config.indexing.create_sparse_index:
                sparse_index_params = {
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP"
                }
                print(f"MilvusProvider: Creating sparse vector index on field 'sparse_vector' with params: {sparse_index_params}")
                self._collection.create_index(field_name="sparse_vector", index_params=sparse_index_params)
                print("MilvusProvider: Sparse vector index created successfully.")

            if self._config.indexing.payload_indexes:
                for payload_config in self._config.indexing.payload_indexes:
                    if payload_config.enable_full_text_index:
                        field_name = payload_config.field_name
                        print(f"MilvusProvider: Creating full-text index on payload field '{field_name}'.")
                        self._collection.create_index(field_name=field_name, index_params={"index_type": "FULL_TEXT"})
                        print(f"MilvusProvider: Full-text index on '{field_name}' created successfully.")

            print("MilvusProvider: Loading collection into memory...")
            self._collection.load()
            print("MilvusProvider: Collection loaded successfully and is ready for operations.")

        except MilvusException as e:
            if self.collection_exists():
                utility.drop_collection(collection_name, using=alias)
            self._collection = None
            raise VectorDBError(f"Failed to create and initialize collection '{collection_name}': {e}")

    def _ensure_collection_handle(self) -> None:
        """A private helper to ensure the self._collection object is valid."""
        if not self.is_ready():
            raise VectorDBConnectionError("Not connected to Milvus. Call connect() first.")
        if self._collection is None:
            raise CollectionDoesNotExistError(f"Collection '{self._sanitized_collection_name}' handle is not loaded. Was it created?")

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]], **kwargs) -> None:
        """
        Adds or updates data, now with support for sparse vectors.

        This method validates the presence of required vector types based on the
        collection's schema, prepares the data in a row-based format (list of dicts),
        and uploads it to Milvus in batches.

        Args:
            vectors: A list of dense vector embeddings. Can be None if the collection
                     is sparse-only.
            payloads: A list of corresponding metadata objects.
            ids: A list of unique identifiers for each data point.
            **kwargs: Must contain `sparse_vectors: List[Dict[int, float]]` if the
                      collection was created with `create_sparse_index=True`.

        Raises:
            UpsertError: If data ingestion fails, or if the required vector types
                         are missing or have mismatched lengths.
        """
        self._ensure_collection_handle()

        if not ids:
            print("MilvusProvider: Upsert called with empty 'ids' list. No action taken.")
            return

        sparse_vectors = kwargs.get("sparse_vectors")

        num_ids = len(ids)
        if len(payloads) != num_ids:
            raise UpsertError(f"Length mismatch: `ids` has {num_ids} elements while `payloads` has {len(payloads)}.")

        if self._config.indexing.create_dense_index:
            if vectors is None:
                raise UpsertError("Collection requires dense vectors, but `vectors` argument was None.")
            if len(vectors) != num_ids:
                raise UpsertError(f"Length mismatch: `ids` has {num_ids} elements while `vectors` has {len(vectors)}.")

        if self._config.indexing.create_sparse_index:
            if sparse_vectors is None:
                raise UpsertError("Collection requires sparse vectors, but `sparse_vectors` kwarg was not provided.")
            if len(sparse_vectors) != num_ids:
                raise UpsertError(f"Length mismatch: `ids` has {num_ids} elements while `sparse_vectors` has {len(sparse_vectors)}.")
        
        try:
            entities_to_upsert = []
            for i in range(num_ids):
                entity = {}
                entity['id'] = str(ids[i])
                entity['chunk'] = str(chunks[i])
                entity.update(payloads[i])
                
                if self._config.indexing.create_dense_index:
                    entity['dense_vector'] = vectors[i]
                if self._config.indexing.create_sparse_index:
                    entity['sparse_vector'] = sparse_vectors[i]
                
                entities_to_upsert.append(entity)

            batch_size = self._config.data_management.batch_size
            total = len(entities_to_upsert)
            print(f"MilvusProvider: Starting upsert of {total} entities in batches of {batch_size}.")

            for i in range(0, total, batch_size):
                batch = entities_to_upsert[i : i + batch_size]
                print(f"MilvusProvider: Upserting batch {i//batch_size + 1}/{ -(-total // batch_size) }, size: {len(batch)}")
                self._collection.upsert(batch)

            self._collection.flush()
            print("MilvusProvider: Upsert complete and data flushed successfully.")

        except MilvusException as e:
            raise UpsertError(f"Failed to upsert data into Milvus: {e}")
        except Exception as e:
            raise UpsertError(f"An unexpected error occurred during data preparation for upsert: {e}")


    def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers.

        Args:
            ids: A list of specific IDs to remove.

        Raises:
            VectorDBError: If the deletion fails.
            VectorDBConnectionError: If not connected to the database.
        """
        self._ensure_collection_handle()
        
        if not ids:
            return
        
        str_ids = [f"'{str(id_val)}'" for id_val in ids]
        expr = f"id in [{','.join(str_ids)}]"
        
        try:
            print(f"MilvusProvider: Deleting {len(ids)} entities with expression: {expr}")
            self._collection.delete(expr)
            self._collection.flush()
            print("MilvusProvider: Deletion complete and data flushed.")
        except MilvusException as e:
            raise VectorDBError(f"Failed to delete entities from Milvus: {e}")
    
    
    def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs.

        Args:
            ids: A list of IDs to retrieve the full records for.

        Returns:
            A list of VectorSearchResult objects containing the fetched data.
            
        Raises:
            VectorDBError: If the fetch operation fails.
        """
        self._ensure_collection_handle()

        if not ids:
            return []
        
        str_ids = [f"'{str(id_val)}'" for id_val in ids]
        expr = f"id in [{','.join(str_ids)}]"
        
        try:
            results = self._collection.query(
                expr=expr,
                output_fields=["*"]
            )
            
            search_results = []
            for res in results:
                payload = {k: v for k, v in res.items() if k not in ['id', 'dense_vector', 'sparse_vector', 'chunk']}
                search_results.append(
                    VectorSearchResult(
                        id=res['id'],
                        score=1.0,
                        payload=payload,
                        vector=res.get('dense_vector'),
                        text=res.get('chunk')
                    )
                )
            return search_results
        except MilvusException as e:
            raise VectorDBError(f"Failed to fetch entities from Milvus: {e}")

    def _build_filter_expression(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[str]:
        """Translates a dictionary-based filter into a Milvus expression string."""
        if not filter_dict:
            return None
        
        clauses = []
        for field, value in filter_dict.items():
            if isinstance(value, dict):
                if len(value) != 1:
                    raise SearchError(f"Filter for field '{field}' must have exactly one operator.")
                op, val = list(value.items())[0]
                if op == "$in":
                    formatted_list = [repr(item) for item in val]
                    clauses.append(f"{field} in [{','.join(formatted_list)}]")
                elif op == "$eq":
                    clauses.append(f"{field} == {repr(val)}")
                elif op == "$ne":
                    clauses.append(f"{field} != {repr(val)}")
                else:
                    raise SearchError(f"Unsupported filter operator '{op}' for field '{field}'.")
            else:
                clauses.append(f"{field} == {repr(value)}")

        return " and ".join(clauses) if clauses else None

    def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Master search dispatcher. Routes queries to the appropriate specialized
        search method based on provided arguments and collection configuration.
        """
        k = top_k if top_k is not None else self._config.search.default_top_k
        if k is None:
            raise ConfigurationError("`top_k` must be provided for search or set as `default_top_k`.")

        final_filter = filter if filter is not None else self._config.search.filter

        is_hybrid = query_vector is not None and query_text is not None
        is_dense = query_vector is not None and query_text is None
        is_full_text = query_vector is None and query_text is not None
        
        if is_hybrid:
            if not self._config.search.hybrid_search_enabled: raise ConfigurationError("Hybrid search is disabled.")
            return self.hybrid_search(query_vector, query_text, k, final_filter, alpha, fusion_method, similarity_threshold, **kwargs)
        elif is_dense:
            if not self._config.search.dense_search_enabled: raise ConfigurationError("Dense search is disabled.")
            return self.dense_search(query_vector, k, final_filter, similarity_threshold, **kwargs)
        elif is_full_text:
            if not self._config.search.full_text_search_enabled: raise ConfigurationError("Full-text search is disabled.")
            return self.full_text_search(query_text, k, final_filter, similarity_threshold, **kwargs)
        else:
            raise SearchError("Search requires at least one of 'query_vector' or 'query_text'.")

    def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """Performs a pure dense vector similarity search."""
        if not self._config.indexing.create_dense_index:
            raise ConfigurationError("Dense search is not possible; no dense index was created for this collection.")
        self._ensure_collection_handle()
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)
        
        params = {"metric_type": self._DISTANCE_MAP[self._config.core.distance_metric], "params": {}}
        index_cfg = self._config.indexing.index_config
        if isinstance(index_cfg, HNSWTuningConfig):
            params["params"]["ef"] = kwargs.get("ef_search", self._config.search.default_ef_search or 128)
        elif isinstance(index_cfg, IVFTuningConfig):
            params["params"]["nprobe"] = kwargs.get("nprobe", self._config.search.default_nprobe or 10)

        filter_expr = self._build_filter_expression(filter)
        
        try:
            results = self._collection.search(
                data=[query_vector],
                anns_field="dense_vector",
                param=params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["*"]
            )
            
            search_results = []
            for hit in results[0]:
                entity_data = hit.entity['entity']
                
                payload = {k: v for k, v in entity_data.items() if k not in ['id', 'dense_vector', 'sparse_vector', 'chunk']}
                
                # Convert distance to similarity score based on distance metric
                # Note: After testing, Milvus with COSINE metric returns cosine similarity directly
                if self._config.core.distance_metric == DistanceMetric.COSINE:
                    # For Milvus cosine metric: hit.distance is actually cosine similarity
                    score = hit.distance
                elif self._config.core.distance_metric == DistanceMetric.EUCLIDEAN:
                    # For Euclidean distance, convert to similarity-like score
                    score = 1.0 / (1.0 + hit.distance)
                elif self._config.core.distance_metric == DistanceMetric.DOT_PRODUCT:
                    # For dot product, higher values are better (already similarity-like)
                    score = hit.distance
                else:
                    # Default to treating as similarity
                    score = hit.distance
                
                if score >= final_similarity_threshold:
                    search_results.append(VectorSearchResult(
                        id=hit.id, 
                        score=score, 
                        payload=payload, 
                        vector=entity_data.get('dense_vector'), 
                        text=entity_data.get('chunk')
                    ))
            return search_results
        except MilvusException as e:
            raise SearchError(f"Failed to perform dense search in Milvus: {e}")

    def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """Performs true full-text search using the best available index."""
        self._ensure_collection_handle()
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)
        
        filter_expr = self._build_filter_expression(filter)

        try:
            if self._config.indexing.create_sparse_index:
                query_sparse_vector = kwargs.get("query_sparse_vector")
                if query_sparse_vector is None:
                    raise SearchError("`full_text_search` on a sparse-indexed collection requires `query_sparse_vector` in kwargs.")
                
                results = self._collection.search(
                    data=[query_sparse_vector],
                    anns_field="sparse_vector",
                    param={"metric_type": "IP"},
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=["*"]
                )
            else:
                search_field = kwargs.get("chunk")
                if not search_field:
                    raise ConfigurationError("`full_text_search` on this collection requires `search_field` in kwargs.")
                
                sanitized_query_text = query_text.replace("'", "''")
                text_expr = f"{search_field} like '%{sanitized_query_text}%'"
                final_expr = f"({text_expr}) and ({filter_expr})" if filter_expr else text_expr
                
                hits = self._collection.query(expr=final_expr, limit=top_k, output_fields=["*"])
                filtered_results = []
                for hit in hits:
                    score = 1.0  # Default score for full-text search
                    if score >= final_similarity_threshold:
                        filtered_results.append(VectorSearchResult(
                            id=hit['id'], 
                            score=score, 
                            payload={k: v for k, v in hit.items() if k not in ['id', 'dense_vector', 'sparse_vector', 'chunk']}, 
                            text=hit['chunk']
                        ))
                return filtered_results

            search_results = []
            for hit in results[0]:
                payload = {k: v for k, v in hit.entity.items() if k not in ['id', 'dense_vector', 'sparse_vector']}
                
                # Apply same scoring logic as dense_search
                if self._config.core.distance_metric == DistanceMetric.COSINE:
                    score = hit.distance  # Milvus returns cosine similarity directly
                elif self._config.core.distance_metric == DistanceMetric.EUCLIDEAN:
                    score = 1.0 / (1.0 + hit.distance)
                elif self._config.core.distance_metric == DistanceMetric.DOT_PRODUCT:
                    score = hit.distance
                else:
                    score = hit.distance
                
                if score >= final_similarity_threshold:
                    search_results.append(VectorSearchResult(id=hit.id, score=score, payload=payload, vector=hit.entity.get('sparse_vector')))
            return search_results

        except MilvusException as e:
            raise SearchError(f"Failed to perform full-text search in Milvus: {e}")

    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[Literal['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """Performs native hybrid search using Milvus's multi-vector search and server-side reranking."""
        self._ensure_collection_handle()
        
        if not (self._config.indexing.create_dense_index and self._config.indexing.create_sparse_index):
            raise ConfigurationError("Hybrid search is only possible on collections with both dense and sparse indexes.")
        
        query_sparse_vector = kwargs.get("query_sparse_vector")
        if query_sparse_vector is None:
            raise SearchError("Hybrid search requires `query_sparse_vector` in kwargs.")

        fusion = fusion_method or self._config.search.default_fusion_method or 'weighted'
        final_alpha = alpha if alpha is not None else self._config.search.default_hybrid_alpha or 0.5
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else (self._config.search.default_similarity_threshold if self._config.search.default_similarity_threshold is not None else 0.5)

        filter_expr = self._build_filter_expression(filter)

        dense_params = {"metric_type": self._DISTANCE_MAP[self._config.core.distance_metric], "params": {}}
        if isinstance(self._config.indexing.index_config, HNSWTuningConfig):
             dense_params["params"]["ef"] = kwargs.get("ef_search", self._config.search.default_ef_search or 128)
        dense_req_params = {
            "data": [query_vector], "anns_field": "dense_vector", "param": dense_params, "limit": top_k
        }
        dense_req = AnnSearchRequest(**dense_req_params)
        sparse_req_params = {
            "data": [query_sparse_vector], "anns_field": "sparse_vector", "param": {"metric_type": "IP"}, "limit": top_k
        }
        sparse_req = AnnSearchRequest(**sparse_req_params)

        if fusion == 'rrf':
            reranker = RRFRanker()
        else:
            reranker = WeightedRanker(final_alpha, 1.0 - final_alpha)

        try:
            hybrid_kwargs = {
                "reqs": [dense_req, sparse_req],
                "rerank": reranker,
                "limit": top_k,
                "output_fields": ["*"]
            }
            if filter_expr:
                hybrid_kwargs["expr"] = filter_expr
            
            results = self._collection.hybrid_search(**hybrid_kwargs)

            search_results = []
            for hit in results[0]:
                payload = {k: v for k, v in hit.entity['entity'].items() if k not in ['id', 'dense_vector', 'sparse_vector', 'chunk']}
                vector = hit.entity['entity'].get('dense_vector')
                text = hit.entity['entity'].get('chunk')
                
                # Apply same scoring logic as dense_search
                if self._config.core.distance_metric == DistanceMetric.COSINE:
                    score = hit.distance  # Milvus returns cosine similarity directly
                elif self._config.core.distance_metric == DistanceMetric.EUCLIDEAN:
                    score = 1.0 / (1.0 + hit.distance)
                elif self._config.core.distance_metric == DistanceMetric.DOT_PRODUCT:
                    score = hit.distance
                else:
                    score = hit.distance
                
                if score >= final_similarity_threshold:
                    search_results.append(VectorSearchResult(id=hit.id, score=score, payload=payload, vector=vector, text=text))
            return search_results
 
        except MilvusException as e:
            raise SearchError(f"Failed to perform hybrid search in Milvus: {e}")