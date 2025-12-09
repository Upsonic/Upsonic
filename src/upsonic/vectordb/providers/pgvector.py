"""
PgVector Provider Implementation

A comprehensive, high-level vector database provider for PostgreSQL with pgvector extension.
Supports async operations, flexible metadata management, advanced indexing, and hybrid search.

This implementation follows the BaseVectorDBProvider interface and integrates best practices
from both SQLAlchemy and pgvector for optimal performance and flexibility.
"""

import asyncio
import json
from hashlib import md5
from math import sqrt
from typing import Any, Dict, List, Optional, Union, Literal, cast, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from sqlalchemy import (
        Column, String, Text, Integer, BigInteger, Float, Boolean, DateTime, 
        Index, MetaData, Table, create_engine, text, select, 
        delete as sa_delete, update as sa_update, func, desc
    )
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session, scoped_session, sessionmaker
    from sqlalchemy.inspection import inspect
    from sqlalchemy.sql.expression import bindparam
    from pgvector.sqlalchemy import Vector

try:
    from sqlalchemy import (
        Column, String, Text, Integer, BigInteger, Float, Boolean, DateTime, 
        Index, MetaData, Table, create_engine, text, select, 
        delete as sa_delete, update as sa_update, func, desc
    )
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session, scoped_session, sessionmaker
    from sqlalchemy.inspection import inspect
    from sqlalchemy.sql.expression import bindparam
    from pgvector.sqlalchemy import Vector
    _PGVECTOR_AVAILABLE = True
except ImportError:
    Column = None  # type: ignore
    String = None  # type: ignore
    Text = None  # type: ignore
    Integer = None  # type: ignore
    BigInteger = None  # type: ignore
    Float = None  # type: ignore
    Boolean = None  # type: ignore
    DateTime = None  # type: ignore
    Index = None  # type: ignore
    MetaData = None  # type: ignore
    Table = None  # type: ignore
    create_engine = None  # type: ignore
    text = None  # type: ignore
    select = None  # type: ignore
    sa_delete = None  # type: ignore
    sa_update = None  # type: ignore
    func = None  # type: ignore
    desc = None  # type: ignore
    postgresql = None  # type: ignore
    Engine = None  # type: ignore
    Session = None  # type: ignore
    scoped_session = None  # type: ignore
    sessionmaker = None  # type: ignore
    inspect = None  # type: ignore
    bindparam = None  # type: ignore
    Vector = None  # type: ignore
    _PGVECTOR_AVAILABLE = False

from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.vectordb.config import PgVectorConfig, HNSWIndexConfig, IVFIndexConfig, DistanceMetric
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.utils.logging_config import get_logger
from upsonic.utils.package.exception import (
    VectorDBConnectionError,
    VectorDBError,
    CollectionDoesNotExistError,
    UpsertError,
    SearchError,
    ConfigurationError
)

logger = get_logger(__name__)


class PgVectorProvider(BaseVectorDBProvider):
    """
    PostgreSQL + pgvector provider with comprehensive features:
    
    - Async-first architecture for high performance
    - Flexible metadata management with custom fields
    - Advanced indexing (HNSW, IVFFlat) with auto-tuning
    - Full-text search with PostgreSQL's GIN indexes
    - Hybrid search combining vector similarity and full-text
    - Batch operations for efficient data ingestion
    - Dynamic schema with version management
    - Extensive filtering and querying capabilities
    
    This provider is fully compatible with SQLAlchemy's API and leverages
    PostgreSQL's native capabilities for optimal performance.
    """

    def __init__(self, config: Union[PgVectorConfig, Dict[str, Any]]):
        """
        Initialize the PgVector provider.
        
        Args:
            config: Either a PgVectorConfig object or a dictionary to create one
        """
        if not _PGVECTOR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="sqlalchemy psycopg pgvector",
                install_command='pip install "upsonic[rag]"',
                feature_name="PGVector vector database provider"
            )
        
        # Convert dict to PgVectorConfig if needed
        if isinstance(config, dict):
            config = PgVectorConfig.from_dict(config)
        
        # Initialize base class
        super().__init__(config)
        self._config: PgVectorConfig = cast(PgVectorConfig, self._config)
        
        # Database settings
        self.schema_name: str = self._config.schema_name
        self.table_name: str = self._config.table_name or self._config.collection_name
        self.connection_string: str = self._config.connection_string.get_secret_value()
        
        # Engine and session (will be initialized in connect())
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[scoped_session] = None
        
        # Table and metadata
        self._metadata: Optional[MetaData] = None
        self._table: Optional[Table] = None
        
        # Index names (will be set during index creation)
        self._vector_index_name: Optional[str] = None
        self._gin_index_name: Optional[str] = None
        
        # Provider metadata
        self.provider_name = self._config.provider_name or f"PgVectorProvider_{self._config.collection_name}"
        self.provider_description = self._config.provider_description
        self.provider_id = self._config.provider_id or self._generate_provider_id()
        
        logger.info(
            f"Initialized PgVectorProvider for collection '{self._config.collection_name}' "
            f"(table: {self.schema_name}.{self.table_name})"
        )

    # ========================================================================
    # Provider Metadata
    # ========================================================================
    
    def _generate_provider_id(self) -> str:
        """Generates a unique provider ID based on connection string and collection."""
        # Create a unique identifier from connection details
        conn_str = getattr(self, 'connection_string', 'default')
        schema = getattr(self, 'schema_name', 'public')
        table = getattr(self, 'table_name', self._config.collection_name)
        
        identifier_parts = [
            conn_str.split("@")[-1] if "@" in conn_str else "local",
            schema,
            table
        ]
        identifier = "#".join(filter(None, identifier_parts))
        return md5(identifier.encode()).hexdigest()[:16]
    
    # ========================================================================
    # Indexed Fields Type Support
    # ========================================================================
    
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
    
    def _get_postgres_column_type(self, field_type: str) -> Any:
        """
        Convert field type string to SQLAlchemy/PostgreSQL column type.
        
        Args:
            field_type: One of 'text', 'keyword', 'integer', 'int', 'bigint', 'float', 'boolean'
        
        Returns:
            SQLAlchemy column type
        """
        type_map = {
            'text': Text,
            'keyword': String,  # VARCHAR with reasonable limit
            'string': String,
            'varchar': String,
            'integer': Integer,
            'int': Integer,
            'int32': Integer,
            'bigint': BigInteger,
            'int64': BigInteger,
            'float': Float,
            'real': Float,
            'double': Float,
            'boolean': Boolean,
            'bool': Boolean,
        }
        return type_map.get(field_type.lower(), Text)
    
    def _get_postgres_index_type(self, field_type: str) -> str:
        """
        Get appropriate PostgreSQL index type for field type.
        
        Args:
            field_type: Field type string
        
        Returns:
            Index type hint ('btree', 'gin', 'hash')
        """
        # PostgreSQL index types:
        # - B-tree: default, good for equality and range queries (numbers, booleans, text)
        # - GIN: good for full-text search and JSONB
        # - Hash: good for equality only
        # - GiST: good for geometric data and full-text
        
        if field_type.lower() in ['text']:
            return 'gin'  # Full-text search
        elif field_type.lower() in ['keyword', 'string', 'varchar']:
            return 'btree'  # Exact match
        elif field_type.lower() in ['integer', 'int', 'bigint', 'int64', 'int32']:
            return 'btree'  # Range queries
        elif field_type.lower() in ['float', 'real', 'double']:
            return 'btree'  # Range queries
        elif field_type.lower() in ['boolean', 'bool']:
            return 'btree'  # Equality
        else:
            return 'btree'  # Default

    # ========================================================================
    # Connection Management
    # ========================================================================

    async def connect(self) -> None:
        """
        Establish connection to PostgreSQL database.
        Creates the engine and session factory.
        """
        try:
            logger.info(f"Connecting to PostgreSQL database...")
            
            # Create engine with connection pooling
            self._engine = await asyncio.to_thread(
                create_engine,
                self.connection_string,
                pool_size=self._config.pool_size,
                max_overflow=self._config.max_overflow,
                pool_timeout=self._config.pool_timeout,
                pool_recycle=self._config.pool_recycle,
                echo=False
            )
            
            # Create session factory
            self._session_factory = scoped_session(
                sessionmaker(bind=self._engine, expire_on_commit=False)
            )
            
            # Initialize metadata and table
            self._metadata = MetaData(schema=self.schema_name)
            self._table = self._get_table_schema()
            
            # Verify connection
            await self._verify_connection()
            
            self._is_connected = True
            logger.info("Successfully connected to PostgreSQL database")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise VectorDBConnectionError(f"Connection failed: {e}")

    async def _verify_connection(self) -> None:
        """Verify the database connection by executing a simple query."""
        try:
            def _verify():
                with self._session_factory() as session:
                    session.execute(text("SELECT 1"))
            
            await asyncio.to_thread(_verify)
            logger.debug("Database connection verified")
        except Exception as e:
            raise VectorDBConnectionError(f"Connection verification failed: {e}")

    async def disconnect(self) -> None:
        """Gracefully close the database connection."""
        try:
            if self._session_factory:
                await asyncio.to_thread(self._session_factory.remove)
            
            if self._engine:
                await asyncio.to_thread(self._engine.dispose)
            
            self._is_connected = False
            logger.info("Disconnected from PostgreSQL database")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            raise VectorDBError(f"Disconnect failed: {e}")

    def connect_sync(self) -> None:
        """Establish connection to PostgreSQL database (sync)."""
        return self._run_async_from_sync(self.connect())

    def disconnect_sync(self) -> None:
        """Gracefully close the database connection (sync)."""
        return self._run_async_from_sync(self.disconnect())

    def is_ready_sync(self) -> bool:
        """Check if the database is connected and responsive (sync)."""
        return self._run_async_from_sync(self.is_ready())

    async def is_ready(self) -> bool:
        """Check if the database is connected and responsive."""
        if not self._is_connected or not self._engine:
            return False
        
        try:
            await self._verify_connection()
            return True
        except:
            return False

    # ========================================================================
    # Schema Management
    # ========================================================================

    def _get_table_schema(self) -> Table:
        """
        Define and return the table schema based on schema version.
        
        Returns:
            SQLAlchemy Table object with the complete schema
        """
        if self._config.schema_version == 1:
            return self._get_table_schema_v1()
        else:
            raise NotImplementedError(
                f"Unsupported schema version: {self._config.schema_version}"
            )

    def _get_table_schema_v1(self) -> Table:
        """
        Schema version 1: Comprehensive schema with all required fields.
        
        Supports typed indexed_fields:
        - Simple format: ["document_name", "document_id"]
        - Advanced format: [{"field": "age", "type": "integer"}, {"field": "score", "type": "float"}]
        
        Fields:
        - id: Primary key (UUID or string)
        - content_id: Unique identifier for content (auto-generated if not provided)
        - document_name: Optional document name (type configurable via indexed_fields)
        - document_id: Optional document identifier (type configurable via indexed_fields)
        - content: The actual text content (required, used for full-text search)
        - embedding: Vector embedding
        - metadata: JSONB field for custom metadata
        - created_at: Timestamp of creation
        - updated_at: Timestamp of last update
        """
        if self._config.vector_size is None:
            raise ValueError("vector_size must be set in config")
        
        # Parse indexed_fields configuration
        indexed_fields_config = self._parse_indexed_fields()
        
        # Determine column types for standard fields based on indexed_fields config
        doc_name_type = String if indexed_fields_config.get("document_name", {}).get("type", "text") in ["keyword", "string", "varchar"] else Text
        doc_id_type = String if indexed_fields_config.get("document_id", {}).get("type", "text") in ["keyword", "string", "varchar"] else Text
        
        table = Table(
            self.table_name,
            self._metadata,
            # Primary key (internal ID)
            Column("id", String, primary_key=True),
            
            # Core identifiers with configurable types
            Column("content_id", String, unique=True, nullable=False, index=True),
            Column("document_name", doc_name_type, nullable=True, index="document_name" in indexed_fields_config),
            Column("document_id", doc_id_type, nullable=True, index="document_id" in indexed_fields_config),
            
            # Content and embedding
            Column("content", Text, nullable=False),
            Column("embedding", Vector(self._config.vector_size), nullable=False),
            
            # Flexible metadata storage
            Column("metadata", postgresql.JSONB, server_default=text("'{}'::jsonb")),
            
            # Timestamps
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
            
            extend_existing=True,
        )
        
        # Add indexes for commonly queried fields
        Index(f"idx_{self.table_name}_id", table.c.id)
        Index(f"idx_{self.table_name}_content_id", table.c.content_id)
        
        # Add indexes for fields specified in indexed_fields config
        if indexed_fields_config:
            for field_name, field_config in indexed_fields_config.items():
                if field_name in ['id', 'content_id']:
                    # Already indexed above
                    continue
                elif field_name in ['document_name', 'document_id']:
                    # Already handled in column definition with index=True
                    continue
                elif field_name == 'content':
                    # Full-text search index for content (will be created via GIN index)
                    if field_config.get("type", "text") == "text":
                        logger.debug(f"Full-text search on 'content' field will use GIN index")
                elif field_name == 'metadata':
                    # GIN index for JSONB metadata (created separately in create_collection)
                    logger.debug("JSONB metadata field will use GIN index")
                else:
                    # Custom field - should be stored in metadata JSONB column
                    logger.debug(
                        f"Custom field '{field_name}' (type: {field_config.get('type')}) "
                        f"should be stored in metadata JSONB column"
                    )
        
        return table

    async def create_collection(self) -> None:
        """
        Create the collection (table) in PostgreSQL.
        
        This method:
        1. Enables the pgvector extension
        2. Creates the schema if it doesn't exist
        3. Creates the table with all columns
        4. Creates necessary indexes
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            # Check if collection exists
            exists = await self.collection_exists()
            
            if exists:
                if self._config.recreate_if_exists:
                    logger.info(f"Collection '{self.table_name}' exists, recreating...")
                    await self.delete_collection()
                else:
                    logger.info(f"Collection '{self.table_name}' already exists")
                    return
            
            # Create collection
            def _create():
                with self._session_factory() as session:
                    with session.begin():
                        # Enable pgvector extension
                        logger.debug("Enabling pgvector extension")
                        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                        
                        # Create schema
                        if self.schema_name and self.schema_name != "public":
                            logger.debug(f"Creating schema: {self.schema_name}")
                            session.execute(
                                text(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")
                            )
                
                # Create table
                logger.debug(f"Creating table: {self.table_name}")
                self._table.create(self._engine)
            
            await asyncio.to_thread(_create)
            
            # Create indexes
            await self._create_vector_index()
            await self._create_gin_index()
            
            # Create metadata indexes if specified
            if self._config.indexed_fields and 'metadata' in self._config.indexed_fields:
                await self._create_metadata_indexes()
            
            logger.info(f"Successfully created collection '{self.table_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise VectorDBError(f"Collection creation failed: {e}")

    async def delete_collection(self) -> None:
        """Permanently delete the collection (table)."""
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            exists = await self.collection_exists()
            if not exists:
                raise CollectionDoesNotExistError(
                    f"Collection '{self.table_name}' does not exist"
                )
            
            def _drop():
                logger.debug(f"Dropping table: {self.schema_name}.{self.table_name}")
                self._table.drop(self._engine)
            
            await asyncio.to_thread(_drop)
            logger.info(f"Successfully deleted collection '{self.table_name}'")
            
        except CollectionDoesNotExistError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorDBError(f"Collection deletion failed: {e}")

    async def collection_exists(self) -> bool:
        """Check if the collection (table) exists."""
        if not self._is_connected or not self._engine:
            return False
        
        try:
            def _check():
                return inspect(self._engine).has_table(
                    self.table_name, 
                    schema=self.schema_name
                )
            
            return await asyncio.to_thread(_check)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def create_collection_sync(self) -> None:
        """Create the collection (table) in PostgreSQL (sync)."""
        return self._run_async_from_sync(self.create_collection())

    def delete_collection_sync(self) -> None:
        """Permanently delete the collection (table) (sync)."""
        return self._run_async_from_sync(self.delete_collection())

    def collection_exists_sync(self) -> bool:
        """Check if the collection (table) exists (sync)."""
        return self._run_async_from_sync(self.collection_exists())

    # ========================================================================
    # Data Operations
    # ========================================================================

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
        Upsert (insert or update) data into the collection.
        
        Args:
            vectors: List of vector embeddings
            payloads: List of metadata dictionaries
            ids: List of unique identifiers
            chunks: Optional list of text content for each vector
            sparse_vectors: Not used in PgVector (reserved for future)
            **kwargs: Additional options:
                - metadata: Additional metadata to merge with payloads
                - batch_size: Override default batch size
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        if not await self.collection_exists():
            raise CollectionDoesNotExistError(
                f"Collection '{self.table_name}' does not exist. Create it first."
            )
        
        # Validate inputs
        if not (len(vectors) == len(payloads) == len(ids)):
            raise UpsertError(
                f"Length mismatch: vectors({len(vectors)}), "
                f"payloads({len(payloads)}), ids({len(ids)})"
            )
        
        if chunks and len(chunks) != len(vectors):
            raise UpsertError(
                f"Length mismatch: chunks({len(chunks)}) != vectors({len(vectors)})"
            )
        
        try:
            # Prepare records
            records = self._prepare_records(
                vectors=vectors,
                payloads=payloads,
                ids=ids,
                chunks=chunks,
                additional_metadata=kwargs.get('metadata')
            )
            
            # Batch upsert
            batch_size = kwargs.get('batch_size', self._config.batch_size)
            await self._batch_upsert(records, batch_size)
            
            logger.info(f"Successfully upserted {len(records)} records")
            
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise UpsertError(f"Failed to upsert data: {e}")

    def upsert_sync(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Upsert (insert or update) data into the collection (sync)."""
        return self._run_async_from_sync(
            self.upsert(vectors, payloads, ids, chunks, sparse_vectors, **kwargs)
        )

    def _prepare_records(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[Union[str, int]],
        chunks: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare records for insertion/update.
        
        This method:
        1. Generates content_id if not provided
        2. Merges metadata from config and kwargs
        3. Extracts document_name and document_id from payload
        4. Creates the content field from chunks or payload
        """
        records = []
        
        for i, (vector, payload, record_id) in enumerate(zip(vectors, payloads, ids)):
            # Get or generate content_id
            content_id = payload.get('content_id')
            if not content_id:
                if self._config.auto_generate_content_id:
                    content_id = self._generate_content_id(payload, chunks[i] if chunks else None)
                else:
                    content_id = str(record_id)
            
            # Extract document identifiers
            document_name = payload.get('document_name')
            document_id = payload.get('document_id')
            
            # Get content (required field)
            if chunks and i < len(chunks):
                content = chunks[i]
            elif 'content' in payload:
                content = payload['content']
            elif 'text' in payload:
                content = payload['text']
            else:
                raise UpsertError(
                    f"No content found for record {i}. "
                    "Provide 'chunks' or include 'content'/'text' in payload"
                )
            
            # Clean content
            content = self._clean_content(content)
            
            # Prepare metadata
            metadata = {}
            
            # Add default metadata from config
            if self._config.default_metadata:
                metadata.update(self._config.default_metadata)
            
            # Add additional metadata from kwargs
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Add payload (excluding special fields)
            excluded_fields = {
                'content_id', 'document_name', 'document_id', 
                'content', 'text', 'embedding', 'vector'
            }
            for key, value in payload.items():
                if key not in excluded_fields:
                    metadata[key] = value
            
            # Create record
            record = {
                'id': str(record_id),
                'content_id': content_id,
                'document_name': document_name,
                'document_id': document_id,
                'content': content,
                'embedding': vector,
                'metadata': metadata
            }
            
            records.append(record)
        
        return records

    def _generate_content_id(
        self, 
        payload: Dict[str, Any], 
        content: Optional[str] = None
    ) -> str:
        """
        Generate a unique content_id based on content hash or UUID.
        
        Args:
            payload: The payload dictionary
            content: The content string
            
        Returns:
            A unique content_id
        """
        if content:
            # Use content hash for deduplication
            return md5(content.encode('utf-8')).hexdigest()
        else:
            # Use UUID for unique identification
            return str(uuid4())

    def _clean_content(self, content: str) -> str:
        """
        Clean content by replacing null characters.
        PostgreSQL doesn't accept null characters in TEXT fields.
        """
        return content.replace("\x00", "\ufffd")

    async def _batch_upsert(
        self, 
        records: List[Dict[str, Any]], 
        batch_size: int
    ) -> None:
        """
        Perform batch upsert operations.
        
        Uses PostgreSQL's ON CONFLICT clause for efficient upserts.
        """
        def _upsert_batch(batch: List[Dict[str, Any]]):
            with self._session_factory() as session:
                with session.begin():
                    # Use INSERT ... ON CONFLICT ... DO UPDATE
                    insert_stmt = postgresql.insert(self._table).values(batch)
                    upsert_stmt = insert_stmt.on_conflict_do_update(
                        index_elements=['content_id'],
                        set_={
                            'id': insert_stmt.excluded.id,
                            'document_name': insert_stmt.excluded.document_name,
                            'document_id': insert_stmt.excluded.document_id,
                            'content': insert_stmt.excluded.content,
                            'embedding': insert_stmt.excluded.embedding,
                            'metadata': insert_stmt.excluded.metadata,
                        }
                    )
                    session.execute(upsert_stmt)
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            logger.debug(f"Upserting batch {i // batch_size + 1} ({len(batch)} records)")
            await asyncio.to_thread(_upsert_batch, batch)

    async def delete(
        self, 
        ids: List[Union[str, int]], 
        **kwargs
    ) -> None:
        """
        Delete records by their IDs.
        
        Args:
            ids: List of IDs to delete
            **kwargs: Additional options:
                - by_content_id: If True, treat ids as content_ids instead of ids
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            by_content_id = kwargs.get('by_content_id', False)
            
            def _delete():
                with self._session_factory() as session:
                    with session.begin():
                        if by_content_id:
                            stmt = sa_delete(self._table).where(
                                self._table.c.content_id.in_(ids)
                            )
                        else:
                            stmt = sa_delete(self._table).where(
                                self._table.c.id.in_([str(id) for id in ids])
                            )
                        session.execute(stmt)
            
            await asyncio.to_thread(_delete)
            logger.info(f"Deleted {len(ids)} records")
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise VectorDBError(f"Failed to delete records: {e}")

    def delete_sync(self, ids: List[Union[str, int]], **kwargs) -> None:
        """Delete records by their IDs (sync)."""
        return self._run_async_from_sync(self.delete(ids, **kwargs))

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete records matching metadata filter (sync)."""
        return self._run_async_from_sync(self.async_delete_by_metadata(metadata))
    
    async def async_delete_by_metadata(
        self, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Delete records matching metadata filter (async).
        
        Args:
            metadata: Dictionary of metadata key-value pairs to match
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            def _delete():
                with self._session_factory() as session:
                    with session.begin():
                        stmt = sa_delete(self._table).where(
                            self._table.c.metadata.contains(metadata)
                        )
                        result = session.execute(stmt)
                        return result.rowcount
            
            count = await asyncio.to_thread(_delete)
            logger.info(f"Deleted {count} records matching metadata filter")
            return True
            
        except Exception as e:
            logger.error(f"Delete by metadata failed: {e}")
            return False

    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all records with the given document_id (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_id(document_id))
    
    async def async_delete_by_document_id(
        self, 
        document_id: str
    ) -> bool:
        """Delete all records with the given document_id."""
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            def _delete():
                with self._session_factory() as session:
                    with session.begin():
                        stmt = sa_delete(self._table).where(
                            self._table.c.document_id == document_id
                        )
                        result = session.execute(stmt)
                        return result.rowcount
            
            count = await asyncio.to_thread(_delete)
            logger.info(f"Deleted {count} records with document_id '{document_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Delete by document_id failed: {e}")
            return False

    async def fetch(
        self, 
        ids: List[Union[str, int]], 
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Retrieve records by their IDs.
        
        Args:
            ids: List of IDs to fetch
            **kwargs: Additional options:
                - by_content_id: If True, treat ids as content_ids
                
        Returns:
            List of VectorSearchResult objects
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            by_content_id = kwargs.get('by_content_id', False)
            
            def _fetch():
                with self._session_factory() as session:
                    if by_content_id:
                        stmt = select(self._table).where(
                            self._table.c.content_id.in_(ids)
                        )
                    else:
                        stmt = select(self._table).where(
                            self._table.c.id.in_([str(id) for id in ids])
                        )
                    result = session.execute(stmt)
                    return result.fetchall()
            
            rows = await asyncio.to_thread(_fetch)
            
            # Convert to VectorSearchResult
            results = []
            for row in rows:
                results.append(
                    VectorSearchResult(
                        id=row.content_id,
                        score=1.0,  # No score for direct fetch
                        payload=self._row_to_payload(row),
                        vector=list(row.embedding) if row.embedding is not None else None,
                        text=row.content
                    )
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            raise VectorDBError(f"Failed to fetch records: {e}")

    def fetch_sync(
        self, 
        ids: List[Union[str, int]], 
        **kwargs
    ) -> List[VectorSearchResult]:
        """Retrieve records by their IDs (sync)."""
        return self._run_async_from_sync(self.fetch(ids, **kwargs))

    def _row_to_payload(self, row) -> Dict[str, Any]:
        """Convert a database row to a payload dictionary."""
        payload = {
            'id': row.id,
            'content_id': row.content_id,
            'document_name': row.document_name,
            'document_id': row.document_id,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'updated_at': row.updated_at.isoformat() if row.updated_at else None,
        }
        
        # Add metadata fields
        if row.metadata:
            payload.update(row.metadata)
        
        return payload

    # ========================================================================
    # Search Operations
    # ========================================================================

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
        Master search method that dispatches to appropriate search function.
        
        Args:
            top_k: Number of results to return
            query_vector: Vector for dense search
            query_text: Text for full-text search
            filter: Metadata filter
            alpha: Weight for hybrid search
            fusion_method: Fusion algorithm for hybrid search
            similarity_threshold: Minimum similarity score
            **kwargs: Additional provider-specific options
            
        Returns:
            List of VectorSearchResult objects
        """
        # Use defaults from config if not provided
        top_k = top_k if top_k is not None else self._config.default_top_k
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        alpha = alpha if alpha is not None else self._config.default_hybrid_alpha
        fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method
        
        # Determine search type
        has_vector = query_vector is not None
        has_text = query_text is not None
        
        # Validate search capabilities
        if has_vector and has_text:
            if not self._config.hybrid_search_enabled:
                raise ConfigurationError("Hybrid search is disabled in configuration")
            return await self.hybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                alpha=alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        elif has_vector:
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled in configuration")
            return await self.dense_search(
                query_vector=query_vector,
                top_k=top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        elif has_text:
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled in configuration")
            return await self.full_text_search(
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
        else:
            raise ConfigurationError(
                "Must provide either query_vector, query_text, or both"
            )

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
        """Master search method that dispatches to appropriate search function (sync)."""
        return self._run_async_from_sync(
            self.search(
                top_k, query_vector, query_text, filter, 
                alpha, fusion_method, similarity_threshold, **kwargs
            )
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
        
        Uses pgvector's distance operators based on the configured metric.
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        # Use config default if not provided
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        
        try:
            def _search():
                with self._session_factory() as session:
                    with session.begin():
                        # Set index parameters
                        self._set_index_params(session)
                        
                        # Build query
                        stmt = select(
                            self._table.c.id,
                            self._table.c.content_id,
                            self._table.c.document_name,
                            self._table.c.document_id,
                            self._table.c.content,
                            self._table.c.embedding,
                            self._table.c.metadata,
                            self._table.c.created_at,
                            self._table.c.updated_at,
                        )
                        
                        # Apply filter
                        if filter:
                            stmt = self._apply_filter(stmt, filter)
                        
                        # Calculate distance and add as column
                        distance_col = self._get_distance_column(query_vector)
                        stmt = stmt.add_columns(distance_col.label('distance'))
                        
                        # Order by distance
                        stmt = stmt.order_by('distance')
                        
                        # Apply similarity threshold if provided
                        if final_similarity_threshold is not None:
                            # Convert similarity to distance based on metric
                            max_distance = self._similarity_to_distance(final_similarity_threshold)
                            stmt = stmt.where(distance_col <= max_distance)
                        
                        # Limit results
                        stmt = stmt.limit(top_k)
                        
                        # Execute
                        result = session.execute(stmt)
                        return result.fetchall()
            
            rows = await asyncio.to_thread(_search)
            
            # Convert to VectorSearchResult
            results = []
            for row in rows:
                # Convert distance to similarity score
                score = self._distance_to_similarity(row.distance)
                
                results.append(
                    VectorSearchResult(
                        id=row.content_id,
                        score=score,
                        payload=self._row_to_payload(row),
                        vector=list(row.embedding) if row.embedding is not None else None,
                        text=row.content
                    )
                )
            
            logger.debug(f"Dense search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise SearchError(f"Dense search failed: {e}")

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
        Perform full-text search using PostgreSQL's text search.
        
        Uses GIN indexes for fast text matching.
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        # Use config default if not provided
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        
        try:
            def _search():
                with self._session_factory() as session:
                    # Build text search query
                    processed_query = self._process_text_query(query_text)
                    
                    # Build statement
                    stmt = select(
                        self._table.c.id,
                        self._table.c.content_id,
                        self._table.c.document_name,
                        self._table.c.document_id,
                        self._table.c.content,
                        self._table.c.embedding,
                        self._table.c.metadata,
                        self._table.c.created_at,
                        self._table.c.updated_at,
                    )
                    
                    # Apply filter
                    if filter:
                        stmt = self._apply_filter(stmt, filter)
                    
                    # Create text search components
                    ts_vector = func.to_tsvector(
                        self._config.content_language, 
                        self._table.c.content
                    )
                    ts_query = func.websearch_to_tsquery(
                        self._config.content_language,
                        bindparam("query", value=processed_query)
                    )
                    text_rank = func.ts_rank_cd(ts_vector, ts_query)
                    
                    # Add rank column
                    stmt = stmt.add_columns(text_rank.label('rank'))
                    
                    # Filter out results with rank 0 (no match at all)
                    # PostgreSQL ts_rank returns 0 for documents that don't match the query
                    stmt = stmt.where(text_rank > 0)
                    
                    # Order by rank
                    stmt = stmt.order_by(desc('rank'))
                    
                    # Apply similarity threshold if provided
                    if final_similarity_threshold is not None:
                        stmt = stmt.where(text_rank >= final_similarity_threshold)
                    
                    # Limit results
                    stmt = stmt.limit(top_k)
                    
                    # Execute
                    result = session.execute(stmt)
                    return result.fetchall()
            
            rows = await asyncio.to_thread(_search)
            
            # Convert to VectorSearchResult
            results = []
            
            # Normalize scores to 0-1 range if we have results
            if rows:
                max_rank = max(float(row.rank) for row in rows) if rows else 1.0
                # Avoid division by zero
                if max_rank == 0:
                    max_rank = 1.0
                
                for row in rows:
                    # Normalize the rank to 0-1 by dividing by max rank
                    normalized_score = float(row.rank) / max_rank
                    results.append(
                        VectorSearchResult(
                            id=row.content_id,
                            score=normalized_score,
                            payload=self._row_to_payload(row),
                            vector=list(row.embedding) if row.embedding is not None else None,
                            text=row.content
                        )
                    )
            
            logger.debug(f"Full-text search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            raise SearchError(f"Full-text search failed: {e}")

    def full_text_search_sync(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """Perform full-text search using PostgreSQL's text search (sync)."""
        return self._run_async_from_sync(
            self.full_text_search(query_text, top_k, filter, similarity_threshold, **kwargs)
        )

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
        Perform hybrid search combining vector similarity and full-text search.
        
        This implementation uses PostgreSQL's native capabilities to compute
        both scores in a single query for optimal performance.
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        # Use defaults from config if not provided
        alpha = alpha if alpha is not None else self._config.default_hybrid_alpha
        fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method
        
        try:
            if fusion_method == 'rrf':
                return await self._hybrid_search_rrf(
                    query_vector, query_text, top_k, filter, similarity_threshold, **kwargs
                )
            else:  # weighted
                return await self._hybrid_search_weighted(
                    query_vector, query_text, top_k, filter, alpha, similarity_threshold, **kwargs
                )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}")

    async def _hybrid_search_weighted(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]],
        alpha: float,
        similarity_threshold: Optional[float],
        **kwargs
    ) -> List[VectorSearchResult]:
        """Hybrid search using weighted score combination."""
        
        def _search():
            with self._session_factory() as session:
                with session.begin():
                    # Set index parameters
                    self._set_index_params(session)
                    
                    # Process text query
                    processed_query = self._process_text_query(query_text)
                    
                    # Build statement
                    stmt = select(
                        self._table.c.id,
                        self._table.c.content_id,
                        self._table.c.document_name,
                        self._table.c.document_id,
                        self._table.c.content,
                        self._table.c.embedding,
                        self._table.c.metadata,
                        self._table.c.created_at,
                        self._table.c.updated_at,
                    )
                    
                    # Apply filter
                    if filter:
                        stmt = self._apply_filter(stmt, filter)
                    
                    # Calculate vector distance
                    distance_col = self._get_distance_column(query_vector)
                    
                    # Calculate vector similarity score (normalized to 0-1)
                    vector_score = self._distance_to_similarity_expression(distance_col)
                    
                    # Calculate text rank
                    ts_vector = func.to_tsvector(
                        self._config.content_language,
                        self._table.c.content
                    )
                    ts_query = func.websearch_to_tsquery(
                        self._config.content_language,
                        bindparam("query", value=processed_query)
                    )
                    text_rank = func.ts_rank_cd(ts_vector, ts_query)
                    
                    # Combine scores with weights
                    text_weight = 1.0 - alpha
                    hybrid_score = (alpha * vector_score) + (text_weight * text_rank)
                    
                    # Add columns
                    stmt = stmt.add_columns(
                        distance_col.label('distance'),
                        text_rank.label('text_rank'),
                        hybrid_score.label('hybrid_score')
                    )
                    
                    # Order by hybrid score
                    stmt = stmt.order_by(desc('hybrid_score'))
                    
                    # Apply threshold
                    if similarity_threshold is not None:
                        stmt = stmt.where(hybrid_score >= similarity_threshold)
                    
                    # Limit results
                    stmt = stmt.limit(top_k)
                    
                    # Execute
                    result = session.execute(stmt)
                    return result.fetchall()
        
        rows = await asyncio.to_thread(_search)
        
        # Convert to VectorSearchResult
        results = []
        for row in rows:
            results.append(
                VectorSearchResult(
                    id=row.content_id,
                    score=float(row.hybrid_score),
                    payload=self._row_to_payload(row),
                    vector=list(row.embedding) if row.embedding is not None else None,
                    text=row.content
                )
            )
        
        logger.debug(f"Hybrid search (weighted) returned {len(results)} results")
        return results

    async def _hybrid_search_rrf(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]],
        similarity_threshold: Optional[float],
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank_i)) for each ranking
        Default k = 60 (common value in literature)
        """
        k = kwargs.get('rrf_k', 60)
        
        # Get separate results
        vector_results = await self.dense_search(
            query_vector=query_vector,
            top_k=top_k * 2,  # Get more results for fusion
            filter=filter,
            **kwargs
        )
        
        text_results = await self.full_text_search(
            query_text=query_text,
            top_k=top_k * 2,
            filter=filter,
            **kwargs
        )
        
        # Create rank maps
        vector_ranks = {r.id: i + 1 for i, r in enumerate(vector_results)}
        text_ranks = {r.id: i + 1 for i, r in enumerate(text_results)}
        
        # Combine using RRF
        rrf_scores: Dict[str, float] = {}
        all_ids = set(vector_ranks.keys()) | set(text_ranks.keys())
        
        for doc_id in all_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += 1.0 / (k + vector_ranks[doc_id])
            if doc_id in text_ranks:
                score += 1.0 / (k + text_ranks[doc_id])
            rrf_scores[doc_id] = score
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create result map for quick lookup
        result_map = {}
        for r in vector_results + text_results:
            if r.id not in result_map:
                result_map[r.id] = r
        
        # Build final results
        results = []
        for doc_id in sorted_ids[:top_k]:
            result = result_map[doc_id]
            # Update score with RRF score
            results.append(
                VectorSearchResult(
                    id=result.id,
                    score=rrf_scores[doc_id],
                    payload=result.payload,
                    vector=result.vector,
                    text=result.text
                )
            )
        
        # Apply threshold if provided
        if similarity_threshold is not None:
            results = [r for r in results if r.score >= similarity_threshold]
        
        logger.debug(f"Hybrid search (RRF) returned {len(results)} results")
        return results

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
        """Perform hybrid search combining vector similarity and full-text search (sync)."""
        return self._run_async_from_sync(
            self.hybrid_search(
                query_vector, query_text, top_k, filter,
                alpha, fusion_method, similarity_threshold, **kwargs
            )
        )

    # ========================================================================
    # Helper Methods for Search
    # ========================================================================

    def _set_index_params(self, session: Session) -> None:
        """Set index-specific parameters for the current session."""
        if isinstance(self._config.index, IVFIndexConfig):
            # Use nprobe from index config, fallback to default
            nprobe = self._config.index.nprobe if self._config.index.nprobe else 10
            session.execute(
                text(f"SET LOCAL ivfflat.probes = {nprobe}")
            )
        elif isinstance(self._config.index, HNSWIndexConfig):
            # Use ef_search from index config, fallback to default
            ef_search = self._config.index.ef_search if self._config.index.ef_search else 40
            session.execute(
                text(f"SET LOCAL hnsw.ef_search = {ef_search}")
            )

    def _get_distance_column(self, query_vector: List[float]):
        """Get the appropriate distance column based on the metric."""
        if self._config.distance_metric == DistanceMetric.COSINE:
            return self._table.c.embedding.cosine_distance(query_vector)
        elif self._config.distance_metric == DistanceMetric.EUCLIDEAN:
            return self._table.c.embedding.l2_distance(query_vector)
        elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            return self._table.c.embedding.max_inner_product(query_vector)
        else:
            raise ConfigurationError(f"Unsupported distance metric: {self._config.distance_metric}")

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score (0-1 range)."""
        if self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            # Inner product: higher is better, normalize to 0-1
            return (distance + 1.0) / 2.0
        else:
            # Distance metrics: lower is better, invert
            return 1.0 / (1.0 + distance)

    def _similarity_to_distance(self, similarity: float) -> float:
        """Convert similarity score to distance."""
        if self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            # Reverse the normalization
            return (similarity * 2.0) - 1.0
        else:
            # Reverse the inversion
            return (1.0 / similarity) - 1.0

    def _distance_to_similarity_expression(self, distance_col):
        """Create a SQL expression to convert distance to similarity."""
        if self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            # Normalize inner product to 0-1
            return (distance_col + 1.0) / 2.0
        else:
            # Invert distance
            return 1.0 / (1.0 + distance_col)

    def _process_text_query(self, query: str) -> str:
        """Process text query for full-text search."""
        if self._config.prefix_match:
            # Enable prefix matching by appending '*' to each word
            words = query.strip().split()
            processed_words = [word + "*" for word in words]
            return " ".join(processed_words)
        return query

    def _apply_filter(self, stmt, filter: Dict[str, Any]):
        """Apply metadata filter to the query."""
        for key, value in filter.items():
            # Check if it's a direct column
            if key in ['document_name', 'document_id', 'content_id']:
                column = getattr(self._table.c, key)
                stmt = stmt.where(column == value)
            else:
                # Metadata field - use JSONB containment
                stmt = stmt.where(
                    self._table.c.metadata[key].astext == str(value)
                )
        return stmt

    # ========================================================================
    # Index Management
    # ========================================================================

    async def _create_vector_index(self) -> None:
        """Create vector index (HNSW or IVFFlat)."""
        try:
            # Generate index name
            index_type = 'hnsw' if isinstance(self._config.index, HNSWIndexConfig) else 'ivfflat'
            self._vector_index_name = f"{self.table_name}_{index_type}_embedding_idx"
            
            # Check if index exists
            if await self._index_exists(self._vector_index_name):
                logger.info(f"Vector index '{self._vector_index_name}' already exists")
                return
            
            # Get distance operator
            distance_op = self._get_distance_operator()
            
            def _create():
                with self._session_factory() as session:
                    with session.begin():
                        if isinstance(self._config.index, HNSWIndexConfig):
                            self._create_hnsw_index(session, distance_op)
                        else:
                            self._create_ivfflat_index(session, distance_op)
            
            await asyncio.to_thread(_create)
            logger.info(f"Created vector index '{self._vector_index_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            raise VectorDBError(f"Vector index creation failed: {e}")

    def _create_hnsw_index(self, session: Session, distance_op: str) -> None:
        """Create HNSW index."""
        config = cast(HNSWIndexConfig, self._config.index)
        
        logger.debug(
            f"Creating HNSW index with m={config.m}, "
            f"ef_construction={config.ef_construction}"
        )
        
        create_sql = text(
            f'CREATE INDEX "{self._vector_index_name}" '
            f'ON {self.schema_name}.{self.table_name} '
            f'USING hnsw (embedding {distance_op}) '
            f'WITH (m = :m, ef_construction = :ef_construction);'
        )
        
        session.execute(
            create_sql,
            {'m': config.m, 'ef_construction': config.ef_construction}
        )

    def _create_ivfflat_index(self, session: Session, distance_op: str) -> None:
        """Create IVFFlat index."""
        config = cast(IVFIndexConfig, self._config.index)
        
        # Calculate number of lists dynamically or use configured value
        num_lists = self._calculate_ivfflat_lists(session)
        
        logger.debug(f"Creating IVFFlat index with lists={num_lists}")
        
        create_sql = text(
            f'CREATE INDEX "{self._vector_index_name}" '
            f'ON {self.schema_name}.{self.table_name} '
            f'USING ivfflat (embedding {distance_op}) '
            f'WITH (lists = :lists);'
        )
        
        session.execute(create_sql, {'lists': num_lists})

    def _calculate_ivfflat_lists(self, session: Session) -> int:
        """
        Calculate optimal number of lists for IVFFlat based on row count.
        
        - Small datasets (< 1M rows): lists = rows / 1000
        - Large datasets (>= 1M rows): lists = sqrt(rows)
        """
        # Get row count
        count_stmt = select(func.count()).select_from(self._table)
        result = session.execute(count_stmt)
        row_count = result.scalar() or 0
        
        logger.debug(f"Row count for IVFFlat calculation: {row_count}")
        
        if row_count < 1000000:
            num_lists = max(int(row_count / 1000), 1)
        else:
            num_lists = max(int(sqrt(row_count)), 1)
        
        return num_lists

    def _get_distance_operator(self) -> str:
        """Get the distance operator string for index creation."""
        metric_map = {
            DistanceMetric.COSINE: 'vector_cosine_ops',
            DistanceMetric.EUCLIDEAN: 'vector_l2_ops',
            DistanceMetric.DOT_PRODUCT: 'vector_ip_ops'
        }
        return metric_map[self._config.distance_metric]

    async def _create_gin_index(self) -> None:
        """Create GIN index for full-text search on content field."""
        try:
            self._gin_index_name = f"{self.table_name}_content_gin_idx"
            
            # Check if index exists
            if await self._index_exists(self._gin_index_name):
                logger.info(f"GIN index '{self._gin_index_name}' already exists")
                return
            
            def _create():
                with self._session_factory() as session:
                    with session.begin():
                        logger.debug("Creating GIN index for full-text search")
                        create_sql = text(
                            f'CREATE INDEX "{self._gin_index_name}" '
                            f'ON {self.schema_name}.{self.table_name} '
                            f'USING GIN (to_tsvector(:language, content));'
                        )
                        session.execute(
                            create_sql,
                            {'language': self._config.content_language}
                        )
            
            await asyncio.to_thread(_create)
            logger.info(f"Created GIN index '{self._gin_index_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create GIN index: {e}")
            raise VectorDBError(f"GIN index creation failed: {e}")

    async def _create_metadata_indexes(self) -> None:
        """Create GIN index for JSONB metadata field."""
        try:
            metadata_index_name = f"{self.table_name}_metadata_gin_idx"
            
            # Check if index exists
            if await self._index_exists(metadata_index_name):
                logger.info(f"Metadata index '{metadata_index_name}' already exists")
                return
            
            def _create():
                with self._session_factory() as session:
                    with session.begin():
                        logger.debug("Creating GIN index for metadata")
                        create_sql = text(
                            f'CREATE INDEX "{metadata_index_name}" '
                            f'ON {self.schema_name}.{self.table_name} '
                            f'USING GIN (metadata);'
                        )
                        session.execute(create_sql)
            
            await asyncio.to_thread(_create)
            logger.info(f"Created metadata index '{metadata_index_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create metadata index: {e}")
            raise VectorDBError(f"Metadata index creation failed: {e}")

    async def _index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        try:
            def _check():
                inspector = inspect(self._engine)
                indexes = inspector.get_indexes(self.table_name, schema=self.schema_name)
                return any(idx['name'] == index_name for idx in indexes)
            
            return await asyncio.to_thread(_check)
        except:
            return False

    def optimize(self, force_recreate: bool = False) -> bool:
        """Optimize the database by creating/recreating indexes (sync)."""
        return self._run_async_from_sync(self.async_optimize(force_recreate))
    
    async def async_optimize(self, force_recreate: bool = False) -> bool:
        """
        Optimize the database by creating/recreating indexes (async).
        
        Args:
            force_recreate: If True, drop and recreate all indexes

        Returns:
            True if optimization was successful, False otherwise
        """
        logger.info("Optimizing PgVector database...")
        try:
            if force_recreate:
                # Drop existing indexes
                if self._vector_index_name:
                    await self._drop_index(self._vector_index_name)
                if self._gin_index_name:
                    await self._drop_index(self._gin_index_name)
            
            # Create indexes
            await self._create_vector_index()
            await self._create_gin_index()
            
            if self._config.indexed_fields and 'metadata' in self._config.indexed_fields:
                await self._create_metadata_indexes()
            
            logger.info("Optimization complete")
            return True
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False

    async def _drop_index(self, index_name: str) -> None:
        """Drop an index."""
        try:
            def _drop():
                with self._session_factory() as session:
                    with session.begin():
                        session.execute(
                            text(f'DROP INDEX IF EXISTS {self.schema_name}."{index_name}";')
                        )
            
            await asyncio.to_thread(_drop)
            logger.info(f"Dropped index '{index_name}'")
        except Exception as e:
            logger.error(f"Failed to drop index '{index_name}': {e}")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    async def get_count(self) -> int:
        """Get the total number of records in the collection."""
        if not self._is_connected:
            return 0
        
        try:
            def _count():
                with self._session_factory() as session:
                    stmt = select(func.count()).select_from(self._table)
                    result = session.execute(stmt)
                    return result.scalar() or 0
            
            return await asyncio.to_thread(_count)
        except:
            return 0

    async def _record_exists(self, column_name: str, value: Any) -> bool:
        """
        Check if a record with the given column value exists.
        
        Args:
            column_name: The name of the column to check
            value: The value to search for
            
        Returns:
            True if the record exists, False otherwise
        """
        if not self._is_connected:
            return False
        
        try:
            def _check():
                with self._session_factory() as session:
                    column = getattr(self._table.c, column_name)
                    stmt = select(1).where(column == value).limit(1)
                    result = session.execute(stmt).first()
                    return result is not None
            
            return await asyncio.to_thread(_check)
        except Exception as e:
            logger.error(f"Error checking if record exists: {e}")
            return False

    def content_id_exists(self, content_id: str) -> bool:
        """Check if a record with the given content_id exists (sync)."""
        return self._run_async_from_sync(self.async_content_id_exists(content_id))
    
    async def async_content_id_exists(self, content_id: str) -> bool:
        """
        Check if a record with the given content_id exists (async).
        
        Args:
            content_id: The content_id to check
            
        Returns:
            True if a record with the content_id exists, False otherwise
        """
        return await self._record_exists('content_id', content_id)

    def document_name_exists(self, document_name: str) -> bool:
        """Check if a record with the given document_name exists (sync)."""
        return self._run_async_from_sync(self.async_document_name_exists(document_name))
    
    async def async_document_name_exists(self, document_name: str) -> bool:
        """
        Check if a record with the given document_name exists (async).
        
        Args:
            document_name: The document_name to check
            
        Returns:
            True if a record with the document_name exists, False otherwise
        """
        return await self._record_exists('document_name', document_name)

    def document_id_exists(self, document_id: str) -> bool:
        """Check if a record with the given document_id exists (sync)."""
        return self._run_async_from_sync(self.async_document_id_exists(document_id))
    
    async def async_document_id_exists(self, document_id: str) -> bool:
        """
        Check if a record with the given document_id exists (async).
        
        Args:
            document_id: The document_id to check
            
        Returns:
            True if a record with the document_id exists, False otherwise
        """
        return await self._record_exists('document_id', document_id)
    
    def get_supported_search_types(self) -> List[str]:
        """Get the supported search types for PgVector (sync)."""
        supported = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported
    
    async def async_get_supported_search_types(self) -> List[str]:
        """Get the supported search types for PgVector (async)."""
        return self.get_supported_search_types()

    async def id_exists(self, id: str) -> bool:
        """
        Check if a record with the given ID exists.
        
        Args:
            id: The ID to check
            
        Returns:
            True if a record with the ID exists, False otherwise
        """
        return await self._record_exists('id', id)

    def delete_by_document_name(self, document_name: str) -> bool:
        """Delete all records with the given document_name (sync)."""
        return self._run_async_from_sync(self.async_delete_by_document_name(document_name))
    
    async def async_delete_by_document_name(self, document_name: str) -> bool:
        """
        Delete all records with the given document_name (async).
        
        Args:
            document_name: The document_name to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            def _delete():
                with self._session_factory() as session:
                    with session.begin():
                        stmt = sa_delete(self._table).where(
                            self._table.c.document_name == document_name
                        )
                        result = session.execute(stmt)
                        return result.rowcount
            
            count = await asyncio.to_thread(_delete)
            logger.info(f"Deleted {count} records with document_name '{document_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Delete by document_name failed: {e}")
            return False
    
    def delete_by_content_id(self, content_id: str) -> bool:
        """Delete all records with the given content_id (sync)."""
        return self._run_async_from_sync(self.async_delete_by_content_id(content_id))
    
    async def async_delete_by_content_id(self, content_id: str) -> bool:
        """
        Delete all records with the given content_id (async).
        
        Args:
            content_id: The content_id to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            def _delete():
                with self._session_factory() as session:
                    with session.begin():
                        stmt = sa_delete(self._table).where(
                            self._table.c.content_id == content_id
                        )
                        result = session.execute(stmt)
                        return result.rowcount
            
            count = await asyncio.to_thread(_delete)
            logger.info(f"Deleted {count} records with content_id '{content_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Delete by content_id failed: {e}")
            return False

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific record (sync)."""
        return self._run_async_from_sync(self.async_update_metadata(content_id, metadata))
    
    async def async_update_metadata(
        self, 
        content_id: str, 
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Update metadata for a specific record (async).
        
        Args:
            content_id: The content_id of the record
            metadata: The metadata to update
            merge: If True, merge with existing metadata; if False, replace

        Returns:
            True if update was successful, False otherwise
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            def _update():
                with self._session_factory() as session:
                    with session.begin():
                        if merge:
                            # Merge with existing metadata
                            stmt = (
                                sa_update(self._table)
                                .where(self._table.c.content_id == content_id)
                                .values(
                                    metadata=func.coalesce(
                                        self._table.c.metadata, text("'{}'::jsonb")
                                    ).op("||")(
                                        bindparam("md", metadata, type_=postgresql.JSONB)
                                    )
                                )
                            )
                        else:
                            # Replace metadata
                            stmt = (
                                sa_update(self._table)
                                .where(self._table.c.content_id == content_id)
                                .values(metadata=metadata)
                            )
                        
                        session.execute(stmt)
            
            await asyncio.to_thread(_update)
            logger.info(f"Updated metadata for content_id '{content_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False
            


    async def clear(self) -> None:
        """Delete all records from the collection without dropping the table."""
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")
        
        try:
            def _clear():
                with self._session_factory() as session:
                    with session.begin():
                        session.execute(sa_delete(self._table))
            
            await asyncio.to_thread(_clear)
            logger.info(f"Cleared all records from collection '{self.table_name}'")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise VectorDBError(f"Clear operation failed: {e}")

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the PgVectorProvider instance.
        
        This is useful when you need to clone the provider for parallel operations.
        Note: The database engine and session are shared, not copied.
        
        Args:
            memo: A dictionary of objects already copied during the current copying pass
            
        Returns:
            A deep-copied instance of PgVectorProvider
        """
        from copy import deepcopy
        
        # Create a new instance without calling __init__
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj
        
        # Deep copy most attributes
        for k, v in self.__dict__.items():
            # Skip SQLAlchemy objects that shouldn't be copied
            if k in {'_metadata', '_table'}:
                continue
            # Reuse engine and session factory (they're thread-safe)
            elif k in {'_engine', '_session_factory'}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))
        
        # Recreate metadata and table for the copied instance
        if self._metadata is not None and self._table is not None:
            copied_obj._metadata = MetaData(schema=copied_obj.schema_name)
            copied_obj._table = copied_obj._get_table_schema()
        
        return copied_obj

    def __repr__(self) -> str:
        return (
            f"PgVectorProvider(collection='{self._config.collection_name}', "
            f"table='{self.schema_name}.{self.table_name}', "
            f"vector_size={self._config.vector_size}, "
            f"metric={self._config.distance_metric.value}, "
            f"index={self._config.index.type.value})"
        )


# Alias for backward compatibility
PgvectorProvider = PgVectorProvider

