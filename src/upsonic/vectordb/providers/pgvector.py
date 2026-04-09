"""
PgVector Provider Implementation

A comprehensive, high-level vector database provider for PostgreSQL with pgvector extension.
Supports async operations, flexible metadata management, advanced indexing, and hybrid search.

Standard properties stored per row:
- chunk_id (String, unique) — unique per-chunk identifier
- chunk_content_hash (String, unique) — MD5 of chunk text for content-based dedup
- document_id (String) — parent document identifier
- doc_content_hash (String) — MD5 of parent document for change detection
- document_name (String) — human-readable source name
- content (Text) — the chunk text
- metadata (JSONB) — all non-standard data

This implementation follows the BaseVectorDBProvider interface and integrates best practices
from both SQLAlchemy and pgvector for optimal performance and flexibility.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid as _uuid
from math import sqrt
from typing import Any, Dict, List, Optional, Union, Literal, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy import (
        Column, String, Text, Integer, BigInteger, Float, Boolean, DateTime,
        Index, MetaData, Table, create_engine, text, select,
        delete as sa_delete, update as sa_update, func, desc, and_
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
        delete as sa_delete, update as sa_update, func, desc, and_
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
from upsonic.utils.printing import info_log, debug_log
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

    - Async-first architecture via asyncio.to_thread wrapping sync SQLAlchemy
    - Centralized client lifecycle with aget_client() singleton pattern
    - Explicit chunk_id / chunk_content_hash / document_id / doc_content_hash storage
    - Content-based deduplication using PostgreSQL native ON CONFLICT
    - Advanced indexing (HNSW, IVFFlat) with auto-tuning
    - Full-text search with PostgreSQL's GIN indexes
    - Hybrid search combining vector similarity and full-text
    - Batch operations for efficient data ingestion
    - Generic afield_exists / adelete_by_field helpers
    """

    _STANDARD_FIELDS: frozenset = frozenset({
        'chunk_id', 'document_id', 'doc_content_hash',
        'chunk_content_hash', 'document_name', 'content', 'metadata',
        'knowledge_base_id',
    })

    def __init__(self, config: Union[PgVectorConfig, Dict[str, Any]]):
        if not _PGVECTOR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="sqlalchemy psycopg pgvector",
                install_command='pip install "upsonic[pgvector]"',
                feature_name="PGVector vector database provider"
            )

        if isinstance(config, dict):
            config = PgVectorConfig.from_dict(config)

        super().__init__(config)
        self._config: PgVectorConfig = cast(PgVectorConfig, self._config)

        self.schema_name: str = self._config.schema_name
        self.table_name: str = self._config.table_name or self._config.collection_name
        self.connection_string: str = self._config.connection_string.get_secret_value()

        self._engine: Optional[Engine] = None
        self._session_factory: Optional[scoped_session] = None

        self._metadata: Optional[MetaData] = None
        self._table: Optional[Table] = None

        self._vector_index_name: Optional[str] = None
        self._gin_index_name: Optional[str] = None

        logger.info(
            f"Initialized PgVectorProvider for collection '{self._config.collection_name}' "
            f"(table: {self.schema_name}.{self.table_name})"
        )

    # ========================================================================
    # Provider Metadata
    # ========================================================================

    def _generate_provider_id(self) -> str:
        conn_str = getattr(self, 'connection_string', 'default')
        schema = getattr(self, 'schema_name', 'public')
        table = getattr(self, 'table_name', self._config.collection_name)

        identifier_parts = [
            conn_str.split("@")[-1] if "@" in conn_str else "local",
            schema,
            table
        ]
        identifier = "#".join(filter(None, identifier_parts))
        return hashlib.md5(identifier.encode()).hexdigest()[:16]

    # ========================================================================
    # Indexed Fields Type Support
    # ========================================================================

    def _parse_indexed_fields(self) -> Dict[str, Dict[str, Any]]:
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

    def _get_postgres_column_type(self, field_type: str) -> Any:
        type_map = {
            'text': Text,
            'keyword': String,
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
        if field_type.lower() in ['text']:
            return 'gin'
        return 'btree'

    # ========================================================================
    # Connection Lifecycle Management
    # ========================================================================

    async def _create_engine_and_session(self) -> None:
        """Creates the SQLAlchemy engine and session factory. Does not verify connectivity."""
        self._engine = await asyncio.to_thread(
            create_engine,
            self.connection_string,
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
            pool_timeout=self._config.pool_timeout,
            pool_recycle=self._config.pool_recycle,
            echo=False
        )
        self._session_factory = scoped_session(
            sessionmaker(bind=self._engine, expire_on_commit=False)
        )
        self._metadata = MetaData(schema=self.schema_name)
        self._table = self._get_table_schema()

    async def aget_client(self) -> scoped_session:
        """
        Singleton entry point for obtaining a verified session factory.
        Creates engine if None, reconnects if unresponsive, verifies readiness.
        """
        if self._engine is None or self._session_factory is None:
            logger.debug("Creating engine and session factory...")
            await self._create_engine_and_session()

        try:
            await self._verify_connection()
        except Exception:
            logger.debug("Existing engine unresponsive, recreating...")
            if self._session_factory:
                try:
                    await asyncio.to_thread(self._session_factory.remove)
                except Exception:
                    pass
            if self._engine:
                try:
                    await asyncio.to_thread(self._engine.dispose)
                except Exception:
                    pass
            self._engine = None
            self._session_factory = None
            self._is_connected = False
            await self._create_engine_and_session()

            try:
                await self._verify_connection()
            except Exception as e:
                self._engine = None
                self._session_factory = None
                self._is_connected = False
                raise VectorDBConnectionError(f"PostgreSQL not ready after recreation: {e}") from e

        self._is_connected = True
        return self._session_factory  # type: ignore[return-value]

    def get_client(self) -> scoped_session:
        """Sync wrapper around aget_client()."""
        return self._run_async_from_sync(self.aget_client())

    async def ais_client_connected(self) -> bool:
        """Read-only check. Does not create or reconnect."""
        if self._engine is None or self._session_factory is None:
            return False
        try:
            await self._verify_connection()
            return True
        except Exception:
            return False

    def is_client_connected(self) -> bool:
        """Sync wrapper around ais_client_connected()."""
        return self._run_async_from_sync(self.ais_client_connected())

    async def aconnect(self) -> None:
        """Idempotent connection. Delegates to aget_client()."""
        if await self.ais_client_connected():
            logger.info("Already connected to PostgreSQL.")
            return

        try:
            await self.aget_client()
            logger.info("Successfully connected to PostgreSQL database")
        except VectorDBConnectionError:
            self._engine = None
            self._session_factory = None
            self._is_connected = False
            raise
        except Exception as e:
            self._engine = None
            self._session_factory = None
            self._is_connected = False
            raise VectorDBConnectionError(f"Connection failed: {e}") from e

    async def _verify_connection(self) -> None:
        try:
            def _verify() -> None:
                with self._session_factory() as session:  # type: ignore[misc]
                    session.execute(text("SELECT 1"))

            await asyncio.to_thread(_verify)
        except Exception as e:
            raise VectorDBConnectionError(f"Connection verification failed: {e}") from e

    async def adisconnect(self) -> None:
        try:
            if self._session_factory:
                await asyncio.to_thread(self._session_factory.remove)
            if self._engine:
                await asyncio.to_thread(self._engine.dispose)
            self._is_connected = False
            logger.info("Disconnected from PostgreSQL database")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            raise VectorDBError(f"Disconnect failed: {e}") from e

    async def ais_ready(self) -> bool:
        return await self.ais_client_connected()

    # ========================================================================
    # Schema Management
    # ========================================================================

    def _get_table_schema(self) -> Table:
        if self._config.schema_version == 1:
            return self._get_table_schema_v1()
        else:
            raise NotImplementedError(
                f"Unsupported schema version: {self._config.schema_version}"
            )

    def _get_table_schema_v1(self) -> Table:
        """
        Schema v1 with chunk-primary, document-optional property hierarchy.

        Primary (chunk-focused, always required):
        - chunk_id (String, unique, indexed)
        - chunk_content_hash (String, unique, indexed) — enables ON CONFLICT dedup
        - content (Text, BM25 searchable)
        - metadata (JSONB)

        Optional (document-focused):
        - document_name (Text/String, nullable, optionally indexed)
        - document_id (Text/String, nullable, optionally indexed)
        - doc_content_hash (String, indexed)
        """
        if self._config.vector_size is None:
            raise ValueError("vector_size must be set in config")

        indexed_fields_config = self._parse_indexed_fields()

        doc_name_type = String if indexed_fields_config.get("document_name", {}).get("type", "text") in ["keyword", "string", "varchar"] else Text
        doc_id_type = String if indexed_fields_config.get("document_id", {}).get("type", "text") in ["keyword", "string", "varchar"] else Text

        table = Table(
            self.table_name,
            self._metadata,
            Column("id", String, primary_key=True),

            # Primary chunk identifiers
            Column("chunk_id", String, unique=True, nullable=False, index=True),
            Column("chunk_content_hash", String, unique=True, nullable=False, index=True),

            # Content and embedding
            Column("content", Text, nullable=False),
            Column("embedding", Vector(self._config.vector_size), nullable=False),

            # Flexible metadata storage
            Column("metadata", postgresql.JSONB, server_default=text("'{}'::jsonb")),

            # Optional document-level identifiers
            Column("document_name", doc_name_type, nullable=True, index="document_name" in indexed_fields_config),
            Column("document_id", doc_id_type, nullable=True, index="document_id" in indexed_fields_config),
            Column("doc_content_hash", String, nullable=True, index=True),

            # KnowledgeBase isolation
            Column("knowledge_base_id", String, nullable=True, index=True),

            # Timestamps
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            Column("updated_at", DateTime(timezone=True), onupdate=func.now()),

            extend_existing=True,
        )

        Index(f"idx_{self.table_name}_id", table.c.id)
        Index(f"idx_{self.table_name}_chunk_id", table.c.chunk_id)
        Index(f"idx_{self.table_name}_chunk_content_hash", table.c.chunk_content_hash)

        if indexed_fields_config:
            for field_name, field_config in indexed_fields_config.items():
                if field_name in {'id', 'chunk_id', 'chunk_content_hash'}:
                    continue
                elif field_name in {'document_name', 'document_id'}:
                    continue
                elif field_name == 'content':
                    if field_config.get("type", "text") == "text":
                        logger.debug("Full-text search on 'content' field will use GIN index")
                elif field_name == 'metadata':
                    logger.debug("JSONB metadata field will use GIN index")

        return table

    async def acreate_collection(self) -> None:
        session_factory = await self.aget_client()

        try:
            exists = await self.acollection_exists()

            if exists:
                if self._config.recreate_if_exists:
                    logger.info(f"Collection '{self.table_name}' exists, recreating...")
                    await self.adelete_collection()
                else:
                    logger.info(f"Collection '{self.table_name}' already exists")
                    return

            def _create() -> None:
                with session_factory() as session:
                    with session.begin():
                        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                        if self.schema_name and self.schema_name != "public":
                            session.execute(
                                text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema_name}";')
                            )

                self._table.create(self._engine)  # type: ignore[union-attr]

            await asyncio.to_thread(_create)

            await self._create_vector_index()
            await self._create_gin_index()

            if self._config.indexed_fields and 'metadata' in self._config.indexed_fields:
                await self._create_metadata_indexes()

            logger.info(f"Successfully created collection '{self.table_name}'")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise VectorDBError(f"Collection creation failed: {e}") from e

    async def adelete_collection(self) -> None:
        await self.aget_client()

        try:
            exists = await self.acollection_exists()
            if not exists:
                raise CollectionDoesNotExistError(
                    f"Collection '{self.table_name}' does not exist"
                )

            def _drop() -> None:
                self._table.drop(self._engine)  # type: ignore[union-attr]

            await asyncio.to_thread(_drop)
            logger.info(f"Successfully deleted collection '{self.table_name}'")

        except CollectionDoesNotExistError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorDBError(f"Collection deletion failed: {e}") from e

    async def acollection_exists(self) -> bool:
        if not self._is_connected or not self._engine:
            return False

        try:
            def _check() -> bool:
                return inspect(self._engine).has_table(
                    self.table_name,
                    schema=self.schema_name
                )

            return await asyncio.to_thread(_check)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    # ========================================================================
    # Data Operations
    # ========================================================================

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
        Upsert data with content-based deduplication via ON CONFLICT (chunk_content_hash).

        Auto-computes chunk_content_hash from chunk text if not provided.
        """
        await self.aget_client()

        if (vectors is None or len(vectors) == 0) and (sparse_vectors is None or len(sparse_vectors) == 0):
            info_log("Nothing to upsert: no dense or sparse vectors provided.", context="PgVectorDB")
            return

        if not await self.acollection_exists():
            logger.warning(f"Collection '{self.table_name}' does not exist. Create it first.")
            return

        n: int = len(vectors) if vectors else len(sparse_vectors)  # type: ignore[arg-type]

        # Length validation for all provided per-item arrays
        for name, arr in (
            ("payloads", payloads),
            ("ids", ids),
            ("chunks", chunks),
            ("document_ids", document_ids),
            ("document_names", document_names),
            ("doc_content_hashes", doc_content_hashes),
            ("chunk_content_hashes", chunk_content_hashes),
            ("knowledge_base_ids", knowledge_base_ids),
        ):
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"Length mismatch: '{name}' has {len(arr)} items, expected {n}"
                )

        try:
            _warned_standard_keys: set = set()
            records = self._prepare_records(
                n=n,
                vectors=vectors,
                payloads=payloads,
                ids=ids,
                chunks=chunks,
                document_ids=document_ids,
                document_names=document_names,
                doc_content_hashes=doc_content_hashes,
                chunk_content_hashes=chunk_content_hashes,
                knowledge_base_ids=knowledge_base_ids,
                extra_metadata=metadata,
                _warned_standard_keys=_warned_standard_keys,
            )

            batch_size: int = self._config.batch_size
            await self._batch_upsert(records, batch_size)

            logger.info(f"Successfully upserted {len(records)} records")

        except UpsertError:
            raise
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise UpsertError(f"Failed to upsert data: {e}") from e

    def _prepare_records(
        self,
        n: int,
        vectors: Optional[List[List[float]]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        chunks: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        document_names: Optional[List[str]] = None,
        doc_content_hashes: Optional[List[str]] = None,
        chunk_content_hashes: Optional[List[str]] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        _warned_standard_keys: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build records with explicit structural IDs. Payload is treated as pure metadata.
        Non-standard keys from payload go into the JSONB metadata column.
        """
        records: List[Dict[str, Any]] = []

        for i in range(n):
            payload: Dict[str, Any] = payloads[i] if payloads and i < len(payloads) else {}
            vector: Optional[List[float]] = vectors[i] if vectors and i < len(vectors) else None

            content: str = chunks[i] if chunks and i < len(chunks) else ""
            content = self._clean_content(content)

            chunk_id_str: str = str(ids[i]) if ids and i < len(ids) else str(_uuid.uuid4())
            chunk_hash: str = (
                chunk_content_hashes[i]
                if chunk_content_hashes and i < len(chunk_content_hashes)
                else hashlib.md5(content.encode("utf-8")).hexdigest()
            )
            doc_id: str = document_ids[i] if document_ids and i < len(document_ids) else ""
            doc_hash: str = doc_content_hashes[i] if doc_content_hashes and i < len(doc_content_hashes) else ""
            doc_name: str = document_names[i] if document_names and i < len(document_names) else ""

            combined_metadata: Dict[str, Any] = {}
            if self._config.default_metadata:
                combined_metadata.update(self._config.default_metadata)

            # If payload contains a nested 'metadata' dict, merge its contents
            # (flatten that specific nesting level to match Milvus/Qdrant semantics).
            nested_meta = payload.get("metadata")
            if isinstance(nested_meta, dict):
                combined_metadata.update(nested_meta)

            # Non-standard payload keys go into JSONB metadata. Standard fields
            # are ignored when present in payload and a warning is emitted.
            for key, value in payload.items():
                if key in self._STANDARD_FIELDS:
                    if key == "metadata":
                        continue
                    if _warned_standard_keys is not None and key not in _warned_standard_keys:
                        _warned_standard_keys.add(key)
                        debug_log(
                            f"Standard field '{key}' found in payload dict and will be ignored. "
                            f"Standard fields must be passed via dedicated parameters "
                            f"(ids, document_ids, document_names, doc_content_hashes, "
                            f"chunk_content_hashes, knowledge_base_ids).",
                            context="PgVectorVectorDB",
                        )
                    continue
                combined_metadata[key] = value

            if extra_metadata:
                combined_metadata.update(extra_metadata)

            kbi: Optional[str] = knowledge_base_ids[i] if knowledge_base_ids else None

            record: Dict[str, Any] = {
                'id': chunk_id_str,
                'chunk_id': chunk_id_str,
                'chunk_content_hash': chunk_hash,
                'content': content,
                'embedding': vector,
                'metadata': combined_metadata,
                'document_name': doc_name or None,
                'document_id': doc_id or None,
                'doc_content_hash': doc_hash or None,
                'knowledge_base_id': kbi,
            }

            records.append(record)

        return records

    def _clean_content(self, content: str) -> str:
        """PostgreSQL doesn't accept null characters in TEXT fields."""
        return content.replace("\x00", "\ufffd")

    async def _batch_upsert(
        self,
        records: List[Dict[str, Any]],
        batch_size: int
    ) -> None:
        """
        Batch upsert using PostgreSQL native ON CONFLICT (chunk_content_hash) DO UPDATE.
        Provides atomic content-based deduplication with zero extra roundtrips.
        """
        def _upsert_batch(batch: List[Dict[str, Any]]) -> None:
            with self._session_factory() as session:  # type: ignore[misc]
                with session.begin():
                    insert_stmt = postgresql.insert(self._table).values(batch)
                    upsert_stmt = insert_stmt.on_conflict_do_update(
                        index_elements=['chunk_content_hash'],
                        set_={
                            'id': insert_stmt.excluded.id,
                            'chunk_id': insert_stmt.excluded.chunk_id,
                            'document_name': insert_stmt.excluded.document_name,
                            'document_id': insert_stmt.excluded.document_id,
                            'doc_content_hash': insert_stmt.excluded.doc_content_hash,
                            'content': insert_stmt.excluded.content,
                            'embedding': insert_stmt.excluded.embedding,
                            'metadata': insert_stmt.excluded.metadata,
                            'knowledge_base_id': insert_stmt.excluded.knowledge_base_id,
                        }
                    )
                    session.execute(upsert_stmt)

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            logger.debug(f"Upserting batch {i // batch_size + 1} ({len(batch)} records)")
            await asyncio.to_thread(_upsert_batch, batch)

    async def adelete(
        self,
        ids: List[Union[str, int]],
    ) -> None:
        """
        Delete records by their IDs (primary key = chunk_id).
        Pre-checks existence and reports accurate counts.
        """
        session_factory = await self.aget_client()

        if not ids:
            return

        try:
            str_ids: List[str] = [str(i) for i in ids]

            def _delete() -> int:
                with session_factory() as session:
                    with session.begin():
                        existing_stmt = select(self._table.c.id).where(
                            self._table.c.id.in_(str_ids)
                        )
                        existing_rows = session.execute(existing_stmt).fetchall()
                        existing_ids: set[str] = {row[0] for row in existing_rows}

                        for sid in str_ids:
                            if sid not in existing_ids:
                                logger.debug(
                                    f"Record with ID '{sid}' does not exist, skipping deletion."
                                )

                        if not existing_ids:
                            return 0

                        del_stmt = sa_delete(self._table).where(
                            self._table.c.id.in_(list(existing_ids))
                        )
                        session.execute(del_stmt)
                        return len(existing_ids)

            deleted_count: int = await asyncio.to_thread(_delete)
            if deleted_count == 0:
                logger.info("No matching records found to delete. No action taken.")
            else:
                logger.info(
                    f"Successfully processed deletion request for {len(str_ids)} IDs. "
                    f"Existed: {deleted_count}, Deleted: {deleted_count}."
                )

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise VectorDBError(f"Failed to delete records: {e}") from e

    async def afetch(
        self,
        ids: List[Union[str, int]],
    ) -> List[VectorSearchResult]:
        """
        Retrieve records by their IDs (primary key = chunk_id).
        """
        session_factory = await self.aget_client()

        if not ids:
            return []

        try:
            def _fetch() -> List[Any]:
                with session_factory() as session:
                    stmt = select(self._table).where(
                        self._table.c.id.in_([str(i) for i in ids])
                    )
                    result = session.execute(stmt)
                    return result.fetchall()

            rows = await asyncio.to_thread(_fetch)

            results: List[VectorSearchResult] = []
            for row in rows:
                results.append(
                    VectorSearchResult(
                        id=row.chunk_id,
                        score=1.0,
                        payload=self._row_to_payload(row),
                        vector=list(row.embedding) if row.embedding is not None else None,
                        text=row.content
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            raise VectorDBError(f"Failed to fetch records: {e}") from e

    def _row_to_payload(self, row: Any) -> Dict[str, Any]:
        raw_meta = getattr(row, "metadata", None)
        kbi = getattr(row, "knowledge_base_id", None)
        return {
            "chunk_id": row.chunk_id or "",
            "document_id": row.document_id or "",
            "document_name": row.document_name or "",
            "content": row.content or "",
            "doc_content_hash": row.doc_content_hash or "",
            "chunk_content_hash": row.chunk_content_hash or "",
            "knowledge_base_id": kbi or "",
            "metadata": dict(raw_meta) if raw_meta else {},
        }

    # ========================================================================
    # Search Operations
    # ========================================================================

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
        top_k = top_k if top_k is not None else self._config.default_top_k
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold
        alpha = alpha if alpha is not None else self._config.default_hybrid_alpha
        fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method

        has_vector = query_vector is not None
        has_text = query_text is not None

        if has_vector and has_text:
            if not self._config.hybrid_search_enabled:
                raise ConfigurationError("Hybrid search is disabled in configuration")
            return await self.ahybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                alpha=alpha,
                fusion_method=fusion_method,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )
        elif has_vector:
            if not self._config.dense_search_enabled:
                raise ConfigurationError("Dense search is disabled in configuration")
            return await self.adense_search(
                query_vector=query_vector,
                top_k=top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
            )
        elif has_text:
            if not self._config.full_text_search_enabled:
                raise ConfigurationError("Full-text search is disabled in configuration")
            return await self.afull_text_search(
                query_text=query_text,
                top_k=top_k,
                filter=filter,
                similarity_threshold=similarity_threshold,
                apply_reranking=apply_reranking,
                sparse_query_vector=sparse_query_vector,
            )
        else:
            raise ConfigurationError(
                "Must provide either query_vector, query_text, or both"
            )

    async def adense_search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
    ) -> List[VectorSearchResult]:
        session_factory = await self.aget_client()

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold

        try:
            def _search() -> List[Any]:
                with session_factory() as session:
                    with session.begin():
                        self._set_index_params(session)

                        stmt = select(
                            self._table.c.id,
                            self._table.c.chunk_id,
                            self._table.c.chunk_content_hash,
                            self._table.c.document_name,
                            self._table.c.document_id,
                            self._table.c.doc_content_hash,
                            self._table.c.content,
                            self._table.c.embedding,
                            self._table.c.metadata,
                            self._table.c.knowledge_base_id,
                            self._table.c.created_at,
                            self._table.c.updated_at,
                        )

                        if filter:
                            stmt = self._apply_filter(stmt, filter)

                        distance_col = self._get_distance_column(query_vector)
                        stmt = stmt.add_columns(distance_col.label('distance'))
                        stmt = stmt.order_by('distance')

                        if final_similarity_threshold is not None and final_similarity_threshold > 0.0:
                            max_distance = self._similarity_to_distance(final_similarity_threshold)
                            if max_distance != float('inf') and max_distance != float('-inf'):
                                stmt = stmt.where(distance_col <= max_distance)

                        stmt = stmt.limit(top_k)
                        result = session.execute(stmt)
                        return result.fetchall()

            rows = await asyncio.to_thread(_search)

            results: List[VectorSearchResult] = []
            for row in rows:
                score = self._distance_to_similarity(row.distance)
                results.append(
                    VectorSearchResult(
                        id=row.chunk_id,
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
            raise SearchError(f"Dense search failed: {e}") from e

    async def afull_text_search(
        self,
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        apply_reranking: bool = True,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        session_factory = await self.aget_client()

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.default_similarity_threshold

        try:
            def _search() -> List[Any]:
                with session_factory() as session:
                    processed_query = self._process_text_query(query_text)

                    stmt = select(
                        self._table.c.id,
                        self._table.c.chunk_id,
                        self._table.c.chunk_content_hash,
                        self._table.c.document_name,
                        self._table.c.document_id,
                        self._table.c.doc_content_hash,
                        self._table.c.content,
                        self._table.c.embedding,
                        self._table.c.metadata,
                        self._table.c.knowledge_base_id,
                        self._table.c.created_at,
                        self._table.c.updated_at,
                    )

                    if filter:
                        stmt = self._apply_filter(stmt, filter)

                    ts_vector = func.to_tsvector(
                        self._config.content_language,
                        self._table.c.content
                    )
                    ts_query = func.websearch_to_tsquery(
                        self._config.content_language,
                        bindparam("query", value=processed_query)
                    )
                    text_rank = func.ts_rank_cd(ts_vector, ts_query)

                    stmt = stmt.add_columns(text_rank.label('rank'))
                    stmt = stmt.where(text_rank > 0)
                    stmt = stmt.order_by(desc('rank'))

                    if final_similarity_threshold is not None:
                        stmt = stmt.where(text_rank >= final_similarity_threshold)

                    stmt = stmt.limit(top_k)
                    result = session.execute(stmt)
                    return result.fetchall()

            rows = await asyncio.to_thread(_search)

            results: List[VectorSearchResult] = []

            if rows:
                max_rank = max(float(row.rank) for row in rows)
                if max_rank == 0:
                    max_rank = 1.0

                for row in rows:
                    normalized_score = float(row.rank) / max_rank
                    results.append(
                        VectorSearchResult(
                            id=row.chunk_id,
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
            raise SearchError(f"Full-text search failed: {e}") from e

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
        await self.aget_client()

        alpha = alpha if alpha is not None else self._config.default_hybrid_alpha
        fusion_method = fusion_method if fusion_method is not None else self._config.default_fusion_method

        try:
            if fusion_method == 'rrf':
                return await self._hybrid_search_rrf(
                    query_vector, query_text, top_k, filter, similarity_threshold
                )
            else:
                return await self._hybrid_search_weighted(
                    query_vector, query_text, top_k, filter, alpha, similarity_threshold
                )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}") from e

    async def _hybrid_search_weighted(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filter: Optional[Dict[str, Any]],
        alpha: float,
        similarity_threshold: Optional[float],
    ) -> List[VectorSearchResult]:
        def _search() -> List[Any]:
            with self._session_factory() as session:  # type: ignore[misc]
                with session.begin():
                    self._set_index_params(session)
                    processed_query = self._process_text_query(query_text)

                    stmt = select(
                        self._table.c.id,
                        self._table.c.chunk_id,
                        self._table.c.chunk_content_hash,
                        self._table.c.document_name,
                        self._table.c.document_id,
                        self._table.c.doc_content_hash,
                        self._table.c.content,
                        self._table.c.embedding,
                        self._table.c.metadata,
                        self._table.c.knowledge_base_id,
                        self._table.c.created_at,
                        self._table.c.updated_at,
                    )

                    if filter:
                        stmt = self._apply_filter(stmt, filter)

                    distance_col = self._get_distance_column(query_vector)
                    vector_score = self._distance_to_similarity_expression(distance_col)

                    ts_vector = func.to_tsvector(
                        self._config.content_language,
                        self._table.c.content
                    )
                    ts_query = func.websearch_to_tsquery(
                        self._config.content_language,
                        bindparam("query", value=processed_query)
                    )
                    text_rank = func.ts_rank_cd(ts_vector, ts_query)

                    text_weight = 1.0 - alpha
                    hybrid_score = (alpha * vector_score) + (text_weight * text_rank)

                    stmt = stmt.add_columns(
                        distance_col.label('distance'),
                        text_rank.label('text_rank'),
                        hybrid_score.label('hybrid_score')
                    )

                    stmt = stmt.order_by(desc('hybrid_score'))

                    if similarity_threshold is not None:
                        stmt = stmt.where(hybrid_score >= similarity_threshold)

                    stmt = stmt.limit(top_k)
                    result = session.execute(stmt)
                    return result.fetchall()

        rows = await asyncio.to_thread(_search)

        raw_scores = [float(row.hybrid_score) for row in rows]

        if raw_scores:
            max_score = max(raw_scores)
            min_score = min(raw_scores)
            score_range = max_score - min_score
        else:
            score_range = 0.0

        results: List[VectorSearchResult] = []
        for i, row in enumerate(rows):
            if score_range > 0:
                normalized_score = (raw_scores[i] - min_score) / score_range
            else:
                normalized_score = 1.0 if raw_scores else 0.0
            normalized_score = max(0.0, min(1.0, normalized_score))

            results.append(
                VectorSearchResult(
                    id=row.chunk_id,
                    score=normalized_score,
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
    ) -> List[VectorSearchResult]:
        k: int = self._config.rrf_k

        vector_results = await self.adense_search(
            query_vector=query_vector,
            top_k=top_k * 2,
            filter=filter,
        )

        text_results = await self.afull_text_search(
            query_text=query_text,
            top_k=top_k * 2,
            filter=filter,
        )

        vector_ranks: Dict[Any, int] = {r.id: i + 1 for i, r in enumerate(vector_results)}
        text_ranks: Dict[Any, int] = {r.id: i + 1 for i, r in enumerate(text_results)}

        rrf_scores: Dict[Any, float] = {}
        all_ids = set(vector_ranks.keys()) | set(text_ranks.keys())

        for doc_id in all_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += 1.0 / (k + vector_ranks[doc_id])
            if doc_id in text_ranks:
                score += 1.0 / (k + text_ranks[doc_id])
            rrf_scores[doc_id] = score

        if rrf_scores:
            max_score = max(rrf_scores.values())
            min_score = min(rrf_scores.values())
            score_range = max_score - min_score
            if score_range > 0:
                rrf_scores = {
                    doc_id: (score - min_score) / score_range
                    for doc_id, score in rrf_scores.items()
                }
            else:
                rrf_scores = {doc_id: 1.0 for doc_id in rrf_scores}

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        result_map: Dict[Any, VectorSearchResult] = {}
        for r in vector_results + text_results:
            if r.id not in result_map:
                result_map[r.id] = r

        results: List[VectorSearchResult] = []
        for doc_id in sorted_ids[:top_k]:
            result = result_map[doc_id]
            results.append(
                VectorSearchResult(
                    id=result.id,
                    score=rrf_scores[doc_id],
                    payload=result.payload,
                    vector=result.vector,
                    text=result.text
                )
            )

        if similarity_threshold is not None:
            results = [r for r in results if r.score >= similarity_threshold]

        logger.debug(f"Hybrid search (RRF) returned {len(results)} results")
        return results

    # ========================================================================
    # Helper Methods for Search
    # ========================================================================

    def _set_index_params(self, session: Session) -> None:
        if isinstance(self._config.index, IVFIndexConfig):
            nprobe = self._config.index.nprobe if self._config.index.nprobe else 10
            session.execute(
                text(f"SET LOCAL ivfflat.probes = {nprobe}")
            )
        elif isinstance(self._config.index, HNSWIndexConfig):
            ef_search = self._config.index.ef_search if self._config.index.ef_search else 40
            session.execute(
                text(f"SET LOCAL hnsw.ef_search = {ef_search}")
            )

    def _get_distance_column(self, query_vector: List[float]) -> Any:
        if self._config.distance_metric == DistanceMetric.COSINE:
            return self._table.c.embedding.cosine_distance(query_vector)
        elif self._config.distance_metric == DistanceMetric.EUCLIDEAN:
            return self._table.c.embedding.l2_distance(query_vector)
        elif self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            return self._table.c.embedding.max_inner_product(query_vector)
        else:
            raise ConfigurationError(f"Unsupported distance metric: {self._config.distance_metric}")

    def _distance_to_similarity(self, distance: float) -> float:
        if self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            import math
            return 1.0 / (1.0 + math.exp(-distance))
        elif self._config.distance_metric == DistanceMetric.COSINE:
            return max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        else:
            return 1.0 / (1.0 + distance)

    def _similarity_to_distance(self, similarity: float) -> float:
        import math

        if similarity <= 0.0:
            return float('inf')

        if self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            if similarity >= 1.0:
                return float('inf')
            return -math.log((1.0 / similarity) - 1.0)
        elif self._config.distance_metric == DistanceMetric.COSINE:
            return (1.0 - similarity) * 2.0
        else:
            return (1.0 / similarity) - 1.0

    def _distance_to_similarity_expression(self, distance_col: Any) -> Any:
        if self._config.distance_metric == DistanceMetric.DOT_PRODUCT:
            return 1.0 / (1.0 + func.exp(-distance_col))
        elif self._config.distance_metric == DistanceMetric.COSINE:
            return func.greatest(0.0, func.least(1.0, 1.0 - (distance_col / 2.0)))
        else:
            return 1.0 / (1.0 + distance_col)

    def _process_text_query(self, query: str) -> str:
        if self._config.prefix_match:
            words = query.strip().split()
            processed_words = [word + "*" for word in words]
            return " ".join(processed_words)
        return query

    def _build_where_clauses(self, filter: Dict[str, Any]) -> List[Any]:
        _COLUMN_FILTER_KEYS: frozenset = frozenset({
            'document_name', 'document_id', 'chunk_id',
            'doc_content_hash', 'chunk_content_hash', 'knowledge_base_id',
        })
        clauses: List[Any] = []
        for key, value in filter.items():
            if key in _COLUMN_FILTER_KEYS:
                column = getattr(self._table.c, key)
                clauses.append(column == value)
            elif key.startswith("metadata."):
                # Support nested JSONB access: 'metadata.a.b' -> metadata->'a'->>'b'
                parts = key.split(".")[1:]
                col = self._table.c.metadata
                for p in parts[:-1]:
                    col = col[p]
                clauses.append(col[parts[-1]].astext == str(value))
            else:
                clauses.append(
                    self._table.c.metadata[key].astext == str(value)
                )
        return clauses

    def _apply_filter(self, stmt: Any, filter: Dict[str, Any]) -> Any:
        clauses = self._build_where_clauses(filter)
        if clauses:
            stmt = stmt.where(and_(*clauses))
        return stmt

    # ========================================================================
    # Index Management
    # ========================================================================

    async def _create_vector_index(self) -> None:
        try:
            index_type = 'hnsw' if isinstance(self._config.index, HNSWIndexConfig) else 'ivfflat'
            self._vector_index_name = f"{self.table_name}_{index_type}_embedding_idx"

            if await self._index_exists(self._vector_index_name):
                logger.info(f"Vector index '{self._vector_index_name}' already exists")
                return

            distance_op = self._get_distance_operator()

            def _create() -> None:
                with self._session_factory() as session:  # type: ignore[misc]
                    with session.begin():
                        if isinstance(self._config.index, HNSWIndexConfig):
                            self._create_hnsw_index(session, distance_op)
                        else:
                            self._create_ivfflat_index(session, distance_op)

            await asyncio.to_thread(_create)
            logger.info(f"Created vector index '{self._vector_index_name}'")

        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            raise VectorDBError(f"Vector index creation failed: {e}") from e

    def _create_hnsw_index(self, session: Session, distance_op: str) -> None:
        config = cast(HNSWIndexConfig, self._config.index)

        m_val = int(config.m)
        ef_val = int(config.ef_construction)

        create_sql = text(
            f'CREATE INDEX "{self._vector_index_name}" '
            f'ON "{self.schema_name}"."{self.table_name}" '
            f'USING hnsw (embedding {distance_op}) '
            f'WITH (m = {m_val}, ef_construction = {ef_val});'
        )

        session.execute(create_sql)

    def _create_ivfflat_index(self, session: Session, distance_op: str) -> None:
        num_lists = self._calculate_ivfflat_lists(session)

        lists_val = int(num_lists)

        create_sql = text(
            f'CREATE INDEX "{self._vector_index_name}" '
            f'ON "{self.schema_name}"."{self.table_name}" '
            f'USING ivfflat (embedding {distance_op}) '
            f'WITH (lists = {lists_val});'
        )

        session.execute(create_sql)

    def _calculate_ivfflat_lists(self, session: Session) -> int:
        count_stmt = select(func.count()).select_from(self._table)
        result = session.execute(count_stmt)
        row_count: int = result.scalar() or 0

        if row_count < 1000000:
            return max(int(row_count / 1000), 1)
        return max(int(sqrt(row_count)), 1)

    def _get_distance_operator(self) -> str:
        metric_map = {
            DistanceMetric.COSINE: 'vector_cosine_ops',
            DistanceMetric.EUCLIDEAN: 'vector_l2_ops',
            DistanceMetric.DOT_PRODUCT: 'vector_ip_ops'
        }
        return metric_map[self._config.distance_metric]

    async def _create_gin_index(self) -> None:
        try:
            self._gin_index_name = f"{self.table_name}_content_gin_idx"

            if await self._index_exists(self._gin_index_name):
                logger.info(f"GIN index '{self._gin_index_name}' already exists")
                return

            def _create() -> None:
                with self._session_factory() as session:  # type: ignore[misc]
                    with session.begin():
                        language = self._config.content_language
                        safe_language = ''.join(c for c in language if c.isalnum() or c == '_')
                        create_sql = text(
                            f'CREATE INDEX "{self._gin_index_name}" '
                            f'ON "{self.schema_name}"."{self.table_name}" '
                            f"USING GIN (to_tsvector('{safe_language}', content));"
                        )
                        session.execute(create_sql)

            await asyncio.to_thread(_create)
            logger.info(f"Created GIN index '{self._gin_index_name}'")

        except Exception as e:
            logger.error(f"Failed to create GIN index: {e}")
            raise VectorDBError(f"GIN index creation failed: {e}") from e

    async def _create_metadata_indexes(self) -> None:
        try:
            metadata_index_name = f"{self.table_name}_metadata_gin_idx"

            if await self._index_exists(metadata_index_name):
                logger.info(f"Metadata index '{metadata_index_name}' already exists")
                return

            def _create() -> None:
                with self._session_factory() as session:  # type: ignore[misc]
                    with session.begin():
                        create_sql = text(
                            f'CREATE INDEX "{metadata_index_name}" '
                            f'ON "{self.schema_name}"."{self.table_name}" '
                            f'USING GIN (metadata);'
                        )
                        session.execute(create_sql)

            await asyncio.to_thread(_create)
            logger.info(f"Created metadata index '{metadata_index_name}'")

        except Exception as e:
            logger.error(f"Failed to create metadata index: {e}")
            raise VectorDBError(f"Metadata index creation failed: {e}") from e

    async def _index_exists(self, index_name: str) -> bool:
        try:
            def _check() -> bool:
                inspector = inspect(self._engine)
                indexes = inspector.get_indexes(self.table_name, schema=self.schema_name)
                return any(idx['name'] == index_name for idx in indexes)

            return await asyncio.to_thread(_check)
        except Exception:
            return False

    async def aoptimize(self, force_recreate: bool = False) -> bool:
        logger.info("Optimizing PgVector database...")
        try:
            if force_recreate:
                if self._vector_index_name:
                    await self._drop_index(self._vector_index_name)
                if self._gin_index_name:
                    await self._drop_index(self._gin_index_name)

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
        try:
            def _drop() -> None:
                with self._session_factory() as session:  # type: ignore[misc]
                    with session.begin():
                        session.execute(
                            text(f'DROP INDEX IF EXISTS {self.schema_name}."{index_name}";')
                        )

            await asyncio.to_thread(_drop)
            logger.info(f"Dropped index '{index_name}'")
        except Exception as e:
            logger.error(f"Failed to drop index '{index_name}': {e}")

    # ========================================================================
    # Generic Field Helpers
    # ========================================================================

    async def afield_exists(self, field_name: str, field_value: Any) -> bool:
        """Generic check if any record with the given field value exists."""
        if not self._is_connected:
            return False

        try:
            def _check() -> bool:
                with self._session_factory() as session:  # type: ignore[misc]
                    column = getattr(self._table.c, field_name)
                    stmt = select(1).where(column == field_value).limit(1)
                    result = session.execute(stmt).first()
                    return result is not None

            return await asyncio.to_thread(_check)
        except Exception as e:
            logger.error(f"Error checking if {field_name}='{field_value}' exists: {e}")
            return False

    async def adelete_by_field(self, field_name: str, field_value: Any) -> bool:
        """
        Generic deletion of all records matching a column value.
        Includes pre-existence check. Idempotent — returns True if no matches.
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")

        try:
            exists = await self.afield_exists(field_name, field_value)
            if not exists:
                logger.warning(f"No records found with {field_name}='{field_value}'")
                return True

            def _delete() -> int:
                with self._session_factory() as session:  # type: ignore[misc]
                    with session.begin():
                        column = getattr(self._table.c, field_name)
                        stmt = sa_delete(self._table).where(column == field_value)
                        result = session.execute(stmt)
                        return result.rowcount

            count = await asyncio.to_thread(_delete)
            logger.info(f"Deleted {count} records with {field_name}='{field_value}'")
            return True

        except VectorDBConnectionError:
            raise
        except Exception as e:
            logger.error(f"Delete by {field_name} failed: {e}")
            return False

    # ========================================================================
    # Utility / Count
    # ========================================================================

    async def aget_count(self) -> int:
        if not self._is_connected:
            return 0

        try:
            def _count() -> int:
                with self._session_factory() as session:  # type: ignore[misc]
                    stmt = select(func.count()).select_from(self._table)
                    result = session.execute(stmt)
                    return result.scalar() or 0

            return await asyncio.to_thread(_count)
        except Exception:
            return 0

    async def aid_exists(self, id: str) -> bool:
        return await self.afield_exists('id', id)

    # ========================================================================
    # Existence Check Methods
    # ========================================================================

    async def achunk_id_exists(self, chunk_id: str) -> bool:
        return await self.afield_exists('chunk_id', chunk_id)

    async def adoc_content_hash_exists(self, doc_content_hash: str) -> bool:
        return await self.afield_exists('doc_content_hash', doc_content_hash)

    async def achunk_content_hash_exists(self, chunk_content_hash: str) -> bool:
        return await self.afield_exists('chunk_content_hash', chunk_content_hash)

    async def adocument_name_exists(self, document_name: str) -> bool:
        return await self.afield_exists('document_name', document_name)

    async def adocument_id_exists(self, document_id: str) -> bool:
        return await self.afield_exists('document_id', document_id)

    # ========================================================================
    # Delete-by-Field Methods
    # ========================================================================

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
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")

        try:
            def _delete() -> int:
                with self._session_factory() as session:  # type: ignore[misc]
                    with session.begin():
                        stmt = sa_delete(self._table)
                        clauses = self._build_where_clauses(metadata)
                        if clauses:
                            stmt = stmt.where(and_(*clauses))
                        result = session.execute(stmt)
                        return result.rowcount

            count = await asyncio.to_thread(_delete)
            logger.info(f"Deleted {count} records matching metadata filter")
            return True

        except Exception as e:
            logger.error(f"Delete by metadata failed: {e}")
            return False

    # ========================================================================
    # Update Metadata
    # ========================================================================

    async def aupdate_metadata(
        self,
        chunk_id: str,
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Update metadata for a record identified by chunk_id.
        Single record since chunk_id is unique.
        """
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")

        try:
            def _update() -> None:
                with self._session_factory() as session:  # type: ignore[misc]
                    with session.begin():
                        if merge:
                            stmt = (
                                sa_update(self._table)
                                .where(self._table.c.chunk_id == chunk_id)
                                .values(
                                    metadata=func.coalesce(
                                        self._table.c.metadata, text("'{}'::jsonb")
                                    ).op("||")(
                                        bindparam("md", metadata, type_=postgresql.JSONB)
                                    )
                                )
                            )
                        else:
                            stmt = (
                                sa_update(self._table)
                                .where(self._table.c.chunk_id == chunk_id)
                                .values(metadata=metadata)
                            )

                        session.execute(stmt)

            await asyncio.to_thread(_update)
            logger.info(f"Updated metadata for chunk_id '{chunk_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False

    # ========================================================================
    # Supported Search Types
    # ========================================================================

    async def aget_supported_search_types(self) -> List[str]:
        supported: List[str] = []
        if self._config.dense_search_enabled:
            supported.append('dense')
        if self._config.full_text_search_enabled:
            supported.append('full_text')
        if self._config.hybrid_search_enabled:
            supported.append('hybrid')
        return supported

    # ========================================================================
    # Clear & Copy
    # ========================================================================

    async def aclear(self) -> None:
        if not self._is_connected:
            raise VectorDBConnectionError("Not connected to database")

        try:
            def _clear() -> None:
                with self._session_factory() as session:  # type: ignore[misc]
                    with session.begin():
                        session.execute(sa_delete(self._table))

            await asyncio.to_thread(_clear)
            logger.info(f"Cleared all records from collection '{self.table_name}'")

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise VectorDBError(f"Clear operation failed: {e}") from e

    def __deepcopy__(self, memo: Dict[int, Any]) -> PgVectorProvider:
        from copy import deepcopy

        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj

        for k, v in self.__dict__.items():
            if k in {'_metadata', '_table'}:
                continue
            elif k in {'_engine', '_session_factory'}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))

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


PgvectorProvider = PgVectorProvider
