"""Synchronous PostgreSQL storage implementation for Upsonic agent framework.

This module provides a synchronous PostgreSQL storage backend using SQLAlchemy
with psycopg2 driver. It supports session persistence and user memory storage
with full JSONB support for efficient querying.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from upsonic.session.base import SessionType, Session
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.schemas import KnowledgeRow

try:
    from sqlalchemy import Column, MetaData, Table, func, select, text
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.engine import Engine, create_engine
    from sqlalchemy.orm import scoped_session, sessionmaker
    from sqlalchemy.schema import Index
except ImportError:
    Column = None  # type: ignore
    MetaData = None  # type: ignore
    Table = None  # type: ignore
    func = None  # type: ignore
    select = None  # type: ignore
    text = None  # type: ignore
    postgresql = None  # type: ignore
    Engine = None  # type: ignore
    create_engine = None  # type: ignore
    scoped_session = None  # type: ignore
    sessionmaker = None  # type: ignore
    Index = None  # type: ignore

from upsonic.storage.base import Storage
from upsonic.storage.postgres.schemas import get_table_schema_definition
from upsonic.storage.postgres.utils import (
    apply_sorting,
    create_schema,
    is_table_available,
    is_valid_table,
    sanitize_postgres_dict,
    sanitize_postgres_list,
    sanitize_postgres_string,
)
from upsonic.storage.schemas import UserMemory
from upsonic.utils.logging_config import get_logger
from upsonic.storage.utils import deserialize_session

_logger = get_logger("upsonic.storage.postgres")


class PostgresStorage(Storage):
    """Synchronous PostgreSQL storage implementation for Upsonic agent framework.
    
    This storage backend uses SQLAlchemy with psycopg2 for sync PostgreSQL operations.
    It provides persistence for agent sessions and user memory data with full JSONB support.
    
    The following order is used to determine the database connection:
        1. Use the db_engine if provided
        2. Use the db_url if provided
        3. Raise an error if neither is provided
    
    Connection Pool Configuration:
        When creating an engine from db_url, the following settings are applied:
        - pool_pre_ping=True: Validates connections before use to handle terminated
          connections (e.g., "terminating connection due to administrator command")
        - pool_recycle=3600: Recycles connections after 1 hour to prevent stale connections
    
    Example:
        ```python
        storage = PostgresStorage(
            db_url="postgresql://user:pass@localhost:5432/db"
        )
        storage._create_all_tables()
        
        # Upsert a session
        session = AgentSession(session_id="abc123", ...)
        result = storage.upsert_session(session)
        
        # Get a session
        session = storage.get_session(session_id="abc123")
        ```
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        db_engine: Optional["Engine"] = None,
        db_schema: Optional[str] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        create_schema: bool = True,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize the sync PostgreSQL storage.
        
        Args:
            db_url: PostgreSQL connection URL (e.g., "postgresql://user:pass@host/db").
            db_engine: Pre-configured SQLAlchemy Engine.
            db_schema: PostgreSQL schema to use (default: "public").
            session_table: Name of the session table.
            user_memory_table: Name of the user memory table.
            knowledge_table: Name of the knowledge document registry table.
            create_schema: Whether to create the schema if it doesn't exist.
            id: Unique identifier for this storage instance.
        
        Raises:
            ValueError: If neither db_url nor db_engine is provided.
            ImportError: If sqlalchemy or psycopg2 is not installed.
        """
        if Column is None or Engine is None:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="sqlalchemy psycopg",
                install_command='pip install "upsonic[postgres-storage]"',
                feature_name="PostgreSQL storage provider"
            )
        
        super().__init__(
            session_table=session_table,
            user_memory_table=user_memory_table,
            knowledge_table=knowledge_table,
            id=id,
        )

        # Determine database engine
        _engine: Optional[Engine] = db_engine
        
        if _engine is None:
            if db_url is not None:
                _engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                )
            else:
                raise ValueError(
                    "One of db_url or db_engine must be provided for PostgreSQL storage"
                )

        self.db_engine: Engine = _engine
        self.db_url: Optional[str] = db_url
        self.db_schema: str = db_schema if db_schema is not None else "public"
        self.create_schema_flag: bool = create_schema
        self.metadata: MetaData = MetaData(schema=self.db_schema)
        
        # Aliases for compatibility with generic model methods
        self.engine: Engine = self.db_engine
        self._metadata: MetaData = self.metadata
        self.schema_name: str = self.db_schema

        # Initialize database session factory (scoped_session for thread safety)
        self.Session: scoped_session = scoped_session(
            sessionmaker(
                bind=self.db_engine,
                expire_on_commit=False,
            )
        )

        # Table references (populated lazily)
        self._session_table: Optional[Table] = None
        self._user_memory_table: Optional[Table] = None
        self._cultural_knowledge_table: Optional[Table] = None
        self._knowledge_table: Optional[Table] = None
        self._tables: Dict[str, Table] = {}  # Cache for generic model tables

    def close(self) -> None:
        """Close database connections and dispose of the connection pool."""
        if self.db_engine is not None:
            self.db_engine.dispose()

    # ======================== Table Management ========================

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the PostgreSQL database."""
        with self.Session() as sess:
            return is_table_available(sess, table_name, self.db_schema)

    def _create_all_tables(self) -> None:
        """Create all required tables for this storage."""
        # Create schema first if needed
        if self.create_schema_flag and self.db_schema != "public":
            with self.Session() as sess, sess.begin():
                create_schema(sess, self.db_schema)

        tables_to_create = [
            (self.session_table_name, "sessions"),
            (self.user_memory_table_name, "user_memories"),
            (self.cultural_knowledge_table_name, "cultural_knowledge"),
            (self.knowledge_table_name, "knowledge"),
        ]

        for table_name, table_type in tables_to_create:
            self._get_or_create_table(
                table_name=table_name,
                table_type=table_type,
                create_if_not_found=True,
            )

    def _create_table(self, table_name: str, table_type: str) -> Table:
        """Create a table with the appropriate schema."""
        try:
            table_schema = get_table_schema_definition(table_type)
            columns: List[Column] = []
            indexes: List[str] = []

            for col_name, col_config in table_schema.items():
                column_args = [col_name, col_config["type"]()]
                column_kwargs: Dict[str, Any] = {}

                if col_config.get("primary_key", False):
                    column_kwargs["primary_key"] = True
                if "nullable" in col_config:
                    column_kwargs["nullable"] = col_config["nullable"]
                if col_config.get("index", False):
                    indexes.append(col_name)

                columns.append(Column(*column_args, **column_kwargs))

            # Create table object with schema
            table = Table(
                table_name, 
                self.metadata, 
                *columns, 
                schema=self.db_schema
            )

            # Add indexes
            for idx_col in indexes:
                idx_name = f"idx_{table_name}_{idx_col}"
                table.append_constraint(Index(idx_name, idx_col))

            # Create schema if needed
            if self.create_schema_flag and self.db_schema != "public":
                with self.Session() as sess, sess.begin():
                    create_schema(sess, self.db_schema)

            # Create table in database
            if not self.table_exists(table_name):
                table.create(self.db_engine, checkfirst=True)
                _logger.debug(f"Successfully created table '{self.db_schema}.{table_name}'")
            else:
                _logger.debug(
                    f"Table '{self.db_schema}.{table_name}' already exists, skipping creation"
                )

            # Create indexes
            for idx in table.indexes:
                try:
                    with self.Session() as sess:
                        exists_query = text(
                            "SELECT 1 FROM pg_indexes "
                            "WHERE schemaname = :schema AND indexname = :index_name"
                        )
                        result = sess.execute(
                            exists_query, 
                            {"schema": self.db_schema, "index_name": idx.name}
                        )
                        if result.scalar() is not None:
                            _logger.debug(
                                f"Index {idx.name} already exists in "
                                f"{self.db_schema}.{table_name}, skipping"
                            )
                            continue

                    idx.create(self.db_engine)
                    _logger.debug(
                        f"Created index: {idx.name} for table {self.db_schema}.{table_name}"
                    )
                except Exception as e:
                    _logger.warning(f"Error creating index {idx.name}: {e}")

            return table

        except Exception as e:
            _logger.error(f"Could not create table '{self.db_schema}.{table_name}': {e}")
            raise e

    def _get_or_create_table(
        self,
        table_name: str,
        table_type: str,
        create_if_not_found: bool = False,
    ) -> Table:
        """Get existing table or create it if needed."""
        with self.Session() as sess:
            table_available = is_table_available(sess, table_name, self.db_schema)

        if not table_available and create_if_not_found:
            return self._create_table(table_name, table_type)

        if not table_available:
            raise ValueError(f"Table {self.db_schema}.{table_name} does not exist")

        # Validate that the existing table has the expected schema
        if not is_valid_table(
            db_engine=self.db_engine,
            table_name=table_name,
            table_type=table_type,
            db_schema=self.db_schema,
        ):
            raise ValueError(
                f"Table {self.db_schema}.{table_name} has an invalid schema"
            )

        try:
            table = Table(
                table_name, 
                self.metadata, 
                schema=self.db_schema,
                autoload_with=self.db_engine
            )
            return table

        except Exception as e:
            _logger.error(
                f"Error loading existing table {self.db_schema}.{table_name}: {e}"
            )
            raise e

    def _get_session_table(self, create_if_not_found: bool = False) -> Table:
        """Get the session table, creating if needed."""
        if self._session_table is None:
            self._session_table = self._get_or_create_table(
                table_name=self.session_table_name,
                table_type="sessions",
                create_if_not_found=create_if_not_found,
            )
        return self._session_table

    def _get_user_memory_table(self, create_if_not_found: bool = False) -> Table:
        """Get the user memory table, creating if needed."""
        if self._user_memory_table is None:
            self._user_memory_table = self._get_or_create_table(
                table_name=self.user_memory_table_name,
                table_type="user_memories",
                create_if_not_found=create_if_not_found,
            )
        return self._user_memory_table

    def _get_cultural_knowledge_table(self, create_if_not_found: bool = False) -> Table:
        """Get the cultural knowledge table, creating if needed."""
        if self._cultural_knowledge_table is None:
            self._cultural_knowledge_table = self._get_or_create_table(
                table_name=self.cultural_knowledge_table_name,
                table_type="cultural_knowledge",
                create_if_not_found=create_if_not_found,
            )
        return self._cultural_knowledge_table

    def _get_knowledge_table(self, create_if_not_found: bool = False) -> Table:
        """Get the knowledge document registry table, creating if needed."""
        if self._knowledge_table is None:
            self._knowledge_table = self._get_or_create_table(
                table_name=self.knowledge_table_name,
                table_type="knowledge",
                create_if_not_found=create_if_not_found,
            )
        return self._knowledge_table

    def _sanitize_session_dict(self, session_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize session dictionary for PostgreSQL storage."""
        result = session_dict.copy()
        
        # Sanitize JSON fields - can be dict or list
        json_fields = [
            "session_data", "agent_data", "team_data", 
            "workflow_data", "metadata", "runs", "messages"
        ]
        for field in json_fields:
            if field in result and result[field] is not None:
                if isinstance(result[field], dict):
                    result[field] = sanitize_postgres_dict(result[field])
                elif isinstance(result[field], list):
                    result[field] = sanitize_postgres_list(result[field])
        
        # Sanitize string fields
        string_fields = ["summary"]
        for field in string_fields:
            if field in result and result[field] is not None:
                result[field] = sanitize_postgres_string(result[field])
        
        return result

    # ======================== Session Methods ========================

    def upsert_session(
        self,
        session: "Session",
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Insert or update a session in the database.
        
        Args:
            session: The session to upsert. Must have session_id set.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The upserted session (deserialized or dict based on flag).
        
        Raises:
            ValueError: If session_id is not provided.
        """
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType

        if not hasattr(session, "session_id") or session.session_id is None:
            raise ValueError("Session must have session_id set for upsert")

        try:
            table = self._get_session_table(create_if_not_found=True)

            # Convert session to dict with serialize_flag=True and sanitize
            session_dict = session.to_dict(serialize_flag=True) if hasattr(session, "to_dict") else dict(session)
            session_dict = self._sanitize_session_dict(session_dict)

            current_time = int(time.time())

            if isinstance(session, AgentSession):
                with self.Session() as sess, sess.begin():
                    stmt = postgresql.insert(table).values(
                        session_id=session_dict.get("session_id"),
                        session_type=SessionType.AGENT.value,
                        agent_id=session_dict.get("agent_id"),
                        team_id=session_dict.get("team_id"),
                        workflow_id=session_dict.get("workflow_id"),
                        user_id=session_dict.get("user_id"),
                        session_data=session_dict.get("session_data"),
                        agent_data=session_dict.get("agent_data"),
                        metadata=session_dict.get("metadata"),
                        runs=session_dict.get("runs"),
                        messages=session_dict.get("messages"),
                        summary=session_dict.get("summary"),
                        usage=session_dict.get("usage"),
                        created_at=session_dict.get("created_at") or current_time,
                        updated_at=current_time,
                    )

                    stmt = stmt.on_conflict_do_update(
                        index_elements=["session_id"],
                        set_=dict(
                            agent_id=session_dict.get("agent_id"),
                            team_id=session_dict.get("team_id"),
                            workflow_id=session_dict.get("workflow_id"),
                            user_id=session_dict.get("user_id"),
                            session_data=session_dict.get("session_data"),
                            agent_data=session_dict.get("agent_data"),
                            metadata=session_dict.get("metadata"),
                            runs=session_dict.get("runs"),
                            messages=session_dict.get("messages"),
                            summary=session_dict.get("summary"),
                            usage=session_dict.get("usage"),
                            updated_at=current_time,
                        ),
                    )

                    stmt = stmt.returning(*table.columns)
                    result = sess.execute(stmt)
                    row = result.fetchone()

                    if row is None:
                        return None

                    session_raw = dict(row._mapping)
                    _logger.debug(f"Upserted session: {session_raw.get('session_id')}")

                    if not deserialize:
                        return session_raw

                    return AgentSession.from_dict(session_raw, deserialize_flag=True)
            else:
                # Fallback for other session types (TeamSession, WorkflowSession) - not yet implemented
                session_type_value = "agent"
                if hasattr(session, "session_type"):
                    st = session.session_type
                    if hasattr(st, "value"):
                        session_type_value = st.value
                    elif isinstance(st, str):
                        session_type_value = st

                with self.Session() as sess, sess.begin():
                    stmt = postgresql.insert(table).values(
                        session_id=session_dict.get("session_id"),
                        session_type=session_type_value,
                        agent_id=session_dict.get("agent_id"),
                        team_id=session_dict.get("team_id"),
                        workflow_id=session_dict.get("workflow_id"),
                        user_id=session_dict.get("user_id"),
                        session_data=session_dict.get("session_data"),
                        agent_data=session_dict.get("agent_data"),
                        team_data=session_dict.get("team_data"),
                        workflow_data=session_dict.get("workflow_data"),
                        metadata=session_dict.get("metadata"),
                        runs=session_dict.get("runs"),
                        messages=session_dict.get("messages"),
                        summary=session_dict.get("summary"),
                        created_at=session_dict.get("created_at") or current_time,
                        updated_at=current_time,
                    )

                    stmt = stmt.on_conflict_do_update(
                        index_elements=["session_id"],
                        set_=dict(
                            session_type=session_type_value,
                            agent_id=session_dict.get("agent_id"),
                            team_id=session_dict.get("team_id"),
                            workflow_id=session_dict.get("workflow_id"),
                            user_id=session_dict.get("user_id"),
                            session_data=session_dict.get("session_data"),
                            agent_data=session_dict.get("agent_data"),
                            team_data=session_dict.get("team_data"),
                            workflow_data=session_dict.get("workflow_data"),
                            metadata=session_dict.get("metadata"),
                            runs=session_dict.get("runs"),
                            messages=session_dict.get("messages"),
                            summary=session_dict.get("summary"),
                            updated_at=current_time,
                        ),
                    )

                    stmt = stmt.returning(*table.columns)
                    result = sess.execute(stmt)
                    row = result.fetchone()

                    if row is None:
                        return None

                    session_raw = dict(row._mapping)
                    _logger.debug(f"Upserted session: {session_raw.get('session_id')}")

                    if not deserialize:
                        return session_raw

                    return deserialize_session(session_raw)

        except Exception as e:
            _logger.error(f"Error upserting session: {e}")
            raise e

    def upsert_sessions(
        self,
        sessions: List["Session"],
        deserialize: bool = True,
    ) -> List[Union["Session", Dict[str, Any]]]:
        """
        Bulk insert or update multiple sessions.
        
        Args:
            sessions: List of sessions to upsert. Each must have session_id set.
            deserialize: If True, return deserialized Session objects.
                        If False, return raw dictionaries.
        
        Returns:
            List of upserted sessions.
        
        Raises:
            ValueError: If any session is missing session_id.
        """
        if not sessions:
            return []

        # Validate all sessions have session_id
        for session in sessions:
            if not hasattr(session, "session_id") or session.session_id is None:
                raise ValueError("All sessions must have session_id set for upsert")

        try:
            table = self._get_session_table(create_if_not_found=True)
            results: List[Union["Session", Dict[str, Any]]] = []
            current_time = int(time.time())

            with self.Session() as sess, sess.begin():
                for session in sessions:
                    # Convert session to dict with serialize_flag=True and sanitize
                    session_dict = session.to_dict(serialize_flag=True) if hasattr(session, "to_dict") else dict(session)
                    session_dict = self._sanitize_session_dict(session_dict)
                    
                    session_type_value = "agent"
                    if hasattr(session, "session_type"):
                        st = session.session_type
                        if hasattr(st, "value"):
                            session_type_value = st.value
                        elif isinstance(st, str):
                            session_type_value = st

                    stmt = postgresql.insert(table).values(
                        session_id=session_dict.get("session_id"),
                        session_type=session_type_value,
                        agent_id=session_dict.get("agent_id"),
                        team_id=session_dict.get("team_id"),
                        workflow_id=session_dict.get("workflow_id"),
                        user_id=session_dict.get("user_id"),
                        session_data=session_dict.get("session_data"),
                        agent_data=session_dict.get("agent_data"),
                        team_data=session_dict.get("team_data"),
                        workflow_data=session_dict.get("workflow_data"),
                        metadata=session_dict.get("metadata"),
                        runs=session_dict.get("runs"),
                        messages=session_dict.get("messages"),
                        summary=session_dict.get("summary"),
                        usage=session_dict.get("usage"),
                        created_at=session_dict.get("created_at") or current_time,
                        updated_at=current_time,
                    )

                    stmt = stmt.on_conflict_do_update(
                        index_elements=["session_id"],
                        set_=dict(
                            session_type=session_type_value,
                            agent_id=session_dict.get("agent_id"),
                            team_id=session_dict.get("team_id"),
                            workflow_id=session_dict.get("workflow_id"),
                            user_id=session_dict.get("user_id"),
                            session_data=session_dict.get("session_data"),
                            agent_data=session_dict.get("agent_data"),
                            team_data=session_dict.get("team_data"),
                            workflow_data=session_dict.get("workflow_data"),
                            metadata=session_dict.get("metadata"),
                            runs=session_dict.get("runs"),
                            messages=session_dict.get("messages"),
                            summary=session_dict.get("summary"),
                            usage=session_dict.get("usage"),
                            updated_at=current_time,
                        ),
                    )

                    stmt = stmt.returning(*table.columns)
                    result = sess.execute(stmt)
                    row = result.fetchone()

                    if row is not None:
                        session_raw = dict(row._mapping)
                        if deserialize:
                            results.append(deserialize_session(session_raw))
                        else:
                            results.append(session_raw)

            _logger.debug(f"Upserted {len(results)} sessions")
            return results

        except Exception as e:
            _logger.error(f"Error upserting sessions: {e}")
            raise e

    def get_session(
        self,
        session_id: Optional[str] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Get a session from the database.
        
        Args:
            session_id: ID of the session to retrieve. If None, returns latest session.
            session_type: Type of session (AGENT, TEAM, WORKFLOW).
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The session if found, None otherwise.
        """
        try:
            table = self._get_session_table(create_if_not_found=True)
        except ValueError:
            return None

        try:
            with self.Session() as sess, sess.begin():
                stmt = select(table)

                if session_id is not None:
                    stmt = stmt.where(table.c.session_id == session_id)
                if session_type is not None:
                    stmt = stmt.where(table.c.session_type == session_type.value)
                if user_id is not None:
                    stmt = stmt.where(table.c.user_id == user_id)
                if agent_id is not None:
                    stmt = stmt.where(table.c.agent_id == agent_id)

                # If no session_id, get latest
                if session_id is None:
                    stmt = stmt.order_by(table.c.created_at.desc()).limit(1)

                result = sess.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

                session_raw = dict(row._mapping)

                if not deserialize:
                    return session_raw

                return deserialize_session(session_raw, session_type)

        except Exception as e:
            _logger.error(f"Error getting session: {e}")
            raise e

    def get_usage(
        self,
        session_id: str,
    ) -> Optional[Any]:
        """
        Get the aggregated usage for a specific session.
        
        Args:
            session_id: The ID of the session.
        
        Returns:
            The RunUsage object for the session, or None if not found.
        """
        from upsonic.usage import RunUsage
        from upsonic.utils.timer import Timer
        import json
        
        table = self._get_session_table()
        if table is None:
            return None

        try:
            with self.Session() as sess:
                stmt = select(table.c.usage).where(table.c.session_id == session_id)
                result = sess.execute(stmt).scalar_one_or_none()

                if result is None:
                    return None

                if isinstance(result, dict):
                    usage_data = result
                else:
                    usage_data = json.loads(result) if result else None

                if usage_data:
                    timer_data = usage_data.pop("timer", None)
                    usage = RunUsage(**usage_data)
                    if timer_data:
                        timer = Timer()
                        timer.start_time = float(timer_data["start_time"]) if timer_data.get("start_time") else None
                        timer.end_time = float(timer_data["end_time"]) if timer_data.get("end_time") else None
                        timer.elapsed_time = timer_data.get("elapsed")
                        usage.timer = timer
                    return usage
                return None
        except Exception as e:
            _logger.error(f"Error getting usage: {e}")
            return None

    def get_sessions(
        self,
        session_ids: Optional[List[str]] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["Session"], Tuple[List[Dict[str, Any]], int]]:
        """
        Get multiple sessions from the database.
        
        Args:
            session_ids: List of session IDs to retrieve. If None, returns all sessions.
            session_type: Filter by session type.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            sort_by: Field to sort by.
            sort_order: Sort order ('asc' or 'desc').
            deserialize: If True, return deserialized Session objects.
                        If False, return tuple of (raw dictionaries, total count).
        
        Returns:
            List of sessions or tuple of (list of dicts, total count).
        """
        try:
            table = self._get_session_table(create_if_not_found=True)
        except ValueError:
            return [] if deserialize else ([], 0)

        try:
            with self.Session() as sess, sess.begin():
                stmt = select(table)

                # Apply filters
                if session_ids is not None and len(session_ids) > 0:
                    stmt = stmt.where(table.c.session_id.in_(session_ids))
                if session_type is not None:
                    stmt = stmt.where(table.c.session_type == session_type.value)
                if user_id is not None:
                    stmt = stmt.where(table.c.user_id == user_id)
                if agent_id is not None:
                    stmt = stmt.where(table.c.agent_id == agent_id)

                # Get total count
                count_stmt = select(func.count()).select_from(stmt.alias())
                count_result = sess.execute(count_stmt)
                total_count = count_result.scalar() or 0

                # Apply sorting
                stmt = apply_sorting(stmt, table, sort_by, sort_order)

                # Apply pagination
                if limit is not None:
                    stmt = stmt.limit(limit)
                    if offset is not None:
                        stmt = stmt.offset(offset)

                result = sess.execute(stmt)
                rows = result.fetchall()

                if not rows:
                    return [] if deserialize else ([], 0)

                sessions_raw = [dict(row._mapping) for row in rows]

                if not deserialize:
                    return sessions_raw, total_count

                return [
                    deserialize_session(s, session_type)
                    for s in sessions_raw
                ]

        except Exception as e:
            _logger.error(f"Error getting sessions: {e}")
            raise e

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database.
        
        Args:
            session_id: ID of the session to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If session_id is not provided.
        """
        if not session_id:
            raise ValueError("session_id is required for delete")

        try:
            table = self._get_session_table(create_if_not_found=False)
        except ValueError:
            return False

        try:
            with self.Session() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.session_id == session_id)
                result = sess.execute(delete_stmt)

                deleted = result.rowcount > 0
                if deleted:
                    _logger.debug(f"Deleted session: {session_id}")
                else:
                    _logger.debug(f"No session found to delete: {session_id}")
                return deleted

        except Exception as e:
            _logger.error(f"Error deleting session: {e}")
            raise e

    def delete_sessions(self, session_ids: List[str]) -> int:
        """
        Delete multiple sessions from the database.
        
        Args:
            session_ids: List of session IDs to delete.
        
        Returns:
            Number of sessions deleted.
        
        Raises:
            ValueError: If session_ids is empty.
        """
        if not session_ids:
            raise ValueError("session_ids is required and cannot be empty")

        try:
            table = self._get_session_table(create_if_not_found=False)
        except ValueError:
            return 0

        try:
            with self.Session() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.session_id.in_(session_ids))
                result = sess.execute(delete_stmt)

                deleted_count = result.rowcount
                _logger.debug(f"Deleted {deleted_count} sessions")
                return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting sessions: {e}")
            raise e

    # ======================== User Memory Methods ========================

    def upsert_user_memory(
        self,
        user_memory: UserMemory,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Insert or update user memory in the database.
        
        Args:
            user_memory: The UserMemory instance to store. Must have user_id set.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The upserted user memory record.
        
        Raises:
            ValueError: If user_id is not provided in user_memory.
        """
        if not user_memory.user_id:
            raise ValueError("user_memory must have user_id set for upsert")

        try:
            table = self._get_user_memory_table(create_if_not_found=True)
            current_time = int(time.time())
            
            # Sanitize the user memory
            sanitized_memory = sanitize_postgres_dict(user_memory.user_memory)

            with self.Session() as sess, sess.begin():
                stmt = postgresql.insert(table).values(
                    user_id=user_memory.user_id,
                    user_memory=sanitized_memory,
                    agent_id=user_memory.agent_id,
                    team_id=user_memory.team_id,
                    created_at=user_memory.created_at or current_time,
                    updated_at=current_time,
                )

                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id"],
                    set_=dict(
                        user_memory=sanitized_memory,
                        agent_id=user_memory.agent_id,
                        team_id=user_memory.team_id,
                        updated_at=current_time,
                    ),
                )

                stmt = stmt.returning(*table.columns)
                result = sess.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

                _logger.debug(f"Upserted user memory for user: {user_memory.user_id}")
                memory_dict = dict(row._mapping)
                if deserialize:
                    return UserMemory.from_dict(memory_dict)
                return memory_dict

        except Exception as e:
            _logger.error(f"Error upserting user memory: {e}")
            raise e

    def upsert_user_memories(
        self,
        user_memories: List[UserMemory],
        deserialize: bool = True,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        """
        Bulk insert or update multiple user memories.
        
        Args:
            user_memories: List of UserMemory instances. Each must have user_id set.
            deserialize: If True, return UserMemory objects. If False, return raw dicts.
        
        Returns:
            List of upserted user memory records.
        
        Raises:
            ValueError: If any record is missing user_id.
        """
        if not user_memories:
            return []

        # Validate all have user_id
        for memory in user_memories:
            if not memory.user_id:
                raise ValueError("All user memories must have user_id set")

        try:
            table = self._get_user_memory_table(create_if_not_found=True)
            results: List[Union[UserMemory, Dict[str, Any]]] = []
            current_time = int(time.time())

            with self.Session() as sess, sess.begin():
                for memory in user_memories:
                    sanitized_memory = sanitize_postgres_dict(memory.user_memory)
                    
                    stmt = postgresql.insert(table).values(
                        user_id=memory.user_id,
                        user_memory=sanitized_memory,
                        agent_id=memory.agent_id,
                        team_id=memory.team_id,
                        created_at=memory.created_at or current_time,
                        updated_at=current_time,
                    )

                    stmt = stmt.on_conflict_do_update(
                        index_elements=["user_id"],
                        set_=dict(
                            user_memory=sanitized_memory,
                            agent_id=memory.agent_id,
                            team_id=memory.team_id,
                            updated_at=current_time,
                        ),
                    )

                    stmt = stmt.returning(*table.columns)
                    result = sess.execute(stmt)
                    row = result.fetchone()

                    if row is not None:
                        memory_dict = dict(row._mapping)
                        if deserialize:
                            results.append(UserMemory.from_dict(memory_dict))
                        else:
                            results.append(memory_dict)

            _logger.debug(f"Upserted {len(results)} user memories")
            return results

        except Exception as e:
            _logger.error(f"Error upserting user memories: {e}")
            raise e

    def get_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Get user memory from the database.
        
        Args:
            user_id: User ID to retrieve. If None, returns latest memory.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return UserMemory object.
                        If False, return raw dictionary.
        
        Returns:
            UserMemory object or dict if found, None otherwise.
        """
        try:
            table = self._get_user_memory_table(create_if_not_found=True)
        except ValueError:
            return None

        try:
            with self.Session() as sess, sess.begin():
                stmt = select(table)

                if user_id is not None:
                    stmt = stmt.where(table.c.user_id == user_id)
                if agent_id is not None:
                    stmt = stmt.where(table.c.agent_id == agent_id)
                if team_id is not None:
                    stmt = stmt.where(table.c.team_id == team_id)

                # If no user_id, get latest
                if user_id is None:
                    stmt = stmt.order_by(table.c.updated_at.desc()).limit(1)

                result = sess.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

                memory_raw = dict(row._mapping)
                
                if not deserialize:
                    return memory_raw
                
                return UserMemory.from_dict(memory_raw)

        except Exception as e:
            _logger.error(f"Error getting user memory: {e}")
            raise e

    def get_user_memories(
        self,
        user_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        deserialize: bool = True,
    ) -> Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
        """
        Get multiple user memories from the database.
        
        Args:
            user_ids: List of user IDs to retrieve. If None, returns all.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            deserialize: If True, return list of UserMemory objects.
                        If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of UserMemory objects or tuple of (list of dicts, total count).
        """
        try:
            table = self._get_user_memory_table(create_if_not_found=False)
        except ValueError:
            return [] if deserialize else ([], 0)

        try:
            with self.Session() as sess, sess.begin():
                stmt = select(table)

                if user_ids is not None and len(user_ids) > 0:
                    stmt = stmt.where(table.c.user_id.in_(user_ids))
                if agent_id is not None:
                    stmt = stmt.where(table.c.agent_id == agent_id)
                if team_id is not None:
                    stmt = stmt.where(table.c.team_id == team_id)

                # Get total count
                count_stmt = select(func.count()).select_from(stmt.alias())
                count_result = sess.execute(count_stmt)
                total_count = count_result.scalar() or 0

                # Apply sorting (by updated_at desc)
                stmt = stmt.order_by(table.c.updated_at.desc())

                # Apply pagination
                if limit is not None:
                    stmt = stmt.limit(limit)
                    if offset is not None:
                        stmt = stmt.offset(offset)

                result = sess.execute(stmt)
                rows = result.fetchall()

                if not rows:
                    return [] if deserialize else ([], 0)

                memories_raw = [dict(row._mapping) for row in rows]

                if not deserialize:
                    return memories_raw, total_count

                return [UserMemory.from_dict(memory) for memory in memories_raw]

        except Exception as e:
            _logger.error(f"Error getting user memories: {e}")
            raise e

    def delete_user_memory(self, user_id: str) -> bool:
        """
        Delete user memory from the database.
        
        Args:
            user_id: User ID of the memory to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If user_id is not provided.
        """
        if not user_id:
            raise ValueError("user_id is required for delete")

        try:
            table = self._get_user_memory_table(create_if_not_found=False)
        except ValueError:
            return False

        try:
            with self.Session() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.user_id == user_id)
                result = sess.execute(delete_stmt)

                deleted = result.rowcount > 0
                if deleted:
                    _logger.debug(f"Deleted user memory: {user_id}")
                else:
                    _logger.debug(f"No user memory found to delete: {user_id}")
                return deleted

        except Exception as e:
            _logger.error(f"Error deleting user memory: {e}")
            raise e

    def delete_user_memories(self, user_ids: List[str]) -> int:
        """
        Delete multiple user memories from the database.
        
        Args:
            user_ids: List of user IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            ValueError: If user_ids is empty.
        """
        if not user_ids:
            raise ValueError("user_ids is required and cannot be empty")

        try:
            table = self._get_user_memory_table(create_if_not_found=False)
        except ValueError:
            return 0

        try:
            with self.Session() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.user_id.in_(user_ids))
                result = sess.execute(delete_stmt)

                deleted_count = result.rowcount
                _logger.debug(f"Deleted {deleted_count} user memories")
                return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting user memories: {e}")
            raise e

    # ======================== Utility Methods ========================

    def clear_all(self) -> None:
        """
        Clear all data from all tables.
        
        This removes all sessions and user memories from the storage.
        Use with caution.
        """
        try:
            # Clear sessions
            try:
                table = self._get_session_table(create_if_not_found=False)
                with self.Session() as sess, sess.begin():
                    sess.execute(table.delete())
                _logger.debug("Cleared all sessions")
            except ValueError:
                pass

            # Clear user memories
            try:
                table = self._get_user_memory_table(create_if_not_found=False)
                with self.Session() as sess, sess.begin():
                    sess.execute(table.delete())
                _logger.debug("Cleared all user memories")
            except ValueError:
                pass

            # Clear knowledge
            try:
                table = self._get_knowledge_table(create_if_not_found=False)
                with self.Session() as sess, sess.begin():
                    sess.execute(table.delete())
                _logger.debug("Cleared all knowledge entries")
            except ValueError:
                pass

            _logger.info("Cleared all data from PostgreSQL storage")

        except Exception as e:
            _logger.error(f"Error clearing all data: {e}")
            raise e

    # ======================== Generic Model Methods ========================

    def _get_generic_model_table(self, collection: str, create_if_not_found: bool = True) -> Table:
        """Get or create a generic model table for a collection."""
        table_name = f"upsonic_models_{collection}"
        
        if table_name in self._tables:
            return self._tables[table_name]
        
        # Check if table exists
        with self.Session() as sess:
            table_exists = is_table_available(sess, table_name, self.schema_name)
        if table_exists:
            # Reflect existing table
            self._metadata.reflect(bind=self.engine, schema=self.schema_name, only=[table_name])
            full_name = f"{self.schema_name}.{table_name}" if self.schema_name else table_name
            if full_name in self._metadata.tables:
                self._tables[table_name] = self._metadata.tables[full_name]
                return self._tables[table_name]
        
        if not create_if_not_found:
            raise ValueError(f"Table {table_name} does not exist")
        
        # Create new table with key and data columns
        from sqlalchemy import String, BigInteger
        from sqlalchemy.dialects.postgresql import JSONB
        
        table = Table(
            table_name,
            self._metadata,
            Column("key", String, primary_key=True, nullable=False),
            Column("collection", String, nullable=False, index=True),
            Column("model_data", JSONB, nullable=False),
            Column("created_at", BigInteger, nullable=False, index=True),
            Column("updated_at", BigInteger, nullable=True, index=True),
            schema=self.schema_name,
            extend_existing=True,
        )
        
        table.create(self.engine, checkfirst=True)
        self._tables[table_name] = table
        _logger.debug(f"Created generic model table: {table_name}")
        return table

    def upsert_model(
        self,
        key: str,
        model: Any,
        collection: str = "generic_models",
    ) -> None:
        """Insert or update a generic Pydantic model in storage."""
        try:
            table = self._get_generic_model_table(collection)
            current_time = int(time.time())
            
            # Serialize model
            if hasattr(model, 'model_dump'):
                model_data = model.model_dump(mode='json')
            elif hasattr(model, 'dict'):
                model_data = model.dict()
            else:
                model_data = dict(model)
            
            model_data = sanitize_postgres_dict(model_data)
            
            record = {
                "key": key,
                "collection": collection,
                "model_data": model_data,
                "updated_at": current_time,
            }
            
            with self.Session() as sess, sess.begin():
                # Check if exists
                stmt = select(table).where(table.c.key == key)
                existing = sess.execute(stmt).fetchone()
                
                if existing:
                    # Update
                    update_stmt = table.update().where(table.c.key == key).values(**record)
                    sess.execute(update_stmt)
                else:
                    # Insert
                    record["created_at"] = current_time
                    sess.execute(table.insert().values(**record))
            
            _logger.debug(f"Upserted model with key '{key}' in collection '{collection}'")
            
        except Exception as e:
            _logger.error(f"Error upserting model: {e}")
            raise

    def get_model(
        self,
        key: str,
        model_type: Any,
        collection: str = "generic_models",
    ) -> Optional[Any]:
        """Retrieve a generic Pydantic model from storage."""
        try:
            table = self._get_generic_model_table(collection, create_if_not_found=False)
            
            with self.Session() as sess:
                stmt = select(table).where(table.c.key == key)
                result = sess.execute(stmt).fetchone()
                
                if result is None:
                    return None
                
                model_data = result.model_data
                return model_type(**model_data)
                
        except ValueError:
            return None
        except Exception as e:
            _logger.error(f"Error getting model: {e}")
            return None

    def delete_model(
        self,
        key: str,
        collection: str = "generic_models",
    ) -> bool:
        """Delete a generic model from storage."""
        try:
            table = self._get_generic_model_table(collection, create_if_not_found=False)
            
            with self.Session() as sess, sess.begin():
                stmt = table.delete().where(table.c.key == key)
                result = sess.execute(stmt)
                
                if result.rowcount > 0:
                    _logger.debug(f"Deleted model with key '{key}' from collection '{collection}'")
                    return True
                return False
                
        except ValueError:
            return False
        except Exception as e:
            _logger.error(f"Error deleting model: {e}")
            return False

    def list_models(
        self,
        model_type: Any,
        collection: str = "generic_models",
    ) -> List[Any]:
        """List all models of a type in a collection."""
        try:
            table = self._get_generic_model_table(collection, create_if_not_found=False)
            
            with self.Session() as sess:
                stmt = select(table).where(table.c.collection == collection)
                results = sess.execute(stmt).fetchall()
                
                models = []
                for row in results:
                    try:
                        models.append(model_type(**row.model_data))
                    except Exception as e:
                        _logger.warning(f"Failed to deserialize model: {e}")
                
                return models
                
        except ValueError:
            return []
        except Exception as e:
            _logger.error(f"Error listing models: {e}")
            return []

    # ======================== Cultural Knowledge Methods ========================

    def delete_cultural_knowledge(self, id: str) -> None:
        """Delete cultural knowledge from the database.
        
        Args:
            id: The ID of the cultural knowledge to delete.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        try:
            table = self._get_cultural_knowledge_table(create_if_not_found=False)

            with self.Session() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.id == id)
                result = sess.execute(delete_stmt)

                success = result.rowcount > 0
                if success:
                    _logger.debug(f"Successfully deleted cultural knowledge id: {id}")
                else:
                    _logger.debug(f"No cultural knowledge found with id: {id}")

        except Exception as e:
            _logger.error(f"Error deleting cultural knowledge: {e}")
            raise e

    def get_cultural_knowledge(
        self,
        id: str,
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Get cultural knowledge from the database.
        
        Args:
            id: The ID of the cultural knowledge to get.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            CulturalKnowledge object or dict if found, None otherwise.
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            table = self._get_cultural_knowledge_table(create_if_not_found=False)

            with self.Session() as sess, sess.begin():
                stmt = select(table).where(table.c.id == id)
                result = sess.execute(stmt).fetchone()
                if result is None:
                    return None

                db_row = dict(result._mapping)
                if not deserialize:
                    return db_row
                return CulturalKnowledge.from_dict(db_row)

        except Exception as e:
            _logger.error(f"Exception reading from cultural knowledge table: {e}")
            raise e

    def get_all_cultural_knowledge(
        self,
        name: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["CulturalKnowledge"], Tuple[List[Dict[str, Any]], int]]:
        """Get all cultural knowledge entries from the database.
        
        Args:
            name: Filter by name.
            limit: Maximum number of records to return.
            page: Page number (1-indexed).
            sort_by: Column to sort by.
            sort_order: Sort order ('asc' or 'desc').
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return list of CulturalKnowledge objects.
                        If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of CulturalKnowledge objects or tuple of (list of dicts, total count).
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            table = self._get_cultural_knowledge_table(create_if_not_found=False)

            with self.Session() as sess, sess.begin():
                stmt = select(table)

                # Filtering
                if name is not None:
                    stmt = stmt.where(table.c.name == name)
                if agent_id is not None:
                    stmt = stmt.where(table.c.agent_id == agent_id)
                if team_id is not None:
                    stmt = stmt.where(table.c.team_id == team_id)

                # Get total count after applying filtering
                count_stmt = select(func.count()).select_from(stmt.alias())
                count_result = sess.execute(count_stmt)
                total_count = count_result.scalar() or 0

                # Sorting
                stmt = apply_sorting(stmt, table, sort_by, sort_order)
                
                # Paginating
                if limit is not None:
                    stmt = stmt.limit(limit)
                    if page is not None:
                        stmt = stmt.offset((page - 1) * limit)

                result = sess.execute(stmt).fetchall()
                if not result:
                    return [] if deserialize else ([], 0)

                db_rows = [dict(record._mapping) for record in result]
                if not deserialize:
                    return db_rows, total_count
                return [CulturalKnowledge.from_dict(row) for row in db_rows]

        except Exception as e:
            _logger.error(f"Error reading from cultural knowledge table: {e}")
            raise e

    def upsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Upsert cultural knowledge into the database.
        
        Args:
            cultural_knowledge: The CulturalKnowledge instance to upsert.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            The upserted CulturalKnowledge object or dict, or None if error occurs.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        import uuid
        
        try:
            table = self._get_cultural_knowledge_table(create_if_not_found=True)

            if cultural_knowledge.id is None:
                cultural_knowledge.id = str(uuid.uuid4())

            # Use to_dict for serialization
            data = cultural_knowledge.to_dict()
            
            # Convert RFC3339 timestamps to epoch seconds for storage
            if "created_at" in data and data["created_at"] is not None:
                if isinstance(data["created_at"], str):
                    from upsonic.utils.dttm import to_epoch_s
                    data["created_at"] = to_epoch_s(data["created_at"])
            if "updated_at" in data and data["updated_at"] is not None:
                if isinstance(data["updated_at"], str):
                    from upsonic.utils.dttm import to_epoch_s
                    data["updated_at"] = to_epoch_s(data["updated_at"])
            else:
                # Set updated_at to current time if not set
                data["updated_at"] = int(time.time())

            # Sanitize string fields for PostgreSQL (remove null bytes)
            if "name" in data:
                data["name"] = sanitize_postgres_string(data.get("name"))
            if "summary" in data:
                data["summary"] = sanitize_postgres_string(data.get("summary"))
            if "content" in data:
                data["content"] = sanitize_postgres_string(data.get("content"))
            if "input" in data:
                data["input"] = sanitize_postgres_string(data.get("input"))
            if "metadata" in data and data["metadata"] is not None:
                data["metadata"] = sanitize_postgres_dict(data["metadata"])
            if "notes" in data and data["notes"] is not None:
                data["notes"] = sanitize_postgres_list(data["notes"])
            if "categories" in data and data["categories"] is not None:
                data["categories"] = sanitize_postgres_list(data["categories"])

            with self.Session() as sess, sess.begin():
                stmt = postgresql.insert(table).values(**data)
                stmt = stmt.on_conflict_do_update(  # type: ignore
                    index_elements=["id"],
                    set_={k: v for k, v in data.items() if k != "id"},
                ).returning(table)

                result = sess.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

            db_row: Dict[str, Any] = dict(row._mapping)
            if not deserialize:
                return db_row
            return CulturalKnowledge.from_dict(db_row)

        except Exception as e:
            _logger.error(f"Error upserting cultural knowledge: {e}")
            raise e

    # ======================== Knowledge Content Methods ========================

    def upsert_knowledge_content(
        self,
        knowledge_row: "KnowledgeRow",
    ) -> Optional["KnowledgeRow"]:
        """Insert or update a knowledge document registry entry."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            table = self._get_knowledge_table(create_if_not_found=True)
            current_time = int(time.time())

            data = knowledge_row.to_dict()
            data.setdefault("created_at", current_time)
            data["updated_at"] = current_time

            if "name" in data:
                data["name"] = sanitize_postgres_string(data.get("name"))
            if "description" in data:
                data["description"] = sanitize_postgres_string(data.get("description"))
            if "source" in data:
                data["source"] = sanitize_postgres_string(data.get("source"))
            if "status_message" in data:
                data["status_message"] = sanitize_postgres_string(data.get("status_message"))
            if "metadata" in data and data["metadata"] is not None:
                data["metadata"] = sanitize_postgres_dict(data["metadata"])

            with self.Session() as sess, sess.begin():
                update_fields = {k: v for k, v in data.items() if k != "id"}
                stmt = (
                    postgresql.insert(table)
                    .values(**data)
                    .on_conflict_do_update(
                        index_elements=["id"],
                        set_=update_fields,
                    )
                    .returning(table)
                )
                result = sess.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

            return KnowledgeRow.from_dict(dict(row._mapping))

        except Exception as e:
            _logger.error(f"Error upserting knowledge content: {e}")
            raise e

    def get_knowledge_content(
        self,
        id: str,
    ) -> Optional["KnowledgeRow"]:
        """Get a knowledge document registry entry by ID."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            table = self._get_knowledge_table(create_if_not_found=False)
        except ValueError:
            return None

        try:
            with self.Session() as sess, sess.begin():
                stmt = select(table).where(table.c.id == id)
                result = sess.execute(stmt).fetchone()
                if result is None:
                    return None
                return KnowledgeRow.from_dict(dict(result._mapping))

        except Exception as e:
            _logger.error(f"Error getting knowledge content: {e}")
            raise e

    def get_knowledge_contents(
        self,
        knowledge_base_id: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> Tuple[List["KnowledgeRow"], int]:
        """Get knowledge document registry entries with filtering and pagination."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            table = self._get_knowledge_table(create_if_not_found=False)
        except ValueError:
            return [], 0

        try:
            with self.Session() as sess, sess.begin():
                stmt = select(table)

                if knowledge_base_id is not None:
                    stmt = stmt.where(table.c.knowledge_base_id == knowledge_base_id)

                count_stmt = select(func.count()).select_from(stmt.alias())
                total_count: int = sess.execute(count_stmt).scalar() or 0

                stmt = apply_sorting(stmt, table, sort_by, sort_order)

                if limit is not None:
                    stmt = stmt.limit(limit)
                    if page is not None:
                        stmt = stmt.offset((page - 1) * limit)

                result = sess.execute(stmt).fetchall()
                if not result:
                    return [], 0

                rows = [KnowledgeRow.from_dict(dict(record._mapping)) for record in result]
                return rows, total_count

        except Exception as e:
            _logger.error(f"Error getting knowledge contents: {e}")
            raise e

    def delete_knowledge_content(self, id: str) -> bool:
        """Delete a knowledge document registry entry by ID."""
        try:
            table = self._get_knowledge_table(create_if_not_found=False)
        except ValueError:
            return False

        try:
            with self.Session() as sess, sess.begin():
                stmt = table.delete().where(table.c.id == id)
                result = sess.execute(stmt)

                deleted: bool = result.rowcount > 0
                if deleted:
                    _logger.debug(f"Deleted knowledge content: {id}")
                return deleted

        except Exception as e:
            _logger.error(f"Error deleting knowledge content: {e}")
            raise e

    def delete_knowledge_contents(self, ids: List[str]) -> int:
        """Delete multiple knowledge document registry entries."""
        if not ids:
            return 0

        try:
            table = self._get_knowledge_table(create_if_not_found=False)
        except ValueError:
            return 0

        try:
            with self.Session() as sess, sess.begin():
                stmt = table.delete().where(table.c.id.in_(ids))
                result = sess.execute(stmt)

                deleted_count: int = result.rowcount
                _logger.debug(f"Deleted {deleted_count} knowledge entries")
                return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting knowledge contents: {e}")
            raise e
