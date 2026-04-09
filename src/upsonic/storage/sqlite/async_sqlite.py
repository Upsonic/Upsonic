"""Async SQLite storage implementation for Upsonic agent framework."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine
    from upsonic.session.base import SessionType, Session
    from upsonic.storage.schemas import KnowledgeRow

try:
    from sqlalchemy import Column, MetaData, Table, func, select, text
    from sqlalchemy.dialects import sqlite
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
    from sqlalchemy.schema import Index
except ImportError:
    Column = None  # type: ignore
    MetaData = None  # type: ignore
    Table = None  # type: ignore
    func = None  # type: ignore
    select = None  # type: ignore
    text = None  # type: ignore
    sqlite = None  # type: ignore
    AsyncEngine = None  # type: ignore
    async_sessionmaker = None  # type: ignore
    create_async_engine = None  # type: ignore
    Index = None  # type: ignore

from upsonic.storage.base import AsyncStorage
from upsonic.storage.schemas import UserMemory
from upsonic.culture.cultural_knowledge import CulturalKnowledge
from upsonic.storage.sqlite.schemas import get_table_schema_definition
from upsonic.storage.sqlite.utils import (
    is_table_available_async,
    apply_sorting
)
from upsonic.utils.logging_config import get_logger
from upsonic.storage.utils import serialize_session_json_fields, deserialize_session_json_fields, deserialize_session

_logger = get_logger("upsonic.storage.sqlite")





class AsyncSqliteStorage(AsyncStorage):
    """Async SQLite storage implementation for Upsonic agent framework.
    
    This storage backend uses SQLAlchemy with aiosqlite for async SQLite operations.
    It provides persistence for agent sessions and user memory data.
    
    The following order is used to determine the database connection:
        1. Use the db_engine if provided
        2. Use the db_url if provided
        3. Use the db_file if provided
        4. Create a new database in the current directory (./upsonic.db)
    
    Example:
        ```python
        storage = AsyncSqliteStorage(db_file="./agent_data.db")
        await storage._create_all_tables()
        
        # Upsert a session
        session = AgentSession(session_id="abc123", ...)
        result = await storage.aupsert_session(session)
        
        # Get a session
        session = await storage.aget_session(session_id="abc123")
        ```
    """

    def __init__(
        self,
        db_file: Optional[str] = None,
        db_engine: Optional["AsyncEngine"] = None,
        db_url: Optional[str] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize the async SQLite storage.
        
        Args:
            db_file: Path to the SQLite database file.
            db_engine: Pre-configured SQLAlchemy AsyncEngine.
            db_url: SQLAlchemy database URL (e.g., "sqlite+aiosqlite:///./data.db").
            session_table: Name of the session table.
            user_memory_table: Name of the user memory table.
            knowledge_table: Name of the knowledge document registry table.
            id: Unique identifier for this storage instance.
        
        Raises:
            ImportError: If sqlalchemy or aiosqlite is not installed.
        """
        if Column is None or AsyncEngine is None:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="sqlalchemy aiosqlite",
                install_command='pip install "upsonic[sqlite-storage]"',
                feature_name="SQLite async storage provider"
            )
        
        super().__init__(
            session_table=session_table,
            user_memory_table=user_memory_table,
            knowledge_table=knowledge_table,
            id=id,
        )

        # Determine database engine
        _engine: Optional[AsyncEngine] = db_engine
        
        if _engine is None:
            if db_url is not None:
                _engine = create_async_engine(db_url, echo=False)
            elif db_file is not None:
                db_path = Path(db_file).resolve()
                db_path.parent.mkdir(parents=True, exist_ok=True)
                db_file = str(db_path)
                _engine = create_async_engine(
                    f"sqlite+aiosqlite:///{db_path}",
                    echo=False,
                )
            else:
                # Default: create db in current directory
                default_db_path = Path("./upsonic.db").resolve()
                _engine = create_async_engine(
                    f"sqlite+aiosqlite:///{default_db_path}",
                    echo=False,
                )
                db_file = str(default_db_path)
                _logger.debug(f"Created SQLite database: {default_db_path}")

        self.db_engine: AsyncEngine = _engine
        self.db_url: Optional[str] = db_url
        self.db_file: Optional[str] = db_file
        self.metadata: MetaData = MetaData()
        
        # Aliases for compatibility with generic model methods
        self.engine: AsyncEngine = self.db_engine
        self._metadata: MetaData = self.metadata

        # Initialize database session factory
        self.async_session_factory = async_sessionmaker(
            bind=self.db_engine,
            expire_on_commit=False,
        )

        # Table references (populated lazily)
        self._session_table: Optional[Table] = None
        self._user_memory_table: Optional[Table] = None
        self._cultural_knowledge_table: Optional[Table] = None
        self._knowledge_table: Optional[Table] = None
        self._tables: Dict[str, Table] = {}  # Cache for generic model tables

    async def close(self) -> None:
        """Close database connections and dispose of the connection pool."""
        if self.db_engine is not None:
            await self.db_engine.dispose()

    # ======================== Table Management ========================

    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the SQLite database."""
        async with self.async_session_factory() as sess:
            return await is_table_available_async(sess, table_name)

    async def _create_all_tables(self) -> None:
        """Create all required tables for this storage."""
        tables_to_create = [
            (self.session_table_name, "sessions"),
            (self.user_memory_table_name, "user_memories"),
            (self.cultural_knowledge_table_name, "cultural_knowledge"),
            (self.knowledge_table_name, "knowledge"),
        ]

        for table_name, table_type in tables_to_create:
            await self._get_or_create_table(
                table_name=table_name,
                table_type=table_type,
                create_if_not_found=True,
            )

    async def _create_table(self, table_name: str, table_type: str) -> Table:
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

            # Create table object
            table = Table(table_name, self.metadata, *columns)

            # Add indexes
            for idx_col in indexes:
                idx_name = f"idx_{table_name}_{idx_col}"
                table.append_constraint(Index(idx_name, idx_col))

            # Create table in database
            if not await self.table_exists(table_name):
                async with self.db_engine.begin() as conn:
                    await conn.run_sync(table.create, checkfirst=True)
                _logger.debug(f"Successfully created table '{table_name}'")
            else:
                _logger.debug(f"Table {table_name} already exists, skipping creation")

            # Create indexes
            for idx in table.indexes:
                try:
                    async with self.async_session_factory() as sess:
                        exists_query = text(
                            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=:index_name"
                        )
                        result = await sess.execute(exists_query, {"index_name": idx.name})
                        if result.scalar() is not None:
                            _logger.debug(f"Index {idx.name} already exists, skipping")
                            continue

                    async with self.db_engine.begin() as conn:
                        await conn.run_sync(idx.create)
                    _logger.debug(f"Created index: {idx.name} for table {table_name}")
                except Exception as e:
                    _logger.warning(f"Error creating index {idx.name}: {e}")

            return table

        except Exception as e:
            _logger.error(f"Could not create table '{table_name}': {e}")
            raise e

    async def _get_or_create_table(
        self,
        table_name: str,
        table_type: str,
        create_if_not_found: bool = False,
    ) -> Table:
        """Get existing table or create it if needed."""
        async with self.async_session_factory() as sess:
            table_available = await is_table_available_async(sess, table_name)

        if not table_available and create_if_not_found:
            return await self._create_table(table_name, table_type)

        if not table_available:
            raise ValueError(f"Table {table_name} does not exist")

        try:
            async with self.db_engine.connect() as conn:

                def load_table(connection: Any) -> Table:
                    return Table(table_name, self.metadata, autoload_with=connection)

                table = await conn.run_sync(load_table)
                return table

        except Exception as e:
            _logger.error(f"Error loading existing table {table_name}: {e}")
            raise e

    async def _get_session_table(self, create_if_not_found: bool = False) -> Table:
        """Get the session table, creating if needed."""
        if self._session_table is None:
            self._session_table = await self._get_or_create_table(
                table_name=self.session_table_name,
                table_type="sessions",
                create_if_not_found=create_if_not_found,
            )
        return self._session_table

    async def _get_user_memory_table(self, create_if_not_found: bool = False) -> Table:
        """Get the user memory table, creating if needed."""
        if self._user_memory_table is None:
            self._user_memory_table = await self._get_or_create_table(
                table_name=self.user_memory_table_name,
                table_type="user_memories",
                create_if_not_found=create_if_not_found,
            )
        return self._user_memory_table


    async def aupsert_session(
        self,
        session: Session,
        deserialize: bool = True,
    ) -> Optional[Union[Session, Dict[str, Any]]]:
        """Insert or update a session in the database."""
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType

        if not hasattr(session, "session_id") or session.session_id is None:
            raise ValueError("Session must have session_id set for upsert")

        try:
            table = await self._get_session_table(create_if_not_found=True)

            # Convert session to dict with serialize_flag=True for proper serialization
            session_dict = session.to_dict(serialize_flag=True) if hasattr(session, "to_dict") else dict(session)
            
            # Serialize JSON fields for database storage
            serialized_session = serialize_session_json_fields(session_dict)
            
            current_time = int(time.time())

            if isinstance(session, AgentSession):
                async with self.async_session_factory() as sess, sess.begin():
                    stmt = sqlite.insert(table).values(
                        session_id=serialized_session.get("session_id"),
                        session_type=SessionType.AGENT.value,
                        agent_id=serialized_session.get("agent_id"),
                        team_id=serialized_session.get("team_id"),
                        workflow_id=serialized_session.get("workflow_id"),
                        user_id=serialized_session.get("user_id"),
                        session_data=serialized_session.get("session_data"),
                        agent_data=serialized_session.get("agent_data"),
                        metadata=serialized_session.get("metadata"),
                        runs=serialized_session.get("runs"),
                        messages=serialized_session.get("messages"),
                        usage=serialized_session.get("usage"),
                        summary=serialized_session.get("summary"),
                        created_at=serialized_session.get("created_at") or current_time,
                        updated_at=current_time,
                    )

                    stmt = stmt.on_conflict_do_update(
                        index_elements=["session_id"],
                        set_=dict(
                            agent_id=serialized_session.get("agent_id"),
                            team_id=serialized_session.get("team_id"),
                            workflow_id=serialized_session.get("workflow_id"),
                            user_id=serialized_session.get("user_id"),
                            session_data=serialized_session.get("session_data"),
                            agent_data=serialized_session.get("agent_data"),
                            metadata=serialized_session.get("metadata"),
                            runs=serialized_session.get("runs"),
                            messages=serialized_session.get("messages"),
                            usage=serialized_session.get("usage"),
                            summary=serialized_session.get("summary"),
                            updated_at=current_time,
                        ),
                    )

                    stmt = stmt.returning(*table.columns)
                    result = await sess.execute(stmt)
                    row = result.fetchone()

                    if row is None:
                        return None

                    session_raw = deserialize_session_json_fields(dict(row._mapping))

                    if not deserialize:
                        return session_raw

                    return AgentSession.from_dict(session_raw, deserialize_flag=True)
            else:
                # Fallback for other session types (TeamSession, WorkflowSession) - not yet implemented
                # Get session type value
                session_type_value = "agent"
                if hasattr(session, "session_type"):
                    st = session.session_type
                    if hasattr(st, "value"):
                        session_type_value = st.value
                    elif isinstance(st, str):
                        session_type_value = st

                async with self.async_session_factory() as sess, sess.begin():
                    stmt = sqlite.insert(table).values(
                        session_id=serialized_session.get("session_id"),
                        session_type=session_type_value,
                        agent_id=serialized_session.get("agent_id"),
                        team_id=serialized_session.get("team_id"),
                        workflow_id=serialized_session.get("workflow_id"),
                        user_id=serialized_session.get("user_id"),
                        session_data=serialized_session.get("session_data"),
                        agent_data=serialized_session.get("agent_data"),
                        team_data=serialized_session.get("team_data"),
                        workflow_data=serialized_session.get("workflow_data"),
                        metadata=serialized_session.get("metadata"),
                        runs=serialized_session.get("runs"),
                        messages=serialized_session.get("messages"),
                        usage=serialized_session.get("usage"),
                        summary=serialized_session.get("summary"),
                        created_at=serialized_session.get("created_at") or current_time,
                        updated_at=current_time,
                    )

                    stmt = stmt.on_conflict_do_update(
                        index_elements=["session_id"],
                        set_=dict(
                            session_type=session_type_value,
                            agent_id=serialized_session.get("agent_id"),
                            team_id=serialized_session.get("team_id"),
                            workflow_id=serialized_session.get("workflow_id"),
                            user_id=serialized_session.get("user_id"),
                            session_data=serialized_session.get("session_data"),
                            agent_data=serialized_session.get("agent_data"),
                            team_data=serialized_session.get("team_data"),
                            workflow_data=serialized_session.get("workflow_data"),
                            metadata=serialized_session.get("metadata"),
                            runs=serialized_session.get("runs"),
                            usage=serialized_session.get("usage"),
                            messages=serialized_session.get("messages"),
                            summary=serialized_session.get("summary"),
                            updated_at=current_time,
                        ),
                    )

                    stmt = stmt.returning(*table.columns)
                    result = await sess.execute(stmt)
                    row = result.fetchone()

                    if row is None:
                        return None

                    session_raw = deserialize_session_json_fields(dict(row._mapping))

                    if not deserialize:
                        return session_raw

                return deserialize_session(session_raw)

        except Exception as e:
            _logger.error(f"Error upserting session: {e}")
            raise e

    async def aupsert_sessions(
        self,
        sessions: List[Session],
        deserialize: bool = True,
    ) -> List[Union[Session, Dict[str, Any]]]:
        """Bulk insert or update multiple sessions."""
        if not sessions:
            return []

        # Validate all sessions have session_id
        for session in sessions:
            if not hasattr(session, "session_id") or session.session_id is None:
                raise ValueError("All sessions must have session_id set for upsert")

        try:
            table = await self._get_session_table(create_if_not_found=True)
            results: List[Union[Session, Dict[str, Any]]] = []
            current_time = int(time.time())

            async with self.async_session_factory() as sess, sess.begin():
                for session in sessions:
                    # Convert session to dict with serialize_flag=True
                    session_dict = session.to_dict(serialize_flag=True) if hasattr(session, "to_dict") else dict(session)
                    
                    # Serialize JSON fields for database storage
                    serialized_session = serialize_session_json_fields(session_dict)
                    
                    session_type_value = "agent"
                    if hasattr(session, "session_type"):
                        st = session.session_type
                        if hasattr(st, "value"):
                            session_type_value = st.value
                        elif isinstance(st, str):
                            session_type_value = st

                    stmt = sqlite.insert(table).values(
                        session_id=serialized_session.get("session_id"),
                        session_type=session_type_value,
                        agent_id=serialized_session.get("agent_id"),
                        team_id=serialized_session.get("team_id"),
                        workflow_id=serialized_session.get("workflow_id"),
                        user_id=serialized_session.get("user_id"),
                        session_data=serialized_session.get("session_data"),
                        agent_data=serialized_session.get("agent_data"),
                        team_data=serialized_session.get("team_data"),
                        workflow_data=serialized_session.get("workflow_data"),
                        metadata=serialized_session.get("metadata"),
                        runs=serialized_session.get("runs"),
                        messages=serialized_session.get("messages"),
                        usage=serialized_session.get("usage"),
                        summary=serialized_session.get("summary"),
                        created_at=serialized_session.get("created_at") or current_time,
                        updated_at=current_time,
                    )

                    stmt = stmt.on_conflict_do_update(
                        index_elements=["session_id"],
                        set_=dict(
                            session_type=session_type_value,
                            agent_id=serialized_session.get("agent_id"),
                            team_id=serialized_session.get("team_id"),
                            workflow_id=serialized_session.get("workflow_id"),
                            user_id=serialized_session.get("user_id"),
                            session_data=serialized_session.get("session_data"),
                            agent_data=serialized_session.get("agent_data"),
                            team_data=serialized_session.get("team_data"),
                            workflow_data=serialized_session.get("workflow_data"),
                            metadata=serialized_session.get("metadata"),
                            runs=serialized_session.get("runs"),
                            messages=serialized_session.get("messages"),
                            usage=serialized_session.get("usage"),
                            summary=serialized_session.get("summary"),
                            updated_at=current_time,
                        ),
                    )

                    stmt = stmt.returning(*table.columns)
                    result = await sess.execute(stmt)
                    row = result.fetchone()

                    if row is not None:
                        session_raw = deserialize_session_json_fields(dict(row._mapping))
                        if deserialize:
                            results.append(deserialize_session(session_raw))
                        else:
                            results.append(session_raw)

            return results

        except Exception as e:
            _logger.error(f"Error upserting sessions: {e}")
            raise e

    async def aget_session(
        self,
        session_id: Optional[str] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union[Session, Dict[str, Any]]]:
        """Get a session from the database."""
        try:
            table = await self._get_session_table(create_if_not_found=True)
        except ValueError:
            return None

        try:
            async with self.async_session_factory() as sess, sess.begin():
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

                result = await sess.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

                session_raw = deserialize_session_json_fields(dict(row._mapping))

                if not deserialize:
                    return session_raw

                return deserialize_session(session_raw, session_type)

        except Exception as e:
            _logger.error(f"Error getting session: {e}")
            raise e

    async def aget_sessions(
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
    ) -> Union[List[Session], Tuple[List[Dict[str, Any]], int]]:
        """Get multiple sessions from the database."""
        try:
            table = await self._get_session_table(create_if_not_found=True)
        except ValueError:
            return [] if deserialize else ([], 0)

        try:
            async with self.async_session_factory() as sess, sess.begin():
                stmt = select(table)

                # Apply filters
                if session_ids is not None:
                    if len(session_ids) == 0:
                        # Empty list means return no results
                        return [] if deserialize else ([], 0)
                    stmt = stmt.where(table.c.session_id.in_(session_ids))
                if session_type is not None:
                    stmt = stmt.where(table.c.session_type == session_type.value)
                if user_id is not None:
                    stmt = stmt.where(table.c.user_id == user_id)
                if agent_id is not None:
                    stmt = stmt.where(table.c.agent_id == agent_id)

                # Get total count
                count_stmt = select(func.count()).select_from(stmt.alias())
                count_result = await sess.execute(count_stmt)
                total_count = count_result.scalar() or 0

                # Apply sorting
                stmt = apply_sorting(stmt, table, sort_by, sort_order)

                # Apply pagination
                if limit is not None:
                    stmt = stmt.limit(limit)
                    if offset is not None:
                        stmt = stmt.offset(offset)

                result = await sess.execute(stmt)
                rows = result.fetchall()

                if not rows:
                    return [] if deserialize else ([], 0)

                sessions_raw = [
                    deserialize_session_json_fields(dict(row._mapping))
                    for row in rows
                ]

                if not deserialize:
                    return sessions_raw, total_count

                return [
                    deserialize_session(s, session_type)
                    for s in sessions_raw
                ]

        except Exception as e:
            _logger.error(f"Error getting sessions: {e}")
            raise e

    async def adelete_session(self, session_id: str) -> bool:
        """Delete a session from the database."""
        if not session_id:
            raise ValueError("session_id is required for delete")

        try:
            table = await self._get_session_table(create_if_not_found=False)
        except ValueError:
            return False

        try:
            async with self.async_session_factory() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.session_id == session_id)
                result = await sess.execute(delete_stmt)

                deleted = result.rowcount > 0
                if deleted:
                    _logger.debug(f"Deleted session: {session_id}")
                else:
                    _logger.debug(f"No session found to delete: {session_id}")
                return deleted

        except Exception as e:
            _logger.error(f"Error deleting session: {e}")
            raise e

    async def adelete_sessions(self, session_ids: List[str]) -> int:
        """Delete multiple sessions from the database."""
        if not session_ids:
            raise ValueError("session_ids is required and cannot be empty")

        try:
            table = await self._get_session_table(create_if_not_found=False)
        except ValueError:
            return 0

        try:
            async with self.async_session_factory() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.session_id.in_(session_ids))
                result = await sess.execute(delete_stmt)

                deleted_count = result.rowcount
                _logger.debug(f"Deleted {deleted_count} sessions")
                return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting sessions: {e}")
            raise e

    # ======================== User Memory Methods ========================

    async def aupsert_user_memory(
        self,
        user_memory: UserMemory,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """Insert or update user memory in the database."""
        if not user_memory.user_id:
            raise ValueError("user_memory must have user_id set for upsert")

        try:
            table = await self._get_user_memory_table(create_if_not_found=True)
            current_time = int(time.time())

            memory_data = user_memory.user_memory
            agent_id = user_memory.agent_id
            team_id = user_memory.team_id
            user_id = user_memory.user_id

            async with self.async_session_factory() as sess, sess.begin():
                stmt = sqlite.insert(table).values(
                    user_id=user_id,
                    user_memory=memory_data,
                    agent_id=agent_id,
                    team_id=team_id,
                    created_at=user_memory.created_at or current_time,
                    updated_at=current_time,
                )

                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id"],
                    set_=dict(
                        user_memory=memory_data,
                        agent_id=agent_id,
                        team_id=team_id,
                        updated_at=current_time,
                    ),
                )

                stmt = stmt.returning(*table.columns)
                result = await sess.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

                memory_dict = dict(row._mapping)
                if deserialize:
                    return UserMemory.from_dict(memory_dict)
                return memory_dict

        except Exception as e:
            _logger.error(f"Error upserting user memory: {e}")
            raise e

    async def aupsert_user_memories(
        self,
        user_memories: List[UserMemory],
        deserialize: bool = True,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        """Bulk insert or update multiple user memories."""
        if not user_memories:
            return []

        # Validate all have user_id
        for memory in user_memories:
            if not memory.user_id:
                raise ValueError("All user memories must have user_id set")

        try:
            table = await self._get_user_memory_table(create_if_not_found=True)
            results: List[Union[UserMemory, Dict[str, Any]]] = []
            current_time = int(time.time())

            async with self.async_session_factory() as sess, sess.begin():
                for memory in user_memories:
                    stmt = sqlite.insert(table).values(
                        user_id=memory.user_id,
                        user_memory=memory.user_memory,
                        agent_id=memory.agent_id,
                        team_id=memory.team_id,
                        created_at=memory.created_at or current_time,
                        updated_at=current_time,
                    )

                    stmt = stmt.on_conflict_do_update(
                        index_elements=["user_id"],
                        set_=dict(
                            user_memory=memory.user_memory,
                            agent_id=memory.agent_id,
                            team_id=memory.team_id,
                            updated_at=current_time,
                        ),
                    )

                    stmt = stmt.returning(*table.columns)
                    result = await sess.execute(stmt)
                    row = result.fetchone()

                    if row is not None:
                        memory_dict = dict(row._mapping)
                        if deserialize:
                            results.append(UserMemory.from_dict(memory_dict))
                        else:
                            results.append(memory_dict)

            return results

        except Exception as e:
            _logger.error(f"Error upserting user memories: {e}")
            raise e

    async def aget_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """Get user memory from the database."""
        try:
            table = await self._get_user_memory_table(create_if_not_found=True)
        except ValueError:
            return None

        try:
            async with self.async_session_factory() as sess, sess.begin():
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

                result = await sess.execute(stmt)
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

    async def aget_user_memories(
        self,
        user_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        deserialize: bool = True,
    ) -> Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
        """Get multiple user memories from the database."""
        try:
            table = await self._get_user_memory_table(create_if_not_found=False)
        except ValueError:
            return [] if deserialize else ([], 0)

        try:
            async with self.async_session_factory() as sess, sess.begin():
                stmt = select(table)

                if user_ids is not None:
                    if len(user_ids) == 0:
                        # Empty list means return no results
                        return [] if deserialize else ([], 0)
                    stmt = stmt.where(table.c.user_id.in_(user_ids))
                if agent_id is not None:
                    stmt = stmt.where(table.c.agent_id == agent_id)
                if team_id is not None:
                    stmt = stmt.where(table.c.team_id == team_id)

                # Get total count
                count_stmt = select(func.count()).select_from(stmt.alias())
                count_result = await sess.execute(count_stmt)
                total_count = count_result.scalar() or 0

                # Apply sorting (by updated_at desc)
                stmt = stmt.order_by(table.c.updated_at.desc())

                # Apply pagination
                if limit is not None:
                    stmt = stmt.limit(limit)
                    if offset is not None:
                        stmt = stmt.offset(offset)

                result = await sess.execute(stmt)
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

    async def adelete_user_memory(self, user_id: str) -> bool:
        """Delete user memory from the database."""
        if not user_id:
            raise ValueError("user_id is required for delete")

        try:
            table = await self._get_user_memory_table(create_if_not_found=False)
        except ValueError:
            return False

        try:
            async with self.async_session_factory() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.user_id == user_id)
                result = await sess.execute(delete_stmt)

                deleted = result.rowcount > 0
                if deleted:
                    _logger.debug(f"Deleted user memory: {user_id}")
                else:
                    _logger.debug(f"No user memory found to delete: {user_id}")
                return deleted

        except Exception as e:
            _logger.error(f"Error deleting user memory: {e}")
            raise e

    async def adelete_user_memories(self, user_ids: List[str]) -> int:
        """Delete multiple user memories from the database."""
        if not user_ids:
            raise ValueError("user_ids is required and cannot be empty")

        try:
            table = await self._get_user_memory_table(create_if_not_found=False)
        except ValueError:
            return 0

        try:
            async with self.async_session_factory() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.user_id.in_(user_ids))
                result = await sess.execute(delete_stmt)

                deleted_count = result.rowcount
                _logger.debug(f"Deleted {deleted_count} user memories")
                return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting user memories: {e}")
            raise e

    # ======================== Utility Methods ========================

    async def aclear_all(self) -> None:
        """Clear all data from all tables."""
        try:
            # Clear sessions
            try:
                table = await self._get_session_table(create_if_not_found=False)
                async with self.async_session_factory() as sess, sess.begin():
                    await sess.execute(table.delete())
                _logger.debug("Cleared all sessions")
            except ValueError:
                pass

            # Clear user memories
            try:
                table = await self._get_user_memory_table(create_if_not_found=False)
                async with self.async_session_factory() as sess, sess.begin():
                    await sess.execute(table.delete())
                _logger.debug("Cleared all user memories")
            except ValueError:
                pass

            # Clear knowledge
            try:
                table = await self._get_knowledge_table(create_if_not_found=False)
                async with self.async_session_factory() as sess, sess.begin():
                    await sess.execute(table.delete())
                _logger.debug("Cleared all knowledge entries")
            except ValueError:
                pass

            _logger.info("Cleared all data from storage")

        except Exception as e:
            _logger.error(f"Error clearing all data: {e}")
            raise e

    # ======================== Generic Model Methods (Async) ========================

    async def _get_generic_model_table(self, collection: str, create_if_not_found: bool = True) -> Table:
        """Get or create a generic model table for a collection."""
        from sqlalchemy import String, JSON, BigInteger
        
        table_name = f"upsonic_models_{collection}"
        
        if table_name in self._tables:
            return self._tables[table_name]
        
        # Check if table exists
        from upsonic.storage.sqlite.utils import is_table_available_async
        async with self.engine.begin() as conn:
            table_exists = await is_table_available_async(conn, table_name)
            if table_exists:
                # Reflect existing table using run_sync
                await conn.run_sync(
                    lambda sync_conn: self._metadata.reflect(bind=sync_conn, only=[table_name])
                )
                if table_name in self._metadata.tables:
                    self._tables[table_name] = self._metadata.tables[table_name]
                    return self._tables[table_name]
        
        if not create_if_not_found:
            raise ValueError(f"Table {table_name} does not exist")
        
        # Create new table
        table = Table(
            table_name,
            self._metadata,
            Column("key", String, primary_key=True, nullable=False),
            Column("collection", String, nullable=False, index=True),
            Column("model_data", JSON, nullable=False),
            Column("created_at", BigInteger, nullable=False, index=True),
            Column("updated_at", BigInteger, nullable=True, index=True),
            extend_existing=True,
        )
        
        async with self.engine.begin() as conn:
            await conn.run_sync(table.create, checkfirst=True)
        
        self._tables[table_name] = table
        _logger.debug(f"Created generic model table: {table_name}")
        return table

    async def aupsert_model(
        self,
        key: str,
        model: Any,
        collection: str = "generic_models",
    ) -> None:
        """Insert or update a generic Pydantic model in storage (async)."""
        import time
        
        try:
            table = await self._get_generic_model_table(collection)
            current_time = int(time.time())
            
            # Serialize model
            if hasattr(model, 'model_dump'):
                model_data = model.model_dump(mode='json')
            elif hasattr(model, 'dict'):
                model_data = model.dict()
            else:
                model_data = dict(model)
            
            record = {
                "key": key,
                "collection": collection,
                "model_data": model_data,
                "updated_at": current_time,
            }
            
            async with self.async_session_factory() as sess, sess.begin():
                # Check if exists
                stmt = select(table).where(table.c.key == key)
                result = await sess.execute(stmt)
                existing = result.fetchone()
                
                if existing:
                    # Update
                    update_stmt = table.update().where(table.c.key == key).values(**record)
                    await sess.execute(update_stmt)
                else:
                    # Insert
                    record["created_at"] = current_time
                    await sess.execute(table.insert().values(**record))
            
            _logger.debug(f"Upserted model with key '{key}' in collection '{collection}'")
            
        except Exception as e:
            _logger.error(f"Error upserting model: {e}")
            raise

    async def aget_model(
        self,
        key: str,
        model_type: Any,
        collection: str = "generic_models",
    ) -> Optional[Any]:
        """Retrieve a generic Pydantic model from storage (async)."""
        try:
            table = await self._get_generic_model_table(collection, create_if_not_found=False)
            
            async with self.async_session_factory() as sess:
                stmt = select(table).where(table.c.key == key)
                result = await sess.execute(stmt)
                row = result.fetchone()
                
                if row is None:
                    return None
                
                model_data = row.model_data
                return model_type(**model_data)
                
        except ValueError:
            return None
        except Exception as e:
            _logger.error(f"Error getting model: {e}")
            return None

    async def adelete_model(
        self,
        key: str,
        collection: str = "generic_models",
    ) -> bool:
        """Delete a generic model from storage (async)."""
        try:
            table = await self._get_generic_model_table(collection, create_if_not_found=False)
            
            async with self.async_session_factory() as sess, sess.begin():
                stmt = table.delete().where(table.c.key == key)
                result = await sess.execute(stmt)
                
                if result.rowcount > 0:
                    _logger.debug(f"Deleted model with key '{key}' from collection '{collection}'")
                    return True
                return False
                
        except ValueError:
            return False
        except Exception as e:
            _logger.error(f"Error deleting model: {e}")
            return False

    async def alist_models(
        self,
        model_type: Any,
        collection: str = "generic_models",
    ) -> List[Any]:
        """List all models of a type in a collection (async)."""
        try:
            table = await self._get_generic_model_table(collection, create_if_not_found=False)
            
            async with self.async_session_factory() as sess:
                stmt = select(table).where(table.c.collection == collection)
                result = await sess.execute(stmt)
                rows = result.fetchall()
                
                models = []
                for row in rows:
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

    async def _get_cultural_knowledge_table(self, create_if_not_found: bool = False) -> Table:
        """Get the cultural knowledge table, creating if needed."""
        if self._cultural_knowledge_table is None:
            self._cultural_knowledge_table = await self._get_or_create_table(
                table_name=self.cultural_knowledge_table_name,
                table_type="cultural_knowledge",
                create_if_not_found=create_if_not_found,
            )
        return self._cultural_knowledge_table

    async def _get_knowledge_table(self, create_if_not_found: bool = False) -> Table:
        """Get the knowledge document registry table, creating if needed."""
        if self._knowledge_table is None:
            self._knowledge_table = await self._get_or_create_table(
                table_name=self.knowledge_table_name,
                table_type="knowledge",
                create_if_not_found=create_if_not_found,
            )
        return self._knowledge_table

    async def adelete_cultural_knowledge(self, id: str) -> None:
        """Delete cultural knowledge from the database (async).
        
        Args:
            id: The ID of the cultural knowledge to delete.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        try:
            table = await self._get_cultural_knowledge_table(create_if_not_found=False)

            async with self.async_session_factory() as sess, sess.begin():
                delete_stmt = table.delete().where(table.c.id == id)
                result = await sess.execute(delete_stmt)

                success = result.rowcount > 0
                if success:
                    _logger.debug(f"Successfully deleted cultural knowledge id: {id}")
                else:
                    _logger.debug(f"No cultural knowledge found with id: {id}")

        except Exception as e:
            _logger.error(f"Error deleting cultural knowledge: {e}")
            raise e

    async def aget_cultural_knowledge(
        self,
        id: str,
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Get cultural knowledge from the database (async).
        
        Args:
            id: The ID of the cultural knowledge to get.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            CulturalKnowledge object or dict if found, None otherwise.
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        try:
            table = await self._get_cultural_knowledge_table(create_if_not_found=False)

            async with self.async_session_factory() as sess, sess.begin():
                stmt = select(table).where(table.c.id == id)
                result = await sess.execute(stmt)
                row = result.fetchone()
                if row is None:
                    return None

                db_row = dict(row._mapping)
                if not deserialize:
                    return db_row
                return CulturalKnowledge.from_dict(db_row)

        except Exception as e:
            _logger.error(f"Exception reading from cultural knowledge table: {e}")
            raise e

    async def aget_all_cultural_knowledge(
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
        """Get all cultural knowledge entries from the database (async).
        
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
        from sqlalchemy import func
        
        try:
            table = await self._get_cultural_knowledge_table(create_if_not_found=False)

            async with self.async_session_factory() as sess, sess.begin():
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
                count_result = await sess.execute(count_stmt)
                total_count = count_result.scalar() or 0

                # Sorting
                stmt = apply_sorting(stmt, table, sort_by, sort_order)
                
                # Paginating
                if limit is not None:
                    stmt = stmt.limit(limit)
                    if page is not None:
                        stmt = stmt.offset((page - 1) * limit)

                result = await sess.execute(stmt)
                rows = result.fetchall()
                if not rows:
                    return [] if deserialize else ([], 0)

                db_rows = [dict(record._mapping) for record in rows]
                if not deserialize:
                    return db_rows, total_count
                return [CulturalKnowledge.from_dict(row) for row in db_rows]

        except Exception as e:
            _logger.error(f"Error reading from cultural knowledge table: {e}")
            raise e

    async def aupsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Upsert cultural knowledge into the database (async).
        
        Args:
            cultural_knowledge: The CulturalKnowledge instance to upsert.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            The upserted CulturalKnowledge object or dict, or None if error occurs.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        import uuid
        
        try:
            table = await self._get_cultural_knowledge_table(create_if_not_found=True)

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

            async with self.async_session_factory() as sess, sess.begin():
                stmt = sqlite.insert(table).values(**data)
                stmt = stmt.on_conflict_do_update(  # type: ignore
                    index_elements=["id"],
                    set_={k: v for k, v in data.items() if k != "id"},
                ).returning(table)

                result = await sess.execute(stmt)
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

    # ======================== Knowledge Content Methods (Async) ========================

    async def aupsert_knowledge_content(
        self,
        knowledge_row: "KnowledgeRow",
    ) -> Optional["KnowledgeRow"]:
        """Insert or update a knowledge document registry entry (async)."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            table = await self._get_knowledge_table(create_if_not_found=True)
            current_time = int(time.time())

            data = knowledge_row.to_dict()
            data.setdefault("created_at", current_time)
            data["updated_at"] = current_time

            async with self.async_session_factory() as sess, sess.begin():
                update_fields = {k: v for k, v in data.items() if k != "id"}
                stmt = (
                    sqlite.insert(table)
                    .values(**data)
                    .on_conflict_do_update(
                        index_elements=["id"],
                        set_=update_fields,
                    )
                    .returning(table)
                )
                result = await sess.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

            return KnowledgeRow.from_dict(dict(row._mapping))

        except Exception as e:
            _logger.error(f"Error upserting knowledge content: {e}")
            raise e

    async def aget_knowledge_content(
        self,
        id: str,
    ) -> Optional["KnowledgeRow"]:
        """Get a knowledge document registry entry by ID (async)."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            table = await self._get_knowledge_table(create_if_not_found=False)
        except ValueError:
            return None

        try:
            async with self.async_session_factory() as sess, sess.begin():
                stmt = select(table).where(table.c.id == id)
                result = await sess.execute(stmt)
                row = result.fetchone()
                if row is None:
                    return None
                return KnowledgeRow.from_dict(dict(row._mapping))

        except Exception as e:
            _logger.error(f"Error getting knowledge content: {e}")
            raise e

    async def aget_knowledge_contents(
        self,
        knowledge_base_id: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> Tuple[List["KnowledgeRow"], int]:
        """Get knowledge document registry entries with filtering and pagination (async)."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            table = await self._get_knowledge_table(create_if_not_found=False)
        except ValueError:
            return [], 0

        try:
            async with self.async_session_factory() as sess, sess.begin():
                stmt = select(table)

                if knowledge_base_id is not None:
                    stmt = stmt.where(table.c.knowledge_base_id == knowledge_base_id)

                count_stmt = select(func.count()).select_from(stmt.alias())
                count_result = await sess.execute(count_stmt)
                total_count: int = count_result.scalar() or 0

                stmt = apply_sorting(stmt, table, sort_by, sort_order)

                if limit is not None:
                    stmt = stmt.limit(limit)
                    if page is not None:
                        stmt = stmt.offset((page - 1) * limit)

                result = await sess.execute(stmt)
                rows = result.fetchall()
                if not rows:
                    return [], 0

                records = [KnowledgeRow.from_dict(dict(record._mapping)) for record in rows]
                return records, total_count

        except Exception as e:
            _logger.error(f"Error getting knowledge contents: {e}")
            raise e

    async def adelete_knowledge_content(self, id: str) -> bool:
        """Delete a knowledge document registry entry by ID (async)."""
        try:
            table = await self._get_knowledge_table(create_if_not_found=False)
        except ValueError:
            return False

        try:
            async with self.async_session_factory() as sess, sess.begin():
                stmt = table.delete().where(table.c.id == id)
                result = await sess.execute(stmt)

                deleted: bool = result.rowcount > 0
                if deleted:
                    _logger.debug(f"Deleted knowledge content: {id}")
                return deleted

        except Exception as e:
            _logger.error(f"Error deleting knowledge content: {e}")
            raise e

    async def adelete_knowledge_contents(self, ids: List[str]) -> int:
        """Delete multiple knowledge document registry entries (async)."""
        if not ids:
            return 0

        try:
            table = await self._get_knowledge_table(create_if_not_found=False)
        except ValueError:
            return 0

        try:
            async with self.async_session_factory() as sess, sess.begin():
                stmt = table.delete().where(table.c.id.in_(ids))
                result = await sess.execute(stmt)

                deleted_count: int = result.rowcount
                _logger.debug(f"Deleted {deleted_count} knowledge entries")
                return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting knowledge contents: {e}")
            raise e
