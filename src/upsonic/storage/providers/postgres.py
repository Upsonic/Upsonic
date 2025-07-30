import time
from typing import Literal, Optional, List

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import (
    BaseSession,
    AgentSession
)

try:
    from sqlalchemy import create_engine, inspect as sqlalchemy_inspect, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import sessionmaker, Session as SqlAlchemySession
    from sqlalchemy.schema import Column, MetaData, Table
    from sqlalchemy.types import String, BigInteger
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.sql.expression import select
except ImportError:
    raise ImportError("`sqlalchemy` and `psycopg2-binary` are required for PostgresStorage. Please install them.")


class PostgresStorage(Storage):
    """
    Production-grade, session-based storage provider using PostgreSQL.

    This provider uses SQLAlchemy Core and PostgreSQL's native JSONB and
    UPSERT capabilities for high-performance, transactional session management.
    """

    def __init__(
        self,
        table_name: str,
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
        schema: str = "public",
        auto_upgrade_schema: bool = False,
        mode: Literal["agent", "team", "workflow", "workflow_v2"] = "agent",
    ):
        """
        Initializes the PostgreSQL storage provider.

        The connection is determined using the following precedence:
        1. An existing `db_engine` object.
        2. A full `db_url` connection string.

        Args:
            table_name: The name of the table for session storage.
            db_url: An optional SQLAlchemy database URL.
            db_engine: An optional, pre-configured SQLAlchemy Engine.
            schema: The PostgreSQL schema to use for the session table.
            auto_upgrade_schema: If True, attempts to run schema migrations automatically.
            mode: The operational mode, which determines the table schema.
        
        Raises:
            ValueError: If neither `db_url` nor `db_engine` is provided.
        """
        super().__init__(mode=mode)

        # THE REFACTORED CONSTRUCTOR LOGIC
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)

        if _engine is None:
            raise ValueError("Must provide either a `db_url` string or a pre-configured SQLAlchemy `db_engine` object.")

        self.db_engine: Engine = _engine

        # Database attributes
        self.table_name: str = table_name
        self.schema: str = schema
        self.metadata: MetaData = MetaData(schema=self.schema)
        self.inspector = sqlalchemy_inspect(self.db_engine)

        # Schema management attributes
        self.auto_upgrade_schema: bool = auto_upgrade_schema

        # Database session and table definition
        self.SqlSession = sessionmaker(bind=self.db_engine)
        self.table: Table = self._get_table_schema()
        
        self.create()
        self.connect()


    def _get_session_model_class(self) -> type[BaseSession]:
        """Returns the appropriate Pydantic session model class based on the current mode."""
        if self.mode == "agent":
            return AgentSession
        raise ValueError(f"Invalid mode '{self.mode}' specified for PostgresStorage.")

    def _get_table_schema(self) -> Table:
        """Dynamically defines the SQLAlchemy Table schema for PostgreSQL."""
        common_columns = [
            Column("session_id", String(64), primary_key=True),
            Column("user_id", String, index=True),
            Column("memory", postgresql.JSONB),
            Column("session_data", postgresql.JSONB),
            Column("extra_data", postgresql.JSONB),
            Column("created_at", postgresql.DOUBLE_PRECISION, nullable=False),
            Column("updated_at", postgresql.DOUBLE_PRECISION, nullable=False, index=True),
        ]
        specific_columns = []
        if self.mode == "agent":
            specific_columns = [
                Column("agent_id", String, index=True),
                Column("agent_data", postgresql.JSONB),
                Column("team_session_id", String(64), index=True, nullable=True),
            ]
        return Table(self.table_name, self.metadata, *common_columns, *specific_columns, extend_existing=True)

    def _table_exists(self) -> bool:
        """Checks if the storage table exists in the database."""
        return self.inspector.has_table(self.table_name, schema=self.schema)


    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self.is_connected():
            return
        
        try:
            with self.db_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            self._connected = True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    def disconnect(self) -> None:
        if not self.is_connected():
            return

        self.db_engine.dispose()
        self._connected = False

    def create(self) -> None:
        """Creates the schema and table if they do not already exist."""
        with self.db_engine.connect() as connection:
            if self.schema and not self.inspector.has_schema(self.schema):
                connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
            self.metadata.create_all(connection, checkfirst=True)
            if hasattr(connection, 'commit'):
                connection.commit()

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[BaseSession]:
        with self.SqlSession() as sess:
            stmt = select(self.table).where(self.table.c.session_id == session_id)
            if user_id:
                stmt = stmt.where(self.table.c.user_id == user_id)
            result = sess.execute(stmt).first()
            if result:
                session_data = dict(result._mapping)
                SessionModel = self._get_session_model_class()
                return SessionModel.from_dict(session_data)
        return None

    def upsert(self, session: BaseSession) -> Optional[BaseSession]:
        session.updated_at = time.time()

        session_dict = session.model_dump(mode="json")
        
        insert_stmt = postgresql.insert(self.table).values(session_dict)
        update_cols = {col.name: col for col in insert_stmt.excluded if col.name not in ["session_id", "created_at"]}
        upsert_stmt = insert_stmt.on_conflict_do_update(index_elements=["session_id"], set_=update_cols)

        try:
            with self.SqlSession() as sess, sess.begin():
                sess.execute(upsert_stmt)
            return self.read(session.session_id)
        except Exception as e:
            raise IOError(f"Failed to upsert session {session.session_id} to PostgreSQL: {e}") from e

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[BaseSession]:
        return self._get_sessions_query(user_id, entity_id)

    def get_recent_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 10) -> List[BaseSession]:
        return self._get_sessions_query(user_id, entity_id, limit)

    def _get_sessions_query(self, user_id: Optional[str], entity_id: Optional[str], limit: Optional[int] = None) -> List[BaseSession]:
        with self.SqlSession() as sess:
            stmt = postgresql.select(self.table).order_by(self.table.c.updated_at.desc())
            if user_id:
                stmt = stmt.where(self.table.c.user_id == user_id)
            if entity_id:
                entity_col_map = {"agent": "agent_id"}
                col_name = entity_col_map.get(self.mode)
                if col_name:
                    stmt = stmt.where(getattr(self.table.c, col_name) == entity_id)
            if limit:
                stmt = stmt.limit(limit)
            results = sess.execute(stmt).fetchall()
            SessionModel = self._get_session_model_class()
            return [SessionModel.from_dict(row._mapping) for row in results]

    def delete_session(self, session_id: str) -> None:
        with self.SqlSession() as sess, sess.begin():
            delete_stmt = self.table.delete().where(self.table.c.session_id == session_id)
            sess.execute(delete_stmt)

    def drop(self) -> None:
        self.metadata.drop_all(self.db_engine, tables=[self.table], checkfirst=True)

    def log_artifact(self, artifact) -> None:
        raise NotImplementedError("Artifact logging should be handled within the session data for this provider.")

    def store_artifact_data(self, artifact_id: str, session_id: str, binary_data: bytes) -> str:
        raise NotImplementedError("Binary artifact storage is not handled by the session provider.")

    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        raise NotImplementedError("Binary artifact storage is not handled by the session provider.")