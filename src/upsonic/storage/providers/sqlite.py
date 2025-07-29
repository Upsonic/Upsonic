import time
import json
from pathlib import Path
from typing import List, Literal, Optional

from upsonic.storage.base import Storage, SchemaMismatchError
from upsonic.storage.sessions import (
    BaseSession,
    AgentSession
)
from upsonic.storage.settings import SQLiteSettings

try:
    from sqlalchemy import create_engine, inspect as sqlalchemy_inspect, text
    from sqlalchemy import Text
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import sessionmaker, Session as SqlAlchemySession
    from sqlalchemy.schema import Column, MetaData, Table
    from sqlalchemy.types import String, Integer, JSON
    from sqlalchemy.dialects import sqlite
    from sqlalchemy.sql.expression import select
except ImportError:
    raise ImportError("`sqlalchemy` is required for SQLiteStorage. Please install it using `pip install sqlalchemy`.")


class SqliteStorage(Storage):
    """
    A session-based storage provider that uses a SQLite database, configured
    via a Pydantic settings object.
    """

    def __init__(self, settings: SQLiteSettings):
        """
        Initializes the SQLite storage provider from a settings object.

        This constructor is designed to be called by the StorageFactory, which
        passes a validated instance of SQLiteSettings. All necessary parameters
        (db path, table name, mode) are extracted from this object.

        Args:
            settings: A validated SQLiteSettings object containing all configuration.
        """
        super().__init__(mode=settings.STORAGE_MODE)

        self.db_path = str(settings.SQLITE_DB_PATH)
        self.table_name = settings.SQLITE_TABLE_NAME

        self.db_uri = f"sqlite:///{self.db_path}" if self.db_path != ":memory:" else "sqlite:///"
        self.db_engine: Engine = create_engine(self.db_uri)
        self.metadata = MetaData()
        self.inspector = sqlalchemy_inspect(self.db_engine)
        self.SqlSession = sessionmaker(bind=self.db_engine)
        
        self.table: Table = self._get_table_schema()

        self.create()
        self.connect()

    def _get_session_model_class(self) -> type[BaseSession]:
        """Returns the appropriate Pydantic session model class based on the current mode."""
        if self.mode == "agent":
            return AgentSession
        raise ValueError(f"Invalid mode '{self.mode}' specified for SqliteStorage.")

    def _get_table_schema(self) -> Table:
        """Dynamically defines the SQLAlchemy Table schema based on the current mode."""
        common_columns = [
            Column("session_id", String, primary_key=True),
            Column("user_id", String, index=True),
            Column("memory", JSON),
            Column("session_data", sqlite.JSON),
            Column("extra_data", sqlite.JSON),
            Column("created_at", sqlite.REAL),
            Column("updated_at", sqlite.REAL, index=True),
        ]
        specific_columns = []
        if self.mode == "agent":
            specific_columns = [Column("agent_id", String, index=True), Column("agent_data", sqlite.JSON), Column("team_session_id", String, index=True, nullable=True)]
        elif self.mode == "team":
            specific_columns = [Column("team_id", String, index=True), Column("team_data", sqlite.JSON), Column("team_session_id", String, index=True, nullable=True)]
        elif self.mode == "workflow":
            specific_columns = [Column("workflow_id", String, index=True), Column("workflow_data", sqlite.JSON)]
        elif self.mode == "workflow_v2":
            specific_columns = [Column("workflow_id", String, index=True), Column("workflow_name", String, index=True), Column("workflow_data", sqlite.JSON), Column("runs", sqlite.JSON)]
        return Table(self.table_name, self.metadata, *common_columns, *specific_columns, extend_existing=True)


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
            raise ConnectionError(f"Failed to connect to SQLite at {self.db_path}: {e}")

    def disconnect(self) -> None:
        if not self.is_connected():
            return
        
        self.db_engine.dispose()
        self._connected = False

    def create(self) -> None:
        self.metadata.create_all(self.db_engine, checkfirst=True)

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
        SessionModel = self._get_session_model_class()
        if not isinstance(session, SessionModel):
            raise TypeError(f"Session object must be of type {SessionModel.__name__} for mode '{self.mode}'")

        session.updated_at = time.time()
        session_dict = session.model_dump(mode="json")

        insert_stmt = sqlite.insert(self.table).values(session_dict)
        update_dict = {key: value for key, value in session_dict.items() if key != "session_id"}

        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["session_id"],
            set_=update_dict
        )
        try:
            with self.SqlSession() as sess, sess.begin():
                sess.execute(upsert_stmt)
            return self.read(session.session_id)
        except Exception as e:
            raise IOError(f"Failed to upsert session {session.session_id}: {e}") from e

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[BaseSession]:
        return self._get_sessions_query(user_id, entity_id)

    def get_recent_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 10) -> List[BaseSession]:
        return self._get_sessions_query(user_id, entity_id, limit)

    def _get_sessions_query(self, user_id: Optional[str], entity_id: Optional[str], limit: Optional[int] = None) -> List[BaseSession]:
        with self.SqlSession() as sess:
            stmt = select(self.table).order_by(self.table.c.updated_at.desc())
            if user_id:
                stmt = stmt.where(self.table.c.user_id == user_id)
            if entity_id:
                entity_col_map = {"agent": "agent_id", "team": "team_id", "workflow": "workflow_id", "workflow_v2": "workflow_id"}
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