from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Optional, Type, Union, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

try:
    import aiosqlite
    _AIOSQLITE_AVAILABLE = True
except ImportError:
    aiosqlite = None  # type: ignore
    _AIOSQLITE_AVAILABLE = False


from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import InteractionSession, UserProfile

T = TypeVar('T', bound=BaseModel)

class SqliteStorage(Storage):
    """
    A hybrid sync/async, file-based storage provider using a single SQLite
    database and the `aiosqlite` driver with proper connection management.
    """

    def __init__(
        self,
        sessions_table_name: str,
        profiles_table_name: str,
        db_file: Optional[str] = None,
    ):
        """
        Initializes the async SQLite storage provider.

        Args:
            sessions_table_name: Name of the table for InteractionSession storage.
            profiles_table_name: Name of the table for UserProfile storage.
            db_file: Path to a local database file. If None, uses in-memory DB.
        """
        if not _AIOSQLITE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="aiosqlite",
                install_command='pip install "upsonic[storage]"',
                feature_name="SQLite storage provider"
            )

        super().__init__()
        self.db_path = ":memory:"
        if db_file:
            db_path_obj = Path(db_file).resolve()
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_path_obj)
        
        self.sessions_table_name = sessions_table_name
        self.profiles_table_name = profiles_table_name
        self._db: Optional[aiosqlite.Connection] = None


    
    def is_connected(self) -> bool: return self._run_async_from_sync(self.is_connected_async())
    def connect(self) -> None: return self._run_async_from_sync(self.connect_async())
    def disconnect(self) -> None: return self._run_async_from_sync(self.disconnect_async())
    def create(self) -> None: return self._run_async_from_sync(self.create_async())
    def read(self, object_id: str, model_type: Type[T]) -> Optional[T]: return self._run_async_from_sync(self.read_async(object_id, model_type))
    def upsert(self, data: Union[InteractionSession, UserProfile]) -> None: return self._run_async_from_sync(self.upsert_async(data))
    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None: return self._run_async_from_sync(self.delete_async(object_id, model_type))
    def drop(self) -> None: return self._run_async_from_sync(self.drop_async())



    async def is_connected_async(self) -> bool:
        return self._db is not None

    async def connect_async(self) -> None:
        if await self.is_connected_async():
            return
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row # Important for dict-like access
        await self.create_async()
        self._connected = True

    async def disconnect_async(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
        self._connected = False

    async def _get_connection(self) -> aiosqlite.Connection:
        """Helper to lazily initialize the database connection."""
        if not await self.is_connected_async():
            await self.connect_async()
        return self._db

    async def create_async(self) -> None:
        db = await self._get_connection()
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.sessions_table_name} (
                session_id TEXT PRIMARY KEY, user_id TEXT, agent_id TEXT,
                team_session_id TEXT, chat_history TEXT, summary TEXT,
                session_data TEXT, extra_data TEXT, created_at REAL, updated_at REAL
            )
        """)
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.profiles_table_name} (
                user_id TEXT PRIMARY KEY, profile_data TEXT,
                created_at REAL, updated_at REAL
            )
        """)
        await db.commit()

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        if model_type is InteractionSession: table, key_col = self.sessions_table_name, "session_id"
        elif model_type is UserProfile: table, key_col = self.profiles_table_name, "user_id"
        else: return None

        db = await self._get_connection()
        sql = f"SELECT * FROM {table} WHERE {key_col} = ?"
        async with db.execute(sql, (object_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                data = dict(row)
                for key, value in data.items():
                    if key in ['chat_history', 'session_data', 'extra_data', 'profile_data'] and isinstance(value, str):
                        data[key] = json.loads(value)
                return model_type.from_dict(data)
        return None

    async def upsert_async(self, data: Union[InteractionSession, UserProfile]) -> None:
        data.updated_at = time.time()
        
        if isinstance(data, InteractionSession):
            table = self.sessions_table_name
            sql = f"""
                INSERT INTO {table} (session_id, user_id, agent_id, team_session_id, chat_history, summary, session_data, extra_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=excluded.user_id, agent_id=excluded.agent_id, team_session_id=excluded.team_session_id,
                    chat_history=excluded.chat_history, summary=excluded.summary, session_data=excluded.session_data,
                    extra_data=excluded.extra_data, updated_at=excluded.updated_at
            """
            params = (
                data.session_id, data.user_id, data.agent_id, data.team_session_id,
                json.dumps(data.chat_history), data.summary, json.dumps(data.session_data),
                json.dumps(data.extra_data), data.created_at, data.updated_at
            )
        elif isinstance(data, UserProfile):
            table = self.profiles_table_name
            sql = f"""
                INSERT INTO {table} (user_id, profile_data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_data=excluded.profile_data, updated_at=excluded.updated_at
            """
            params = (data.user_id, json.dumps(data.profile_data), data.created_at, data.updated_at)
        else:
            raise TypeError(f"Unsupported data type for upsert: {type(data).__name__}")

        db = await self._get_connection()
        await db.execute(sql, params)
        await db.commit()

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        if model_type is InteractionSession: table, key_col = self.sessions_table_name, "session_id"
        elif model_type is UserProfile: table, key_col = self.profiles_table_name, "user_id"
        else: return

        db = await self._get_connection()
        sql = f"DELETE FROM {table} WHERE {key_col} = ?"
        await db.execute(sql, (object_id,))
        await db.commit()

    async def drop_async(self) -> None:
        db = await self._get_connection()
        await db.execute(f"DROP TABLE IF EXISTS {self.sessions_table_name}")
        await db.execute(f"DROP TABLE IF EXISTS {self.profiles_table_name}")
        await db.commit()
