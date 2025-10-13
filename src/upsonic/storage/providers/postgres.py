from __future__ import annotations

import time
import json
from typing import Optional, Type, Union, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

try:
    import asyncpg
    _ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG_AVAILABLE = False

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import InteractionSession, UserProfile

T = TypeVar('T', bound=BaseModel)

class PostgresStorage(Storage):
    """
    A hybrid sync/async, production-grade storage provider using PostgreSQL
    and the `asyncpg` driver with a connection pool.
    """

    def __init__(self, sessions_table_name: str, profiles_table_name: str, db_url: str, schema: str = "public"):
        """
        Initializes the async PostgreSQL storage provider.

        Args:
            sessions_table_name: The name of the table for InteractionSession storage.
            profiles_table_name: The name of the table for UserProfile storage.
            db_url: An asyncpg-compatible database URL (e.g., "postgresql://user:pass@host:port/db").
            schema: The PostgreSQL schema to use for the tables.
        """
        if not _ASYNCPG_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="asyncpg",
                install_command='pip install "upsonic[storage]"',
                feature_name="PostgreSQL storage provider"
            )

        super().__init__()
        self.db_url = db_url
        self.sessions_table_name = f'"{schema}"."{sessions_table_name}"'
        self.profiles_table_name = f'"{schema}"."{profiles_table_name}"'
        self.schema = schema
        self._pool = None



    def is_connected(self) -> bool:
        return self._run_async_from_sync(self.is_connected_async())
    def connect(self) -> None:
        return self._run_async_from_sync(self.connect_async())
    def disconnect(self) -> None:
        return self._run_async_from_sync(self.disconnect_async())
    def create(self) -> None:
        return self._run_async_from_sync(self.create_async())
    def read(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        return self._run_async_from_sync(self.read_async(object_id, model_type))
    def upsert(self, data: Union[InteractionSession, UserProfile]) -> None:
        return self._run_async_from_sync(self.upsert_async(data))
    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None:
        return self._run_async_from_sync(self.delete_async(object_id, model_type))
    def drop(self) -> None:
        return self._run_async_from_sync(self.drop_async())
    


    async def is_connected_async(self) -> bool:
        return self._pool is not None and not self._pool._closing
    
    async def connect_async(self) -> None:
        if await self.is_connected_async():
            return
        try:
            self._pool = await asyncpg.create_pool(self.db_url)
            # Verify connection and ensure schema/tables exist
            await self.create_async()
            self._connected = True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    async def disconnect_async(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._connected = False

    async def _get_pool(self):
        """Helper to lazily initialize the connection pool."""
        if not await self.is_connected_async():
            await self.connect_async()
        return self._pool

    async def create_async(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.sessions_table_name} (
                    session_id TEXT PRIMARY KEY, user_id TEXT, agent_id TEXT,
                    team_session_id TEXT, chat_history TEXT, summary TEXT,
                    session_data TEXT, extra_data TEXT, created_at REAL, updated_at REAL
                )
            """)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.profiles_table_name} (
                    user_id TEXT PRIMARY KEY, profile_data TEXT,
                    created_at REAL, updated_at REAL
                )
            """)

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        if model_type is InteractionSession:
            table, key_col = self.sessions_table_name, "session_id"
        elif model_type is UserProfile:
            table, key_col = self.profiles_table_name, "user_id"
        else:
            return None
        
        pool = await self._get_pool()
        sql = f"SELECT * FROM {table} WHERE {key_col} = $1"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, object_id)
            if row:
                data = dict(row)
                for key in ['chat_history', 'session_data', 'extra_data', 'profile_data']:
                    if key in data and isinstance(data[key], str):
                        try:
                            data[key] = json.loads(data[key])
                        except Exception:
                            pass
                return model_type.from_dict(data)
        return None
    async def upsert_async(self, data: Union[InteractionSession, UserProfile]) -> None:
        data.updated_at = time.time()

        if isinstance(data, InteractionSession):
            table = self.sessions_table_name
            sql = f"""
                INSERT INTO {table} (session_id, user_id, agent_id, team_session_id, chat_history, summary, session_data, extra_data, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=EXCLUDED.user_id, agent_id=EXCLUDED.agent_id, team_session_id=EXCLUDED.team_session_id,
                    chat_history=EXCLUDED.chat_history, summary=EXCLUDED.summary, session_data=EXCLUDED.session_data,
                    extra_data=EXCLUDED.extra_data, updated_at=EXCLUDED.updated_at
            """
            params = (data.session_id, data.user_id, data.agent_id, data.team_session_id, json.dumps(data.chat_history), data.summary, json.dumps(data.session_data), json.dumps(data.extra_data), data.created_at, data.updated_at)
        elif isinstance(data, UserProfile):
            table = self.profiles_table_name
            sql = f"""
                INSERT INTO {table} (user_id, profile_data, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_data=EXCLUDED.profile_data, updated_at=EXCLUDED.updated_at
            """
            params = (data.user_id, json.dumps(data.profile_data), data.created_at, data.updated_at)
        else:
            raise TypeError(f"Unsupported data type for upsert: {type(data).__name__}")
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        if model_type is InteractionSession:
            table, key_col = self.sessions_table_name, "session_id"
        elif model_type is UserProfile:
            table, key_col = self.profiles_table_name, "user_id"
        else:
            return
            
        pool = await self._get_pool()
        sql = f"DELETE FROM {table} WHERE {key_col} = $1"
        async with pool.acquire() as conn:
            await conn.execute(sql, object_id)

    async def drop_async(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self.sessions_table_name}")
            await conn.execute(f"DROP TABLE IF EXISTS {self.profiles_table_name}")
            