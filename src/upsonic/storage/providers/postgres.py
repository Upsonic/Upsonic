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

    def __init__(self, db_url: str, schema: str = "public", sessions_table_name: Optional[str] = None, profiles_table_name: Optional[str] = None):
        """
        Initializes the async PostgreSQL storage provider.

        Args:
            db_url: An asyncpg-compatible database URL (e.g., "postgresql://user:pass@host:port/db").
            schema: The PostgreSQL schema to use for the tables.
            sessions_table_name: The name of the table for InteractionSession storage.
            profiles_table_name: The name of the table for UserProfile storage.
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
        self.sessions_table_name = f'"{schema}"."{sessions_table_name or "sessions"}"'
        self.profiles_table_name = f'"{schema}"."{profiles_table_name or "profiles"}"'
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

    def _get_table_info(self, model_type: Type[BaseModel]) -> Optional[tuple[str, str]]:
        """Get table name and primary key for a model type."""
        if model_type is InteractionSession:
            return (self.sessions_table_name, "session_id")
        elif model_type is UserProfile:
            return (self.profiles_table_name, "user_id")
        else:
            # Generic model
            table_name = f'"{self.schema}"."{model_type.__name__.lower()}_storage"'
            # Auto-detect primary key
            if hasattr(model_type, 'model_fields'):
                for field in ['path', 'id', 'key', 'name']:
                    if field in model_type.model_fields:
                        return (table_name, field)
            return (table_name, "id")
    
    async def _ensure_table(self, model_type: Type[BaseModel]) -> str:
        """Ensure table exists for model type."""
        table_info = self._get_table_info(model_type)
        if table_info is None:
            raise TypeError(f"Cannot determine table for {model_type.__name__}")
        
        table_name, key_col = table_info
        
        if model_type not in [InteractionSession, UserProfile]:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        {key_col} TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at REAL,
                        updated_at REAL
                    )
                """)
        
        return table_name
    
    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        table_info = self._get_table_info(model_type)
        if table_info is None:
            return None
        
        table, key_col = table_info
        
        # Ensure table exists for generic models
        if model_type not in [InteractionSession, UserProfile]:
            await self._ensure_table(model_type)
        
        pool = await self._get_pool()
        sql = f"SELECT * FROM {table} WHERE {key_col} = $1"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, object_id)
            if row:
                data = dict(row)
                
                if model_type in [InteractionSession, UserProfile]:
                    # Handle known types
                    for key in ['chat_history', 'session_data', 'extra_data', 'profile_data']:
                        if key in data and isinstance(data[key], str):
                            try:
                                data[key] = json.loads(data[key])
                            except Exception:
                                pass
                    if hasattr(model_type, 'from_dict'):
                        return model_type.from_dict(data)
                    else:
                        return model_type.model_validate(data)
                else:
                    # Generic model
                    if 'data' in data and isinstance(data['data'], str):
                        obj_data = json.loads(data['data'])
                        return model_type.model_validate(obj_data)
        return None
    async def upsert_async(self, data: BaseModel) -> None:
        if hasattr(data, 'updated_at'):
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
            # Generic model support
            table_name = await self._ensure_table(type(data))
            _, key_col = self._get_table_info(type(data))
            
            key_value = getattr(data, key_col)
            data_json = data.model_dump_json()
            created_at = getattr(data, 'created_at', time.time())
            updated_at = getattr(data, 'updated_at', time.time())
            
            sql = f"""
                INSERT INTO {table_name} ({key_col}, data, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT({key_col}) DO UPDATE SET
                    data=EXCLUDED.data, updated_at=EXCLUDED.updated_at
            """
            params = (key_value, data_json, created_at, updated_at)
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        table_info = self._get_table_info(model_type)
        if table_info is None:
            return
        
        table, key_col = table_info
            
        pool = await self._get_pool()
        sql = f"DELETE FROM {table} WHERE {key_col} = $1"
        async with pool.acquire() as conn:
            await conn.execute(sql, object_id)
    
    async def list_all_async(self, model_type: Type[T]) -> list[T]:
        """List all objects of a specific type."""
        table_info = self._get_table_info(model_type)
        if table_info is None:
            return []
        
        table_name, key_col = table_info
        
        # Ensure table exists
        await self._ensure_table(model_type)
        
        pool = await self._get_pool()
        sql = f"SELECT * FROM {table_name}"
        results = []
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)
            
            for row in rows:
                data = dict(row)
                
                if model_type in [InteractionSession, UserProfile]:
                    # Handle known types
                    for key in ['chat_history', 'session_data', 'extra_data', 'profile_data']:
                        if key in data and isinstance(data[key], str):
                            try:
                                data[key] = json.loads(data[key])
                            except Exception:
                                pass
                    
                    if hasattr(model_type, 'from_dict'):
                        obj = model_type.from_dict(data)
                    else:
                        obj = model_type.model_validate(data)
                else:
                    # Generic model
                    if 'data' in data and isinstance(data['data'], str):
                        try:
                            obj_data = json.loads(data['data'])
                            obj = model_type.model_validate(obj_data)
                        except Exception:
                            continue
                    else:
                        continue
                
                results.append(obj)
        
        return results

    async def drop_async(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self.sessions_table_name}")
            await conn.execute(f"DROP TABLE IF EXISTS {self.profiles_table_name}")
            