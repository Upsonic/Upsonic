from __future__ import annotations

import time
import json
from typing import List, Optional, Type, Union, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg
    from upsonic.culture.cultural_knowledge import CulturalKnowledge

try:
    import asyncpg
    _ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG_AVAILABLE = False

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.session.agent import AgentSession

T = TypeVar('T', bound=BaseModel)

class PostgresStorage(Storage):
    """
    A hybrid sync/async, production-grade storage provider using PostgreSQL
    and the `asyncpg` driver with a connection pool.
    
    This storage provider is designed to be flexible and dynamic:
    - Can accept a pre-existing asyncpg connection pool or create one from connection details
    - Only creates AgentSession tables when they are actually used
    - Supports generic Pydantic models for custom storage needs
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    def __init__(
        self,
        pool: Optional['asyncpg.Pool'] = None,
        db_url: Optional[str] = None,
        schema: str = "public",
        agent_sessions_table_name: Optional[str] = None,
        cultural_knowledge_table_name: Optional[str] = None,
    ):
        """
        Initializes the async PostgreSQL storage provider.

        Args:
            pool: Optional pre-existing asyncpg.Pool. If provided, this pool will be used
                instead of creating a new one. User is responsible for pool lifecycle
                management when providing their own pool.
            db_url: An asyncpg-compatible database URL (e.g., "postgresql://user:pass@host:port/db").
                Required if pool is not provided. Ignored if pool is provided.
            schema: The PostgreSQL schema to use for the tables. Defaults to "public".
            agent_sessions_table_name: The name of the table for AgentSession storage.
                Only used if AgentSession objects are stored. Defaults to "agent_sessions".
            cultural_knowledge_table_name: The name of the table for CulturalKnowledge storage.
                Only used if CulturalKnowledge objects are stored. Defaults to "cultural_knowledge".
        """
        if not _ASYNCPG_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="asyncpg",
                install_command='pip install "upsonic[storage]"',
                feature_name="PostgreSQL storage provider"
            )

        super().__init__()
        
        self._pool = pool
        self._owns_pool = (pool is None)
        
        if not pool and not db_url:
            raise ValueError("Either 'pool' or 'db_url' must be provided")
        self.db_url = db_url
        self.schema = schema
        
        self.agent_sessions_table_name = f'"{schema}"."{agent_sessions_table_name or "agent_sessions"}"'
        self.cultural_knowledge_table_name = f'"{schema}"."{cultural_knowledge_table_name or "cultural_knowledge"}"'
        
        self._agent_sessions_table_initialized = False
        self._cultural_knowledge_table_initialized = False



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
    def upsert(self, data: BaseModel) -> None:
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
        
        if not self._owns_pool:
            self._connected = True
            return
        
        try:
            self._pool = await asyncpg.create_pool(self.db_url)
            await self._ensure_schema()
            self._connected = True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    async def disconnect_async(self) -> None:
        if not self._owns_pool:
            return
        
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._connected = False

    async def _get_pool(self):
        """Get connection pool with auto-reconnect if needed."""
        needs_reconnect = False
        
        if self._pool is None:
            needs_reconnect = True
        elif self._pool._closing:
            needs_reconnect = True
        else:
            # Test connection with a simple query
            try:
                async with self._pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
            except Exception:
                needs_reconnect = True
        
        if needs_reconnect:
            # Close existing pool if it exists
            if self._pool is not None:
                try:
                    await self._pool.close()
                except Exception:
                    pass
                self._pool = None
            
            # Reconnect
            await self.connect_async()
        
        return self._pool
    
    async def _ensure_schema(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

    async def create_async(self) -> None:
        await self._ensure_schema()
    
    async def _ensure_agent_sessions_table(self) -> None:
        if self._agent_sessions_table_initialized:
            return
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.agent_sessions_table_name} (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL,
                    updated_at REAL
                )
            """)
        self._agent_sessions_table_initialized = True
    
    async def _ensure_cultural_knowledge_table(self) -> None:
        if self._cultural_knowledge_table_initialized:
            return
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.cultural_knowledge_table_name} (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    content TEXT,
                    summary TEXT,
                    categories TEXT,
                    notes TEXT,
                    metadata TEXT,
                    input TEXT,
                    agent_id TEXT,
                    team_id TEXT,
                    created_at BIGINT,
                    updated_at BIGINT
                )
            """)
        self._cultural_knowledge_table_initialized = True

    def _get_table_info(self, model_type: Type[BaseModel]) -> Optional[tuple[str, str]]:
        if model_type.__name__ == "AgentSession":
            return (self.agent_sessions_table_name, "session_id")
        else:
            table_name = f'"{self.schema}"."{model_type.__name__.lower()}_storage"'
            if hasattr(model_type, 'model_fields'):
                for field in ['path', 'id', 'key', 'name']:
                    if field in model_type.model_fields:
                        return (table_name, field)
            return (table_name, "id")
    
    async def _ensure_table(self, model_type: Type[BaseModel]) -> str:
        table_info = self._get_table_info(model_type)
        if table_info is None:
            raise TypeError(f"Cannot determine table for {model_type.__name__}")
        
        table_name, key_col = table_info
        
        if model_type.__name__ != "AgentSession":
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
        import base64
        table_info = self._get_table_info(model_type)
        if table_info is None:
            return None
        
        table, key_col = table_info
        
        if model_type.__name__ == "AgentSession":
            await self._ensure_agent_sessions_table()
        else:
            await self._ensure_table(model_type)
        
        pool = await self._get_pool()
        sql = f"SELECT data FROM {table} WHERE {key_col} = $1"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, object_id)
            if row:
                data_str = row['data']
                
                if model_type.__name__ == "AgentSession":
                    # Use deserialize: base64 decode then deserialize
                    serialized_bytes = base64.b64decode(data_str.encode('utf-8'))
                    return model_type.deserialize(serialized_bytes)
                else:
                    if 'data' in row and isinstance(row['data'], str):
                        obj_data = json.loads(row['data'])
                        return model_type.model_validate(obj_data)
        return None

    async def upsert_async(self, data: BaseModel) -> None:
        import base64
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()

        if type(data).__name__ == "AgentSession":
            await self._ensure_agent_sessions_table()
            
            # Use serialize: serialize to bytes, then base64 encode for TEXT storage
            serialized_bytes = data.serialize()
            serialized_str = base64.b64encode(serialized_bytes).decode('utf-8')
            
            table = self.agent_sessions_table_name
            sql = f"""
                INSERT INTO {table} (session_id, data, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT(session_id) DO UPDATE SET
                    data=EXCLUDED.data, updated_at=EXCLUDED.updated_at
            """
            params = (
                data.session_id,
                serialized_str,
                data.created_at or time.time(),
                data.updated_at or time.time()
            )
        else:
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
        
        if model_type.__name__ == "AgentSession":
            await self._ensure_agent_sessions_table()
            
        pool = await self._get_pool()
        sql = f"DELETE FROM {table} WHERE {key_col} = $1"
        async with pool.acquire() as conn:
            await conn.execute(sql, object_id)
    
    async def list_all_async(self, model_type: Type[T]) -> list[T]:
        import base64
        table_info = self._get_table_info(model_type)
        if table_info is None:
            return []
        
        table_name, key_col = table_info
        
        if model_type.__name__ == "AgentSession":
            await self._ensure_agent_sessions_table()
        else:
            await self._ensure_table(model_type)
        
        pool = await self._get_pool()
        sql = f"SELECT data FROM {table_name}"
        results = []
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)
            
            for row in rows:
                data_str = row['data']
                
                if model_type.__name__ == "AgentSession":
                    # Use deserialize: base64 decode then deserialize
                    try:
                        serialized_bytes = base64.b64decode(data_str.encode('utf-8'))
                        obj = model_type.deserialize(serialized_bytes)
                        results.append(obj)
                    except Exception:
                        continue
                else:
                    if isinstance(data_str, str):
                        try:
                            obj_data = json.loads(data_str)
                            obj = model_type.model_validate(obj_data)
                            results.append(obj)
                        except Exception:
                            continue
        
        return results

    async def drop_async(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self.agent_sessions_table_name}")
            await conn.execute(f"DROP TABLE IF EXISTS {self.cultural_knowledge_table_name}")
        
        self._agent_sessions_table_initialized = False
        self._cultural_knowledge_table_initialized = False

    # =========================================================================
    # Cultural Knowledge Methods
    # =========================================================================

    async def read_cultural_knowledge_async(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        await self._ensure_cultural_knowledge_table()
        
        pool = await self._get_pool()
        sql = f"SELECT * FROM {self.cultural_knowledge_table_name} WHERE id = $1"
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, knowledge_id)
            if row:
                data = dict(row)
                for key in ['categories', 'notes', 'metadata']:
                    if key in data and isinstance(data[key], str):
                        try:
                            data[key] = json.loads(data[key])
                        except:
                            pass
                return CulturalKnowledge.from_dict(data)
        return None

    async def upsert_cultural_knowledge_async(self, knowledge: "CulturalKnowledge") -> None:
        await self._ensure_cultural_knowledge_table()
        
        knowledge.bump_updated_at()
        
        pool = await self._get_pool()
        
        sql = f"""
            INSERT INTO {self.cultural_knowledge_table_name} 
            (id, name, content, summary, categories, notes, metadata, input, agent_id, team_id, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT(id) DO UPDATE SET
                name=EXCLUDED.name, content=EXCLUDED.content, summary=EXCLUDED.summary,
                categories=EXCLUDED.categories, notes=EXCLUDED.notes, metadata=EXCLUDED.metadata,
                input=EXCLUDED.input, agent_id=EXCLUDED.agent_id, team_id=EXCLUDED.team_id,
                updated_at=EXCLUDED.updated_at
        """
        
        params = (
            knowledge.id,
            knowledge.name,
            knowledge.content,
            knowledge.summary,
            json.dumps(knowledge.categories) if knowledge.categories else None,
            json.dumps(knowledge.notes) if knowledge.notes else None,
            json.dumps(knowledge.metadata) if knowledge.metadata else None,
            knowledge.input,
            knowledge.agent_id,
            knowledge.team_id,
            knowledge.created_at,
            knowledge.updated_at,
        )
        
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def delete_cultural_knowledge_async(self, knowledge_id: str) -> None:
        await self._ensure_cultural_knowledge_table()
        
        pool = await self._get_pool()
        sql = f"DELETE FROM {self.cultural_knowledge_table_name} WHERE id = $1"
        
        async with pool.acquire() as conn:
            await conn.execute(sql, knowledge_id)

    async def list_all_cultural_knowledge_async(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        await self._ensure_cultural_knowledge_table()
        
        pool = await self._get_pool()
        
        if name:
            sql = f"SELECT * FROM {self.cultural_knowledge_table_name} WHERE name ILIKE $1"
            params = (f"%{name}%",)
        else:
            sql = f"SELECT * FROM {self.cultural_knowledge_table_name}"
            params = ()
        
        results = []
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            
            for row in rows:
                data = dict(row)
                for key in ['categories', 'notes', 'metadata']:
                    if key in data and isinstance(data[key], str):
                        try:
                            data[key] = json.loads(data[key])
                        except:
                            pass
                results.append(CulturalKnowledge.from_dict(data))
        
        return results

    async def clear_cultural_knowledge_async(self) -> None:
        await self._ensure_cultural_knowledge_table()
        
        pool = await self._get_pool()
        sql = f"DELETE FROM {self.cultural_knowledge_table_name}"
        
        async with pool.acquire() as conn:
            await conn.execute(sql)
