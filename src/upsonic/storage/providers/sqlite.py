from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Optional, Type, Union, TypeVar, TYPE_CHECKING, List

if TYPE_CHECKING:
    import aiosqlite
    from upsonic.culture.cultural_knowledge import CulturalKnowledge

try:
    import aiosqlite
    _AIOSQLITE_AVAILABLE = True
except ImportError:
    aiosqlite = None  # type: ignore
    _AIOSQLITE_AVAILABLE = False


from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.session.agent import AgentSession

T = TypeVar('T', bound=BaseModel)

class SqliteStorage(Storage):
    """
    A hybrid sync/async, file-based storage provider using a single SQLite
    database and the `aiosqlite` driver with proper connection management.
    
    This storage provider is designed to be flexible and dynamic:
    - Can accept a pre-existing aiosqlite connection or create one from connection details
    - Only creates AgentSession tables when they are actually used
    - Supports generic Pydantic models for custom storage needs
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    def __init__(
        self,
        db: Optional['aiosqlite.Connection'] = None,
        db_file: Optional[str] = None,
        agent_sessions_table_name: Optional[str] = None,
        cultural_knowledge_table_name: Optional[str] = None,
    ):
        """
        Initializes the async SQLite storage provider.

        Args:
            db: Optional pre-existing aiosqlite.Connection. If provided, this connection
                will be used instead of creating a new one. User is responsible for
                connection lifecycle management when providing their own connection.
            db_file: Path to a local database file. If None and db is None, uses in-memory DB.
                Ignored if db is provided.
            agent_sessions_table_name: Name of the table for AgentSession storage.
                Defaults to "agent_sessions".
            cultural_knowledge_table_name: Name of the table for CulturalKnowledge storage.
                Only used if CulturalKnowledge objects are stored. Defaults to "cultural_knowledge".
        """
        if not _AIOSQLITE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="aiosqlite",
                install_command='pip install "upsonic[storage]"',
                feature_name="SQLite storage provider"
            )

        super().__init__()
        
        self._db: Optional[aiosqlite.Connection] = db
        self._owns_connection = (db is None)
        
        self.db_path = ":memory:"
        if db_file and not db:
            db_path_obj = Path(db_file).resolve()
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_path_obj)
        
        self.agent_sessions_table_name = agent_sessions_table_name or "agent_sessions"
        self.cultural_knowledge_table_name = cultural_knowledge_table_name or "cultural_knowledge"
        
        self._agent_sessions_table_initialized = False
        self._cultural_knowledge_table_initialized = False


    
    def is_connected(self) -> bool: return self._run_async_from_sync(self.is_connected_async())
    def connect(self) -> None: return self._run_async_from_sync(self.connect_async())
    def disconnect(self) -> None: return self._run_async_from_sync(self.disconnect_async())
    def create(self) -> None: return self._run_async_from_sync(self.create_async())
    def read(self, object_id: str, model_type: Type[T]) -> Optional[T]: return self._run_async_from_sync(self.read_async(object_id, model_type))
    def upsert(self, data: BaseModel) -> None: return self._run_async_from_sync(self.upsert_async(data))
    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None: return self._run_async_from_sync(self.delete_async(object_id, model_type))
    def drop(self) -> None: return self._run_async_from_sync(self.drop_async())



    async def is_connected_async(self) -> bool:
        return self._db is not None

    async def connect_async(self) -> None:
        if await self.is_connected_async():
            return
        
        if not self._owns_connection:
            self._connected = True
            return
        
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        self._connected = True

    async def disconnect_async(self) -> None:
        if not self._owns_connection:
            return
        
        if self._db:
            await self._db.close()
            self._db = None
        self._connected = False

    async def _get_connection(self) -> aiosqlite.Connection:
        # Check if connection needs to be established or re-established
        needs_reconnect = False
        
        if self._db is None:
            needs_reconnect = True
        else:
            # Check if aiosqlite connection's internal thread is still running
            # The _running attribute indicates if the background thread is active
            if hasattr(self._db, '_running') and not self._db._running:
                needs_reconnect = True
            elif hasattr(self._db, '_connection') and self._db._connection is None:
                needs_reconnect = True
            else:
                # Try a simple query to verify connection is usable
                try:
                    async with self._db.execute("SELECT 1") as cursor:
                        await cursor.fetchone()
                except Exception:
                    needs_reconnect = True
        
        if needs_reconnect:
            # Close existing connection if it exists
            if self._db is not None:
                try:
                    await self._db.close()
                except Exception:
                    pass
                self._db = None
            
            # Create new connection
            await self.connect_async()
        
        return self._db

    async def create_async(self) -> None:
        await self._get_connection()
    
    async def _ensure_agent_sessions_table(self) -> None:
        if self._agent_sessions_table_initialized:
            return
        
        db = await self._get_connection()
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.agent_sessions_table_name} (
                session_id TEXT PRIMARY KEY,
                agent_id TEXT,
                user_id TEXT,
                workflow_id TEXT,
                session_data TEXT,
                metadata TEXT,
                user_profile TEXT,
                agent_data TEXT,
                runs TEXT,
                summary TEXT,
                messages TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
        """)
        await db.commit()
        self._agent_sessions_table_initialized = True
    
    async def _ensure_cultural_knowledge_table(self) -> None:
        if self._cultural_knowledge_table_initialized:
            return
        
        db = await self._get_connection()
        await db.execute(f"""
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
                created_at INTEGER,
                updated_at INTEGER
            )
        """)
        await db.commit()
        self._cultural_knowledge_table_initialized = True

    def _get_table_info_for_model(self, model_type: Type[BaseModel]) -> Optional[tuple[str, str]]:
        if model_type.__name__ == "AgentSession" or (hasattr(model_type, '__mro__') and any(c.__name__ == "AgentSession" for c in model_type.__mro__)):
            return (self.agent_sessions_table_name, "session_id")
        else:
            table_name = f"{model_type.__name__.lower()}_storage"
            if hasattr(model_type, 'model_fields'):
                fields = model_type.model_fields
                for id_field in ['path', 'id', 'key', 'name']:
                    if id_field in fields:
                        return (table_name, id_field)
            return (table_name, "id")
    
    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        table_info = self._get_table_info_for_model(model_type)
        if table_info is None:
            return None
        
        table, key_col = table_info

        # Get connection first to ensure it's established
        db = await self._get_connection()
        
        if model_type.__name__ == "AgentSession":
            await self._ensure_agent_sessions_table()
        else:
            await self._ensure_table_for_model(model_type)

        sql = f"SELECT * FROM {table} WHERE {key_col} = ?"
        async with db.execute(sql, (object_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                data = dict(row)
                
                if model_type.__name__ == "AgentSession":
                    import base64
                    # Use deserialize() method from runs column (base64 encoded)
                    # This is the primary and only reliable method for AgentSession
                    runs_data = data.get('runs')
                    if runs_data and isinstance(runs_data, str):
                        try:
                            # runs column contains base64-encoded serialized AgentSession
                            serialized_bytes = base64.b64decode(runs_data.encode('utf-8'))
                            return model_type.deserialize(serialized_bytes)
                        except Exception as e:
                            from upsonic.utils.printing import warning_log
                            warning_log(f"Failed to deserialize AgentSession from SQLite: {e}", "SqliteStorage")
                            return None
                    
                    # If runs column is empty/None, session doesn't exist in new format
                    return None
                else:
                    if 'data' in data and isinstance(data['data'], str):
                        try:
                            obj_data = json.loads(data['data'])
                            return model_type.model_validate(obj_data)
                        except Exception:
                            return None
        return None

    async def _ensure_table_for_model(self, model_type: Type[BaseModel]) -> str:
        table_info = self._get_table_info_for_model(model_type)
        if table_info is None:
            raise TypeError(f"Cannot determine table for model: {model_type.__name__}")
        
        table_name, key_col = table_info
        
        if model_type.__name__ != "AgentSession":
            db = await self._get_connection()
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {key_col} TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            await db.commit()
        
        return table_name
    
    async def upsert_async(self, data: BaseModel) -> None:
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()
        
        if type(data).__name__ == "AgentSession":
            await self._ensure_agent_sessions_table()
            
            import base64
            # Use serialize() method to get bytes, then base64 encode for SQLite TEXT storage
            serialized_bytes = data.serialize()
            serialized_str = base64.b64encode(serialized_bytes).decode('utf-8')
            
            table = self.agent_sessions_table_name
            sql = f"""
                INSERT INTO {table} (
                    session_id, agent_id, user_id, workflow_id,
                    session_data, metadata, user_profile, agent_data, runs, summary, messages,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    agent_id=excluded.agent_id, user_id=excluded.user_id, workflow_id=excluded.workflow_id,
                    session_data=excluded.session_data, metadata=excluded.metadata, user_profile=excluded.user_profile,
                    agent_data=excluded.agent_data, runs=excluded.runs, summary=excluded.summary, messages=excluded.messages,
                    updated_at=excluded.updated_at
            """
            # Store key fields for querying + base64-encoded serialized data in runs column
            params = (
                data.session_id,
                data.agent_id,
                data.user_id,
                data.workflow_id,
                None,  # session_data
                None,  # metadata  
                None,  # user_profile
                None,  # agent_data
                serialized_str,  # Full serialized AgentSession (base64)
                data.summary,
                None,  # messages
                data.created_at,
                int(data.updated_at) if data.updated_at else None
            )
        else:
            table_name = await self._ensure_table_for_model(type(data))
            table_info = self._get_table_info_for_model(type(data))
            _, key_col = table_info
            
            key_value = getattr(data, key_col)
            
            data_json = data.model_dump_json()
            
            created_at = getattr(data, 'created_at', time.time())
            updated_at = getattr(data, 'updated_at', time.time())
            
            sql = f"""
                INSERT INTO {table_name} ({key_col}, data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT({key_col}) DO UPDATE SET
                    data=excluded.data, updated_at=excluded.updated_at
            """
            params = (key_value, data_json, created_at, updated_at)

        db = await self._get_connection()
        await db.execute(sql, params)
        await db.commit()

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        table_info = self._get_table_info_for_model(model_type)
        if table_info is None:
            return
        
        table, key_col = table_info
        
        if model_type.__name__ == "AgentSession":
            await self._ensure_agent_sessions_table()

        db = await self._get_connection()
        sql = f"DELETE FROM {table} WHERE {key_col} = ?"
        await db.execute(sql, (object_id,))
        await db.commit()
    
    async def list_all_async(self, model_type: Type[T]) -> List[T]:
        table_info = self._get_table_info_for_model(model_type)
        if table_info is None:
            return []
        
        table_name, key_col = table_info
        
        if model_type.__name__ == "AgentSession":
            await self._ensure_agent_sessions_table()
        else:
            await self._ensure_table_for_model(model_type)
        
        db = await self._get_connection()
        
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        ) as cursor:
            if not await cursor.fetchone():
                return []
        
        sql = f"SELECT * FROM {table_name}"
        results = []
        
        async with db.execute(sql) as cursor:
            async for row in cursor:
                data = dict(row)
                
                if model_type.__name__ == "AgentSession":
                    import base64
                    # Use deserialize() method from runs column (base64 encoded)
                    runs_data = data.get('runs')
                    if runs_data and isinstance(runs_data, str):
                        try:
                            # runs column contains base64-encoded serialized AgentSession
                            serialized_bytes = base64.b64decode(runs_data.encode('utf-8'))
                            obj = model_type.deserialize(serialized_bytes)
                            results.append(obj)
                        except Exception:
                            # Skip invalid entries
                            continue
                    else:
                        # Skip entries without serialized data
                        continue
                else:
                    if 'data' in data and isinstance(data['data'], str):
                        try:
                            obj_data = json.loads(data['data'])
                            obj = model_type.model_validate(obj_data)
                            results.append(obj)
                        except Exception:
                            continue
                    else:
                        try:
                            obj = model_type.model_validate(data)
                            results.append(obj)
                        except Exception:
                            continue
        
        return results

    async def drop_async(self) -> None:
        db = await self._get_connection()
        
        await db.execute(f"DROP TABLE IF EXISTS {self.agent_sessions_table_name}")
        await db.execute(f"DROP TABLE IF EXISTS {self.cultural_knowledge_table_name}")
        
        self._agent_sessions_table_initialized = False
        self._cultural_knowledge_table_initialized = False
        
        await db.commit()

    # =========================================================================
    # Cultural Knowledge Methods
    # =========================================================================

    async def read_cultural_knowledge_async(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        await self._ensure_cultural_knowledge_table()
        
        db = await self._get_connection()
        sql = f"SELECT * FROM {self.cultural_knowledge_table_name} WHERE id = ?"
        
        async with db.execute(sql, (knowledge_id,)) as cursor:
            row = await cursor.fetchone()
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
        
        db = await self._get_connection()
        
        sql = f"""
            INSERT INTO {self.cultural_knowledge_table_name} 
            (id, name, content, summary, categories, notes, metadata, input, agent_id, team_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, content=excluded.content, summary=excluded.summary,
                categories=excluded.categories, notes=excluded.notes, metadata=excluded.metadata,
                input=excluded.input, agent_id=excluded.agent_id, team_id=excluded.team_id,
                updated_at=excluded.updated_at
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
        
        await db.execute(sql, params)
        await db.commit()

    async def delete_cultural_knowledge_async(self, knowledge_id: str) -> None:
        await self._ensure_cultural_knowledge_table()
        
        db = await self._get_connection()
        sql = f"DELETE FROM {self.cultural_knowledge_table_name} WHERE id = ?"
        await db.execute(sql, (knowledge_id,))
        await db.commit()

    async def list_all_cultural_knowledge_async(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        await self._ensure_cultural_knowledge_table()
        
        db = await self._get_connection()
        
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self.cultural_knowledge_table_name,)
        ) as cursor:
            if not await cursor.fetchone():
                return []
        
        if name:
            sql = f"SELECT * FROM {self.cultural_knowledge_table_name} WHERE name LIKE ?"
            params = (f"%{name}%",)
        else:
            sql = f"SELECT * FROM {self.cultural_knowledge_table_name}"
            params = ()
        
        results = []
        
        async with db.execute(sql, params) as cursor:
            async for row in cursor:
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
        
        db = await self._get_connection()
        sql = f"DELETE FROM {self.cultural_knowledge_table_name}"
        await db.execute(sql)
        await db.commit()
