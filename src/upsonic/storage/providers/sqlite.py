from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Optional, Type, Union, TypeVar, TYPE_CHECKING, List

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
    
    This storage provider is designed to be flexible and dynamic:
    - Can accept a pre-existing aiosqlite connection or create one from connection details
    - Only creates InteractionSession/UserProfile tables when they are actually used
    - Supports generic Pydantic models for custom storage needs
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    def __init__(
        self,
        db: Optional['aiosqlite.Connection'] = None,
        db_file: Optional[str] = None,
        sessions_table_name: Optional[str] = None,
        profiles_table_name: Optional[str] = None,
    ):
        """
        Initializes the async SQLite storage provider.

        Args:
            db: Optional pre-existing aiosqlite.Connection. If provided, this connection
                will be used instead of creating a new one. User is responsible for
                connection lifecycle management when providing their own connection.
            db_file: Path to a local database file. If None and db is None, uses in-memory DB.
                Ignored if db is provided.
            sessions_table_name: Name of the table for InteractionSession storage.
                Only used if InteractionSession objects are stored. Defaults to "sessions".
            profiles_table_name: Name of the table for UserProfile storage.
                Only used if UserProfile objects are stored. Defaults to "profiles".
        """
        if not _AIOSQLITE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="aiosqlite",
                install_command='pip install "upsonic[storage]"',
                feature_name="SQLite storage provider"
            )

        super().__init__()
        
        # Store connection and track ownership for lifecycle management
        self._db: Optional[aiosqlite.Connection] = db
        self._owns_connection = (db is None)  # True if we create it, False if user provided
        
        # Connection details for creating our own connection if needed
        self.db_path = ":memory:"
        if db_file and not db:
            db_path_obj = Path(db_file).resolve()
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_path_obj)
        
        # Table names for InteractionSession/UserProfile (lazy initialization)
        self.sessions_table_name = sessions_table_name or "sessions"
        self.profiles_table_name = profiles_table_name or "profiles"
        
        # Track which built-in tables have been initialized
        self._sessions_table_initialized = False
        self._profiles_table_initialized = False


    
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
        """
        Establishes connection to the database.
        If user provided a connection, this is a no-op.
        Otherwise, creates a new connection using db_path.
        """
        if await self.is_connected_async():
            return
        
        if not self._owns_connection:
            # User provided connection, already set in __init__
            self._connected = True
            return
        
        # Create our own connection
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row  # Important for dict-like access
        self._connected = True

    async def disconnect_async(self) -> None:
        """
        Closes the database connection.
        If user provided the connection, this is a no-op (user manages lifecycle).
        """
        if not self._owns_connection:
            # User manages their own connection lifecycle
            return
        
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
        """
        Creates database schema.
        Note: InteractionSession and UserProfile tables are created lazily
        only when first accessed. This allows the storage to be used for
        generic purposes without creating unused infrastructure.
        """
        # Ensure connection exists, but don't create any tables yet
        # Tables will be created on-demand when accessed
        await self._get_connection()
    
    async def _ensure_sessions_table(self) -> None:
        """Lazily creates the sessions table on first access."""
        if self._sessions_table_initialized:
            return
        
        db = await self._get_connection()
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.sessions_table_name} (
                session_id TEXT PRIMARY KEY, user_id TEXT, agent_id TEXT,
                team_session_id TEXT, chat_history TEXT, summary TEXT,
                session_data TEXT, extra_data TEXT, created_at REAL, updated_at REAL
            )
        """)
        await db.commit()
        self._sessions_table_initialized = True
    
    async def _ensure_profiles_table(self) -> None:
        """Lazily creates the profiles table on first access."""
        if self._profiles_table_initialized:
            return
        
        db = await self._get_connection()
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.profiles_table_name} (
                user_id TEXT PRIMARY KEY, profile_data TEXT,
                created_at REAL, updated_at REAL
            )
        """)
        await db.commit()
        self._profiles_table_initialized = True

    def _get_table_info_for_model(self, model_type: Type[BaseModel]) -> Optional[tuple[str, str]]:
        """
        Get table name and primary key column for a model type.
        
        Args:
            model_type: Pydantic model class
            
        Returns:
            Tuple of (table_name, key_column) or None if not supported
        """
        if model_type is InteractionSession:
            return (self.sessions_table_name, "session_id")
        elif model_type is UserProfile:
            return (self.profiles_table_name, "user_id")
        else:
            # Generic support for arbitrary models
            # Use model class name as table name and "path" as key
            table_name = f"{model_type.__name__.lower()}_storage"
            # Determine primary key field - look for common patterns
            if hasattr(model_type, 'model_fields'):
                fields = model_type.model_fields
                # Try common ID field names
                for id_field in ['path', 'id', 'key', 'name']:
                    if id_field in fields:
                        return (table_name, id_field)
            # Fallback to generic "id"
            return (table_name, "id")
    
    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        table_info = self._get_table_info_for_model(model_type)
        if table_info is None:
            return None
        
        table, key_col = table_info
        
        # Ensure table exists - lazy initialization for built-in types
        if model_type is InteractionSession:
            await self._ensure_sessions_table()
        elif model_type is UserProfile:
            await self._ensure_profiles_table()
        else:
            # Generic models - create table if needed
            await self._ensure_table_for_model(model_type)

        db = await self._get_connection()
        sql = f"SELECT * FROM {table} WHERE {key_col} = ?"
        async with db.execute(sql, (object_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                data = dict(row)
                
                if model_type in [InteractionSession, UserProfile]:
                    # Handle known types with special fields
                    for key, value in data.items():
                        if key in ['chat_history', 'session_data', 'extra_data', 'profile_data'] and isinstance(value, str):
                            try:
                                data[key] = json.loads(value)
                            except:
                                pass
                    
                    if hasattr(model_type, 'from_dict'):
                        return model_type.from_dict(data)
                    else:
                        return model_type.model_validate(data)
                else:
                    # Generic model - data stored as JSON in 'data' column
                    if 'data' in data and isinstance(data['data'], str):
                        try:
                            obj_data = json.loads(data['data'])
                            return model_type.model_validate(obj_data)
                        except Exception:
                            return None
        return None

    async def _ensure_table_for_model(self, model_type: Type[BaseModel]) -> str:
        """
        Ensure table exists for a model type and return table name.
        
        For arbitrary models, creates a generic JSON storage table.
        
        Args:
            model_type: Pydantic model class
            
        Returns:
            Table name
        """
        table_info = self._get_table_info_for_model(model_type)
        if table_info is None:
            raise TypeError(f"Cannot determine table for model: {model_type.__name__}")
        
        table_name, key_col = table_info
        
        # For non-standard models, create generic JSON storage table
        if model_type not in [InteractionSession, UserProfile]:
            db = await self._get_connection()
            # Create generic table: key + data (JSON) + timestamps
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
        # Update timestamp if it exists
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()
        
        if isinstance(data, InteractionSession):
            # Ensure sessions table exists before upserting
            await self._ensure_sessions_table()
            
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
            # Ensure profiles table exists before upserting
            await self._ensure_profiles_table()
            
            table = self.profiles_table_name
            sql = f"""
                INSERT INTO {table} (user_id, profile_data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_data=excluded.profile_data, updated_at=excluded.updated_at
            """
            params = (data.user_id, json.dumps(data.profile_data), data.created_at, data.updated_at)
        else:
            # Generic model support - ensure table exists
            table_name = await self._ensure_table_for_model(type(data))
            table_info = self._get_table_info_for_model(type(data))
            _, key_col = table_info
            
            # Get the key value
            key_value = getattr(data, key_col)
            
            # Serialize entire model as JSON
            data_json = data.model_dump_json()
            
            # Get timestamps
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
        
        # Ensure table exists - lazy initialization for built-in types
        if model_type is InteractionSession:
            await self._ensure_sessions_table()
        elif model_type is UserProfile:
            await self._ensure_profiles_table()
        else:
            # For generic models, only delete if table exists
            # No need to create table just to delete from it
            pass

        db = await self._get_connection()
        sql = f"DELETE FROM {table} WHERE {key_col} = ?"
        await db.execute(sql, (object_id,))
        await db.commit()
    
    async def list_all_async(self, model_type: Type[T]) -> List[T]:
        """
        List all objects of a specific type from storage.
        
        Args:
            model_type: The Pydantic model class to query
            
        Returns:
            List of all objects of the specified type
        """
        table_info = self._get_table_info_for_model(model_type)
        if table_info is None:
            return []
        
        table_name, key_col = table_info
        
        # Ensure table exists - lazy initialization for built-in types
        if model_type is InteractionSession:
            await self._ensure_sessions_table()
        elif model_type is UserProfile:
            await self._ensure_profiles_table()
        else:
            # Generic models - create table if needed
            await self._ensure_table_for_model(model_type)
        
        db = await self._get_connection()
        
        # Check if table exists
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        ) as cursor:
            if not await cursor.fetchone():
                # Table doesn't exist yet
                return []
        
        # Query all rows
        sql = f"SELECT * FROM {table_name}"
        results = []
        
        async with db.execute(sql) as cursor:
            async for row in cursor:
                data = dict(row)
                
                if model_type in [InteractionSession, UserProfile]:
                    # Handle known types
                    for key, value in data.items():
                        if key in ['chat_history', 'session_data', 'extra_data', 'profile_data'] and isinstance(value, str):
                            try:
                                data[key] = json.loads(value)
                            except:
                                pass
                    
                    if hasattr(model_type, 'from_dict'):
                        obj = model_type.from_dict(data)
                    else:
                        obj = model_type.model_validate(data)
                else:
                    # Generic model - data is stored as JSON
                    if 'data' in data and isinstance(data['data'], str):
                        try:
                            obj_data = json.loads(data['data'])
                            obj = model_type.model_validate(obj_data)
                        except Exception:
                            continue
                    else:
                        # Try to parse the whole row
                        try:
                            obj = model_type.model_validate(data)
                        except Exception:
                            continue
                
                results.append(obj)
        
        return results

    async def drop_async(self) -> None:
        """
        Drops all tables managed by this storage provider.
        Only drops InteractionSession/UserProfile tables if they were actually created.
        """
        db = await self._get_connection()
        
        # Drop built-in tables (they may or may not exist)
        await db.execute(f"DROP TABLE IF EXISTS {self.sessions_table_name}")
        await db.execute(f"DROP TABLE IF EXISTS {self.profiles_table_name}")
        
        # Reset initialization flags
        self._sessions_table_initialized = False
        self._profiles_table_initialized = False
        
        await db.commit()
