import asyncio
import time
from collections import OrderedDict
from typing import Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import (
    InteractionSession,
    UserProfile
)

T = TypeVar('T', bound=BaseModel)

class InMemoryStorage(Storage):
    """
    A hybrid sync/async, ephemeral, thread-safe storage provider that lives in memory.

    This provider implements both a synchronous and an asynchronous API. The
    synchronous methods are convenient wrappers that intelligently manage the
    event loop to run the core async logic.
    """

    def __init__(self, max_sessions: Optional[int] = None, max_profiles: Optional[int] = None):
        """
        Initializes the in-memory storage provider.

        Args:
            max_sessions: Max InteractionSessions to store. Acts as a fixed-size LRU cache.
            max_profiles: Max UserProfiles to store. Acts as a fixed-size LRU cache.
        """
        super().__init__()
        self.max_sessions = max_sessions
        self._sessions: Dict[str, InteractionSession] = OrderedDict() if self.max_sessions else {}
        self.max_profiles = max_profiles
        self._user_profiles: Dict[str, UserProfile] = OrderedDict() if self.max_profiles else {}
        # Generic storage for arbitrary Pydantic models
        # Structure: {model_type_name: {object_id: model_instance}}
        self._generic_storage: Dict[str, Dict[str, BaseModel]] = {}
        self._lock: Optional[asyncio.Lock] = None


    @property
    def lock(self) -> asyncio.Lock:
        """
        Lazily initializes and returns an asyncio.Lock, ensuring it is always
        bound to the currently running event loop.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(current_loop)
        
        if self._lock is None or self._lock._loop is not current_loop:
            self._lock = asyncio.Lock()
            
        return self._lock
    
    def _get_primary_key_field(self, model_type: Type[BaseModel]) -> str:
        """
        Determine the primary key field for a model type.
        
        Args:
            model_type: Pydantic model class
            
        Returns:
            Name of the primary key field
        """
        # For known types
        if model_type is InteractionSession:
            return "session_id"
        elif model_type is UserProfile:
            return "user_id"
        
        # For generic types, auto-detect
        if hasattr(model_type, 'model_fields'):
            for field_name in ['path', 'id', 'key', 'name']:
                if field_name in model_type.model_fields:
                    return field_name
        
        # Fallback
        return "id"
    
    def _get_storage_key(self, model_type: Type[BaseModel]) -> str:
        """
        Get storage key (model type name) for generic storage dict.
        
        Args:
            model_type: Pydantic model class
            
        Returns:
            Storage key for this model type
        """
        return model_type.__name__.lower()


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
        """Checks the internal connected flag."""
        return self._connected

    async def connect_async(self) -> None:
        """Marks the provider as connected. A no-op for in-memory."""
        self._connected = True

    async def disconnect_async(self) -> None:
        """Marks the provider as disconnected. A no-op for in-memory."""
        self._connected = False

    async def create_async(self) -> None:
        """For in-memory, there is no persistent schema to create. This is a no-op."""
        pass

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        """Asynchronously reads an object from the corresponding in-memory dictionary."""
        async with self.lock:
            if model_type is InteractionSession:
                item = self._sessions.get(object_id)
                if item:
                    if self.max_sessions:
                        self._sessions.move_to_end(object_id)
                    return item.model_copy(deep=True)
            elif model_type is UserProfile:
                item = self._user_profiles.get(object_id)
                if item:
                    if self.max_profiles:
                        self._user_profiles.move_to_end(object_id)
                    return item.model_copy(deep=True)
            else:
                # Generic model support
                storage_key = self._get_storage_key(model_type)
                if storage_key in self._generic_storage:
                    item = self._generic_storage[storage_key].get(object_id)
                    if item:
                        return item.model_copy(deep=True)
        return None

    async def upsert_async(self, data: BaseModel) -> None:
        """Asynchronously upserts an object into the corresponding in-memory dictionary."""
        async with self.lock:
            if hasattr(data, 'updated_at'):
                data.updated_at = time.time()
            
            data_copy = data.model_copy(deep=True)
            
            if isinstance(data, InteractionSession):
                self._sessions[data.session_id] = data_copy
                if self.max_sessions:
                    self._sessions.move_to_end(data.session_id)
                    if len(self._sessions) > self.max_sessions:
                        self._sessions.popitem(last=False)
            elif isinstance(data, UserProfile):
                self._user_profiles[data.user_id] = data_copy
                if self.max_profiles:
                    self._user_profiles.move_to_end(data.user_id)
                    if len(self._user_profiles) > self.max_profiles:
                        self._user_profiles.popitem(last=False)
            else:
                # Generic model support
                storage_key = self._get_storage_key(type(data))
                
                # Initialize storage for this model type if needed
                if storage_key not in self._generic_storage:
                    self._generic_storage[storage_key] = {}
                
                # Get primary key value
                primary_key_field = self._get_primary_key_field(type(data))
                object_id = getattr(data, primary_key_field)
                
                # Store
                self._generic_storage[storage_key][object_id] = data_copy
    
    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        """Asynchronously deletes an object from the corresponding in-memory dictionary."""
        async with self.lock:
            if model_type is InteractionSession and object_id in self._sessions:
                del self._sessions[object_id]
            elif model_type is UserProfile and object_id in self._user_profiles:
                del self._user_profiles[object_id]
            else:
                # Generic model support
                storage_key = self._get_storage_key(model_type)
                if storage_key in self._generic_storage:
                    if object_id in self._generic_storage[storage_key]:
                        del self._generic_storage[storage_key][object_id]

    async def list_all_async(self, model_type: Type[T]) -> list[T]:
        """List all objects of a specific type."""
        async with self.lock:
            if model_type is InteractionSession:
                return [session.model_copy(deep=True) for session in self._sessions.values()]
            elif model_type is UserProfile:
                return [profile.model_copy(deep=True) for profile in self._user_profiles.values()]
            else:
                # Generic model support
                storage_key = self._get_storage_key(model_type)
                if storage_key in self._generic_storage:
                    return [obj.model_copy(deep=True) for obj in self._generic_storage[storage_key].values()]
                return []

    async def drop_async(self) -> None:
        """Asynchronously clears all sessions and user profiles from memory."""
        async with self.lock:
            self._sessions.clear()
            self._user_profiles.clear()
            self._generic_storage.clear()