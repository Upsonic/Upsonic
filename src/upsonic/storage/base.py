from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Type, TypeVar, Union, overload, Any

from pydantic import BaseModel, Field

from upsonic.utils.async_utils import AsyncExecutionMixin
from upsonic.storage.session.sessions import (
    InteractionSession,
    UserProfile
)
from upsonic.storage.types import SessionId, UserId

T = TypeVar('T', bound=BaseModel)



class Storage(AsyncExecutionMixin, ABC):
    """
    The "Contract" for a hybrid sync/async, unified, type-driven Memory and
    Profile Archive.

    This ABC defines two sets of methods:
    1. A clean, synchronous API (`connect`, `read`, `upsert`, etc.) for
       ease of use in standard Python scripts and applications.
    2. A high-performance, purely asynchronous API with an `_async` suffix
       (`connect_async`, `read_async`, etc.) for use in asyncio-native
       applications.

    Concrete implementations must provide the logic for the `_async` methods.
    The synchronous methods can then be implemented as simple wrappers.
    
    Storage Provider Design Principles:
    ------------------------------------
    Storage providers are designed to be flexible and dynamic:
    
    1. **Bring Your Own Client**: Providers can accept pre-existing database 
       clients/connections, allowing users to integrate with their existing 
       infrastructure. When a client is provided, the user manages its lifecycle.
    
    2. **Lazy Initialization**: InteractionSession and UserProfile tables/collections
       are only created when first accessed, not during initialization. This allows
       storages to be used for generic purposes without creating unused infrastructure.
    
    3. **Generic Model Support**: All providers support arbitrary Pydantic models,
       not just InteractionSession and UserProfile. This makes them truly general-purpose.
    
    4. **Dual Purpose**: Providers can be used for both custom storage needs AND
       built-in chat/profile features simultaneously in the same database/connection.
    """

    def __init__(self):
        """Initializes the storage provider's state."""
        self._connected = False



    @abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @overload
    def read(self, object_id: SessionId, model_type: Type[InteractionSession]) -> Optional[InteractionSession]: ...
    @overload
    def read(self, object_id: UserId, model_type: Type[UserProfile]) -> Optional[UserProfile]: ...
    @abstractmethod
    def read(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        raise NotImplementedError

    @overload
    def upsert(self, data: InteractionSession) -> None: ...
    @overload
    def upsert(self, data: UserProfile) -> None: ...
    @overload
    def upsert(self, data: BaseModel) -> None: ...
    @abstractmethod
    def upsert(self, data: BaseModel) -> None:
        raise NotImplementedError

    @overload
    def delete(self, object_id: SessionId, model_type: Type[InteractionSession]) -> None: ...
    @overload
    def delete(self, object_id: UserId, model_type: Type[UserProfile]) -> None: ...
    @abstractmethod
    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop(self) -> None:
        raise NotImplementedError



    @abstractmethod
    async def is_connected_async(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def connect_async(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def disconnect_async(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def create_async(self) -> None:
        raise NotImplementedError

    @overload
    async def read_async(self, object_id: SessionId, model_type: Type[InteractionSession]) -> Optional[InteractionSession]: ...
    @overload
    async def read_async(self, object_id: UserId, model_type: Type[UserProfile]) -> Optional[UserProfile]: ...
    @abstractmethod
    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        raise NotImplementedError

    @overload
    async def upsert_async(self, data: InteractionSession) -> None: ...
    @overload
    async def upsert_async(self, data: UserProfile) -> None: ...
    @overload
    async def upsert_async(self, data: BaseModel) -> None: ...
    @abstractmethod
    async def upsert_async(self, data: BaseModel) -> None:
        raise NotImplementedError

    @overload
    async def delete_async(self, object_id: SessionId, model_type: Type[InteractionSession]) -> None: ...
    @overload
    async def delete_async(self, object_id: UserId, model_type: Type[UserProfile]) -> None: ...
    @abstractmethod
    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def drop_async(self) -> None:
        raise NotImplementedError
    
    # Generic query methods for arbitrary Pydantic models
    async def list_all_async(self, model_type: Type[T]) -> List[T]:
        """
        List all objects of a specific type.
        
        This method enables querying all instances of any Pydantic model type.
        Useful for FilesystemEntry, custom models, etc.
        
        Args:
            model_type: The Pydantic model class to query
            
        Returns:
            List of all objects of the specified type
            
        Note: Default implementation returns empty list.
              Storage providers should override for full functionality.
        """
        # Default implementation (for backward compatibility)
        # Providers can override for actual querying
        return []
    
    def list_all(self, model_type: Type[T]) -> List[T]:
        """Synchronous wrapper for list_all_async."""
        return self._run_async_from_sync(self.list_all_async(model_type))