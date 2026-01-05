from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Type, TypeVar, Union, overload, Any, TYPE_CHECKING

from pydantic import BaseModel

from upsonic.utils.async_utils import AsyncExecutionMixin
from upsonic.session.agent import AgentSession

if TYPE_CHECKING:
    from upsonic.culture.cultural_knowledge import CulturalKnowledge

T = TypeVar('T', bound=BaseModel)



class Storage(AsyncExecutionMixin, ABC):
    """
    The "Contract" for a hybrid sync/async, unified, type-driven Memory Archive.

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
    
    2. **Lazy Initialization**: AgentSession tables/collections are only created 
       when first accessed, not during initialization. This allows storages to be 
       used for generic purposes without creating unused infrastructure.
    
    3. **Generic Model Support**: All providers support arbitrary Pydantic models,
       not just AgentSession. This makes them truly general-purpose.
    
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
    def read(self, object_id: str, model_type: Type[AgentSession]) -> Optional[AgentSession]: ...
    @overload
    def read(self, object_id: str, model_type: Type[T]) -> Optional[T]: ...
    @abstractmethod
    def read(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        raise NotImplementedError

    @overload
    def upsert(self, data: AgentSession) -> None: ...
    @overload
    def upsert(self, data: BaseModel) -> None: ...
    @abstractmethod
    def upsert(self, data: BaseModel) -> None:
        raise NotImplementedError

    @overload
    def delete(self, object_id: str, model_type: Type[AgentSession]) -> None: ...
    @overload
    def delete(self, object_id: str, model_type: Type[T]) -> None: ...
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
    async def read_async(self, object_id: str, model_type: Type[AgentSession]) -> Optional[AgentSession]: ...
    @overload
    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]: ...
    @abstractmethod
    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        raise NotImplementedError

    @overload
    async def upsert_async(self, data: AgentSession) -> None: ...
    @overload
    async def upsert_async(self, data: BaseModel) -> None: ...
    @abstractmethod
    async def upsert_async(self, data: BaseModel) -> None:
        raise NotImplementedError

    @overload
    async def delete_async(self, object_id: str, model_type: Type[AgentSession]) -> None: ...
    @overload
    async def delete_async(self, object_id: str, model_type: Type[T]) -> None: ...
    @abstractmethod
    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def drop_async(self) -> None:
        raise NotImplementedError
    
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
        return []
    
    def list_all(self, model_type: Type[T]) -> List[T]:
        """Synchronous wrapper for list_all_async."""
        return self._run_async_from_sync(self.list_all_async(model_type))


    async def read_cultural_knowledge_async(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        """
        Read a cultural knowledge entry by ID.
        
        Args:
            knowledge_id: The unique identifier of the cultural knowledge
            
        Returns:
            The CulturalKnowledge instance if found, None otherwise
            
        Note: Default implementation returns None. Providers should override.
        """
        return None
    
    def read_cultural_knowledge(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        """Synchronous wrapper for read_cultural_knowledge_async."""
        return self._run_async_from_sync(self.read_cultural_knowledge_async(knowledge_id))

    async def upsert_cultural_knowledge_async(self, knowledge: "CulturalKnowledge") -> None:
        """
        Insert or update a cultural knowledge entry.
        
        If a knowledge entry with the same ID exists, it will be updated.
        Otherwise, a new entry will be created.
        
        Args:
            knowledge: The CulturalKnowledge instance to upsert
            
        Note: Default implementation does nothing. Providers should override.
        """
        pass
    
    def upsert_cultural_knowledge(self, knowledge: "CulturalKnowledge") -> None:
        """Synchronous wrapper for upsert_cultural_knowledge_async."""
        return self._run_async_from_sync(self.upsert_cultural_knowledge_async(knowledge))

    async def delete_cultural_knowledge_async(self, knowledge_id: str) -> None:
        """
        Delete a cultural knowledge entry by ID.
        
        Args:
            knowledge_id: The unique identifier of the cultural knowledge to delete
            
        Note: Default implementation does nothing. Providers should override.
        """
        pass
    
    def delete_cultural_knowledge(self, knowledge_id: str) -> None:
        """Synchronous wrapper for delete_cultural_knowledge_async."""
        return self._run_async_from_sync(self.delete_cultural_knowledge_async(knowledge_id))

    async def list_all_cultural_knowledge_async(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        """
        List all cultural knowledge entries.
        
        Args:
            name: Optional filter by name (case-insensitive partial match)
            
        Returns:
            List of all CulturalKnowledge instances, optionally filtered by name
            
        Note: Default implementation returns empty list. Providers should override.
        """
        return []
    
    def list_all_cultural_knowledge(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        """Synchronous wrapper for list_all_cultural_knowledge_async."""
        return self._run_async_from_sync(self.list_all_cultural_knowledge_async(name))

    async def clear_cultural_knowledge_async(self) -> None:
        """
        Delete all cultural knowledge entries.
        
        Warning: This is a destructive operation. Use with caution.
        
        Note: Default implementation does nothing. Providers should override.
        """
        pass
    
    def clear_cultural_knowledge(self) -> None:
        """Synchronous wrapper for clear_cultural_knowledge_async."""
        return self._run_async_from_sync(self.clear_cultural_knowledge_async())
    
    # ========================================================================
    # AgentSession convenience methods (concise API)
    # ========================================================================
    
    async def read_agent_session_async(
        self, session_id: str, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Optional[AgentSession]:
        """Read an agent session by session_id."""
        return await self.read_async(session_id, AgentSession)
    
    def read_agent_session(self, session_id: str, **kwargs) -> Optional[AgentSession]:
        return self._run_async_from_sync(self.read_agent_session_async(session_id, **kwargs))
    
    async def upsert_agent_session_async(self, session: AgentSession) -> None:
        """Insert or update an agent session (auto-sets updated_at)."""
        import time
        session.updated_at = time.time()
        await self.upsert_async(session)
    
    def upsert_agent_session(self, session: AgentSession) -> None:
        self._run_async_from_sync(self.upsert_agent_session_async(session))
    
    async def delete_agent_session_async(self, session_id: str) -> None:
        """Delete an agent session by session_id."""
        await self.delete_async(session_id, AgentSession)
    
    def delete_agent_session(self, session_id: str) -> None:
        self._run_async_from_sync(self.delete_agent_session_async(session_id))
    
    async def list_agent_sessions_async(
        self, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[AgentSession]:
        """List sessions, optionally filtered by agent_id/user_id."""
        sessions = await self.list_all_async(AgentSession)
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        return sessions
    
    def list_agent_sessions(self, **kwargs) -> List[AgentSession]:
        return self._run_async_from_sync(self.list_agent_sessions_async(**kwargs))
    
    async def find_agent_session_async(
        self, session_id: Optional[str] = None, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Optional[AgentSession]:
        """Find session by session_id (priority) or agent_id/user_id filter."""
        if session_id:
            return await self.read_async(session_id, AgentSession)
        sessions = await self.list_agent_sessions_async(agent_id, user_id)
        return sessions[0] if sessions else None
    
    def find_agent_session(self, **kwargs) -> Optional[AgentSession]:
        return self._run_async_from_sync(self.find_agent_session_async(**kwargs))
    
    async def clear_agent_sessions_async(
        self, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> int:
        """Delete all sessions matching criteria. Returns count deleted."""
        sessions = await self.list_agent_sessions_async(agent_id, user_id)
        for s in sessions:
            await self.delete_async(s.session_id, AgentSession)
        return len(sessions)
    
    def clear_agent_sessions(self, **kwargs) -> int:
        return self._run_async_from_sync(self.clear_agent_sessions_async(**kwargs))