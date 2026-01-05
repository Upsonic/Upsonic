import asyncio
import copy
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Type, TypeVar, Union, TYPE_CHECKING

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.session.agent import AgentSession

if TYPE_CHECKING:
    from upsonic.culture.cultural_knowledge import CulturalKnowledge

T = TypeVar('T', bound=BaseModel)

class InMemoryStorage(Storage):
    """
    A hybrid sync/async, ephemeral, thread-safe storage provider that lives in memory.

    This provider implements both a synchronous and an asynchronous API. The
    synchronous methods are convenient wrappers that intelligently manage the
    event loop to run the core async logic.
    """

    def __init__(self, max_sessions: Optional[int] = None):
        """
        Initializes the in-memory storage provider.

        Args:
            max_sessions: Max AgentSessions to store. Acts as a fixed-size LRU cache.
        """
        super().__init__()
        self.max_sessions = max_sessions
        self._sessions: Dict[str, AgentSession] = OrderedDict() if self.max_sessions else {}
        self._generic_storage: Dict[str, Dict[str, BaseModel]] = {}
        self._cultural_knowledge: Dict[str, "CulturalKnowledge"] = {}
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
        """Determine the primary key field for a model type."""
        if model_type.__name__ == "AgentSession":
            return "session_id"
        
        if hasattr(model_type, 'model_fields'):
            for field_name in ['path', 'id', 'key', 'name']:
                if field_name in model_type.model_fields:
                    return field_name
        
        return "id"
    
    def _get_storage_key(self, model_type: Type[BaseModel]) -> str:
        """Get storage key (model type name) for generic storage dict."""
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

    def upsert(self, data: BaseModel) -> None:
        return self._run_async_from_sync(self.upsert_async(data))
    
    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None:
        return self._run_async_from_sync(self.delete_async(object_id, model_type))

    def drop(self) -> None:
        return self._run_async_from_sync(self.drop_async())



    async def is_connected_async(self) -> bool:
        return self._connected

    async def connect_async(self) -> None:
        self._connected = True

    async def disconnect_async(self) -> None:
        self._connected = False

    async def create_async(self) -> None:
        pass

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        async with self.lock:
            if model_type.__name__ == "AgentSession":
                item = self._sessions.get(object_id)
                if item:
                    if self.max_sessions:
                        self._sessions.move_to_end(object_id)
                    return copy.deepcopy(item)
            else:
                storage_key = self._get_storage_key(model_type)
                if storage_key in self._generic_storage:
                    item = self._generic_storage[storage_key].get(object_id)
                    if item:
                        return item.model_copy(deep=True)
        return None

    async def upsert_async(self, data: BaseModel) -> None:
        async with self.lock:
            if hasattr(data, 'updated_at'):
                data.updated_at = time.time()
            
            if type(data).__name__ == "AgentSession":
                data_copy = copy.deepcopy(data)
                self._sessions[data.session_id] = data_copy
                if self.max_sessions:
                    self._sessions.move_to_end(data.session_id)
                    if len(self._sessions) > self.max_sessions:
                        self._sessions.popitem(last=False)
            else:
                data_copy = data.model_copy(deep=True)
                storage_key = self._get_storage_key(type(data))
                
                if storage_key not in self._generic_storage:
                    self._generic_storage[storage_key] = {}
                
                primary_key_field = self._get_primary_key_field(type(data))
                object_id = getattr(data, primary_key_field)
                
                self._generic_storage[storage_key][object_id] = data_copy
    
    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        async with self.lock:
            if model_type.__name__ == "AgentSession" and object_id in self._sessions:
                del self._sessions[object_id]
            else:
                storage_key = self._get_storage_key(model_type)
                if storage_key in self._generic_storage:
                    if object_id in self._generic_storage[storage_key]:
                        del self._generic_storage[storage_key][object_id]

    async def list_all_async(self, model_type: Type[T]) -> list[T]:
        async with self.lock:
            if model_type.__name__ == "AgentSession":
                return [copy.deepcopy(session) for session in self._sessions.values()]
            else:
                storage_key = self._get_storage_key(model_type)
                if storage_key in self._generic_storage:
                    return [obj.model_copy(deep=True) for obj in self._generic_storage[storage_key].values()]
                return []

    async def drop_async(self) -> None:
        async with self.lock:
            self._sessions.clear()
            self._generic_storage.clear()
            self._cultural_knowledge.clear()

    # =========================================================================
    # Cultural Knowledge Methods
    # =========================================================================

    async def read_cultural_knowledge_async(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        async with self.lock:
            knowledge = self._cultural_knowledge.get(knowledge_id)
            if knowledge:
                return copy.deepcopy(knowledge)
            return None

    async def upsert_cultural_knowledge_async(self, knowledge: "CulturalKnowledge") -> None:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        async with self.lock:
            knowledge.bump_updated_at()
            self._cultural_knowledge[knowledge.id] = copy.deepcopy(knowledge)

    async def delete_cultural_knowledge_async(self, knowledge_id: str) -> None:
        async with self.lock:
            if knowledge_id in self._cultural_knowledge:
                del self._cultural_knowledge[knowledge_id]

    async def list_all_cultural_knowledge_async(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        async with self.lock:
            results = []
            for knowledge in self._cultural_knowledge.values():
                if name is not None:
                    if knowledge.name is None:
                        continue
                    if name.lower() not in knowledge.name.lower():
                        continue
                results.append(copy.deepcopy(knowledge))
            return results

    async def clear_cultural_knowledge_async(self) -> None:
        async with self.lock:
            self._cultural_knowledge.clear()
