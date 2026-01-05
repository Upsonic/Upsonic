from __future__ import annotations

import json
import time
from typing import List, Optional, Dict, Any, Type, Union, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.session.agent import AgentSession

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from upsonic.culture.cultural_knowledge import CulturalKnowledge

try:
    from redis.asyncio import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    _REDIS_AVAILABLE = True
except ImportError:
    Redis = None  # type: ignore
    RedisConnectionError = None  # type: ignore
    _REDIS_AVAILABLE = False


T = TypeVar('T', bound=BaseModel)

class RedisStorage(Storage):
    """
    A hybrid sync/async, high-performance storage provider using Redis and
    its native async client, with proper connection lifecycle management.
    
    This storage provider is designed to be flexible and dynamic:
    - Can accept a pre-existing Redis client or create one from connection details
    - Uses key prefixes to organize data (sessions, generic models)
    - Supports generic Pydantic models for custom storage needs
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    def __init__(
        self,
        redis_client: Optional['Redis'] = None,
        prefix: str = "upsonic",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        expire: Optional[int] = None,
    ):
        """
        Initializes the async Redis storage provider.

        Args:
            redis_client: Optional pre-existing Redis client. If provided, this client
                will be used instead of creating a new one. User is responsible for
                client lifecycle management when providing their own client.
            prefix: A prefix to namespace all keys for this application instance.
                Defaults to "upsonic".
            host: The Redis server hostname. Ignored if redis_client is provided.
            port: The Redis server port. Ignored if redis_client is provided.
            db: The Redis database number to use. Ignored if redis_client is provided.
            password: Optional password for Redis authentication. Ignored if redis_client is provided.
            ssl: If True, uses an SSL connection. Ignored if redis_client is provided.
            expire: Optional TTL in seconds for all created keys.
        """
        if not _REDIS_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="redis",
                install_command='pip install "upsonic[storage]"',
                feature_name="Redis storage provider"
            )

        super().__init__()
        self.prefix = prefix
        self.expire = expire
        
        self._owns_client = (redis_client is None)
        
        if redis_client:
            self.redis_client: Redis = redis_client
        else:
            self.redis_client: Redis = Redis(
                host=host, port=port, db=db, password=password,
                ssl=ssl, decode_responses=True
            )

    def _get_primary_key_field(self, model_type: Type[BaseModel]) -> str:
        if model_type.__name__ == "AgentSession":
            return "session_id"
        
        if hasattr(model_type, 'model_fields'):
            for field_name in ['path', 'id', 'key', 'name']:
                if field_name in model_type.model_fields:
                    return field_name
        return "id"
    
    def _get_key(self, object_id: str, model_type: Type[BaseModel]) -> str:
        if model_type.__name__ == "AgentSession":
            return f"{self.prefix}:session:{object_id}"
        else:
            model_name = model_type.__name__.lower()
            return f"{self.prefix}:model:{model_name}:{object_id}"
    
    def _serialize(self, data: Dict[str, Any]) -> str:
        return json.dumps(data)
    
    def _deserialize(self, data: str) -> Dict[str, Any]:
        return json.loads(data)



    def is_connected(self) -> bool: return self._run_async_from_sync(self.is_connected_async())
    def connect(self) -> None: return self._run_async_from_sync(self.connect_async())
    def disconnect(self) -> None: return self._run_async_from_sync(self.disconnect_async())
    def create(self) -> None: return self._run_async_from_sync(self.create_async())
    def read(self, object_id: str, model_type: Type[T]) -> Optional[T]: return self._run_async_from_sync(self.read_async(object_id, model_type))
    def upsert(self, data: BaseModel) -> None: return self._run_async_from_sync(self.upsert_async(data))
    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None: return self._run_async_from_sync(self.delete_async(object_id, model_type))
    def drop(self) -> None: return self._run_async_from_sync(self.drop_async())
    


    async def is_connected_async(self) -> bool:
        if not self._connected:
            return False
        try:
            await self.redis_client.ping()
            return True
        except (RedisConnectionError, ConnectionRefusedError):
            self._connected = False
            return False

    async def connect_async(self) -> None:
        if self._connected and await self.is_connected_async():
            return
        try:
            await self.redis_client.ping()
            self._connected = True
        except (RedisConnectionError, ConnectionRefusedError) as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    async def disconnect_async(self) -> None:
        if not self._owns_client:
            return
        
        await self.redis_client.close()
    
    async def _ensure_connection(self) -> None:
        """Ensure connection is valid, reconnect if needed."""
        if not await self.is_connected_async():
            # If we own the client and it's disconnected, recreate it
            if self._owns_client:
                # Get connection details from old client
                connection_kwargs = {}
                if hasattr(self.redis_client, 'connection_pool'):
                    pool = self.redis_client.connection_pool
                    if hasattr(pool, 'connection_kwargs'):
                        connection_kwargs = pool.connection_kwargs.copy()
                
                # Close old client if possible
                try:
                    await self.redis_client.close()
                except Exception:
                    pass
                
                # Recreate client
                if connection_kwargs:
                    self.redis_client = Redis(**connection_kwargs)
                else:
                    # Fallback - just try to reconnect
                    pass
            
            await self.connect_async()

    async def create_async(self) -> None:
        await self._ensure_connection()

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        import base64
        await self._ensure_connection()
        key = self._get_key(object_id, model_type)
        data_str = await self.redis_client.get(key)
        if data_str is None:
            return None
        try:
            if model_type.__name__ == "AgentSession":
                # Use deserialize: base64 decode then deserialize
                data_dict = self._deserialize(data_str)
                if "data" in data_dict:
                    serialized_bytes = base64.b64decode(data_dict["data"].encode('utf-8'))
                    return model_type.deserialize(serialized_bytes)
                else:
                    # Fallback to old format
                    if hasattr(model_type, 'from_dict'):
                        return model_type.from_dict(data_dict)
                    return model_type.model_validate(data_dict)
            else:
                data_dict = self._deserialize(data_str)
                if hasattr(model_type, 'from_dict'):
                    return model_type.from_dict(data_dict)
                else:
                    return model_type.model_validate(data_dict)
        except (json.JSONDecodeError, TypeError) as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not parse key {key}. Error: {e}", "RedisStorage")
            return None

    async def upsert_async(self, data: BaseModel) -> None:
        import base64
        await self._ensure_connection()
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()
        
        if type(data).__name__ == "AgentSession":
            # Use serialize: serialize to bytes, then base64 encode
            serialized_bytes = data.serialize()
            serialized_str = base64.b64encode(serialized_bytes).decode('utf-8')
            data_dict = {
                "data": serialized_str,
                "session_id": data.session_id,
                "created_at": data.created_at or time.time(),
                "updated_at": data.updated_at or time.time()
            }
            key = self._get_key(data.session_id, type(data))
        else:
            data_dict = data.model_dump(mode="json")
            primary_key_field = self._get_primary_key_field(type(data))
            object_id = getattr(data, primary_key_field)
            key = self._get_key(object_id, type(data))
        
        json_string = self._serialize(data_dict)
        await self.redis_client.set(key, json_string, ex=self.expire)

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        await self._ensure_connection()
        key = self._get_key(object_id, model_type)
        await self.redis_client.delete(key)
    
    async def list_all_async(self, model_type: Type[T]) -> list[T]:
        import base64
        await self._ensure_connection()
        if model_type.__name__ == "AgentSession":
            pattern = f"{self.prefix}:session:*"
        else:
            model_name = model_type.__name__.lower()
            pattern = f"{self.prefix}:model:{model_name}:*"
        
        results = []
        
        async for key in self.redis_client.scan_iter(match=pattern):
            try:
                data_str = await self.redis_client.get(key)
                if data_str:
                    data_dict = self._deserialize(data_str)
                    
                    if model_type.__name__ == "AgentSession" and "data" in data_dict:
                        # Use deserialize: base64 decode then deserialize
                        serialized_bytes = base64.b64decode(data_dict["data"].encode('utf-8'))
                        obj = model_type.deserialize(serialized_bytes)
                    else:
                        if hasattr(model_type, 'from_dict'):
                            obj = model_type.from_dict(data_dict)
                        else:
                            obj = model_type.model_validate(data_dict)
                    
                    results.append(obj)
            except Exception:
                continue
        
        return results

    async def drop_async(self) -> None:
        keys_to_delete = [key async for key in self.redis_client.scan_iter(match=f"{self.prefix}:*")]
        if keys_to_delete:
            await self.redis_client.delete(*keys_to_delete)

    # =========================================================================
    # Cultural Knowledge Methods
    # =========================================================================

    def _get_culture_key(self, knowledge_id: str) -> str:
        return f"{self.prefix}:culture:{knowledge_id}"

    async def read_cultural_knowledge_async(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        key = self._get_culture_key(knowledge_id)
        data_str = await self.redis_client.get(key)
        
        if data_str is None:
            return None
        
        try:
            data_dict = self._deserialize(data_str)
            return CulturalKnowledge.from_dict(data_dict)
        except (json.JSONDecodeError, TypeError) as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not parse cultural knowledge key {key}. Error: {e}", "RedisStorage")
            return None

    async def upsert_cultural_knowledge_async(self, knowledge: "CulturalKnowledge") -> None:
        knowledge.bump_updated_at()
        
        data_dict = knowledge.to_dict()
        json_string = self._serialize(data_dict)
        
        key = self._get_culture_key(knowledge.id)
        await self.redis_client.set(key, json_string, ex=self.expire)

    async def delete_cultural_knowledge_async(self, knowledge_id: str) -> None:
        key = self._get_culture_key(knowledge_id)
        await self.redis_client.delete(key)

    async def list_all_cultural_knowledge_async(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        pattern = f"{self.prefix}:culture:*"
        results = []
        
        async for key in self.redis_client.scan_iter(match=pattern):
            try:
                data_str = await self.redis_client.get(key)
                if data_str:
                    data_dict = self._deserialize(data_str)
                    knowledge = CulturalKnowledge.from_dict(data_dict)
                    
                    if name is not None:
                        if knowledge.name is None:
                            continue
                        if name.lower() not in knowledge.name.lower():
                            continue
                    
                    results.append(knowledge)
            except Exception:
                continue
        
        return results

    async def clear_cultural_knowledge_async(self) -> None:
        pattern = f"{self.prefix}:culture:*"
        keys_to_delete = [key async for key in self.redis_client.scan_iter(match=pattern)]
        if keys_to_delete:
            await self.redis_client.delete(*keys_to_delete)
