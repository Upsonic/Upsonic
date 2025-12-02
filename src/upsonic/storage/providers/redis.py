from __future__ import annotations

import json
import time
from typing import Optional, Dict, Any, Type, Union, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import InteractionSession, UserProfile

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError

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
    - Uses key prefixes to organize data (sessions, profiles, generic models)
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
        
        # Store client and track ownership for lifecycle management
        self._owns_client = (redis_client is None)  # True if we create it, False if user provided
        
        if redis_client:
            self.redis_client: Redis = redis_client
        else:
            # Create our own client (configured but not connected on initialization)
            self.redis_client: Redis = Redis(
                host=host, port=port, db=db, password=password,
                ssl=ssl, decode_responses=True
            )

    def _get_primary_key_field(self, model_type: Type[BaseModel]) -> str:
        """Determine the primary key field for a model type."""
        if model_type is InteractionSession:
            return "session_id"
        elif model_type is UserProfile:
            return "user_id"
        
        # Auto-detect for generic types
        if hasattr(model_type, 'model_fields'):
            for field_name in ['path', 'id', 'key', 'name']:
                if field_name in model_type.model_fields:
                    return field_name
        return "id"
    
    def _get_key(self, object_id: str, model_type: Type[BaseModel]) -> str:
        if model_type is InteractionSession:
            return f"{self.prefix}:session:{object_id}"
        elif model_type is UserProfile:
            return f"{self.prefix}:profile:{object_id}"
        else:
            # Generic model: prefix:model:model_type_name:object_id
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
    def upsert(self, data: Union[InteractionSession, UserProfile]) -> None: return self._run_async_from_sync(self.upsert_async(data))
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
        """
        Closes the Redis client connection.
        If user provided the client, this is a no-op (user manages lifecycle).
        """
        if not self._owns_client:
            # User manages their own client lifecycle
            return
        
        await self.redis_client.close()
        self._connected = False

    async def create_async(self) -> None:
        await self.connect_async()

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        key = self._get_key(object_id, model_type)
        data_str = await self.redis_client.get(key)
        if data_str is None:
            return None
        try:
            data_dict = self._deserialize(data_str)
            
            # Try from_dict first, fallback to model_validate
            if hasattr(model_type, 'from_dict'):
                return model_type.from_dict(data_dict)
            else:
                return model_type.model_validate(data_dict)
        except (json.JSONDecodeError, TypeError) as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not parse key {key}. Error: {e}", "RedisStorage")
            return None

    async def upsert_async(self, data: BaseModel) -> None:
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()
        
        data_dict = data.model_dump(mode="json")
        json_string = self._serialize(data_dict)

        if isinstance(data, InteractionSession):
            key = self._get_key(data.session_id, InteractionSession)
        elif isinstance(data, UserProfile):
            key = self._get_key(data.user_id, UserProfile)
        else:
            # Generic model support
            primary_key_field = self._get_primary_key_field(type(data))
            object_id = getattr(data, primary_key_field)
            key = self._get_key(object_id, type(data))
        
        await self.redis_client.set(key, json_string, ex=self.expire)

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        key = self._get_key(object_id, model_type)
        await self.redis_client.delete(key)
    
    async def list_all_async(self, model_type: Type[T]) -> list[T]:
        """List all objects of a specific type."""
        if model_type is InteractionSession:
            pattern = f"{self.prefix}:session:*"
        elif model_type is UserProfile:
            pattern = f"{self.prefix}:profile:*"
        else:
            # Generic model
            model_name = model_type.__name__.lower()
            pattern = f"{self.prefix}:model:{model_name}:*"
        
        results = []
        
        # Scan for all matching keys
        async for key in self.redis_client.scan_iter(match=pattern):
            try:
                data_str = await self.redis_client.get(key)
                if data_str:
                    data_dict = self._deserialize(data_str)
                    
                    if hasattr(model_type, 'from_dict'):
                        obj = model_type.from_dict(data_dict)
                    else:
                        obj = model_type.model_validate(data_dict)
                    
                    results.append(obj)
            except Exception:
                continue
        
        return results

    async def drop_async(self) -> None:
        """Asynchronously deletes ALL keys associated with this provider's prefix."""
        keys_to_delete = [key async for key in self.redis_client.scan_iter(match=f"{self.prefix}:*")]
        if keys_to_delete:
            await self.redis_client.delete(*keys_to_delete)
