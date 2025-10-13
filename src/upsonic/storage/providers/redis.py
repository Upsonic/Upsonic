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
    """

    def __init__(
        self,
        prefix: str,
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
            prefix: A prefix to namespace all keys for this application instance.
            host: The Redis server hostname.
            port: The Redis server port.
            db: The Redis database number to use.
            password: Optional password for Redis authentication.
            ssl: If True, uses an SSL connection.
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
        # Client is configured but not connected on initialization
        self.redis_client: Redis = Redis(
            host=host, port=port, db=db, password=password,
            ssl=ssl, decode_responses=True
        )

    def _get_key(self, object_id: str, model_type: Type[BaseModel]) -> str:
        if model_type is InteractionSession: return f"{self.prefix}:session:{object_id}"
        elif model_type is UserProfile: return f"{self.prefix}:profile:{object_id}"
        raise TypeError(f"Unsupported model type for key generation: {model_type.__name__}")
    
    def _serialize(self, data: Dict[str, Any]) -> str: return json.dumps(data)
    def _deserialize(self, data: str) -> Dict[str, Any]: return json.loads(data)



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
            return model_type.from_dict(data_dict)
        except (json.JSONDecodeError, TypeError) as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not parse key {key}. Error: {e}", "RedisStorage")
            return None

    async def upsert_async(self, data: Union[InteractionSession, UserProfile]) -> None:
        data.updated_at = time.time()
        data_dict = data.model_dump(mode="json")
        json_string = self._serialize(data_dict)

        if isinstance(data, InteractionSession): key = self._get_key(data.session_id, InteractionSession)
        elif isinstance(data, UserProfile): key = self._get_key(data.user_id, UserProfile)
        else: raise TypeError(f"Unsupported data type for upsert: {type(data).__name__}")
        
        await self.redis_client.set(key, json_string, ex=self.expire)

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        key = self._get_key(object_id, model_type)
        await self.redis_client.delete(key)

    async def drop_async(self) -> None:
        """Asynchronously deletes ALL keys associated with this provider's prefix."""
        keys_to_delete = [key async for key in self.redis_client.scan_iter(match=f"{self.prefix}:*")]
        if keys_to_delete:
            await self.redis_client.delete(*keys_to_delete)
