import json
import time
from typing import List, Optional, Dict, Any, Literal

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import (
    BaseSession,
    AgentSession
)

try:
    from redis import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
except ImportError:
    raise ImportError("`redis` is required for RedisStorage. Please install it using `pip install redis`.")


class RedisStorage(Storage):
    """
    A session-based storage provider using Redis.

    This provider implements the Storage contract by storing each session as a
    single JSON string in a Redis key. It is designed for high performance on
    reads and writes but performs filtering operations on the client side.
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
        mode: Literal["agent", "team", "workflow", "workflow_v2"] = "agent",
    ):
        """
        Initializes the Redis storage provider.

        Args:
            prefix: A prefix to namespace all keys for this application instance.
            host: The Redis server hostname.
            port: The Redis server port.
            db: The Redis database number to use.
            password: Optional password for Redis authentication.
            ssl: If True, uses an SSL connection.
            expire: Optional TTL in seconds for all session keys.
            mode: The operational mode.
        """
        super().__init__(mode=mode)

        self.prefix = prefix
        self.expire = expire
        self.redis_client: Redis = Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            ssl=ssl,
            decode_responses=True 
        )

        self.connect()

    def _get_key(self, session_id: str) -> str:
        """Generates the final Redis key for a given session ID."""
        return f"{self.prefix}:{self.mode}:session:{session_id}"

    def _serialize(self, data: Dict[str, Any]) -> str:
        """Serializes a dictionary to a JSON string."""
        return json.dumps(data)

    def _deserialize(self, data: str) -> Dict[str, Any]:
        """Deserializes a JSON string to a dictionary."""
        return json.loads(data)

    def _get_session_model_class(self) -> type[BaseSession]:
        """Returns the appropriate Pydantic session model class based on the current mode."""
        if self.mode == "agent":
            return AgentSession
        raise ValueError(f"Invalid mode '{self.mode}' specified for RedisStorage.")


    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self.is_connected():
            return

        try:
            self.redis_client.ping()
            self._connected = True
        except RedisConnectionError as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    def disconnect(self) -> None:
        if not self.is_connected():
            return

        try:
            self.redis_client.close()
        except Exception:
            pass
        self._connected = False

    def create(self) -> None:
        """For Redis, this is a no-op but we can use it to verify connection."""
        self.connect()

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[BaseSession]:
        key = self._get_key(session_id)
        data = self.redis_client.get(key)
        if data is None:
            return None

        session_data = self._deserialize(data)
        if user_id and session_data.get("user_id") != user_id:
            return None

        SessionModel = self._get_session_model_class()
        return SessionModel.from_dict(session_data)

    def upsert(self, session: BaseSession) -> Optional[BaseSession]:
        SessionModel = self._get_session_model_class()
        if not isinstance(session, SessionModel):
            raise TypeError(f"Session object must be of type {SessionModel.__name__} for mode '{self.mode}'")

        session.updated_at = int(time.time())
        session_dict = session.model_dump(mode="json")
        json_string = self._serialize(session_dict)
        key = self._get_key(session.session_id)

        try:
            self.redis_client.set(key, json_string, ex=self.expire)
            return session
        except Exception as e:
            raise IOError(f"Failed to upsert session {session.session_id} to Redis: {e}")

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[BaseSession]:
        """
        Retrieves all sessions, with inefficient client-side filtering.
        WARNING: This operation can be slow on large datasets as it scans all keys.
        """
        sessions: List[BaseSession] = []
        SessionModel = self._get_session_model_class()
        entity_key = f"{self.mode}_id" if self.mode else None

        for key in self.redis_client.scan_iter(match=f"{self.prefix}:{self.mode}:session:*"):
            data = self.redis_client.get(key)
            if not data:
                continue

            session_data = self._deserialize(data)

            # Client-side filtering
            if user_id and session_data.get("user_id") != user_id:
                continue
            if entity_id and entity_key and session_data.get(entity_key) != entity_id:
                continue

            sessions.append(SessionModel.from_dict(session_data))
        return sessions

    def get_recent_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 10) -> List[BaseSession]:
        """
        Retrieves recent sessions by fetching all, then sorting in memory.
        WARNING: Highly inefficient. Not recommended for production use on large datasets.
        Consider using a different indexing strategy (e.g., Sorted Sets) for this functionality.
        """
        all_sessions = self.get_all_sessions(user_id=user_id, entity_id=entity_id)
        # Sort in memory by the 'updated_at' timestamp, descending
        all_sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return all_sessions[:limit]

    def delete_session(self, session_id: str) -> None:
        key = self._get_key(session_id)
        self.redis_client.delete(key)

    def drop(self) -> None:
        """Deletes ALL session keys associated with this provider's prefix and mode."""
        keys_to_delete = list(self.redis_client.scan_iter(match=f"{self.prefix}:{self.mode}:session:*"))
        if keys_to_delete:
            self.redis_client.delete(*keys_to_delete)

    def log_artifact(self, artifact) -> None:
        raise NotImplementedError("Artifact logging should be handled within the session data for this provider.")

    def store_artifact_data(self, artifact_id: str, session_id: str, binary_data: bytes) -> str:
        raise NotImplementedError("Binary artifact storage is not handled by the session provider.")

    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        raise NotImplementedError("Binary artifact storage is not handled by the session provider.")