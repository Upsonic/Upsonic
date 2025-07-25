import json
import time
from typing import List, Optional

try:
    from redis import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
except ImportError:
    raise ImportError("`redis` not installed. Please install it using `pip install redis`")

from upsonic.storage.base import Storage
from upsonic.storage.settings import RedisSettings
from upsonic.storage.session.llm import LLMConversation, LLMTurn, Artifact


class RedisStorage(Storage):
    """
    High-performance storage provider for the LLM Interaction Archive using Redis.
    - Conversation metadata is stored in Hashes.
    - Turns are stored in append-only Redis Streams for maximum performance.
    """
    def __init__(self, settings: RedisSettings):
        super().__init__()
        self._set_mode(settings.STORAGE_MODE)
        self.prefix = settings.REDIS_PREFIX
        self.expire = settings.REDIS_EXPIRE
        self.redis_client: Optional[Redis] = None
        self._connection_args = {"decode_responses": True}
        self._redis_url = str(settings.REDIS_DSN)

    def _key_conv(self, conversation_id: str) -> str:
        """Key for the LLMConversation metadata Hash."""
        return f"{self.prefix}:{self.mode}:conv:{conversation_id}"

    def _key_turns(self, conversation_id: str) -> str:
        """Key for the LLMTurns Stream."""
        return f"{self.prefix}:{self.mode}:turns:{conversation_id}"

    def _key_artifacts(self, conversation_id: str) -> str:
        """Key for the Artifacts Hash (storing artifact_id -> artifact_json)."""
        return f"{self.prefix}:{self.mode}:artifacts:{conversation_id}"

    def _key_index_all_convs(self) -> str:
        """Key for the Sorted Set used to index all conversations by update time."""
        return f"{self.prefix}:{self.mode}:zset:convs_by_update"

    def connect(self) -> None:
        if self.redis_client:
            self._connected = True
            return
        try:
            self.redis_client = Redis.from_url(self._redis_url, **self._connection_args)
            self.redis_client.ping()
            self._connected = True
        except RedisConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    def disconnect(self) -> None:
        if self.redis_client:
            self.redis_client.close()
        self._connected = False

    def is_connected(self) -> bool:
        """Returns the current connection state."""
        return self._connected

    def _get_client(self) -> Redis:
        if not self.redis_client:
            raise ConnectionError("Storage is not connected.")
        return self.redis_client
    
    def _key_artifact_data(self, artifact_id: str) -> str:
        """Key for the raw binary data of an artifact."""
        return f"{self.prefix}:{self.mode}:artifact_data:{artifact_id}"

    

    def store_artifact_data(self, artifact_id: str, conversation_id: str, binary_data: bytes) -> str:

        """Stores binary data directly in a Redis key."""
        client = self._get_client()
        key = self._key_artifact_data(artifact_id)
        
        client.set(key, binary_data, ex=self.expire)
        
        return f"redis://{artifact_id}"

    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        """Retrieves binary data from a Redis key."""
        if not storage_uri.startswith("redis://"):
            raise ValueError("RedisStorage can only handle 'redis://' URIs for artifacts.")
            
        client = self._get_client()
        artifact_id = storage_uri[len("redis://"):]
        key = self._key_artifact_data(artifact_id)
        
        data = client.get(key)
        
        if data is None:
            raise FileNotFoundError(f"Artifact not found in Redis for URI: {storage_uri}")
            
        if isinstance(data, str):
            return data.encode('utf-8')
        return data

    def start_conversation(self, conversation: LLMConversation) -> None:
        client = self._get_client()
        key = self._key_conv(conversation.conversation_id)
        
        metadata_str = json.dumps(conversation.metadata)
        tags_str = json.dumps(conversation.tags)
        
        pipe = client.pipeline()
        pipe.hset(key, mapping={
            "conversation_id": conversation.conversation_id,
            "user_id": conversation.user_id or "",
            "entity_id": conversation.entity_id or "",
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "summary": conversation.summary or "",
            "tags": tags_str,
            "metadata": metadata_str,
        })
        pipe.zadd(self._key_index_all_convs(), {conversation.conversation_id: conversation.updated_at})
        
        if self.expire:
            pipe.expire(key, self.expire)
        
        pipe.execute()

    def append_turn(self, conversation_id: str, turn: LLMTurn) -> None:
        client = self._get_client()
        key_turns = self._key_turns(conversation_id)
        key_conv = self._key_conv(conversation_id)
        updated_at = int(time.time())

        turn_json = turn.model_dump_json()

        pipe = client.pipeline()
        pipe.xadd(key_turns, {"turn_data": turn_json})
        
        pipe.hset(key_conv, "updated_at", updated_at)
        pipe.zadd(self._key_index_all_convs(), {conversation_id: updated_at})
        
        pipe.execute()

    def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        client = self._get_client()
        key_conv = self._key_conv(conversation_id)
        key_turns = self._key_turns(conversation_id)
        
        pipe = client.pipeline()
        pipe.hgetall(key_conv)
        pipe.xrange(key_turns, min='-', max='+')
        results = pipe.execute()
        
        conv_hash = results[0]
        turn_stream_entries = results[1]

        if not conv_hash:
            return None

        turns = []
        for _, turn_fields in turn_stream_entries:
            turn_data_json = turn_fields.get("turn_data")
            if turn_data_json:
                turns.append(LLMTurn.model_validate_json(turn_data_json))

        return LLMConversation(
            conversation_id=conv_hash["conversation_id"],
            user_id=conv_hash.get("user_id"),
            entity_id=conv_hash.get("entity_id"),
            created_at=int(conv_hash["created_at"]),
            updated_at=int(conv_hash["updated_at"]),
            summary=conv_hash.get("summary"),
            tags=json.loads(conv_hash.get("tags", "[]")),
            metadata=json.loads(conv_hash.get("metadata", "{}")),
            turns=turns
        )

    def list_conversations(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[LLMConversation]:
        if user_id or entity_id:
            print("Warning: RedisStorage list_conversations does not currently support user_id/entity_id filtering.")

        client = self._get_client()
        conv_ids = client.zrevrange(self._key_index_all_convs(), offset, offset + limit - 1)
        if not conv_ids:
            return []

        pipe = client.pipeline()
        for conv_id in conv_ids:
            pipe.hgetall(self._key_conv(conv_id))
        
        results = pipe.execute()
        
        conversations = []
        for conv_hash in results:
            if conv_hash:
                conversations.append(LLMConversation(
                    conversation_id=conv_hash["conversation_id"],
                    user_id=conv_hash.get("user_id"),
                    entity_id=conv_hash.get("entity_id"),
                    created_at=int(conv_hash["created_at"]),
                    updated_at=int(conv_hash["updated_at"]),
                    summary=conv_hash.get("summary"),
                    tags=json.loads(conv_hash.get("tags", "[]")),
                    metadata=json.loads(conv_hash.get("metadata", "{}")),
                    turns=[]
                ))
        return conversations

    def log_artifact(self, artifact: Artifact) -> None:
        client = self._get_client()
        key = self._key_artifacts(artifact.conversation_id)
        artifact_json = artifact.model_dump_json()
        client.hset(key, artifact.artifact_id, artifact_json)

    def drop(self) -> None:
        """Deletes ALL keys associated with this storage prefix and mode."""
        client = self._get_client()
        for key in client.scan_iter(match=f"{self.prefix}:{self.mode}:*"):
            client.delete(key)