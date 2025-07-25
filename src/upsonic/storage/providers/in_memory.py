import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set

from upsonic.storage.base import Storage
from upsonic.storage.settings import InMemorySettings
from upsonic.storage.session.llm import LLMConversation, LLMTurn, Artifact


class InMemoryStorage(Storage):
    """
    Ephemeral, thread-safe storage for the LLM Interaction Archive.
    - Stores full LLMConversation objects in memory.
    - Ideal for unit testing, rapid prototyping, and transient workflows.
    """
    def __init__(self, settings: InMemorySettings):
        super().__init__()
        self._set_mode(settings.STORAGE_MODE)
        self.max_conversations = settings.IN_MEMORY_MAX_SESSIONS
        
        self._conversations: Dict[str, LLMConversation] = OrderedDict() if self.max_conversations else {}
        self._artifacts: Dict[str, List[Artifact]] = {}
        self._artifact_data: Dict[str, bytes] = {}
        
        self._user_index: Dict[str, Set[str]] = {}
        self._entity_index: Dict[str, Set[str]] = {}
        
        self._lock = threading.Lock()

    def get_lock(self) -> threading.Lock:
        return self._lock

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        """Returns the current connection state."""
        return self._connected

    def _add_to_indexes(self, conversation: LLMConversation):
        if conversation.user_id:
            self._user_index.setdefault(conversation.user_id, set()).add(conversation.conversation_id)
        if conversation.entity_id:
            self._entity_index.setdefault(conversation.entity_id, set()).add(conversation.conversation_id)

    def _remove_from_indexes(self, conversation: LLMConversation):
        if conversation.user_id and conversation.user_id in self._user_index:
            self._user_index[conversation.user_id].discard(conversation.conversation_id)
            if not self._user_index[conversation.user_id]:
                del self._user_index[conversation.user_id]
        if conversation.entity_id and conversation.entity_id in self._entity_index:
            self._entity_index[conversation.entity_id].discard(conversation.conversation_id)
            if not self._entity_index[conversation.entity_id]:
                del self._entity_index[conversation.entity_id]


    def store_artifact_data(self, artifact_id: str, conversation_id: str, binary_data: bytes) -> str:
        """Stores binary data in an in-memory dictionary."""
        storage_uri = f"memory://{artifact_id}"
        with self._lock:
            self._artifact_data[storage_uri] = binary_data
        return storage_uri

    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        """Retrieves binary data from the in-memory dictionary."""
        with self._lock:
            data = self._artifact_data.get(storage_uri)
            if data is None:
                raise FileNotFoundError(f"Artifact not found in memory for URI: {storage_uri}")
            return data
    
    def start_conversation(self, conversation: LLMConversation) -> None:
        with self._lock:
            convo_copy = conversation.model_copy(deep=True)
            convo_copy.turns = []
            
            if convo_copy.conversation_id in self._conversations:
                return

            self._conversations[convo_copy.conversation_id] = convo_copy
            self._add_to_indexes(convo_copy)
            
            if self.max_conversations and len(self._conversations) > self.max_conversations:
                evicted_cid, evicted_convo = self._conversations.popitem(last=False)
                self._remove_from_indexes(evicted_convo)
                if evicted_cid in self._artifacts:
                    del self._artifacts[evicted_cid]

    def append_turn(self, conversation_id: str, turn: LLMTurn) -> None:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation:
                conversation.turns.append(turn.model_copy(deep=True))
                conversation.updated_at = int(time.time())
                if self.max_conversations:
                    self._conversations.move_to_end(conversation_id)

    def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation:
                if self.max_conversations:
                    self._conversations.move_to_end(conversation_id)
                return conversation.model_copy(deep=True)
            return None

    def list_conversations(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[LLMConversation]:
        with self._lock:
            if user_id and entity_id:
                user_cids = self._user_index.get(user_id, set())
                entity_cids = self._entity_index.get(entity_id, set())
                candidate_cids = user_cids.intersection(entity_cids)
            elif user_id:
                candidate_cids = self._user_index.get(user_id, set())
            elif entity_id:
                candidate_cids = self._entity_index.get(entity_id, set())
            else:
                candidate_cids = set(self._conversations.keys())
            
            filtered_convos = [self._conversations[cid] for cid in candidate_cids if cid in self._conversations]

        filtered_convos.sort(key=lambda c: c.updated_at, reverse=True)
        
        paginated_convos = filtered_convos[offset : offset + limit]
        
        return [
            convo.model_copy(update={'turns': []}, deep=True)
            for convo in paginated_convos
        ]
        
    def log_artifact(self, artifact: Artifact) -> None:
        with self._lock:
            self._artifacts.setdefault(artifact.conversation_id, []).append(artifact.model_copy(deep=True))

    def drop(self) -> None:
        """Clears all in-memory data structures."""
        with self._lock:
            self._conversations.clear()
            self._artifacts.clear()
            self._user_index.clear()
            self._entity_index.clear()