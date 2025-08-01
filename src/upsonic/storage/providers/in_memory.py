import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Literal

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import (
    BaseSession,
    AgentSession
)

class InMemoryStorage(Storage):
    """
    An ephemeral, thread-safe, session-based storage provider that lives in memory,
    configured via direct arguments.
    """

    def __init__(
        self,
        max_sessions: Optional[int] = None,
        mode: Literal["agent", "team", "workflow", "workflow_v2"] = "agent",
    ):
        """
        Initializes the in-memory storage provider.

        Args:
            max_sessions: The maximum number of sessions to store. If set, the
                          storage acts as a fixed-size LRU cache.
            mode: The operational mode.
        """
        super().__init__(mode=mode)
        
        self.max_sessions = max_sessions
        self._sessions: Dict[str, BaseSession] = OrderedDict() if self.max_sessions else {}
        self._lock = threading.Lock()

        self.connect()


    def _get_session_model_class(self) -> type[BaseSession]:
        """Returns the appropriate Pydantic session model class based on the current mode."""
        if self.mode == "agent":
            return AgentSession
        raise ValueError(f"Invalid mode '{self.mode}' specified for InMemoryStorage.")


    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """For in-memory, this is a no-op that just marks the state as connected."""
        self._connected = True

    def disconnect(self) -> None:
        """For in-memory, this is a no-op that just marks the state as disconnected."""
        self._connected = False

    def create(self) -> None:
        """For in-memory, there is no persistent schema to create. This is a no-op."""
        pass

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[BaseSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if user_id and session.user_id != user_id:
                    return None

                if self.max_sessions:
                    self._sessions.move_to_end(session_id)
                
                return session.model_copy(deep=True)
        return None

    def upsert(self, session: BaseSession) -> Optional[BaseSession]:
        SessionModel = self._get_session_model_class()
        if not isinstance(session, SessionModel):
            raise TypeError(f"Session object must be of type {SessionModel.__name__} for mode '{self.mode}'")

        with self._lock:
            session.updated_at = int(time.time())
            session_copy = session.model_copy(deep=True)
            self._sessions[session.session_id] = session_copy

            if self.max_sessions:
                self._sessions.move_to_end(session.session_id)
                if len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)
            
            return session_copy

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[BaseSession]:
        with self._lock:
            all_sessions = list(self._sessions.values())

        filtered_sessions: List[BaseSession] = []
        entity_key = f"{self.mode}_id" if self.mode else None

        for session in all_sessions:
            if user_id and session.user_id != user_id:
                continue
            if entity_id and entity_key and getattr(session, entity_key, None) != entity_id:
                continue
            filtered_sessions.append(session.model_copy(deep=True))
            
        return filtered_sessions

    def get_recent_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 10) -> List[BaseSession]:
        all_sessions = self.get_all_sessions(user_id=user_id, entity_id=entity_id)
        all_sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return all_sessions[:limit]

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def drop(self) -> None:
        """Clears all sessions from memory."""
        with self._lock:
            self._sessions.clear()


    def log_artifact(self, artifact) -> None:
        raise NotImplementedError("Artifact logging should be handled within the session data for this provider.")

    def store_artifact_data(self, artifact_id: str, session_id: str, binary_data: bytes) -> str:
        raise NotImplementedError("Binary artifact storage is not handled by the session provider.")

    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        raise NotImplementedError("Binary artifact storage is not handled by the session provider.")