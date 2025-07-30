import json
import time
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import (
    BaseSession,
    AgentSession
)

class JSONStorage(Storage):
    """
    A simple, human-readable storage provider using one JSON file per session.

    This provider implements the Storage contract by storing each session as a
    single, self-contained JSON file on the local filesystem. It's ideal for
    development, debugging, and simple, single-node applications.
    """

    def __init__(
        self,
        directory_path: str,
        pretty_print: bool = True,
        mode: Literal["agent", "team", "workflow", "workflow_v2"] = "agent",
    ):
        """
        Initializes the JSON storage provider.

        Args:
            directory_path: The root directory where session files will be stored.
            pretty_print: If True, the output JSON will be indented for readability.
            mode: The operational mode.
        """
        super().__init__(mode=mode)
        
        # THE CHANGE: Parameters are now used directly.
        self.base_path = Path(directory_path).resolve()
        self.sessions_path = self.base_path / mode
        self._pretty_print = pretty_print
        self._json_indent = 4 if self._pretty_print else None

        self.create()
        self.connect()


    def _get_session_path(self, session_id: str) -> Path:
        """Generates the full, absolute path to a session file."""
        return self.sessions_path / f"{session_id}.json"

    def _serialize(self, data: Dict[str, Any]) -> str:
        """Serializes a dictionary to a JSON string, respecting pretty-printing settings."""
        return json.dumps(data, indent=self._json_indent)

    def _deserialize(self, data: str) -> Dict[str, Any]:
        """Deserializes a JSON string to a dictionary."""
        return json.loads(data)

    def _get_session_model_class(self) -> type[BaseSession]:
        """Returns the appropriate Pydantic session model class based on the current mode."""
        if self.mode == "agent":
            return AgentSession
        # ... add other modes as needed
        raise ValueError(f"Invalid mode '{self.mode}' specified for JSONStorage.")


    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """For file-based storage, connecting just means ensuring the directory exists."""
        self.create()
        self._connected = True

    def disconnect(self) -> None:
        """For file-based storage, there is no active connection to disconnect."""
        self._connected = False

    def create(self) -> None:
        """Ensures the base and mode-specific directories exist."""
        self.sessions_path.mkdir(parents=True, exist_ok=True)

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[BaseSession]:
        session_file = self._get_session_path(session_id)
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = self._deserialize(f.read())

            if user_id and session_data.get("user_id") != user_id:
                return None

            SessionModel = self._get_session_model_class()
            return SessionModel.from_dict(session_data)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not read or parse corrupt session file {session_file}. Error: {e}")
            return None

    def upsert(self, session: BaseSession) -> Optional[BaseSession]:
        SessionModel = self._get_session_model_class()
        if not isinstance(session, SessionModel):
            raise TypeError(f"Session object must be of type {SessionModel.__name__} for mode '{self.mode}'")

        session.updated_at = int(time.time())
        session_dict = session.model_dump(mode="json")
        json_string = self._serialize(session_dict)
        session_file = self._get_session_path(session.session_id)

        try:
            # 'w' mode will create the file or overwrite it if it exists, which is the 'upsert' behavior.
            with open(session_file, "w", encoding="utf-8") as f:
                f.write(json_string)
            return session
        except IOError as e:
            raise IOError(f"Failed to write session file to {session_file}: {e}")

    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[BaseSession]:
        """Retrieves all sessions, with inefficient client-side filtering by reading every file."""
        sessions: List[BaseSession] = []
        SessionModel = self._get_session_model_class()
        entity_key = f"{self.mode}_id" if self.mode else None

        for session_file in self.sessions_path.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = self._deserialize(f.read())

                if user_id and session_data.get("user_id") != user_id:
                    continue
                if entity_id and entity_key and session_data.get(entity_key) != entity_id:
                    continue
                
                sessions.append(SessionModel.from_dict(session_data))
            except (IOError, json.JSONDecodeError):
                continue
        
        return sessions

    def get_recent_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 10) -> List[BaseSession]:
        """Retrieves recent sessions by fetching all, then sorting in memory."""
        all_sessions = self.get_all_sessions(user_id=user_id, entity_id=entity_id)
        all_sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return all_sessions[:limit]

    def delete_session(self, session_id: str) -> None:
        try:
            session_file = self._get_session_path(session_id)
            session_file.unlink(missing_ok=True)
        except OSError as e:
            print(f"Error: Could not delete session file for ID {session_id}. Reason: {e}")
            
    def drop(self) -> None:
        """Deletes the entire directory for the current mode and all its contents."""
        if self.sessions_path.exists():
            shutil.rmtree(self.sessions_path)
        # Re-create the directory so the provider can still be used.
        self.create()


    def log_artifact(self, artifact) -> None:
        raise NotImplementedError("Artifact logging should be handled within the session data for this provider.")

    def store_artifact_data(self, artifact_id: str, session_id: str, binary_data: bytes) -> str:
        raise NotImplementedError("Binary artifact storage is not handled by the session provider.")

    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        raise NotImplementedError("Binary artifact storage is not handled by the session provider.")