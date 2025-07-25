import msvcrt
import json
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from upsonic.storage.base import Storage
from upsonic.storage.settings import JSONSettings
from upsonic.storage.session.llm import LLMConversation, LLMTurn, Artifact

# Thread-local storage for file locks
_thread_local = threading.local()


class JSONStorage(Storage):
    """
    Human-readable, file-system storage for the LLM Interaction Archive.
    - Each conversation is a directory.
    - Conversation metadata is in a '_conversation.json' file.
    - Each turn is a separate JSON file within the conversation's directory.
    """
    def __init__(self, settings: JSONSettings):
        super().__init__()
        self._set_mode(settings.STORAGE_MODE)
        self.base_path = settings.JSON_DIRECTORY_PATH.resolve()
        self.artifacts_base_path = settings.JSON_ARTIFACT_PATH.resolve()
        self._pretty_print = settings.JSON_PRETTY_PRINT
        self._json_indent = 4 if self._pretty_print else None
        
        self.conversations_path = self.base_path / self.mode
        self._lock_path = self.conversations_path / "_storage.lock"

    def _acquire_lock(self):
        if not hasattr(_thread_local, 'lock_file'):
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            _thread_local.lock_file = self._lock_path.open('r+')
        try:
            msvcrt.locking(_thread_local.lock_file.fileno(), msvcrt.LK_LOCK, 1)
        except OSError as e:
            raise IOError(f"Could not acquire lock on {self._lock_path}: {e}")

    def _release_lock(self):
        if hasattr(_thread_local, 'lock_file'):
            try:
                msvcrt.locking(_thread_local.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError as e:
                raise IOError(f"Could not release lock on {self._lock_path}: {e}")

            _thread_local.lock_file.close()
            del _thread_local.lock_file

    def connect(self) -> None:
        try:
            self.conversations_path.mkdir(parents=True, exist_ok=True)
            self._lock_path.touch()
            self._connected = True
        except OSError as e:
            raise ConnectionError(f"Could not access storage directory {self.conversations_path}: {e}")

    def disconnect(self) -> None:
        self._connected = False
    
    def is_connected(self) -> bool:
        """Returns the current connection state."""
        return self._connected

    def _get_conv_dir(self, conversation_id: str) -> Path:
        return self.conversations_path / conversation_id

    def _get_conv_meta_path(self, conversation_id: str) -> Path:
        return self._get_conv_dir(conversation_id) / "_conversation.json"

    def _get_turn_path(self, conversation_id: str, turn: LLMTurn) -> Path:
        filename = f"{turn.timestamp_start}_{turn.turn_id}.json"
        return self._get_conv_dir(conversation_id) / filename

    def _get_artifacts_path(self, conversation_id: str) -> Path:
        return self._get_conv_dir(conversation_id) / "_artifacts.json"


    def store_artifact_data(self, artifact_id: str, conversation_id: str, binary_data: bytes) -> str:
        """Stores the artifact as a file on disk."""
        self._acquire_lock()
        try:
            conv_artifact_path = self.artifacts_base_path / conversation_id
            conv_artifact_path.mkdir(parents=True, exist_ok=True)
            
            file_path = conv_artifact_path / artifact_id
            
            with file_path.open("wb") as f:
                f.write(binary_data)
            
            return file_path.as_uri()
        finally:
            self._release_lock()

    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        """Retrieves an artifact from a file URI."""
        if not storage_uri.startswith("file://"):
            raise ValueError("JSONStorage can only handle 'file://' URIs for artifacts.")
        
        file_path_str = storage_uri[len("file://"):]
        if os.name == 'nt' and file_path_str.startswith('/'):
            file_path_str = file_path_str[1:]
        
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file not found at: {file_path}")
            
        with file_path.open("rb") as f:
            return f.read()
    

    def start_conversation(self, conversation: LLMConversation) -> None:
        self._acquire_lock()
        try:
            conv_dir = self._get_conv_dir(conversation.conversation_id)
            conv_dir.mkdir(exist_ok=True)
            meta_path = self._get_conv_meta_path(conversation.conversation_id)
            
            meta_data = conversation.model_dump(exclude={"turns"})
            
            with meta_path.open('w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=self._json_indent)
        finally:
            self._release_lock()

    def append_turn(self, conversation_id: str, turn: LLMTurn) -> None:
        self._acquire_lock()
        try:
            conv_dir = self._get_conv_dir(conversation_id)
            if not conv_dir.exists():
                raise FileNotFoundError(f"Cannot append turn, conversation directory not found for ID: {conversation_id}")

            turn_path = self._get_turn_path(conversation_id, turn)
            with turn_path.open('w', encoding='utf-8') as f:
                f.write(turn.model_dump_json(indent=self._json_indent))
            
            meta_path = self._get_conv_meta_path(conversation_id)
            if meta_path.exists():
                with meta_path.open('r+', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    meta_data['updated_at'] = int(time.time())
                    f.seek(0)
                    json.dump(meta_data, f, indent=self._json_indent)
                    f.truncate()
        finally:
            self._release_lock()

    def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        conv_dir = self._get_conv_dir(conversation_id)
        meta_path = self._get_conv_meta_path(conversation_id)

        if not meta_path.exists():
            return None

        with meta_path.open('r', encoding='utf-8') as f:
            meta_data = json.load(f)

        turns = []
        turn_files = sorted(conv_dir.glob("*.json"))
        for turn_file in turn_files:
            if turn_file.name.startswith("_"):
                continue
            with turn_file.open('r', encoding='utf-8') as f:
                turns.append(LLMTurn.model_validate_json(f.read()))

        meta_data['turns'] = turns
        return LLMConversation.model_validate(meta_data)

    def list_conversations(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[LLMConversation]:
        self._acquire_lock()
        try:
            all_conv_meta = []
            for conv_dir in self.conversations_path.iterdir():
                if not conv_dir.is_dir():
                    continue
                
                meta_path = conv_dir / "_conversation.json"
                if meta_path.exists():
                    with meta_path.open('r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                        user_match = (not user_id) or (meta_data.get('user_id') == user_id)
                        entity_match = (not entity_id) or (meta_data.get('entity_id') == entity_id)
                        if user_match and entity_match:
                            all_conv_meta.append(meta_data)
            
            all_conv_meta.sort(key=lambda x: x.get('updated_at', 0), reverse=True)
            
            paginated_meta = all_conv_meta[offset : offset + limit]

            return [LLMConversation.model_validate(meta) for meta in paginated_meta]
        finally:
            self._release_lock()

    def log_artifact(self, artifact: Artifact) -> None:
        self._acquire_lock()
        try:
            artifacts_path = self._get_artifacts_path(artifact.conversation_id)
            
            if artifacts_path.exists():
                with artifacts_path.open('r', encoding='utf-8') as f:
                    artifacts_list = json.load(f)
            else:
                artifacts_list = []
                
            artifacts_list.append(artifact.model_dump(mode="json"))
            
            with artifacts_path.open('w', encoding='utf-8') as f:
                json.dump(artifacts_list, f, indent=self._json_indent)
        finally:
            self._release_lock()

    def drop(self) -> None:
        """Deletes the entire directory for the current mode."""
        self._acquire_lock()
        try:
            if self.conversations_path.exists():
                shutil.rmtree(self.conversations_path)
        finally:
            self._release_lock()