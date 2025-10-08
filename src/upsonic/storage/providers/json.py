import asyncio
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Type, Union, TypeVar

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import (
    InteractionSession,
    UserProfile
)

T = TypeVar('T', bound=BaseModel)

class JSONStorage(Storage):
    """
    A hybrid sync/async, file-based storage provider using one JSON file per object.

    This provider implements both a synchronous and an asynchronous API. The
    synchronous methods are convenient wrappers that intelligently manage the
    event loop to run the core async logic. The core async logic uses
    `asyncio.to_thread` to ensure file I/O operations are non-blocking.
    """

    def __init__(self, directory_path: str, pretty_print: bool = True):
        """
        Initializes the JSON storage provider.

        Args:
            directory_path: The root directory where data will be stored.
            pretty_print: If True, JSON files will be indented for readability.
        """
        super().__init__()
        self.base_path = Path(directory_path).resolve()
        self.sessions_path = self.base_path / "sessions"
        self.profiles_path = self.base_path / "profiles"
        self._pretty_print = pretty_print
        self._json_indent = 4 if self._pretty_print else None
        self._lock: Optional[asyncio.Lock] = None
        
        self.sessions_path.mkdir(parents=True, exist_ok=True)
        self.profiles_path.mkdir(parents=True, exist_ok=True)
        self._connected = True

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

    def _get_path(self, object_id: str, model_type: Type[BaseModel]) -> Path:
        if model_type is InteractionSession: return self.sessions_path / f"{object_id}.json"
        elif model_type is UserProfile: return self.profiles_path / f"{object_id}.json"
        raise TypeError(f"Unsupported model type for path generation: {model_type.__name__}")
    
    def _serialize(self, data: Dict[str, Any]) -> str: return json.dumps(data, indent=self._json_indent)
    def _deserialize(self, data: str) -> Dict[str, Any]: return json.loads(data)



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
    def upsert(self, data: Union[InteractionSession, UserProfile]) -> None:
        return self._run_async_from_sync(self.upsert_async(data))
    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None:
        return self._run_async_from_sync(self.delete_async(object_id, model_type))
    def drop(self) -> None:
        return self._run_async_from_sync(self.drop_async())
        


    async def is_connected_async(self) -> bool:
        return self._connected
    
    async def connect_async(self) -> None:
        if self._connected: return
        await self.create_async()
        self._connected = True

    async def disconnect_async(self) -> None:
        self._connected = False

    async def create_async(self) -> None:
        await asyncio.to_thread(self.sessions_path.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(self.profiles_path.mkdir, parents=True, exist_ok=True)

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        file_path = self._get_path(object_id, model_type)
        async with self.lock:
            if not await asyncio.to_thread(file_path.exists): return None
            try:
                content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                data = self._deserialize(content)
                return model_type.from_dict(data)
            except (json.JSONDecodeError, TypeError) as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Could not parse file {file_path}. Error: {e}", "JSONStorage")
                return None

    async def upsert_async(self, data: Union[InteractionSession, UserProfile]) -> None:
        data.updated_at = time.time()
        data_dict = data.model_dump(mode="json")
        json_string = self._serialize(data_dict)

        if isinstance(data, InteractionSession): file_path = self._get_path(data.session_id, InteractionSession)
        elif isinstance(data, UserProfile): file_path = self._get_path(data.user_id, UserProfile)
        else: raise TypeError(f"Unsupported data type for upsert: {type(data).__name__}")
        
        async with self.lock:
            try:
                await asyncio.to_thread(file_path.write_text, json_string, encoding="utf-8")
            except IOError as e:
                raise IOError(f"Failed to write file to {file_path}: {e}")

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        file_path = self._get_path(object_id, model_type)
        async with self.lock:
            if await asyncio.to_thread(file_path.exists):
                try: 
                    await asyncio.to_thread(file_path.unlink)
                except OSError as e: 
                    from upsonic.utils.printing import error_log
                    error_log(f"Could not delete file {file_path}. Reason: {e}", "JSONStorage")

    async def drop_async(self) -> None:
        async with self.lock:
            if await asyncio.to_thread(self.sessions_path.exists): 
                await asyncio.to_thread(shutil.rmtree, self.sessions_path)
            if await asyncio.to_thread(self.profiles_path.exists): 
                await asyncio.to_thread(shutil.rmtree, self.profiles_path)
        await self.create_async()
