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
    
    This storage provider is designed to be flexible and dynamic:
    - Only creates InteractionSession/UserProfile directories when they are actually used
    - Supports generic Pydantic models for custom storage needs
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    def __init__(self, directory_path: Optional[str] = None, pretty_print: bool = True):
        """
        Initializes the JSON storage provider.

        Args:
            directory_path: The root directory where data will be stored. If None, uses "data" in the current working directory.
            pretty_print: If True, JSON files will be indented for readability.
        """
        super().__init__()
        self.base_path = Path(directory_path or "data").resolve()
        self.sessions_path = self.base_path / "sessions"
        self.profiles_path = self.base_path / "profiles"
        self.generic_path = self.base_path / "generic"  # For arbitrary models
        self._pretty_print = pretty_print
        self._json_indent = 4 if self._pretty_print else None
        self._lock: Optional[asyncio.Lock] = None
        
        # Create base directory but not subdirectories yet (lazy initialization)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Track which built-in directories have been initialized
        self._sessions_dir_initialized = False
        self._profiles_dir_initialized = False
        
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
    
    def _encode_id_for_filename(self, object_id: str) -> str:
        """
        Encode object ID to be safe for filename.
        
        Handles IDs that contain slashes (like file paths).
        """
        import urllib.parse
        return urllib.parse.quote(object_id, safe='')
    
    def _decode_id_from_filename(self, filename: str) -> str:
        """Decode filename back to object ID."""
        import urllib.parse
        # Remove .json extension
        if filename.endswith('.json'):
            filename = filename[:-5]
        return urllib.parse.unquote(filename)
    
    def _get_path(self, object_id: str, model_type: Type[BaseModel]) -> Path:
        if model_type is InteractionSession:
            return self.sessions_path / f"{object_id}.json"
        elif model_type is UserProfile:
            return self.profiles_path / f"{object_id}.json"
        else:
            # Generic model: generic/{model_type_name}/{encoded_id}.json
            model_folder = self.generic_path / model_type.__name__.lower()
            model_folder.mkdir(parents=True, exist_ok=True)
            encoded_id = self._encode_id_for_filename(object_id)
            return model_folder / f"{encoded_id}.json"
    
    def _serialize(self, data: Dict[str, Any]) -> str:
        return json.dumps(data, indent=self._json_indent)
    
    def _deserialize(self, data: str) -> Dict[str, Any]:
        return json.loads(data)



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
        """
        Creates storage directories.
        Note: InteractionSession and UserProfile directories are created lazily
        only when first accessed. This allows the storage to be used for
        generic purposes without creating unused infrastructure.
        """
        # Ensure base directory exists, but don't create subdirectories yet
        # Subdirectories will be created on-demand when accessed
        await asyncio.to_thread(self.base_path.mkdir, parents=True, exist_ok=True)
    
    async def _ensure_sessions_dir(self) -> None:
        """Lazily creates the sessions directory on first access."""
        if self._sessions_dir_initialized:
            return
        
        await asyncio.to_thread(self.sessions_path.mkdir, parents=True, exist_ok=True)
        self._sessions_dir_initialized = True
    
    async def _ensure_profiles_dir(self) -> None:
        """Lazily creates the profiles directory on first access."""
        if self._profiles_dir_initialized:
            return
        
        await asyncio.to_thread(self.profiles_path.mkdir, parents=True, exist_ok=True)
        self._profiles_dir_initialized = True

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        # Ensure directory exists - lazy initialization for built-in types
        if model_type is InteractionSession:
            await self._ensure_sessions_dir()
        elif model_type is UserProfile:
            await self._ensure_profiles_dir()
        # Generic models create their own directories in _get_path
        
        file_path = self._get_path(object_id, model_type)
        async with self.lock:
            if not await asyncio.to_thread(file_path.exists):
                return None
            try:
                content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                data = self._deserialize(content)
                
                # Try from_dict first, fallback to model_validate
                if hasattr(model_type, 'from_dict'):
                    return model_type.from_dict(data)
                else:
                    return model_type.model_validate(data)
            except (json.JSONDecodeError, TypeError) as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Could not parse file {file_path}. Error: {e}", "JSONStorage")
                return None

    async def upsert_async(self, data: BaseModel) -> None:
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()
        
        data_dict = data.model_dump(mode="json")
        json_string = self._serialize(data_dict)

        if isinstance(data, InteractionSession):
            # Ensure sessions directory exists before upserting
            await self._ensure_sessions_dir()
            file_path = self._get_path(data.session_id, InteractionSession)
        elif isinstance(data, UserProfile):
            # Ensure profiles directory exists before upserting
            await self._ensure_profiles_dir()
            file_path = self._get_path(data.user_id, UserProfile)
        else:
            # Generic model support - _get_path creates directory if needed
            primary_key_field = self._get_primary_key_field(type(data))
            object_id = getattr(data, primary_key_field)
            file_path = self._get_path(object_id, type(data))
        
        async with self.lock:
            try:
                await asyncio.to_thread(file_path.write_text, json_string, encoding="utf-8")
            except IOError as e:
                raise IOError(f"Failed to write file to {file_path}: {e}")

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        # Ensure directory exists - lazy initialization for built-in types
        if model_type is InteractionSession:
            await self._ensure_sessions_dir()
        elif model_type is UserProfile:
            await self._ensure_profiles_dir()
        # For generic models, only delete if directory exists
        
        file_path = self._get_path(object_id, model_type)
        async with self.lock:
            if await asyncio.to_thread(file_path.exists):
                try: 
                    await asyncio.to_thread(file_path.unlink)
                except OSError as e: 
                    from upsonic.utils.printing import error_log
                    error_log(f"Could not delete file {file_path}. Reason: {e}", "JSONStorage")
    
    async def list_all_async(self, model_type: Type[T]) -> list[T]:
        """List all objects of a specific type."""
        async with self.lock:
            if model_type is InteractionSession:
                # Ensure sessions directory exists
                await self._ensure_sessions_dir()
                folder = self.sessions_path
            elif model_type is UserProfile:
                # Ensure profiles directory exists
                await self._ensure_profiles_dir()
                folder = self.profiles_path
            else:
                # Generic model
                folder = self.generic_path / model_type.__name__.lower()
                if not await asyncio.to_thread(folder.exists):
                    return []
            
            results = []
            
            # Iterate through all .json files
            try:
                files = await asyncio.to_thread(list, folder.glob("*.json"))
                
                for file_path in files:
                    try:
                        content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                        data = self._deserialize(content)
                        
                        if hasattr(model_type, 'from_dict'):
                            obj = model_type.from_dict(data)
                        else:
                            obj = model_type.model_validate(data)
                        
                        results.append(obj)
                    except Exception:
                        continue
            except Exception:
                return []
            
            return results

    async def drop_async(self) -> None:
        """
        Drops all directories managed by this storage provider.
        Only drops InteractionSession/UserProfile directories if they were actually created.
        """
        async with self.lock:
            if await asyncio.to_thread(self.sessions_path.exists): 
                await asyncio.to_thread(shutil.rmtree, self.sessions_path)
            if await asyncio.to_thread(self.profiles_path.exists): 
                await asyncio.to_thread(shutil.rmtree, self.profiles_path)
            if await asyncio.to_thread(self.generic_path.exists):
                await asyncio.to_thread(shutil.rmtree, self.generic_path)
        
        # Reset initialization flags
        self._sessions_dir_initialized = False
        self._profiles_dir_initialized = False
        
        await self.create_async()
