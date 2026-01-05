import asyncio
import json
import time
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Type, Union, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.session.agent import AgentSession

if TYPE_CHECKING:
    from upsonic.culture.cultural_knowledge import CulturalKnowledge

T = TypeVar('T', bound=BaseModel)

class JSONStorage(Storage):
    """
    A hybrid sync/async, file-based storage provider using one JSON file per object.

    This provider implements both a synchronous and an asynchronous API. The
    synchronous methods are convenient wrappers that intelligently manage the
    event loop to run the core async logic. The core async logic uses
    `asyncio.to_thread` to ensure file I/O operations are non-blocking.
    
    This storage provider is designed to be flexible and dynamic:
    - Only creates AgentSession directories when they are actually used
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
        self.generic_path = self.base_path / "generic"
        self.cultural_knowledge_path = self.base_path / "cultural_knowledge"
        self._pretty_print = pretty_print
        self._json_indent = 4 if self._pretty_print else None
        self._lock: Optional[asyncio.Lock] = None
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self._sessions_dir_initialized = False
        self._cultural_knowledge_dir_initialized = False
        
        self._connected = True

    @property
    def lock(self) -> asyncio.Lock:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(current_loop)
        
        if self._lock is None or self._lock._loop is not current_loop:
            self._lock = asyncio.Lock()
            
        return self._lock

    def _get_primary_key_field(self, model_type: Type[BaseModel]) -> str:
        if model_type.__name__ == "AgentSession":
            return "session_id"
        
        if hasattr(model_type, 'model_fields'):
            for field_name in ['path', 'id', 'key', 'name']:
                if field_name in model_type.model_fields:
                    return field_name
        return "id"
    
    def _encode_id_for_filename(self, object_id: str) -> str:
        import urllib.parse
        return urllib.parse.quote(object_id, safe='')
    
    def _decode_id_from_filename(self, filename: str) -> str:
        import urllib.parse
        if filename.endswith('.json'):
            filename = filename[:-5]
        return urllib.parse.unquote(filename)
    
    def _get_path(self, object_id: str, model_type: Type[BaseModel]) -> Path:
        if model_type.__name__ == "AgentSession":
            return self.sessions_path / f"{object_id}.json"
        else:
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
    def upsert(self, data: BaseModel) -> None:
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
        await asyncio.to_thread(self.base_path.mkdir, parents=True, exist_ok=True)
    
    async def _ensure_sessions_dir(self) -> None:
        if self._sessions_dir_initialized:
            return
        
        await asyncio.to_thread(self.sessions_path.mkdir, parents=True, exist_ok=True)
        self._sessions_dir_initialized = True
    
    async def _ensure_cultural_knowledge_dir(self) -> None:
        if self._cultural_knowledge_dir_initialized:
            return
        
        await asyncio.to_thread(self.cultural_knowledge_path.mkdir, parents=True, exist_ok=True)
        self._cultural_knowledge_dir_initialized = True

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        import base64
        if model_type.__name__ == "AgentSession":
            await self._ensure_sessions_dir()
        
        file_path = self._get_path(object_id, model_type)
        async with self.lock:
            if not await asyncio.to_thread(file_path.exists):
                return None
            try:
                content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                data = self._deserialize(content)
                
                # Check for new serialized format
                if isinstance(data, dict) and "__serialized__" in data:
                    serialized_bytes = base64.b64decode(data["__serialized__"].encode('utf-8'))
                    return model_type.deserialize(serialized_bytes)
                
                # Fallback to old format
                if hasattr(model_type, 'from_dict'):
                    return model_type.from_dict(data)
                else:
                    return model_type.model_validate(data)
            except (json.JSONDecodeError, TypeError) as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Could not parse file {file_path}. Error: {e}", "JSONStorage")
                return None

    async def upsert_async(self, data: BaseModel) -> None:
        import base64
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()
        
        if type(data).__name__ == "AgentSession":
            await self._ensure_sessions_dir()
            # Use serialize() to get bytes, then base64 encode
            serialized_bytes = data.serialize()
            data_dict = {"__serialized__": base64.b64encode(serialized_bytes).decode('utf-8')}
            file_path = self._get_path(data.session_id, type(data))
        else:
            data_dict = data.model_dump(mode="json")
            primary_key_field = self._get_primary_key_field(type(data))
            object_id = getattr(data, primary_key_field)
            file_path = self._get_path(object_id, type(data))
        
        json_string = self._serialize(data_dict)
        
        async with self.lock:
            try:
                await asyncio.to_thread(file_path.write_text, json_string, encoding="utf-8")
            except IOError as e:
                raise IOError(f"Failed to write file to {file_path}: {e}")

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        if model_type.__name__ == "AgentSession":
            await self._ensure_sessions_dir()
        
        file_path = self._get_path(object_id, model_type)
        async with self.lock:
            if await asyncio.to_thread(file_path.exists):
                try: 
                    await asyncio.to_thread(file_path.unlink)
                except OSError as e: 
                    from upsonic.utils.printing import error_log
                    error_log(f"Could not delete file {file_path}. Reason: {e}", "JSONStorage")
    
    async def list_all_async(self, model_type: Type[T]) -> list[T]:
        async with self.lock:
            if model_type.__name__ == "AgentSession":
                await self._ensure_sessions_dir()
                folder = self.sessions_path
            else:
                folder = self.generic_path / model_type.__name__.lower()
                if not await asyncio.to_thread(folder.exists):
                    return []
            
            results = []
            
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
        async with self.lock:
            if await asyncio.to_thread(self.sessions_path.exists): 
                await asyncio.to_thread(shutil.rmtree, self.sessions_path)
            if await asyncio.to_thread(self.generic_path.exists):
                await asyncio.to_thread(shutil.rmtree, self.generic_path)
            if await asyncio.to_thread(self.cultural_knowledge_path.exists):
                await asyncio.to_thread(shutil.rmtree, self.cultural_knowledge_path)
        
        self._sessions_dir_initialized = False
        self._cultural_knowledge_dir_initialized = False
        
        await self.create_async()

    # =========================================================================
    # Cultural Knowledge Methods
    # =========================================================================

    async def read_cultural_knowledge_async(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        await self._ensure_cultural_knowledge_dir()
        file_path = self.cultural_knowledge_path / f"{knowledge_id}.json"
        
        async with self.lock:
            if not await asyncio.to_thread(file_path.exists):
                return None
            try:
                content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                data = self._deserialize(content)
                return CulturalKnowledge.from_dict(data)
            except (json.JSONDecodeError, TypeError) as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Could not parse cultural knowledge file {file_path}. Error: {e}", "JSONStorage")
                return None

    async def upsert_cultural_knowledge_async(self, knowledge: "CulturalKnowledge") -> None:
        await self._ensure_cultural_knowledge_dir()
        
        knowledge.bump_updated_at()
        
        data_dict = knowledge.to_dict()
        json_string = self._serialize(data_dict)
        
        file_path = self.cultural_knowledge_path / f"{knowledge.id}.json"
        
        async with self.lock:
            try:
                await asyncio.to_thread(file_path.write_text, json_string, encoding="utf-8")
            except IOError as e:
                raise IOError(f"Failed to write cultural knowledge file to {file_path}: {e}")

    async def delete_cultural_knowledge_async(self, knowledge_id: str) -> None:
        await self._ensure_cultural_knowledge_dir()
        file_path = self.cultural_knowledge_path / f"{knowledge_id}.json"
        
        async with self.lock:
            if await asyncio.to_thread(file_path.exists):
                try:
                    await asyncio.to_thread(file_path.unlink)
                except OSError as e:
                    from upsonic.utils.printing import error_log
                    error_log(f"Could not delete cultural knowledge file {file_path}. Reason: {e}", "JSONStorage")

    async def list_all_cultural_knowledge_async(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        await self._ensure_cultural_knowledge_dir()
        
        results = []
        
        async with self.lock:
            try:
                files = await asyncio.to_thread(list, self.cultural_knowledge_path.glob("*.json"))
                
                for file_path in files:
                    try:
                        content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                        data = self._deserialize(content)
                        knowledge = CulturalKnowledge.from_dict(data)
                        
                        if name is not None:
                            if knowledge.name is None:
                                continue
                            if name.lower() not in knowledge.name.lower():
                                continue
                        
                        results.append(knowledge)
                    except Exception:
                        continue
            except Exception:
                return []
        
        return results

    async def clear_cultural_knowledge_async(self) -> None:
        await self._ensure_cultural_knowledge_dir()
        
        async with self.lock:
            if await asyncio.to_thread(self.cultural_knowledge_path.exists):
                await asyncio.to_thread(shutil.rmtree, self.cultural_knowledge_path)
                await asyncio.to_thread(self.cultural_knowledge_path.mkdir, parents=True, exist_ok=True)
