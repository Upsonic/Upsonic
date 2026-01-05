from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, TYPE_CHECKING

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.session.agent import AgentSession

if TYPE_CHECKING:
    from upsonic.culture.cultural_knowledge import CulturalKnowledge

try:
    from mem0 import AsyncMemoryClient, Memory
    HAS_ASYNC_CLIENT = True
    _MEM0_AVAILABLE = True
except ImportError:
    try:
        from mem0 import Memory, MemoryClient as AsyncMemoryClient
        HAS_ASYNC_CLIENT = False
        _MEM0_AVAILABLE = True
    except ImportError:
        AsyncMemoryClient = None
        Memory = None
        MemoryClient = None
        HAS_ASYNC_CLIENT = False
        _MEM0_AVAILABLE = False


T = TypeVar('T', bound=BaseModel)


class Mem0Storage(Storage):
    """
    Mem0 storage provider.
    
    This storage provider is designed to be flexible and dynamic:
    - Can accept a pre-existing Mem0 client (Memory or AsyncMemoryClient) or create one
    - Supports generic Pydantic models through custom categories
    - Uses AgentSession category when needed
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    CATEGORY_SESSION = "upsonic_agent_session"
    CATEGORY_CULTURE = "upsonic_cultural_knowledge"

    def __init__(
        self,
        client: Optional[Union['Memory', 'AsyncMemoryClient']] = None,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        local_config: Optional[Dict[str, Any]] = None,
        namespace: str = "upsonic",
        infer: bool = False,
        custom_categories: Optional[List[str]] = None,
        enable_caching: bool = True,
        cache_ttl: int = 300,
        output_format: str = "v1.1",
        version: str = "v2",
    ):
        """
        Initialize Mem0 storage provider.

        Args:
            client: Optional pre-existing Memory or AsyncMemoryClient. If provided, this client
                will be used instead of creating a new one. User is responsible for
                client lifecycle management when providing their own client.
            api_key: Mem0 Platform API key (if using hosted service). Ignored if client is provided.
            org_id: Organization ID for Mem0 Platform. Ignored if client is provided.
            project_id: Project ID for Mem0 Platform. Ignored if client is provided.
            local_config: Configuration dict for Open Source Mem0. Ignored if client is provided.
            namespace: Application namespace for organizing memories.
            infer: Enable LLM-based memory inference (False for structured storage).
            custom_categories: Additional custom categories for the project.
            enable_caching: Enable internal ID caching for faster lookups.
            cache_ttl: Cache time-to-live in seconds (0 = no expiry).
            output_format: Mem0 output format version.
            version: Mem0 API version.
        """
        if not _MEM0_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="mem0ai",
                install_command='pip install "upsonic[mem0]" or pip install mem0ai',
                feature_name="Mem0 storage provider"
            )

        super().__init__()

        self.namespace = namespace
        self.infer = infer
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.output_format = output_format
        self.version = version
        
        self._id_to_memory_id: Dict[str, str] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        self._client: Optional[Union[Memory, AsyncMemoryClient]] = client
        self._owns_client = (client is None)
        self._is_platform_client = False
        
        if client and hasattr(client, 'api_key'):
            self._is_platform_client = True
        
        self._api_key = api_key
        self._org_id = org_id
        self._project_id = project_id
        self._local_config = local_config
        self._custom_categories = custom_categories or []
        
        all_categories = [self.CATEGORY_SESSION, self.CATEGORY_CULTURE]
        if self._custom_categories:
            all_categories.extend(self._custom_categories)
        self._all_categories = list(set(all_categories))

    async def _initialize_client(self) -> Union[Memory, AsyncMemoryClient]:
        try:
            if self._api_key:
                from upsonic.utils.printing import info_log
                info_log("Initializing Mem0 Platform async client", "Mem0Storage")
                
                client_kwargs = {"api_key": self._api_key}
                if self._org_id:
                    client_kwargs["org_id"] = self._org_id
                if self._project_id:
                    client_kwargs["project_id"] = self._project_id
                    
                client = AsyncMemoryClient(**client_kwargs)
                self._is_platform_client = True
                
                if self._all_categories:
                    try:
                        # Use new API: client.project.update() instead of deprecated update_project()
                        if hasattr(client, 'project') and hasattr(client.project, 'update'):
                            await client.project.update(custom_categories=self._all_categories)
                        else:
                            # Fallback to old method if new API not available
                            await client.update_project(custom_categories=self._all_categories)
                    except Exception as e:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Could not update project categories: {e}", "Mem0Storage")
                
                return client
                
            elif self._local_config:
                from upsonic.utils.printing import info_log
                info_log("Initializing Mem0 Open Source client with config", "Mem0Storage")
                return Memory.from_config(self._local_config)
                
            else:
                from upsonic.utils.printing import info_log
                info_log("Initializing Mem0 Open Source client with defaults", "Mem0Storage")
                return Memory()
                
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to initialize Mem0 client: {e}", "Mem0Storage")
            raise ConnectionError(f"Failed to initialize Mem0 client: {e}") from e

    async def _get_client(self) -> Union[Memory, AsyncMemoryClient]:
        if self._client is None:
            self._client = await self._initialize_client()
        return self._client
    
    async def _call_get_all(self, client: Union[Memory, AsyncMemoryClient], **kwargs) -> Any:
        """Helper to call get_all on both sync and async clients."""
        import asyncio
        if HAS_ASYNC_CLIENT and isinstance(client, AsyncMemoryClient):
            return await client.get_all(**kwargs)
        else:
            # Sync client - run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: client.get_all(**kwargs))
    
    async def _call_add(self, client: Union[Memory, AsyncMemoryClient], **kwargs) -> Any:
        """Helper to call add on both sync and async clients."""
        import asyncio
        if HAS_ASYNC_CLIENT and isinstance(client, AsyncMemoryClient):
            return await client.add(**kwargs)
        else:
            # Sync client - run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: client.add(**kwargs))
    
    async def _call_update(self, client: Union[Memory, AsyncMemoryClient], **kwargs) -> Any:
        """Helper to call update on both sync and async clients."""
        import asyncio
        if HAS_ASYNC_CLIENT and isinstance(client, AsyncMemoryClient):
            return await client.update(**kwargs)
        else:
            # Sync client - run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: client.update(**kwargs))
    
    async def _call_delete(self, client: Union[Memory, AsyncMemoryClient], **kwargs) -> Any:
        """Helper to call delete on both sync and async clients."""
        import asyncio
        if HAS_ASYNC_CLIENT and isinstance(client, AsyncMemoryClient):
            return await client.delete(**kwargs)
        else:
            # Sync client - run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: client.delete(**kwargs))
    
    async def _ensure_connection(self) -> None:
        """Ensure client is available and connected."""
        if not self._connected or self._client is None:
            await self.connect_async()



    def _cache_memory_id(self, object_id: str, memory_id: str) -> None:
        if self.enable_caching:
            self._id_to_memory_id[object_id] = memory_id
            self._cache_timestamps[object_id] = time.time()

    def _get_cached_memory_id(self, object_id: str) -> Optional[str]:
        if not self.enable_caching:
            return None
            
        memory_id = self._id_to_memory_id.get(object_id)
        if memory_id is None:
            return None
            
        if self.cache_ttl > 0:
            cached_at = self._cache_timestamps.get(object_id, 0)
            if time.time() - cached_at > self.cache_ttl:
                self._clear_cache_entry(object_id)
                return None
                
        return memory_id

    def _clear_cache_entry(self, object_id: str) -> None:
        self._id_to_memory_id.pop(object_id, None)
        self._cache_timestamps.pop(object_id, None)

    def _clear_all_cache(self) -> None:
        self._id_to_memory_id.clear()
        self._cache_timestamps.clear()



    def _serialize_session(self, session: AgentSession) -> tuple[str, Dict[str, Any]]:
        import base64
        # Use serialize: serialize to bytes, then base64 encode for storage
        serialized_bytes = session.serialize()
        serialized_str = base64.b64encode(serialized_bytes).decode('utf-8')
        
        memory_data = {
            "type": "agent_session",
            "session_id": session.session_id,
            "data": serialized_str,
        }
        memory_text = json.dumps(memory_data, ensure_ascii=False)
        
        metadata = {
            "category": self.CATEGORY_SESSION,
            "namespace": self.namespace,
            "session_id": session.session_id,
            "user_id": session.user_id,
            "agent_id": session.agent_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }
        
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return memory_text, metadata

    def _deserialize_session(self, memory_dict: Dict[str, Any]) -> AgentSession:
        import base64
        memory_text = memory_dict.get("memory", "{}")
        try:
            memory_data = json.loads(memory_text) if isinstance(memory_text, str) else {}
        except json.JSONDecodeError:
            memory_data = {}
        
        session_data = memory_data.get("data", "")
        if session_data:
            # Use deserialize: base64 decode then deserialize
            serialized_bytes = base64.b64decode(session_data.encode('utf-8'))
            return AgentSession.deserialize(serialized_bytes)
        return AgentSession(session_id="unknown")

    def _get_primary_key_field(self, model_type: Type[BaseModel]) -> str:
        if model_type.__name__ == "AgentSession":
            return "session_id"
        
        if hasattr(model_type, 'model_fields'):
            for field_name in ['path', 'id', 'key', 'name']:
                if field_name in model_type.model_fields:
                    return field_name
        return "id"
    
    def _serialize_generic_model(self, model: BaseModel) -> tuple[str, Dict[str, Any]]:
        model_type = type(model)
        model_name = model_type.__name__
        
        memory_data = {
            "type": "generic_model",
            "model_type": model_name,
            "data": model.model_dump(mode="json")
        }
        memory_text = json.dumps(memory_data, ensure_ascii=False)
        
        primary_key_field = self._get_primary_key_field(model_type)
        object_id = getattr(model, primary_key_field, None)
        
        metadata = {
            "category": f"upsonic_generic_{model_name.lower()}",
            "namespace": self.namespace,
            "model_type": model_name,
            "object_id": str(object_id) if object_id else None,
            "created_at": getattr(model, 'created_at', time.time()),
            "updated_at": getattr(model, 'updated_at', time.time()),
        }
        
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return memory_text, metadata
    
    def _deserialize_generic_model(self, memory_dict: Dict[str, Any], model_type: Type[T]) -> T:
        memory_text = memory_dict.get("memory", "{}")
        try:
            memory_data = json.loads(memory_text) if isinstance(memory_text, str) else {}
        except json.JSONDecodeError:
            memory_data = {}
        
        model_data = memory_data.get("data", {})
        
        return model_type.model_validate(model_data)



    async def _resolve_memory_id(
        self, 
        object_id: str, 
        model_type: Type[BaseModel]
    ) -> Optional[str]:
        cached_id = self._get_cached_memory_id(object_id)
        if cached_id:
            return cached_id
        
        try:
            import asyncio
            client = await self._get_client()
            
            if model_type.__name__ == "AgentSession":
                composite_user_id = f"{self.namespace}:session:{object_id}"
                target_category = self.CATEGORY_SESSION
            else:
                model_name = model_type.__name__.lower()
                composite_user_id = f"{self.namespace}:model:{model_name}:{object_id}"
                target_category = f"upsonic_generic_{model_name}"
            
            try:
                if self._is_platform_client:
                    # Platform API requires filters parameter
                    # Use AND filter with user_id
                    filters = {"AND": [{"user_id": composite_user_id}]}
                    results = await asyncio.wait_for(
                        client.get_all(filters=filters),
                        timeout=10.0
                    )
                else:
                    # Open Source API uses user_id directly
                    results = await asyncio.wait_for(
                        self._call_get_all(client, user_id=composite_user_id),
                        timeout=10.0
                    )
            except asyncio.TimeoutError:
                return None
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Mem0 get_all failed in _resolve_memory_id: {e}", "Mem0Storage")
                return None
            
            if isinstance(results, list):
                memories = results
            elif isinstance(results, dict):
                memories = results.get("results", results.get("memories", []))
            else:
                return None
            
            # Filter by category
            filtered_memories = [
                m for m in memories 
                if m.get("metadata", {}).get("category") == target_category
            ]
            
            if filtered_memories:
                memory = max(filtered_memories, key=lambda m: m.get("updated_at", 0) or m.get("created_at", 0))
                memory_id = memory.get("id")
                if memory_id:
                    self._cache_memory_id(object_id, memory_id)
                    return memory_id
            
            return None
            
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Failed to resolve memory ID for {object_id}: {e}", "Mem0Storage")
            return None


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
        if not self._connected:
            return False
        
        try:
            if self._is_platform_client:
                return True
            else:
                return self._client is not None
        except Exception:
            self._connected = False
            return False

    async def connect_async(self) -> None:
        if self._connected and await self.is_connected_async():
            return
        
        try:
            _ = await self._get_client()
            self._connected = True
            
            from upsonic.utils.printing import info_log
            mode = "Platform" if self._is_platform_client else "Open Source"
            info_log(f"Connected to Mem0 ({mode} mode)", "Mem0Storage")
            
        except Exception as e:
            self._connected = False
            from upsonic.utils.printing import error_log
            error_log(f"Failed to connect to Mem0: {e}", "Mem0Storage")
            raise ConnectionError(f"Failed to connect to Mem0: {e}") from e

    async def disconnect_async(self) -> None:
        self._connected = False
        self._clear_all_cache()
        
        from upsonic.utils.printing import info_log
        info_log("Disconnected from Mem0", "Mem0Storage")

    async def create_async(self) -> None:
        await self._ensure_connection()

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        try:
            import asyncio
            await self._ensure_connection()
            
            if model_type.__name__ == "AgentSession":
                composite_user_id = f"{self.namespace}:session:{object_id}"
                target_category = self.CATEGORY_SESSION
            else:
                model_name = model_type.__name__.lower()
                composite_user_id = f"{self.namespace}:model:{model_name}:{object_id}"
                target_category = f"upsonic_generic_{model_name}"
            
            client = await self._get_client()
            
            # Try to get memories using the appropriate API
            try:
                if self._is_platform_client:
                    # Platform API requires filters parameter
                    filters = {"AND": [{"user_id": composite_user_id}]}
                    results = await asyncio.wait_for(
                        client.get_all(filters=filters),
                        timeout=10.0
                    )
                else:
                    # Open Source API uses user_id directly
                    results = await asyncio.wait_for(
                        self._call_get_all(client, user_id=composite_user_id),
                        timeout=10.0
                    )
            except asyncio.TimeoutError:
                from upsonic.utils.printing import warning_log
                warning_log(f"Mem0 get_all timeout for {object_id}", "Mem0Storage")
                return None
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Mem0 get_all failed: {e}", "Mem0Storage")
                return None
            
            if isinstance(results, list):
                memories = results
            elif isinstance(results, dict):
                memories = results.get("results", results.get("memories", []))
            else:
                return None
            
            if not memories:
                return None
            
            # Filter by category
            filtered_memories = [
                m for m in memories 
                if m.get("metadata", {}).get("category") == target_category
            ]
            
            if not filtered_memories:
                return None
            
            memory_dict = max(filtered_memories, key=lambda m: m.get("updated_at", 0) or m.get("created_at", 0))
            
            memory_id = memory_dict.get("id")
            if memory_id:
                self._cache_memory_id(object_id, memory_id)
            
            if model_type.__name__ == "AgentSession":
                return self._deserialize_session(memory_dict)
            else:
                return self._deserialize_generic_model(memory_dict, model_type)
            
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Failed to read from Mem0 (id={object_id}): {e}", "Mem0Storage")
            return None

    async def upsert_async(self, data: BaseModel) -> None:
        try:
            await self._ensure_connection()
            if hasattr(data, 'updated_at'):
                data.updated_at = time.time()
            
            if type(data).__name__ == "AgentSession":
                memory_text, metadata = self._serialize_session(data)
                object_id = data.session_id
                model_type = AgentSession
                
                user_id = f"{self.namespace}:session:{data.session_id}"
                agent_id = None
                
            else:
                memory_text, metadata = self._serialize_generic_model(data)
                model_type = type(data)
                
                primary_key_field = self._get_primary_key_field(model_type)
                object_id = getattr(data, primary_key_field)
                
                model_name = model_type.__name__.lower()
                user_id = f"{self.namespace}:model:{model_name}:{object_id}"
                agent_id = None
            
            memory_id = await self._resolve_memory_id(object_id, model_type)
            
            if memory_id:
                update_params = {
                    "memory_id": memory_id,
                    "text": memory_text,
                }
                
                if self._is_platform_client:
                    update_params["metadata"] = metadata
                
                client = await self._get_client()
                await self._call_update(client, **update_params)
                
                from upsonic.utils.printing import info_log
                info_log(f"Updated memory: {object_id}", "Mem0Storage")
                
            else:
                messages = [{"role": "user", "content": memory_text}]
                
                add_params = {
                    "messages": messages,
                    "metadata": metadata,
                    "infer": self.infer,
                }
                
                if user_id:
                    add_params["user_id"] = user_id
                if agent_id:
                    add_params["agent_id"] = agent_id
                
                
                client = await self._get_client()
                result = await self._call_add(client, **add_params)
                
                new_memory_id = None
                if isinstance(result, dict):
                    new_memory_id = result.get("id")
                    if not new_memory_id:
                        results_array = result.get("results", [])
                        if results_array and len(results_array) > 0:
                            new_memory_id = results_array[0].get("id")
                elif isinstance(result, list) and len(result) > 0:
                    new_memory_id = result[0].get("id")
                
                if new_memory_id:
                    self._cache_memory_id(object_id, new_memory_id)
                    # Also store in metadata for easier retrieval
                    if self._is_platform_client:
                        # Update metadata with memory_id for direct lookup
                        try:
                            await self._call_update(
                                client,
                                memory_id=new_memory_id,
                                metadata={**metadata, "memory_id": new_memory_id, "object_id": object_id}
                            )
                        except Exception:
                            pass  # Ignore update errors
                
                from upsonic.utils.printing import info_log
                info_log(f"Added memory: {object_id}", "Mem0Storage")
                
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to upsert to Mem0: {e}", "Mem0Storage")
            raise

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        try:
            await self._ensure_connection()
            memory_id = await self._resolve_memory_id(object_id, model_type)
            
            if memory_id:
                client = await self._get_client()
                await client.delete(memory_id=memory_id)
                
                self._clear_cache_entry(object_id)
                
                from upsonic.utils.printing import info_log
                info_log(f"Deleted memory: {object_id}", "Mem0Storage")
            else:
                from upsonic.utils.printing import warning_log
                warning_log(f"Memory not found for deletion: {object_id}", "Mem0Storage")
                
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to delete from Mem0 (id={object_id}): {e}", "Mem0Storage")
            raise

    async def drop_async(self) -> None:
        try:
            import asyncio
            from upsonic.utils.printing import warning_log, info_log
            warning_log("Dropping ALL memories in namespace", "Mem0Storage")
            
            # Get all cached memory IDs before clearing cache
            cached_memory_ids = list(self._id_to_memory_id.values()) if self.enable_caching else []
            
            self._clear_all_cache()
            
            if self._is_platform_client:
                client = await self._get_client()
                deleted_count = 0
                
                # Delete all cached memories by ID
                for memory_id in cached_memory_ids:
                    try:
                        await client.delete(memory_id=memory_id)
                        deleted_count += 1
                    except Exception:
                        pass
                
                info_log(f"Deleted {deleted_count} cached memories from Platform", "Mem0Storage")
                
            else:
                client = await self._get_client()
                if hasattr(client, 'reset'):
                    await client.reset()
                    info_log("Reset Mem0 Open Source instance", "Mem0Storage")
                else:
                    warning_log("Reset not supported in this Mem0 version", "Mem0Storage")
                    
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to drop Mem0 storage: {e}", "Mem0Storage")
            # Don't raise - drop is best-effort

    # =========================================================================
    # Cultural Knowledge Methods
    # =========================================================================
    
    def _serialize_cultural_knowledge(self, knowledge: "CulturalKnowledge") -> tuple[str, Dict[str, Any]]:
        memory_data = {
            "type": "cultural_knowledge",
            "id": knowledge.id,
            "name": knowledge.name,
            "content": knowledge.content,
            "summary": knowledge.summary,
            "categories": knowledge.categories,
            "notes": knowledge.notes,
            "input": knowledge.input,
        }
        memory_text = json.dumps(memory_data, ensure_ascii=False)
        
        metadata = {
            "category": self.CATEGORY_CULTURE,
            "namespace": self.namespace,
            "knowledge_id": knowledge.id,
            "name": knowledge.name,
            "metadata": json.dumps(knowledge.metadata) if knowledge.metadata else None,
            "agent_id": knowledge.agent_id,
            "team_id": knowledge.team_id,
            "created_at": knowledge.created_at,
            "updated_at": knowledge.updated_at,
        }
        
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return memory_text, metadata
    
    def _deserialize_cultural_knowledge(self, memory_dict: Dict[str, Any]) -> "CulturalKnowledge":
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        memory_text = memory_dict.get("memory", "{}")
        try:
            memory_data = json.loads(memory_text) if isinstance(memory_text, str) else {}
        except json.JSONDecodeError:
            memory_data = {}
        
        metadata = memory_dict.get("metadata", {})
        
        extra_metadata = None
        if metadata.get("metadata"):
            try:
                extra_metadata = json.loads(metadata["metadata"])
            except json.JSONDecodeError:
                pass
        
        return CulturalKnowledge(
            id=metadata.get("knowledge_id", memory_data.get("id")),
            name=memory_data.get("name"),
            content=memory_data.get("content"),
            summary=memory_data.get("summary"),
            categories=memory_data.get("categories"),
            notes=memory_data.get("notes"),
            metadata=extra_metadata,
            input=memory_data.get("input"),
            agent_id=metadata.get("agent_id"),
            team_id=metadata.get("team_id"),
            created_at=metadata.get("created_at"),
            updated_at=metadata.get("updated_at"),
        )

    async def read_cultural_knowledge_async(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        try:
            composite_user_id = f"{self.namespace}:culture:{knowledge_id}"
            
            client = await self._get_client()
            results = await self._call_get_all(client, user_id=composite_user_id)
            
            if isinstance(results, list):
                memories = results
            elif isinstance(results, dict):
                memories = results.get("results", [])
            else:
                return None
            
            if not memories:
                return None
            
            filtered_memories = [
                m for m in memories 
                if m.get("metadata", {}).get("category") == self.CATEGORY_CULTURE
            ]
            
            if not filtered_memories:
                return None
            
            memory_dict = max(filtered_memories, key=lambda m: m.get("updated_at", 0))
            
            memory_id = memory_dict.get("id")
            if memory_id:
                self._cache_memory_id(knowledge_id, memory_id)
            
            return self._deserialize_cultural_knowledge(memory_dict)
            
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Failed to read cultural knowledge from Mem0 (id={knowledge_id}): {e}", "Mem0Storage")
            return None

    async def _resolve_culture_memory_id(self, knowledge_id: str) -> Optional[str]:
        cached_id = self._get_cached_memory_id(knowledge_id)
        if cached_id:
            return cached_id
        
        try:
            composite_user_id = f"{self.namespace}:culture:{knowledge_id}"
            
            client = await self._get_client()
            results = await self._call_get_all(client, user_id=composite_user_id)
            
            if isinstance(results, list):
                memories = results
            elif isinstance(results, dict):
                memories = results.get("results", [])
            else:
                return None
            
            filtered_memories = [
                m for m in memories 
                if m.get("metadata", {}).get("category") == self.CATEGORY_CULTURE
            ]
            
            if not filtered_memories:
                return None
            
            memory_dict = max(filtered_memories, key=lambda m: m.get("updated_at", 0))
            memory_id = memory_dict.get("id")
            
            if memory_id:
                self._cache_memory_id(knowledge_id, memory_id)
            
            return memory_id
            
        except Exception:
            return None

    async def upsert_cultural_knowledge_async(self, knowledge: "CulturalKnowledge") -> None:
        try:
            knowledge.bump_updated_at()
            
            memory_text, metadata = self._serialize_cultural_knowledge(knowledge)
            composite_user_id = f"{self.namespace}:culture:{knowledge.id}"
            
            memory_id = await self._resolve_culture_memory_id(knowledge.id)
            
            client = await self._get_client()
            
            if memory_id:
                update_params = {
                    "memory_id": memory_id,
                    "text": memory_text,
                }
                
                if self._is_platform_client:
                    update_params["metadata"] = metadata
                
                await client.update(**update_params)
                
                from upsonic.utils.printing import info_log
                info_log(f"Updated cultural knowledge: {knowledge.id}", "Mem0Storage")
                
            else:
                messages = [{"role": "user", "content": memory_text}]
                
                add_params = {
                    "messages": messages,
                    "metadata": metadata,
                    "infer": self.infer,
                    "user_id": composite_user_id,
                }
                
                result = await client.add(**add_params)
                
                new_memory_id = None
                if isinstance(result, dict):
                    new_memory_id = result.get("id")
                    if not new_memory_id:
                        results_array = result.get("results", [])
                        if results_array and len(results_array) > 0:
                            new_memory_id = results_array[0].get("id")
                elif isinstance(result, list) and len(result) > 0:
                    new_memory_id = result[0].get("id")
                
                if new_memory_id:
                    self._cache_memory_id(knowledge.id, new_memory_id)
                
                from upsonic.utils.printing import info_log
                info_log(f"Added cultural knowledge: {knowledge.id}", "Mem0Storage")
                
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to upsert cultural knowledge to Mem0: {e}", "Mem0Storage")
            raise

    async def delete_cultural_knowledge_async(self, knowledge_id: str) -> None:
        try:
            memory_id = await self._resolve_culture_memory_id(knowledge_id)
            
            if memory_id:
                client = await self._get_client()
                await client.delete(memory_id=memory_id)
                
                self._clear_cache_entry(knowledge_id)
                
                from upsonic.utils.printing import info_log
                info_log(f"Deleted cultural knowledge: {knowledge_id}", "Mem0Storage")
            else:
                from upsonic.utils.printing import warning_log
                warning_log(f"Cultural knowledge not found for deletion: {knowledge_id}", "Mem0Storage")
                
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to delete cultural knowledge from Mem0: {e}", "Mem0Storage")
            raise

    async def list_all_cultural_knowledge_async(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        try:
            client = await self._get_client()
            
            all_results = await self._call_get_all(client)
            
            if isinstance(all_results, dict):
                all_memories = all_results.get("results", [])
            elif isinstance(all_results, list):
                all_memories = all_results
            else:
                all_memories = []
            
            filtered_memories = [
                m for m in all_memories
                if m.get("metadata", {}).get("namespace") == self.namespace
                and m.get("metadata", {}).get("category") == self.CATEGORY_CULTURE
            ]
            
            results = []
            for memory_dict in filtered_memories:
                try:
                    knowledge = self._deserialize_cultural_knowledge(memory_dict)
                    
                    if name is not None:
                        if knowledge.name is None:
                            continue
                        if name.lower() not in knowledge.name.lower():
                            continue
                    
                    results.append(knowledge)
                except Exception:
                    continue
            
            return results
            
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Failed to list cultural knowledge from Mem0: {e}", "Mem0Storage")
            return []

    async def clear_cultural_knowledge_async(self) -> None:
        try:
            from upsonic.utils.printing import warning_log
            warning_log("Clearing all cultural knowledge", "Mem0Storage")
            
            client = await self._get_client()
            
            all_results = await self._call_get_all(client)
            
            if isinstance(all_results, dict):
                all_memories = all_results.get("results", [])
            elif isinstance(all_results, list):
                all_memories = all_results
            else:
                all_memories = []
            
            culture_memories = [
                m for m in all_memories
                if m.get("metadata", {}).get("namespace") == self.namespace
                and m.get("metadata", {}).get("category") == self.CATEGORY_CULTURE
            ]
            
            for memory in culture_memories:
                memory_id = memory.get("id")
                if memory_id:
                    try:
                        await client.delete(memory_id=memory_id)
                        knowledge_id = memory.get("metadata", {}).get("knowledge_id")
                        if knowledge_id:
                            self._clear_cache_entry(knowledge_id)
                    except Exception as e:
                        warning_log(f"Failed to delete cultural knowledge {memory_id}: {e}", "Mem0Storage")
            
            from upsonic.utils.printing import info_log
            info_log(f"Cleared {len(culture_memories)} cultural knowledge entries", "Mem0Storage")
            
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to clear cultural knowledge from Mem0: {e}", "Mem0Storage")
            raise
