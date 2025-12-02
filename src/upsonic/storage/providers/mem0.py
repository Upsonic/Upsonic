from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import InteractionSession, UserProfile
from upsonic.storage.types import SessionId, UserId

try:
    from mem0 import AsyncMemoryClient, Memory
    HAS_ASYNC_CLIENT = True
    _MEM0_AVAILABLE = True
except ImportError:
    try:
        # Fallback: Try regular imports if async not available
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
    - Uses InteractionSession/UserProfile categories when needed
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    # Category constants for type discrimination
    CATEGORY_SESSION = "upsonic_interaction_session"
    CATEGORY_PROFILE = "upsonic_user_profile"

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
        
        # Internal caches for performance optimization
        self._id_to_memory_id: Dict[str, str] = {}  # object_id → memory_id
        self._cache_timestamps: Dict[str, float] = {}  # object_id → timestamp
        
        # Store client and track ownership for lifecycle management
        self._client: Optional[Union[Memory, AsyncMemoryClient]] = client
        self._owns_client = (client is None)  # True if we create it, False if user provided
        self._is_platform_client = False
        
        # Detect if user provided a platform client
        if client and hasattr(client, 'api_key'):
            self._is_platform_client = True
        
        # Store initialization parameters (only used if no client provided)
        self._api_key = api_key
        self._org_id = org_id
        self._project_id = project_id
        self._local_config = local_config
        self._custom_categories = custom_categories or []
        
        # Add our categories to custom categories
        all_categories = [self.CATEGORY_SESSION, self.CATEGORY_PROFILE]
        if self._custom_categories:
            all_categories.extend(self._custom_categories)
        self._all_categories = list(set(all_categories))

    async def _initialize_client(self) -> Union[Memory, AsyncMemoryClient]:
        """
        Initialize the appropriate Mem0 client based on configuration.
        
        Returns:
            Initialized Memory or AsyncMemoryClient instance.
            
        Raises:
            ConnectionError: If client initialization fails.
        """
        try:
            if self._api_key:
                # Platform mode: Use AsyncMemoryClient
                from upsonic.utils.printing import info_log
                info_log("Initializing Mem0 Platform async client", "Mem0Storage")
                
                client_kwargs = {"api_key": self._api_key}
                if self._org_id:
                    client_kwargs["org_id"] = self._org_id
                if self._project_id:
                    client_kwargs["project_id"] = self._project_id
                    
                client = AsyncMemoryClient(**client_kwargs)
                self._is_platform_client = True
                
                # Configure custom categories for the project
                if self._all_categories:
                    try:
                        await client.update_project(custom_categories=self._all_categories)
                    except Exception as e:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Could not update project categories: {e}", "Mem0Storage")
                
                return client
                
            elif self._local_config:
                # Open source mode with config
                from upsonic.utils.printing import info_log
                info_log("Initializing Mem0 Open Source client with config", "Mem0Storage")
                return Memory.from_config(self._local_config)
                
            else:
                # Open source mode with defaults
                from upsonic.utils.printing import info_log
                info_log("Initializing Mem0 Open Source client with defaults", "Mem0Storage")
                return Memory()
                
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to initialize Mem0 client: {e}", "Mem0Storage")
            raise ConnectionError(f"Failed to initialize Mem0 client: {e}") from e

    async def _get_client(self) -> Union[Memory, AsyncMemoryClient]:
        """
        Lazy-loaded Mem0 client with async initialization.
        If user provided a client, returns that. Otherwise, creates one.
        """
        if self._client is None:
            # We own the client, so we need to create it
            self._client = await self._initialize_client()
        return self._client



    def _cache_memory_id(self, object_id: str, memory_id: str) -> None:
        """Cache a memory ID for fast future lookups."""
        if self.enable_caching:
            self._id_to_memory_id[object_id] = memory_id
            self._cache_timestamps[object_id] = time.time()

    def _get_cached_memory_id(self, object_id: str) -> Optional[str]:
        """Retrieve cached memory ID if valid."""
        if not self.enable_caching:
            return None
            
        memory_id = self._id_to_memory_id.get(object_id)
        if memory_id is None:
            return None
            
        # Check TTL
        if self.cache_ttl > 0:
            cached_at = self._cache_timestamps.get(object_id, 0)
            if time.time() - cached_at > self.cache_ttl:
                # Cache expired
                self._clear_cache_entry(object_id)
                return None
                
        return memory_id

    def _clear_cache_entry(self, object_id: str) -> None:
        """Clear a single cache entry."""
        self._id_to_memory_id.pop(object_id, None)
        self._cache_timestamps.pop(object_id, None)

    def _clear_all_cache(self) -> None:
        """Clear all cache entries."""
        self._id_to_memory_id.clear()
        self._cache_timestamps.clear()



    def _serialize_session(self, session: InteractionSession) -> tuple[str, Dict[str, Any]]:
        """
        Serialize InteractionSession into Mem0 memory text and metadata.
        
        Returns:
            Tuple of (memory_text, metadata_dict)
        """
        # Store chat history and summary in the memory text
        memory_data = {
            "type": "interaction_session",
            "session_id": session.session_id,
            "chat_history": session.chat_history,
            "summary": session.summary,
        }
        memory_text = json.dumps(memory_data, ensure_ascii=False)
        
        # Store all other fields in metadata
        metadata = {
            "category": self.CATEGORY_SESSION,
            "namespace": self.namespace,
            "session_id": session.session_id,
            "user_id": session.user_id,
            "agent_id": session.agent_id,
            "team_session_id": session.team_session_id,
            "session_data": json.dumps(session.session_data),
            "extra_data": json.dumps(session.extra_data),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }
        
        # Remove None values to avoid Mem0 issues
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return memory_text, metadata

    def _deserialize_session(self, memory_dict: Dict[str, Any]) -> InteractionSession:
        """
        Deserialize Mem0 memory back into InteractionSession.
        
        Args:
            memory_dict: Memory dict from Mem0 (contains 'memory' and 'metadata')
            
        Returns:
            Reconstructed InteractionSession instance.
        """
        # Parse memory text
        memory_text = memory_dict.get("memory", "{}")
        try:
            memory_data = json.loads(memory_text) if isinstance(memory_text, str) else {}
        except json.JSONDecodeError:
            memory_data = {}
        
        # Get metadata
        metadata = memory_dict.get("metadata", {})
        
        # Reconstruct session
        return InteractionSession(
            session_id=SessionId(metadata.get("session_id", memory_data.get("session_id", ""))),
            user_id=UserId(metadata["user_id"]) if metadata.get("user_id") else None,
            agent_id=metadata.get("agent_id"),
            team_session_id=metadata.get("team_session_id"),
            chat_history=memory_data.get("chat_history", []),
            summary=memory_data.get("summary"),
            session_data=json.loads(metadata["session_data"]) if metadata.get("session_data") else {},
            extra_data=json.loads(metadata["extra_data"]) if metadata.get("extra_data") else {},
            created_at=metadata.get("created_at", time.time()),
            updated_at=metadata.get("updated_at", time.time()),
        )

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
    
    def _serialize_generic_model(self, model: BaseModel) -> tuple[str, Dict[str, Any]]:
        """
        Serialize a generic Pydantic model into Mem0 memory text and metadata.
        
        Args:
            model: Any Pydantic BaseModel instance
            
        Returns:
            Tuple of (memory_text, metadata_dict)
        """
        model_type = type(model)
        model_name = model_type.__name__
        
        # Store entire model as JSON in memory text
        memory_data = {
            "type": "generic_model",
            "model_type": model_name,
            "data": model.model_dump(mode="json")
        }
        memory_text = json.dumps(memory_data, ensure_ascii=False)
        
        # Get primary key for lookup
        primary_key_field = self._get_primary_key_field(model_type)
        object_id = getattr(model, primary_key_field, None)
        
        # Store metadata
        metadata = {
            "category": f"upsonic_generic_{model_name.lower()}",
            "namespace": self.namespace,
            "model_type": model_name,
            "object_id": str(object_id) if object_id else None,
            "created_at": getattr(model, 'created_at', time.time()),
            "updated_at": getattr(model, 'updated_at', time.time()),
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return memory_text, metadata
    
    def _deserialize_generic_model(self, memory_dict: Dict[str, Any], model_type: Type[T]) -> T:
        """
        Deserialize a generic model from Mem0 memory dict.
        
        Args:
            memory_dict: Mem0 memory dictionary
            model_type: Pydantic model class to deserialize into
            
        Returns:
            Reconstructed model instance
        """
        # Parse memory text
        memory_text = memory_dict.get("memory", "{}")
        try:
            memory_data = json.loads(memory_text) if isinstance(memory_text, str) else {}
        except json.JSONDecodeError:
            memory_data = {}
        
        # Extract model data
        model_data = memory_data.get("data", {})
        
        # Reconstruct model
        return model_type.model_validate(model_data)

    def _serialize_profile(self, profile: UserProfile) -> tuple[str, Dict[str, Any]]:
        """
        Serialize UserProfile into Mem0 memory text and metadata.
        
        Returns:
            Tuple of (memory_text, metadata_dict)
        """
        # Store profile data in memory text
        memory_data = {
            "type": "user_profile",
            "user_id": profile.user_id,
            "profile_data": profile.profile_data,
        }
        memory_text = json.dumps(memory_data, ensure_ascii=False)
        
        # Store metadata
        metadata = {
            "category": self.CATEGORY_PROFILE,
            "namespace": self.namespace,
            "user_id": profile.user_id,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return memory_text, metadata

    def _deserialize_profile(self, memory_dict: Dict[str, Any]) -> UserProfile:
        """
        Deserialize Mem0 memory back into UserProfile.
        
        Args:
            memory_dict: Memory dict from Mem0 (contains 'memory' and 'metadata')
            
        Returns:
            Reconstructed UserProfile instance.
        """
        # Parse memory text
        memory_text = memory_dict.get("memory", "{}")
        try:
            memory_data = json.loads(memory_text) if isinstance(memory_text, str) else {}
        except json.JSONDecodeError:
            memory_data = {}
        
        # Get metadata
        metadata = memory_dict.get("metadata", {})
        
        # Reconstruct profile
        return UserProfile(
            user_id=UserId(metadata.get("user_id", memory_data.get("user_id", ""))),
            profile_data=memory_data.get("profile_data", {}),
            created_at=metadata.get("created_at", time.time()),
            updated_at=metadata.get("updated_at", time.time()),
        )



    async def _resolve_memory_id(
        self, 
        object_id: str, 
        model_type: Type[BaseModel]
    ) -> Optional[str]:
        """
        Resolve object_id to Mem0 memory_id using cache or get_all.
        
        Args:
            object_id: Session ID, User ID, or generic model object ID
            model_type: InteractionSession, UserProfile, or any Pydantic BaseModel
            
        Returns:
            Mem0 memory_id or None if not found
        """
        # Check cache first
        cached_id = self._get_cached_memory_id(object_id)
        if cached_id:
            return cached_id
        
        # Get from Mem0 using composite user_id
        try:
            client = await self._get_client()
            
            if model_type is InteractionSession:
                composite_user_id = f"{self.namespace}:session:{object_id}"
            elif model_type is UserProfile:
                composite_user_id = f"{self.namespace}:profile:{object_id}"
            else:
                # Generic model support
                model_name = model_type.__name__.lower()
                composite_user_id = f"{self.namespace}:model:{model_name}:{object_id}"
            
            results = await client.get_all(user_id=composite_user_id)
            
            # Parse results - get_all() returns a list directly
            if isinstance(results, list):
                memories = results
            elif isinstance(results, dict):
                memories = results.get("results", [])
            else:
                return None
            
            # Filter by category to ensure correct type
            target_category = self.CATEGORY_SESSION if model_type is InteractionSession else self.CATEGORY_PROFILE
            filtered_memories = [
                m for m in memories 
                if m.get("metadata", {}).get("category") == target_category
            ]
            
            if filtered_memories:
                # Get most recent
                memory = max(filtered_memories, key=lambda m: m.get("updated_at", 0))
                memory_id = memory.get("id")
                if memory_id:
                    # Cache it
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

    def upsert(self, data: Union[InteractionSession, UserProfile]) -> None:
        return self._run_async_from_sync(self.upsert_async(data))

    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None:
        return self._run_async_from_sync(self.delete_async(object_id, model_type))

    def drop(self) -> None:
        return self._run_async_from_sync(self.drop_async())



    async def is_connected_async(self) -> bool:
        """
        Check if the Mem0 client is connected and operational.
        
        Returns:
            True if connected, False otherwise.
        """
        if not self._connected:
            return False
        
        try:
            # Test connection with a simple query
            if self._is_platform_client:
                # Platform client is stateless, always connected if initialized
                return True
            else:
                # For OSS, we assume it's connected if initialized
                return self._client is not None
        except Exception:
            self._connected = False
            return False

    async def connect_async(self) -> None:
        """
        Initialize connection to Mem0.
        
        Raises:
            ConnectionError: If connection fails.
        """
        if self._connected and await self.is_connected_async():
            return
        
        try:
            # Initialize client (lazy loading)
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
        """
        Disconnect from Mem0 and cleanup resources.
        """
        self._connected = False
        self._clear_all_cache()
        
        # No explicit disconnect needed for Mem0 clients
        from upsonic.utils.printing import info_log
        info_log("Disconnected from Mem0", "Mem0Storage")

    async def create_async(self) -> None:
        """
        Initialize Mem0 storage (ensure connection).
        """
        await self.connect_async()

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        """
        Read an object from Mem0 using native Mem0 fields (user_id, agent_id).
        
        For sessions: user_id = namespace:session:session_id
        For profiles: user_id = namespace:profile:user_id
        For generic models: user_id = namespace:model:model_type:object_id
        
        Args:
            object_id: Session ID, User ID, or generic model object ID
            model_type: InteractionSession, UserProfile, or any Pydantic BaseModel class
            
        Returns:
            The requested object or None if not found.
        """
        try:
            # Create composite user_id for lookup
            if model_type is InteractionSession:
                composite_user_id = f"{self.namespace}:session:{object_id}"
                target_category = self.CATEGORY_SESSION
            elif model_type is UserProfile:
                composite_user_id = f"{self.namespace}:profile:{object_id}"
                target_category = self.CATEGORY_PROFILE
            else:
                # Generic model support
                model_name = model_type.__name__.lower()
                composite_user_id = f"{self.namespace}:model:{model_name}:{object_id}"
                target_category = f"upsonic_generic_{model_name}"
            
            client = await self._get_client()
            # Use get_all with user_id to retrieve memories
            # get_all() with user_id parameter should return all memories for that user_id
            results = await client.get_all(user_id=composite_user_id)
            
            # Parse results - get_all() returns a list directly, not a dict
            if isinstance(results, list):
                memories = results
            elif isinstance(results, dict):
                memories = results.get("results", [])
            else:
                return None
            
            if not memories:
                return None
            
            # Filter by category in metadata to ensure we get the right type
            filtered_memories = [
                m for m in memories 
                if m.get("metadata", {}).get("category") == target_category
            ]
            
            if not filtered_memories:
                return None
            
            # Get the most recent memory (sorted by updated_at)
            memory_dict = max(filtered_memories, key=lambda m: m.get("updated_at", 0))
            
            # Cache the memory ID
            memory_id = memory_dict.get("id")
            if memory_id:
                self._cache_memory_id(object_id, memory_id)
            
            # Deserialize based on type
            if model_type is InteractionSession:
                return self._deserialize_session(memory_dict)
            elif model_type is UserProfile:
                return self._deserialize_profile(memory_dict)
            else:
                # Generic model support
                return self._deserialize_generic_model(memory_dict, model_type)
            
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Failed to read from Mem0 (id={object_id}): {e}", "Mem0Storage")
            return None

    async def upsert_async(self, data: BaseModel) -> None:
        """
        Insert or update an object in Mem0.
        
        Args:
            data: InteractionSession, UserProfile, or any Pydantic BaseModel instance to store.
            
        Raises:
            TypeError: If unsupported data type is provided.
        """
        try:
            if hasattr(data, 'updated_at'):
                data.updated_at = time.time()
            
            if isinstance(data, InteractionSession):
                memory_text, metadata = self._serialize_session(data)
                object_id = data.session_id
                model_type = InteractionSession
                
                # For sessions, create a composite user_id that includes the session
                # Format: namespace:session:session_id
                user_id = f"{self.namespace}:session:{data.session_id}"
                agent_id = None  # Don't use agent_id to avoid complications
                
            elif isinstance(data, UserProfile):
                memory_text, metadata = self._serialize_profile(data)
                object_id = data.user_id
                model_type = UserProfile
                
                # For profiles, create a composite user_id
                # Format: namespace:profile:user_id
                user_id = f"{self.namespace}:profile:{data.user_id}"
                agent_id = None
                
            else:
                # Generic Pydantic model support
                memory_text, metadata = self._serialize_generic_model(data)
                model_type = type(data)
                
                # Get object ID from model
                primary_key_field = self._get_primary_key_field(model_type)
                object_id = getattr(data, primary_key_field)
                
                # For generic models, create composite user_id
                # Format: namespace:model:model_type:object_id
                model_name = model_type.__name__.lower()
                user_id = f"{self.namespace}:model:{model_name}:{object_id}"
                agent_id = None
            
            # Check if object already exists
            memory_id = await self._resolve_memory_id(object_id, model_type)
            
            if memory_id:
                # UPDATE existing memory
                update_params = {
                    "memory_id": memory_id,
                    "text": memory_text,  # Use 'text' not 'data'
                }
                
                # Platform-specific metadata update
                if self._is_platform_client:
                    update_params["metadata"] = metadata
                
                client = await self._get_client()
                await client.update(**update_params)
                
                from upsonic.utils.printing import info_log
                info_log(f"Updated memory: {object_id}", "Mem0Storage")
                
            else:
                # ADD new memory
                # Format as conversation for Mem0
                # The memory_text contains serialized session data as JSON.
                # Since infer=False, Mem0 won't process this semantically - it's pure storage.
                messages = [{"role": "user", "content": memory_text}]
                
                add_params = {
                    "messages": messages,
                    "metadata": metadata,
                    "infer": self.infer,  # Disable inference for structured storage
                }
                
                # Add native Mem0 fields if available
                if user_id:
                    add_params["user_id"] = user_id
                if agent_id:
                    add_params["agent_id"] = agent_id
                
                
                client = await self._get_client()
                result = await client.add(**add_params)
                
                # Parse result to get memory ID
                new_memory_id = None
                if isinstance(result, dict):
                    # Try direct id field first
                    new_memory_id = result.get("id")
                    # If not found, try results array
                    if not new_memory_id:
                        results_array = result.get("results", [])
                        if results_array and len(results_array) > 0:
                            new_memory_id = results_array[0].get("id")
                elif isinstance(result, list) and len(result) > 0:
                    new_memory_id = result[0].get("id")
                
                if new_memory_id:
                    self._cache_memory_id(object_id, new_memory_id)
                
                from upsonic.utils.printing import info_log
                info_log(f"Added memory: {object_id}", "Mem0Storage")
                
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to upsert to Mem0: {e}", "Mem0Storage")
            raise

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        """
        Delete an object from Mem0.
        
        Args:
            object_id: Session ID or User ID to delete
            model_type: InteractionSession or UserProfile class
        """
        try:
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
        """
        Delete ALL memories associated with this namespace.
        
        WARNING: This is a destructive operation that cannot be undone.
        """
        try:
            from upsonic.utils.printing import warning_log
            warning_log("Dropping ALL memories in namespace", "Mem0Storage")
            
            self._clear_all_cache()
            
            # For Platform: Get all memories and delete them
            if self._is_platform_client:
                client = await self._get_client()
                
                # Get all memories for this namespace (we'll filter by category in metadata)
                all_results = await client.get_all()
                
                # Parse results
                if isinstance(all_results, dict):
                    all_memories_raw = all_results.get("results", [])
                elif isinstance(all_results, list):
                    all_memories_raw = all_results
                else:
                    all_memories_raw = []
                
                # Filter by our namespace and categories
                all_memories = [
                    m for m in all_memories_raw
                    if m.get("metadata", {}).get("namespace") == self.namespace
                    and m.get("metadata", {}).get("category") in [self.CATEGORY_SESSION, self.CATEGORY_PROFILE]
                ]
                
                # Delete each memory
                for memory in all_memories:
                    memory_id = memory.get("id")
                    if memory_id:
                        try:
                            await client.delete(memory_id=memory_id)
                        except Exception as e:
                            warning_log(f"Failed to delete memory {memory_id}: {e}", "Mem0Storage")
                
                from upsonic.utils.printing import info_log
                info_log(f"Dropped {len(all_memories)} memories", "Mem0Storage")
                
            else:
                # For OSS: Use reset if available
                client = await self._get_client()
                if hasattr(client, 'reset'):
                    await client.reset()
                    from upsonic.utils.printing import info_log
                    info_log("Reset Mem0 Open Source instance", "Mem0Storage")
                else:
                    from upsonic.utils.printing import warning_log
                    warning_log("Reset not supported in this Mem0 version", "Mem0Storage")
                    
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Failed to drop Mem0 storage: {e}", "Mem0Storage")
            raise

