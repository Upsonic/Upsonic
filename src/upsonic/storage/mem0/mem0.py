"""Mem0 storage implementation for Upsonic agent framework.

This module provides a synchronous Mem0 storage backend that supports both:
- Self-hosted Mem0 (via Memory class from mem0 library)
- Mem0 Platform (via MemoryClient class from mem0 library)

Mem0 stores data as memories with metadata, which we use to simulate
table-like behavior for sessions and user memories.
"""
from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from mem0 import Memory, MemoryClient
    from mem0.configs.base import MemoryConfig
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.session.base import Session, SessionType
    from upsonic.storage.schemas import KnowledgeRow

from upsonic.storage.base import Storage
from upsonic.storage.mem0.utils import (
    apply_pagination,
    build_cultural_knowledge_filters,
    build_knowledge_filters,
    build_session_filters,
    build_user_memory_filters,
    deserialize_cultural_knowledge_from_mem0,
    deserialize_knowledge_from_mem0,
    deserialize_session_from_mem0,
    deserialize_session_to_object,
    deserialize_user_memory_from_mem0,
    generate_cultural_knowledge_memory_id,
    generate_knowledge_memory_id,
    generate_session_memory_id,
    generate_user_memory_id,
    serialize_cultural_knowledge_to_mem0,
    serialize_knowledge_to_mem0,
    serialize_session_to_mem0,
    serialize_user_memory_to_mem0,
    sort_records_by_field,
)
from upsonic.storage.schemas import UserMemory
from upsonic.utils.logging_config import get_logger

_logger = get_logger("upsonic.storage.mem0")


class Mem0Storage(Storage):
    """Synchronous Mem0 storage implementation for Upsonic agent framework.
    
    This storage backend uses the Mem0 library for memory persistence.
    It supports both self-hosted Mem0 (Memory class) and Mem0 Platform
    (MemoryClient class).
    
    Mem0 stores data as memories with content and metadata. This implementation
    uses metadata filtering to simulate table-like behavior for:
    - Sessions (AgentSession, TeamSession, WorkflowSession)
    - User memories
    
    Client Creation Modes:
        1. Pass an existing memory_client (Memory or MemoryClient instance)
        2. Pass api_key to create a MemoryClient (Platform)
        3. Pass config to create a Memory instance (Self-hosted)
    
    Example usage with internal MemoryClient creation (Platform):
        ```python
        storage = Mem0Storage(api_key="your-api-key")
        
        # With additional platform options
        storage = Mem0Storage(
            api_key="your-api-key",
            org_id="your-org-id",
            project_id="your-project-id",
        )
        ```
    
    Example usage with internal Memory creation (Self-hosted):
        ```python
        from mem0.configs.base import MemoryConfig
        
        config = MemoryConfig(...)
        storage = Mem0Storage(config=config)
        ```
    
    Example usage with external client:
        ```python
        from mem0 import Memory, MemoryClient
        
        # With self-hosted
        memory = Memory()
        storage = Mem0Storage(memory_client=memory)
        
        # With platform
        client = MemoryClient(api_key="your-api-key")
        storage = Mem0Storage(memory_client=client)
        ```
    """

    def __init__(
        self,
        memory_client: Optional[Union["Memory", "MemoryClient"]] = None,
        # Platform client (MemoryClient) parameters
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        # Self-hosted (Memory) parameters
        config: Optional["MemoryConfig"] = None,
        # Common parameters
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        cultural_knowledge_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        default_user_id: str = "upsonic_default",
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize the Mem0 storage backend.
        
        The client can be created in three ways (in priority order):
        1. Pass an existing memory_client (Memory or MemoryClient instance)
        2. Pass api_key to create a MemoryClient (Platform)
        3. Pass config to create a Memory instance (Self-hosted)
        
        Args:
            memory_client: Pre-configured Mem0 Memory or MemoryClient instance.
            api_key: Mem0 Platform API key. Creates MemoryClient if provided.
                    Can also use MEM0_API_KEY environment variable.
            host: Mem0 Platform host URL (default: "https://api.mem0.ai").
            org_id: Mem0 Platform organization ID.
            project_id: Mem0 Platform project ID.
            config: MemoryConfig for self-hosted Memory instance.
            session_table: Name of the session table (used in metadata for filtering).
            user_memory_table: Name of the user memory table (used in metadata).
            cultural_knowledge_table: Name of the cultural knowledge table (used in metadata).
            default_user_id: Default user_id for Mem0 operations (required by Mem0).
            id: Unique identifier for this storage instance.
        
        Raises:
            ValueError: If no valid client configuration is provided.
            ImportError: If mem0 library is not installed.
        """
        super().__init__(
            session_table=session_table,
            user_memory_table=user_memory_table,
            cultural_knowledge_table=cultural_knowledge_table,
            knowledge_table=knowledge_table,
            id=id,
        )
        
        # Create or use provided memory client
        self.memory_client: Union["Memory", "MemoryClient"] = self._create_or_use_client(
            memory_client=memory_client,
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
            config=config,
        )
        self.default_user_id: str = default_user_id
        
        # Determine client type for API compatibility
        self._is_platform_client: bool = self._check_is_platform_client()
        
        _logger.debug(
            f"Initialized Mem0Storage with {'platform' if self._is_platform_client else 'self-hosted'} client"
        )

    def _create_or_use_client(
        self,
        memory_client: Optional[Union["Memory", "MemoryClient"]],
        api_key: Optional[str],
        host: Optional[str],
        org_id: Optional[str],
        project_id: Optional[str],
        config: Optional["MemoryConfig"],
    ) -> Union["Memory", "MemoryClient"]:
        """
        Create a new memory client or use the provided one.
        
        Priority order:
        1. Use memory_client if provided
        2. Create MemoryClient if api_key is provided
        3. Create Memory if config is provided
        4. Create Memory with default config
        
        Args:
            memory_client: Existing client instance
            api_key: Platform API key
            host: Platform host URL
            org_id: Platform org ID
            project_id: Platform project ID
            config: Self-hosted Memory config
        
        Returns:
            Memory or MemoryClient instance
        
        Raises:
            ValueError: If configuration is invalid
            ImportError: If mem0 is not installed
        """
        # 1. Use existing client if provided
        if memory_client is not None:
            _logger.debug("Using provided memory client")
            return memory_client
        
        # Try to import mem0
        try:
            from mem0 import Memory, MemoryClient
        except ImportError:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="mem0ai",
                install_command='pip install "upsonic[mem0-storage]"',
                feature_name="Mem0 storage provider"
            )
        
        # 2. Create MemoryClient if api_key provided (or from env)
        effective_api_key = api_key or os.getenv("MEM0_API_KEY")
        if effective_api_key:
            _logger.debug("Creating MemoryClient (platform)")
            client_kwargs: Dict[str, Any] = {"api_key": effective_api_key}
            if host:
                client_kwargs["host"] = host
            if org_id:
                client_kwargs["org_id"] = org_id
            if project_id:
                client_kwargs["project_id"] = project_id
            return MemoryClient(**client_kwargs)
        
        # 3. Create Memory with config if provided
        if config is not None:
            _logger.debug("Creating Memory (self-hosted) with custom config")
            return Memory(config=config)
        
        # 4. Create Memory with default config
        _logger.debug("Creating Memory (self-hosted) with default config")
        return Memory()

    def _check_is_platform_client(self) -> bool:
        """Check if the memory client is a platform client (MemoryClient) or self-hosted (Memory)."""
        client_class_name = self.memory_client.__class__.__name__
        return client_class_name == "MemoryClient"

    # ======================== Table Management ========================

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists (always returns True for Mem0).
        
        Mem0 doesn't have explicit tables - data is organized via metadata.
        This method returns True as tables are "virtual" and always available.
        
        Args:
            table_name: Name of the table to check.
        
        Returns:
            Always True for Mem0 storage.
        """
        return True

    def _create_all_tables(self) -> None:
        """
        Create all required tables (no-op for Mem0).
        
        Mem0 doesn't require explicit table creation. Tables are virtual
        and defined through metadata filtering.
        """
        _logger.debug("Mem0 does not require explicit table creation")

    def close(self) -> None:
        """
        Close the Mem0 client connection.
        
        Note: Most Mem0 clients don't require explicit cleanup.
        """
        _logger.debug("Mem0 storage closed")

    # ======================== Session Methods ========================

    def upsert_session(
        self,
        session: "Session",
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Insert or update a session in Mem0.
        
        Args:
            session: The session to upsert. Must have session_id set.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The upserted session (deserialized or dict based on flag).
        
        Raises:
            ValueError: If session_id is not provided.
        """
        if not hasattr(session, "session_id") or session.session_id is None:
            raise ValueError("Session must have session_id set for upsert")
        
        try:
            # Serialize session for Mem0
            serialized = serialize_session_to_mem0(session, self.session_table_name)
            memory_id = serialized["memory_id"]
            chunks = serialized.get("_chunks")
            
            # Check if memory already exists to determine if this is update or insert
            existing = self._get_memory_by_id(memory_id)
            
            # Handle chunked data - store chunks first
            if chunks:
                chunk_ids = serialized["metadata"].get("_chunk_ids", [])
                for i, (chunk_id, chunk_data) in enumerate(zip(chunk_ids, chunks)):
                    chunk_metadata = {
                        "_type": "session_chunk",
                        "_parent_id": memory_id,
                        "_chunk_index": i,
                        "_upsonic_memory_id": chunk_id,
                        "_data": chunk_data,
                    }
                    # Delete existing chunk if it exists
                    self._delete_memory(chunk_id)
                    self._add_memory(
                        memory_id=chunk_id,
                        content=f"Session chunk {i} for {session.session_id}",
                        metadata=chunk_metadata,
                    )
                _logger.debug(f"Stored {len(chunks)} chunks for session: {session.session_id}")
            
            if existing:
                # Update existing memory
                self._update_memory(
                    memory_id=memory_id,
                    content=serialized["content"],
                    metadata=serialized["metadata"],
                )
                _logger.debug(f"Updated session: {session.session_id}")
            else:
                # Add new memory
                self._add_memory(
                    memory_id=memory_id,
                    content=serialized["content"],
                    metadata=serialized["metadata"],
                )
                _logger.debug(f"Added session: {session.session_id}")
            
            # Retrieve and return the stored session
            stored = self._get_memory_by_id(memory_id)
            if stored is None:
                return None
            
            # If chunked, retrieve and reassemble chunks
            if stored.get("metadata", {}).get("_chunked"):
                stored = self._reassemble_chunked_session(stored)
            
            session_dict = deserialize_session_from_mem0(stored)
            
            if not deserialize:
                return session_dict
            
            return deserialize_session_to_object(session_dict)
        
        except Exception as e:
            _logger.error(f"Error upserting session: {e}")
            raise e
    
    def _reassemble_chunked_session(self, stored: Dict[str, Any]) -> Dict[str, Any]:
        """Reassemble a chunked session by retrieving and combining all chunks."""
        metadata = stored.get("metadata", {})
        chunk_ids = metadata.get("_chunk_ids", [])
        
        if not chunk_ids:
            return stored
        
        # Retrieve all chunks
        chunks_data = []
        for chunk_id in chunk_ids:
            chunk = self._get_memory_by_id(chunk_id)
            if chunk:
                chunk_metadata = chunk.get("metadata", {})
                chunk_data = chunk_metadata.get("_data", "")
                chunk_index = chunk_metadata.get("_chunk_index", 0)
                chunks_data.append((chunk_index, chunk_data))
        
        # Sort by index and combine
        chunks_data.sort(key=lambda x: x[0])
        combined_data = "".join(data for _, data in chunks_data)
        
        # Update stored metadata with reassembled data
        result = stored.copy()
        result["metadata"] = metadata.copy()
        result["metadata"]["_data"] = combined_data
        result["metadata"]["_chunked"] = False  # Mark as reassembled
        
        return result

    def upsert_sessions(
        self,
        sessions: List["Session"],
        deserialize: bool = True,
    ) -> List[Union["Session", Dict[str, Any]]]:
        """
        Bulk insert or update multiple sessions.
        
        Args:
            sessions: List of sessions to upsert. Each must have session_id set.
            deserialize: If True, return deserialized Session objects.
                        If False, return raw dictionaries.
        
        Returns:
            List of upserted sessions.
        
        Raises:
            ValueError: If any session is missing session_id.
        """
        if not sessions:
            return []
        
        # Validate all sessions have session_id
        for session in sessions:
            if not hasattr(session, "session_id") or session.session_id is None:
                raise ValueError("All sessions must have session_id set for upsert")
        
        results: List[Union["Session", Dict[str, Any]]] = []
        
        for session in sessions:
            try:
                result = self.upsert_session(session, deserialize=deserialize)
                if result is not None:
                    results.append(result)
            except Exception as e:
                _logger.error(f"Error upserting session {session.session_id}: {e}")
                raise e
        
        _logger.debug(f"Upserted {len(results)} sessions")
        return results

    def get_session(
        self,
        session_id: Optional[str] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Get a session from Mem0.
        
        Args:
            session_id: ID of the session to retrieve. If None, returns latest session.
            session_type: Type of session (AGENT, TEAM, WORKFLOW).
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The session if found, None otherwise.
        """
        try:
            if session_id is not None:
                # Direct lookup by memory_id
                memory_id = generate_session_memory_id(session_id, self.session_table_name)
                stored = self._get_memory_by_id(memory_id)
                
                if stored is None:
                    return None
                
                # Handle chunked sessions
                if stored.get("metadata", {}).get("_chunked"):
                    stored = self._reassemble_chunked_session(stored)
                
                session_dict = deserialize_session_from_mem0(stored)
                
                if not deserialize:
                    return session_dict
                
                return deserialize_session_to_object(session_dict, session_type)
            
            # Search with filters and return latest
            filters = build_session_filters(
                table_name=self.session_table_name,
                session_type=session_type,
                user_id=user_id,
                agent_id=agent_id,
            )
            
            records = self._search_memories(filters=filters)
            
            if not records:
                return None
            
            # Sort by created_at desc and get first
            sorted_records = sort_records_by_field(records, "created_at", "desc")
            
            if not sorted_records:
                return None
            
            stored = sorted_records[0]
            
            # Handle chunked sessions
            if stored.get("metadata", {}).get("_chunked"):
                stored = self._reassemble_chunked_session(stored)
            
            session_dict = deserialize_session_from_mem0(stored)
            
            if not deserialize:
                return session_dict
            
            return deserialize_session_to_object(session_dict, session_type)
        
        except Exception as e:
            _logger.error(f"Error getting session: {e}")
            raise e

    def get_sessions(
        self,
        session_ids: Optional[List[str]] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["Session"], Tuple[List[Dict[str, Any]], int]]:
        """
        Get multiple sessions from Mem0.
        
        Args:
            session_ids: List of session IDs to retrieve. If None, returns all sessions.
            session_type: Filter by session type.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            sort_by: Field to sort by.
            sort_order: Sort order ('asc' or 'desc').
            deserialize: If True, return deserialized Session objects.
                        If False, return tuple of (raw dictionaries, total count).
        
        Returns:
            List of sessions or tuple of (list of dicts, total count).
        """
        try:
            records: List[Dict[str, Any]] = []
            
            if session_ids is not None and len(session_ids) > 0:
                # Fetch specific sessions by ID
                for session_id in session_ids:
                    memory_id = generate_session_memory_id(session_id, self.session_table_name)
                    stored = self._get_memory_by_id(memory_id)
                    if stored:
                        records.append(stored)
            else:
                # Search with filters
                filters = build_session_filters(
                    table_name=self.session_table_name,
                    session_type=session_type,
                    user_id=user_id,
                    agent_id=agent_id,
                )
                records = self._search_memories(filters=filters)
            
            if not records:
                return [] if deserialize else ([], 0)
            
            # Get total count before pagination
            total_count = len(records)
            
            # Sort records
            sorted_records = sort_records_by_field(
                records,
                sort_by=sort_by or "created_at",
                sort_order=sort_order or "desc",
            )
            
            # Apply pagination
            paginated_records = apply_pagination(sorted_records, limit, offset)
            
            # Handle chunked sessions
            reassembled_records: List[Dict[str, Any]] = []
            for r in paginated_records:
                if r.get("metadata", {}).get("_chunked"):
                    r = self._reassemble_chunked_session(r)
                reassembled_records.append(r)
            
            # Deserialize all records
            session_dicts = [deserialize_session_from_mem0(r) for r in reassembled_records]
            
            if not deserialize:
                return session_dicts, total_count
            
            return [
                deserialize_session_to_object(s, session_type)
                for s in session_dicts
            ]
        
        except Exception as e:
            _logger.error(f"Error getting sessions: {e}")
            raise e

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from Mem0.
        Also deletes any associated chunk memories.
        
        Args:
            session_id: ID of the session to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If session_id is not provided.
        """
        if not session_id:
            raise ValueError("session_id is required for delete")
        
        try:
            memory_id: str = generate_session_memory_id(session_id, self.session_table_name)
            
            # Check for chunks and delete them first
            stored: Optional[Dict[str, Any]] = self._get_memory_by_id(memory_id)
            if stored:
                chunk_ids: List[str] = stored.get("metadata", {}).get("_chunk_ids", [])
                for chunk_id in chunk_ids:
                    self._delete_memory(chunk_id)
            
            return self._delete_memory(memory_id)
        
        except Exception as e:
            _logger.error(f"Error deleting session: {e}")
            raise e

    def delete_sessions(self, session_ids: List[str]) -> int:
        """
        Delete multiple sessions from Mem0.
        
        Args:
            session_ids: List of session IDs to delete.
        
        Returns:
            Number of sessions deleted.
        
        Raises:
            ValueError: If session_ids is empty.
        """
        if not session_ids:
            raise ValueError("session_ids is required and cannot be empty")
        
        deleted_count = 0
        
        for session_id in session_ids:
            try:
                if self.delete_session(session_id):
                    deleted_count += 1
            except Exception as e:
                _logger.warning(f"Error deleting session {session_id}: {e}")
        
        _logger.debug(f"Deleted {deleted_count} sessions")
        return deleted_count

    # ======================== User Memory Methods ========================

    def upsert_user_memory(
        self,
        user_memory: UserMemory,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Insert or update user memory in Mem0.
        
        Args:
            user_memory: The UserMemory instance to store. Must have user_id set.
            deserialize: If True, return full record. If False, return raw dict.
        
        Returns:
            The upserted user memory record.
        
        Raises:
            ValueError: If user_id is not provided in user_memory.
        """
        if not user_memory.user_id:
            raise ValueError("user_memory must have user_id set for upsert")
        
        try:
            # Serialize user memory for Mem0
            serialized = serialize_user_memory_to_mem0(
                user_id=user_memory.user_id,
                user_memory=user_memory.user_memory,
                table_name=self.user_memory_table_name,
                agent_id=user_memory.agent_id,
                team_id=user_memory.team_id,
            )
            memory_id = serialized["memory_id"]
            
            # Check if memory already exists
            existing = self._get_memory_by_id(memory_id)
            
            if existing:
                existing_data = deserialize_user_memory_from_mem0(existing)
                created_at = existing_data.get("created_at", user_memory.created_at or int(time.time()))
                
                from upsonic.storage.mem0.utils import _epoch_to_iso
                serialized["metadata"]["created_at"] = _epoch_to_iso(created_at) if isinstance(created_at, int) else created_at
                
                data_str = serialized["metadata"].get("_data")
                if data_str:
                    data_dict = json.loads(data_str)
                    data_dict["created_at"] = created_at
                    serialized["metadata"]["_data"] = json.dumps(data_dict)
                
                self._update_memory(
                    memory_id=memory_id,
                    content=serialized["content"],
                    metadata=serialized["metadata"],
                )
                _logger.debug(f"Updated user memory: {user_memory.user_id}")
            else:
                # Add new memory
                self._add_memory(
                    memory_id=memory_id,
                    content=serialized["content"],
                    metadata=serialized["metadata"],
                )
                _logger.debug(f"Added user memory: {user_memory.user_id}")
            
            # Retrieve and return the stored memory
            stored = self._get_memory_by_id(memory_id)
            if stored is None:
                return None
            
            memory_dict = deserialize_user_memory_from_mem0(stored)
            if deserialize:
                return UserMemory.from_dict(memory_dict)
            return memory_dict
        
        except Exception as e:
            _logger.error(f"Error upserting user memory: {e}")
            raise e

    def upsert_user_memories(
        self,
        user_memories: List[UserMemory],
        deserialize: bool = True,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        """
        Bulk insert or update multiple user memories.
        
        Args:
            user_memories: List of UserMemory instances. Each must have user_id set.
            deserialize: If True, return full records.
        
        Returns:
            List of upserted user memory records.
        
        Raises:
            ValueError: If any record is missing user_id.
        """
        if not user_memories:
            return []
        
        # Validate all have user_id
        for memory in user_memories:
            if not memory.user_id:
                raise ValueError("All user memories must have user_id set")
        
        results: List[Union[UserMemory, Dict[str, Any]]] = []
        
        for memory in user_memories:
            try:
                result = self.upsert_user_memory(
                    user_memory=memory,
                    deserialize=deserialize,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                _logger.error(f"Error upserting user memory for {memory.user_id}: {e}")
                raise e
        
        _logger.debug(f"Upserted {len(results)} user memories")
        return results

    def get_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Get user memory from Mem0.
        
        Args:
            user_id: User ID to retrieve. If None, returns latest memory.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The user memory record if found, None otherwise.
        """
        try:
            if user_id is not None:
                # Direct lookup by memory_id
                memory_id = generate_user_memory_id(user_id, self.user_memory_table_name)
                stored = self._get_memory_by_id(memory_id)
                
                if stored is None:
                    return None
                
                memory_dict = deserialize_user_memory_from_mem0(stored)
                if deserialize:
                    return UserMemory.from_dict(memory_dict)
                return memory_dict
            
            # Search with filters and return latest
            filters = build_user_memory_filters(
                table_name=self.user_memory_table_name,
                agent_id=agent_id,
                team_id=team_id,
            )
            
            records = self._search_memories(filters=filters)
            
            if not records:
                return None
            
            # Sort by updated_at desc and get first
            sorted_records = sort_records_by_field(records, "updated_at", "desc")
            
            if not sorted_records:
                return None
            
            memory_dict = deserialize_user_memory_from_mem0(sorted_records[0])
            if deserialize:
                return UserMemory.from_dict(memory_dict)
            return memory_dict
        
        except Exception as e:
            _logger.error(f"Error getting user memory: {e}")
            raise e

    def get_user_memories(
        self,
        user_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        deserialize: bool = True,
    ) -> Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
        """
        Get multiple user memories from Mem0.
        
        Args:
            user_ids: List of user IDs to retrieve. If None, returns all.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            deserialize: If True, return list of UserMemory objects. 
                         If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of UserMemory objects if deserialize=True, otherwise tuple of (dicts, count).
        """
        try:
            records: List[Dict[str, Any]] = []
            
            if user_ids is not None and len(user_ids) > 0:
                # Fetch specific user memories by ID
                for user_id in user_ids:
                    memory_id = generate_user_memory_id(user_id, self.user_memory_table_name)
                    stored = self._get_memory_by_id(memory_id)
                    if stored:
                        records.append(stored)
            else:
                # Search with filters
                filters = build_user_memory_filters(
                    table_name=self.user_memory_table_name,
                    agent_id=agent_id,
                    team_id=team_id,
                )
                records = self._search_memories(filters=filters)
            
            if not records:
                if deserialize:
                    return []
                return [], 0
            
            # Get total count before pagination
            total_count = len(records)
            
            # Sort by updated_at desc
            sorted_records = sort_records_by_field(records, "updated_at", "desc")
            
            # Apply pagination
            paginated_records = apply_pagination(sorted_records, limit, offset)
            
            # Deserialize all records
            memory_dicts = [deserialize_user_memory_from_mem0(r) for r in paginated_records]
            
            if deserialize:
                return [UserMemory.from_dict(m) for m in memory_dicts]
            return memory_dicts, total_count
        
        except Exception as e:
            _logger.error(f"Error getting user memories: {e}")
            raise e

    def delete_user_memory(self, user_id: str) -> bool:
        """
        Delete user memory from Mem0.
        
        Args:
            user_id: User ID of the memory to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If user_id is not provided.
        """
        if not user_id:
            raise ValueError("user_id is required for delete")
        
        try:
            memory_id = generate_user_memory_id(user_id, self.user_memory_table_name)
            return self._delete_memory(memory_id)
        
        except Exception as e:
            _logger.error(f"Error deleting user memory: {e}")
            raise e

    def delete_user_memories(self, user_ids: List[str]) -> int:
        """
        Delete multiple user memories from Mem0.
        
        Args:
            user_ids: List of user IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            ValueError: If user_ids is empty.
        """
        if not user_ids:
            raise ValueError("user_ids is required and cannot be empty")
        
        deleted_count = 0
        
        for user_id in user_ids:
            try:
                if self.delete_user_memory(user_id):
                    deleted_count += 1
            except Exception as e:
                _logger.warning(f"Error deleting user memory {user_id}: {e}")
        
        _logger.debug(f"Deleted {deleted_count} user memories")
        return deleted_count

    # ======================== Cultural Knowledge Methods ========================

    def delete_cultural_knowledge(self, id: str) -> None:
        """
        Delete cultural knowledge from Mem0.
        
        Args:
            id: The ID of the cultural knowledge to delete.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        try:
            memory_id = generate_cultural_knowledge_memory_id(id, self.cultural_knowledge_table_name)
            success = self._delete_memory(memory_id)
            if success:
                _logger.debug(f"Successfully deleted cultural knowledge id: {id}")
            else:
                _logger.debug(f"No cultural knowledge found with id: {id}")
        except Exception as e:
            _logger.error(f"Error deleting cultural knowledge: {e}")
            raise e

    def get_cultural_knowledge(
        self,
        id: str,
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """
        Get cultural knowledge from Mem0.
        
        Args:
            id: The ID of the cultural knowledge to get.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            CulturalKnowledge object or dict if found, None otherwise.
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            memory_id = generate_cultural_knowledge_memory_id(id, self.cultural_knowledge_table_name)
            stored = self._get_memory_by_id(memory_id)
            
            if stored is None:
                return None
            
            db_row = deserialize_cultural_knowledge_from_mem0(stored)
            
            if not deserialize:
                return db_row
            return CulturalKnowledge.from_dict(db_row)
        
        except Exception as e:
            _logger.error(f"Error getting cultural knowledge: {e}")
            raise e

    def get_all_cultural_knowledge(
        self,
        name: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["CulturalKnowledge"], Tuple[List[Dict[str, Any]], int]]:
        """
        Get all cultural knowledge from Mem0.
        
        Args:
            name: Filter by name.
            limit: Maximum number of records to return.
            page: Page number (1-indexed).
            sort_by: Field to sort by.
            sort_order: Sort order ('asc' or 'desc').
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return list of CulturalKnowledge objects.
                        If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of CulturalKnowledge objects or tuple of (list of dicts, total count).
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            # Build filters
            filters = build_cultural_knowledge_filters(
                table_name=self.cultural_knowledge_table_name,
                name=name,
                agent_id=agent_id,
                team_id=team_id,
            )
            
            records = self._search_memories(filters=filters)
            
            if not records:
                return [] if deserialize else ([], 0)
            
            # Get total count before pagination
            total_count = len(records)
            
            # Sort records
            sorted_records = sort_records_by_field(
                records,
                sort_by=sort_by or "created_at",
                sort_order=sort_order or "desc",
            )
            
            # Apply pagination (convert page to offset)
            offset = None
            if page is not None and limit is not None:
                offset = (page - 1) * limit
            
            paginated_records = apply_pagination(sorted_records, limit, offset)
            
            # Deserialize all records
            db_rows = [deserialize_cultural_knowledge_from_mem0(r) for r in paginated_records]
            
            if not deserialize:
                return db_rows, total_count
            
            return [CulturalKnowledge.from_dict(row) for row in db_rows]
        
        except Exception as e:
            _logger.error(f"Error getting all cultural knowledge: {e}")
            raise e

    def upsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """
        Upsert cultural knowledge into Mem0.
        
        Args:
            cultural_knowledge: The CulturalKnowledge instance to upsert.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            The upserted CulturalKnowledge object or dict, or None if error.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        import uuid
        
        try:
            if cultural_knowledge.id is None:
                cultural_knowledge.id = str(uuid.uuid4())
            
            # Serialize for Mem0
            serialized = serialize_cultural_knowledge_to_mem0(
                cultural_knowledge, self.cultural_knowledge_table_name
            )
            memory_id = serialized["memory_id"]
            
            # Check if memory already exists
            existing = self._get_memory_by_id(memory_id)
            
            if existing:
                existing_data = deserialize_cultural_knowledge_from_mem0(existing)
                created_at = existing_data.get("created_at", int(time.time()))
                
                from upsonic.storage.mem0.utils import _epoch_to_iso
                data_str = serialized["metadata"].get("_data")
                if data_str:
                    data_dict = json.loads(data_str)
                    data_dict["created_at"] = created_at
                    serialized["metadata"]["_data"] = json.dumps(data_dict)
                    serialized["metadata"]["created_at"] = _epoch_to_iso(created_at) if isinstance(created_at, int) else created_at
                
                self._update_memory(
                    memory_id=memory_id,
                    content=serialized["content"],
                    metadata=serialized["metadata"],
                )
                _logger.debug(f"Updated cultural knowledge: {cultural_knowledge.id}")
            else:
                # Add new memory
                self._add_memory(
                    memory_id=memory_id,
                    content=serialized["content"],
                    metadata=serialized["metadata"],
                )
                _logger.debug(f"Added cultural knowledge: {cultural_knowledge.id}")
            
            # Retrieve and return the stored record
            stored = self._get_memory_by_id(memory_id)
            if stored is None:
                return None
            
            db_row = deserialize_cultural_knowledge_from_mem0(stored)
            
            if not deserialize:
                return db_row
            return CulturalKnowledge.from_dict(db_row)
        
        except Exception as e:
            _logger.error(f"Error upserting cultural knowledge: {e}")
            raise e

    # ======================== Knowledge Content Methods ========================

    def upsert_knowledge_content(
        self,
        knowledge_row: "KnowledgeRow",
    ) -> Optional["KnowledgeRow"]:
        """Insert or update a knowledge document registry entry."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            serialized = serialize_knowledge_to_mem0(
                knowledge_row, self.knowledge_table_name
            )
            memory_id = serialized["memory_id"]

            existing = self._get_memory_by_id(memory_id)

            if existing:
                existing_data = deserialize_knowledge_from_mem0(existing)
                created_at = existing_data.get("created_at", int(time.time()))

                from upsonic.storage.mem0.utils import _epoch_to_iso
                data_str = serialized["metadata"].get("_data")
                if data_str:
                    data_dict = json.loads(data_str)
                    data_dict["created_at"] = created_at
                    serialized["metadata"]["_data"] = json.dumps(data_dict, default=str)
                    serialized["metadata"]["created_at"] = (
                        _epoch_to_iso(created_at)
                        if isinstance(created_at, int)
                        else created_at
                    )

                self._update_memory(
                    memory_id=memory_id,
                    content=serialized["content"],
                    metadata=serialized["metadata"],
                )
                _logger.debug(f"Updated knowledge content: {knowledge_row.id}")
            else:
                self._add_memory(
                    memory_id=memory_id,
                    content=serialized["content"],
                    metadata=serialized["metadata"],
                )
                _logger.debug(f"Added knowledge content: {knowledge_row.id}")

            stored = self._get_memory_by_id(memory_id)
            if stored is None:
                return None

            db_row = deserialize_knowledge_from_mem0(stored)
            return KnowledgeRow.from_dict(db_row)

        except Exception as e:
            _logger.error(f"Error upserting knowledge content: {e}")
            raise e

    def get_knowledge_content(
        self,
        id: str,
    ) -> Optional["KnowledgeRow"]:
        """Get a knowledge document registry entry by ID."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            memory_id = generate_knowledge_memory_id(id, self.knowledge_table_name)
            stored = self._get_memory_by_id(memory_id)

            if stored is None:
                return None

            db_row = deserialize_knowledge_from_mem0(stored)
            if not db_row:
                return None

            return KnowledgeRow.from_dict(db_row)

        except Exception as e:
            _logger.error(f"Error getting knowledge content: {e}")
            raise e

    def get_knowledge_contents(
        self,
        knowledge_base_id: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> Tuple[List["KnowledgeRow"], int]:
        """Get knowledge document registry entries with filtering and pagination."""
        from upsonic.storage.schemas import KnowledgeRow

        try:
            filters = build_knowledge_filters(
                table_name=self.knowledge_table_name,
                knowledge_base_id=knowledge_base_id,
            )

            records = self._search_memories(filters=filters)

            if not records:
                return [], 0

            total_count: int = len(records)

            sorted_records = sort_records_by_field(
                records,
                sort_by=sort_by or "created_at",
                sort_order=sort_order or "desc",
            )

            offset: Optional[int] = None
            if page is not None and limit is not None:
                offset = (page - 1) * limit

            paginated_records = apply_pagination(sorted_records, limit, offset)

            rows = [
                KnowledgeRow.from_dict(deserialize_knowledge_from_mem0(r))
                for r in paginated_records
            ]
            return rows, total_count

        except Exception as e:
            _logger.error(f"Error getting knowledge contents: {e}")
            raise e

    def delete_knowledge_content(self, id: str) -> bool:
        """Delete a knowledge document registry entry by ID."""
        try:
            memory_id = generate_knowledge_memory_id(id, self.knowledge_table_name)
            return self._delete_memory(memory_id)

        except Exception as e:
            _logger.error(f"Error deleting knowledge content: {e}")
            raise e

    def delete_knowledge_contents(self, ids: List[str]) -> int:
        """Delete multiple knowledge document registry entries."""
        if not ids:
            return 0

        deleted_count: int = 0
        for doc_id in ids:
            try:
                if self.delete_knowledge_content(doc_id):
                    deleted_count += 1
            except Exception as e:
                _logger.warning(f"Error deleting knowledge content {doc_id}: {e}")

        _logger.debug(f"Deleted {deleted_count} knowledge entries")
        return deleted_count

    # ======================== Utility Methods ========================

    def clear_all(self) -> None:
        """
        Clear all data from all tables.
        
        This removes all sessions, user memories, and cultural knowledge from storage.
        Use with caution.
        """
        try:
            # Get all sessions and delete them
            session_filters = build_session_filters(table_name=self.session_table_name)
            session_records = self._search_memories(filters=session_filters)
            
            for record in session_records:
                memory_id = record.get("id") or record.get("memory_id")
                if memory_id:
                    self._delete_memory(memory_id)
            
            _logger.debug(f"Cleared {len(session_records)} sessions")
            
            # Get all user memories and delete them
            user_memory_filters = build_user_memory_filters(table_name=self.user_memory_table_name)
            user_memory_records = self._search_memories(filters=user_memory_filters)
            
            for record in user_memory_records:
                memory_id = record.get("id") or record.get("memory_id")
                if memory_id:
                    self._delete_memory(memory_id)
            
            _logger.debug(f"Cleared {len(user_memory_records)} user memories")
            
            # Get all cultural knowledge and delete them
            cultural_knowledge_filters = build_cultural_knowledge_filters(
                table_name=self.cultural_knowledge_table_name
            )
            cultural_knowledge_records = self._search_memories(filters=cultural_knowledge_filters)
            
            for record in cultural_knowledge_records:
                memory_id = record.get("id") or record.get("memory_id")
                if memory_id:
                    self._delete_memory(memory_id)
            
            _logger.debug(f"Cleared {len(cultural_knowledge_records)} cultural knowledge entries")

            # Get all knowledge entries and delete them
            knowledge_filters = build_knowledge_filters(
                table_name=self.knowledge_table_name
            )
            knowledge_records = self._search_memories(filters=knowledge_filters)

            for record in knowledge_records:
                memory_id = record.get("id") or record.get("memory_id")
                if memory_id:
                    self._delete_memory(memory_id)

            _logger.debug(f"Cleared {len(knowledge_records)} knowledge entries")
            _logger.info("Cleared all data from Mem0 storage")
        
        except Exception as e:
            _logger.error(f"Error clearing all data: {e}")
            raise e

    # ======================== Private Mem0 API Methods ========================
    # 
    # These methods handle the API differences between:
    # - MemoryClient (Platform): Supports memory_id on add(), uses text param for update()
    # - Memory (Self-hosted): Does NOT support memory_id on add(), uses data param for update()
    #
    # For self-hosted Memory, we use metadata filtering to find records by our session_id
    # since we cannot specify our own memory_id during creation.

    def _add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Add a new memory to Mem0.
        
        Args:
            memory_id: Unique identifier for the memory (stored in metadata for lookup)
            content: The content to store
            metadata: Metadata for filtering (must include _upsonic_memory_id)
        
        Returns:
            The created memory record or None
        
        Note:
            Both Platform and Self-hosted use metadata-based ID tracking because:
            - Platform API's memory_id handling is unreliable for custom IDs
            - Self-hosted doesn't support memory_id parameter at all
            We store our custom ID in metadata._upsonic_memory_id for lookup.
        """
        try:
            if self._is_platform_client:
                result = self.memory_client.add(
                    messages=content,
                    user_id=self.default_user_id,
                    metadata=metadata,
                    infer=False,
                    async_mode=False,
                )
            else:
                result = self.memory_client.add(
                    messages=content,
                    user_id=self.default_user_id,
                    metadata=metadata,
                    infer=False,
                )
            
            _logger.debug(f"Added memory: {memory_id}")
            return result
        
        except Exception as e:
            _logger.error(f"Error adding memory {memory_id}: {e}")
            raise e

    def _update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing memory in Mem0.
        
        Args:
            memory_id: Our custom memory ID (stored in metadata as _upsonic_memory_id)
            content: The new content
            metadata: The new metadata
        
        Returns:
            The updated memory record or None
        
        Note:
            Both Platform and Self-hosted use metadata-based ID resolution.
            Platform uses 'text' param, Self-hosted uses 'data' param.
        """
        try:
            # Find the actual Mem0 memory_id from our custom ID
            actual_memory_id = self._resolve_memory_id(memory_id)
            
            if actual_memory_id is None:
                _logger.warning(f"Memory not found for update: {memory_id}")
                return None
            
            if self._is_platform_client:
                result = self.memory_client.update(
                    memory_id=actual_memory_id,
                    metadata=metadata,
                )
            else:
                self.memory_client.delete(memory_id=actual_memory_id)
                result = self.memory_client.add(
                    messages=content,
                    user_id=self.default_user_id,
                    metadata=metadata,
                    infer=False,
                )
            
            _logger.debug(f"Updated memory: {memory_id}")
            return result
        
        except Exception as e:
            _logger.error(f"Error updating memory {memory_id}: {e}")
            raise e

    def _get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by its ID.
        
        Args:
            memory_id: Our custom memory ID (stored in metadata as _upsonic_memory_id)
        
        Returns:
            The memory record or None if not found
        
        Note:
            Both Platform and Self-hosted search by _upsonic_memory_id in metadata.
        """
        try:
            return self._find_memory_by_upsonic_id(memory_id)
        except Exception as e:
            _logger.debug(f"Error getting memory {memory_id}: {e}")
            return None

    def _delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id: Our custom memory ID (stored in metadata as _upsonic_memory_id)
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            # Find actual Mem0 ID first
            actual_memory_id = self._resolve_memory_id(memory_id)
            
            if actual_memory_id is None:
                _logger.debug(f"Memory not found for delete: {memory_id}")
                return False
            
            self.memory_client.delete(memory_id=actual_memory_id)
            
            _logger.debug(f"Deleted memory: {memory_id}")
            return True
        
        except Exception as e:
            if "not found" in str(e).lower():
                return False
            _logger.warning(f"Error deleting memory {memory_id}: {e}")
            return False

    def _resolve_memory_id(self, upsonic_memory_id: str) -> Optional[str]:
        """
        Resolve our custom memory_id to the actual Mem0 memory_id.
        
        We store our ID in metadata and need to find the actual
        Mem0-generated ID for operations.
        
        Args:
            upsonic_memory_id: Our custom memory ID (stored in metadata)
        
        Returns:
            The actual Mem0 memory_id or None if not found
        """
        memory = self._find_memory_by_upsonic_id(upsonic_memory_id)
        if memory:
            return memory.get("id")
        return None

    def _find_memory_by_upsonic_id(self, upsonic_memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a memory record by our custom upsonic_memory_id in metadata.
        
        Args:
            upsonic_memory_id: Our custom memory ID to search for
        
        Returns:
            The memory record or None if not found
        """
        try:
            if self._is_platform_client:
                # Platform API: Use proper filter format with AND wrapper
                result = self.memory_client.get_all(
                    filters={
                        "AND": [
                            {"user_id": self.default_user_id},
                            {"metadata": {"_upsonic_memory_id": upsonic_memory_id}},
                        ]
                    }
                )
            else:
                # Self-hosted API: Get all for user and filter manually
                result = self.memory_client.get_all(
                    user_id=self.default_user_id,
                    limit=10000,
                )
            
            # Handle different return formats
            if isinstance(result, dict):
                memories = result.get("results", result.get("memories", []))
            elif isinstance(result, list):
                memories = result
            else:
                memories = []
            
            # For Platform API with proper filters, we should get exact match
            if self._is_platform_client and memories:
                return memories[0] if memories else None
            
            # For Self-hosted, find the memory with matching _upsonic_memory_id in metadata
            for memory in memories:
                metadata = memory.get("metadata", {})
                if metadata.get("_upsonic_memory_id") == upsonic_memory_id:
                    return memory
            
            return None
        
        except Exception as e:
            _logger.debug(f"Error finding memory by upsonic ID: {e}")
            return None

    def _search_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Search memories with optional filters.
        
        Args:
            filters: Metadata filters (for client-side filtering)
            query: Optional search query (for semantic search)
            limit: Maximum number of results
        
        Returns:
            List of matching memory records
        """
        try:
            if self._is_platform_client:
                # Platform API: Use proper filter format with AND wrapper
                api_filters: Dict[str, Any] = {
                    "AND": [
                        {"user_id": self.default_user_id},
                    ]
                }
                
                # Add metadata filters if provided
                if filters:
                    for key, value in filters.items():
                        api_filters["AND"].append({"metadata": {key: value}})
                
                result = self.memory_client.get_all(filters=api_filters)
            else:
                # Self-hosted API: Get all for user
                result = self.memory_client.get_all(
                    user_id=self.default_user_id,
                    limit=limit,
                )
            
            # Handle different return formats
            if isinstance(result, dict):
                memories = result.get("results", result.get("memories", []))
            elif isinstance(result, list):
                memories = result
            else:
                memories = []
            
            # For Self-hosted, apply metadata filters manually
            if not self._is_platform_client and filters and memories:
                filtered = []
                for memory in memories:
                    metadata = memory.get("metadata", {})
                    
                    # Check if all filter conditions match
                    matches = True
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            matches = False
                            break
                    
                    if matches:
                        filtered.append(memory)
                
                return filtered
            
            return memories
        
        except Exception as e:
            _logger.error(f"Error searching memories: {e}")
            return []

