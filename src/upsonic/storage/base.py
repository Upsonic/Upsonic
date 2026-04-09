"""Base abstract classes for storage implementations."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

if TYPE_CHECKING:
    from upsonic.session.base import SessionType
    from upsonic.storage.schemas import UserMemory, KnowledgeRow
    from upsonic.session.base import Session
    from upsonic.culture.cultural_knowledge import CulturalKnowledge




class Storage(ABC):
    """Base abstract class for synchronous storage implementations.
    
    This class defines the interface for all storage backends that need
    to persist agent sessions and user memory data.
    """

    def __init__(
        self,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        cultural_knowledge_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize the storage backend.
        
        Args:
            session_table: Name of the table to store sessions.
            user_memory_table: Name of the table to store user memories.
            cultural_knowledge_table: Name of the table to store cultural knowledge.
            knowledge_table: Name of the table to store knowledge base document registry.
            id: Unique identifier for this storage instance.
        """
        self.id: str = id or str(uuid4())
        self.session_table_name: str = session_table or "upsonic_sessions"
        self.user_memory_table_name: str = user_memory_table or "upsonic_user_memories"
        self.cultural_knowledge_table_name: str = cultural_knowledge_table or "upsonic_cultural_knowledge"
        self.knowledge_table_name: str = knowledge_table or "upsonic_knowledge"

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check.
        
        Returns:
            True if the table exists, False otherwise.
        """
        raise NotImplementedError

    def _create_all_tables(self) -> None:
        """Create all required tables for this storage. Override in subclasses."""
        pass

    def close(self) -> None:
        """
        Close database connections and release resources.
        
        Override in subclasses to properly dispose of connection pools.
        Should be called during application shutdown.
        """
        pass

    # --- Session Methods ---

    @abstractmethod
    def upsert_session(
        self,
        session: "Session",
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Insert or update a session in the database.
        
        Args:
            session: The session to upsert. Must have session_id set.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The upserted session (deserialized or dict based on flag).
        
        Raises:
            ValueError: If session_id is not provided.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def get_session(
        self,
        session_id: Optional[str] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Get a session from the database.
        
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
        raise NotImplementedError

    @abstractmethod
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
        Get multiple sessions from the database.
        
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
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database.
        
        Args:
            session_id: ID of the session to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If session_id is not provided.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_sessions(self, session_ids: List[str]) -> int:
        """
        Delete multiple sessions from the database.
        
        Args:
            session_ids: List of session IDs to delete.
        
        Returns:
            Number of sessions deleted.
        
        Raises:
            ValueError: If session_ids is empty.
        """
        raise NotImplementedError

    # --- User Memory Methods ---

    @abstractmethod
    def upsert_user_memory(
        self,
        user_memory: "UserMemory",
        deserialize: bool = True,
    ) -> Optional[Union["UserMemory", Dict[str, Any]]]:
        """
        Insert or update user memory in the database.
        
        Args:
            user_memory: The UserMemory instance to store. Must have user_id set.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The upserted user memory record.
        
        Raises:
            ValueError: If user_id is not provided in user_memory.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_user_memories(
        self,
        user_memories: List["UserMemory"],
        deserialize: bool = True,
    ) -> List[Union["UserMemory", Dict[str, Any]]]:
        """
        Bulk insert or update multiple user memories.
        
        Args:
            user_memories: List of UserMemory instances. Each must have user_id set.
            deserialize: If True, return UserMemory objects. If False, return raw dicts.
        
        Returns:
            List of upserted user memory records.
        
        Raises:
            ValueError: If any record is missing user_id.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["UserMemory", Dict[str, Any]]]:
        """
        Get user memory from the database.
        
        Args:
            user_id: User ID to retrieve. If None, returns latest memory.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return UserMemory object.
                        If False, return raw dictionary.
        
        Returns:
            UserMemory object or dict if found, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_memories(
        self,
        user_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        deserialize: bool = True,
    ) -> Union[List["UserMemory"], Tuple[List[Dict[str, Any]], int]]:
        """
        Get multiple user memories from the database.
        
        Args:
            user_ids: List of user IDs to retrieve. If None, returns all.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            deserialize: If True, return list of UserMemory objects.
                        If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of UserMemory objects or tuple of (list of dicts, total count).
        """
        raise NotImplementedError

    @abstractmethod
    def delete_user_memory(self, user_id: str) -> bool:
        """
        Delete user memory from the database.
        
        Args:
            user_id: User ID of the memory to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If user_id is not provided.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_user_memories(self, user_ids: List[str]) -> int:
        """
        Delete multiple user memories from the database.
        
        Args:
            user_ids: List of user IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            ValueError: If user_ids is empty.
        """
        raise NotImplementedError

    # --- Cultural Knowledge Methods ---

    @abstractmethod
    def delete_cultural_knowledge(self, id: str) -> None:
        """
        Delete cultural knowledge from the database.
        
        Args:
            id: ID of the cultural knowledge to delete.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        raise NotImplementedError

    def delete_cultural_knowledges(self, ids: List[str]) -> int:
        """
        Delete multiple cultural knowledge entries from the database.
        
        Args:
            ids: List of IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Note: Default implementation calls delete_cultural_knowledge for each ID.
              Providers may override for optimized bulk deletion.
        """
        count = 0
        for id in ids:
            try:
                self.delete_cultural_knowledge(id)
                count += 1
            except Exception:
                pass
        return count

    @abstractmethod
    def get_cultural_knowledge(
        self,
        id: str,
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """
        Get cultural knowledge from the database.
        
        Args:
            id: ID of the cultural knowledge to retrieve.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            CulturalKnowledge object or dict if found, None otherwise.
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        raise NotImplementedError

    @abstractmethod
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
        Get all cultural knowledge entries from the database.
        
        Args:
            name: Filter by name.
            limit: Maximum number of records to return.
            page: Page number (1-indexed).
            sort_by: Column to sort by.
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
        raise NotImplementedError

    @abstractmethod
    def upsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """
        Insert or update cultural knowledge in the database.
        
        Args:
            cultural_knowledge: The CulturalKnowledge instance to store.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            The upserted CulturalKnowledge object or dict, or None if error occurs.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        raise NotImplementedError

    # --- Knowledge Content Methods ---

    @abstractmethod
    def upsert_knowledge_content(
        self,
        knowledge_row: "KnowledgeRow",
    ) -> Optional["KnowledgeRow"]:
        """
        Insert or update a knowledge document registry entry.
        
        Args:
            knowledge_row: The KnowledgeRow instance to upsert.
        
        Returns:
            The upserted KnowledgeRow, or None if the operation fails.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        raise NotImplementedError

    @abstractmethod
    def get_knowledge_content(
        self,
        id: str,
    ) -> Optional["KnowledgeRow"]:
        """
        Get a knowledge document registry entry by ID.
        
        Args:
            id: The document ID to retrieve.
        
        Returns:
            KnowledgeRow if found, None otherwise.
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        raise NotImplementedError

    @abstractmethod
    def get_knowledge_contents(
        self,
        knowledge_base_id: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> Tuple[List["KnowledgeRow"], int]:
        """
        Get knowledge document registry entries with filtering and pagination.
        
        Args:
            knowledge_base_id: Filter by knowledge base ID.
            limit: Maximum number of records to return.
            page: Page number (1-indexed).
            sort_by: Column to sort by.
            sort_order: Sort order ('asc' or 'desc').
        
        Returns:
            Tuple of (list of KnowledgeRow, total count).
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_knowledge_content(self, id: str) -> bool:
        """
        Delete a knowledge document registry entry by ID.
        
        Args:
            id: The document ID to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_knowledge_contents(self, ids: List[str]) -> int:
        """
        Delete multiple knowledge document registry entries.
        
        Args:
            ids: List of document IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        raise NotImplementedError

    # --- Utility Methods ---

    @abstractmethod
    def clear_all(self) -> None:
        """
        Clear all data from all tables.
        
        This removes all sessions and user memories from the storage.
        Use with caution.
        """
        raise NotImplementedError

    # --- Usage Methods ---

    def get_usage(
        self,
        session_id: str,
    ) -> Optional[Any]:
        """
        Get the session-level aggregated usage for a session.
        
        Args:
            session_id: ID of the session to get usage for.
        
        Returns:
            RunUsage object if found, None otherwise.
        
        Note: Default implementation gets session and returns usage.
              Providers may override for optimized retrieval.
        """
        session = self.get_session(session_id=session_id, deserialize=True)
        if session is None:
            return None
        return getattr(session, 'usage', None)

    # --- Generic Model Storage Methods ---
    # These methods allow storing any Pydantic BaseModel in a generic table/collection.
    # Used by components like MemoryBackend for FilesystemEntry storage.

    def upsert_model(
        self,
        key: str,
        model: Any,
        collection: str = "generic_models",
    ) -> None:
        """
        Insert or update a generic Pydantic model in storage.
        
        Args:
            key: Unique identifier for the model instance.
            model: The Pydantic BaseModel instance to store.
            collection: Collection/table name for grouping models.
        
        Note: Default implementation does nothing. Providers should override.
        """
        pass

    def get_model(
        self,
        key: str,
        model_type: Any,
        collection: str = "generic_models",
    ) -> Optional[Any]:
        """
        Retrieve a generic Pydantic model from storage.
        
        Args:
            key: Unique identifier for the model instance.
            model_type: The Pydantic model class to deserialize into.
            collection: Collection/table name where model is stored.
        
        Returns:
            The model instance if found, None otherwise.
        
        Note: Default implementation returns None. Providers should override.
        """
        return None

    def delete_model(
        self,
        key: str,
        collection: str = "generic_models",
    ) -> bool:
        """
        Delete a generic model from storage.
        
        Args:
            key: Unique identifier for the model instance.
            collection: Collection/table name where model is stored.
        
        Returns:
            True if deleted, False if not found.
        
        Note: Default implementation returns False. Providers should override.
        """
        return False

    def list_models(
        self,
        model_type: Any,
        collection: str = "generic_models",
    ) -> List[Any]:
        """
        List all models of a type in a collection.
        
        Args:
            model_type: The Pydantic model class to deserialize into.
            collection: Collection/table name to list from.
        
        Returns:
            List of model instances.
        
        Note: Default implementation returns empty list. Providers should override.
        """
        return []


class AsyncStorage(ABC):
    """Base abstract class for asynchronous storage implementations.
    
    This class defines the interface for all async storage backends that need
    to persist agent sessions and user memory data.
    """

    def __init__(
        self,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        cultural_knowledge_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize the async storage backend.
        
        Args:
            session_table: Name of the table to store sessions.
            user_memory_table: Name of the table to store user memories.
            cultural_knowledge_table: Name of the table to store cultural knowledge.
            knowledge_table: Name of the table to store knowledge base document registry.
            id: Unique identifier for this storage instance.
        """
        self.id: str = id or str(uuid4())
        self.session_table_name: str = session_table or "upsonic_sessions"
        self.user_memory_table_name: str = user_memory_table or "upsonic_user_memories"
        self.cultural_knowledge_table_name: str = cultural_knowledge_table or "upsonic_cultural_knowledge"
        self.knowledge_table_name: str = knowledge_table or "upsonic_knowledge"

    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check.
        
        Returns:
            True if the table exists, False otherwise.
        """
        raise NotImplementedError

    async def _create_all_tables(self) -> None:
        """Create all required tables for this storage. Override in subclasses."""
        pass

    async def close(self) -> None:
        """
        Close database connections and release resources.
        
        Override in subclasses to properly dispose of connection pools.
        Should be called during application shutdown.
        """
        pass

    # --- Session Methods ---

    @abstractmethod
    async def aupsert_session(
        self,
        session: "Session",
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Insert or update a session in the database (async).
        
        Args:
            session: The session to upsert. Must have session_id set.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The upserted session (deserialized or dict based on flag).
        
        Raises:
            ValueError: If session_id is not provided.
        """
        raise NotImplementedError

    @abstractmethod
    async def aupsert_sessions(
        self,
        sessions: List["Session"],
        deserialize: bool = True,
    ) -> List[Union["Session", Dict[str, Any]]]:
        """
        Bulk insert or update multiple sessions (async).
        
        Args:
            sessions: List of sessions to upsert. Each must have session_id set.
            deserialize: If True, return deserialized Session objects.
                        If False, return raw dictionaries.
        
        Returns:
            List of upserted sessions.
        
        Raises:
            ValueError: If any session is missing session_id.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_session(
        self,
        session_id: Optional[str] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Get a session from the database (async).
        
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
        raise NotImplementedError

    @abstractmethod
    async def aget_sessions(
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
        Get multiple sessions from the database (async).
        
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
        raise NotImplementedError

    @abstractmethod
    async def adelete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database (async).
        
        Args:
            session_id: ID of the session to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If session_id is not provided.
        """
        raise NotImplementedError

    @abstractmethod
    async def adelete_sessions(self, session_ids: List[str]) -> int:
        """
        Delete multiple sessions from the database (async).
        
        Args:
            session_ids: List of session IDs to delete.
        
        Returns:
            Number of sessions deleted.
        
        Raises:
            ValueError: If session_ids is empty.
        """
        raise NotImplementedError

    # --- User Memory Methods ---

    @abstractmethod
    async def aupsert_user_memory(
        self,
        user_memory: "UserMemory",
        deserialize: bool = True,
    ) -> Optional[Union["UserMemory", Dict[str, Any]]]:
        """
        Insert or update user memory in the database (async).
        
        Args:
            user_memory: The UserMemory instance to store. Must have user_id set.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The upserted user memory record.
        
        Raises:
            ValueError: If user_id is not provided in user_memory.
        """
        raise NotImplementedError

    @abstractmethod
    async def aupsert_user_memories(
        self,
        user_memories: List["UserMemory"],
        deserialize: bool = True,
    ) -> List[Union["UserMemory", Dict[str, Any]]]:
        """
        Bulk insert or update multiple user memories (async).
        
        Args:
            user_memories: List of UserMemory instances. Each must have user_id set.
            deserialize: If True, return UserMemory objects. If False, return raw dicts.
        
        Returns:
            List of upserted user memory records.
        
        Raises:
            ValueError: If any record is missing user_id.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["UserMemory", Dict[str, Any]]]:
        """
        Get user memory from the database (async).
        
        Args:
            user_id: User ID to retrieve. If None, returns latest memory.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return UserMemory object.
                        If False, return raw dictionary.
        
        Returns:
            UserMemory object or dict if found, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_user_memories(
        self,
        user_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        deserialize: bool = True,
    ) -> Union[List["UserMemory"], Tuple[List[Dict[str, Any]], int]]:
        """
        Get multiple user memories from the database (async).
        
        Args:
            user_ids: List of user IDs to retrieve. If None, returns all.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            deserialize: If True, return list of UserMemory objects.
                        If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of UserMemory objects or tuple of (list of dicts, total count).
        """
        raise NotImplementedError

    @abstractmethod
    async def adelete_user_memory(self, user_id: str) -> bool:
        """
        Delete user memory from the database (async).
        
        Args:
            user_id: User ID of the memory to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If user_id is not provided.
        """
        raise NotImplementedError

    @abstractmethod
    async def adelete_user_memories(self, user_ids: List[str]) -> int:
        """
        Delete multiple user memories from the database (async).
        
        Args:
            user_ids: List of user IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            ValueError: If user_ids is empty.
        """
        raise NotImplementedError

    # --- Cultural Knowledge Methods (Async) ---

    @abstractmethod
    async def adelete_cultural_knowledge(self, id: str) -> None:
        """
        Delete cultural knowledge from the database (async).
        
        Args:
            id: ID of the cultural knowledge to delete.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        raise NotImplementedError

    async def adelete_cultural_knowledges(self, ids: List[str]) -> int:
        """
        Delete multiple cultural knowledge entries from the database (async).
        
        Args:
            ids: List of IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Note: Default implementation calls adelete_cultural_knowledge for each ID.
              Providers may override for optimized bulk deletion.
        """
        count = 0
        for id in ids:
            try:
                await self.adelete_cultural_knowledge(id)
                count += 1
            except Exception:
                pass
        return count

    @abstractmethod
    async def aget_cultural_knowledge(
        self,
        id: str,
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """
        Get cultural knowledge from the database (async).
        
        Args:
            id: ID of the cultural knowledge to retrieve.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            CulturalKnowledge object or dict if found, None otherwise.
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_all_cultural_knowledge(
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
        Get all cultural knowledge entries from the database (async).
        
        Args:
            name: Filter by name.
            limit: Maximum number of records to return.
            page: Page number (1-indexed).
            sort_by: Column to sort by.
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
        raise NotImplementedError

    @abstractmethod
    async def aupsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """
        Insert or update cultural knowledge in the database (async).
        
        Args:
            cultural_knowledge: The CulturalKnowledge instance to store.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            The upserted CulturalKnowledge object or dict, or None if error occurs.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        raise NotImplementedError

    # --- Knowledge Content Methods (Async) ---

    @abstractmethod
    async def aupsert_knowledge_content(
        self,
        knowledge_row: "KnowledgeRow",
    ) -> Optional["KnowledgeRow"]:
        """
        Insert or update a knowledge document registry entry (async).
        
        Args:
            knowledge_row: The KnowledgeRow instance to upsert.
        
        Returns:
            The upserted KnowledgeRow, or None if the operation fails.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_knowledge_content(
        self,
        id: str,
    ) -> Optional["KnowledgeRow"]:
        """
        Get a knowledge document registry entry by ID (async).
        
        Args:
            id: The document ID to retrieve.
        
        Returns:
            KnowledgeRow if found, None otherwise.
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_knowledge_contents(
        self,
        knowledge_base_id: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> Tuple[List["KnowledgeRow"], int]:
        """
        Get knowledge document registry entries with filtering and pagination (async).
        
        Args:
            knowledge_base_id: Filter by knowledge base ID.
            limit: Maximum number of records to return.
            page: Page number (1-indexed).
            sort_by: Column to sort by.
            sort_order: Sort order ('asc' or 'desc').
        
        Returns:
            Tuple of (list of KnowledgeRow, total count).
        
        Raises:
            Exception: If an error occurs during retrieval.
        """
        raise NotImplementedError

    @abstractmethod
    async def adelete_knowledge_content(self, id: str) -> bool:
        """
        Delete a knowledge document registry entry by ID (async).
        
        Args:
            id: The document ID to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        raise NotImplementedError

    @abstractmethod
    async def adelete_knowledge_contents(self, ids: List[str]) -> int:
        """
        Delete multiple knowledge document registry entries (async).
        
        Args:
            ids: List of document IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        raise NotImplementedError

    # --- Utility Methods ---

    @abstractmethod
    async def aclear_all(self) -> None:
        """
        Clear all data from all tables (async).
        
        This removes all sessions and user memories from the storage.
        Use with caution.
        """
        raise NotImplementedError

    # --- Usage Methods ---

    async def aget_usage(
        self,
        session_id: str,
    ) -> Optional[Any]:
        """
        Get the session-level aggregated usage for a session (async).
        
        Args:
            session_id: ID of the session to get usage for.
        
        Returns:
            RunUsage object if found, None otherwise.
        
        Note: Default implementation gets session and returns usage.
              Providers may override for optimized retrieval.
        """
        session = await self.aget_session(session_id=session_id, deserialize=True)
        if session is None:
            return None
        return getattr(session, 'usage', None)

    # --- Generic Model Storage Methods (Async) ---
    # These methods allow storing any Pydantic BaseModel in a generic table/collection.
    # Used by components like MemoryBackend for FilesystemEntry storage.

    async def aupsert_model(
        self,
        key: str,
        model: Any,
        collection: str = "generic_models",
    ) -> None:
        """
        Insert or update a generic Pydantic model in storage (async).
        
        Args:
            key: Unique identifier for the model instance.
            model: The Pydantic BaseModel instance to store.
            collection: Collection/table name for grouping models.
        
        Note: Default implementation does nothing. Providers should override.
        """
        pass

    async def aget_model(
        self,
        key: str,
        model_type: Any,
        collection: str = "generic_models",
    ) -> Optional[Any]:
        """
        Retrieve a generic Pydantic model from storage (async).
        
        Args:
            key: Unique identifier for the model instance.
            model_type: The Pydantic model class to deserialize into.
            collection: Collection/table name where model is stored.
        
        Returns:
            The model instance if found, None otherwise.
        
        Note: Default implementation returns None. Providers should override.
        """
        return None

    async def adelete_model(
        self,
        key: str,
        collection: str = "generic_models",
    ) -> bool:
        """
        Delete a generic model from storage (async).
        
        Args:
            key: Unique identifier for the model instance.
            collection: Collection/table name where model is stored.
        
        Returns:
            True if deleted, False if not found.
        
        Note: Default implementation returns False. Providers should override.
        """
        return False

    async def alist_models(
        self,
        model_type: Any,
        collection: str = "generic_models",
    ) -> List[Any]:
        """
        List all models of a type in a collection (async).
        
        Args:
            model_type: The Pydantic model class to deserialize into.
            collection: Collection/table name to list from.
        
        Returns:
            List of model instances.
        
        Note: Default implementation returns empty list. Providers should override.
        """
        return []

