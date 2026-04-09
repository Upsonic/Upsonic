"""In-memory storage implementation for Upsonic agent framework."""
from __future__ import annotations

import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from upsonic.session.base import Session, SessionType
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.schemas import KnowledgeRow

from upsonic.storage.base import Storage
from upsonic.storage.in_memory.utils import (
    apply_pagination,
    apply_sorting,
    deep_copy_record,
    deep_copy_records,
)
from upsonic.storage.schemas import UserMemory
from upsonic.storage.utils import deserialize_session
from upsonic.utils.logging_config import get_logger

_logger = get_logger("upsonic.storage.in_memory")

class InMemoryStorage(Storage):
    """In-memory storage implementation for Upsonic agent framework.
    
    This storage backend uses Python lists and dictionaries for in-memory
    storage of agent sessions and user memory data. Data is NOT persistent
    and will be lost when the application terminates.
    
    Use this storage for:
    - Development and testing
    - Temporary agent sessions
    - Scenarios where persistence is not required
    
    Example:
        ```python
        storage = InMemoryStorage()
        
        # Upsert a session
        session = AgentSession(session_id="abc123", ...)
        result = storage.upsert_session(session)
        
        # Get a session
        session = storage.get_session(session_id="abc123")
        
        # Clear all data
        storage.clear_all()
        ```
    """

    def __init__(
        self,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize the in-memory storage.
        
        Args:
            session_table: Name of the session table (for compatibility).
            user_memory_table: Name of the user memory table (for compatibility).
            knowledge_table: Name of the knowledge table (for compatibility).
            id: Unique identifier for this storage instance.
        """
        super().__init__(
            session_table=session_table,
            user_memory_table=user_memory_table,
            knowledge_table=knowledge_table,
            id=id,
        )

        self._sessions: List[Dict[str, Any]] = []
        self._user_memories: List[Dict[str, Any]] = []
        self._cultural_knowledge: List[Dict[str, Any]] = []
        self._knowledge: List[Dict[str, Any]] = []
        self._generic_models: Dict[str, Dict[str, Any]] = {}

        _logger.info(f"Initialized InMemoryStorage with id: {self.id}")

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists. Always returns True for in-memory storage.
        
        Args:
            table_name: Name of the table to check.
        
        Returns:
            True (in-memory storage is always available).
        """
        return True

    def close(self) -> None:
        """Close storage. Clears all data for in-memory storage."""
        self._sessions.clear()
        self._user_memories.clear()
        self._cultural_knowledge.clear()
        self._knowledge.clear()
        _logger.info(f"InMemoryStorage with id: {self.id} closed and cleared")

    def _get_session_type_value(self, session: "Session") -> str:
        """
        Get the session type value string from a session object.
        
        Args:
            session: Session object.
        
        Returns:
            Session type string value.
        """
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType as SessionTypeEnum

        if isinstance(session, AgentSession):
            return SessionTypeEnum.AGENT.value
        # Add TeamSession and WorkflowSession checks when available
        return SessionTypeEnum.AGENT.value

    def upsert_session(
        self,
        session: "Session",
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Insert or update a session in in-memory storage.
        
        Args:
            session: The session to upsert. Must have session_id set.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The upserted session (deserialized or dict based on flag).
        
        Raises:
            ValueError: If session_id is not provided.
        """
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType

        if session is None:
            return None

        session_id = getattr(session, "session_id", None)
        if not session_id:
            raise ValueError("session_id is required for upsert_session")

        try:
            # Convert session to dict with serialize_flag=True
            session_dict = session.to_dict(serialize_flag=True)

            current_time = int(time.time())

            # Build session record based on session type
            if isinstance(session, AgentSession):
                # Only use AgentSession-specific attributes
                session_record: Dict[str, Any] = {
                    "session_id": session_dict.get("session_id"),
                    "session_type": SessionType.AGENT.value,
                    "agent_id": session_dict.get("agent_id"),
                    "team_id": session_dict.get("team_id"),
                    "workflow_id": session_dict.get("workflow_id"),
                    "user_id": session_dict.get("user_id"),
                    "session_data": session_dict.get("session_data"),
                    "agent_data": session_dict.get("agent_data"),
                    "metadata": session_dict.get("metadata"),
                    "runs": session_dict.get("runs"),
                    "summary": session_dict.get("summary"),
                    "messages": session_dict.get("messages"),
                    "usage": session_dict.get("usage"),
                    "created_at": session_dict.get("created_at") or current_time,
                    "updated_at": current_time,
                }
            else:
                # Fallback for other session types
                session_type_value = self._get_session_type_value(session)
                if "session_type" not in session_dict or session_dict.get("session_type") is None:
                    session_dict["session_type"] = {"session_type": session_type_value}
                session_record = session_dict

            # Find existing session to update
            session_updated = False
            for i, existing_session in enumerate(self._sessions):
                if existing_session.get("session_id") == session_id:
                    # Update existing session
                    session_record["created_at"] = existing_session.get("created_at", current_time)
                    session_record["updated_at"] = current_time
                    self._sessions[i] = deepcopy(session_record)
                    session_updated = True
                    _logger.debug(f"Updated session: {session_id}")
                    break

            if not session_updated:
                # Insert new session
                session_record["created_at"] = session_record.get("created_at") or current_time
                session_record["updated_at"] = session_record.get("updated_at") or current_time
                self._sessions.append(deepcopy(session_record))
                _logger.debug(f"Inserted new session: {session_id}")

            session_dict_copy = deep_copy_record(session_record)

            if not deserialize:
                return session_dict_copy

            if isinstance(session, AgentSession):
                return AgentSession.from_dict(session_dict_copy, deserialize_flag=True)
            return deserialize_session(session_dict_copy)

        except Exception as e:
            _logger.error(f"Error upserting session: {e}")
            raise

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
        """
        if not sessions:
            return []

        try:
            _logger.info(f"Bulk upserting {len(sessions)} sessions")
            results: List[Union["Session", Dict[str, Any]]] = []

            for session in sessions:
                if session is not None:
                    try:
                        result = self.upsert_session(session, deserialize=deserialize)
                        if result is not None:
                            results.append(result)
                    except ValueError:
                        raise
                    except Exception as e:
                        _logger.warning(f"Error upserting session {getattr(session, 'session_id', 'unknown')}: {e}")

            return results

        except ValueError:
            raise
        except Exception as e:
            _logger.error(f"Error bulk upserting sessions: {e}")
            return []

    def get_session(
        self,
        session_id: Optional[str] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Get a session from in-memory storage.
        
        Args:
            session_id: ID of the session to retrieve. If None, returns latest session.
            session_type: Filter by session type.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The session if found, None otherwise.
        """
        try:
            if not self._sessions:
                return None

            # If session_id is provided, do direct lookup
            if session_id is not None:
                for session_data in self._sessions:
                    if session_data.get("session_id") == session_id:
                        # Apply additional filters if provided
                        if user_id is not None and session_data.get("user_id") != user_id:
                            continue
                        if agent_id is not None and session_data.get("agent_id") != agent_id:
                            continue
                        if session_type is not None:
                            stored_type = self._extract_session_type_value(session_data)
                            if stored_type != session_type.value:
                                continue

                        session_copy = deep_copy_record(session_data)

                        if not deserialize:
                            return session_copy

                        return deserialize_session(session_copy, session_type)

                return None

            # No session_id provided - return latest session matching filters
            filtered_sessions = self._filter_sessions(
                session_type=session_type,
                user_id=user_id,
                agent_id=agent_id,
            )

            if not filtered_sessions:
                return None

            # Sort by updated_at descending to get latest
            sorted_sessions = apply_sorting(filtered_sessions, sort_by="updated_at", sort_order="desc")
            latest_session = deep_copy_record(sorted_sessions[0])

            if not deserialize:
                return latest_session

            return deserialize_session(latest_session, session_type)

        except Exception as e:
            _logger.error(f"Error getting session: {e}")
            raise

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
        Get multiple sessions from in-memory storage.
        
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
            # Start with all sessions or filtered by IDs
            if session_ids is not None:
                filtered_sessions = [
                    s for s in self._sessions
                    if s.get("session_id") in session_ids
                ]
            else:
                filtered_sessions = self._sessions.copy()

            # Apply filters
            filtered_sessions = self._filter_sessions(
                sessions=filtered_sessions,
                session_type=session_type,
                user_id=user_id,
                agent_id=agent_id,
            )

            total_count = len(filtered_sessions)

            # Apply sorting
            filtered_sessions = apply_sorting(filtered_sessions, sort_by, sort_order)

            # Apply pagination
            filtered_sessions = apply_pagination(filtered_sessions, limit, offset)

            # Deep copy to prevent mutation
            result_sessions = deep_copy_records(filtered_sessions)

            if not deserialize:
                return result_sessions, total_count

            # Deserialize all sessions
            deserialized_sessions: List["Session"] = []
            for session_data in result_sessions:
                try:
                    session = deserialize_session(session_data, session_type)
                    if session is not None:
                        deserialized_sessions.append(session)
                except Exception as e:
                    _logger.warning(f"Failed to deserialize session: {e}")

            return deserialized_sessions

        except Exception as e:
            _logger.error(f"Error getting sessions: {e}")
            raise

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from in-memory storage.
        
        Args:
            session_id: ID of the session to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If session_id is not provided.
        """
        if not session_id:
            raise ValueError("session_id is required for delete_session")

        try:
            original_count = len(self._sessions)
            self._sessions = [
                s for s in self._sessions
                if s.get("session_id") != session_id
            ]

            if len(self._sessions) < original_count:
                _logger.debug(f"Deleted session: {session_id}")
                return True
            else:
                _logger.debug(f"No session found to delete with id: {session_id}")
                return False

        except Exception as e:
            _logger.error(f"Error deleting session: {e}")
            raise

    def delete_sessions(self, session_ids: List[str]) -> int:
        """
        Delete multiple sessions from in-memory storage.
        
        Args:
            session_ids: List of session IDs to delete.
        
        Returns:
            Number of sessions deleted.
        
        Raises:
            ValueError: If session_ids is empty.
        """
        if not session_ids:
            raise ValueError("session_ids cannot be empty for delete_sessions")

        try:
            original_count = len(self._sessions)
            session_ids_set = set(session_ids)
            self._sessions = [
                s for s in self._sessions
                if s.get("session_id") not in session_ids_set
            ]

            deleted_count = original_count - len(self._sessions)
            _logger.debug(f"Deleted {deleted_count} sessions")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting sessions: {e}")
            raise

    def _filter_sessions(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter sessions based on provided criteria.
        
        Args:
            sessions: List of sessions to filter. Uses self._sessions if None.
            session_type: Filter by session type.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            workflow_id: Filter by workflow ID.
        
        Returns:
            Filtered list of sessions.
        """
        if sessions is None:
            sessions = self._sessions

        filtered: List[Dict[str, Any]] = []

        for session_data in sessions:
            # Filter by session_type
            if session_type is not None:
                stored_type = self._extract_session_type_value(session_data)
                if stored_type != session_type.value:
                    continue

            # Filter by user_id
            if user_id is not None and session_data.get("user_id") != user_id:
                continue

            # Filter by agent_id
            if agent_id is not None and session_data.get("agent_id") != agent_id:
                continue

            # Filter by team_id
            if team_id is not None and session_data.get("team_id") != team_id:
                continue

            # Filter by workflow_id
            if workflow_id is not None and session_data.get("workflow_id") != workflow_id:
                continue

            filtered.append(session_data)

        return filtered

    def _extract_session_type_value(self, session_data: Dict[str, Any]) -> str:
        """
        Extract session type value string from session data.
        
        Args:
            session_data: Session dictionary.
        
        Returns:
            Session type value string (e.g., "agent", "team", "workflow").
        """
        session_type_data = session_data.get("session_type")

        if session_type_data is None:
            return "agent"

        if isinstance(session_type_data, dict):
            return session_type_data.get("session_type", "agent")

        if isinstance(session_type_data, str):
            return session_type_data

        return "agent"

    # ======================== User Memory Methods ========================

    def upsert_user_memory(
        self,
        user_memory: UserMemory,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Insert or update user memory in in-memory storage.
        
        Args:
            user_memory: The memory data to store. Must have 'user_id' key.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The upserted user memory record.
        
        Raises:
            ValueError: If user_id is not provided in user_memory.
        """
        if not user_memory.user_id:
            raise ValueError("user_memory must have user_id set for upsert")

        try:
            current_time = int(time.time())

            # Build the memory record (use 'user_memory' to match UserMemory schema)
            memory_record: Dict[str, Any] = {
                "user_id": user_memory.user_id,
                "user_memory": user_memory.user_memory,
                "agent_id": user_memory.agent_id,
                "team_id": user_memory.team_id,
                "created_at": user_memory.created_at or current_time,
                "updated_at": current_time,
            }

            # Find existing memory to update
            memory_updated = False
            for i, existing_memory in enumerate(self._user_memories):
                if existing_memory.get("user_id") == user_memory.user_id:
                    # Match on agent_id and team_id if provided
                    if user_memory.agent_id is not None and existing_memory.get("agent_id") != user_memory.agent_id:
                        continue
                    if user_memory.team_id is not None and existing_memory.get("team_id") != user_memory.team_id:
                        continue

                    # Update existing memory
                    memory_record["created_at"] = existing_memory.get("created_at", user_memory.created_at or current_time)
                    self._user_memories[i] = deepcopy(memory_record)
                    memory_updated = True
                    _logger.debug(f"Updated user memory for user: {user_memory.user_id}")
                    break

            if not memory_updated:
                # Insert new memory
                self._user_memories.append(deepcopy(memory_record))
                _logger.debug(f"Inserted new user memory for user: {user_memory.user_id}")

            memory_dict = deep_copy_record(memory_record)
            if deserialize:
                return UserMemory.from_dict(memory_dict)
            return memory_dict

        except Exception as e:
            _logger.error(f"Error upserting user memory: {e}")
            raise

    def upsert_user_memories(
        self,
        user_memories: List[UserMemory],
        deserialize: bool = True,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        """
        Bulk insert or update multiple user memories.
        
        Args:
            user_memories: List of user memory dicts. Each must have 'user_id'.
            deserialize: If True, return UserMemory objects. If False, return raw dicts.
        
        Returns:
            List of upserted user memory records.
        
        Raises:
            ValueError: If any record is missing user_id.
        """
        if not user_memories:
            return []

        try:
            _logger.info(f"Bulk upserting {len(user_memories)} user memories")
            results: List[Union[UserMemory, Dict[str, Any]]] = []

            for memory in user_memories:
                if not memory.user_id:
                    raise ValueError("user_id is required in each UserMemory instance")

                result = self.upsert_user_memory(
                    user_memory=memory,
                    deserialize=deserialize,
                )
                if result is not None:
                    results.append(result)

            return results

        except Exception as e:
            _logger.error(f"Error bulk upserting user memories: {e}")
            raise

    def get_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Get user memory from in-memory storage.
        
        Args:
            user_id: User ID to retrieve. If None, returns latest memory.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return UserMemory object.
                        If False, return raw dictionary.
        
        Returns:
            UserMemory object or dict if found, None otherwise.
        """
        try:
            if not self._user_memories:
                return None

            # If user_id is provided, do direct lookup
            if user_id is not None:
                for memory_data in self._user_memories:
                    if memory_data.get("user_id") == user_id:
                        # Apply additional filters if provided
                        if agent_id is not None and memory_data.get("agent_id") != agent_id:
                            continue
                        if team_id is not None and memory_data.get("team_id") != team_id:
                            continue

                        memory_raw = deep_copy_record(memory_data)
                        
                        if not deserialize:
                            return memory_raw
                        
                        return UserMemory.from_dict(memory_raw)

                return None

            # No user_id provided - return latest memory matching filters
            filtered_memories = self._filter_user_memories(
                agent_id=agent_id,
                team_id=team_id,
            )

            if not filtered_memories:
                return None

            # Sort by updated_at descending to get latest
            sorted_memories = apply_sorting(filtered_memories, sort_by="updated_at", sort_order="desc")
            memory_raw = deep_copy_record(sorted_memories[0])
            
            if not deserialize:
                return memory_raw
            
            return UserMemory.from_dict(memory_raw)

        except Exception as e:
            _logger.error(f"Error getting user memory: {e}")
            raise

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
        Get multiple user memories from in-memory storage.
        
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
        try:
            # Start with all memories or filtered by IDs
            if user_ids is not None:
                filtered_memories = [
                    m for m in self._user_memories
                    if m.get("user_id") in user_ids
                ]
            else:
                filtered_memories = self._user_memories.copy()

            # Apply filters
            filtered_memories = self._filter_user_memories(
                memories=filtered_memories,
                agent_id=agent_id,
                team_id=team_id,
            )

            total_count = len(filtered_memories)

            # Sort by updated_at descending by default
            filtered_memories = apply_sorting(filtered_memories, sort_by="updated_at", sort_order="desc")

            # Apply pagination
            filtered_memories = apply_pagination(filtered_memories, limit, offset)

            # Deep copy to prevent mutation
            result_memories = deep_copy_records(filtered_memories)

            if not deserialize:
                return result_memories, total_count

            return [UserMemory.from_dict(memory) for memory in result_memories]

        except Exception as e:
            _logger.error(f"Error getting user memories: {e}")
            raise

    def delete_user_memory(self, user_id: str) -> bool:
        """
        Delete user memory from in-memory storage.
        
        Args:
            user_id: User ID of the memory to delete.
        
        Returns:
            True if deleted successfully, False otherwise.
        
        Raises:
            ValueError: If user_id is not provided.
        """
        if not user_id:
            raise ValueError("user_id is required for delete_user_memory")

        try:
            original_count = len(self._user_memories)
            self._user_memories = [
                m for m in self._user_memories
                if m.get("user_id") != user_id
            ]

            if len(self._user_memories) < original_count:
                _logger.debug(f"Deleted user memory for user: {user_id}")
                return True
            else:
                _logger.debug(f"No user memory found to delete for user: {user_id}")
                return False

        except Exception as e:
            _logger.error(f"Error deleting user memory: {e}")
            raise

    def delete_user_memories(self, user_ids: List[str]) -> int:
        """
        Delete multiple user memories from in-memory storage.
        
        Args:
            user_ids: List of user IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            ValueError: If user_ids is empty.
        """
        if not user_ids:
            raise ValueError("user_ids cannot be empty for delete_user_memories")

        try:
            original_count = len(self._user_memories)
            user_ids_set = set(user_ids)
            self._user_memories = [
                m for m in self._user_memories
                if m.get("user_id") not in user_ids_set
            ]

            deleted_count = original_count - len(self._user_memories)
            _logger.debug(f"Deleted {deleted_count} user memories")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting user memories: {e}")
            raise

    def _filter_user_memories(
        self,
        memories: Optional[List[Dict[str, Any]]] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter user memories based on provided criteria.
        
        Args:
            memories: List of memories to filter. Uses self._user_memories if None.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
        
        Returns:
            Filtered list of user memories.
        """
        if memories is None:
            memories = self._user_memories

        filtered: List[Dict[str, Any]] = []

        for memory_data in memories:
            # Filter by agent_id
            if agent_id is not None and memory_data.get("agent_id") != agent_id:
                continue

            # Filter by team_id
            if team_id is not None and memory_data.get("team_id") != team_id:
                continue

            filtered.append(memory_data)

        return filtered

    # ======================== Utility Methods ========================

    def clear_all(self) -> None:
        """
        Clear all data from all tables.
        
        This removes all sessions, user memories, and knowledge from storage.
        """
        try:
            self._sessions.clear()
            self._user_memories.clear()
            self._knowledge.clear()
            _logger.info("Cleared all data from InMemoryStorage")

        except Exception as e:
            _logger.error(f"Error clearing all data: {e}")
            raise

    def get_session_count(self) -> int:
        """
        Get the total number of sessions in storage.
        
        Returns:
            Number of sessions.
        """
        return len(self._sessions)

    def get_user_memory_count(self) -> int:
        """
        Get the total number of user memories in storage.
        
        Returns:
            Number of user memories.
        """
        return len(self._user_memories)

    # ======================== Generic Model Methods ========================

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
        """
        try:
            if collection not in self._generic_models:
                self._generic_models[collection] = {}
            
            # Store model as dict
            if hasattr(model, 'model_dump'):
                model_data = model.model_dump()
            elif hasattr(model, 'dict'):
                model_data = model.dict()
            else:
                model_data = dict(model)
            
            self._generic_models[collection][key] = model_data
            _logger.debug(f"Upserted model with key '{key}' in collection '{collection}'")
            
        except Exception as e:
            _logger.error(f"Error upserting model: {e}")
            raise

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
        """
        try:
            if collection not in self._generic_models:
                return None
            
            model_data = self._generic_models[collection].get(key)
            if model_data is None:
                return None
            
            # Deserialize to model type
            return model_type(**model_data)
            
        except Exception as e:
            _logger.error(f"Error getting model: {e}")
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
        """
        try:
            if collection not in self._generic_models:
                return False
            
            if key in self._generic_models[collection]:
                del self._generic_models[collection][key]
                _logger.debug(f"Deleted model with key '{key}' from collection '{collection}'")
                return True
            
            return False
            
        except Exception as e:
            _logger.error(f"Error deleting model: {e}")
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
        """
        try:
            if collection not in self._generic_models:
                return []
            
            models = []
            for model_data in self._generic_models[collection].values():
                try:
                    models.append(model_type(**model_data))
                except Exception as e:
                    _logger.warning(f"Failed to deserialize model: {e}")
            
            return models
            
        except Exception as e:
            _logger.error(f"Error listing models: {e}")
            return []

    # ======================== Cultural Knowledge Methods ========================

    def delete_cultural_knowledge(self, id: str) -> None:
        """Delete cultural knowledge from in-memory storage."""
        try:
            self._cultural_knowledge = [
                ck for ck in self._cultural_knowledge if ck.get("id") != id
            ]
            _logger.debug(f"Deleted cultural knowledge with id: {id}")
        except Exception as e:
            _logger.error(f"Error deleting cultural knowledge: {e}")
            raise e

    def get_cultural_knowledge(
        self,
        id: str,
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Get cultural knowledge from in-memory storage."""
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            for ck_data in self._cultural_knowledge:
                if ck_data.get("id") == id:
                    ck_data_copy = deep_copy_record(ck_data)
                    if not deserialize:
                        return ck_data_copy
                    return CulturalKnowledge.from_dict(ck_data_copy)
            return None
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
        """Get all cultural knowledge from in-memory storage."""
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            filtered_ck: List[Dict[str, Any]] = []
            for ck_data in self._cultural_knowledge:
                if name is not None and ck_data.get("name") != name:
                    continue
                if agent_id is not None and ck_data.get("agent_id") != agent_id:
                    continue
                if team_id is not None and ck_data.get("team_id") != team_id:
                    continue
                filtered_ck.append(ck_data)

            # Apply sorting
            if sort_by:
                filtered_ck = apply_sorting(filtered_ck, sort_by, sort_order)

            total_count = len(filtered_ck)

            # Apply pagination
            if limit and page:
                start = (page - 1) * limit
                filtered_ck = filtered_ck[start : start + limit]
            elif limit:
                filtered_ck = filtered_ck[:limit]

            if not deserialize:
                return deep_copy_records(filtered_ck), total_count
            return [CulturalKnowledge.from_dict(deep_copy_record(ck)) for ck in filtered_ck]

        except Exception as e:
            _logger.error(f"Error getting all cultural knowledge: {e}")
            raise e

    def upsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Upsert cultural knowledge into in-memory storage."""
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        import uuid
        import time
        
        try:
            if cultural_knowledge.id is None:
                cultural_knowledge.id = str(uuid.uuid4())

            # Use to_dict for serialization
            data = cultural_knowledge.to_dict()
            
            # Convert RFC3339 timestamps to epoch seconds for storage
            if "created_at" in data and data["created_at"] is not None:
                if isinstance(data["created_at"], str):
                    from upsonic.utils.dttm import to_epoch_s
                    data["created_at"] = to_epoch_s(data["created_at"])
            if "updated_at" in data and data["updated_at"] is not None:
                if isinstance(data["updated_at"], str):
                    from upsonic.utils.dttm import to_epoch_s
                    data["updated_at"] = to_epoch_s(data["updated_at"])
            else:
                data["updated_at"] = int(time.time())

            # Remove existing entry with same id
            self._cultural_knowledge = [
                ck for ck in self._cultural_knowledge if ck.get("id") != cultural_knowledge.id
            ]

            # Add new entry
            self._cultural_knowledge.append(data)

            if not deserialize:
                return deep_copy_record(data)
            return CulturalKnowledge.from_dict(data)

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
            current_time = int(time.time())
            data = knowledge_row.to_dict()
            data.setdefault("created_at", current_time)
            data["updated_at"] = current_time

            self._knowledge = [
                item for item in self._knowledge if item.get("id") != data["id"]
            ]
            self._knowledge.append(deepcopy(data))

            _logger.debug(f"Upserted knowledge content: {data['id']}")
            return KnowledgeRow.from_dict(data)

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
            for item in self._knowledge:
                if item.get("id") == id:
                    return KnowledgeRow.from_dict(deep_copy_record(item))
            return None

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
            filtered: List[Dict[str, Any]] = []
            for item in self._knowledge:
                if knowledge_base_id is not None and item.get("knowledge_base_id") != knowledge_base_id:
                    continue
                filtered.append(item)

            total_count: int = len(filtered)

            sorted_items = apply_sorting(
                filtered,
                sort_by=sort_by or "created_at",
                sort_order=sort_order or "desc",
            )

            paginated = apply_pagination(sorted_items, limit, page)

            rows = [KnowledgeRow.from_dict(deep_copy_record(item)) for item in paginated]
            return rows, total_count

        except Exception as e:
            _logger.error(f"Error getting knowledge contents: {e}")
            raise e

    def delete_knowledge_content(self, id: str) -> bool:
        """Delete a knowledge document registry entry by ID."""
        try:
            original_count = len(self._knowledge)
            self._knowledge = [
                item for item in self._knowledge if item.get("id") != id
            ]

            if len(self._knowledge) < original_count:
                _logger.debug(f"Deleted knowledge content: {id}")
                return True
            return False

        except Exception as e:
            _logger.error(f"Error deleting knowledge content: {e}")
            raise e

    def delete_knowledge_contents(self, ids: List[str]) -> int:
        """Delete multiple knowledge document registry entries."""
        if not ids:
            return 0

        try:
            ids_set = set(ids)
            original_count = len(self._knowledge)
            self._knowledge = [
                item for item in self._knowledge if item.get("id") not in ids_set
            ]

            deleted_count: int = original_count - len(self._knowledge)
            _logger.debug(f"Deleted {deleted_count} knowledge entries")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting knowledge contents: {e}")
            raise e
