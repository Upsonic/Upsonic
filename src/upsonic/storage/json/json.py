"""Synchronous JSON file storage implementation for Upsonic agent framework."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from upsonic.session.base import SessionType, Session
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.schemas import KnowledgeRow

from upsonic.storage.base import Storage
from upsonic.storage.json.utils import (
    apply_pagination,
    apply_sorting,
    filter_sessions,
    filter_user_memories,
)
from upsonic.storage.utils import deserialize_session, deserialize_session_json_fields
from upsonic.storage.schemas import UserMemory
from upsonic.utils.logging_config import get_logger

_logger = get_logger("upsonic.storage.json")


class JSONStorage(Storage):
    """Synchronous JSON file storage implementation for Upsonic agent framework.
    
    This storage backend uses JSON files for persistent storage. It provides
    a simple file-based persistence mechanism for agent sessions and user memory data.
    
    The JSON files are stored in a directory structure:
        db_path/
            {session_table_name}.json
            {user_memory_table_name}.json
    
    Example:
        ```python
        storage = JSONStorage(db_path="./agent_data")
        
        # Upsert a session
        session = AgentSession(session_id="abc123", ...)
        result = storage.upsert_session(session)
        
        # Get a session
        session = storage.get_session(session_id="abc123")
        
        # Get latest session if no ID provided
        latest = storage.get_session()
        ```
    
    Args:
        db_path: Path to the directory where JSON files will be stored.
            Defaults to "./upsonic_json_db" in the current working directory.
        session_table: Name of the JSON file for sessions (without .json extension).
        user_memory_table: Name of the JSON file for user memories.
        id: Unique identifier for this storage instance.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize the JSON file storage.
        
        Args:
            db_path: Path to the directory where JSON files will be stored.
            session_table: Name of the session table (JSON file name without extension).
            user_memory_table: Name of the user memory table (JSON file name without extension).
            knowledge_table: Name of the knowledge table (JSON file name without extension).
            id: Unique identifier for this storage instance.
        """
        super().__init__(
            session_table=session_table,
            user_memory_table=user_memory_table,
            knowledge_table=knowledge_table,
            id=id,
        )

        # Set up the database directory path
        self.db_path: Path = Path(
            db_path or os.path.join(os.getcwd(), "upsonic_json_db")
        )

    # ======================== Table Management ========================

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table (JSON file) exists.
        
        For JSON storage, we always return True as files are created on demand.
        
        Args:
            table_name: Name of the table to check.
        
        Returns:
            True (files are created on demand).
        """
        return True

    def _create_all_tables(self) -> None:
        """Create all required tables (JSON files) for this storage."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self._read_json_file(self.session_table_name, create_if_not_found=True)
        self._read_json_file(self.user_memory_table_name, create_if_not_found=True)
        self._read_json_file(self.knowledge_table_name, create_if_not_found=True)

    def close(self) -> None:
        """Close the storage (no-op for JSON file storage)."""
        pass

    # ======================== File I/O Methods ========================

    def _read_json_file(
        self,
        table_name: str,
        create_if_not_found: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Read data from a JSON file, creating it if it doesn't exist.
        
        Args:
            table_name: The name of the JSON file (without .json extension).
            create_if_not_found: Whether to create the file if it doesn't exist.
        
        Returns:
            List of dictionaries from the JSON file.
        
        Raises:
            json.JSONDecodeError: If the JSON file is malformed.
            FileNotFoundError: If file doesn't exist and create_if_not_found is False.
        """
        file_path = self.db_path / f"{table_name}.json"

        # Create directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        except FileNotFoundError:
            if create_if_not_found:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
                return []
            raise

        except json.JSONDecodeError as e:
            _logger.error(f"Error reading JSON file {file_path}: {e}")
            raise e

    def _write_json_file(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
    ) -> None:
        """
        Write data to a JSON file.
        
        Args:
            table_name: The name of the JSON file (without .json extension).
            data: List of dictionaries to write to the JSON file.
        
        Raises:
            Exception: If an error occurs while writing.
        """
        file_path = self.db_path / f"{table_name}.json"

        # Create directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            _logger.error(f"Error writing to JSON file {file_path}: {e}")
            raise e

    def _get_session_type_value(self, session: "Session") -> str:
        """
        Extract session type value string from session object.
        
        Args:
            session: Session object to extract type from.
        
        Returns:
            String value of the session type.
        """
        session_type_value = "agent"
        if hasattr(session, "session_type"):
            st = session.session_type
            if hasattr(st, "value"):
                session_type_value = st.value
            elif isinstance(st, str):
                session_type_value = st
        return session_type_value

    # ======================== Session Methods ========================

    def upsert_session(
        self,
        session: "Session",
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Insert or update a session in the JSON file.
        
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

        if not hasattr(session, "session_id") or session.session_id is None:
            raise ValueError("Session must have session_id set for upsert")

        try:
            sessions = self._read_json_file(
                self.session_table_name,
                create_if_not_found=True,
            )

            # Convert session to dict with serialize_flag=True
            session_dict = (
                session.to_dict(serialize_flag=True) if hasattr(session, "to_dict") else dict(session)
            )
            current_time = int(time.time())

            # Prepare the session record based on session type
            if isinstance(session, AgentSession):
                session_record = {
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
                    "created_at": session_dict.get("created_at") or current_time,
                    "updated_at": current_time,
                }
            else:
                # Fallback for other session types (TeamSession, WorkflowSession) - not yet implemented
                session_type_value = self._get_session_type_value(session)
                session_record = {
                    "session_id": session_dict.get("session_id"),
                    "session_type": session_type_value,
                    "agent_id": session_dict.get("agent_id"),
                    "team_id": session_dict.get("team_id"),
                    "workflow_id": session_dict.get("workflow_id"),
                    "user_id": session_dict.get("user_id"),
                    "session_data": session_dict.get("session_data"),
                    "agent_data": session_dict.get("agent_data"),
                    "team_data": session_dict.get("team_data"),
                    "workflow_data": session_dict.get("workflow_data"),
                    "metadata": session_dict.get("metadata"),
                    "runs": session_dict.get("runs"),
                    "summary": session_dict.get("summary"),
                    "messages": session_dict.get("messages"),
                    "created_at": session_dict.get("created_at") or current_time,
                    "updated_at": current_time,
                }

            # Find and update or insert
            session_updated = False
            for i, existing_session in enumerate(sessions):
                if existing_session.get("session_id") == session_record["session_id"]:
                    # Preserve original created_at
                    session_record["created_at"] = existing_session.get(
                        "created_at", current_time
                    )
                    sessions[i] = session_record
                    session_updated = True
                    _logger.debug(f"Updated session: {session_record['session_id']}")
                    break

            if not session_updated:
                sessions.append(session_record)
                _logger.debug(f"Created session: {session_record['session_id']}")

            self._write_json_file(self.session_table_name, sessions)

            session_raw = deserialize_session_json_fields(session_record)

            if not deserialize:
                return session_raw

            if isinstance(session, AgentSession):
                return AgentSession.from_dict(session_raw, deserialize_flag=True)
            return deserialize_session(session_raw)

        except Exception as e:
            _logger.error(f"Error upserting session: {e}")
            raise e

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

        try:
            existing_sessions = self._read_json_file(
                self.session_table_name,
                create_if_not_found=True,
            )
            results: List[Union["Session", Dict[str, Any]]] = []
            current_time = int(time.time())

            # Build a map of existing sessions for efficient lookup
            sessions_map: Dict[str, int] = {
                s.get("session_id"): i for i, s in enumerate(existing_sessions)
            }

            for session in sessions:
                session_dict = (
                    session.to_dict() if hasattr(session, "to_dict") else dict(session)
                )
                session_type_value = self._get_session_type_value(session)

                session_record = {
                    "session_id": session_dict.get("session_id"),
                    "session_type": session_type_value,
                    "agent_id": session_dict.get("agent_id"),
                    "team_id": session_dict.get("team_id"),
                    "workflow_id": session_dict.get("workflow_id"),
                    "user_id": session_dict.get("user_id"),
                    "session_data": session_dict.get("session_data"),
                    "agent_data": session_dict.get("agent_data"),
                    "team_data": session_dict.get("team_data"),
                    "workflow_data": session_dict.get("workflow_data"),
                    "metadata": session_dict.get("metadata"),
                    "runs": session_dict.get("runs"),
                    "summary": session_dict.get("summary"),
                    "messages": session_dict.get("messages"),
                    "created_at": session_dict.get("created_at") or current_time,
                    "updated_at": current_time,
                }

                session_id = session_record["session_id"]

                if session_id in sessions_map:
                    # Update existing
                    idx = sessions_map[session_id]
                    session_record["created_at"] = existing_sessions[idx].get(
                        "created_at", current_time
                    )
                    existing_sessions[idx] = session_record
                else:
                    # Insert new
                    existing_sessions.append(session_record)
                    sessions_map[session_id] = len(existing_sessions) - 1

                session_raw = deserialize_session_json_fields(session_record)

                if deserialize:
                    results.append(deserialize_session(session_raw))
                else:
                    results.append(session_raw)

            self._write_json_file(self.session_table_name, existing_sessions)

            _logger.debug(f"Upserted {len(results)} sessions")
            return results

        except Exception as e:
            _logger.error(f"Error upserting sessions: {e}")
            raise e

    def get_session(
        self,
        session_id: Optional[str] = None,
        session_type: Optional["SessionType"] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Get a session from the JSON file.
        
        If session_id is not provided, returns the latest session
        matching any other filter criteria.
        
        Args:
            session_id: ID of the session to retrieve.
            session_type: Optional filter by session type.
            user_id: Optional filter by user ID.
            agent_id: Optional filter by agent ID.
            deserialize: If True, return deserialized Session object.
                        If False, return raw dictionary.
        
        Returns:
            The session if found, None otherwise.
        """
        try:
            sessions = self._read_json_file(
                self.session_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return None

        try:
            # Get session type value for filtering
            session_type_value: Optional[str] = None
            if session_type is not None:
                session_type_value = (
                    session_type.value
                    if hasattr(session_type, "value")
                    else str(session_type)
                )

            # Filter sessions
            filtered = filter_sessions(
                sessions=sessions,
                session_ids=[session_id] if session_id else None,
                session_type=session_type_value,
                user_id=user_id,
                agent_id=agent_id,
            )

            if not filtered:
                return None

            # If no session_id provided, get the latest
            if session_id is None:
                filtered = apply_sorting(filtered, sort_by="created_at", sort_order="desc")
                session_raw = filtered[0]
            else:
                session_raw = filtered[0]

            session_raw = deserialize_session_json_fields(session_raw)

            if not deserialize:
                return session_raw

            return deserialize_session(session_raw, session_type)

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
        Get multiple sessions from the JSON file.
        
        If session_ids is not provided, returns all sessions matching
        the filter criteria.
        
        Args:
            session_ids: List of session IDs to retrieve.
            session_type: Optional filter by session type.
            user_id: Optional filter by user ID.
            agent_id: Optional filter by agent ID.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            sort_by: Field to sort by (default: "created_at").
            sort_order: Sort order - "asc" or "desc" (default: "desc").
            deserialize: If True, return list of deserialized Session objects.
                        If False, return tuple of (raw dictionaries, total count).
        
        Returns:
            List of sessions or tuple of (list of dicts, total count).
        """
        try:
            sessions = self._read_json_file(
                self.session_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return [] if deserialize else ([], 0)

        try:
            # Get session type value for filtering
            session_type_value: Optional[str] = None
            if session_type is not None:
                session_type_value = (
                    session_type.value
                    if hasattr(session_type, "value")
                    else str(session_type)
                )

            # Filter sessions
            filtered = filter_sessions(
                sessions=sessions,
                session_ids=session_ids,
                session_type=session_type_value,
                user_id=user_id,
                agent_id=agent_id,
            )

            total_count = len(filtered)

            # Apply sorting
            filtered = apply_sorting(filtered, sort_by, sort_order)

            # Apply pagination
            filtered = apply_pagination(filtered, limit, offset)

            if not filtered:
                return [] if deserialize else ([], 0)

            # Deserialize JSON fields
            sessions_raw = [deserialize_session_json_fields(s) for s in filtered]

            if not deserialize:
                return sessions_raw, total_count

            return [deserialize_session(s, session_type) for s in sessions_raw]

        except Exception as e:
            _logger.error(f"Error getting sessions: {e}")
            raise e

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the JSON file.
        
        Args:
            session_id: ID of the session to delete.
        
        Returns:
            True if deleted successfully, False if not found.
        
        Raises:
            ValueError: If session_id is not provided.
        """
        if not session_id:
            raise ValueError("session_id is required for delete")

        try:
            sessions = self._read_json_file(
                self.session_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return False

        try:
            original_count = len(sessions)
            sessions = [s for s in sessions if s.get("session_id") != session_id]

            if len(sessions) < original_count:
                self._write_json_file(self.session_table_name, sessions)
                _logger.debug(f"Deleted session: {session_id}")
                return True

            _logger.debug(f"No session found to delete: {session_id}")
            return False

        except Exception as e:
            _logger.error(f"Error deleting session: {e}")
            raise e

    def delete_sessions(self, session_ids: List[str]) -> int:
        """
        Delete multiple sessions from the JSON file.
        
        Args:
            session_ids: List of session IDs to delete.
        
        Returns:
            Number of sessions deleted.
        
        Raises:
            ValueError: If session_ids is empty or not provided.
        """
        if not session_ids:
            raise ValueError("session_ids is required and cannot be empty")

        try:
            sessions = self._read_json_file(
                self.session_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return 0

        try:
            original_count = len(sessions)
            session_ids_set = set(session_ids)
            sessions = [s for s in sessions if s.get("session_id") not in session_ids_set]

            deleted_count = original_count - len(sessions)

            if deleted_count > 0:
                self._write_json_file(self.session_table_name, sessions)

            _logger.debug(f"Deleted {deleted_count} sessions")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting sessions: {e}")
            raise e

    # ======================== User Memory Methods ========================

    def upsert_user_memory(
        self,
        user_memory: UserMemory,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Insert or update user memory in the JSON file.
        
        Args:
            user_memory: The memory data to store. Must have 'user_id' key.
            deserialize: If True, return full record.
        
        Returns:
            The upserted user memory record.
        
        Raises:
            ValueError: If user_id is not provided in user_memory.
        """
        if not user_memory.user_id:
            raise ValueError("user_memory must have user_id set for upsert")

        try:
            memories = self._read_json_file(
                self.user_memory_table_name,
                create_if_not_found=True,
            )
            current_time = int(time.time())

            # Prepare the memory record
            memory_record = {
                "user_id": user_memory.user_id,
                "user_memory": user_memory.user_memory,
                "agent_id": user_memory.agent_id,
                "team_id": user_memory.team_id,
                "created_at": user_memory.created_at or current_time,
                "updated_at": current_time,
            }

            # Find and update or insert
            memory_updated = False
            for i, existing_memory in enumerate(memories):
                if existing_memory.get("user_id") == user_memory.user_id:
                    # Preserve original created_at
                    memory_record["created_at"] = existing_memory.get(
                        "created_at", user_memory.created_at or current_time
                    )
                    memories[i] = memory_record
                    memory_updated = True
                    _logger.debug(f"Updated user memory: {user_memory.user_id}")
                    break

            if not memory_updated:
                memories.append(memory_record)
                _logger.debug(f"Created user memory: {user_memory.user_id}")

            self._write_json_file(self.user_memory_table_name, memories)

            if deserialize:
                return UserMemory.from_dict(memory_record)
            return memory_record

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
            user_memories: List of user memory dicts. Each must have 'user_id'.
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

        try:
            existing_memories = self._read_json_file(
                self.user_memory_table_name,
                create_if_not_found=True,
            )
            results: List[Union[UserMemory, Dict[str, Any]]] = []
            current_time = int(time.time())

            # Build a map of existing memories for efficient lookup
            memories_map: Dict[str, int] = {
                m.get("user_id"): i for i, m in enumerate(existing_memories)
            }

            for memory in user_memories:
                memory_record = {
                    "user_id": memory.user_id,
                    "user_memory": memory.user_memory,
                    "agent_id": memory.agent_id,
                    "team_id": memory.team_id,
                    "created_at": memory.created_at or current_time,
                    "updated_at": current_time,
                }

                user_id = memory_record["user_id"]

                if user_id in memories_map:
                    # Update existing
                    idx = memories_map[user_id]
                    memory_record["created_at"] = existing_memories[idx].get(
                        "created_at", memory.created_at or current_time
                    )
                    existing_memories[idx] = memory_record
                else:
                    # Insert new
                    existing_memories.append(memory_record)
                    memories_map[user_id] = len(existing_memories) - 1

                if deserialize:
                    results.append(UserMemory.from_dict(memory_record))
                else:
                    results.append(memory_record)

            self._write_json_file(self.user_memory_table_name, existing_memories)

            _logger.debug(f"Upserted {len(results)} user memories")
            return results

        except Exception as e:
            _logger.error(f"Error upserting user memories: {e}")
            raise e

    def get_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Get user memory from the JSON file.
        
        If user_id is not provided, returns the latest user memory
        matching any other filter criteria.
        
        Args:
            user_id: User ID to retrieve.
            agent_id: Optional filter by agent ID.
            team_id: Optional filter by team ID.
            deserialize: If True, return UserMemory object.
                        If False, return raw dictionary.
        
        Returns:
            UserMemory object or dict if found, None otherwise.
        """
        try:
            memories = self._read_json_file(
                self.user_memory_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return None

        try:
            # Filter memories
            filtered = filter_user_memories(
                memories=memories,
                user_ids=[user_id] if user_id else None,
                agent_id=agent_id,
                team_id=team_id,
            )

            if not filtered:
                return None

            # If no user_id provided, get the latest
            if user_id is None:
                filtered = apply_sorting(filtered, sort_by="updated_at", sort_order="desc")
                memory_raw = filtered[0]
            else:
                memory_raw = filtered[0]

            if not deserialize:
                return memory_raw

            return UserMemory.from_dict(memory_raw)

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
        Get multiple user memories from the JSON file.
        
        If user_ids is not provided, returns all user memories matching
        the filter criteria.
        
        Args:
            user_ids: List of user IDs to retrieve.
            agent_id: Optional filter by agent ID.
            team_id: Optional filter by team ID.
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            deserialize: If True, return list of UserMemory objects.
                        If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of UserMemory objects or tuple of (list of dicts, total count).
        """
        try:
            memories = self._read_json_file(
                self.user_memory_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return [] if deserialize else ([], 0)

        try:
            # Filter memories
            filtered = filter_user_memories(
                memories=memories,
                user_ids=user_ids,
                agent_id=agent_id,
                team_id=team_id,
            )

            total_count = len(filtered)

            # Apply sorting (by updated_at desc)
            filtered = apply_sorting(filtered, sort_by="updated_at", sort_order="desc")

            # Apply pagination
            filtered = apply_pagination(filtered, limit, offset)

            if not filtered:
                return [] if deserialize else ([], 0)

            if not deserialize:
                return filtered, total_count

            return [UserMemory.from_dict(m) for m in filtered]

        except Exception as e:
            _logger.error(f"Error getting user memories: {e}")
            raise e

    def delete_user_memory(self, user_id: str) -> bool:
        """
        Delete user memory from the JSON file.
        
        Args:
            user_id: User ID of the memory to delete.
        
        Returns:
            True if deleted successfully, False if not found.
        
        Raises:
            ValueError: If user_id is not provided.
        """
        if not user_id:
            raise ValueError("user_id is required for delete")

        try:
            memories = self._read_json_file(
                self.user_memory_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return False

        try:
            original_count = len(memories)
            memories = [m for m in memories if m.get("user_id") != user_id]

            if len(memories) < original_count:
                self._write_json_file(self.user_memory_table_name, memories)
                _logger.debug(f"Deleted user memory: {user_id}")
                return True

            _logger.debug(f"No user memory found to delete: {user_id}")
            return False

        except Exception as e:
            _logger.error(f"Error deleting user memory: {e}")
            raise e

    def delete_user_memories(self, user_ids: List[str]) -> int:
        """
        Delete multiple user memories from the JSON file.
        
        Args:
            user_ids: List of user IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            ValueError: If user_ids is empty or not provided.
        """
        if not user_ids:
            raise ValueError("user_ids is required and cannot be empty")

        try:
            memories = self._read_json_file(
                self.user_memory_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return 0

        try:
            original_count = len(memories)
            user_ids_set = set(user_ids)
            memories = [m for m in memories if m.get("user_id") not in user_ids_set]

            deleted_count = original_count - len(memories)

            if deleted_count > 0:
                self._write_json_file(self.user_memory_table_name, memories)

            _logger.debug(f"Deleted {deleted_count} user memories")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting user memories: {e}")
            raise e

    # ======================== Utility Methods ========================

    def clear_all(self) -> None:
        """
        Clear all data from all tables (JSON files).
        
        This removes all sessions and user memories from the storage.
        """
        try:
            # Clear sessions
            try:
                self._write_json_file(self.session_table_name, [])
                _logger.debug("Cleared all sessions")
            except Exception:
                pass

            # Clear user memories
            try:
                self._write_json_file(self.user_memory_table_name, [])
                _logger.debug("Cleared all user memories")
            except Exception:
                pass

            # Clear knowledge
            try:
                self._write_json_file(self.knowledge_table_name, [])
                _logger.debug("Cleared all knowledge entries")
            except Exception:
                pass

            _logger.info("Cleared all data from storage")

        except Exception as e:
            _logger.error(f"Error clearing all data: {e}")
            raise e

    # ======================== Generic Model Methods ========================

    def _get_model_file_name(self, collection: str) -> str:
        """Get the JSON file name for a model collection."""
        return f"models_{collection}"

    def upsert_model(
        self,
        key: str,
        model: Any,
        collection: str = "generic_models",
    ) -> None:
        """Insert or update a generic Pydantic model in storage."""
        try:
            file_name = self._get_model_file_name(collection)
            current_time = int(time.time())
            
            # Serialize model
            if hasattr(model, 'model_dump'):
                model_data = model.model_dump(mode='json')
            elif hasattr(model, 'dict'):
                model_data = model.dict()
            else:
                model_data = dict(model)
            
            # Load existing data
            models = self._read_json_file(file_name)
            
            # Find and update or append
            found = False
            for i, m in enumerate(models):
                if m.get("key") == key:
                    models[i] = {
                        "key": key,
                        "collection": collection,
                        "model_data": model_data,
                        "created_at": m.get("created_at", current_time),
                        "updated_at": current_time,
                    }
                    found = True
                    break
            
            if not found:
                models.append({
                    "key": key,
                    "collection": collection,
                    "model_data": model_data,
                    "created_at": current_time,
                    "updated_at": current_time,
                })
            
            self._write_json_file(file_name, models)
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
        """Retrieve a generic Pydantic model from storage."""
        try:
            file_name = self._get_model_file_name(collection)
            models = self._read_json_file(file_name)
            
            for m in models:
                if m.get("key") == key:
                    model_data = m.get("model_data")
                    if model_data:
                        return model_type(**model_data)
            
            return None
            
        except Exception as e:
            _logger.error(f"Error getting model: {e}")
            return None

    def delete_model(
        self,
        key: str,
        collection: str = "generic_models",
    ) -> bool:
        """Delete a generic model from storage."""
        try:
            file_name = self._get_model_file_name(collection)
            models = self._read_json_file(file_name)
            
            original_count = len(models)
            models = [m for m in models if m.get("key") != key]
            
            if len(models) < original_count:
                self._write_json_file(file_name, models)
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
        """List all models of a type in a collection."""
        try:
            file_name = self._get_model_file_name(collection)
            models_data = self._read_json_file(file_name)
            
            models = []
            for m in models_data:
                if m.get("collection") == collection:
                    try:
                        model_data = m.get("model_data")
                        if model_data:
                            models.append(model_type(**model_data))
                    except Exception as e:
                        _logger.warning(f"Failed to deserialize model: {e}")
            
            return models
            
        except Exception as e:
            _logger.error(f"Error listing models: {e}")
            return []

    # ======================== Cultural Knowledge Methods ========================

    def delete_cultural_knowledge(self, id: str) -> None:
        """Delete cultural knowledge from JSON storage.
        
        Args:
            id: The ID of the cultural knowledge to delete.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        try:
            file_name = "cultural_knowledge"
            all_items = self._read_json_file(file_name)
            filtered_items = [ck for ck in all_items if ck.get("id") != id]
            self._write_json_file(file_name, filtered_items)
            _logger.debug(f"Deleted cultural knowledge with id: {id}")
        except Exception as e:
            _logger.error(f"Error deleting cultural knowledge: {e}")
            raise e

    def get_cultural_knowledge(
        self,
        id: str,
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Get cultural knowledge from JSON storage.
        
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
            file_name = "cultural_knowledge"
            all_items = self._read_json_file(file_name)
            for item in all_items:
                if item.get("id") == id:
                    if not deserialize:
                        return item
                    return CulturalKnowledge.from_dict(item)
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
        """Get all cultural knowledge entries from JSON storage.
        
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
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            file_name = "cultural_knowledge"
            all_items = self._read_json_file(file_name)

            # Filter
            filtered: List[Dict[str, Any]] = []
            for item in all_items:
                if name is not None and item.get("name") != name:
                    continue
                if agent_id is not None and item.get("agent_id") != agent_id:
                    continue
                if team_id is not None and item.get("team_id") != team_id:
                    continue
                filtered.append(item)

            # Sort
            if sort_by:
                filtered = apply_sorting(filtered, sort_by, sort_order)

            total_count = len(filtered)

            # Paginate
            if limit and page:
                start = (page - 1) * limit
                filtered = filtered[start : start + limit]
            elif limit:
                filtered = filtered[:limit]

            if not deserialize:
                return filtered, total_count
            return [CulturalKnowledge.from_dict(item) for item in filtered]

        except Exception as e:
            _logger.error(f"Error getting all cultural knowledge: {e}")
            raise e

    def upsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Upsert cultural knowledge into JSON storage.
        
        Args:
            cultural_knowledge: The CulturalKnowledge instance to upsert.
            deserialize: If True, return CulturalKnowledge object. If False, return dict.
        
        Returns:
            The upserted CulturalKnowledge object or dict, or None if error occurs.
        
        Raises:
            Exception: If an error occurs during upsert.
        """
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        import uuid
        import time
        
        try:
            if cultural_knowledge.id is None:
                cultural_knowledge.id = str(uuid.uuid4())

            file_name = "cultural_knowledge"
            all_items = self._read_json_file(file_name, create_if_not_found=True)

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
            all_items = [ck for ck in all_items if ck.get("id") != cultural_knowledge.id]

            # Add new entry
            all_items.append(data)

            self._write_json_file(file_name, all_items)

            if not deserialize:
                return data
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

            all_items = self._read_json_file(
                self.knowledge_table_name,
                create_if_not_found=True,
            )

            all_items = [item for item in all_items if item.get("id") != data["id"]]
            all_items.append(data)

            self._write_json_file(self.knowledge_table_name, all_items)
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
            all_items = self._read_json_file(
                self.knowledge_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return None

        try:
            for item in all_items:
                if item.get("id") == id:
                    return KnowledgeRow.from_dict(item)
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
            all_items = self._read_json_file(
                self.knowledge_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return [], 0

        try:
            filtered: List[Dict[str, Any]] = []
            for item in all_items:
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

            rows = [KnowledgeRow.from_dict(item) for item in paginated]
            return rows, total_count

        except Exception as e:
            _logger.error(f"Error getting knowledge contents: {e}")
            raise e

    def delete_knowledge_content(self, id: str) -> bool:
        """Delete a knowledge document registry entry by ID."""
        try:
            all_items = self._read_json_file(
                self.knowledge_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return False

        try:
            original_count = len(all_items)
            all_items = [item for item in all_items if item.get("id") != id]

            if len(all_items) < original_count:
                self._write_json_file(self.knowledge_table_name, all_items)
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
            all_items = self._read_json_file(
                self.knowledge_table_name,
                create_if_not_found=False,
            )
        except FileNotFoundError:
            return 0

        try:
            ids_set = set(ids)
            original_count = len(all_items)
            all_items = [item for item in all_items if item.get("id") not in ids_set]

            deleted_count: int = original_count - len(all_items)
            if deleted_count > 0:
                self._write_json_file(self.knowledge_table_name, all_items)

            _logger.debug(f"Deleted {deleted_count} knowledge entries")
            return deleted_count

        except Exception as e:
            _logger.error(f"Error deleting knowledge contents: {e}")
            raise e
