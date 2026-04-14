"""Redis storage implementation for Upsonic agent framework.

This module provides a Redis-based storage backend that implements
the Storage abstract base class for persisting agent sessions and
user memory data.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from upsonic.storage.base import Storage
from upsonic.storage.redis.schemas import (
    CULTURAL_KNOWLEDGE_INDEX_FIELDS,
    KNOWLEDGE_INDEX_FIELDS,
    SESSION_INDEX_FIELDS,
    USER_MEMORY_INDEX_FIELDS,
)
from upsonic.storage.schemas import UserMemory
from upsonic.storage.redis.utils import (
    apply_filters,
    apply_pagination,
    apply_sorting,
    create_index_entries,
    deserialize_data,
    generate_redis_key,
    get_all_keys_for_table,
    remove_index_entries,
    serialize_data,
)
from upsonic.utils.logging_config import get_logger
from upsonic.storage.utils import deserialize_session

if TYPE_CHECKING:
    from redis import Redis, RedisCluster
    from upsonic.session.base import SessionType, Session
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.schemas import KnowledgeRow

_logger = get_logger("upsonic.storage.redis")



class RedisStorage(Storage):
    """
    Redis-based storage implementation for agent sessions and user memory.
    
    This class provides a persistent storage backend using Redis for storing:
    - Agent sessions (including team and workflow sessions)
    - User memory/profile data
    
    Features:
    - Automatic key prefixing for namespace isolation
    - Optional TTL (expire) for automatic data expiration
    - Index-based lookups for efficient filtering
    - Support for both single Redis and Redis Cluster
    
    Example:
        >>> from upsonic.storage.redis import RedisStorage
        >>> from redis import Redis
        >>> 
        >>> # Using Redis URL
        >>> storage = RedisStorage(db_url="redis://localhost:6379/0")
        >>> 
        >>> # Using existing Redis client
        >>> client = Redis.from_url("redis://localhost:6379/0")
        >>> storage = RedisStorage(redis_client=client)
        >>> 
        >>> # With custom prefix and TTL
        >>> storage = RedisStorage(
        ...     db_url="redis://localhost:6379/0",
        ...     db_prefix="myapp",
        ...     expire=86400,  # 24 hours
        ... )
    """

    def __init__(
        self,
        id: Optional[str] = None,
        redis_client: Optional[Union["Redis", "RedisCluster"]] = None,
        db_url: Optional[str] = None,
        db_prefix: str = "upsonic",
        expire: Optional[int] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
    ) -> None:
        """
        Initialize the Redis storage backend.
        
        Connection priority:
            1. Use redis_client if provided
            2. Use db_url to create a new client
            3. Raise ValueError if neither is provided
        
        Note: db_url only supports single-node Redis. For Redis Cluster,
        provide a pre-configured redis_client.
        
        Args:
            id: Unique identifier for this storage instance.
            redis_client: Existing Redis/RedisCluster client instance.
            db_url: Redis connection URL (e.g., "redis://localhost:6379/0").
            db_prefix: Prefix for all Redis keys (default: "upsonic").
            expire: TTL in seconds for Redis keys (optional).
            session_table: Name override for session table (default: "sessions").
            user_memory_table: Name override for user memory table (default: "user_memories").
        
        Raises:
            ValueError: If neither redis_client nor db_url is provided.
            ImportError: If redis package is not installed.
        """
        # Generate ID from connection info if not provided
        if id is None:
            base_seed = db_url or str(redis_client)
            id = f"redis_{hash(f'{base_seed}#{db_prefix}') % 100000}"
        
        super().__init__(
            session_table=session_table or "sessions",
            user_memory_table=user_memory_table or "user_memories",
            knowledge_table=knowledge_table,
            id=id,
        )
        
        self.db_prefix = db_prefix
        self.expire = expire
        
        # Initialize Redis client
        if redis_client is not None:
            self.redis_client = redis_client
        elif db_url is not None:
            try:
                from redis import Redis
            except ImportError:
                from upsonic.utils.printing import import_error
                import_error(
                    package_name="redis",
                    install_command='pip install "upsonic[redis-storage]"',
                    feature_name="Redis storage provider"
                )
                raise  # This will never be reached, but for type checking
            self.redis_client = Redis.from_url(db_url, decode_responses=True)
        else:
            raise ValueError("Either redis_client or db_url must be provided")
        
        _logger.debug(f"Initialized RedisStorage with prefix '{db_prefix}'")

    # --- DB Utility Methods ---

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in Redis.
        
        Redis doesn't have traditional tables, so this always returns True.
        Data existence is checked at the key level.
        
        Args:
            table_name: Name of the table to check.
        
        Returns:
            Always True for Redis.
        """
        return True

    def _get_table_name(self, table_type: str) -> str:
        """
        Get the table name for a given table type.
        
        Args:
            table_type: Type of table ("sessions" or "user_memories").
        
        Returns:
            The configured table name.
        
        Raises:
            ValueError: If table_type is unknown.
        """
        if table_type == "sessions":
            return self.session_table_name
        elif table_type == "user_memories":
            return self.user_memory_table_name
        else:
            raise ValueError(f"Unknown table type: {table_type}")

    def _store_record(
        self,
        table_type: str,
        record_id: str,
        data: Dict[str, Any],
        index_fields: Optional[List[str]] = None,
    ) -> bool:
        """
        Store a record in Redis with optional indexing.
        
        Args:
            table_type: Type of table to store in.
            record_id: Unique ID for the record.
            data: Record data as dictionary.
            index_fields: Fields to create indexes for.
        
        Returns:
            True if stored successfully, False otherwise.
        """
        try:
            key = generate_redis_key(
                prefix=self.db_prefix,
                table_type=table_type,
                key_id=record_id,
            )
            serialized = serialize_data(data)
            
            self.redis_client.set(key, serialized, ex=self.expire)
            
            if index_fields:
                create_index_entries(
                    redis_client=self.redis_client,
                    prefix=self.db_prefix,
                    table_type=table_type,
                    record_id=record_id,
                    record_data=data,
                    index_fields=index_fields,
                    expire=self.expire,
                )
            
            return True
        except Exception as e:
            _logger.error(f"Error storing record {record_id}: {e}")
            return False

    def _get_record(
        self,
        table_type: str,
        record_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a record from Redis by ID.
        
        Args:
            table_type: Type of table to read from.
            record_id: ID of the record to get.
        
        Returns:
            Record data as dictionary, or None if not found.
        """
        try:
            key = generate_redis_key(
                prefix=self.db_prefix,
                table_type=table_type,
                key_id=record_id,
            )
            
            data = self.redis_client.get(key)
            if data is None:
                return None
            
            return deserialize_data(data)
        except Exception as e:
            _logger.error(f"Error getting record {record_id}: {e}")
            return None

    def _delete_record(
        self,
        table_type: str,
        record_id: str,
        index_fields: Optional[List[str]] = None,
    ) -> bool:
        """
        Delete a record from Redis.
        
        Args:
            table_type: Type of table to delete from.
            record_id: ID of the record to delete.
            index_fields: Fields that were indexed (to clean up).
        
        Returns:
            True if deleted, False if not found or error.
        """
        try:
            # Clean up indexes first
            if index_fields:
                record_data = self._get_record(table_type, record_id)
                if record_data:
                    remove_index_entries(
                        redis_client=self.redis_client,
                        prefix=self.db_prefix,
                        table_type=table_type,
                        record_id=record_id,
                        record_data=record_data,
                        index_fields=index_fields,
                    )
            
            key = generate_redis_key(
                prefix=self.db_prefix,
                table_type=table_type,
                key_id=record_id,
            )
            result = self.redis_client.delete(key)
            
            return result is not None and result > 0
        except Exception as e:
            _logger.error(f"Error deleting record {record_id}: {e}")
            return False

    def _get_all_records(
        self,
        table_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all records for a table type.
        
        Args:
            table_type: Type of table to read from.
        
        Returns:
            List of all records as dictionaries.
        """
        try:
            keys = get_all_keys_for_table(
                redis_client=self.redis_client,
                prefix=self.db_prefix,
                table_type=table_type,
            )
            
            records = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    records.append(deserialize_data(data))
            
            return records
        except Exception as e:
            _logger.error(f"Error getting all records for {table_type}: {e}")
            return []

    def close(self) -> None:
        """
        Close the Redis connection.
        
        Should be called during application shutdown to release resources.
        """
        try:
            if hasattr(self.redis_client, 'close'):
                self.redis_client.close()
            _logger.debug("Closed Redis connection")
        except Exception as e:
            _logger.error(f"Error closing Redis connection: {e}")

    def _session_to_dict(self, session: "Session") -> Dict[str, Any]:
        """
        Convert a Session object to a storage dictionary.
        
        Args:
            session: Session object to convert.
        
        Returns:
            Dictionary representation for storage.
        """
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType

        session_dict = session.to_dict(serialize_flag=True)
        
        current_time = int(time.time())
        
        if isinstance(session, AgentSession):
            return {
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
                "messages": session_dict.get("messages"),
                "summary": session_dict.get("summary"),
                "usage": session_dict.get("usage"),
                "created_at": session_dict.get("created_at") or current_time,
                "updated_at": current_time,
            }
        else:
            # Fallback for other session types
            return {
                "session_id": session_dict.get("session_id"),
                "session_type": session_dict.get("session_type"),
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
                "messages": session_dict.get("messages"),
                "summary": session_dict.get("summary"),
                "created_at": session_dict.get("created_at") or current_time,
                "updated_at": current_time,
            }

    def upsert_session(
        self,
        session: "Session",
        deserialize: bool = True,
    ) -> Optional[Union["Session", Dict[str, Any]]]:
        """
        Insert or update a session in Redis.
        
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

        if not session.session_id:
            raise ValueError("session_id is required for upsert_session")
        
        try:
            data = self._session_to_dict(session)
            
            success = self._store_record(
                table_type="sessions",
                record_id=session.session_id,
                data=data,
                index_fields=SESSION_INDEX_FIELDS,
            )
            
            if not success:
                return None
            
            _logger.debug(f"Upserted session: {session.session_id}")
            
            if not deserialize:
                return data
            
            if isinstance(session, AgentSession):
                return AgentSession.from_dict(data, deserialize_flag=True)
            return deserialize_session(data)
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
        
        Note: Redis doesn't have efficient bulk operations like SQL databases.
        This method performs individual upserts for each session.
        
        Args:
            sessions: List of sessions to upsert.
            deserialize: If True, return deserialized Session objects.
        
        Returns:
            List of upserted sessions.
        """
        if not sessions:
            return []
        
        _logger.info(
            f"Redis bulk operations are sequential. "
            f"Upserting {len(sessions)} sessions individually."
        )
        
        results = []
        for session in sessions:
            if session is not None and session.session_id:
                result = self.upsert_session(session, deserialize=deserialize)
                if result is not None:
                    results.append(result)
        
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
        Get a session from Redis.
        
        If session_id is provided, fetches that specific session.
        If session_id is None, returns the latest session matching filters.
        
        Args:
            session_id: ID of the session to retrieve.
            session_type: Filter by session type.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            deserialize: If True, return deserialized Session object.
        
        Returns:
            The session if found, None otherwise.
        """
        try:
            # If session_id provided, get directly
            if session_id is not None:
                session_raw = self._get_record("sessions", session_id)
                if session_raw is None:
                    return None
                
                # Apply filters
                if user_id and session_raw.get("user_id") != user_id:
                    return None
                if agent_id and session_raw.get("agent_id") != agent_id:
                    return None
                if session_type:
                    session_type_data = session_raw.get("session_type")
                    if isinstance(session_type_data, dict):
                        type_value = session_type_data.get("session_type")
                    else:
                        type_value = session_type_data
                    if type_value != session_type.value:
                        return None
                
                if not deserialize:
                    return session_raw
                
                return deserialize_session(session_raw)
            
            # No session_id - get latest matching filters
            all_sessions = self._get_all_records("sessions")
            
            # Apply filters
            conditions: Dict[str, Any] = {}
            if user_id:
                conditions["user_id"] = user_id
            if agent_id:
                conditions["agent_id"] = agent_id
            
            filtered = apply_filters(all_sessions, conditions)
            
            # Filter by session_type
            if session_type:
                filtered = [
                    s for s in filtered
                    if self._extract_session_type(s) == session_type.value
                ]
            
            if not filtered:
                return None
            
            # Sort by updated_at descending and get latest
            sorted_sessions = apply_sorting(filtered, sort_by="updated_at", sort_order="desc")
            session_raw = sorted_sessions[0]
            
            if not deserialize:
                return session_raw
            
            return deserialize_session(session_raw)
        except Exception as e:
            _logger.error(f"Error getting session: {e}")
            raise

    def _extract_session_type(self, session_raw: Dict[str, Any]) -> Optional[str]:
        """Extract session type value from raw session data."""
        session_type_data = session_raw.get("session_type")
        if isinstance(session_type_data, dict):
            return session_type_data.get("session_type")
        return session_type_data

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
        Get multiple sessions from Redis.
        
        If session_ids is provided, fetches those specific sessions.
        If session_ids is None, returns all sessions matching filters.
        
        Args:
            session_ids: List of session IDs to retrieve.
            session_type: Filter by session type.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            sort_by: Field to sort by (default: "updated_at").
            sort_order: Sort order - "asc" or "desc" (default: "desc").
            deserialize: If True, return deserialized Session objects.
                        If False, return tuple of (dicts, total_count).
        
        Returns:
            List of sessions or tuple of (list of dicts, total count).
        """
        try:
            # Get records
            if session_ids is not None:
                # session_ids is explicitly provided (could be empty list)
                if len(session_ids) == 0:
                    # Empty list means no IDs to fetch, return empty
                    all_records = []
                else:
                    # Fetch specific sessions
                    all_records = []
                    for sid in session_ids:
                        record = self._get_record("sessions", sid)
                        if record:
                            all_records.append(record)
            else:
                # session_ids is None - get all sessions
                all_records = self._get_all_records("sessions")
            
            # Apply filters
            conditions: Dict[str, Any] = {}
            if user_id:
                conditions["user_id"] = user_id
            if agent_id:
                conditions["agent_id"] = agent_id
            
            filtered = apply_filters(all_records, conditions)
            
            # Filter by session_type
            if session_type:
                filtered = [
                    s for s in filtered
                    if self._extract_session_type(s) == session_type.value
                ]
            
            total_count = len(filtered)
            
            # Sort
            sorted_records = apply_sorting(filtered, sort_by=sort_by, sort_order=sort_order)
            
            # Paginate
            paginated = apply_pagination(sorted_records, limit=limit, offset=offset)
            
            if not deserialize:
                return paginated, total_count
            
            # Deserialize
            return [deserialize_session(r) for r in paginated]
        except Exception as e:
            _logger.error(f"Error getting sessions: {e}")
            raise

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from Redis.
        
        Args:
            session_id: ID of the session to delete.
        
        Returns:
            True if deleted, False if not found.
        
        Raises:
            ValueError: If session_id is not provided.
        """
        if not session_id:
            raise ValueError("session_id is required for delete_session")
        
        try:
            success = self._delete_record(
                table_type="sessions",
                record_id=session_id,
                index_fields=SESSION_INDEX_FIELDS,
            )
            
            if success:
                _logger.debug(f"Deleted session: {session_id}")
            else:
                _logger.debug(f"Session not found: {session_id}")
            
            return success
        except Exception as e:
            _logger.error(f"Error deleting session: {e}")
            raise

    def delete_sessions(self, session_ids: List[str]) -> int:
        """
        Delete multiple sessions from Redis.
        
        Args:
            session_ids: List of session IDs to delete.
        
        Returns:
            Number of sessions deleted.
        
        Raises:
            ValueError: If session_ids is empty.
        """
        if not session_ids:
            raise ValueError("session_ids list cannot be empty")
        
        deleted_count = 0
        for session_id in session_ids:
            if self._delete_record(
                table_type="sessions",
                record_id=session_id,
                index_fields=SESSION_INDEX_FIELDS,
            ):
                deleted_count += 1
        
        _logger.debug(f"Deleted {deleted_count} of {len(session_ids)} sessions")
        return deleted_count

    # --- User Memory Methods ---

    def upsert_user_memory(
        self,
        user_memory: UserMemory,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Insert or update user memory in Redis.
        
        Args:
            user_memory: The memory data to store. Must have 'user_id' key.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The upserted user memory record.
        
        Raises:
            ValueError: If user_id is not provided in user_memory.
        """
        user_id = user_memory.user_id
        if not user_id:
            raise ValueError("user_memory must have 'user_id' for upsert")
        
        try:
            current_time = int(time.time())
            
            memory_data = user_memory.user_memory
            agent_id = user_memory.agent_id
            team_id = user_memory.team_id
            
            # Check for existing record to preserve created_at
            existing = self._get_record("user_memories", user_id)
            created_at = existing.get("created_at") if existing else current_time
            
            data = {
                "user_id": user_id,
                "user_memory": memory_data,
                "agent_id": agent_id,
                "team_id": team_id,
                "created_at": created_at,
                "updated_at": current_time,
            }
            
            success = self._store_record(
                table_type="user_memories",
                record_id=user_memory.user_id,
                data=data,
                index_fields=USER_MEMORY_INDEX_FIELDS,
            )
            
            if not success:
                return None
            
            _logger.debug(f"Upserted user memory for: {user_memory.user_id}")
            if deserialize:
                return UserMemory.from_dict(data)
            return data
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
        
        Each dict in user_memories must have 'user_id' key.
        Other optional keys: 'memory', 'agent_id', 'team_id'.
        
        Args:
            user_memories: List of user memory dicts.
            deserialize: If True, return UserMemory objects. If False, return raw dicts.
        
        Returns:
            List of upserted user memory records.
        
        Raises:
            ValueError: If any record is missing user_id.
        """
        if not user_memories:
            return []
        
        results: List[Union[UserMemory, Dict[str, Any]]] = []
        for mem in user_memories:
            if not mem.user_id:
                raise ValueError("Each UserMemory instance must have user_id set")
            
            result = self.upsert_user_memory(
                user_memory=mem,
                deserialize=deserialize,
            )
            if result:
                results.append(result)
        
        return results

    def get_user_memory(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """
        Get user memory from Redis.
        
        If user_id is provided, fetches that specific record.
        If user_id is None, returns the latest memory matching filters.
        
        Args:
            user_id: User ID to retrieve.
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            deserialize: If True, return UserMemory object. If False, return raw dict.
        
        Returns:
            The user memory record if found, None otherwise.
        """
        try:
            # If user_id provided, get directly
            if user_id is not None:
                record = self._get_record("user_memories", user_id)
                if record is None:
                    return None
                
                # Apply filters
                if agent_id and record.get("agent_id") != agent_id:
                    return None
                if team_id and record.get("team_id") != team_id:
                    return None
                
                if deserialize:
                    return UserMemory.from_dict(record)
                return record
            
            # No user_id - get latest matching filters
            all_records = self._get_all_records("user_memories")
            
            conditions: Dict[str, Any] = {}
            if agent_id:
                conditions["agent_id"] = agent_id
            if team_id:
                conditions["team_id"] = team_id
            
            filtered = apply_filters(all_records, conditions)
            
            if not filtered:
                return None
            
            # Sort by updated_at descending and get latest
            sorted_records = apply_sorting(filtered, sort_by="updated_at", sort_order="desc")
            memory_dict = sorted_records[0]
            if deserialize:
                return UserMemory.from_dict(memory_dict)
            return memory_dict
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
        Get multiple user memories from Redis.
        
        If user_ids is provided, fetches those specific records.
        If user_ids is None, returns all records matching filters.
        
        Args:
            user_ids: List of user IDs to retrieve.
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
            # Get records
            if user_ids is not None:
                # user_ids is explicitly provided (could be empty list)
                if len(user_ids) == 0:
                    # Empty list means no IDs to fetch, return empty
                    if deserialize:
                        return []
                    return [], 0
                else:
                    # Fetch specific records
                    all_records = []
                    for uid in user_ids:
                        record = self._get_record("user_memories", uid)
                        if record:
                            all_records.append(record)
            else:
                # user_ids is None - get all records
                all_records = self._get_all_records("user_memories")
            
            # Apply filters
            conditions: Dict[str, Any] = {}
            if agent_id:
                conditions["agent_id"] = agent_id
            if team_id:
                conditions["team_id"] = team_id
            
            filtered = apply_filters(all_records, conditions)
            total_count = len(filtered)
            
            # Sort by updated_at descending
            sorted_records = apply_sorting(filtered, sort_by="updated_at", sort_order="desc")
            
            # Paginate
            paginated = apply_pagination(sorted_records, limit=limit, offset=offset)
            
            if not paginated:
                if deserialize:
                    return []
                return [], 0
            
            if deserialize:
                return [UserMemory.from_dict(record) for record in paginated]
            return paginated, total_count
        except Exception as e:
            _logger.error(f"Error getting user memories: {e}")
            raise

    def delete_user_memory(self, user_id: str) -> bool:
        """
        Delete user memory from Redis.
        
        Args:
            user_id: User ID of the memory to delete.
        
        Returns:
            True if deleted, False if not found.
        
        Raises:
            ValueError: If user_id is not provided.
        """
        if not user_id:
            raise ValueError("user_id is required for delete_user_memory")
        
        try:
            success = self._delete_record(
                table_type="user_memories",
                record_id=user_id,
                index_fields=USER_MEMORY_INDEX_FIELDS,
            )
            
            if success:
                _logger.debug(f"Deleted user memory for: {user_id}")
            else:
                _logger.debug(f"User memory not found for: {user_id}")
            
            return success
        except Exception as e:
            _logger.error(f"Error deleting user memory: {e}")
            raise

    def delete_user_memories(self, user_ids: List[str]) -> int:
        """
        Delete multiple user memories from Redis.
        
        Args:
            user_ids: List of user IDs to delete.
        
        Returns:
            Number of records deleted.
        
        Raises:
            ValueError: If user_ids is empty.
        """
        if not user_ids:
            raise ValueError("user_ids list cannot be empty")
        
        deleted_count = 0
        for user_id in user_ids:
            if self._delete_record(
                table_type="user_memories",
                record_id=user_id,
                index_fields=USER_MEMORY_INDEX_FIELDS,
            ):
                deleted_count += 1
        
        _logger.debug(f"Deleted {deleted_count} of {len(user_ids)} user memories")
        return deleted_count

    # --- Utility Methods ---

    def clear_all(self) -> None:
        """
        Clear all data from all tables in Redis.
        
        This removes all sessions and user memories from storage.
        Use with caution - this operation is irreversible.
        """
        try:
            # Clear sessions
            session_keys = get_all_keys_for_table(
                redis_client=self.redis_client,
                prefix=self.db_prefix,
                table_type="sessions",
            )
            if session_keys:
                self.redis_client.delete(*session_keys)
            
            # Clear session indexes
            session_idx_pattern = f"{self.db_prefix}:sessions:index:*"
            self._delete_keys_by_pattern(session_idx_pattern)
            
            # Clear user memories
            memory_keys = get_all_keys_for_table(
                redis_client=self.redis_client,
                prefix=self.db_prefix,
                table_type="user_memories",
            )
            if memory_keys:
                self.redis_client.delete(*memory_keys)
            
            # Clear user memory indexes
            memory_idx_pattern = f"{self.db_prefix}:user_memories:index:*"
            self._delete_keys_by_pattern(memory_idx_pattern)

            # Clear knowledge
            knowledge_keys = get_all_keys_for_table(
                redis_client=self.redis_client,
                prefix=self.db_prefix,
                table_type="knowledge",
            )
            if knowledge_keys:
                self.redis_client.delete(*knowledge_keys)

            # Clear knowledge indexes
            knowledge_idx_pattern = f"{self.db_prefix}:knowledge:index:*"
            self._delete_keys_by_pattern(knowledge_idx_pattern)

            _logger.info("Cleared all data from Redis storage")
        except Exception as e:
            _logger.error(f"Error clearing all data: {e}")
            raise

    def _delete_keys_by_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern using SCAN.
        
        Args:
            pattern: Redis key pattern (e.g., "upsonic:sessions:index:*").
        
        Returns:
            Number of keys deleted.
        """
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self.redis_client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        return deleted

    # ======================== Generic Model Methods ========================

    def _get_model_key(self, key: str, collection: str) -> str:
        """Generate Redis key for a generic model."""
        return f"{self.db_prefix}:models:{collection}:{key}"

    def upsert_model(
        self,
        key: str,
        model: Any,
        collection: str = "generic_models",
    ) -> None:
        """Insert or update a generic Pydantic model in storage."""
        import time
        import json
        
        try:
            redis_key = self._get_model_key(key, collection)
            current_time = int(time.time())
            
            # Serialize model
            if hasattr(model, 'model_dump'):
                model_data = model.model_dump(mode='json')
            elif hasattr(model, 'dict'):
                model_data = model.dict()
            else:
                model_data = dict(model)
            
            # Get existing to preserve created_at
            existing = self.redis_client.get(redis_key)
            created_at = current_time
            if existing:
                try:
                    existing_data = json.loads(existing)
                    created_at = existing_data.get("created_at", current_time)
                except Exception:
                    pass
            
            document = {
                "key": key,
                "collection": collection,
                "model_data": model_data,
                "created_at": created_at,
                "updated_at": current_time,
            }
            
            self.redis_client.set(redis_key, json.dumps(document))
            
            if self.ttl:
                self.redis_client.expire(redis_key, self.ttl)
            
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
        import json
        
        try:
            redis_key = self._get_model_key(key, collection)
            
            data = self.redis_client.get(redis_key)
            if data is None:
                return None
            
            document = json.loads(data)
            model_data = document.get("model_data")
            if model_data is None:
                return None
            
            return model_type(**model_data)
            
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
            redis_key = self._get_model_key(key, collection)
            
            result = self.redis_client.delete(redis_key)
            
            if result > 0:
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
        import json
        
        try:
            pattern = f"{self.db_prefix}:models:{collection}:*"
            
            models = []
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                for key in keys:
                    try:
                        data = self.redis_client.get(key)
                        if data:
                            document = json.loads(data)
                            model_data = document.get("model_data")
                            if model_data:
                                models.append(model_type(**model_data))
                    except Exception as e:
                        _logger.warning(f"Failed to deserialize model: {e}")
                if cursor == 0:
                    break
            
            return models
            
        except Exception as e:
            _logger.error(f"Error listing models: {e}")
            return []

    # ======================== Cultural Knowledge Methods ========================

    def delete_cultural_knowledge(self, id: str) -> None:
        """Delete cultural knowledge from Redis.
        
        Args:
            id: The ID of the cultural knowledge to delete.
        
        Raises:
            Exception: If an error occurs during deletion.
        """
        try:
            key = generate_redis_key(self.db_prefix, "cultural_knowledge", id)
            
            # Get existing data to remove index entries
            existing_data = self.redis_client.get(key)
            if existing_data:
                record_data = deserialize_data(existing_data)
                remove_index_entries(
                    redis_client=self.redis_client,
                    prefix=self.db_prefix,
                    table_type="cultural_knowledge",
                    record_id=id,
                    record_data=record_data,
                    index_fields=CULTURAL_KNOWLEDGE_INDEX_FIELDS,
                )
                self.redis_client.delete(key)
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
        """Get cultural knowledge from Redis.
        
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
            key = generate_redis_key(self.db_prefix, "cultural_knowledge", id)
            data = self.redis_client.get(key)
            
            if data is None:
                return None

            db_row = deserialize_data(data)
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
        """Get all cultural knowledge entries from Redis.
        
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
            # Get all cultural knowledge keys
            keys = get_all_keys_for_table(
                redis_client=self.redis_client,
                prefix=self.db_prefix,
                table_type="cultural_knowledge",
            )
            
            all_items: List[Dict[str, Any]] = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    all_items.append(deserialize_data(data))

            # Apply filters
            filtered_items: List[Dict[str, Any]] = []
            for item in all_items:
                if name is not None and item.get("name") != name:
                    continue
                if agent_id is not None and item.get("agent_id") != agent_id:
                    continue
                if team_id is not None and item.get("team_id") != team_id:
                    continue
                filtered_items.append(item)

            total_count = len(filtered_items)

            # Apply sorting
            sorted_items = apply_sorting(
                records=filtered_items,
                sort_by=sort_by or "created_at",
                sort_order=sort_order or "desc",
            )
            
            # Apply pagination
            paginated_items = apply_pagination(
                records=sorted_items,
                limit=limit,
                page=page,
            )

            if not deserialize:
                return paginated_items, total_count
            return [CulturalKnowledge.from_dict(item) for item in paginated_items]

        except Exception as e:
            _logger.error(f"Error getting all cultural knowledge: {e}")
            raise e

    def upsert_cultural_knowledge(
        self,
        cultural_knowledge: "CulturalKnowledge",
        deserialize: bool = True,
    ) -> Optional[Union["CulturalKnowledge", Dict[str, Any]]]:
        """Upsert cultural knowledge into Redis.
        
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

            key = generate_redis_key(self.db_prefix, "cultural_knowledge", cultural_knowledge.id)
            
            # Remove old index entries if record exists
            existing_data = self.redis_client.get(key)
            if existing_data:
                old_record = deserialize_data(existing_data)
                remove_index_entries(
                    redis_client=self.redis_client,
                    prefix=self.db_prefix,
                    table_type="cultural_knowledge",
                    record_id=cultural_knowledge.id,
                    record_data=old_record,
                    index_fields=CULTURAL_KNOWLEDGE_INDEX_FIELDS,
                )

            # Store the data
            serialized_data = serialize_data(data)
            if self.expire:
                self.redis_client.setex(key, self.expire, serialized_data)
            else:
                self.redis_client.set(key, serialized_data)

            # Create new index entries
            create_index_entries(
                redis_client=self.redis_client,
                prefix=self.db_prefix,
                table_type="cultural_knowledge",
                record_id=cultural_knowledge.id,
                record_data=data,
                index_fields=CULTURAL_KNOWLEDGE_INDEX_FIELDS,
                expire=self.expire,
            )

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

            key = generate_redis_key(self.db_prefix, "knowledge", data["id"])

            existing_data = self.redis_client.get(key)
            if existing_data:
                old_record = deserialize_data(existing_data)
                remove_index_entries(
                    redis_client=self.redis_client,
                    prefix=self.db_prefix,
                    table_type="knowledge",
                    record_id=data["id"],
                    record_data=old_record,
                    index_fields=KNOWLEDGE_INDEX_FIELDS,
                )

            serialized = serialize_data(data)
            if self.expire:
                self.redis_client.setex(key, self.expire, serialized)
            else:
                self.redis_client.set(key, serialized)

            create_index_entries(
                redis_client=self.redis_client,
                prefix=self.db_prefix,
                table_type="knowledge",
                record_id=data["id"],
                record_data=data,
                index_fields=KNOWLEDGE_INDEX_FIELDS,
                expire=self.expire,
            )

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
            key = generate_redis_key(self.db_prefix, "knowledge", id)
            data = self.redis_client.get(key)

            if data is None:
                return None

            return KnowledgeRow.from_dict(deserialize_data(data))

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
            keys = get_all_keys_for_table(
                redis_client=self.redis_client,
                prefix=self.db_prefix,
                table_type="knowledge",
            )

            all_items: List[Dict[str, Any]] = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    all_items.append(deserialize_data(data))

            filtered_items: List[Dict[str, Any]] = []
            for item in all_items:
                if knowledge_base_id is not None and item.get("knowledge_base_id") != knowledge_base_id:
                    continue
                filtered_items.append(item)

            total_count: int = len(filtered_items)

            sorted_items = apply_sorting(
                records=filtered_items,
                sort_by=sort_by or "created_at",
                sort_order=sort_order or "desc",
            )

            paginated_items = apply_pagination(
                records=sorted_items,
                limit=limit,
                page=page,
            )

            rows = [KnowledgeRow.from_dict(item) for item in paginated_items]
            return rows, total_count

        except Exception as e:
            _logger.error(f"Error getting knowledge contents: {e}")
            raise e

    def delete_knowledge_content(self, id: str) -> bool:
        """Delete a knowledge document registry entry by ID."""
        try:
            key = generate_redis_key(self.db_prefix, "knowledge", id)

            existing_data = self.redis_client.get(key)
            if existing_data:
                record_data = deserialize_data(existing_data)
                remove_index_entries(
                    redis_client=self.redis_client,
                    prefix=self.db_prefix,
                    table_type="knowledge",
                    record_id=id,
                    record_data=record_data,
                    index_fields=KNOWLEDGE_INDEX_FIELDS,
                )
                self.redis_client.delete(key)
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

        deleted_count: int = 0
        for doc_id in ids:
            if self.delete_knowledge_content(doc_id):
                deleted_count += 1

        _logger.debug(f"Deleted {deleted_count} knowledge entries")
        return deleted_count
