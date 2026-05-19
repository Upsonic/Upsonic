"""Session manager for Chat class with storage binding.

This module provides comprehensive session management that syncs with storage.
All operations update both local state and persisted storage data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from upsonic.storage.base import Storage
    from upsonic.session.agent import AgentSession
    from upsonic.usage import RunUsage

from .message import ChatMessage


class SessionState(Enum):
    """Session state enumeration."""
    IDLE = "idle"
    AWAITING_RESPONSE = "awaiting_response"
    STREAMING = "streaming"
    ERROR = "error"


class SessionManager:
    """
    Comprehensive session management for Chat class with storage binding.
    
    This class serves as the bridge between Chat and storage, ensuring:
    - All data comes from storage (single source of truth)
    - All operations update both local state and storage
    - ChatMessage is only used for developer-friendly access (conversion on demand)
    
    Key Design Principles:
    1. Storage is the source of truth for session data
    2. Local state (metrics, concurrent invocations) is for runtime only
    3. Messages are stored as ModelMessage in storage, converted to ChatMessage on access
    4. Every mutation operation syncs to storage
    5. Session timing is managed automatically (start_time, end_time)
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        storage: "Storage",
        *,
        debug: bool = False,
        debug_level: int = 1,
        max_concurrent_invocations: int = 1,
    ) -> None:
        """
        Initialize SessionManager with storage binding.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            storage: Storage backend for persistence (required)
            debug: Enable debug logging
            debug_level: Debug verbosity (1 = standard, 2 = detailed)
            max_concurrent_invocations: Maximum concurrent invocations allowed
        """
        if storage is None:
            raise ValueError("storage is required for SessionManager")
        
        self.session_id = session_id
        self.user_id = user_id
        self._storage = storage
        self.debug = debug
        self.debug_level = debug_level if debug else 1
        
        # Runtime state (not persisted)
        self._state = SessionState.IDLE
        self._concurrent_invocations = 0
        self._max_concurrent_invocations = max_concurrent_invocations
        
        # Session timing (managed automatically)
        self._start_time: float = time.time()
        self._end_time: Optional[float] = None
        self._last_activity_time: float = time.time()
        self._response_times: List[float] = []
        self._is_closed: bool = False
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(
                f"SessionManager initialized: session_id={session_id}, user_id={user_id}",
                "SessionManager",
                debug=self.debug,
                debug_level=self.debug_level
            )
    

    
    def _get_session(self) -> Optional["AgentSession"]:
        """Get the current session from storage (sync)."""
        from upsonic.session.base import SessionType
        
        return self._storage.get_session(
            session_id=self.session_id,
            session_type=SessionType.AGENT,
            deserialize=True
        )
    
    async def _aget_session(self) -> Optional["AgentSession"]:
        """Get the current session from storage (async)."""
        from upsonic.session.base import SessionType
        from upsonic.storage.base import AsyncStorage
        
        if isinstance(self._storage, AsyncStorage):
            return await self._storage.aget_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
        else:
            return self._storage.get_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
    
    def _upsert_session(self, session: "AgentSession") -> None:
        """Upsert session to storage (sync)."""
        from upsonic.utils.dttm import now_epoch_s
        
        session.updated_at = now_epoch_s()
        self._storage.upsert_session(session, deserialize=True)
    
    async def _aupsert_session(self, session: "AgentSession") -> None:
        """Upsert session to storage (async)."""
        from upsonic.storage.base import AsyncStorage
        from upsonic.utils.dttm import now_epoch_s
        
        session.updated_at = now_epoch_s()
        if isinstance(self._storage, AsyncStorage):
            await self._storage.aupsert_session(session, deserialize=True)
        else:
            self._storage.upsert_session(session, deserialize=True)
    
    def _get_or_create_session(self) -> "AgentSession":
        """Get existing session or create a new one (sync)."""
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType
        from upsonic.utils.dttm import now_epoch_s
        
        session = self._get_session()
        if session is None:
            session = AgentSession(
                session_id=self.session_id,
                user_id=self.user_id,
                session_type=SessionType.AGENT,
                created_at=now_epoch_s(),
                messages=[],
                runs={},
            )
            self._upsert_session(session)
        return session
    
    async def _aget_or_create_session(self) -> "AgentSession":
        """Get existing session or create a new one (async)."""
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType
        from upsonic.utils.dttm import now_epoch_s
        
        session = await self._aget_session()
        if session is None:
            session = AgentSession(
                session_id=self.session_id,
                user_id=self.user_id,
                session_type=SessionType.AGENT,
                created_at=now_epoch_s(),
                messages=[],
                runs={},
            )
            await self._aupsert_session(session)
        return session
    

    
    @property
    def state(self) -> SessionState:
        """Get current session state (runtime only)."""
        return self._state
    
    def transition_state(self, new_state: SessionState) -> None:
        """Transition to a new session state (runtime only)."""
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(
                f"Session state transition: {self._state.value} -> {new_state.value}",
                "SessionManager"
            )
        self._state = new_state
    
    def can_accept_invocation(self) -> bool:
        """Check if session can accept a new invocation."""
        return (
            self._state != SessionState.ERROR and
            self._concurrent_invocations < self._max_concurrent_invocations
        )
    
    def start_invocation(self) -> None:
        """Start a new invocation (runtime tracking)."""
        self._concurrent_invocations += 1
        self._update_activity()
    
    def end_invocation(self) -> None:
        """End an invocation (runtime tracking)."""
        self._concurrent_invocations = max(0, self._concurrent_invocations - 1)
        self._update_activity()
    

    
    @property
    def all_messages(self) -> List[ChatMessage]:
        """
        Get all messages in the session as ChatMessage objects.
        
        Retrieves ModelMessage list from storage and converts to ChatMessage.
        This is the developer-friendly interface for accessing chat history.
        
        Returns:
            List of ChatMessage objects
        """
        session = self._get_session()
        if session is None or not session.messages:
            return []
        return ChatMessage.from_model_messages(session.messages)
    
    async def aget_all_messages(self) -> List[ChatMessage]:
        """Get all messages in the session as ChatMessage objects (async)."""
        session = await self._aget_session()
        if session is None or not session.messages:
            return []
        return ChatMessage.from_model_messages(session.messages)
    
    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent messages as ChatMessage objects."""
        session = self._get_session()
        if session is None or not session.messages:
            return []
        
        messages = session.messages[-count:] if count > 0 else session.messages
        return ChatMessage.from_model_messages(messages)
    
    async def aget_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent messages as ChatMessage objects (async)."""
        session = await self._aget_session()
        if session is None or not session.messages:
            return []
        
        messages = session.messages[-count:] if count > 0 else session.messages
        return ChatMessage.from_model_messages(messages)
    
    def get_message_count(self) -> int:
        """Get the number of messages in the session."""
        session = self._get_session()
        if session is None or not session.messages:
            return 0
        return len(session.messages)
    
    async def aget_message_count(self) -> int:
        """Get the number of messages in the session (async)."""
        session = await self._aget_session()
        if session is None or not session.messages:
            return 0
        return len(session.messages)
    

    
    def get_raw_messages(self) -> List[Any]:
        """
        Get raw ModelMessage list from storage.
        
        This is for direct manipulation of message history.
        
        Returns:
            List of ModelMessage objects (for manipulation)
        """
        session = self._get_session()
        if session is None or not session.messages:
            return []
        return list(session.messages)
    
    async def aget_raw_messages(self) -> List[Any]:
        """Get raw ModelMessage list from storage (async)."""
        session = await self._aget_session()
        if session is None or not session.messages:
            return []
        return list(session.messages)
    
    def set_messages(self, messages: List[Any]) -> None:
        """
        Set the message list in storage.
        
        Use this after manipulating messages (e.g., removing attachments).
        
        Args:
            messages: List of ModelMessage objects to set
        """
        session = self._get_or_create_session()
        session.messages = messages
        self._upsert_session(session)
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"Session messages updated: {len(messages)} messages", "SessionManager")
    
    async def aset_messages(self, messages: List[Any]) -> None:
        """Set the message list in storage (async)."""
        session = await self._aget_or_create_session()
        session.messages = messages
        await self._aupsert_session(session)
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"Session messages updated (async): {len(messages)} messages", "SessionManager")
    
    def delete_message(self, message_index: int) -> bool:
        """
        Delete a message by index.
        
        Args:
            message_index: The index of the message to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        session = self._get_session()
        if session is None or not session.messages:
            return False
        
        if message_index < 0 or message_index >= len(session.messages):
            return False
        
        messages = list(session.messages)
        del messages[message_index]
        session.messages = messages
        self._upsert_session(session)
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"Deleted message at index {message_index}", "SessionManager")
        
        return True
    
    async def adelete_message(self, message_index: int) -> bool:
        """Delete a message by index (async)."""
        session = await self._aget_session()
        if session is None or not session.messages:
            return False
        
        if message_index < 0 or message_index >= len(session.messages):
            return False
        
        messages = list(session.messages)
        del messages[message_index]
        session.messages = messages
        await self._aupsert_session(session)
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"Deleted message at index {message_index} (async)", "SessionManager")
        
        return True
    
    def remove_attachment_from_message(
        self,
        message_index: int,
        attachment_index: int
    ) -> bool:
        """
        Remove an attachment from a specific message.
        
        This modifies the message's content parts to remove the attachment
        at the specified index.
        
        Args:
            message_index: The index of the message containing the attachment
            attachment_index: The index of the attachment within the message
            
        Returns:
            True if removal was successful, False otherwise
        """
        from upsonic.messages.messages import (
            ModelRequest,
            UserPromptPart,
            ImageUrl,
            AudioUrl,
            DocumentUrl,
            VideoUrl,
            BinaryContent,
        )
        
        session = self._get_session()
        if session is None or not session.messages:
            return False
        
        if message_index < 0 or message_index >= len(session.messages):
            return False
        
        messages = list(session.messages)
        message = messages[message_index]
        
        if not isinstance(message, ModelRequest):
            return False
        
        for part_idx, part in enumerate(message.parts):
            if isinstance(part, UserPromptPart):
                content = part.content
                
                if isinstance(content, str):
                    return False
                
                if not hasattr(content, '__iter__') or isinstance(content, str):
                    return False
                
                new_content: List[Any] = []
                current_attachment_idx = 0
                
                for item in content:
                    is_attachment = isinstance(item, (
                        ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent
                    ))
                    
                    if is_attachment:
                        if current_attachment_idx != attachment_index:
                            new_content.append(item)
                        current_attachment_idx += 1
                    else:
                        new_content.append(item)
                
                if current_attachment_idx <= attachment_index:
                    return False
                
                if len(new_content) == 1 and isinstance(new_content[0], str):
                    part.content = new_content[0]
                else:
                    part.content = new_content
                
                self._upsert_session(session)
                
                if self.debug:
                    from upsonic.utils.printing import debug_log
                    debug_log(
                        f"Removed attachment {attachment_index} from message {message_index}",
                        "SessionManager"
                    )
                
                return True
        
        return False
    
    async def aremove_attachment_from_message(
        self,
        message_index: int,
        attachment_index: int
    ) -> bool:
        """Remove an attachment from a specific message (async)."""
        from upsonic.messages.messages import (
            ModelRequest,
            UserPromptPart,
            ImageUrl,
            AudioUrl,
            DocumentUrl,
            VideoUrl,
            BinaryContent,
        )
        
        session = await self._aget_session()
        if session is None or not session.messages:
            return False
        
        if message_index < 0 or message_index >= len(session.messages):
            return False
        
        messages = list(session.messages)
        message = messages[message_index]
        
        if not isinstance(message, ModelRequest):
            return False
        
        for part_idx, part in enumerate(message.parts):
            if isinstance(part, UserPromptPart):
                content = part.content
                
                if isinstance(content, str):
                    return False
                
                if not hasattr(content, '__iter__') or isinstance(content, str):
                    return False
                
                new_content: List[Any] = []
                current_attachment_idx = 0
                
                for item in content:
                    is_attachment = isinstance(item, (
                        ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent
                    ))
                    
                    if is_attachment:
                        if current_attachment_idx != attachment_index:
                            new_content.append(item)
                        current_attachment_idx += 1
                    else:
                        new_content.append(item)
                
                if current_attachment_idx <= attachment_index:
                    return False
                
                if len(new_content) == 1 and isinstance(new_content[0], str):
                    part.content = new_content[0]
                else:
                    part.content = new_content
                
                await self._aupsert_session(session)
                
                if self.debug:
                    from upsonic.utils.printing import debug_log
                    debug_log(
                        f"Removed attachment {attachment_index} from message {message_index} (async)",
                        "SessionManager"
                    )
                
                return True
        
        return False
    
    def remove_attachment_by_path(self, path: str) -> int:
        """
        Remove all attachments matching the given path from all messages.
        
        This is a simplified API - just pass the path and all occurrences
        are found and removed across all messages.
        
        Args:
            path: The file path or identifier to remove (e.g., "/path/to/file.pdf")
            
        Returns:
            int: Number of attachments removed
        """
        from upsonic.messages.messages import (
            ModelRequest,
            UserPromptPart,
            ImageUrl,
            AudioUrl,
            DocumentUrl,
            VideoUrl,
            BinaryContent,
        )
        
        session = self._get_session()
        if session is None or not session.messages:
            return 0
        
        removed_count = 0
        modified = False
        
        for message in session.messages:
            if not isinstance(message, ModelRequest):
                continue
            
            for part in message.parts:
                if not isinstance(part, UserPromptPart):
                    continue
                
                content = part.content
                
                if isinstance(content, str):
                    continue
                
                if not hasattr(content, '__iter__') or isinstance(content, str):
                    continue
                
                new_content: List[Any] = []
                
                for item in content:
                    is_attachment = isinstance(item, (
                        ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent
                    ))
                    
                    if is_attachment:
                        identifier = getattr(item, 'identifier', None) or getattr(item, 'url', '')
                        if path in identifier or identifier in path or path == identifier:
                            removed_count += 1
                            modified = True
                            continue
                    
                    new_content.append(item)
                
                if len(new_content) == 1 and isinstance(new_content[0], str):
                    part.content = new_content[0]
                elif new_content:
                    part.content = new_content
                else:
                    part.content = ""
        
        if modified:
            self._upsert_session(session)
            
            if self.debug:
                from upsonic.utils.printing import debug_log
                debug_log(
                    f"Removed {removed_count} attachments matching '{path}'",
                    "SessionManager"
                )
        
        return removed_count
    
    async def aremove_attachment_by_path(self, path: str) -> int:
        """Remove all attachments matching the given path (async)."""
        from upsonic.messages.messages import (
            ModelRequest,
            UserPromptPart,
            ImageUrl,
            AudioUrl,
            DocumentUrl,
            VideoUrl,
            BinaryContent,
        )
        
        session = await self._aget_session()
        if session is None or not session.messages:
            return 0
        
        removed_count = 0
        modified = False
        
        for message in session.messages:
            if not isinstance(message, ModelRequest):
                continue
            
            for part in message.parts:
                if not isinstance(part, UserPromptPart):
                    continue
                
                content = part.content
                
                if isinstance(content, str):
                    continue
                
                if not hasattr(content, '__iter__') or isinstance(content, str):
                    continue
                
                new_content: List[Any] = []
                
                for item in content:
                    is_attachment = isinstance(item, (
                        ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent
                    ))
                    
                    if is_attachment:
                        identifier = getattr(item, 'identifier', None) or getattr(item, 'url', '')
                        if path in identifier or identifier in path or path == identifier:
                            removed_count += 1
                            modified = True
                            continue
                    
                    new_content.append(item)
                
                if len(new_content) == 1 and isinstance(new_content[0], str):
                    part.content = new_content[0]
                elif new_content:
                    part.content = new_content
                else:
                    part.content = ""
        
        if modified:
            await self._aupsert_session(session)
            
            if self.debug:
                from upsonic.utils.printing import debug_log
                debug_log(
                    f"Removed {removed_count} attachments matching '{path}' (async)",
                    "SessionManager"
                )
        
        return removed_count
    
    def clear_history(self) -> None:
        """
        Clear the chat history from storage.
        
        This clears session.messages while preserving other session data.
        """
        session = self._get_session()
        if session is None:
            return
        
        session.messages = []
        self._upsert_session(session)
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log("Session message history cleared", "SessionManager")
    
    async def aclear_history(self) -> None:
        """Clear the chat history from storage (async)."""
        session = await self._aget_session()
        if session is None:
            return
        
        session.messages = []
        await self._aupsert_session(session)
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log("Session message history cleared (async)", "SessionManager")
    
    def reset_session(self) -> None:
        """
        Reset the session to initial state.
        
        This deletes the session from storage and resets all timing.
        A new start_time is set, end_time is cleared.
        """
        # Delete from storage
        self._storage.delete_session(self.session_id)
        
        # Reset all runtime state and timing
        self._state = SessionState.IDLE
        self._concurrent_invocations = 0
        self._response_times.clear()
        
        # Reset timing - new session starts now
        self._start_time = time.time()
        self._end_time = None
        self._last_activity_time = time.time()
        self._is_closed = False
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log("Session reset to initial state", "SessionManager")
    
    async def areset_session(self) -> None:
        """Reset the session to initial state (async)."""
        from upsonic.storage.base import AsyncStorage
        
        # Delete from storage
        if isinstance(self._storage, AsyncStorage):
            await self._storage.adelete_session(self.session_id)
        else:
            self._storage.delete_session(self.session_id)
        
        # Reset all runtime state and timing
        self._state = SessionState.IDLE
        self._concurrent_invocations = 0
        self._response_times.clear()
        
        # Reset timing - new session starts now
        self._start_time = time.time()
        self._end_time = None
        self._last_activity_time = time.time()
        self._is_closed = False
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log("Session reset to initial state (async)", "SessionManager")
    

    
    @property
    def input_tokens(self) -> int:
        """Get total input tokens from session usage."""
        session = self._get_session()
        if session is None or session.usage is None:
            return 0
        return session.usage.input_tokens
    
    @property
    def output_tokens(self) -> int:
        """Get total output tokens from session usage."""
        session = self._get_session()
        if session is None or session.usage is None:
            return 0
        return session.usage.output_tokens
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens (input + output) from session usage."""
        return self.input_tokens + self.output_tokens
    
    @property
    def total_cost(self) -> float:
        """Get total cost from session usage."""
        session = self._get_session()
        if session is None or session.usage is None:
            return 0.0
        return session.usage.cost or 0.0
    
    @property
    def total_requests(self) -> int:
        """Get total API requests from session usage."""
        session = self._get_session()
        if session is None or session.usage is None:
            return 0
        return session.usage.requests
    
    @property
    def total_tool_calls(self) -> int:
        """Get total tool calls from session usage."""
        session = self._get_session()
        if session is None or session.usage is None:
            return 0
        return session.usage.tool_calls
    
    @property
    def run_duration(self) -> Optional[float]:
        """Get total run duration from session usage (seconds)."""
        session = self._get_session()
        if session is None or session.usage is None:
            return None
        return session.usage.duration
    
    @property
    def time_to_first_token(self) -> Optional[float]:
        """Get time to first token from session usage (seconds)."""
        session = self._get_session()
        if session is None or session.usage is None:
            return None
        return session.usage.time_to_first_token
    
    @property
    def cache_write_tokens(self) -> int:
        """Get cache write tokens from session usage."""
        session = self._get_session()
        if session is None or session.usage is None:
            return 0
        return session.usage.cache_write_tokens
    
    @property
    def cache_read_tokens(self) -> int:
        """Get cache read tokens from session usage."""
        session = self._get_session()
        if session is None or session.usage is None:
            return 0
        return session.usage.cache_read_tokens
    
    @property
    def reasoning_tokens(self) -> int:
        """Get reasoning tokens from session usage (for o1/o3 models)."""
        session = self._get_session()
        if session is None or session.usage is None:
            return 0
        return session.usage.reasoning_tokens
    
    def get_usage(self) -> Optional["RunUsage"]:
        """Get the full RunUsage object from session."""
        session = self._get_session()
        if session is None:
            return None
        return session.usage
    
    async def aget_usage(self) -> Optional["RunUsage"]:
        """Get the full RunUsage object from session (async)."""
        session = await self._aget_session()
        if session is None:
            return None
        return session.usage
    
    def start_response_timer(self) -> float:
        """Start timing a response."""
        return time.time()
    
    def end_response_timer(self, start_time: float) -> float:
        """End timing a response and record the duration."""
        response_time = time.time() - start_time
        self._response_times.append(response_time)
        return response_time
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if not self._response_times:
            return 0.0
        return sum(self._response_times) / len(self._response_times)
    
    
    def _update_activity(self) -> None:
        """Update last activity time."""
        self._last_activity_time = time.time()
    
    @property
    def start_time(self) -> float:
        """Get session start time (Unix timestamp)."""
        return self._start_time
    
    @property
    def end_time(self) -> Optional[float]:
        """Get session end time (Unix timestamp, None if still active)."""
        return self._end_time
    
    @property
    def last_activity(self) -> float:
        """Get time since last activity in seconds."""
        return time.time() - self._last_activity_time
    
    @property
    def last_activity_time(self) -> float:
        """Get last activity timestamp (Unix timestamp)."""
        return self._last_activity_time
    
    @property
    def duration(self) -> float:
        """
        Get session duration in seconds.
        
        If the session is closed, returns the duration from start to end.
        If the session is active, returns the duration from start to now.
        """
        if self._end_time is not None:
            return self._end_time - self._start_time
        return time.time() - self._start_time
    
    @property
    def is_closed(self) -> bool:
        """Check if the session has been closed."""
        return self._is_closed
    

    
    def close_session(self) -> None:
        """
        Close the session.
        
        Sets the end time and marks the session as closed.
        After closing, duration will be fixed to start_time -> end_time.
        """
        if not self._is_closed:
            self._end_time = time.time()
            self._is_closed = True
        
        self._state = SessionState.IDLE
        self._concurrent_invocations = 0
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(
                f"Session closed: duration={self.duration:.2f}s",
                "SessionManager"
            )
    
    async def aclose_session(self) -> None:
        """Close the session (async)."""
        if not self._is_closed:
            self._end_time = time.time()
            self._is_closed = True
        
        self._state = SessionState.IDLE
        self._concurrent_invocations = 0
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(
                f"Session closed (async): duration={self.duration:.2f}s",
                "SessionManager"
            )
    
    def is_session_active(self) -> bool:
        """Check if the session is still active."""
        return self._state != SessionState.ERROR
    
    def reopen_session(self) -> None:
        """
        Reopen a closed session.
        
        This allows resuming a session that was previously closed.
        The session duration continues from where it left off (cumulative).
        """
        if not self._is_closed:
            if self.debug:
                from upsonic.utils.printing import debug_log
                debug_log("Session is already open", "SessionManager")
            return
        
        # Calculate how long the session was closed
        closed_duration = self.duration  # This was frozen at close time
        
        # Reopen by setting a new start time that accounts for elapsed time
        # This makes duration continue from where it left off
        self._start_time = time.time() - closed_duration
        self._end_time = None
        self._is_closed = False
        self._last_activity_time = time.time()
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(
                f"Session reopened: continuing from duration={closed_duration:.2f}s",
                "SessionManager"
            )
    
    async def areopen_session(self) -> None:
        """Reopen a closed session (async)."""
        if not self._is_closed:
            if self.debug:
                from upsonic.utils.printing import debug_log
                debug_log("Session is already open (async)", "SessionManager")
            return
        
        closed_duration = self.duration
        self._start_time = time.time() - closed_duration
        self._end_time = None
        self._is_closed = False
        self._last_activity_time = time.time()
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(
                f"Session reopened (async): continuing from duration={closed_duration:.2f}s",
                "SessionManager"
            )
    

    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the session."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self._state.value,
            "concurrent_invocations": self._concurrent_invocations,
            "max_concurrent_invocations": self._max_concurrent_invocations,
            "message_count": self.get_message_count(),
            "duration": self.duration,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "is_closed": self._is_closed,
            "last_activity": self.last_activity,
            "can_accept_invocation": self.can_accept_invocation()
        }
