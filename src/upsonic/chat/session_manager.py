import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from upsonic.usage import RequestUsage
from upsonic.models import Model
from .message import ChatMessage
from .cost_calculator import CostTracker


class SessionState(Enum):
    """Session state enumeration."""
    IDLE = "idle"
    AWAITING_RESPONSE = "awaiting_response"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class SessionMetrics:
    """Session metrics and analytics."""
    session_id: str
    user_id: str
    start_time: float
    end_time: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    message_count: int = 0
    average_response_time: float = 0.0
    last_activity_time: float = field(default_factory=time.time)
    
    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    @property
    def messages_per_minute(self) -> float:
        """Calculate messages per minute."""
        duration_minutes = self.duration / 60.0
        return self.message_count / duration_minutes if duration_minutes > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "message_count": self.message_count,
            "average_response_time": self.average_response_time,
            "messages_per_minute": self.messages_per_minute,
            "last_activity_time": self.last_activity_time
        }


class SessionManager:
    """
    Comprehensive session management for Chat class.
    
    This class handles all session-related functionality including:
    - Session state management
    - Message history tracking
    - Cost and token tracking
    - Session metrics and analytics
    - Session lifecycle management
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        debug: bool = False,
        max_concurrent_invocations: int = 1
    ):
        """
        Initialize SessionManager.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            debug: Enable debug logging
            max_concurrent_invocations: Maximum concurrent invocations allowed
        """
        self.session_id = session_id
        self.user_id = user_id
        self.debug = debug
        
        # Session state
        self._state = SessionState.IDLE
        self._concurrent_invocations = 0
        self._max_concurrent_invocations = max_concurrent_invocations
        
        # Message history
        self._message_history: List[ChatMessage] = []
        
        # Cost tracking
        self._cost_tracker = CostTracker()
        
        # Session metrics
        self._session_start_time = time.time()
        self._last_activity_time = time.time()
        self._response_times: List[float] = []
        
        self._metrics = SessionMetrics(
            session_id=session_id,
            user_id=user_id,
            start_time=self._session_start_time
        )
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"SessionManager initialized: session_id={session_id}, user_id={user_id}", "SessionManager")
    
    # State Management
    @property
    def state(self) -> SessionState:
        """Get current session state."""
        return self._state
    
    def transition_state(self, new_state: SessionState) -> None:
        """Transition to a new session state."""
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"Session state transition: {self._state.value} -> {new_state.value}", "SessionManager")
        self._state = new_state
    
    def can_accept_invocation(self) -> bool:
        """Check if session can accept a new invocation."""
        return (
            self._state != SessionState.ERROR and
            self._concurrent_invocations < self._max_concurrent_invocations
        )
    
    def start_invocation(self) -> None:
        """Start a new invocation."""
        self._concurrent_invocations += 1
        self._update_activity()
    
    def end_invocation(self) -> None:
        """End an invocation."""
        self._concurrent_invocations -= 1
        self._update_activity()
    
    def set_error_state(self) -> None:
        """Set session to error state."""
        self.transition_state(SessionState.ERROR)
    
    def reset_session(self) -> None:
        """Reset session to initial state."""
        self._state = SessionState.IDLE
        self._concurrent_invocations = 0
        self._message_history.clear()
        self._cost_tracker = CostTracker()
        self._response_times.clear()
        self._session_start_time = time.time()
        self._last_activity_time = time.time()
        
        self._metrics = SessionMetrics(
            session_id=self.session_id,
            user_id=self.user_id,
            start_time=self._session_start_time
        )
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log("Session reset to initial state", "SessionManager")
    
    # Message History Management
    @property
    def all_messages(self) -> List[ChatMessage]:
        """Get all messages in the session (read-only)."""
        return self._message_history.copy()
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session history."""
        self._message_history.append(message)
        self._metrics.message_count = len(self._message_history)
        self._update_activity()
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"Added {message.role} message to session history", "SessionManager")
    
    def clear_history(self) -> None:
        """Clear the message history."""
        self._message_history.clear()
        self._metrics.message_count = 0
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log("Session message history cleared", "SessionManager")
    
    def get_message_count(self) -> int:
        """Get the number of messages in the session."""
        return len(self._message_history)
    
    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent messages."""
        return self._message_history[-count:] if count > 0 else self._message_history
    
    # Cost and Token Tracking
    def add_usage(self, usage: RequestUsage, model: Optional[Model] = None) -> None:
        """Add usage information to the session."""
        if not usage:
            return
        
        # Update cost tracker (single source of truth)
        self._cost_tracker.add_usage(usage, model)
        
        # Update session metrics to match cost tracker
        self._metrics.total_input_tokens = self._cost_tracker.input_tokens
        self._metrics.total_output_tokens = self._cost_tracker.output_tokens
        self._metrics.total_cost = self._cost_tracker.total_cost
        
        if self.debug:
            input_tokens = usage.input_tokens or 0
            output_tokens = usage.output_tokens or 0
            # Get the cost for this individual usage
            individual_cost = self._cost_tracker._cost_history[-1].estimated_cost if self._cost_tracker._cost_history else 0.0
            from upsonic.utils.printing import info_log
            info_log(f"Added usage: {input_tokens} in, {output_tokens} out, ${individual_cost:.4f} (cumulative: {self._cost_tracker.input_tokens} in, {self._cost_tracker.output_tokens} out, ${self._cost_tracker.total_cost:.4f})", "SessionManager")
    
    @property
    def input_tokens(self) -> int:
        """Get total input tokens for the session."""
        return self._cost_tracker.input_tokens
    
    @property
    def output_tokens(self) -> int:
        """Get total output tokens for the session."""
        return self._cost_tracker.output_tokens
    
    @property
    def total_cost(self) -> float:
        """Get total cost for the session."""
        return self._cost_tracker.total_cost
    
    def get_cost_history(self) -> List[Dict[str, Any]]:
        """Get detailed cost history."""
        return self._cost_tracker.get_cost_history()
    
    # Response Time Tracking
    def start_response_timer(self) -> float:
        """Start timing a response."""
        return time.time()
    
    def end_response_timer(self, start_time: float) -> float:
        """End timing a response and record the duration."""
        response_time = time.time() - start_time
        self._response_times.append(response_time)
        
        # Update average response time
        self._metrics.average_response_time = sum(self._response_times) / len(self._response_times)
        
        return response_time
    
    # Activity Tracking
    def _update_activity(self) -> None:
        """Update last activity time."""
        self._last_activity_time = time.time()
        self._metrics.last_activity_time = self._last_activity_time
    
    @property
    def last_activity(self) -> float:
        """Get last activity timestamp."""
        return self._last_activity_time
    
    @property
    def session_duration(self) -> float:
        """Get session duration in seconds."""
        return self._metrics.duration
    
    # Session Metrics
    def get_session_metrics(self) -> SessionMetrics:
        """Get current session metrics."""
        return self._metrics
    
    def get_session_summary(self) -> str:
        """Get a formatted session summary."""
        metrics = self._metrics
        return f"""Session Summary:
  Duration: {metrics.duration:.1f}s
  Messages: {metrics.message_count}
  Tokens: {metrics.total_input_tokens + metrics.total_output_tokens}
  Cost: ${metrics.total_cost:.4f}
  Avg Response Time: {metrics.average_response_time:.2f}s
  Messages/min: {metrics.messages_per_minute:.1f}"""
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self._state.value,
            "concurrent_invocations": self._concurrent_invocations,
            "metrics": self._metrics.to_dict(),
            "message_count": len(self._message_history),
            "cost_history": self.get_cost_history()
        }
    
    # Session Lifecycle
    def close_session(self) -> None:
        """Close the session and finalize metrics."""
        self._metrics.end_time = time.time()
        self._state = SessionState.IDLE
        self._concurrent_invocations = 0
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"Session closed: {self.get_session_summary()}", "SessionManager")
    
    def is_session_active(self) -> bool:
        """Check if the session is still active."""
        return self._state != SessionState.ERROR and self._concurrent_invocations >= 0
    
    # Debug and Monitoring
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the session."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self._state.value,
            "concurrent_invocations": self._concurrent_invocations,
            "max_concurrent_invocations": self._max_concurrent_invocations,
            "message_count": len(self._message_history),
            "session_duration": self.session_duration,
            "last_activity": self.last_activity,
            "can_accept_invocation": self.can_accept_invocation()
        }
