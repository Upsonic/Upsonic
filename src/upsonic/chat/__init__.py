from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat import Chat
    from .message import ChatMessage
    from .session_manager import SessionManager, SessionState, SessionMetrics
    from .cost_calculator import CostTracker, format_cost, format_tokens

def _get_chat_classes():
    """Lazy import of chat classes."""
    from .chat import Chat
    from .message import ChatMessage
    from .session_manager import SessionManager, SessionState, SessionMetrics
    from .cost_calculator import CostTracker, format_cost, format_tokens
    
    return {
        'Chat': Chat,
        'ChatMessage': ChatMessage,
        'SessionManager': SessionManager,
        'SessionState': SessionState,
        'SessionMetrics': SessionMetrics,
        'CostTracker': CostTracker,
        'format_cost': format_cost,
        'format_tokens': format_tokens,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    chat_classes = _get_chat_classes()
    if name in chat_classes:
        return chat_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "Chat",
    "SessionManager",
    "SessionState",
    "SessionMetrics",
    "ChatMessage",
    "CostTracker",
    "format_cost",
    "format_tokens"
]
