"""Chat module for Upsonic agent framework.

This module provides the Chat class and related components for managing
conversational AI sessions with storage binding, memory integration, and cost tracking.

Key Components:
- Chat: High-level interface for conversational AI sessions
- SessionManager: Session management with storage binding
- SessionState: Session state enumeration
- ChatMessage: Developer-friendly message representation
- CostTracker: Cost calculation utilities

For token / cost / duration totals, read fields off ``chat.usage``
(:class:`upsonic.usage_registry.AggregatedUsage`).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat import Chat, InvokeResult
    from .message import ChatMessage, ChatAttachment
    from .session_manager import SessionManager, SessionState
    from .cost_calculator import CostTracker, format_cost, format_tokens


def _get_chat_classes() -> dict:
    """Lazy import of chat classes."""
    from .chat import Chat, InvokeResult
    from .message import ChatMessage, ChatAttachment
    from .session_manager import SessionManager, SessionState
    from .cost_calculator import CostTracker, format_cost, format_tokens

    return {
        'Chat': Chat,
        'InvokeResult': InvokeResult,
        'ChatMessage': ChatMessage,
        'ChatAttachment': ChatAttachment,
        'SessionManager': SessionManager,
        'SessionState': SessionState,
        'CostTracker': CostTracker,
        'format_cost': format_cost,
        'format_tokens': format_tokens,
    }


def __getattr__(name: str) -> Any:
    """Lazy loading of chat classes."""
    chat_classes = _get_chat_classes()
    if name in chat_classes:
        return chat_classes[name]

    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Available: {list(chat_classes.keys())}"
    )


__all__ = [
    "Chat",
    "InvokeResult",
    "ChatMessage",
    "ChatAttachment",
    "SessionManager",
    "SessionState",
    "CostTracker",
    "format_cost",
    "format_tokens",
]
