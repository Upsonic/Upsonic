"""Memory module for Upsonic agent framework.

This module provides memory orchestration for session and user memory operations.

Key Components:
    - Memory: Main orchestrator with runtime session type selection
    - SessionMemoryFactory: Factory for creating session memory instances
    - BaseMemoryStrategy: Shared base for all memory strategy types
    - BaseSessionMemory: Abstract base for session memory implementations
    - AgentSessionMemory: Session memory for AgentSession
    - BaseUserMemory: Abstract base for user memory implementations
    - UserMemory: User memory implementation
    - PreparedSessionInputs: Structured output from session memory get operations
    - BaseCultureMemory: Abstract base for culture memory implementations
    - CultureMemory: Culture memory implementation
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory import Memory
    from .factory import SessionMemoryFactory
    from .strategy.base import BaseMemoryStrategy
    from .session.base import BaseSessionMemory, PreparedSessionInputs
    from .session.agent import AgentSessionMemory
    from .user.base import BaseUserMemory
    from .user.user import UserMemory
    from .culture.base import BaseCultureMemory
    from .culture.culture import CultureMemory


def __getattr__(name: str) -> Any:
    """Lazy loading of memory classes."""
    if name == "Memory":
        from .memory import Memory
        return Memory
    
    if name == "SessionMemoryFactory":
        from .factory import SessionMemoryFactory
        return SessionMemoryFactory
    
    if name == "BaseMemoryStrategy":
        from .strategy.base import BaseMemoryStrategy
        return BaseMemoryStrategy
    
    if name == "BaseSessionMemory":
        from .session.base import BaseSessionMemory
        return BaseSessionMemory
    
    if name == "PreparedSessionInputs":
        from .session.base import PreparedSessionInputs
        return PreparedSessionInputs
    
    if name == "AgentSessionMemory":
        from .session.agent import AgentSessionMemory
        return AgentSessionMemory
    
    if name == "BaseUserMemory":
        from .user.base import BaseUserMemory
        return BaseUserMemory
    
    if name == "UserMemory":
        from .user.user import UserMemory
        return UserMemory
    
    if name == "BaseCultureMemory":
        from .culture.base import BaseCultureMemory
        return BaseCultureMemory
    
    if name == "CultureMemory":
        from .culture.culture import CultureMemory
        return CultureMemory
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )


__all__ = [
    "Memory",
    "SessionMemoryFactory",
    "BaseMemoryStrategy",
    "BaseSessionMemory",
    "PreparedSessionInputs",
    "AgentSessionMemory",
    "BaseUserMemory",
    "UserMemory",
    "BaseCultureMemory",
    "CultureMemory",
]
