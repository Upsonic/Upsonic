from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .sessions import (
        UserProfile,
        InteractionSession
    )

def _get_session_classes():
    """Lazy import of session classes."""
    from .sessions import (
        UserProfile,
        InteractionSession
    )
    
    return {
        'UserProfile': UserProfile,
        'InteractionSession': InteractionSession,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    session_classes = _get_session_classes()
    if name in session_classes:
        return session_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "UserProfile",
    "InteractionSession"
]