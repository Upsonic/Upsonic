from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase, KBState

_LAZY_MAP = {
    "KnowledgeBase": "KnowledgeBase",
    "KBState": "KBState",
}

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    if name in _LAZY_MAP:
        from . import knowledge_base as _mod
        return getattr(_mod, _LAZY_MAP[name])
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "KnowledgeBase",
    "KBState",
]