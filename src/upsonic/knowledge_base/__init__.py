from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    if name == "KnowledgeBase":
        from .knowledge_base import KnowledgeBase
        return KnowledgeBase
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "KnowledgeBase"
]