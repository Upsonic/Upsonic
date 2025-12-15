from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory import Memory

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    if name == "Memory":
        from .memory import Memory
        return Memory
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "Memory",
]