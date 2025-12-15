from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .canvas import Canvas

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    if name == "Canvas":
        from .canvas import Canvas
        return Canvas
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = ['Canvas']