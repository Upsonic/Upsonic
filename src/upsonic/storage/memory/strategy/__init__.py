"""Memory strategy base types for Upsonic storage."""
from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy loading of strategy classes."""
    if name == "BaseMemoryStrategy":
        from .base import BaseMemoryStrategy

        return BaseMemoryStrategy

    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
    )


__all__ = [
    "BaseMemoryStrategy",
]
