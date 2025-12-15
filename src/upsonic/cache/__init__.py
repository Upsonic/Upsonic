"""
Upsonic Cache Module

This module provides session-level cache management for Task objects.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cache_manager import CacheManager

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    if name == "CacheManager":
        from .cache_manager import CacheManager
        return CacheManager
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = ["CacheManager"]
