"""
Culture module for Upsonic AI Agent Framework.

This module provides cultural knowledge management capabilities that enable
agents to share universal knowledge, principles, and best practices across
all interactions. Unlike Memory (user-specific facts), Culture stores
universal principles that benefit all agents.

Notice: Culture is an experimental feature and is subject to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cultural_knowledge import CulturalKnowledge
    from .manager import CultureManager


def _get_culture_classes():
    """Lazy import of culture classes."""
    from .cultural_knowledge import CulturalKnowledge
    from .manager import CultureManager
    
    return {
        'CulturalKnowledge': CulturalKnowledge,
        'CultureManager': CultureManager,
    }


def __getattr__(name: str) -> Any:
    """Lazy loading of culture classes."""
    culture_classes = _get_culture_classes()
    if name in culture_classes:
        return culture_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Available: {list(culture_classes.keys())}"
    )


__all__ = [
    "CulturalKnowledge",
    "CultureManager",
]
