"""
Autonomous Agent module for the Upsonic AI Agent Framework.

This module provides a pre-configured agent with:
- Default InMemoryStorage for session persistence
- Built-in filesystem tools for file operations
- Built-in shell command execution tools
- Memory management out of the box
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .autonomous_agent import AutonomousAgent
    from .filesystem_toolkit import AutonomousFilesystemToolKit
    from .shell_toolkit import AutonomousShellToolKit


def _get_classes() -> dict[str, Any]:
    """Lazy import of autonomous agent classes."""
    from .autonomous_agent import AutonomousAgent
    from .filesystem_toolkit import AutonomousFilesystemToolKit
    from .shell_toolkit import AutonomousShellToolKit
    
    return {
        "AutonomousAgent": AutonomousAgent,
        "AutonomousFilesystemToolKit": AutonomousFilesystemToolKit,
        "AutonomousShellToolKit": AutonomousShellToolKit,
    }


def __getattr__(name: str) -> Any:
    """Lazy loading of autonomous agent classes."""
    classes = _get_classes()
    if name in classes:
        return classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Available: {list(classes.keys())}"
    )


__all__ = [
    "AutonomousAgent",
    "AutonomousFilesystemToolKit",
    "AutonomousShellToolKit",
]
