"""
Prebuilt Autonomous Agent module for the Upsonic AI Agent Framework.

Extends :class:`~upsonic.agent.autonomous_agent.AutonomousAgent` to bootstrap
its system prompt and first message from a git repo subfolder and expose
run / run_async / run_stream / run_stream_async / run_console entry points.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prebuilt_autonomous_agent import PrebuiltAutonomousAgent


def _get_classes() -> dict[str, Any]:
    """Lazy import of prebuilt autonomous agent classes."""
    from .prebuilt_autonomous_agent import PrebuiltAutonomousAgent

    return {
        "PrebuiltAutonomousAgent": PrebuiltAutonomousAgent,
    }


def __getattr__(name: str) -> Any:
    """Lazy loading of prebuilt autonomous agent classes."""
    classes = _get_classes()
    if name in classes:
        return classes[name]

    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Available: {list(classes.keys())}"
    )


__all__ = [
    "PrebuiltAutonomousAgent",
]
