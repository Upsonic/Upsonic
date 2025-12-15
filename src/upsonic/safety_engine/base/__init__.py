"""
Base classes for AI Safety Engine
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .rule_base import RuleBase
    from .action_base import ActionBase
    from .policy import Policy

def _get_base_classes():
    """Lazy import of base safety engine classes."""
    from .rule_base import RuleBase
    from .action_base import ActionBase
    from .policy import Policy
    
    return {
        'RuleBase': RuleBase,
        'ActionBase': ActionBase,
        'Policy': Policy,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = ["RuleBase", "ActionBase", "Policy"]