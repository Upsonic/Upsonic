"""
Team module for multi-agent operations using the Upsonic client.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from upsonic.team.team import Team
    from upsonic.team.context_sharing import ContextSharing
    from upsonic.team.task_assignment import TaskAssignment
    from upsonic.team.result_combiner import ResultCombiner

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    if name == "Team":
        from upsonic.team.team import Team
        return Team
    elif name == "ContextSharing":
        from upsonic.team.context_sharing import ContextSharing
        return ContextSharing
    elif name == "TaskAssignment":
        from upsonic.team.task_assignment import TaskAssignment
        return TaskAssignment
    elif name == "ResultCombiner":
        from upsonic.team.result_combiner import ResultCombiner
        return ResultCombiner
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'Team',
    'ContextSharing', 
    'TaskAssignment',
    'ResultCombiner'
]
