"""
Tools for DeepAgent

Provides complete toolkits for DeepAgent functionality:

Filesystem Tools:
- ls: List files in a directory
- read_file: Read file content with pagination
- write_file: Create/overwrite files
- edit_file: Modify existing files with exact string replacement
- glob: Find files matching patterns
- grep: Search for text within files

Planning Tools:
- write_todos: Task decomposition and planning

Subagent Tools:
- task: Spawn ephemeral subagents for complex isolated tasks

All tools integrate with the backend system and use proper security validation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .filesystem_toolkit import FilesystemToolKit
    from .planning_toolkit import PlanningToolKit, Todo, TodoList
    from .subagent_toolkit import SubagentToolKit

def _get_toolkit_classes():
    """Lazy import of toolkit classes."""
    from .filesystem_toolkit import FilesystemToolKit
    from .planning_toolkit import PlanningToolKit, Todo, TodoList
    from .subagent_toolkit import SubagentToolKit
    
    return {
        'FilesystemToolKit': FilesystemToolKit,
        'PlanningToolKit': PlanningToolKit,
        'Todo': Todo,
        'TodoList': TodoList,
        'SubagentToolKit': SubagentToolKit,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    toolkit_classes = _get_toolkit_classes()
    if name in toolkit_classes:
        return toolkit_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "FilesystemToolKit",
    "PlanningToolKit",
    "Todo",
    "TodoList",
    "SubagentToolKit",
]

