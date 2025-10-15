"""
Deep Agent Module

This module provides Deep Agent functionality for handling complex, multi-step tasks
with advanced capabilities including:
- Todo management for task planning
- Virtual filesystem for file operations  
- Subagent spawning for context isolation

Main exports:
- DeepAgent: Main deep agent class
- create_deep_agent: Convenience function (alias for DeepAgent)
- State classes and tools for advanced usage
"""

from upsonic.agent.deep_agent.deep_agent import DeepAgent, create_deep_agent
from upsonic.agent.deep_agent.state import DeepAgentState, Todo
from upsonic.agent.deep_agent.tools import (
    write_todos,
    ls,
    read_file,
    write_file,
    edit_file,
    create_task_tool
)

__all__ = [
    # Main classes
    'DeepAgent',
    'create_deep_agent',
    
    # State management
    'DeepAgentState',
    'Todo',
    
    # Tools
    'write_todos',
    'ls',
    'read_file',
    'write_file',
    'edit_file',
    'create_task_tool',
]
