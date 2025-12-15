from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .call_manager import CallManager
    from .task_manager import TaskManager
    from .reliability_manager import ReliabilityManager
    from .llm_manager import LLMManager
    from .system_prompt_manager import SystemPromptManager
    from .context_manager import ContextManager
    from .memory_manager import MemoryManager

def _get_context_manager_classes():
    """Lazy import of context manager classes."""
    from .call_manager import CallManager
    from .task_manager import TaskManager
    from .reliability_manager import ReliabilityManager
    from .llm_manager import LLMManager
    from .system_prompt_manager import SystemPromptManager
    from .context_manager import ContextManager
    from .memory_manager import MemoryManager
    
    return {
        'CallManager': CallManager,
        'TaskManager': TaskManager,
        'ReliabilityManager': ReliabilityManager,
        'LLMManager': LLMManager,
        'SystemPromptManager': SystemPromptManager,
        'ContextManager': ContextManager,
        'MemoryManager': MemoryManager,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    context_manager_classes = _get_context_manager_classes()
    if name in context_manager_classes:
        return context_manager_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    'SystemPromptManager',
    'ContextManager',
    'CallManager',
    'TaskManager',
    'ReliabilityManager',
    'MemoryManager',
    'LLMManager'
] 