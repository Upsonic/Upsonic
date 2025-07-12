"""
Context processing module exports.
"""

# Main context processing function (backward compatibility)
from .context import context_proceess

# New strategy-based context processing
from .context_processor import ContextProcessor
from .context import get_context_processor, set_context_processor

# Base strategy interface
from .base_strategy import ContextProcessingStrategy

# Available strategies
from .task import TaskContextStrategy
from .agent import AgentContextStrategy
from .default_prompt import DefaultPromptContextStrategy
from .knowledge_base_strategy import KnowledgeBaseContextStrategy

# Legacy functions for backward compatibility
from .agent import turn_agent_to_string
from .task import turn_task_to_string
from .default_prompt import default_prompt, DefaultPrompt


__all__ = [
    # Main interface
    "context_proceess",
    "ContextProcessor",
    "get_context_processor",
    "set_context_processor",
    # Strategy pattern
    "ContextProcessingStrategy",
    "TaskContextStrategy",
    "AgentContextStrategy",
    "DefaultPromptContextStrategy",
    "KnowledgeBaseContextStrategy",
    # Legacy functions
    "turn_agent_to_string",
    "turn_task_to_string",
    "default_prompt",
    "DefaultPrompt",
]
