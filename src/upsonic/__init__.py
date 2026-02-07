import warnings
import importlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from upsonic.utils.logging_config import *

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

__version__ = "0.1.0"

_lazy_imports = {}

# Load .env file from current working directory (where user runs their script)
# This ensures .env is found even when package is installed in site-packages
cwd = Path(os.getcwd())
env_path = cwd / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)
else:
    # Fallback: search from current directory upwards (default behavior)
    load_dotenv(override=False)

def _lazy_import(module_name: str, class_name: str = None):
    """Lazy import function to defer heavy imports until actually needed."""
    def _import():
        if module_name not in _lazy_imports:
            _lazy_imports[module_name] = importlib.import_module(module_name)
        
        if class_name:
            return getattr(_lazy_imports[module_name], class_name)
        return _lazy_imports[module_name]
    
    return _import

def _get_Task():
    task_cls = _lazy_import("upsonic.tasks.tasks", "Task")()
    # Ensure all dependencies are imported before rebuilding
    try:
        # Import dependencies to resolve forward references
        _lazy_import("upsonic.embeddings.factory", "EmbeddingProvider")()
        _lazy_import("upsonic.agent.agent", "Agent")()
        _lazy_import("upsonic.cache.cache_manager", "CacheManager")()
        _lazy_import("upsonic.tools.base", "Tool")()
        # Now rebuild the model
        task_cls.model_rebuild()
    except Exception:
        pass
    return task_cls

def _get_KnowledgeBase():
    return _lazy_import("upsonic.knowledge_base.knowledge_base", "KnowledgeBase")()

def _get_Agent():
    agent_cls = _lazy_import("upsonic.agent.agent", "Agent")()
    # After Agent is imported, rebuild Task model to resolve forward references
    try:
        from upsonic.tasks.tasks import Task
        Task.model_rebuild()
    except Exception:
        pass
    return agent_cls

def _get_Graph():
    return _lazy_import("upsonic.graph.graph", "Graph")()

def _get_Team():
    return _lazy_import("upsonic.team.team", "Team")()

def _get_Chat():
    return _lazy_import("upsonic.chat.chat", "Chat")()

def _get_Direct():
    return _lazy_import("upsonic.direct", "Direct")()

def _get_Simulation():
    return _lazy_import("upsonic.simulation.simulation", "Simulation")()

def _get_RalphLoop():
    return _lazy_import("upsonic.ralph.loop", "RalphLoop")()

def _get_AutonomousAgent():
    return _lazy_import("upsonic.agent.autonomous_agent.autonomous_agent", "AutonomousAgent")()

def hello() -> str:
    return "Hello from upsonic!"

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes.
    
    Only Agent, Task, KnowledgeBase, Graph, Team, Chat, Direct, cancel_run are directly available.
    All other classes must be imported from their sub-modules.
    """
    
    # Only these classes are directly available
    if name == "Task":
        return _get_Task()
    elif name == "KnowledgeBase":
        return _get_KnowledgeBase()
    elif name == "Agent":
        return _get_Agent()
    elif name == "Graph":
        return _get_Graph()
    elif name == "Team":
        return _get_Team()
    elif name == "Chat":
        return _get_Chat()
    elif name == "Direct":
        return _get_Direct()
    elif name == "Simulation":
        return _get_Simulation()
    elif name == "RalphLoop":
        return _get_RalphLoop()
    elif name == "AutonomousAgent":
        return _get_AutonomousAgent()
    
    # All other imports must come from sub-modules
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module. "
        f"For example: from upsonic.agent.agent import Agent"
    )

__all__ = [
    "hello",
    "Task",
    "KnowledgeBase",
    "Agent",
    "AutonomousAgent",
    "Graph",
    "Team",
    "Chat",
    "Direct",
    "Simulation",
    "RalphLoop",
]