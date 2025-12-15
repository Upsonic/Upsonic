import warnings
import importlib
from typing import Any

from dotenv import load_dotenv

from upsonic.utils.logging_config import *

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

__version__ = "0.1.0"

_lazy_imports = {}

load_dotenv()

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
    return _lazy_import("upsonic.tasks.tasks", "Task")()

def _get_KnowledgeBase():
    return _lazy_import("upsonic.knowledge_base.knowledge_base", "KnowledgeBase")()

def _get_Agent():
    return _lazy_import("upsonic.agent.agent", "Agent")()

def _get_Graph():
    return _lazy_import("upsonic.graph.graph", "Graph")()

def _get_Team():
    return _lazy_import("upsonic.team.team", "Team")()

def _get_Chat():
    return _lazy_import("upsonic.chat.chat", "Chat")()

def _get_Direct():
    return _lazy_import("upsonic.direct", "Direct")()

def hello() -> str:
    return "Hello from upsonic!"

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes.
    
    Only Agent, Task, KnowledgeBase, Graph, Team, Chat, Direct are directly available.
    All other classes must be imported from their sub-modules.
    """
    
    # Only these 7 classes are directly available
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
    
    # All other imports must come from sub-modules
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module. "
        f"For example: from upsonic.agent.run_result import AgentRunResult"
    )

__all__ = [
    "hello",
    "Task",
    "KnowledgeBase",
    "Agent",
    "Graph",
    "Team",
    "Chat",
    "Direct",
]