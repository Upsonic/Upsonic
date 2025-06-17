import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



from .tasks.tasks import Task

from .knowledge_base.knowledge_base import KnowledgeBase
from .direct.direct_llm_cal import Direct
from .direct.direct_llm_cal import Direct as Agent
from .graph.graph import Graph, DecisionFunc, DecisionLLM, TaskNode, TaskChain, State
from .canvas.canvas import Canvas
from .team.team import Team

# Export error handling components for advanced users
from .utils.package.exception import (
    UupsonicError, 
    AgentExecutionError, 
    ModelConnectionError, 
    TaskProcessingError, 
    ConfigurationError, 
    RetryExhaustedError,
    NoAPIKeyException
)
from .utils.error_wrapper import upsonic_error_handler




def hello() -> str:
    return "Hello from upsonic!"


__all__ = [
    "hello", 
    "Task", 
    "KnowledgeBase", 
    "Direct", 
    "Agent",
    "Graph",
    "DecisionFunc",
    "DecisionLLM",
    "TaskNode",
    "TaskChain",
    "State",
    "Canvas",
    "MultiAgent",
    # Error handling exports
    "UupsonicError",
    "AgentExecutionError", 
    "ModelConnectionError", 
    "TaskProcessingError", 
    "ConfigurationError", 
    "RetryExhaustedError",
    "NoAPIKeyException",
    "upsonic_error_handler"
]
