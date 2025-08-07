import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



from upsonic.tasks.tasks import Task

from upsonic.knowledge_base.knowledge_base import KnowledgeBase
from upsonic.agent.agent import Direct
from upsonic.agent.agent import Direct as Agent
from upsonic.graph.graph import Graph, DecisionFunc, DecisionLLM, TaskNode, TaskChain, State
from upsonic.canvas.canvas import Canvas
from upsonic.team.team import Team

# Export error handling components for advanced users
from upsonic.utils.package.exception import (
    UupsonicError, 
    AgentExecutionError, 
    ModelConnectionError, 
    TaskProcessingError, 
    ConfigurationError, 
    RetryExhaustedError,
    NoAPIKeyException
)
from upsonic.utils.error_wrapper import upsonic_error_handler


from .storage import (
    Storage,
    InMemoryStorage,
    JSONStorage,
    PostgresStorage,
    RedisStorage,
    SqliteStorage,
    SessionId,
    UserId,
    InteractionSession,
    UserProfile,
    Memory
)




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
    "Team",
    "UupsonicError",
    "AgentExecutionError", 
    "ModelConnectionError", 
    "TaskProcessingError", 
    "ConfigurationError", 
    "RetryExhaustedError",
    "NoAPIKeyException",
    "upsonic_error_handler",
    "Memory",
    "Storage",
    "InMemoryStorage",
    "JSONStorage",
    "PostgresStorage",
    "RedisStorage",
    "SqliteStorage",
    "InteractionSession",
    "UserProfile",
    "SessionId",
    "UserId",
]
