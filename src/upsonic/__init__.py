import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


from .client.base import UpsonicClient
from .client.tasks.task_response import ObjectResponse, StrResponse, IntResponse, FloatResponse, BoolResponse, StrInListResponse
from .client.tasks.tasks import Task
from .client.agent_configuration.agent_configuration import AgentConfiguration
from .client.agent_configuration.agent_configuration import AgentConfiguration as Agent
from .client.knowledge_base.knowledge_base import KnowledgeBase
from .client.direct_llm_call.direct_llm_cal import Direct
from .client.multi_agent.multi_agent import MultiAgent


from .client.storage.storage import ClientConfig

from pydantic import Field


def hello() -> str:
    return "Hello from upsonic!"


__all__ = ["hello", "UpsonicClient", "ObjectResponse", "StrResponse", "IntResponse", "FloatResponse", "BoolResponse", "Task", "StrInListResponse", "AgentConfiguration", "Field", "KnowledgeBase", "ClientConfig", "Agent", "Direct", "MultiAgent"]
