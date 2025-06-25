# context/strategies.py
import json
from ..tasks.tasks import Task
from .default_prompt import DefaultPrompt
from ..knowledge_base.knowledge_base import KnowledgeBase
from .strategy import ContextStrategy
from ..agent.base import AgentBase


class TaskStrategy(ContextStrategy):
    def matches(self, obj) -> bool:
        return isinstance(obj, Task)

    def format(self, obj) -> str:
        payload = {
            "id": obj.task_id,
            "description": obj.description,
            "images": obj.images,
            "response": str(obj.response),
        }
        return f"<Tasks>Task ID ({obj.task_id}): {json.dumps(payload)}</Tasks>"


class AgentStrategy(ContextStrategy):
    def matches(self, obj) -> bool:
        return isinstance(obj, AgentBase)

    def format(self, obj) -> str:
        payload = {
            "id": getattr(obj, "agent_id", None),
            "name": getattr(obj, "name", None),
            "company_url": getattr(obj, "company_url", None),
            "company_objective": getattr(obj, "company_objective", None),
            "company_description": getattr(obj, "company_description", None),
            "system_prompt": getattr(obj, "system_prompt", None),
        }
        return f"<Agents>Agent ID ({payload['id']}): {json.dumps(payload)}</Agents>"


class DefaultPromptStrategy(ContextStrategy):
    def matches(self, obj) -> bool:
        return isinstance(obj, DefaultPrompt)

    def format(self, obj) -> str:
        return f"<Default Prompt>{obj.prompt}</Default Prompt>"


class KnowledgeBaseStrategy(ContextStrategy):
    def matches(self, obj) -> bool:
        return isinstance(obj, KnowledgeBase)

    def format(self, obj) -> str:
        return f"<Knowledge Base>{obj.markdown()}</Knowledge Base>"
