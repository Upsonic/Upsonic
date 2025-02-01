from dataclasses import Field
import uuid
from pydantic import BaseModel


from typing import Any, List, Dict, Optional, Type, Union


from ..knowledge_base.knowledge_base import KnowledgeBase


class AgentConfiguration(BaseModel):
    agent_id_: str = None
    job_title: str
    company_url: str = None
    company_objective: str = None
    name: str = ""
    contact: str = ""
    model: str = "openai/gpt-4o"

    sub_task: bool = True

    reflection: bool = False

    memory: bool = False

    caching: bool = True
    cache_expiry: int = 60 * 60

    knowledge_base: KnowledgeBase = None
    tools: List[Any] = []

    context_compress: bool = True


    @property
    def retries(self):
        if self.reflection:
            return 5
        else:
            return 2


    @property
    def agent_id(self):
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_