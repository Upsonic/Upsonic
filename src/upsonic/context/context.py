from .agent import turn_agent_to_string
from ..tasks.tasks import Task
from ..direct.direct_llm_cal import Direct as Agent
from ..direct.direct_llm_cal import Direct
from .task import turn_task_to_string
from .default_prompt import default_prompt, DefaultPrompt
from ..knowledge_base.knowledge_base import KnowledgeBase


class ContextBuilder(object):
    def __init__(self, context):
        self._context = context if context is not None else []
        self._context.append(default_prompt())
        self.task_context = "<Tasks>"
        self.agent_context = "<Agents>"
        self.default_prompt_context = "<Default Prompt>"
        self.knowledge_base_context = "<Knowledge Base>"

    def build(self):

        for each in self._context:
            if isinstance(each, Task):
                self.task_context += f"Task ID ({each.get_task_id()}): " + turn_task_to_string(each) + "\n"
            if isinstance(each, Agent) or isinstance(each, Direct):
                self.agent_context += f"Agent ID ({each.get_agent_id()}): " + turn_agent_to_string(each) + "\n"
            if isinstance(each, DefaultPrompt):
                self.default_prompt_context += f"Default Prompt: {each.prompt}\n"
            if isinstance(each, KnowledgeBase):
                self.knowledge_base_context += f"Knowledge Base: {each.markdown()}\n"

        self.task_context += "</Tasks>"
        self.agent_context += "</Agents>"
        self.default_prompt_context += "</Default Prompt>"
        self.knowledge_base_context += "</Knowledge Base>"

        return ("<Context>" + self.agent_context +
                self.task_context + self.default_prompt_context +
                self.knowledge_base_context + "</Context>")
