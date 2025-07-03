from .context_visitor import ContextVisitor
from ..context.agent import turn_agent_to_string
from ..context.task import turn_task_to_string

class ContextStringBuilder(ContextVisitor):
    def __init__(self):
        self.task_context = "<Tasks>"
        self.agent_context = "<Agents>"
        self.default_prompt_context = "<Default Prompt>"
        self.knowledge_base_context = "<Knowledge Base>"

    def visit_task(self, task):
        self.task_context += f"Task ID ({task.get_task_id()}): {turn_task_to_string(task)}\n"
        
    def visit_direct(self, agent):
        self.agent_context += f"Agent ID ({agent.get_agent_id()}): {turn_agent_to_string(agent)}\n"

    def visit_defaultprompt(self, prompt):
        self.default_prompt_context += f"Default Prompt: {prompt.prompt}\n"

    def visit_knowledgebase(self, kb):
        self.knowledge_base_context += f"Knowledge Base: {kb.markdown()}\n"

    def build(self):
        self.task_context += "</Tasks>"
        self.agent_context += "</Agents>"
        self.default_prompt_context += "</Default Prompt>"
        self.knowledge_base_context += "</Knowledge Base>"

        return (
            "<Context>" +
            self.agent_context +
            self.task_context +
            self.default_prompt_context +
            self.knowledge_base_context +
            "</Context>"
        )
