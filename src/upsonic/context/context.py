from .agent import turn_agent_to_string
from ..tasks.tasks import Task
from ..direct.direct_llm_cal import Direct as Agent
from ..direct.direct_llm_cal import Direct
from .task import turn_task_to_string
from .default_prompt import default_prompt, DefaultPrompt
from ..knowledge_base.knowledge_base import KnowledgeBase


def context_proceess(context):


    if context is None:
        context = []


    context.append(default_prompt())

    
    TOTAL_CONTEXT = "<Context>"



    KNOWLEDGE_BASE_CONTEXT = "<Knowledge Base>"
    AGENT_CONTEXT = "<Agents>"

    TASK_CONTEXT  = "<Tasks>"

    DEFAULT_PROMPT_CONTEXT = "<Default Prompt>"

    for each in context:
        if isinstance(each, Task):
            TASK_CONTEXT += f"Task ID ({each.get_task_id()}): " + turn_task_to_string(each) + "\n"
        if isinstance(each, Agent) or isinstance(each, Direct):
            AGENT_CONTEXT += f"Agent ID ({each.get_agent_id()}): " + turn_agent_to_string(each) + "\n"
        if isinstance(each, DefaultPrompt):
            DEFAULT_PROMPT_CONTEXT += f"Default Prompt: {each.prompt}\n"
        if isinstance(each, KnowledgeBase):
            KNOWLEDGE_BASE_CONTEXT += f"Knowledge Base: {each.markdown()}\n"

    
    TASK_CONTEXT += "</Tasks>"
    AGENT_CONTEXT += "</Agents>"
    DEFAULT_PROMPT_CONTEXT += "</Default Prompt>"
    KNOWLEDGE_BASE_CONTEXT += "</Knowledge Base>"


    TOTAL_CONTEXT += AGENT_CONTEXT
    TOTAL_CONTEXT += TASK_CONTEXT
    TOTAL_CONTEXT += DEFAULT_PROMPT_CONTEXT
    TOTAL_CONTEXT += KNOWLEDGE_BASE_CONTEXT
    TOTAL_CONTEXT += "</Context>"



    
    return TOTAL_CONTEXT

    



