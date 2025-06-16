from .agent import turn_agent_to_string
from ..tasks.tasks import Task
from ..direct.direct_llm_cal import Direct as Agent
from ..direct.direct_llm_cal import Direct
from .task import turn_task_to_string

def context_proceess(context):


    if context is None:
        return ""

    
    TOTAL_CONTEXT = "<Context>"


    AGENT_CONTEXT = "<Agents>"

    TASK_CONTEXT  = "<Tasks>"

    for each in context:
        if isinstance(each, Task):
            TASK_CONTEXT += f"Task ID ({each.get_task_id()}): " + turn_task_to_string(each) + "\n"
        if isinstance(each, Agent) or isinstance(each, Direct):
            AGENT_CONTEXT += f"Agent ID ({each.get_agent_id()}): " + turn_agent_to_string(each) + "\n"


    
    TASK_CONTEXT += "</Tasks>"
    AGENT_CONTEXT += "</Agents>"




    TOTAL_CONTEXT += AGENT_CONTEXT
    TOTAL_CONTEXT += TASK_CONTEXT
    TOTAL_CONTEXT += "</Context>"



    
    return TOTAL_CONTEXT

    



