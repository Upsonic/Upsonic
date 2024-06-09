from langchain.tools import tool
from .agent.agent import custom_tools

def Tool(func):
    """
    A decorator function to register a tool with the custom tools list.

    Parameters:
    - func (callable): The function to be registered as a tool.

    Returns:
    - callable: The input function `func` unchanged.
    """
    global custom_tools
    custom_tools.append(tool(func))
    return func
