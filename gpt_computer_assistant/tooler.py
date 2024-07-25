from langchain.tools import tool

try:
    from .utils.db import load_api_key
    from .llm import get_model
except ImportError:
    from utils.db import load_api_key
    from llm import get_model


def Tool(func):
    """
    A decorator function to register a tool with the custom tools list.

    Parameters:
    - func (callable): The function to be registered as a tool.

    Returns:
    - callable: The input function `func` unchanged.
    """
    from .agent.agent import custom_tools_
    global custom_tools_
    custom_tools_.append(tool(func))
    return func

