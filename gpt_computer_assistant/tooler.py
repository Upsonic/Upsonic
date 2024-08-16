from langchain.tools import tool

try:
    from .utils.db import load_api_key
    from .llm import get_model
    from .top_bar_wrapper import wrapper
except ImportError:
    from top_bar_wrapper import wrapper


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
    func = wrapper(func)
    custom_tools_.append(tool(func))
    return func
