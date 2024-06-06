
from langchain.tools import tool


from .agent.agent import custom_tools



def Tool(func):
     global custom_tools
     custom_tools.append(tool(func))
     return func

