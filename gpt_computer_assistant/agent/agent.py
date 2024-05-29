from ..llm import *




from langgraph.checkpoint.sqlite import SqliteSaver





from langgraph.prebuilt import chat_agent_executor


from upsonic import Tiger
tools = Tiger()
tools.enable_auto_requirements = False
tools = tools.langchain()
def get_agent_executor():
    return chat_agent_executor.create_tool_calling_executor(get_model(), tools)



"""
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
python_repl = PythonREPL()
# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

from langgraph.prebuilt import chat_agent_executor
def get_agent_executor():
    return chat_agent_executor.create_tool_calling_executor(get_model(), [repl_tool])
"""


