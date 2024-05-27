from ..llm import *

from upsonic import Tiger
tools = Tiger()
tools.enable_auto_requirements = False
tools = tools.langchain()


from langgraph.checkpoint.sqlite import SqliteSaver

from ..utils.db import memory_db

memory = SqliteSaver.from_conn_string(memory_db)



from langgraph.prebuilt import chat_agent_executor
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools, checkpointer=memory)




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
agent_executor = chat_agent_executor.create_tool_calling_executor(model, [repl_tool], checkpointer=memory)
"""
