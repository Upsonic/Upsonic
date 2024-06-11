try:
    from ..llm import get_model
    from ..utils.db import load_model_settings
    from ..llm_settings import llm_settings
    from ..tooler import click_on_a_text_on_the_screen, click_on_a_icon_on_the_screen
except ImportError:
    from llm import get_model
    from utils.db import load_model_settings
    from llm_settings import llm_settings
    from tooler import click_on_a_text_on_the_screen, click_on_a_icon_on_the_screen


from langgraph.checkpoint.sqlite import SqliteSaver


from langchain.agents import AgentExecutor, create_json_chat_agent


from langgraph.prebuilt import chat_agent_executor


custom_tools = []

try:
    from upsonic import Tiger
    tools = Tiger()
    tools.enable_auto_requirements = True
    tools = tools.langchain()
except:
    from langchain.agents import Tool
    from langchain_experimental.utilities import PythonREPL

    python_repl = PythonREPL()
    # You can create the tool to pass to an agent
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )
    tools = [repl_tool]


prompt_cache = {}


def get_prompt(name):
    global prompt_cache
    if name in prompt_cache:
        return prompt_cache[name]
    else:
        from langchain import hub

        prompt = hub.pull(name)
        prompt_cache[name] = prompt
        return prompt


def get_agent_executor():
    global custom_tools, tools
    tools += custom_tools
    model = load_model_settings()


    if llm_settings[model]["provider"] == "openai":
        tools += [click_on_a_text_on_the_screen, click_on_a_icon_on_the_screen]


    if llm_settings[model]["provider"] == "openai" or llm_settings[model]["provider"] == "groq":
        return chat_agent_executor.create_tool_calling_executor(get_model(), tools)



    if llm_settings[model]["provider"] == "ollama":
        from langchain import hub

        prompt = get_prompt("hwchase17/react-chat-json")
        the_agent = create_json_chat_agent(get_model(), tools, prompt)
        return AgentExecutor(
            agent=the_agent, tools=tools, verbose=True, handle_parsing_errors=True
        )





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
