try:
    from ..llm import get_model
    from ..utils.db import *
    from ..llm_settings import llm_settings
    from ..tooler import *
    from ..display_tools import *
    from ..cu.computer import *
    from ..teams import *
    from .agent_tools import get_tools
    from ..mcp.tool import mcp_tools
    from ..standard_tools import get_standard_tools
    
except ImportError:
    from llm import get_model
    from utils.db import *
    from llm_settings import llm_settings
    from tooler import *
    from display_tools import *
    from cu.computer import *
    from teams import *
    from agent.agent_tools import get_tools
    from mcp.tool import mcp_tools
    from standard_tools import get_standard_tools


from langgraph.prebuilt import chat_agent_executor


custom_tools_ = []


def custom_tools():
    global custom_tools_
    the_list = []
    the_list += custom_tools_
    return the_list


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


def get_agent_executor(the_anthropic_model=False):
    tools = get_tools()
    tools += custom_tools()

    model = load_model_settings()

    if is_predefined_agents_setting_active() and llm_settings[model]["tools"]:
        try:
            import crewai

            tools += [search_on_internet_and_report_team, generate_code_with_aim_team]
        except ImportError:
            pass


    if the_anthropic_model:
        tools += []
        model_catch = get_model(the_model="claude-3-5-sonnet-20241022")

        print("Anthropic model catch", model_catch)
        print("Anthropic tools len", len(tools))
        return chat_agent_executor.create_tool_calling_executor(model_catch, tools)
    else:
        tools += [click_to_text, click_to_icon, click_to_area] + mcp_tools() + get_standard_tools()





    if (
        llm_settings[model]["provider"] == "openai"
        or llm_settings[model]["provider"] == "groq"
        or llm_settings[model]["provider"] == "azureai",
        llm_settings[model]["provider"] == "anthropic",
    ):
        return chat_agent_executor.create_tool_calling_executor(get_model(), tools)

    if llm_settings[model]["provider"] == "ollama":
        print("Ollama tool len", len(tools))
        return chat_agent_executor.create_tool_calling_executor(get_model(), tools)
