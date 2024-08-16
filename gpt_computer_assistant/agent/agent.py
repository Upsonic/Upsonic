try:
    from ..llm import get_model
    from ..utils.db import *
    from ..llm_settings import llm_settings
    from ..tooler import *
    from ..display_tools import *
    from ..teams import *
    from .agent_tools import get_tools
except ImportError:
    from llm import get_model
    from utils.db import *
    from llm_settings import llm_settings
    from tooler import *
    from display_tools import *
    from teams import *
    from agent.agent_tools import get_tools


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


def get_agent_executor():
    tools = get_tools()
    tools += custom_tools()

    model = load_model_settings()

    if is_predefined_agents_setting_active() and llm_settings[model]["tools"]:
        try:
            import crewai

            tools += [search_on_internet_and_report_team, generate_code_with_aim_team]
        except ImportError:
            pass

    if llm_settings[model]["provider"] == "openai":
        tools += [
            click_on_a_text_on_the_screen,
            click_on_a_icon_on_the_screen,
            move_on_a_text_on_the_screen,
            move_on_a_icon_on_the_screen,
            mouse_scroll,
        ]

    tools += [get_texts_on_the_screen]

    if (
        llm_settings[model]["provider"] == "openai"
        or llm_settings[model]["provider"] == "groq"
    ):
        return chat_agent_executor.create_tool_calling_executor(get_model(), tools)

    if llm_settings[model]["provider"] == "ollama":
        print("Ollama tool len", len(tools))
        return chat_agent_executor.create_tool_calling_executor(get_model(), tools)
