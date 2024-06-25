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
    from agent_tools import get_tools


from langchain.agents import AgentExecutor, create_json_chat_agent


from langgraph.prebuilt import chat_agent_executor


custom_tools = []




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
    global custom_tools
    tools = get_tools()
    tools += custom_tools

    if is_predefined_agents_setting_active():
        try:
            import crewai
            tools += [search_on_internet_and_report_team, generate_code_with_aim_team]
        except ImportError:
            pass


    model = load_model_settings()


    if llm_settings[model]["provider"] == "openai":
        tools += [click_on_a_text_on_the_screen, click_on_a_icon_on_the_screen, move_on_a_text_on_the_screen, move_on_a_icon_on_the_screen, mouse_scroll]


    if llm_settings[model]["provider"] == "openai" or llm_settings[model]["provider"] == "groq":
        return chat_agent_executor.create_tool_calling_executor(get_model(), tools)



    if llm_settings[model]["provider"] == "ollama":
        from langchain import hub

        prompt = get_prompt("hwchase17/react-chat-json")
        the_agent = create_json_chat_agent(get_model(), tools, prompt)
        return AgentExecutor(
            agent=the_agent, tools=tools, verbose=True, handle_parsing_errors=True
        )


