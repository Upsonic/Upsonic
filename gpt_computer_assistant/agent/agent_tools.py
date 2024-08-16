try:
    from ..utils.db import *
    from ..tooler import *
    from ..display_tools import *
    from ..teams import *
    from ..llm_settings import each_message_extension, llm_settings
except ImportError:
    from utils.db import *

    from tooler import *
    from display_tools import *
    from teams import *
    from llm_settings import llm_settings


custom_tools = []


def load_tiger_tools():
    try:
        from upsonic import Tiger

        tools = Tiger()
        tools.enable_auto_requirements = True
        tools = tools.langchain()
        return tools
    except:
        return False


def load_default_tools():
    from ..standard_tools import get_standard_tools

    return get_standard_tools()


cached_tiger_tools = None


def get_tiger_tools():
    global cached_tiger_tools
    if cached_tiger_tools is None:
        cached_tiger_tools = load_tiger_tools()
    return cached_tiger_tools


if is_online_tools_setting_active():
    get_tiger_tools()


def get_tools():
    model = load_model_settings()

    if not llm_settings[model]["tools"]:
        return []

    if is_online_tools_setting_active():
        tools = get_tiger_tools()
        if not tools:
            tools = load_default_tools()
    else:
        tools = load_default_tools()

    return tools
