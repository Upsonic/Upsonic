try:
    from .utils.db import *

except ImportError:
    from utils.db import *

llm_settings = {
    "gpt-4o": {
        "show_name": "gpt-4o (OpenAI)",
        "vision": True,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "claude-3-5-sonnet-20241022": {
        "show_name": "claude-3-5-sonnet-20241022 (Anthropic)",
        "vision": True,
        "provider": "anthropic",
        "tools": True,
        "stream": False,
    },
    "gpt-4o-azureopenai": {
        "show_name": "gpt-4o (AzureAI)",
        "vision": True,
        "provider": "azureai",
        "tools": True,
        "stream": True,
    },
    "gpt-4o-mini": {
        "show_name": "gpt-4o-mini (OpenAI)",
        "vision": True,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "gpt-4-turbo": {
        "show_name": "gpt-4-turbo (OpenAI)",
        "vision": False,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "gpt-3.5": {
        "show_name": "gpt-3.5 (OpenAI)",
        "vision": False,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "gpt-3.5-turbo": {
        "show_name": "gpt-3.5-turbo (OpenAI)",
        "vision": False,
        "provider": "openai",
        "tools": True,
        "stream": True,
    },
    "llama3": {
        "show_name": "Llama3 (Ollama)",
        "vision": False,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "llama3.1": {
        "show_name": "Llama3.1 (Ollama)",
        "vision": False,
        "provider": "ollama",
        "tools": True,
        "stream": False,
    },
    "qwen2:1.5b": {
        "show_name": "Qwen2 1.5b (Ollama)",
        "vision": False,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "llava": {
        "show_name": "Llava (Ollama)",
        "vision": True,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "bakllava": {
        "show_name": "BakLLaVA (Ollama)",
        "vision": True,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "llava-llama3": {
        "show_name": "Llava-Llama3 (Ollama)",
        "vision": True,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "llava-phi3": {
        "show_name": "LLaVA-Phi-3 (Ollama)",
        "vision": True,
        "provider": "ollama",
        "tools": False,
        "stream": False,
    },
    "gemini-pro": {
        "show_name": "gemini-pro (Google)",
        "vision": True,
        "provider": "google",
        "tools": True,
        "stream": True,
    },
    "mixtral-8x7b-groq": {
        "show_name": "Mixtral 8x7b (Groq)",
        "vision": False,
        "provider": "groq",
        "tools": True,
        "stream": True,
    },
}


def get_openai_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "openai"]

def get_azureai_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "azureai"]


def get_ollama_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "ollama"]


def get_google_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "google"]


def get_groq_models():
    return [k for k, v in llm_settings.items() if v["provider"] == "groq"]


llm_show_name_ = {}
for k, v in llm_settings.items():
    llm_show_name_[v["show_name"]] = k

llm_show_name = llm_show_name_


def first_message():
    from .character import name, developer, get_website_content
    from .cu.computer import width, height, display_num
    model = load_model_settings()


    if llm_settings[model]["provider"] == "azureai":
        return ""

    the_text = f"""
You are {name()} that developed by {developer()}, you are the first live AI assistant in everyone computer that can complete any task by using tools. 

Before any task, write a plan for your tasks and do it step by step. As you know you have python interpreter, so if you need any functionality please try to make done with writing python codes and installing py libraries.

Don't forget, you are capable to make any task.

Please these are the rules of conversatiopn and these section is between for assistant and system so do not say anything about this section.



Use a mouse and keyboard to interact with a computer.
* The screen's resolution is {{ width }}x{{ height }}.
* The display number is {{ display_num }}

- If the user absolutely try to click somthing, you can use mouse_move_and_left_click action.

* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.

* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.


Your GitHub Repository:
https://github.com/Upsonic/gpt-computer-assistant


"""

    the_website_content = get_website_content()
    if the_website_content:
        the_text += f"""
# The Website Content of the User

{the_website_content}

"""

    return the_text


def each_message_extension():
    model = load_model_settings()


    if llm_settings[model]["provider"] == "azureai":
        return ""

    return """

    # Usings Answer
    Please start with <Answer> in your last responses. DONT FORGET IT AND DONT TALK ABOUT THIS RULE OR REFFERENCE


    """
