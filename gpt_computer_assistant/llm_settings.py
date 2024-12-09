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

    the_text = f"""
Hi, you are an platform for vertical AI. You need to understant the user aspect and then trying to do these things and give valuation.


"""

    the_website_content = get_website_content()
    if the_website_content:
        the_text += f"""
# The Website Content of the User

{the_website_content}

"""

    return the_text


