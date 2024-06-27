from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

try:
    from .utils.db import load_api_key, load_openai_url, load_model_settings, load_groq_api_key, load_google_api_key
    from .custom_callback import customcallback
except ImportError:
    from utils.db import load_api_key, load_openai_url, load_model_settings, load_groq_api_key, load_google_api_key
    from custom_callback import customcallback



the_callback = customcallback(strip_tokens=False, answer_prefix_tokens=["Answer"])



def get_model(high_context=False):
    the_model = load_model_settings()
    the_api_key = load_api_key()
    the_groq_api_key = load_groq_api_key()
    the_google_api_key = load_google_api_key()
    the_openai_url = load_openai_url()

    def open_ai_base(high_context):
        if the_openai_url == "default":
            true_model = the_model
            if high_context:
                true_model = "gpt-4-turbo"
            return {"model": true_model, "api_key": the_api_key, "max_retries":15, "streaming":True, "callbacks":[the_callback]}
        else:
            return {"model": the_model, "api_key": the_api_key, "max_retries":15, "streaming":True, "callbacks":[the_callback], "base_url": the_openai_url}

    args_mapping = {
        ChatOpenAI: open_ai_base(high_context=high_context),
        ChatOllama: {"model": the_model},
        ChatGroq: {"temperature": 0, "model_name": the_model.replace("-groq", ""), "groq_api_key": the_openai_url},
        ChatGoogleGenerativeAI:{"model": the_model, "google_api_key": the_google_api_key}
    }
    model_mapping = {
        # OpenAI
        "gpt-4o": (ChatOpenAI, args_mapping[ChatOpenAI]),
        "gpt-4-turbo": (ChatOpenAI, args_mapping[ChatOpenAI]),
        "gpt-3.5": (ChatOpenAI, args_mapping[ChatOpenAI]),
        "gpt-3.5-turbo": (ChatOpenAI, args_mapping[ChatOpenAI]),

        # Google Generative AI - Llama
        "llava": (ChatOllama, args_mapping[ChatOllama]),
        "llama3": (ChatOllama, args_mapping[ChatOllama]),
        "bakllava": (ChatOllama, args_mapping[ChatOllama]),

        # Google Generative AI - Gemini
        "gemini-pro": (ChatGoogleGenerativeAI, args_mapping[ChatGoogleGenerativeAI]),

        # Groq
        "mixtral-8x7b-groq": (ChatGroq, args_mapping[ChatGroq])
    }

    model_class, args = model_mapping[the_model]
    return model_class(**args) if model_class else None


def get_client():
    the_api_key = load_api_key()
    the_openai_url = load_openai_url()
    if the_openai_url == "default":
        return OpenAI(api_key=the_api_key)
    else:
        return OpenAI(api_key=the_api_key, base_url=the_openai_url)
