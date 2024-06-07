from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

try:
    from .utils.db import load_api_key, load_openai_url, load_model_settings, load_groq_api_key
except ImportError:
    from utils.db import load_api_key, load_openai_url, load_model_settings, load_groq_api_key


def get_model():
    the_model = load_model_settings()
    if the_model == "gpt-4o":
        the_api_key = load_api_key()
        the_openai_url = load_openai_url()
        if the_openai_url == "default":
            return ChatOpenAI(model="gpt-4o", api_key=the_api_key, base_url=the_openai_url)
        else:
            return ChatOpenAI(model="gpt-4o", api_key=the_api_key, base_url=the_openai_url)
    elif the_model == "llava":
        return ChatOllama(model="llava")
    elif the_model == "bakllava":
        return ChatOllama(model="bakllava")
    elif the_model == "mixtral-8x7b-groq":
        the_api_key = load_groq_api_key()
        return ChatGroq(
            temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=the_api_key
        )


def get_client():
    the_api_key = load_api_key()
    the_openai_url = load_openai_url()
    if the_openai_url == "default":
        return OpenAI(api_key=the_api_key)
    else:
        return OpenAI(api_key=the_api_key, base_url=the_openai_url)