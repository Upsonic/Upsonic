from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from .utils.db import load_api_key, load_model_settings





def get_model():
    the_model = load_model_settings()
    if the_model == "gpt-4o":
        the_api_key = load_api_key()
        return ChatOpenAI(model="gpt-4o", api_key=the_api_key)
    elif the_model == "llava":
        return ChatOllama(model="llava")
    elif the_model == "bakllava":
        return ChatOllama(model="bakllava")




def get_client():
    the_api_key = load_api_key()
    return OpenAI(api_key=the_api_key)
