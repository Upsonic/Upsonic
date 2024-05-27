from openai import OpenAI
from langchain_openai import ChatOpenAI

from .utils.db import load_api_key





def get_model():
    the_api_key = load_api_key()
    return ChatOpenAI(model="gpt-4o", api_key=the_api_key)


def get_client():
    the_api_key = load_api_key()
    return OpenAI(api_key=the_api_key)
