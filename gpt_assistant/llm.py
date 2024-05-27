from openai import OpenAI
from langchain_openai import ChatOpenAI

from .utils.db import load_api_key

the_api_key = load_api_key()

model = ChatOpenAI(model="gpt-4o", api_key=the_api_key)

client = OpenAI(api_key=the_api_key)
