import os
from dotenv import load_dotenv

load_dotenv()

def model_set(model):
    if model is None:
        model = os.getenv("LLM_MODEL_KEY").split(":")[0] if os.getenv("LLM_MODEL_KEY", None) else "openai/gpt-4o"
    return model