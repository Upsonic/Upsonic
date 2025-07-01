import os
from dotenv import load_dotenv

load_dotenv()

def model_set(model):
    if model is None:
        model = os.getenv("LLM_MODEL_KEY").split(":")[0] if os.getenv("LLM_MODEL_KEY", None) else "openai/gpt-4o"
        
        try:

            from celery import current_task

            task_id = current_task.request.id
            task_args = current_task.request.args
            task_kwargs = current_task.request.kwargs

            
            if task_kwargs.get("bypass_llm_model", None) is not None:
                model = task_kwargs.get("bypass_llm_model")


        except Exception as e:
            pass

    return model