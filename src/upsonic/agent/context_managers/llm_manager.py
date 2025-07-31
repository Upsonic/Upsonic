import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class LLMManager:
    def __init__(self, default_model, requested_model: Optional[str] = None):
        self.default_model = default_model
        self.requested_model = requested_model
        self.selected_model = None
        
    def _model_set(self, model):
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
        
    def get_model(self):
        return self.selected_model
    
    @asynccontextmanager
    async def manage_llm(self):
        # LLM Selection logic
        if self.requested_model is None:
            self.selected_model = self._model_set(self.default_model)
        else:
            self.selected_model = self._model_set(self.requested_model)
        
        try:
            yield self
        finally:
            # Any cleanup logic for LLM resources can go here
            pass 