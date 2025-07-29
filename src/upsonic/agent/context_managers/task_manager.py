from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

class TaskManager:
    def __init__(self, task, agent, *, turn_data: Optional[Dict[str, Any]] = None):
        self.task = task
        self.agent = agent
        self.model_response = None
        self.turn_data = turn_data

        
    def process_response(self, model_response):
        self.model_response = model_response

        if self.turn_data is not None and self.model_response is None:
            if hasattr(self.task, 'error_message') and self.task.error_message:
                self.turn_data['error'] = str(self.task.error_message)

        return self.model_response


    @asynccontextmanager
    async def manage_task(self):
        # Start the task
        self.task.task_start(self.agent)
        
        try:
            yield self
        finally:
            # Set task response and end the task if we have a model response
            if self.model_response is not None:
                self.task.task_response(self.model_response)
                self.task.task_end() 