from contextlib import asynccontextmanager

class TaskManager:
    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
        self.model_response = None

        
    def process_response(self, model_response):
        self.model_response = model_response

        return self.model_response


    @asynccontextmanager
    async def manage_task(self):
        # Task start/end is now managed by pipeline steps (InitializationStep and FinalizationStep)
        # This context manager now only handles task response processing

        try:
            yield self
        finally:
            # Set task response if we have a model response
            if self.model_response is not None:
                self.task.task_response(self.model_response) 