from contextlib import asynccontextmanager


class TaskHandler:
    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
        self.model_response = None
        
    def process_response(self, model_response):
        self.model_response = model_response
        return self.model_response


class TaskManager:
    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
    
    @asynccontextmanager
    async def manage_task(self):
        task_handler = TaskHandler(self.task, self.agent)
        
        # Start the task
        self.task.task_start(self.agent)
        
        try:
            yield task_handler
        finally:
            # Set task response and end the task if we have a model response
            if task_handler.model_response is not None:
                self.task.task_response(task_handler.model_response)
                self.task.task_end() 