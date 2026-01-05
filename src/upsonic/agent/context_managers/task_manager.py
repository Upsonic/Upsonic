from contextlib import asynccontextmanager

class TaskManager:
    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
        self.model_response = None

        
    def process_response(self, model_response):
        self.model_response = model_response

        return self.model_response

    async def aprepare(self) -> None:
        """Prepare the task before the LLM call."""
        # Task start/end is now managed by pipeline steps (InitializationStep and FinalizationStep)
        pass
    
    async def afinalize(self) -> None:
        """Finalize the task after the LLM call."""
        # Set task response if we have a model response
        if self.model_response is not None:
            self.task.task_response(self.model_response)
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aprepare())
    
    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.afinalize())

    @asynccontextmanager
    async def manage_task(self):
        """
        Async context manager for task lifecycle.
        
        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        """
        await self.aprepare()

        try:
            yield self
        finally:
            await self.afinalize()