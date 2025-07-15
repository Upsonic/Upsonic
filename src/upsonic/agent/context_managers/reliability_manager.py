from contextlib import asynccontextmanager
from upsonic.reliability_layer.reliability_layer import ReliabilityProcessor


class ReliabilityManager:
    def __init__(self, task, reliability_layer, model):
        self.task = task
        self.reliability_layer = reliability_layer
        self.model = model
        self.processed_task = None
        
    async def process_task(self, task):
        self.task = task
        # Process the task through the reliability layer
        processed_result = await ReliabilityProcessor.process_task(
            task, 
            self.reliability_layer,
            self.model
        )
        self.processed_task = processed_result
        return processed_result
    
    @asynccontextmanager
    async def manage_reliability(self):
        try:
            yield self
        finally:
            # The processing is now handled by the process_task method
            pass 