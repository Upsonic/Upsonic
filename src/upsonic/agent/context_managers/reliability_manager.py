from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

# Heavy imports moved to lazy loading for faster startup
if TYPE_CHECKING:
    from upsonic.reliability_layer.reliability_layer import ReliabilityProcessor
    from upsonic.models import Model
else:
    # Use string annotations to avoid importing heavy modules
    ReliabilityProcessor = "ReliabilityProcessor"
    Model = "Model"

class ReliabilityManager:
    def __init__(self, task, reliability_layer, model: "Model"):
        """
        Initializes the ReliabilityManager.

        Args:
            task: The task being executed.
            reliability_layer: The configured reliability layer.
            model: The instantiated model object.
        """
        self.task = task
        self.reliability_layer = reliability_layer
        self.model = model
        self.processed_task = None
        
    async def process_task(self, task):
        self.task = task
        # Lazy import for heavy modules
        from upsonic.reliability_layer.reliability_layer import ReliabilityProcessor
        
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
            pass 