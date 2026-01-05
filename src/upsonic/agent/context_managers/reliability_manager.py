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
    
    async def aprepare(self) -> None:
        """Prepare the reliability layer before the LLM call."""
        pass
    
    async def afinalize(self) -> None:
        """Finalize the reliability layer after the LLM call."""
        pass
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aprepare())
    
    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.afinalize())
    
    @asynccontextmanager
    async def manage_reliability(self):
        """
        Async context manager for reliability layer lifecycle.
        
        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        """
        await self.aprepare()
        
        try:
            yield self
        finally:
            await self.afinalize()