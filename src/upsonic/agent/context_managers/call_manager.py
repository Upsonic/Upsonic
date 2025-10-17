import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

# Heavy imports moved to lazy loading for faster startup
if TYPE_CHECKING:
    from upsonic.models import Model
    from upsonic.utils.printing import call_end
    from upsonic.utils.llm_usage import llm_usage
    from upsonic.utils.tool_usage import tool_usage
else:
    # Use string annotations to avoid importing heavy modules
    Model = "Model"
    call_end = "call_end"
    llm_usage = "llm_usage"
    tool_usage = "tool_usage"


class CallManager:
    def __init__(self, model: "Model", task, debug=False, show_tool_calls=True):
        """
        Initializes the CallManager.

        Args:
            model: The instantiated model object for this call.
            task: The task being executed.
            debug: Whether debug mode is enabled.
            show_tool_calls: Whether to show tool calls.
        """
        self.model = model
        self.task = task
        self.show_tool_calls = show_tool_calls
        self.debug = debug
        self.start_time = None
        self.end_time = None
        self.model_response = None
        
    def process_response(self, model_response):
        self.model_response = model_response
        return self.model_response
    
    @asynccontextmanager
    async def manage_call(self):
        self.start_time = time.time()
        
        try:
            yield self
        finally:
            self.end_time = time.time()
            
            # Only call call_end if we have a model response
            if self.model_response is not None:
                # Lazy import for heavy modules
                from upsonic.utils.llm_usage import llm_usage
                from upsonic.utils.tool_usage import tool_usage
                from upsonic.utils.printing import call_end
                
                # Calculate usage and tool usage
                usage = llm_usage(self.model_response)
                if self.show_tool_calls:
                    tool_usage_result = tool_usage(self.model_response, self.task)
                else:
                    tool_usage_result = None
                # Call the end logging
                call_end(
                    self.model_response.output,
                    self.model,
                    self.task.response_format,
                    self.start_time,
                    self.end_time,
                    usage,
                    tool_usage_result,
                    self.debug,
                    self.task.price_id
                )