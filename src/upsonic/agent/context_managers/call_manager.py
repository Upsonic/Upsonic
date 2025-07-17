import time
from contextlib import asynccontextmanager
from upsonic.utils.printing import call_end
from upsonic.utils.llm_usage import llm_usage
from upsonic.utils.tool_usage import tool_usage


class CallManager:
    def __init__(self, model, task, debug=False):
        self.model = model
        self.task = task
        self.debug = debug
        self.start_time = None
        self.end_time = None
        self.model_response = None
        self.historical_message_count = 0
        
    def process_response(self, model_response):
        self.model_response = model_response
        return self.model_response
    
    @asynccontextmanager
    async def manage_call(self, memory_handler=None):
        self.start_time = time.time()
        self.historical_message_count = memory_handler.historical_message_count if memory_handler else 0
        
        try:
            yield self
        finally:
            self.end_time = time.time()
            
            # Only call call_end if we have a model response
            if self.model_response is not None:
                # Calculate usage and tool usage
                usage = llm_usage(self.model_response, self.historical_message_count)
                tool_usage_result = tool_usage(self.model_response, self.task, self.historical_message_count)
                
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