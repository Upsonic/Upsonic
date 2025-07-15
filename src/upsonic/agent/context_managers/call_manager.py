import time
from contextlib import asynccontextmanager
from upsonic.utils.printing import call_end
from upsonic.utils.llm_usage import llm_usage
from upsonic.utils.tool_usage import tool_usage


class CallHandler:
    def __init__(self, model, task, debug, memory_handler=None):
        self.model = model
        self.task = task
        self.debug = debug
        self.start_time = None
        self.end_time = None
        self.model_response = None
        self.historical_message_count = memory_handler.historical_message_count if memory_handler else 0
        
    def process_response(self, model_response):
        self.model_response = model_response
        return self.model_response


class CallManager:
    def __init__(self, model, task, debug=False):
        self.model = model
        self.task = task
        self.debug = debug
    
    @asynccontextmanager
    async def manage_call(self, memory_handler=None):
        call_handler = CallHandler(self.model, self.task, self.debug, memory_handler)
        call_handler.start_time = time.time()
        
        try:
            yield call_handler
        finally:
            call_handler.end_time = time.time()
            
            # Only call call_end if we have a model response
            if call_handler.model_response is not None:
                # Calculate usage and tool usage
                usage = llm_usage(call_handler.model_response, call_handler.historical_message_count)
                tool_usage_result = tool_usage(call_handler.model_response, call_handler.task, call_handler.historical_message_count)
                
                # Call the end logging
                call_end(
                    call_handler.model_response.output,
                    call_handler.model,
                    call_handler.task.response_format,
                    call_handler.start_time,
                    call_handler.end_time,
                    usage,
                    tool_usage_result,
                    call_handler.debug,
                    call_handler.task.price_id
                ) 