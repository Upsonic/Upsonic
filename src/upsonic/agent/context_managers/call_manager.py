import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from upsonic.storage.session.llm import LLMUsageStats, LLMToolCall
from upsonic.utils.printing import call_end
from upsonic.utils.llm_usage import llm_usage
from upsonic.utils.tool_usage import tool_usage


class CallManager:
    def __init__(self, model, task, debug=False, *, turn_data: Optional[Dict[str, Any]] = None):
        self.model = model
        self.task = task
        self.debug = debug
        self.start_time = None
        self.end_time = None
        self.model_response = None
        self.historical_message_count = 0
        self.turn_data = turn_data
        
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
                
                if self.turn_data is not None:
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    self.turn_data['usage_stats'] = LLMUsageStats(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                    
                    self.turn_data['content'] = self.model_response.output

                    if tool_usage_result:
                        parsed_tool_calls = [
                            LLMToolCall(
                                tool_name=tc.get("tool_name", "unknown_tool"),
                                arguments=tc.get("params", {}),
                                tool_output=tc.get("tool_result")
                            ) 
                            for tc in tool_usage_result
                        ]
                        self.turn_data['tool_calls'] = parsed_tool_calls

                    if self.task.total_cost is not None:
                        self.turn_data['cost'] = self.task.total_cost

                
                # Call the end logging
                call_end(
                    self.model_response.output,
                    self.model,
                    self.task.response_format,
                    self.start_time,
                    self.end_time,
                    usage,
                    list(tool_usage_result.values()),
                    self.debug,
                    self.task.price_id
                ) 