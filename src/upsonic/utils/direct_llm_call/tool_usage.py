def tool_usage(model_response, task):


        # Extract tool calls from model_response.all_messages()
        tool_usage_value = []
        all_messages = model_response.all_messages()
        
        # Process messages to extract tool calls and their results
        tool_calls_map = {}  # Map tool_call_id to tool call info
        
        for message in all_messages:
            if hasattr(message, 'parts'):
                for part in message.parts:
                    # Check if this is a tool call
                    if hasattr(part, 'tool_name') and hasattr(part, 'tool_call_id') and hasattr(part, 'args'):
                        tool_calls_map[part.tool_call_id] = {
                            "tool_name": part.tool_name,
                            "params": part.args,
                            "tool_result": None  # Will be filled when we find the return
                        }
                    # Check if this is a tool return
                    elif hasattr(part, 'tool_call_id') and hasattr(part, 'content') and part.tool_call_id in tool_calls_map:
                        tool_calls_map[part.tool_call_id]["tool_result"] = part.content
        
        # Convert to list format
        tool_usage_value = list(tool_calls_map.values())
        
        # Store tool calls in the task
        for tool_call in tool_usage_value:
            task.add_tool_call(tool_call)


        return tool_usage_value
