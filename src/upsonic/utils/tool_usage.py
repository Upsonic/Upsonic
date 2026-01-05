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

        # If no tool calls found in messages, check AgentRunOutput.tools as fallback
        # This handles cases where ToolCallPart is not in messages but ToolExecution objects exist
        if not tool_calls_map and hasattr(model_response, 'tools') and model_response.tools:
            for tool_exec in model_response.tools:
                if tool_exec.tool_call_id and tool_exec.tool_name:
                    tool_calls_map[tool_exec.tool_call_id] = {
                        "tool_name": tool_exec.tool_name,
                        "params": tool_exec.tool_args or {},
                        "tool_result": tool_exec.result
                    }
        # Also try to match tool returns from messages with tool executions
        elif tool_calls_map and hasattr(model_response, 'tools') and model_response.tools:
            # Match tool returns from messages with tool executions to get complete info
            for tool_exec in model_response.tools:
                if tool_exec.tool_call_id in tool_calls_map:
                    # Update with tool execution info if available
                    if tool_exec.tool_args:
                        tool_calls_map[tool_exec.tool_call_id]["params"] = tool_exec.tool_args
                    if tool_exec.result and not tool_calls_map[tool_exec.tool_call_id].get("tool_result"):
                        tool_calls_map[tool_exec.tool_call_id]["tool_result"] = tool_exec.result
                elif tool_exec.tool_call_id and tool_exec.tool_name:
                    # Tool execution exists but not in messages - add it
                    tool_calls_map[tool_exec.tool_call_id] = {
                        "tool_name": tool_exec.tool_name,
                        "params": tool_exec.tool_args or {},
                        "tool_result": tool_exec.result
                    }

        # Convert to list format
        tool_usage_value = list(tool_calls_map.values())
        # Store tool calls in the task
        for tool_call in tool_usage_value:
            task.add_tool_call(tool_call)

        return tool_usage_value
