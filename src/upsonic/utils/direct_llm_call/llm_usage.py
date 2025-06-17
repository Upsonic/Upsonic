def llm_usage(model_response, historical_message_count=0):

        # Extract all messages from model_response
        all_messages = model_response.all_messages()
        
        # Only process messages from the current interaction (skip historical messages)
        current_interaction_messages = all_messages[historical_message_count:]
        
        # Initialize token counters
        input_tokens = 0
        output_tokens = 0
        
        # Process messages to extract actual token usage
        for message in current_interaction_messages:
            # For request messages, estimate tokens from content
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'content') and isinstance(part.content, str):
                        # Simple estimation for current request tokens
                        part_tokens = int(len(part.content.split()) * 1.33)
                        input_tokens += part_tokens
            
            # For response messages, only count response tokens (not the inflated request_tokens)
            if hasattr(message, 'usage') and message.usage:
                resp_tokens = getattr(message.usage, 'response_tokens', 0)
                output_tokens += resp_tokens
        
        # Build usage result structure
        usage_result = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        
        return usage_result