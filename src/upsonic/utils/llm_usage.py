def llm_usage(model_response):

        # Extract all messages from model_response
        all_messages = model_response.all_messages()
        
        # Initialize token counters
        input_tokens = 0
        output_tokens = 0
        
        # Process messages to extract actual token usage from usage data
        for message in all_messages:
            # Extract actual token counts from usage data
            if hasattr(message, 'usage') and message.usage:
                # Get actual request tokens (input tokens)
                request_tokens = getattr(message.usage, 'request_tokens', 0)
                input_tokens += request_tokens
                
                # Get actual response tokens (output tokens)
                response_tokens = getattr(message.usage, 'response_tokens', 0)
                output_tokens += response_tokens
        
        # Build usage result structure
        usage_result = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        
        return usage_result