def llm_usage(model_response):
    """
    Extract usage information from a model response.
    
    This function extracts token usage information from ModelResponse objects,
    supporting both current and deprecated field names for backward compatibility.
    """
    all_messages = model_response.all_messages()
    
    input_tokens = 0
    output_tokens = 0
    
    for message in all_messages:
        if hasattr(message, 'usage') and message.usage:
            usage_obj = message.usage
            
            if hasattr(usage_obj, 'input_tokens'):
                input_tokens += usage_obj.input_tokens
            elif hasattr(usage_obj, 'request_tokens'):
                input_tokens += usage_obj.request_tokens
            
            if hasattr(usage_obj, 'output_tokens'):
                output_tokens += usage_obj.output_tokens
            elif hasattr(usage_obj, 'response_tokens'):
                output_tokens += usage_obj.response_tokens
    
    usage_result = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
    
    return usage_result