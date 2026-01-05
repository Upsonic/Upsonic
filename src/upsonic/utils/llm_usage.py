from typing import Dict, Any


def llm_usage(model_response, cumulative: bool = False) -> Dict[str, int]:
    """
    Extract usage information from a model response.
    
    This function extracts token usage information from ModelResponse objects,
    supporting both current and deprecated field names for backward compatibility.
    
    Args:
        model_response: A RunResult or StreamRunResult object containing messages.
        cumulative: If False (default), only count tokens from the last run (task-specific).
                   If True, count tokens from all runs (agent-cumulative).
    
    Returns:
        Dict with 'input_tokens' and 'output_tokens' counts.
    """
    # Use new_messages() for task-specific metrics, all_messages() for cumulative
    if cumulative:
        messages = model_response.all_messages()
    else:
        messages = model_response.new_messages()
    
    input_tokens = 0
    output_tokens = 0
    
    for message in messages:
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