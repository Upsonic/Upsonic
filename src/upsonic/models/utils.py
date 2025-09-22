"""
Utility functions for model operations and cost estimation.
"""

from typing import Any, Dict, Optional
from decimal import Decimal


def get_estimated_cost(
    input_tokens: int, 
    output_tokens: int, 
    model_provider: Optional[Any] = None
) -> str:
    """
    Estimate the cost of a model interaction based on token usage.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_provider: The model provider instance (optional)
        
    Returns:
        Estimated cost as a formatted string (e.g., "~0.0123")
    """
    # Default pricing for common models (per 1K tokens)
    # These are approximate rates and should be updated regularly
    default_pricing = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }
    
    # Try to get model name from provider
    model_name = "gpt-4o"  # default
    if model_provider and hasattr(model_provider, 'model_name'):
        model_name = model_provider.model_name
    
    # Get pricing for the model
    pricing = default_pricing.get(model_name, default_pricing["gpt-4o"])
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    # Format as string with ~ prefix
    return f"~{total_cost:.4f}"


def format_token_usage(usage: Dict[str, int]) -> str:
    """
    Format token usage information for display.
    
    Args:
        usage: Dictionary containing token usage information
        
    Returns:
        Formatted string
    """
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    total_tokens = input_tokens + output_tokens
    
    return f"Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {total_tokens:,}"


def validate_model_settings(settings: Dict[str, Any]) -> bool:
    """
    Validate model settings for common issues.
    
    Args:
        settings: Dictionary of model settings
        
    Returns:
        True if settings are valid, False otherwise
    """
    # Check temperature range
    if 'temperature' in settings:
        temp = settings['temperature']
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            return False
    
    # Check max_tokens
    if 'max_tokens' in settings:
        max_tokens = settings['max_tokens']
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            return False
    
    # Check top_p range
    if 'top_p' in settings:
        top_p = settings['top_p']
        if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
            return False
    
    return True
