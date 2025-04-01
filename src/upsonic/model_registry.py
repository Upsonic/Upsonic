from pydantic_ai.settings import ModelSettings

# Define model settings in a centralized dictionary for easier maintenance
MODEL_SETTINGS = {
    "openai": ModelSettings(parallel_tool_calls=False),
    "anthropic": ModelSettings(parallel_tool_calls=False),
    # Add other provider settings as needed
}

# Comprehensive model registry for easier maintenance and extension
MODEL_REGISTRY = {
    # OpenAI models
    "openai/gpt-4o": {
        "provider": "openai", 
        "model_name": "gpt-4o", 
        "api_key": "OPENAI_API_KEY", 
        "capabilities": [],
        "pricing": {"input": 0.0000025, "output": 0.00001}
    },
    "gpt-4o": {
        "provider": "openai", 
        "model_name": "gpt-4o", 
        "api_key": "OPENAI_API_KEY", 
        "capabilities": [],
        "pricing": {"input": 0.0000025, "output": 0.00001}
    },
    "openai/o3-mini": {
        "provider": "openai", 
        "model_name": "o3-mini", 
        "api_key": "OPENAI_API_KEY", 
        "capabilities": [],
        "pricing": {"input": 0.0000011, "output": 0.00000055}
    },
    "openai/gpt-4o-mini": {
        "provider": "openai", 
        "model_name": "gpt-4o-mini", 
        "api_key": "OPENAI_API_KEY", 
        "capabilities": [],
        "pricing": {"input": 0.00000015, "output": 0.00000008}
    },
    
    # Azure OpenAI models
    "azure/gpt-4o": {
        "provider": "azure_openai", 
        "model_name": "gpt-4o", 
        "capabilities": [],
        "pricing": {"input": 0.0000025, "output": 0.00001}
    },
    "gpt-4o-azure": {
        "provider": "azure_openai", 
        "model_name": "gpt-4o", 
        "capabilities": [],
        "pricing": {"input": 0.0000025, "output": 0.00001}
    },
    "azure/gpt-4o-mini": {
        "provider": "azure_openai", 
        "model_name": "gpt-4o-mini", 
        "capabilities": [],
        "pricing": {"input": 0.00000015, "output": 0.00000008}
    },
    
    # Deepseek model
    "deepseek/deepseek-chat": {
        "provider": "deepseek", 
        "model_name": "deepseek-chat", 
        "capabilities": [],
        "pricing": {"input": 0.00000027, "output": 0.00000028}
    },
    
    # Anthropic models
    "claude/claude-3-5-sonnet": {
        "provider": "anthropic", 
        "model_name": "claude-3-5-sonnet-latest", 
        "capabilities": ["computer_use"],
        "pricing": {"input": 0.000003, "output": 0.000015}
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic", 
        "model_name": "claude-3-5-sonnet-latest", 
        "capabilities": ["computer_use"],
        "pricing": {"input": 0.000003, "output": 0.000015}
    },
    
    # Bedrock Anthropic models
    "bedrock/claude-3-5-sonnet": {
        "provider": "bedrock_anthropic", 
        "model_name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
        "capabilities": ["computer_use"],
        "pricing": {"input": 0.000003, "output": 0.000015}
    },
    "claude-3-5-sonnet-aws": {
        "provider": "bedrock_anthropic", 
        "model_name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
        "capabilities": ["computer_use"],
        "pricing": {"input": 0.000003, "output": 0.000015}
    },
}

# Helper functions for model registry access

def get_model_registry_entry(llm_model: str):
    """Get model registry entry or return None if not found."""
    if llm_model in MODEL_REGISTRY:
        return MODEL_REGISTRY[llm_model]
    
    # Try case-insensitive match as fallback
    llm_model_lower = llm_model.lower()
    for model_id, details in MODEL_REGISTRY.items():
        if model_id.lower() == llm_model_lower:
            return details
    
    print(f"Warning: Model '{llm_model}' not found in registry")
    return None

def get_model_family(provider_type: str):
    """Get all models of a specific provider type."""
    return [model for model, info in MODEL_REGISTRY.items() if info["provider"] == provider_type]

# Pre-computed model family groupings
OPENAI_MODELS = [model for model, info in MODEL_REGISTRY.items() if info["provider"] in ["openai", "azure_openai", "deepseek"]]
ANTHROPIC_MODELS = [model for model, info in MODEL_REGISTRY.items() if info["provider"] in ["anthropic", "bedrock_anthropic"]]

def get_model_settings(llm_model: str, tools=None):
    """Get the appropriate model settings based on the model type."""
    # If no tools are provided, no model settings are needed
    if not tools:
        return None
    
    # Convert model name to lowercase for case-insensitive matching    
    llm_model_lower = llm_model.lower()
    
    # Check if the model belongs to the OpenAI family
    for model_id in OPENAI_MODELS:
        if model_id.lower() in llm_model_lower:
            return MODEL_SETTINGS["openai"]
    
    # Check if the model belongs to the Anthropic family
    for model_id in ANTHROPIC_MODELS:
        if model_id.lower() in llm_model_lower:
            return MODEL_SETTINGS["anthropic"]
    
    # Log when no settings match is found
    print(f"Warning: No model settings found for {llm_model}")
    return None

def get_pricing(llm_model: str):
    """Get pricing information for a model."""
    model_info = get_model_registry_entry(llm_model)
    if model_info and "pricing" in model_info:
        return model_info["pricing"]
    return None

def get_estimated_cost(input_tokens: int, output_tokens: int, llm_model: str):
    """Calculate estimated cost for token usage with a specific model."""
    pricing = get_pricing(llm_model)
    if not pricing:
        return "Unknown"
    
    input_cost = pricing["input"] * input_tokens
    output_cost = pricing["output"] * output_tokens
    total = input_cost + output_cost

    # to 4 decimal places
    return f"~${round(total, 4)}"

def has_capability(llm_model: str, capability: str):
    """Check if a model has a specific capability."""
    model_info = get_model_registry_entry(llm_model)
    if model_info and "capabilities" in model_info:
        return capability in model_info["capabilities"]
    return False
