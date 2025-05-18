from pydantic_ai.settings import ModelSettings
from decimal import Decimal
from pydantic_ai.models.openai import OpenAIModelSettings
from pydantic_ai.models.anthropic import AnthropicModelSettings


from typing import Literal
# Define all available model names
ModelNames = Literal[
    "openai/gpt-4o",
    "openai/gpt-4.5-preview",
    "openai/o3-mini",
    "openai/gpt-4o-mini",
    "azure/gpt-4o",
    "azure/gpt-4o-mini",
    "claude/claude-3-5-sonnet",
    "claude/claude-3-7-sonnet",
    "bedrock/claude-3-5-sonnet",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-1.5-flash",
    "ollama/llama3.2",
    "ollama/llama3.1:70b",
    "ollama/llama3.1",
    "ollama/llama3.3",
    "ollama/qwen2.5",
    "deepseek/deepseek-chat",
    "openrouter/anthropic/claude-3-sonnet",
    "openrouter/meta-llama/llama-3.1-8b-instruct",
    "openrouter/google/gemini-pro",
    "openrouter/<provider>/<model>",
]



# Define model settings in a centralized dictionary for easier maintenance
MODEL_SETTINGS = {
    "openai": OpenAIModelSettings(parallel_tool_calls=False),
    "anthropic": AnthropicModelSettings(parallel_tool_calls=False),
    "openrouter": OpenAIModelSettings(parallel_tool_calls=False),
    # Add other provider settings as needed
}

# OpenAI models that don't support parallel tool calls
OPENAI_NON_PARALLEL_MODELS = {
    "o3-mini": True,
}

# Comprehensive model registry for easier maintenance and extension
MODEL_REGISTRY = {
    # OpenAI models
    "openai/gpt-4o": {
        "provider": "openai", 
        "model_name": "gpt-4o", 
        "capabilities": [],
        "pricing": {"input": 2.50, "output": 10.00},
        "required_environment_variables": ["OPENAI_API_KEY"]
    },
    "openai/gpt-4.5-preview": {
        "provider": "openai", 
        "model_name": "gpt-4.5-preview", 
        "capabilities": [],
        "pricing": {"input": 75.00, "output": 150.00},
        "required_environment_variables": ["OPENAI_API_KEY"]
    },

    "openai/o3-mini": {
        "provider": "openai", 
        "model_name": "o3-mini", 
        "capabilities": [],
        "pricing": {"input": 1.1, "output": 4.4},
        "required_environment_variables": ["OPENAI_API_KEY"]
    },
    "openai/gpt-4o-mini": {
        "provider": "openai", 
        "model_name": "gpt-4o-mini", 
        "capabilities": [],
        "pricing": {"input": 0.15, "output": 0.60},
        "required_environment_variables": ["OPENAI_API_KEY"]
    },
    
    # Azure OpenAI models
    "azure/gpt-4o": {
        "provider": "azure_openai", 
        "model_name": "gpt-4o", 
        "capabilities": [],
        "pricing": {"input": 2.50, "output": 10.00},
        "required_environment_variables": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_KEY"]
    },

    "azure/gpt-4o-mini": {
        "provider": "azure_openai", 
        "model_name": "gpt-4o-mini", 
        "capabilities": [],
        "pricing":{"input": 0.15, "output": 0.60},
        "required_environment_variables": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_KEY"]
    },
    
    # Deepseek model
    "deepseek/deepseek-chat": {
        "provider": "deepseek", 
        "model_name": "deepseek-chat", 
        "capabilities": [],
        "pricing": {"input": 0.27, "output": 1.10},
        "required_environment_variables": ["DEEPSEEK_API_KEY"]
    },

    "gemini/gemini-2.0-flash": {
        "provider": "gemini", 
        "model_name": "gemini-2.0-flash", 
        "capabilities": [],
        "pricing": {"input": 0.10, "output": 0.40},
        "required_environment_variables": ["GOOGLE_GLA_API_KEY"]
    },

    "gemini/gemini-1.5-pro": {
        "provider": "gemini", 
        "model_name": "gemini-1.5-pro", 
        "capabilities": [],
        "pricing": {"input": 1.25, "output": 5.00},
        "required_environment_variables": ["GOOGLE_GLA_API_KEY"]
    },

    "gemini/gemini-1.5-flash": {
        "provider": "gemini", 
        "model_name": "gemini-1.5-flash", 
        "capabilities": [],
        "pricing": {"input": 0.075, "output": 0.30},
        "required_environment_variables": ["GOOGLE_GLA_API_KEY"]
    },
    
    "ollama/llama3.2": {
        "provider": "ollama", 
        "model_name": "llama3.2", 
        "capabilities": [],
        "pricing": {"input": 0.0, "output": 0.0},
        "required_environment_variables": []
    },

    "ollama/llama3.1:70b": {
        "provider": "ollama", 
        "model_name": "llama3.1:70b", 
        "capabilities": [],
        "pricing": {"input": 0.0, "output": 0.0},
        "required_environment_variables": []
    },

    "ollama/llama3.1": {
        "provider": "ollama", 
        "model_name": "llama3.1", 
        "capabilities": [],
        "pricing": {"input": 0.0, "output": 0.0},
        "required_environment_variables": []
    },

    "ollama/llama3.3": {
        "provider": "ollama", 
        "model_name": "llama3.3", 
        "capabilities": [],
        "pricing": {"input": 0.0, "output": 0.0},
        "required_environment_variables": []
    },

    "ollama/qwen2.5": {
        "provider": "ollama", 
        "model_name": "qwen2.5", 
        "capabilities": [],
        "pricing": {"input": 0.0, "output": 0.0},
        "required_environment_variables": []
    },

    # Anthropic models
    "claude/claude-3-5-sonnet": {
        "provider": "anthropic", 
        "model_name": "claude-3-5-sonnet-latest", 
        "capabilities": ["computer_use"],
        "pricing": {"input": 3.00, "output": 15.00},
        "required_environment_variables": ["ANTHROPIC_API_KEY"]
    },
    "claude/claude-3-7-sonnet": {
        "provider": "anthropic", 
        "model_name": "claude-3-7-sonnet-latest", 
        "capabilities": ["computer_use"],
        "pricing": {"input": 3.00, "output": 15.00},
        "required_environment_variables": ["ANTHROPIC_API_KEY"]
    },

    
    # Bedrock Anthropic models
    "bedrock/claude-3-5-sonnet": {
        "provider": "bedrock_anthropic", 
        "model_name": "us.anthropic.claude-3-5-sonnet-20240620-v1:0", 
        "capabilities": ["computer_use"],
        "pricing": {"input": 3.00, "output": 15.00},
        "required_environment_variables": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]
    },

    # OpenRouter models
    "openrouter/anthropic/claude-3-sonnet": {
        "provider": "openrouter",
        "model_name": "anthropic/claude-3-sonnet",
        "capabilities": [],
        "pricing": {"input": 0.0, "output": 0.0},
        "required_environment_variables": ["OPENROUTER_API_KEY"]
    },
    "openrouter/meta-llama/llama-3.1-8b-instruct": {
        "provider": "openrouter",
        "model_name": "meta-llama/llama-3.1-8b-instruct",
        "capabilities": [],
        "pricing": {"input": 0.0, "output": 0.0},
        "required_environment_variables": ["OPENROUTER_API_KEY"]
    },
    "openrouter/google/gemini-pro": {
        "provider": "openrouter",
        "model_name": "google/gemini-pro",
        "capabilities": [],
        "pricing": {"input": 0.0, "output": 0.0},
        "required_environment_variables": ["OPENROUTER_API_KEY"]
    },
}

# Helper functions for model registry access

def get_model_registry_entry(llm_model: str):
    """Get model registry entry or return None if not found."""
    if llm_model in MODEL_REGISTRY:
        return MODEL_REGISTRY[llm_model]
    
    # Handle dynamic OpenRouter models
    if llm_model.startswith("openrouter/"):
        # Extract the model name after openrouter/
        model_name = llm_model.split("openrouter/", 1)[1]
        return {
            "provider": "openrouter",
            "model_name": model_name,  # Use the full model name as provided
            "capabilities": [],
            "pricing": {"input": 0.0, "output": 0.0},
            "required_environment_variables": ["OPENROUTER_API_KEY"]
        }
    
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
    
    # Get model info from registry
    model_info = get_model_registry_entry(llm_model)
    if not model_info:
        return None

    # Filter out ComputerUse tools if model doesn't support computer_use capability
    filtered_tools = []
    for tool in tools:
        if "ComputerUse" in tool and not has_capability(llm_model, "computer_use"):
            continue
        filtered_tools.append(tool)
    
    # If no tools remain after filtering, return None
    if not filtered_tools:
        return None

    # Special handling for OpenAI models that don't support parallel tool calls
    if model_info["provider"] == "openai" and model_info["model_name"] in OPENAI_NON_PARALLEL_MODELS:
        return OpenAIModelSettings()
    
    # For all other models, return provider settings
    provider = model_info["provider"]
    if provider in MODEL_SETTINGS:
        return MODEL_SETTINGS[provider]
    
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
    
    # Convert token counts to millions for pricing calculation using Decimal
    input_tokens_millions = Decimal(str(input_tokens)) / Decimal('1000000')
    output_tokens_millions = Decimal(str(output_tokens)) / Decimal('1000000')
    
    input_cost = Decimal(str(pricing["input"])) * input_tokens_millions
    output_cost = Decimal(str(pricing["output"])) * output_tokens_millions
    total = input_cost + output_cost

    # to 4 decimal places
    return f"~${float(round(total, 4))}"

def has_capability(llm_model: str, capability: str):
    """Check if a model has a specific capability."""
    model_info = get_model_registry_entry(llm_model)
    if model_info and "capabilities" in model_info:
        return capability in model_info["capabilities"]
    return False
