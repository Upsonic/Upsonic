from __future__ import annotations
from typing import Dict, Type, List, Any, Optional, Union
import os

from .base import EmbeddingProvider, EmbeddingConfig
from ..utils.package.exception import ConfigurationError


_PROVIDER_REGISTRY: Dict[str, Type[EmbeddingProvider]] = {}
_PROVIDER_CONFIGS: Dict[str, Type[EmbeddingConfig]] = {}


def register_provider(name: str, provider_class: Type[EmbeddingProvider], config_class: Type[EmbeddingConfig]):
    """Register an embedding provider."""
    _PROVIDER_REGISTRY[name] = provider_class
    _PROVIDER_CONFIGS[name] = config_class


def _lazy_import_providers():
    """Lazy import all providers to populate registry."""
    global _PROVIDER_REGISTRY, _PROVIDER_CONFIGS
    
    if _PROVIDER_REGISTRY:
        return
    
    try:
        from .openai_provider import OpenAIEmbedding, OpenAIEmbeddingConfig
        register_provider("openai", OpenAIEmbedding, OpenAIEmbeddingConfig)
    except ImportError:
        pass
    
    try:
        from .azure_openai_provider import AzureOpenAIEmbedding, AzureOpenAIEmbeddingConfig
        register_provider("azure_openai", AzureOpenAIEmbedding, AzureOpenAIEmbeddingConfig)
        register_provider("azure", AzureOpenAIEmbedding, AzureOpenAIEmbeddingConfig)
    except ImportError:
        pass
    
    try:
        from .bedrock_provider import BedrockEmbedding, BedrockEmbeddingConfig
        register_provider("bedrock", BedrockEmbedding, BedrockEmbeddingConfig)
        register_provider("aws", BedrockEmbedding, BedrockEmbeddingConfig)
    except ImportError:
        pass
    
    try:
        from .huggingface_provider import HuggingFaceEmbedding, HuggingFaceEmbeddingConfig
        register_provider("huggingface", HuggingFaceEmbedding, HuggingFaceEmbeddingConfig)
        register_provider("hf", HuggingFaceEmbedding, HuggingFaceEmbeddingConfig)
    except ImportError:
        pass
    
    try:
        from .fastembed_provider import FastEmbedProvider, FastEmbedConfig
        register_provider("fastembed", FastEmbedProvider, FastEmbedConfig)
        register_provider("qdrant", FastEmbedProvider, FastEmbedConfig)
    except ImportError:
        pass
    
    try:
        from .ollama_provider import OllamaEmbedding, OllamaEmbeddingConfig
        register_provider("ollama", OllamaEmbedding, OllamaEmbeddingConfig)
    except ImportError:
        pass
    
    try:
        from .gemini_provider import GeminiEmbedding, GeminiEmbeddingConfig
        register_provider("gemini", GeminiEmbedding, GeminiEmbeddingConfig)
        register_provider("google", GeminiEmbedding, GeminiEmbeddingConfig)
    except ImportError:
        pass



def list_available_providers() -> List[str]:
    """
    List all available embedding providers.
    
    Returns:
        List of provider names that can be used with create_embedding_provider()
    """
    _lazy_import_providers()
    return list(_PROVIDER_REGISTRY.keys())


def get_provider_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about all available providers.
    
    Returns:
        Dictionary with provider information including dependencies and features
    """
    _lazy_import_providers()
    
    provider_info = {
        "openai": {
            "description": "OpenAI embedding models (GPT, Ada)",
            "dependencies": ["openai"],
            "features": ["API", "Rate limiting", "Cost tracking"],
            "models": ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        },
        "azure_openai": {
            "description": "Azure OpenAI embedding service",
            "dependencies": ["openai", "azure-identity"],
            "features": ["Enterprise auth", "Managed identity", "Private endpoints"],
            "models": ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        },
        "bedrock": {
            "description": "AWS Bedrock embedding models",
            "dependencies": ["boto3"],
            "features": ["IAM integration", "Multiple models", "Enterprise security"],
            "models": ["amazon.titan-embed-text-v1", "cohere.embed-english-v3"]
        },
        "huggingface": {
            "description": "HuggingFace transformers and API",
            "dependencies": ["transformers", "torch", "requests"],
            "features": ["Local models", "API access", "GPU support", "Quantization"],
            "models": ["sentence-transformers/*", "Custom models"]
        },
        "fastembed": {
            "description": "FastEmbed by Qdrant (ONNX optimized)",
            "dependencies": ["fastembed"],
            "features": ["Fast inference", "Minimal dependencies", "ONNX runtime"],
            "models": ["BAAI/bge-*", "sentence-transformers/*"]
        },
        "ollama": {
            "description": "Local Ollama embedding models",
            "dependencies": ["aiohttp", "requests"],
            "features": ["Local deployment", "Model management", "No API costs"],
            "models": ["nomic-embed-text", "mxbai-embed-large"]
        },
        "gemini": {
            "description": "Google Gemini embedding models",
            "dependencies": ["google-genai"],
            "features": ["Content safety", "Multi-modal", "Enterprise features", "Caching", "Batch processing", "Task-specific optimization"],
            "models": ["gemini-embedding-001", "text-embedding-005", "text-multilingual-embedding-002"]
        },
    }
    
    available_info = {}
    for provider in list_available_providers():
        base_provider = provider.split("_")[0] if "_" in provider else provider
        if base_provider in provider_info:
            available_info[provider] = provider_info[base_provider]
    
    return available_info


def create_embedding_provider(
    provider: str,
    config: Optional[Union[EmbeddingConfig, Dict[str, Any]]] = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Create an embedding provider using the factory pattern.
    
    This function provides a unified interface for creating any embedding provider
    with automatic configuration and sensible defaults.
    
    Args:
        provider: Provider name (e.g., "openai", "huggingface", "ollama")
        config: Provider configuration object or dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured embedding provider instance
        
    Examples:
        # Simple provider creation
        embedding = create_embedding_provider("openai")
        
        # With specific model
        embedding = create_embedding_provider(
            "openai", 
            model_name="text-embedding-3-large"
        )
        
        # With configuration object
        from upsonic.embeddings import OpenAIEmbeddingConfig
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            batch_size=100
        )
        embedding = create_embedding_provider("openai", config=config)
        
        # With dictionary configuration
        embedding = create_embedding_provider(
            "huggingface",
            config={
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "device": "cuda"
            }
        )
    """
    _lazy_import_providers()
    
    provider = provider.lower().replace("-", "_")
    
    if provider not in _PROVIDER_REGISTRY:
        available = ", ".join(list_available_providers())
        raise ConfigurationError(
            f"Unknown provider '{provider}'. Available providers: {available}",
            error_code="UNKNOWN_PROVIDER"
        )
    
    provider_class = _PROVIDER_REGISTRY[provider]
    config_class = _PROVIDER_CONFIGS[provider]
    
    if config is None:
        config = config_class(**kwargs)
    elif isinstance(config, dict):
        merged_config = {**config, **kwargs}
        config = config_class(**merged_config)
    elif kwargs:
        print(f"Warning: Both config object and kwargs provided. Using config object, ignoring kwargs: {list(kwargs.keys())}")
    
    return provider_class(config=config)


def create_openai_embedding(**kwargs) -> EmbeddingProvider:
    """Create OpenAI embedding provider with defaults."""
    return create_embedding_provider("openai", **kwargs)


def create_azure_openai_embedding(**kwargs) -> EmbeddingProvider:
    """Create Azure OpenAI embedding provider with defaults."""
    return create_embedding_provider("azure_openai", **kwargs)


def create_bedrock_embedding(**kwargs) -> EmbeddingProvider:
    """Create AWS Bedrock embedding provider with defaults."""
    return create_embedding_provider("bedrock", **kwargs)


def create_huggingface_embedding(**kwargs) -> EmbeddingProvider:
    """Create HuggingFace embedding provider with defaults."""
    return create_embedding_provider("huggingface", **kwargs)


def create_fastembed_provider(**kwargs) -> EmbeddingProvider:
    """Create FastEmbed provider with defaults."""
    return create_embedding_provider("fastembed", **kwargs)


def create_ollama_embedding(**kwargs) -> EmbeddingProvider:
    """Create Ollama embedding provider with defaults."""
    return create_embedding_provider("ollama", **kwargs)


def create_gemini_embedding(**kwargs) -> EmbeddingProvider:
    """Create Gemini embedding provider with defaults."""
    return create_embedding_provider("gemini", **kwargs)


def create_gemini_vertex_embedding(**kwargs) -> EmbeddingProvider:
    """Create Gemini embedding provider with Vertex AI."""
    return create_embedding_provider("gemini", use_vertex_ai=True, **kwargs)



def create_best_available_embedding(
    use_case: str = "general",
    preference: str = "balanced",
    **kwargs
) -> EmbeddingProvider:
    """
    Create the best available embedding provider based on use case and preference.
    
    Args:
        use_case: Use case type ("general", "enterprise", "local", "cost_effective")
        preference: Performance preference ("fast", "quality", "balanced", "cheap")
        **kwargs: Additional configuration parameters
        
    Returns:
        Best available embedding provider for the specified criteria
    """
    _lazy_import_providers()
    available = list_available_providers()
    
    preferences = {
        "enterprise": {
            "fast": ["azure_openai", "bedrock", "openai", "gemini"],
            "quality": ["openai", "azure_openai", "gemini", "bedrock"],
            "balanced": ["azure_openai", "openai", "bedrock", "gemini"],
            "cheap": ["bedrock", "azure_openai", "huggingface", "fastembed"]
        },
        "local": {
            "fast": ["fastembed", "ollama", "huggingface"],
            "quality": ["huggingface", "ollama", "fastembed"],
            "balanced": ["ollama", "huggingface", "fastembed"],
            "cheap": ["fastembed", "ollama", "huggingface"]
        },
        "cost_effective": {
            "fast": ["fastembed", "ollama", "huggingface", "gemini"],
            "quality": ["huggingface", "gemini", "ollama", "fastembed"],
            "balanced": ["ollama", "fastembed", "gemini", "huggingface"],
            "cheap": ["fastembed", "ollama", "huggingface"]
        },
        "general": {
            "fast": ["openai", "fastembed", "azure_openai", "ollama"],
            "quality": ["openai", "huggingface", "gemini", "azure_openai"],
            "balanced": ["openai", "ollama", "fastembed", "huggingface"],
            "cheap": ["fastembed", "ollama", "gemini", "huggingface"]
        }
    }
    
    preference_order = preferences.get(use_case, preferences["general"]).get(preference, preferences["general"]["balanced"])
    
    for provider in preference_order:
        if provider in available:
            print(f"Selected {provider} embedding provider for {use_case} use case with {preference} preference")
            
            use_case_defaults = {
                "enterprise": {"batch_size": 100, "cache_embeddings": True},
                "local": {"batch_size": 32, "show_progress": True},
                "cost_effective": {"batch_size": 50, "cache_embeddings": True},
                "general": {"batch_size": 50}
            }
            
            defaults = use_case_defaults.get(use_case, {})
            merged_kwargs = {**defaults, **kwargs}
            
            return create_embedding_provider(provider, **merged_kwargs)
    
    if available:
        provider = available[0]
        print(f"No preferred provider available, using {provider}")
        return create_embedding_provider(provider, **kwargs)
    
    raise ConfigurationError(
        "No embedding providers available. Please install at least one provider package.",
        error_code="NO_PROVIDERS_AVAILABLE"
    )


def auto_detect_best_embedding(**kwargs) -> EmbeddingProvider:
    """
    Automatically detect and create the best embedding provider based on environment.
    
    This function checks for available API keys, local models, and hardware to
    automatically select the most appropriate embedding provider.
    
    Returns:
        Best automatically detected embedding provider
    """
    _lazy_import_providers()
    available = list_available_providers()
    
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_azure = bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
    has_gemini = bool(os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY"))
    has_aws = bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"))
    
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass
    
    if has_openai and "openai" in available:

        return create_embedding_provider("openai", **kwargs)
    
    elif has_azure and "azure_openai" in available:

        return create_embedding_provider("azure_openai", **kwargs)
    
    elif has_gemini and "gemini" in available:
        return create_embedding_provider("gemini", **kwargs)
    
    elif has_aws and "bedrock" in available:
        return create_embedding_provider("bedrock", **kwargs)
    
    elif "fastembed" in available:
        config = {"enable_gpu": has_gpu} if has_gpu else {}
        return create_embedding_provider("fastembed", config=config, **kwargs)
    
    elif "ollama" in available:
        return create_embedding_provider("ollama", **kwargs)
    
    elif "huggingface" in available:
        config = {"device": "cuda" if has_gpu else "cpu"}
        return create_embedding_provider("huggingface", config=config, **kwargs)
    
    else:
        raise ConfigurationError(
            "No embedding providers available or configured. Please install a provider package and/or configure API credentials.",
            error_code="NO_CONFIGURED_PROVIDERS"
        )


def get_embedding_recommendations(
    use_case: str = "general",
    budget: str = "medium",
    privacy: str = "standard"
) -> List[Dict[str, Any]]:
    """
    Get embedding provider recommendations based on requirements.
    
    Args:
        use_case: "general", "enterprise", "research", "production"
        budget: "low", "medium", "high"
        privacy: "standard", "high", "maximum"
        
    Returns:
        List of recommended providers with reasoning
    """
    recommendations = []
    
    if privacy == "maximum":
        recommendations.extend([
            {
                "provider": "fastembed",
                "reason": "Local execution, no data sent to external services",
                "pros": ["Fast", "No API costs", "Private"],
                "cons": ["Limited model selection"]
            },
            {
                "provider": "ollama", 
                "reason": "Local model management with good model variety",
                "pros": ["Local", "Good models", "Easy management"],
                "cons": ["Requires setup", "Resource intensive"]
            }
        ])
    
    elif privacy == "high":
        recommendations.extend([
            {
                "provider": "azure_openai",
                "reason": "Enterprise privacy controls and compliance",
                "pros": ["Enterprise features", "Compliance", "High quality"],
                "cons": ["Requires Azure subscription"]
            },
            {
                "provider": "bedrock",
                "reason": "AWS enterprise security and compliance",
                "pros": ["AWS integration", "Multiple models", "Enterprise security"],
                "cons": ["AWS complexity"]
            }
        ])
    
    if budget == "low":
        recommendations.extend([
            {
                "provider": "fastembed",
                "reason": "No API costs, efficient local execution",
                "pros": ["Free", "Fast", "Efficient"],
                "cons": ["Limited models"]
            },
            {
                "provider": "gemini",
                "reason": "Competitive pricing for API-based service",
                "pros": ["Affordable", "Good quality", "Google infrastructure"],
                "cons": ["API dependency"]
            }
        ])
    
    elif budget == "high":
        recommendations.extend([
            {
                "provider": "openai",
                "reason": "Premium quality and features",
                "pros": ["Highest quality", "Latest models", "Excellent API"],
                "cons": ["Higher cost"]
            },
            {
                "provider": "azure_openai",
                "reason": "Enterprise features with premium quality",
                "pros": ["Enterprise grade", "High quality", "Azure integration"],
                "cons": ["Higher cost", "Setup complexity"]
            }
        ])
    
    if use_case == "enterprise":
        recommendations.extend([
            {
                "provider": "azure_openai",
                "reason": "Built for enterprise with compliance features",
                "pros": ["Enterprise auth", "Compliance", "SLA"],
                "cons": ["Complex setup"]
            },
            {
                "provider": "bedrock",
                "reason": "AWS enterprise integration",
                "pros": ["AWS ecosystem", "Multiple models", "Enterprise security"],
                "cons": ["AWS knowledge required"]
            }
        ])
    
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec["provider"] not in seen:
            seen.add(rec["provider"])
            unique_recommendations.append(rec)
    
    return unique_recommendations[:5]
