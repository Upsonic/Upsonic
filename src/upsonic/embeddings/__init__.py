from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode, EmbeddingMetrics
    from .openai_provider import OpenAIEmbedding, OpenAIEmbeddingConfig
    from .azure_openai_provider import AzureOpenAIEmbedding
    from .bedrock_provider import BedrockEmbedding
    from .huggingface_provider import HuggingFaceEmbedding
    from .fastembed_provider import FastEmbedProvider
    from .ollama_provider import OllamaEmbedding
    from .gemini_provider import (
        GeminiEmbedding, 
        GeminiEmbeddingConfig,
        create_gemini_vertex_embedding,
        create_gemini_document_embedding,
        create_gemini_query_embedding,
        create_gemini_semantic_embedding,
        create_gemini_cloud_embedding
    )

if TYPE_CHECKING:
    from .base import (
        EmbeddingProvider,
        EmbeddingConfig,
        EmbeddingMode,
        EmbeddingMetrics
    )
    from .factory import (
        create_embedding_provider, 
        list_available_providers,
        get_provider_info,
        create_best_available_embedding,
        auto_detect_best_embedding,
        get_embedding_recommendations,
        create_openai_embedding,
        create_azure_openai_embedding, 
        create_bedrock_embedding,
        create_huggingface_embedding,
        create_fastembed_provider,
        create_ollama_embedding,
        create_gemini_embedding,
        create_gemini_vertex_embedding,
    )

def _get_base_classes():
    """Lazy import of base embedding classes."""
    from .base import (
        EmbeddingProvider,
        EmbeddingConfig,
        EmbeddingMode,
        EmbeddingMetrics
    )
    
    return {
        'EmbeddingProvider': EmbeddingProvider,
        'EmbeddingConfig': EmbeddingConfig,
        'EmbeddingMode': EmbeddingMode,
        'EmbeddingMetrics': EmbeddingMetrics,
    }

def _get_factory_functions():
    """Lazy import of factory functions."""
    from .factory import (
        create_embedding_provider, 
        list_available_providers,
        get_provider_info,
        create_best_available_embedding,
        auto_detect_best_embedding,
        get_embedding_recommendations,
        create_openai_embedding,
        create_azure_openai_embedding, 
        create_bedrock_embedding,
        create_huggingface_embedding,
        create_fastembed_provider,
        create_ollama_embedding,
        create_gemini_embedding,
        create_gemini_vertex_embedding,
    )
    
    return {
        'create_embedding_provider': create_embedding_provider,
        'list_available_providers': list_available_providers,
        'get_provider_info': get_provider_info,
        'create_best_available_embedding': create_best_available_embedding,
        'auto_detect_best_embedding': auto_detect_best_embedding,
        'get_embedding_recommendations': get_embedding_recommendations,
        'create_openai_embedding': create_openai_embedding,
        'create_azure_openai_embedding': create_azure_openai_embedding,
        'create_bedrock_embedding': create_bedrock_embedding,
        'create_huggingface_embedding': create_huggingface_embedding,
        'create_fastembed_provider': create_fastembed_provider,
        'create_ollama_embedding': create_ollama_embedding,
        'create_gemini_embedding': create_gemini_embedding,
        'create_gemini_vertex_embedding': create_gemini_vertex_embedding,
    }
def _get_openai_embedding():
    from .openai_provider import OpenAIEmbedding
    return OpenAIEmbedding

def _get_openai_embedding_config():
    from .openai_provider import OpenAIEmbeddingConfig
    return OpenAIEmbeddingConfig

def _get_azure_openai_embedding():
    from .azure_openai_provider import AzureOpenAIEmbedding
    return AzureOpenAIEmbedding

def _get_azure_openai_embedding_config():
    from .azure_openai_provider import AzureOpenAIEmbeddingConfig
    return AzureOpenAIEmbeddingConfig

def _get_bedrock_embedding():
    from .bedrock_provider import BedrockEmbedding
    return BedrockEmbedding

def _get_bedrock_embedding_config():
    from .bedrock_provider import BedrockEmbeddingConfig
    return BedrockEmbeddingConfig

def _get_huggingface_embedding():
    from .huggingface_provider import HuggingFaceEmbedding
    return HuggingFaceEmbedding

def _get_huggingface_embedding_config():
    from .huggingface_provider import HuggingFaceEmbeddingConfig
    return HuggingFaceEmbeddingConfig

def _get_fastembed_provider():
    from .fastembed_provider import FastEmbedProvider
    return FastEmbedProvider

def _get_fastembed_config():
    from .fastembed_provider import FastEmbedConfig
    return FastEmbedConfig

def _get_ollama_embedding():
    from .ollama_provider import OllamaEmbedding
    return OllamaEmbedding

def _get_ollama_embedding_config():
    from .ollama_provider import OllamaEmbeddingConfig
    return OllamaEmbeddingConfig

def _get_gemini_embedding():
    from .gemini_provider import GeminiEmbedding
    return GeminiEmbedding

def _get_gemini_embedding_config():
    from .gemini_provider import GeminiEmbeddingConfig
    return GeminiEmbeddingConfig

def _get_gemini_vertex_embedding():
    from .gemini_provider import create_gemini_vertex_embedding
    return create_gemini_vertex_embedding

def _get_gemini_document_embedding():
    from .gemini_provider import create_gemini_document_embedding
    return create_gemini_document_embedding

def _get_gemini_query_embedding():
    from .gemini_provider import create_gemini_query_embedding
    return create_gemini_query_embedding

def _get_gemini_semantic_embedding():
    from .gemini_provider import create_gemini_semantic_embedding
    return create_gemini_semantic_embedding

def _get_gemini_cloud_embedding():
    from .gemini_provider import create_gemini_cloud_embedding
    return create_gemini_cloud_embedding

def _get_azure_embedding_with_managed_identity():
    from .azure_openai_provider import create_azure_embedding_with_managed_identity
    return create_azure_embedding_with_managed_identity

def _get_create_titan_embedding():
    from .bedrock_provider import create_titan_embedding
    return create_titan_embedding

def _get_create_cohere_embedding():
    from .bedrock_provider import create_cohere_embedding
    return create_cohere_embedding

def _get_create_mpnet_embedding():
    from .huggingface_provider import create_mpnet_embedding
    return create_mpnet_embedding

def _get_create_minilm_embedding():
    from .huggingface_provider import create_minilm_embedding
    return create_minilm_embedding

def _get_create_huggingface_api_embedding():
    from .huggingface_provider import create_huggingface_api_embedding
    return create_huggingface_api_embedding

def _get_create_bge_large_embedding():
    from .fastembed_provider import create_bge_large_embedding
    return create_bge_large_embedding

def _get_create_e5_embedding():
    from .fastembed_provider import create_e5_embedding
    return create_e5_embedding

def _get_create_gpu_accelerated_embedding():
    from .fastembed_provider import create_gpu_accelerated_embedding
    return create_gpu_accelerated_embedding

def _get_create_sparse_embedding():
    from .fastembed_provider import create_sparse_embedding
    return create_sparse_embedding

def _get_create_nomic_embedding():
    from .ollama_provider import create_nomic_embedding
    return create_nomic_embedding

def _get_create_mxbai_embedding():
    from .ollama_provider import create_mxbai_embedding
    return create_mxbai_embedding

def _get_create_arctic_embedding():
    from .ollama_provider import create_arctic_embedding
    return create_arctic_embedding


def __getattr__(name: str) -> Any:
    """Lazy loading of provider classes and functions."""
    # Base classes
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    # Factory functions
    factory_functions = _get_factory_functions()
    if name in factory_functions:
        return factory_functions[name]
    
    # Provider classes (lazy loaded)
    lazy_loaders = {
        "OpenAIEmbedding": _get_openai_embedding,
        "OpenAIEmbeddingConfig": _get_openai_embedding_config,
        "AzureOpenAIEmbedding": _get_azure_openai_embedding,
        "AzureOpenAIEmbeddingConfig": _get_azure_openai_embedding_config,
        "BedrockEmbedding": _get_bedrock_embedding,
        "BedrockEmbeddingConfig": _get_bedrock_embedding_config,
        "HuggingFaceEmbedding": _get_huggingface_embedding,
        "HuggingFaceEmbeddingConfig": _get_huggingface_embedding_config,
        "FastEmbedProvider": _get_fastembed_provider,
        "FastEmbedConfig": _get_fastembed_config,
        "OllamaEmbedding": _get_ollama_embedding,
        "OllamaEmbeddingConfig": _get_ollama_embedding_config,
        "GeminiEmbedding": _get_gemini_embedding,
        "GeminiEmbeddingConfig": _get_gemini_embedding_config,
        "create_gemini_vertex_embedding": _get_gemini_vertex_embedding,
        "create_gemini_document_embedding": _get_gemini_document_embedding,
        "create_gemini_query_embedding": _get_gemini_query_embedding,
        "create_gemini_semantic_embedding": _get_gemini_semantic_embedding,
        "create_gemini_cloud_embedding": _get_gemini_cloud_embedding,
        "create_azure_embedding_with_managed_identity": _get_azure_embedding_with_managed_identity,
        "create_titan_embedding": _get_create_titan_embedding,
        "create_cohere_embedding": _get_create_cohere_embedding,
        "create_mpnet_embedding": _get_create_mpnet_embedding,
        "create_minilm_embedding": _get_create_minilm_embedding,
        "create_huggingface_api_embedding": _get_create_huggingface_api_embedding,
        "create_bge_large_embedding": _get_create_bge_large_embedding,
        "create_e5_embedding": _get_create_e5_embedding,
        "create_gpu_accelerated_embedding": _get_create_gpu_accelerated_embedding,
        "create_sparse_embedding": _get_create_sparse_embedding,
        "create_nomic_embedding": _get_create_nomic_embedding,
        "create_mxbai_embedding": _get_create_mxbai_embedding,
        "create_arctic_embedding": _get_create_arctic_embedding,
    }
    
    if name in lazy_loaders:
        return lazy_loaders[name]()
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )


__all__ = [
    # Base classes (always available)
    "EmbeddingProvider",
    "EmbeddingConfig", 
    "EmbeddingMode",
    "EmbeddingMetrics",
    
    "OpenAIEmbedding",
    "OpenAIEmbeddingConfig",
    "AzureOpenAIEmbedding",
    "AzureOpenAIEmbeddingConfig", 
    "BedrockEmbedding",
    "BedrockEmbeddingConfig",
    "HuggingFaceEmbedding",
    "HuggingFaceEmbeddingConfig",
    "FastEmbedProvider",
    "FastEmbedConfig",
    "OllamaEmbedding",
    "OllamaEmbeddingConfig",
    "GeminiEmbedding",
    "GeminiEmbeddingConfig",
    "create_embedding_provider",
    "list_available_providers",
    "get_provider_info",
    "create_best_available_embedding",
    "auto_detect_best_embedding",
    "get_embedding_recommendations",
    "create_openai_embedding",
    "create_azure_openai_embedding", 
    "create_bedrock_embedding",
    "create_huggingface_embedding",
    "create_fastembed_provider",
    "create_ollama_embedding",
    "create_gemini_embedding",
    "create_gemini_vertex_embedding",
    "create_gemini_document_embedding",
    "create_gemini_query_embedding",
    "create_gemini_semantic_embedding",
    "create_gemini_cloud_embedding",
    "create_azure_embedding_with_managed_identity",
    "create_titan_embedding",
    "create_cohere_embedding",
    "create_mpnet_embedding",
    "create_minilm_embedding",
    "create_huggingface_api_embedding",
    "create_bge_large_embedding",
    "create_e5_embedding",
    "create_gpu_accelerated_embedding",
    "create_sparse_embedding",
    "create_nomic_embedding",
    "create_mxbai_embedding",
    "create_arctic_embedding",
]

