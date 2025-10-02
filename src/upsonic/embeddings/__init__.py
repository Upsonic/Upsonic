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
def _get_openai_embedding():
    from .openai_provider import OpenAIEmbedding
    return OpenAIEmbedding

def _get_openai_embedding_config():
    from .openai_provider import OpenAIEmbeddingConfig
    return OpenAIEmbeddingConfig

def _get_azure_openai_embedding():
    from .azure_openai_provider import AzureOpenAIEmbedding
    return AzureOpenAIEmbedding

def _get_bedrock_embedding():
    from .bedrock_provider import BedrockEmbedding
    return BedrockEmbedding

def _get_huggingface_embedding():
    from .huggingface_provider import HuggingFaceEmbedding
    return HuggingFaceEmbedding

def _get_fastembed_provider():
    from .fastembed_provider import FastEmbedProvider
    return FastEmbedProvider

def _get_ollama_embedding():
    from .ollama_provider import OllamaEmbedding
    return OllamaEmbedding

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


def __getattr__(name: str) -> Any:
    """Lazy loading of provider classes and functions."""
    lazy_loaders = {
        "OpenAIEmbedding": _get_openai_embedding,
        "OpenAIEmbeddingConfig": _get_openai_embedding_config,
        "AzureOpenAIEmbedding": _get_azure_openai_embedding,
        "BedrockEmbedding": _get_bedrock_embedding,
        "HuggingFaceEmbedding": _get_huggingface_embedding,
        "FastEmbedProvider": _get_fastembed_provider,
        "OllamaEmbedding": _get_ollama_embedding,
        "GeminiEmbedding": _get_gemini_embedding,
        "GeminiEmbeddingConfig": _get_gemini_embedding_config,
        "create_gemini_vertex_embedding": _get_gemini_vertex_embedding,
        "create_gemini_document_embedding": _get_gemini_document_embedding,
        "create_gemini_query_embedding": _get_gemini_query_embedding,
        "create_gemini_semantic_embedding": _get_gemini_semantic_embedding,
        "create_gemini_cloud_embedding": _get_gemini_cloud_embedding,
    }
    
    if name in lazy_loaders:
        return lazy_loaders[name]()
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Base classes (always available)
    "EmbeddingProvider",
    "EmbeddingConfig", 
    "EmbeddingMode",
    "EmbeddingMetrics",
    
    "OpenAIEmbedding",
    "OpenAIEmbeddingConfig",
    "AzureOpenAIEmbedding", 
    "BedrockEmbedding",
    "HuggingFaceEmbedding",
    "FastEmbedProvider",
    "OllamaEmbedding",
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
]
