from .base import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingMetrics
)

from .openai_provider import OpenAIEmbedding
from .azure_openai_provider import AzureOpenAIEmbedding
from .bedrock_provider import BedrockEmbedding
from .huggingface_provider import HuggingFaceEmbedding
from .fastembed_provider import FastEmbedProvider
from .ollama_provider import OllamaEmbedding
from .gemini_provider import GeminiEmbedding

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
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingConfig", 
    "EmbeddingMode",
    "EmbeddingMetrics",
    
    "OpenAIEmbedding",
    "AzureOpenAIEmbedding", 
    "BedrockEmbedding",
    "HuggingFaceEmbedding",
    "FastEmbedProvider",
    "OllamaEmbedding",
    "GeminiEmbedding",
    "MilvusEmbedding",
    
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
]

PROVIDER_REGISTRY = {
    "openai": OpenAIEmbedding,
    "azure_openai": AzureOpenAIEmbedding,
    "azure": AzureOpenAIEmbedding,
    "bedrock": BedrockEmbedding,
    "aws": BedrockEmbedding,
    "huggingface": HuggingFaceEmbedding,
    "hf": HuggingFaceEmbedding,
    "fastembed": FastEmbedProvider,
    "qdrant": FastEmbedProvider,
    "ollama": OllamaEmbedding,
    "gemini": GeminiEmbedding,
    "google": GeminiEmbedding,
}
