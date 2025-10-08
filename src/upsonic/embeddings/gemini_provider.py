from __future__ import annotations
import asyncio
import os
from typing import List, Dict, Any, Optional
import time

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import google.auth
    from google.auth import default
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

from upsonic.utils.printing import warning_log, info_log, debug_log

from pydantic import BaseModel, Field, field_validator
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode
from ..utils.package.exception import ConfigurationError, ModelConnectionError


class GeminiEmbeddingConfig(EmbeddingConfig):
    """Configuration for Google Gemini embedding models."""
    
    api_key: Optional[str] = Field(None, description="Google AI API key")
    
    model_name: str = Field("gemini-embedding-001", description="Gemini embedding model")
    
    enable_safety_filtering: bool = Field(True, description="Enable Google's safety filtering")
    safety_settings: Dict[str, str] = Field(
        default_factory=lambda: {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
        },
        description="Safety filtering settings"
    )
    
    task_type: str = Field("SEMANTIC_SIMILARITY", description="Embedding task type")
    title: Optional[str] = Field(None, description="Optional title for context")
    
    enable_batch_processing: bool = Field(True, description="Enable batch processing optimization")
    
    use_google_cloud_auth: bool = Field(False, description="Use Google Cloud authentication")
    project_id: Optional[str] = Field(None, description="Google Cloud project ID")
    location: str = Field("us-central1", description="Google Cloud location")
    
    requests_per_minute: int = Field(60, description="Requests per minute limit")
    
    use_vertex_ai: bool = Field(False, description="Use Vertex AI API instead of Gemini Developer API")
    api_version: str = Field("v1beta", description="API version to use (v1beta, v1, v1alpha)")
    enable_caching: bool = Field(False, description="Enable response caching")
    cache_ttl_seconds: int = Field(3600, description="Cache TTL in seconds")
    
    output_dimensionality: Optional[int] = Field(None, description="Output embedding dimension (128-3072)")
    embedding_config: Optional[Dict[str, Any]] = Field(None, description="Additional embedding configuration")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate Gemini model names."""
        valid_models = [
            "gemini-embedding-001",
            "text-embedding-005",
            "text-multilingual-embedding-002",
            "embedding-001"
        ]
        
        if v not in valid_models:
            warning_log(f"'{v}' may not be a valid Gemini embedding model. Known models: {valid_models}", context="GeminiEmbedding")
        
        return v
    
    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v):
        """Validate embedding task type."""
        valid_tasks = [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT", 
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
            "CODE_RETRIEVAL_QUERY"
        ]
        
        if v not in valid_tasks:
            raise ValueError(f"Invalid task_type: {v}. Valid options: {valid_tasks}")
        
        return v
    
    @field_validator('api_version')
    @classmethod
    def validate_api_version(cls, v):
        """Validate API version."""
        valid_versions = ["v1beta", "v1", "v1alpha"]
        if v not in valid_versions:
            raise ValueError(f"Invalid api_version: {v}. Valid options: {valid_versions}")
        return v
    
    @field_validator('output_dimensionality')
    @classmethod
    def validate_output_dimensionality(cls, v):
        """Validate output dimensionality."""
        if v is not None:
            if not (128 <= v <= 3072):
                raise ValueError(f"output_dimensionality must be between 128 and 3072, got {v}")
        return v


class GeminiEmbedding(EmbeddingProvider):
    
    config: GeminiEmbeddingConfig
    
    def __init__(self, config: Optional[GeminiEmbeddingConfig] = None, **kwargs):
        if not GEMINI_AVAILABLE:
            raise ConfigurationError(
                "Google GenAI package not found. Install with: pip install google-genai",
                error_code="DEPENDENCY_MISSING"
            )
        
        if config is None:
            config = GeminiEmbeddingConfig(**kwargs)
        
        super().__init__(config=config)
        
        self._setup_authentication()
        self._setup_client()
        
        self._request_times: List[float] = []
        self._model_info: Optional[Dict[str, Any]] = None
        self._request_count = 0
        self._character_count = 0
    
    def _setup_authentication(self):
        """Setup Google authentication."""
        if self.config.use_google_cloud_auth or self.config.use_vertex_ai:
            if not GOOGLE_AUTH_AVAILABLE:
                raise ConfigurationError(
                    "Google Auth package not found. Install with: pip install google-auth",
                    error_code="DEPENDENCY_MISSING"
                )
            
            try:
                credentials, project = default()
                self.credentials = credentials
                self.project_id = self.config.project_id or project
                
                if not self.project_id:
                    raise ConfigurationError(
                        "Google Cloud project ID not found. Set GOOGLE_CLOUD_PROJECT or provide project_id",
                        error_code="PROJECT_ID_MISSING"
                    )
                
                info_log(f"Using Google Cloud authentication for project: {self.project_id}", context="GeminiEmbedding")
                
            except Exception as e:
                raise ConfigurationError(
                    f"Google Cloud authentication failed: {str(e)}",
                    error_code="GOOGLE_CLOUD_AUTH_ERROR",
                    original_error=e
                )
        else:
            api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "Google AI API key not found. Set GOOGLE_API_KEY environment variable or pass api_key in config.",
                    error_code="API_KEY_MISSING"
                )
            
            info_log("Using Google AI API key authentication", context="GeminiEmbedding")
    
    def _setup_client(self):
        """Setup Gemini client with new google-genai library."""
        try:
            http_options = types.HttpOptions(api_version=self.config.api_version)
            
            if self.config.use_google_cloud_auth or self.config.use_vertex_ai:
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.config.location,
                    http_options=http_options
                )
                info_log(f"Gemini client configured for Vertex AI with model: {self.config.model_name}", context="GeminiEmbedding")
            else:
                api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
                self.client = genai.Client(
                    api_key=api_key,
                    http_options=http_options
                )
                info_log(f"Gemini client configured for Gemini Developer API with model: {self.config.model_name}", context="GeminiEmbedding")
            
            if self.config.enable_caching:
                self._setup_caching()
            
        except Exception as e:
            raise ConfigurationError(
                f"Gemini client setup failed: {str(e)}",
                error_code="GEMINI_CLIENT_ERROR",
                original_error=e
            )
    
    def _setup_caching(self):
        """Setup response caching."""
        try:
            cache_config = types.CacheConfig(
                ttl_seconds=self.config.cache_ttl_seconds
            )
            self.cache = self.client.caches.create(config=cache_config)
            info_log(f"Response caching enabled with TTL: {self.config.cache_ttl_seconds}s", context="GeminiEmbedding")
        except Exception as e:
            warning_log(f"Could not setup caching: {e}", context="GeminiEmbedding")
            self.cache = None
    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """Gemini supports all embedding modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY, EmbeddingMode.SYMMETRIC, EmbeddingMode.CLUSTERING]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get Google Gemini embedding pricing."""
        pricing_map = {
            "gemini-embedding-001": 0.00015, 
            "text-embedding-005": None,      
            "text-multilingual-embedding-002": None
        }
        
        price_per_1k_chars = pricing_map.get(self.config.model_name, 0.00001)
        
        price_per_million_tokens = price_per_1k_chars * 1000 * 4
        
        return {
            "per_1k_characters": price_per_1k_chars,
            "per_million_tokens": price_per_million_tokens,
            "currency": "USD",
            "billing_unit": "characters",
            "note": "Gemini pricing is based on characters, not tokens",
            "updated": "2024-12-01",
            "api_type": "Vertex AI" if self.config.use_vertex_ai else "Gemini Developer API"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Gemini model."""
        if self._model_info is None:
            model_info_map = {
                "gemini-embedding-001": {
                    "dimensions": [3072, 1536, 768],   # via MRL (Matryoshka Representation Learning)
                    "max_input_tokens": 2048,
                    "description": "GA unified Gemini embedding model (multilingual + code + text)",
                    "languages": "100+ languages",
                    "supported_tasks": [
                        "retrieval",
                        "semantic_similarity",
                        "classification",
                        "clustering",
                        "multilingual",
                        "code",
                        "question_answering",
                        "fact_verification"
                    ]
                },
                "text-embedding-005": {
                    "dimensions": 768,
                    "max_input_tokens": 2048,
                    "description": "Legacy English/code embedding model",
                    "languages": "English",
                    "supported_tasks": [
                        "retrieval",
                        "semantic_similarity",
                        "classification",
                        "clustering",
                        "code"
                    ]
                },
                "text-multilingual-embedding-002": {
                    "dimensions": 768,
                    "max_input_tokens": 2048,
                    "description": "Legacy multilingual embedding model",
                    "languages": "100+ languages",
                    "supported_tasks": [
                        "retrieval",
                        "semantic_similarity",
                        "classification",
                        "clustering"
                    ]
                }
            }
            
            self._model_info = model_info_map.get(self.config.model_name, {
                "dimensions": 768,
                "max_input_chars": 2048,
                "description": "Unknown Gemini embedding model",
                "languages": "Multiple",
                "supported_tasks": ["retrieval", "semantic_similarity"]
            })
            
            self._model_info.update({
                "model_name": self.config.model_name,
                "provider": "Google Gemini",
                "type": "embedding",
                "task_type": self.config.task_type,
                "safety_filtering": self.config.enable_safety_filtering,
                "authentication": "Google Cloud" if self.config.use_google_cloud_auth else "API Key",
                "api_type": "Vertex AI" if self.config.use_vertex_ai else "Gemini Developer API",
                "api_version": self.config.api_version,
                "caching_enabled": self.config.enable_caching,
                "output_dimensionality": self.config.output_dimensionality
            })
        
        return self._model_info
    
    def _get_task_type_for_mode(self, mode: EmbeddingMode) -> str:
        """Map embedding mode to Gemini task type."""
        mode_mapping = {
            EmbeddingMode.DOCUMENT: "RETRIEVAL_DOCUMENT",
            EmbeddingMode.QUERY: "RETRIEVAL_QUERY", 
            EmbeddingMode.SYMMETRIC: "SEMANTIC_SIMILARITY",
            EmbeddingMode.CLUSTERING: "CLUSTERING"
        }
        
        return mode_mapping.get(mode, self.config.task_type)
    
    async def _check_rate_limits(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        
        minute_ago = current_time - 60
        self._request_times = [t for t in self._request_times if t > minute_ago]
        
        if len(self._request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self._request_times.append(current_time)
    
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """
        Embed a batch of texts using Google Gemini with new API.
        
        Args:
            texts: List of text strings to embed
            mode: Embedding mode for optimization
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        await self._check_rate_limits()
        
        try:
            task_type = self._get_task_type_for_mode(mode)
            
            all_embeddings = []
            
            if self.config.enable_batch_processing and len(texts) > 1:
                try:
                    embeddings = await self._embed_texts_batch(texts, task_type)
                    all_embeddings.extend(embeddings)
                except Exception as batch_error:
                    warning_log(f"Batch processing failed, falling back to individual requests: {batch_error}", context="GeminiEmbedding")
                    for text in texts:
                        embedding = await self._embed_single_text(text, task_type)
                        all_embeddings.append(embedding)
            else:
                for text in texts:
                    embedding = await self._embed_single_text(text, task_type)
                    all_embeddings.append(embedding)
            
            self._request_count += len(texts)
            self._character_count += sum(len(text) for text in texts)
            
            return all_embeddings
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise ModelConnectionError(
                    f"Gemini API rate limit exceeded: {str(e)}",
                    error_code="GEMINI_RATE_LIMIT",
                    original_error=e
                )
            elif "safety" in str(e).lower():
                raise ConfigurationError(
                    f"Content filtered by Gemini safety settings: {str(e)}",
                    error_code="GEMINI_SAFETY_ERROR",
                    original_error=e
                )
            else:
                raise ModelConnectionError(
                    f"Gemini embedding failed: {str(e)}",
                    error_code="GEMINI_EMBEDDING_ERROR",
                    original_error=e
                )
    
    async def _embed_single_text(self, text: str, task_type: str) -> List[float]:
        """Embed a single text using Gemini with new API."""
        try:
            if not text or not text.strip():
                default_dim = 768  # Default dimension
                if self.config.output_dimensionality:
                    default_dim = self.config.output_dimensionality
                elif self.config.model_name == "gemini-embedding-001":
                    default_dim = 3072
                return [0.0] * default_dim
            
            config = None
            if self.config.output_dimensionality or self.config.embedding_config:
                config = types.EmbedContentConfig()
                if self.config.output_dimensionality:
                    config.output_dimensionality = self.config.output_dimensionality
                if self.config.embedding_config:
                    for key, value in self.config.embedding_config.items():
                        setattr(config, key, value)
            
            response = self.client.models.embed_content(
                model=self.config.model_name,
                contents=text,
                config=config
            )
            
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values
            else:
                raise ModelConnectionError(
                    "No embeddings returned from Gemini API",
                    error_code="GEMINI_NO_EMBEDDINGS"
                )
            
        except Exception as e:
            raise ModelConnectionError(
                f"Single text embedding failed: {str(e)}",
                error_code="GEMINI_SINGLE_EMBED_ERROR",
                original_error=e
            )
    
    async def _embed_texts_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Embed multiple texts in batch using new API."""
        try:
            config = None
            if self.config.output_dimensionality or self.config.embedding_config:
                config = types.EmbedContentConfig()
                if self.config.output_dimensionality:
                    config.output_dimensionality = self.config.output_dimensionality
                if self.config.embedding_config:
                    for key, value in self.config.embedding_config.items():
                        setattr(config, key, value)
            
            results = []
            for text in texts:
                response = self.client.models.embed_content(
                    model=self.config.model_name,
                    contents=text,
                    config=config
                )
                if response.embeddings and len(response.embeddings) > 0:
                    results.append(response.embeddings[0].values)
                else:
                    raise ModelConnectionError(
                        "No embeddings returned from Gemini API",
                        error_code="GEMINI_NO_EMBEDDINGS"
                    )
            
            return results
            
        except Exception as e:
            warning_log(f"Batch embedding failed, using individual requests: {e}", context="GeminiEmbedding")
            results = []
            for text in texts:
                embedding = await self._embed_single_text(text, task_type)
                results.append(embedding)
            return results
    
    async def validate_connection(self) -> bool:
        """Validate Gemini connection and model access."""
        try:
            test_result = await self._embed_batch(["test connection"], EmbeddingMode.QUERY)
            return len(test_result) > 0 and len(test_result[0]) > 0
            
        except Exception as e:
            debug_log(f"Gemini connection validation failed: {str(e)}", context="GeminiEmbedding")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics."""
        pricing = self.pricing_info
        estimated_cost = (self._character_count / 1000) * pricing["per_1k_characters"]
        
        return {
            "total_requests": self._request_count,
            "total_characters": self._character_count,
            "estimated_cost_usd": estimated_cost,
            "average_chars_per_request": self._character_count / max(self._request_count, 1),
            "model_name": self.config.model_name,
            "task_type": self.config.task_type,
            "safety_filtering": self.config.enable_safety_filtering,
            "api_type": "Vertex AI" if self.config.use_vertex_ai else "Gemini Developer API",
            "caching_enabled": self.config.enable_caching,
            "output_dimensionality": self.config.output_dimensionality
        }
    
    def get_safety_info(self) -> Dict[str, Any]:
        """Get content safety and filtering information."""
        return {
            "safety_filtering_enabled": self.config.enable_safety_filtering,
            "safety_settings": self.config.safety_settings,
            "content_policies": [
                "Harassment protection",
                "Hate speech detection", 
                "Sexually explicit content filtering",
                "Dangerous content blocking"
            ],
            "compliance": [
                "Google AI Principles",
                "Responsible AI practices",
                "Content policy enforcement"
            ]
        }
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available Gemini embedding models."""
        try:
            models = []
            for model in self.client.models.list():
                if 'embedding' in model.name.lower():
                    models.append({
                        "name": model.name,
                        "display_name": model.display_name,
                        "description": model.description,
                        "supported_generation_methods": model.supported_generation_methods,
                        "input_token_limit": getattr(model, 'input_token_limit', 'unknown'),
                        "output_token_limit": getattr(model, 'output_token_limit', 'unknown')
                    })
            
            return models
            
        except Exception as e:
            debug_log(f"Could not list Gemini models: {e}", context="GeminiEmbedding")
            return []

    async def close(self):
        """
        Clean up Gemini client and clear resources.
        """
        if hasattr(self, 'client') and self.client:
            try:
                if hasattr(self.client, 'aclose'):
                    await self.client.aclose()
                elif hasattr(self.client, 'close'):
                    if asyncio.iscoroutinefunction(self.client.close):
                        await self.client.close()
                    else:
                        self.client.close()
                
                del self.client
                self.client = None
            except Exception as e:
                warning_log(f"Error closing Gemini client: {e}", context="GeminiEmbedding")
        
        if hasattr(self, 'cache') and self.cache:
            try:
                if hasattr(self.cache, 'aclose'):
                    await self.cache.aclose()
                elif hasattr(self.cache, 'close'):
                    if asyncio.iscoroutinefunction(self.cache.close):
                        await self.cache.close()
                    else:
                        self.cache.close()
                
                del self.cache
                self.cache = None
            except Exception as e:
                warning_log(f"Error closing Gemini cache: {e}", context="GeminiEmbedding")
        
        if hasattr(self, '_credential') and self._credential:
            try:
                if hasattr(self._credential, 'aclose'):
                    await self._credential.aclose()
                elif hasattr(self._credential, 'close'):
                    if asyncio.iscoroutinefunction(self._credential.close):
                        await self._credential.close()
                    else:
                        self._credential.close()
                
                del self._credential
                self._credential = None
            except Exception as e:
                warning_log(f"Error closing Google credential: {e}", context="GeminiEmbedding")
        
        if hasattr(self, '_model_info'):
            self._model_info = None
        
        await super().close()


def create_gemini_document_embedding(api_key: Optional[str] = None, **kwargs) -> GeminiEmbedding:
    """Create Gemini embedding optimized for documents."""
    config = GeminiEmbeddingConfig(
        api_key=api_key,
        model_name="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        **kwargs
    )
    return GeminiEmbedding(config=config)


def create_gemini_query_embedding(api_key: Optional[str] = None, **kwargs) -> GeminiEmbedding:
    """Create Gemini embedding optimized for queries."""
    config = GeminiEmbeddingConfig(
        api_key=api_key,
        model_name="gemini-embedding-001",
        task_type="RETRIEVAL_QUERY",
        **kwargs
    )
    return GeminiEmbedding(config=config)


def create_gemini_semantic_embedding(api_key: Optional[str] = None, **kwargs) -> GeminiEmbedding:
    """Create Gemini embedding for semantic similarity."""
    config = GeminiEmbeddingConfig(
        api_key=api_key,
        model_name="gemini-embedding-001",
        task_type="SEMANTIC_SIMILARITY",
        **kwargs
    )
    return GeminiEmbedding(config=config)


def create_gemini_cloud_embedding(
    project_id: str,
    location: str = "us-central1",
    **kwargs
) -> GeminiEmbedding:
    """Create Gemini embedding with Google Cloud authentication."""
    config = GeminiEmbeddingConfig(
        use_google_cloud_auth=True,
        use_vertex_ai=True,
        project_id=project_id,
        location=location,
        model_name="gemini-embedding-001",
        **kwargs
    )
    return GeminiEmbedding(config=config)


def create_gemini_vertex_embedding(
    project_id: str,
    location: str = "us-central1",
    **kwargs
) -> GeminiEmbedding:
    """Create Gemini embedding with Vertex AI."""
    config = GeminiEmbeddingConfig(
        use_vertex_ai=True,
        project_id=project_id,
        location=location,
        model_name="gemini-embedding-001",
        **kwargs
    )
    return GeminiEmbedding(config=config)
