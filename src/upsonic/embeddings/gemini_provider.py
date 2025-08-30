from __future__ import annotations
import asyncio
import os
from typing import List, Dict, Any, Optional
import time

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import google.auth
    from google.auth import default
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

from pydantic import BaseModel, Field, validator
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode
from ..utils.error_wrapper import upsonic_error_handler
from ..utils.package.exception import ConfigurationError, ModelConnectionError


class GeminiEmbeddingConfig(EmbeddingConfig):
    """Configuration for Google Gemini embedding models."""
    
    api_key: Optional[str] = Field(None, description="Google AI API key")
    
    model_name: str = Field("text-embedding-004", description="Gemini embedding model")
    
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
    
    task_type: str = Field("RETRIEVAL_DOCUMENT", description="Embedding task type")
    title: Optional[str] = Field(None, description="Optional title for context")
    
    enable_batch_processing: bool = Field(True, description="Enable batch processing optimization")
    
    use_google_cloud_auth: bool = Field(False, description="Use Google Cloud authentication")
    project_id: Optional[str] = Field(None, description="Google Cloud project ID")
    location: str = Field("us-central1", description="Google Cloud location")
    
    requests_per_minute: int = Field(60, description="Requests per minute limit")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate Gemini model names."""
        valid_models = [
            "text-embedding-004",
            "text-embedding-preview-0409",
            "embedding-001"
        ]
        
        if v not in valid_models:
            print(f"Warning: '{v}' may not be a valid Gemini embedding model. Known models: {valid_models}")
        
        return v
    
    @validator('task_type')
    def validate_task_type(cls, v):
        """Validate embedding task type."""
        valid_tasks = [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT", 
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING"
        ]
        
        if v not in valid_tasks:
            raise ValueError(f"Invalid task_type: {v}. Valid options: {valid_tasks}")
        
        return v


class GeminiEmbedding(EmbeddingProvider):
    
    config: GeminiEmbeddingConfig
    
    def __init__(self, config: Optional[GeminiEmbeddingConfig] = None, **kwargs):
        if not GEMINI_AVAILABLE:
            raise ConfigurationError(
                "Google GenerativeAI package not found. Install with: pip install google-generativeai",
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
        if self.config.use_google_cloud_auth:
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
                
                print(f"Using Google Cloud authentication for project: {self.project_id}")
                
            except Exception as e:
                raise ConfigurationError(
                    f"Google Cloud authentication failed: {str(e)}",
                    error_code="GOOGLE_CLOUD_AUTH_ERROR",
                    original_error=e
                )
        else:
            api_key = self.config.api_key or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "Google AI API key not found. Set GOOGLE_AI_API_KEY environment variable or pass api_key in config.",
                    error_code="API_KEY_MISSING"
                )
            
            genai.configure(api_key=api_key)
            print("Using Google AI API key authentication")
    
    def _setup_client(self):
        """Setup Gemini client."""
        try:
            safety_settings = []
            if self.config.enable_safety_filtering:
                for category, threshold in self.config.safety_settings.items():
                    try:
                        harm_category = getattr(HarmCategory, category)
                        harm_threshold = getattr(HarmBlockThreshold, threshold)
                        safety_settings.append({
                            "category": harm_category,
                            "threshold": harm_threshold
                        })
                    except AttributeError:
                        print(f"Warning: Unknown safety setting: {category}={threshold}")
            
            self.safety_settings = safety_settings
            
            self.generation_config = {
                "temperature": 0.0,
            }
            
            print(f"Gemini client configured for model: {self.config.model_name}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Gemini client setup failed: {str(e)}",
                error_code="GEMINI_CLIENT_ERROR",
                original_error=e
            )
    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """Gemini supports all embedding modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY, EmbeddingMode.SYMMETRIC, EmbeddingMode.CLUSTERING]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get Google Gemini embedding pricing."""
        pricing_map = {
            "text-embedding-004": 0.00001,
            "text-embedding-preview-0409": 0.00001,
            "embedding-001": 0.0000125
        }
        
        price_per_1k_chars = pricing_map.get(self.config.model_name, 0.00001)
        
        price_per_million_tokens = price_per_1k_chars * 1000 * 4
        
        return {
            "per_1k_characters": price_per_1k_chars,
            "per_million_tokens": price_per_million_tokens,
            "currency": "USD",
            "billing_unit": "characters",
            "note": "Gemini pricing is based on characters, not tokens",
            "updated": "2024-01-01"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Gemini model."""
        if self._model_info is None:
            model_info_map = {
                "text-embedding-004": {
                    "dimensions": 768,
                    "max_input_chars": 2048,
                    "description": "Latest Gemini text embedding model",
                    "languages": "100+ languages"
                },
                "text-embedding-preview-0409": {
                    "dimensions": 768,
                    "max_input_chars": 2048,
                    "description": "Preview Gemini embedding model",
                    "languages": "100+ languages"
                },
                "embedding-001": {
                    "dimensions": 768,
                    "max_input_chars": 1024,
                    "description": "Gemini embedding model v1",
                    "languages": "100+ languages"
                }
            }
            
            self._model_info = model_info_map.get(self.config.model_name, {
                "dimensions": 768,
                "max_input_chars": 2048,
                "description": "Unknown Gemini embedding model",
                "languages": "Multiple"
            })
            
            self._model_info.update({
                "model_name": self.config.model_name,
                "provider": "Google Gemini",
                "type": "embedding",
                "task_type": self.config.task_type,
                "safety_filtering": self.config.enable_safety_filtering,
                "authentication": "Google Cloud" if self.config.use_google_cloud_auth else "API Key"
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
    
    @upsonic_error_handler(max_retries=3, show_error_details=True)
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """
        Embed a batch of texts using Google Gemini.
        
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
                    embeddings = self._embed_texts_batch(texts, task_type)
                    all_embeddings.extend(embeddings)
                except Exception as batch_error:
                    print(f"Batch processing failed, falling back to individual requests: {batch_error}")
                    for text in texts:
                        embedding = self._embed_single_text(text, task_type)
                        all_embeddings.append(embedding)
            else:
                for text in texts:
                    embedding = self._embed_single_text(text, task_type)
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
    
    def _embed_single_text(self, text: str, task_type: str) -> List[float]:
        """Embed a single text using Gemini."""
        try:
            result = genai.embed_content(
                model=f"models/{self.config.model_name}",
                content=text,
                task_type=task_type,
                title=self.config.title
            )
            
            return result['embedding']
            
        except Exception as e:
            raise ModelConnectionError(
                f"Single text embedding failed: {str(e)}",
                error_code="GEMINI_SINGLE_EMBED_ERROR",
                original_error=e
            )
    
    def _embed_texts_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Embed multiple texts in batch (if supported)."""
        try:
            results = []
            for text in texts:
                result = genai.embed_content(
                    model=f"models/{self.config.model_name}",
                    content=text,
                    task_type=task_type,
                    title=self.config.title
                )
                results.append(result['embedding'])
            
            return results
            
        except Exception as e:
            raise ModelConnectionError(
                f"Batch text embedding failed: {str(e)}",
                error_code="GEMINI_BATCH_EMBED_ERROR",
                original_error=e
            )
    
    async def validate_connection(self) -> bool:
        """Validate Gemini connection and model access."""
        try:
            test_result = await self._embed_batch(["test connection"], EmbeddingMode.QUERY)
            return len(test_result) > 0 and len(test_result[0]) > 0
            
        except Exception as e:
            print(f"Gemini connection validation failed: {str(e)}")
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
            "safety_filtering": self.config.enable_safety_filtering
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
            for model in genai.list_models():
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
            print(f"Could not list Gemini models: {e}")
            return []


def create_gemini_document_embedding(api_key: Optional[str] = None, **kwargs) -> GeminiEmbedding:
    """Create Gemini embedding optimized for documents."""
    config = GeminiEmbeddingConfig(
        api_key=api_key,
        model_name="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT",
        **kwargs
    )
    return GeminiEmbedding(config=config)


def create_gemini_query_embedding(api_key: Optional[str] = None, **kwargs) -> GeminiEmbedding:
    """Create Gemini embedding optimized for queries."""
    config = GeminiEmbeddingConfig(
        api_key=api_key,
        model_name="text-embedding-004",
        task_type="RETRIEVAL_QUERY",
        **kwargs
    )
    return GeminiEmbedding(config=config)


def create_gemini_semantic_embedding(api_key: Optional[str] = None, **kwargs) -> GeminiEmbedding:
    """Create Gemini embedding for semantic similarity."""
    config = GeminiEmbeddingConfig(
        api_key=api_key,
        model_name="text-embedding-004",
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
        project_id=project_id,
        location=location,
        model_name="text-embedding-004",
        **kwargs
    )
    return GeminiEmbedding(config=config)
