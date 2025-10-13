from __future__ import annotations
import asyncio
import os
from typing import List, Dict, Any, Optional
import time

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pydantic import Field, field_validator
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode
from ..utils.package.exception import ConfigurationError, ModelConnectionError
from upsonic.utils.printing import debug_log, warning_log


class OpenAIEmbeddingConfig(EmbeddingConfig):
    """Configuration for OpenAI embedding models."""
    
    api_key: Optional[str] = Field(None, description="OpenAI API key")
    organization: Optional[str] = Field(None, description="OpenAI organization ID")
    base_url: Optional[str] = Field(None, description="Custom OpenAI API base URL")
    
    model_name: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    
    enable_rate_limiting: bool = Field(True, description="Enable intelligent rate limiting")
    requests_per_minute: int = Field(3000, description="Max requests per minute")
    tokens_per_minute: int = Field(1000000, description="Max tokens per minute")
    
    parallel_requests: int = Field(5, description="Number of parallel requests")
    request_timeout: float = Field(60.0, description="Request timeout in seconds")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate and auto-correct model names."""
        model_mapping = {
            "text-embedding-ada-002": "text-embedding-ada-002",
            "text-embedding-3-small": "text-embedding-3-small",
            "text-embedding-3-large": "text-embedding-3-large",
            "ada-002": "text-embedding-ada-002",
            "3-small": "text-embedding-3-small",
            "3-large": "text-embedding-3-large",
            "small": "text-embedding-3-small",
            "large": "text-embedding-3-large"
        }
        
        if v in model_mapping:
            return model_mapping[v]
        
        valid_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small", 
            "text-embedding-3-large"
        ]
        
        if v not in valid_models:
            raise ValueError(f"Invalid OpenAI model: {v}. Valid models: {valid_models}")
        
        return v


class OpenAIEmbedding(EmbeddingProvider):
    
    config: OpenAIEmbeddingConfig
    
    def __init__(self, config: Optional[OpenAIEmbeddingConfig] = None, **kwargs):
        if not OPENAI_AVAILABLE:
            raise ConfigurationError(
                "OpenAI package not found. Install with: pip install openai",
                error_code="DEPENDENCY_MISSING"
            )
        
        if config is None:
            config = OpenAIEmbeddingConfig(**kwargs)
        
        super().__init__(config=config)
        
        self._setup_client()
        
        self._request_times: List[float] = []
        self._token_usage: List[tuple] = []
        
        self._model_info: Optional[Dict[str, Any]] = None
    
    def _setup_client(self):
        """Setup the OpenAI client with proper configuration."""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key in config.",
                error_code="API_KEY_MISSING"
            )
        
        client_params = {
            "api_key": api_key,
            "timeout": self.config.request_timeout
        }
        
        if self.config.organization:
            client_params["organization"] = self.config.organization
            
        if self.config.base_url:
            client_params["base_url"] = self.config.base_url
        
        self.client = AsyncOpenAI(**client_params)
    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """OpenAI supports document and query modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get current OpenAI embedding pricing."""
        pricing_map = {
            "text-embedding-ada-002": 0.10,
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13
        }
        
        price = pricing_map.get(self.config.model_name, 0.10)
        
        return {
            "per_million_tokens": price,
            "currency": "USD",
            "updated": "2024-01-01"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current OpenAI model."""
        if self._model_info is None:
            model_info_map = {
                "text-embedding-ada-002": {
                    "dimensions": 1536,
                    "max_tokens": 8191,
                    "description": "Most capable 2nd generation embedding model, replacing 16 first generation models"
                },
                "text-embedding-3-small": {
                    "dimensions": 1536,
                    "max_tokens": 8191,
                    "description": "High performance embedding model with smaller dimensions"
                },
                "text-embedding-3-large": {
                    "dimensions": 3072,
                    "max_tokens": 8191,
                    "description": "Most capable embedding model for both english and non-english tasks"
                }
            }
            
            self._model_info = model_info_map.get(self.config.model_name, {
                "dimensions": 1536,
                "max_tokens": 8191,
                "description": "Unknown OpenAI embedding model"
            })
            
            self._model_info.update({
                "model_name": self.config.model_name,
                "provider": "OpenAI",
                "type": "embedding"
            })
        
        return self._model_info
    
    async def _check_rate_limits(self, estimated_tokens: int) -> None:
        """Check and enforce rate limits."""
        if not self.config.enable_rate_limiting:
            return
        
        current_time = time.time()
        
        minute_ago = current_time - 60
        self._request_times = [t for t in self._request_times if t > minute_ago]
        self._token_usage = [(t, tokens) for t, tokens in self._token_usage if t > minute_ago]
        
        if len(self._request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        current_tokens = sum(tokens for _, tokens in self._token_usage)
        if current_tokens + estimated_tokens > self.config.tokens_per_minute:
            sleep_time = 60 - (current_time - self._token_usage[0][0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self._request_times.append(current_time)
        self._token_usage.append((current_time, estimated_tokens))
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate token count for texts."""
        total_chars = sum(len(text) for text in texts)
        return int(total_chars / 4)
    
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """
        Embed a batch of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            mode: Embedding mode (not used by OpenAI, but kept for compatibility)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        estimated_tokens = self._estimate_tokens(texts)
        await self._check_rate_limits(estimated_tokens)
        
        try:
            request_params = {
                "model": self.config.model_name,
                "input": texts,
                "encoding_format": "float"
            }
            
            response = await self.client.embeddings.create(**request_params)
            
            embeddings = [data.embedding for data in response.data]
            
            if hasattr(response, 'usage') and response.usage:
                self._metrics.total_tokens += response.usage.total_tokens
            
            return embeddings
            
        except openai.RateLimitError as e:
            wait_time = 60
            if "retry-after" in str(e):
                try:
                    import re
                    wait_match = re.search(r'retry-after[:\s]+(\d+)', str(e), re.IGNORECASE)
                    if wait_match:
                        wait_time = int(wait_match.group(1))
                except:
                    pass
            
            await asyncio.sleep(min(wait_time, 300))
            raise
            
        except openai.AuthenticationError as e:
            raise ConfigurationError(
                f"OpenAI authentication failed: {str(e)}",
                error_code="AUTHENTICATION_ERROR",
                original_error=e
            )
            
        except openai.BadRequestError as e:
            raise ModelConnectionError(
                f"OpenAI API request error: {str(e)}",
                error_code="API_REQUEST_ERROR", 
                original_error=e
            )
            
        except Exception as e:
            raise ModelConnectionError(
                f"OpenAI embedding failed: {str(e)}",
                error_code="EMBEDDING_FAILED",
                original_error=e
            )
    
    async def validate_connection(self) -> bool:
        """Validate OpenAI connection and model access."""
        try:
            test_result = await self._embed_batch(["test connection"], EmbeddingMode.QUERY)
            return len(test_result) > 0 and len(test_result[0]) > 0
            
        except Exception as e:
            debug_log(f"OpenAI connection validation failed: {str(e)}", context="OpenAIEmbedding")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics for this session."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        recent_requests = len([t for t in self._request_times if t > minute_ago])
        recent_tokens = sum(tokens for t, tokens in self._token_usage if t > minute_ago)
        
        return {
            "total_requests": len(self._request_times),
            "recent_requests_per_minute": recent_requests,
            "total_tokens_used": self._metrics.total_tokens,
            "recent_tokens_per_minute": recent_tokens,
            "rate_limit_requests": self.config.requests_per_minute,
            "rate_limit_tokens": self.config.tokens_per_minute,
            "model_name": self.config.model_name
        }
    
    def estimate_cost_detailed(self, num_texts: int, avg_text_length: int = 100) -> Dict[str, Any]:
        """Detailed cost estimation with breakdown."""
        base_estimate = self.estimate_cost(num_texts, avg_text_length)
        
        pricing = self.pricing_info
        cost_per_token = pricing["per_million_tokens"] / 1_000_000
        
        return {
            **base_estimate,
            "cost_per_token": cost_per_token,
            "cost_breakdown": {
                "base_cost": base_estimate["estimated_cost"],
                "buffer_cost": base_estimate["estimated_cost"] * 0.1,
                "total_with_buffer": base_estimate["estimated_cost"] * 1.1
            },
            "comparison": {
                "vs_ada_002": self._compare_model_cost("text-embedding-ada-002", num_texts, avg_text_length),
                "vs_3_small": self._compare_model_cost("text-embedding-3-small", num_texts, avg_text_length),
                "vs_3_large": self._compare_model_cost("text-embedding-3-large", num_texts, avg_text_length)
            }
        }
    
    def _compare_model_cost(self, model_name: str, num_texts: int, avg_text_length: int) -> Dict[str, float]:
        """Compare cost with another model."""
        pricing_map = {
            "text-embedding-ada-002": 0.10,
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13
        }
        
        if model_name not in pricing_map:
            return {"cost": 0.0, "difference": 0.0, "percentage": 0.0}
        
        avg_tokens = avg_text_length / 4
        total_tokens = num_texts * avg_tokens
        cost = (total_tokens / 1_000_000) * pricing_map[model_name]
        
        current_cost = (total_tokens / 1_000_000) * pricing_map[self.config.model_name]
        
        difference = cost - current_cost
        percentage = (difference / current_cost * 100) if current_cost > 0 else 0
        
        return {
            "cost": cost,
            "difference": difference,
            "percentage": percentage
        }
    
    async def close(self):
        """
        Clean up OpenAI client connections and resources.
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
                warning_log(f"Error closing OpenAI client: {e}", context="OpenAIEmbedding")
        
        await super().close()


def create_openai_embedding(
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    **kwargs
) -> OpenAIEmbedding:
    """
    Quick factory function for OpenAI embeddings.
    
    Args:
        model_name: OpenAI model name
        api_key: OpenAI API key (optional, uses env var if not provided)
        **kwargs: Additional configuration options
        
    Returns:
        Configured OpenAIEmbedding instance
    """
    config = OpenAIEmbeddingConfig(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )
    return OpenAIEmbedding(config=config)


def create_ada_002_embedding(**kwargs) -> OpenAIEmbedding:
    """Create OpenAI Ada-002 embedding provider."""
    return create_openai_embedding(model_name="text-embedding-ada-002", **kwargs)


def create_3_small_embedding(**kwargs) -> OpenAIEmbedding:
    """Create OpenAI text-embedding-3-small provider."""
    return create_openai_embedding(model_name="text-embedding-3-small", **kwargs)


def create_3_large_embedding(**kwargs) -> OpenAIEmbedding:
    """Create OpenAI text-embedding-3-large provider."""
    return create_openai_embedding(model_name="text-embedding-3-large", **kwargs)
