from __future__ import annotations
import asyncio
import os
from typing import List, Dict, Any, Optional
import time

try:
    import openai
    from openai import AsyncAzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
    from azure.core.credentials import AccessToken
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

from upsonic.utils.printing import debug_log, warning_log

from pydantic import Field, field_validator
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode

from ..utils.package.exception import ConfigurationError, ModelConnectionError


class AzureOpenAIEmbeddingConfig(EmbeddingConfig):
    """Configuration for Azure OpenAI embedding models."""
    
    api_key: Optional[str] = Field(None, description="Azure OpenAI API key")
    azure_endpoint: Optional[str] = Field(None, description="Azure OpenAI endpoint URL")
    api_version: str = Field("2024-02-01", description="Azure OpenAI API version")
    deployment_name: Optional[str] = Field(None, description="Azure deployment name")
    
    use_managed_identity: bool = Field(False, description="Use Azure Managed Identity")
    tenant_id: Optional[str] = Field(None, description="Azure tenant ID")
    client_id: Optional[str] = Field(None, description="Azure client ID for managed identity")
    
    model_name: str = Field("text-embedding-ada-002", description="Embedding model name")
    
    enable_content_filtering: bool = Field(True, description="Enable Azure content filtering")
    data_residency_region: Optional[str] = Field(None, description="Data residency region")
    
    parallel_requests: int = Field(3, description="Parallel requests (Azure has lower limits)")
    requests_per_minute: int = Field(240, description="Requests per minute for Azure")
    tokens_per_minute: int = Field(240000, description="Tokens per minute for Azure")
    
    @field_validator('azure_endpoint')
    @classmethod
    def validate_azure_endpoint(cls, v):
        """Validate Azure endpoint format."""
        if v and not v.startswith('https://'):
            raise ValueError("Azure endpoint must start with https://")
        if v and not v.endswith('.openai.azure.com/'):
            if not v.endswith('.openai.azure.com'):
                v = v.rstrip('/') + '.openai.azure.com/'
            else:
                v = v + '/'
        return v
    
    @field_validator('deployment_name')
    @classmethod
    def validate_deployment_name(cls, v):
        """Validate deployment name format."""
        if v and not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Deployment name can only contain letters, numbers, hyphens, and underscores")
        return v


class AzureOpenAIEmbedding(EmbeddingProvider):
    
    config: AzureOpenAIEmbeddingConfig
    
    def __init__(self, config: Optional[AzureOpenAIEmbeddingConfig] = None, **kwargs):
        if not AZURE_OPENAI_AVAILABLE:
            raise ConfigurationError(
                "OpenAI package not found. Install with: pip install openai",
                error_code="DEPENDENCY_MISSING"
            )
        
        if config is None:
            config = AzureOpenAIEmbeddingConfig(**kwargs)
        
        super().__init__(config=config)
        
        self._setup_client()
        
        self._request_times: List[float] = []
        self._token_usage: List[tuple] = []
        
        self._model_info: Optional[Dict[str, Any]] = None
        
        self._credential = None
        if self.config.use_managed_identity:
            self._setup_managed_identity()
    
    def _setup_managed_identity(self):
        """Setup Azure Managed Identity authentication."""
        if not AZURE_IDENTITY_AVAILABLE:
            raise ConfigurationError(
                "Azure Identity package not found. Install with: pip install azure-identity",
                error_code="DEPENDENCY_MISSING"
            )
        
        try:
            if self.config.client_id:
                self._credential = ManagedIdentityCredential(client_id=self.config.client_id)
            else:
                self._credential = DefaultAzureCredential()
        except Exception as e:
            raise ConfigurationError(
                f"Failed to setup Azure Managed Identity: {str(e)}",
                error_code="AZURE_AUTH_ERROR",
                original_error=e
            )
    
    def _setup_client(self):
        """Setup the Azure OpenAI client with proper configuration."""
        azure_endpoint = self.config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ConfigurationError(
                "Azure endpoint not found. Set AZURE_OPENAI_ENDPOINT environment variable or pass azure_endpoint in config.",
                error_code="ENDPOINT_MISSING"
            )
        self.config.azure_endpoint = azure_endpoint

        client_params = {
            "azure_endpoint": azure_endpoint,
            "api_version": self.config.api_version,
            "timeout": self.config.timeout
        }
        
        if self.config.use_managed_identity:
            if not self._credential:
                self._setup_managed_identity()
            client_params["azure_ad_token_provider"] = self._get_azure_token
        else:
            api_key = self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY environment variable or pass api_key in config.",
                    error_code="API_KEY_MISSING"
                )
            self.config.api_key = api_key
            client_params["api_key"] = api_key
        
        self.client = AsyncAzureOpenAI(**client_params)
    
    async def _get_azure_token(self) -> str:
        """Get Azure AD token for authentication."""
        if not self._credential:
            raise ConfigurationError("Azure credential not initialized")
        
        try:
            token = await self._credential.get_token("https://cognitiveservices.azure.com/.default")
            return token.token
        except Exception as e:
            raise ConfigurationError(
                f"Failed to get Azure AD token: {str(e)}",
                error_code="AZURE_TOKEN_ERROR",
                original_error=e
            )
    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """Azure OpenAI supports document and query modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get Azure OpenAI embedding pricing (may differ from standard OpenAI)."""
        pricing_map = {
            "text-embedding-ada-002": 0.10,
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13
        }
        
        price = pricing_map.get(self.config.model_name, 0.10)
        
        return {
            "per_million_tokens": price,
            "currency": "USD",
            "note": "Azure pricing may vary by region and contract",
            "updated": "2024-01-01"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Azure OpenAI model."""
        if self._model_info is None:
            model_info_map = {
                "text-embedding-ada-002": {
                    "dimensions": 1536,
                    "max_tokens": 8191,
                    "description": "Azure OpenAI Ada-002 embedding model"
                },
                "text-embedding-3-small": {
                    "dimensions": 1536,
                    "max_tokens": 8191,
                    "description": "Azure OpenAI text-embedding-3-small model"
                },
                "text-embedding-3-large": {
                    "dimensions": 3072,
                    "max_tokens": 8191,
                    "description": "Azure OpenAI text-embedding-3-large model"
                }
            }
            
            self._model_info = model_info_map.get(self.config.model_name, {
                "dimensions": 1536,
                "max_tokens": 8191,
                "description": "Unknown Azure OpenAI embedding model"
            })
            
            self._model_info.update({
                "model_name": self.config.model_name,
                "deployment_name": self.config.deployment_name,
                "provider": "Azure OpenAI",
                "type": "embedding",
                "azure_endpoint": self.config.azure_endpoint,
                "api_version": self.config.api_version,
                "content_filtering": self.config.enable_content_filtering
            })
        
        return self._model_info
    
    async def _check_rate_limits(self, estimated_tokens: int) -> None:
        """Check and enforce Azure-specific rate limits."""
        if not self.config.enable_retry_with_backoff:
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
        Embed a batch of texts using Azure OpenAI API.
        
        Args:
            texts: List of text strings to embed
            mode: Embedding mode (not used by Azure OpenAI, but kept for compatibility)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        estimated_tokens = self._estimate_tokens(texts)
        await self._check_rate_limits(estimated_tokens)
        
        try:
            model_param = self.config.deployment_name or self.config.model_name
            
            request_params = {
                "model": model_param,
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
            if self.config.use_managed_identity:
                try:
                    await self._get_azure_token()
                    raise
                except:
                    pass
            
            raise ConfigurationError(
                f"Azure OpenAI authentication failed: {str(e)}",
                error_code="AZURE_AUTH_ERROR",
                original_error=e
            )
            
        except openai.BadRequestError as e:
            raise ModelConnectionError(
                f"Azure OpenAI API request error: {str(e)}",
                error_code="AZURE_API_ERROR", 
                original_error=e
            )
            
        except Exception as e:
            raise ModelConnectionError(
                f"Azure OpenAI embedding failed: {str(e)}",
                error_code="AZURE_EMBEDDING_FAILED",
                original_error=e
            )
    
    async def validate_connection(self) -> bool:
        """Validate Azure OpenAI connection and deployment access."""
        try:
            test_result = await self._embed_batch(["test connection"], EmbeddingMode.QUERY)
            return len(test_result) > 0 and len(test_result[0]) > 0
            
        except Exception as e:
            debug_log(f"Azure OpenAI connection validation failed: {str(e)}", context="AzureOpenAIEmbedding")
            return False
    
    def get_azure_info(self) -> Dict[str, Any]:
        """Get Azure-specific configuration information."""
        return {
            "azure_endpoint": self.config.azure_endpoint,
            "api_version": self.config.api_version,
            "deployment_name": self.config.deployment_name,
            "use_managed_identity": self.config.use_managed_identity,
            "content_filtering_enabled": self.config.enable_content_filtering,
            "data_residency_region": self.config.data_residency_region,
            "authentication_method": "Managed Identity" if self.config.use_managed_identity else "API Key"
        }
    
    def get_compliance_info(self) -> Dict[str, Any]:
        """Get compliance and security information."""
        return {
            "content_filtering": self.config.enable_content_filtering,
            "data_residency": self.config.data_residency_region,
            "authentication": "Azure AD" if self.config.use_managed_identity else "API Key",
            "encryption": "Azure encryption at rest and in transit",
            "compliance_certifications": [
                "SOC 2 Type 2",
                "ISO 27001",
                "HIPAA",
                "FedRAMP",
                "GDPR compliant"
            ]
        }


    async def close(self):
        """
        Clean up Azure OpenAI client connections and resources.
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
                warning_log(f"Error closing Azure OpenAI client: {e}", context="AzureOpenAIEmbedding")
        
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
                warning_log(f"Error closing Azure credential: {e}", context="AzureOpenAIEmbedding")
        
        await super().close()


def create_azure_openai_embedding(
    azure_endpoint: str,
    deployment_name: str,
    api_key: Optional[str] = None,
    use_managed_identity: bool = False,
    **kwargs
) -> AzureOpenAIEmbedding:
    """
    Quick factory function for Azure OpenAI embeddings.
    
    Args:
        azure_endpoint: Azure OpenAI endpoint URL
        deployment_name: Azure deployment name
        api_key: Azure OpenAI API key (optional if using managed identity)
        use_managed_identity: Use Azure Managed Identity authentication
        **kwargs: Additional configuration options
        
    Returns:
        Configured AzureOpenAIEmbedding instance
    """
    config = AzureOpenAIEmbeddingConfig(
        azure_endpoint=azure_endpoint,
        deployment_name=deployment_name,
        api_key=api_key,
        use_managed_identity=use_managed_identity,
        **kwargs
    )
    return AzureOpenAIEmbedding(config=config)


def create_azure_embedding_with_managed_identity(
    azure_endpoint: str,
    deployment_name: str,
    **kwargs
) -> AzureOpenAIEmbedding:
    """Create Azure OpenAI embedding with managed identity."""
    return create_azure_openai_embedding(
        azure_endpoint=azure_endpoint,
        deployment_name=deployment_name,
        use_managed_identity=True,
        **kwargs
    )
