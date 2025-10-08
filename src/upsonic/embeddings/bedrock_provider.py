from __future__ import annotations
import asyncio
import json
from typing import List, Dict, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from pydantic import BaseModel, Field, field_validator
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode
from ..utils.package.exception import ConfigurationError, ModelConnectionError
from upsonic.utils.printing import warning_log, info_log, debug_log


class BedrockEmbeddingConfig(EmbeddingConfig):
    """Configuration for AWS Bedrock embedding models."""
    
    aws_access_key_id: Optional[str] = Field(None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS secret access key")
    aws_session_token: Optional[str] = Field(None, description="AWS session token")
    region_name: str = Field("us-east-1", description="AWS region")
    profile_name: Optional[str] = Field(None, description="AWS profile name")
    
    model_name: str = Field("amazon.titan-embed-text-v1", description="Bedrock embedding model")
    
    model_id: Optional[str] = Field(None, description="Full Bedrock model ID (overrides model_name)")
    inference_profile: Optional[str] = Field(None, description="Bedrock inference profile")
    
    enable_guardrails: bool = Field(True, description="Enable Bedrock guardrails")
    guardrail_id: Optional[str] = Field(None, description="Custom guardrail ID")
    
    enable_model_caching: bool = Field(True, description="Enable model response caching")
    prefer_provisioned_throughput: bool = Field(False, description="Prefer provisioned throughput models")
    
    enable_cloudwatch_logging: bool = Field(True, description="Enable CloudWatch logging")
    log_group_name: Optional[str] = Field(None, description="CloudWatch log group name")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate and map Bedrock model names."""
        model_mapping = {
            "titan-embed-text-v1": "amazon.titan-embed-text-v1",
            "titan-embed-text-v2": "amazon.titan-embed-text-v2:0",
            "marengo-embed-2-7-v1:0": "twelvelabs.marengo-embed-2-7-v1:0",
            "cohere-embed-english": "cohere.embed-english-v3",
            "cohere-embed-multilingual": "cohere.embed-multilingual-v3",
            "titan": "amazon.titan-embed-text-v1",
            "cohere": "cohere.embed-english-v3"
        }
        
        if v in model_mapping:
            return model_mapping[v]
        
        if "." in v:
            return v
        
        warning_log(f"Unknown model '{v}', defaulting to amazon.titan-embed-text-v1", context="BedrockEmbedding")
        return "amazon.titan-embed-text-v1"
    
    @field_validator('region_name')
    @classmethod
    def validate_region(cls, v):
        """Validate AWS region for Bedrock availability."""
        bedrock_regions = [
            "us-east-1", "us-west-2", "ap-southeast-1", "ap-northeast-1", 
            "eu-central-1", "eu-west-1", "eu-west-3"
        ]
        
        if v not in bedrock_regions:
            warning_log(f"Region '{v}' may not support Bedrock. Supported regions: {bedrock_regions}", context="BedrockEmbedding")
        
        return v


class BedrockEmbedding(EmbeddingProvider):
    
    config: BedrockEmbeddingConfig
    
    def __init__(self, config: Optional[BedrockEmbeddingConfig] = None, **kwargs):
        if not BOTO3_AVAILABLE:
            raise ConfigurationError(
                "Boto3 package not found. Install with: pip install boto3",
                error_code="DEPENDENCY_MISSING"
            )
        
        if config is None:
            config = BedrockEmbeddingConfig(**kwargs)
        
        super().__init__(config=config)
        
        self._setup_aws_session()
        self._setup_bedrock_client()
        
        self._model_info: Optional[Dict[str, Any]] = None
        
        self._invocation_count = 0
        self._total_input_tokens = 0
    
    def _setup_aws_session(self):
        """Setup AWS session with proper credentials."""
        session_kwargs = {}
        
        if self.config.profile_name:
            session_kwargs["profile_name"] = self.config.profile_name
        elif self.config.aws_access_key_id and self.config.aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": self.config.aws_access_key_id,
                "aws_secret_access_key": self.config.aws_secret_access_key
            })
            if self.config.aws_session_token:
                session_kwargs["aws_session_token"] = self.config.aws_session_token
        
        try:
            self.session = boto3.Session(**session_kwargs)
            
            sts_client = self.session.client('sts', region_name=self.config.region_name)
            identity = sts_client.get_caller_identity()
            info_log(f"AWS authentication successful. Account: {identity.get('Account')}", context="BedrockEmbedding")
            
        except NoCredentialsError:
            raise ConfigurationError(
                "AWS credentials not found. Configure credentials via AWS CLI, environment variables, or IAM roles.",
                error_code="AWS_CREDENTIALS_MISSING"
            )
        except Exception as e:
            raise ConfigurationError(
                f"AWS session setup failed: {str(e)}",
                error_code="AWS_SESSION_ERROR",
                original_error=e
            )
    
    def _setup_bedrock_client(self):
        """Setup Bedrock runtime client."""
        try:
            self.bedrock_client = self.session.client(
                'bedrock-runtime',
                region_name=self.config.region_name
            )
            
            self.bedrock_info_client = self.session.client(
                'bedrock',
                region_name=self.config.region_name
            )
            
        except Exception as e:
            raise ConfigurationError(
                f"Bedrock client setup failed: {str(e)}",
                error_code="BEDROCK_CLIENT_ERROR",
                original_error=e
            )
    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """Bedrock supports document and query modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get AWS Bedrock embedding pricing."""
        pricing_map = {
            "amazon.titan-embed-text-v1": 0.0001,
            "amazon.titan-embed-text-v2:0": 0.00002,
            "twelvelabs.marengo-embed-2-7-v1:0": 0.0007,
            "cohere.embed-english-v3": 0.0001,
            "cohere.embed-multilingual-v3": 0.0001
        }
        
        model_id = self.config.model_id or self.config.model_name
        price_per_1k = pricing_map.get(model_id, 0.0001)
        
        return {
            "per_1k_tokens": price_per_1k,
            "per_million_tokens": price_per_1k * 1000,
            "currency": "USD",
            "region": self.config.region_name,
            "note": "Pricing varies by region and model",
            "updated": "2024-01-01"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Bedrock model."""
        if self._model_info is None:
            model_id = self.config.model_id or self.config.model_name
            
            model_info_map = {
                "amazon.titan-embed-text-v1": {
                    "dimensions": 1536,
                    "max_tokens": 8192,
                    "description": "Amazon Titan Text Embeddings v1",
                    "languages": ["English"],
                    "provider": "Amazon"
                },
                "amazon.titan-embed-text-v2:0": {
                    "dimensions": 1024,
                    "max_tokens": 8192,
                    "description": "Amazon Titan Text Embeddings v2 - Optimized",
                    "languages": ["English"],
                    "provider": "Amazon"
                },
                "twelvelabs.marengo-embed-2-7-v1:0": {
                    "dimensions": 1024,
                    "max_tokens": 2048,
                    "description": "Marengo Embed 2.7",
                    "languages": ["English"],
                    "provider": "Amazon"
                },
                "cohere.embed-english-v3": {
                    "dimensions": 1024,
                    "max_tokens": 512,
                    "description": "Cohere Embed English v3",
                    "languages": ["English"],
                    "provider": "Cohere"
                },
                "cohere.embed-multilingual-v3": {
                    "dimensions": 1024,
                    "max_tokens": 512,
                    "description": "Cohere Embed Multilingual v3",
                    "languages": ["100+ languages"],
                    "provider": "Cohere"
                }
            }
            
            self._model_info = model_info_map.get(model_id, {
                "dimensions": 1536,
                "max_tokens": 8192,
                "description": "Unknown Bedrock embedding model",
                "languages": ["Unknown"],
                "provider": "Unknown"
            })
            
            self._model_info.update({
                "model_id": model_id,
                "provider": "AWS Bedrock",
                "type": "embedding",
                "region": self.config.region_name
            })
            
            try:
                response = self.bedrock_info_client.get_foundation_model(modelIdentifier=model_id)
                model_details = response.get('modelDetails', {})
                self._model_info.update({
                    "model_arn": model_details.get('modelArn'),
                    "model_status": model_details.get('modelLifecycle', {}).get('status'),
                    "supported_customizations": model_details.get('customizationsSupported', []),
                    "supported_inference_types": model_details.get('inferenceTypesSupported', [])
                })
            except Exception as e:
                debug_log(f"Could not fetch live model info: {e}", context="BedrockEmbedding")
        
        return self._model_info
    
    def _prepare_titan_request(self, texts: List[str]) -> Dict[str, Any]:
        """Prepare request body for Titan models."""
        return {
            "inputText": texts[0] if len(texts) == 1 else texts,
            "dimensions": self.get_model_info().get("dimensions", 1536),
            "normalize": self.config.normalize_embeddings
        }
    
    def _prepare_cohere_request(self, texts: List[str]) -> Dict[str, Any]:
        """Prepare request body for Cohere models."""
        return {
            "texts": texts,
            "input_type": "search_document"
        }
    
    def _prepare_request_body(self, texts: List[str], mode: EmbeddingMode) -> Dict[str, Any]:
        """Prepare request body based on the model provider."""
        model_id = self.config.model_id or self.config.model_name
        
        if model_id.startswith("amazon.titan"):
            return self._prepare_titan_request(texts)
        elif model_id.startswith("cohere"):
            body = self._prepare_cohere_request(texts)
            if mode == EmbeddingMode.QUERY:
                body["input_type"] = "search_query"
            return body
        else:
            return {"texts": texts}
    
    def _extract_embeddings(self, response_body: Dict[str, Any], num_texts: int) -> List[List[float]]:
        """Extract embeddings from response based on model type."""
        model_id = self.config.model_id or self.config.model_name
        
        if model_id.startswith("amazon.titan"):
            if "embedding" in response_body:
                return [response_body["embedding"]]
            elif "embeddings" in response_body:
                return response_body["embeddings"]
            else:
                raise ModelConnectionError("Unexpected Titan response format")
                
        elif model_id.startswith("cohere"):
            if "embeddings" in response_body:
                return response_body["embeddings"]
            else:
                raise ModelConnectionError("Unexpected Cohere response format")
        else:
            for key in ["embeddings", "embedding", "vectors"]:
                if key in response_body:
                    result = response_body[key]
                    if isinstance(result, list):
                        return result if isinstance(result[0], list) else [result]
            
            raise ModelConnectionError("Could not extract embeddings from response")
    
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """
        Embed a batch of texts using AWS Bedrock.
        
        Args:
            texts: List of text strings to embed
            mode: Embedding mode for optimization
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        model_id = self.config.model_id or self.config.model_name
        
        try:
            all_embeddings = []
            
            batch_size = min(self.config.batch_size, 25)
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                request_body = self._prepare_request_body(batch_texts, mode)
                
                response = self.bedrock_client.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                
                batch_embeddings = self._extract_embeddings(response_body, len(batch_texts))
                all_embeddings.extend(batch_embeddings)
                
                self._invocation_count += 1
                self._total_input_tokens += sum(len(text.split()) for text in batch_texts)
                
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'ThrottlingException':
                await asyncio.sleep(2)
                raise
            elif error_code in ['ValidationException', 'ModelNotReadyException']:
                raise ConfigurationError(
                    f"Bedrock model error: {error_message}",
                    error_code=f"BEDROCK_{error_code.upper()}",
                    original_error=e
                )
            else:
                raise ModelConnectionError(
                    f"Bedrock API error ({error_code}): {error_message}",
                    error_code=f"BEDROCK_{error_code.upper()}",
                    original_error=e
                )
                
        except Exception as e:
            raise ModelConnectionError(
                f"Bedrock embedding failed: {str(e)}",
                error_code="BEDROCK_EMBEDDING_ERROR",
                original_error=e
            )
    
    async def validate_connection(self) -> bool:
        """Validate Bedrock connection and model access."""
        try:
            test_result = await self._embed_batch(["test connection"], EmbeddingMode.QUERY)
            return len(test_result) > 0 and len(test_result[0]) > 0
            
        except Exception as e:
            debug_log(f"Bedrock connection validation failed: {str(e)}", context="BedrockEmbedding")
            return False
    
    def get_aws_info(self) -> Dict[str, Any]:
        """Get AWS-specific configuration information."""
        try:
            sts_client = self.session.client('sts', region_name=self.config.region_name)
            identity = sts_client.get_caller_identity()
            
            return {
                "account_id": identity.get('Account'),
                "user_arn": identity.get('Arn'),
                "region": self.config.region_name,
                "model_id": self.config.model_id or self.config.model_name,
                "profile_name": self.config.profile_name,
                "guardrails_enabled": self.config.enable_guardrails,
                "cloudwatch_logging": self.config.enable_cloudwatch_logging
            }
        except Exception as e:
            return {"error": f"Could not retrieve AWS info: {e}"}
    
    def get_cost_estimate(self) -> Dict[str, Any]:
        """Get detailed cost estimation for current usage."""
        pricing = self.pricing_info
        cost_per_1k = pricing["per_1k_tokens"]
        
        estimated_cost = (self._total_input_tokens / 1000) * cost_per_1k
        
        return {
            "total_invocations": self._invocation_count,
            "total_input_tokens": self._total_input_tokens,
            "estimated_cost_usd": estimated_cost,
            "cost_per_1k_tokens": cost_per_1k,
            "average_tokens_per_invocation": self._total_input_tokens / max(self._invocation_count, 1),
            "model_id": self.config.model_id or self.config.model_name,
            "region": self.config.region_name
        }
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available embedding models in Bedrock."""
        try:
            response = self.bedrock_info_client.list_foundation_models(
                byInferenceType='ON_DEMAND',
                byOutputModality='EMBEDDING'
            )
            
            models = []
            for model in response.get('modelSummaries', []):
                models.append({
                    "model_id": model.get('modelId'),
                    "model_name": model.get('modelName'),
                    "provider_name": model.get('providerName'),
                    "input_modalities": model.get('inputModalities', []),
                    "output_modalities": model.get('outputModalities', []),
                    "customizations_supported": model.get('customizationsSupported', []),
                    "inference_types_supported": model.get('inferenceTypesSupported', [])
                })
            
            return models
            
        except Exception as e:
            debug_log(f"Could not list models: {e}", context="BedrockEmbedding")
            return []

    async def close(self):
        """
        Clean up AWS Bedrock connections and resources.
        """
        if hasattr(self, 'bedrock_client') and self.bedrock_client:
            try:
                if hasattr(self.bedrock_client, 'aclose'):
                    await self.bedrock_client.aclose()
                elif hasattr(self.bedrock_client, 'close'):
                    self.bedrock_client.close()
            except Exception as e:
                warning_log(f"Error closing Bedrock client: {e}", context="BedrockEmbedding")
        
        if hasattr(self, 'bedrock_info_client') and self.bedrock_info_client:
            try:
                if hasattr(self.bedrock_info_client, 'aclose'):
                    await self.bedrock_info_client.aclose()
                elif hasattr(self.bedrock_info_client, 'close'):
                    self.bedrock_info_client.close()
            except Exception as e:
                warning_log(f"Error closing Bedrock info client: {e}", context="BedrockEmbedding")
        
        if hasattr(self, 'session') and self.session:
            try:
                if hasattr(self.session, 'aclose'):
                    await self.session.aclose()
                elif hasattr(self.session, 'close'):
                    self.session.close()
            except Exception as e:
                warning_log(f"Error closing AWS session: {e}", context="BedrockEmbedding")
        
        await super().close()


def create_titan_embedding(
    region_name: str = "us-east-1",
    model_version: str = "v1",
    **kwargs
) -> BedrockEmbedding:
    """Create AWS Titan embedding provider."""
    model_map = {
        "v1": "amazon.titan-embed-text-v1",
        "v2": "amazon.titan-embed-text-v2:0",
    }
    
    config = BedrockEmbeddingConfig(
        model_name=model_map.get(model_version, "amazon.titan-embed-text-v1"),
        region_name=region_name,
        **kwargs
    )
    return BedrockEmbedding(config=config)


def create_cohere_embedding(
    language: str = "english",
    region_name: str = "us-east-1",
    **kwargs
) -> BedrockEmbedding:
    """Create Cohere embedding provider."""
    model_map = {
        "english": "cohere.embed-english-v3",
        "multilingual": "cohere.embed-multilingual-v3"
    }
    
    config = BedrockEmbeddingConfig(
        model_name=model_map.get(language, "cohere.embed-english-v3"),
        region_name=region_name,
        **kwargs
    )
    return BedrockEmbedding(config=config)
