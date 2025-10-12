from __future__ import annotations
import asyncio
import os
from typing import List, Dict, Any, Optional
import numpy as np


try:
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    import torch.nn.functional as F
    import torch
    from huggingface_hub import hf_hub_download, login, InferenceClient
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    BitsAndBytesConfig = None
    F = None
    torch = None
    hf_hub_download = None
    login = None
    InferenceClient = None
    _TRANSFORMERS_AVAILABLE = False
from pydantic import Field, field_validator

from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingMode
from upsonic.utils.printing import info_log, debug_log
from ..utils.package.exception import ConfigurationError, ModelConnectionError


class HuggingFaceEmbeddingConfig(EmbeddingConfig):
    """Configuration for HuggingFace embedding models."""
    
    model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="HuggingFace model name or path")
    
    hf_token: Optional[str] = Field(None, description="HuggingFace API token")
    
    use_api: bool = Field(False, description="Use HuggingFace Inference API instead of local model")
    use_local: bool = Field(True, description="Use local model execution")
    
    device: Optional[str] = Field(None, description="Device to run model on (auto-detected if None)")
    torch_dtype: str = Field("float32", description="PyTorch data type (float16, float32, bfloat16)")
    trust_remote_code: bool = Field(False, description="Trust remote code in model")
    
    max_seq_length: Optional[int] = Field(None, description="Maximum sequence length")
    pooling_strategy: str = Field("mean", description="Pooling strategy (mean, cls, max)")
    normalize_embeddings: bool = Field(True, description="Normalize embeddings to unit length")
    
    enable_quantization: bool = Field(False, description="Enable model quantization")
    quantization_bits: int = Field(8, description="Quantization bits (4, 8, 16)")
    enable_gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save memory")
    
    wait_for_model: bool = Field(True, description="Wait for model to load if using API")
    timeout: int = Field(None, description="Timeout for model")

    cache_dir: Optional[str] = Field(None, description="Model cache directory")
    force_download: bool = Field(False, description="Force re-download of model")
    
    @field_validator('pooling_strategy')
    @classmethod
    def validate_pooling_strategy(cls, v):
        valid_strategies = ['mean', 'cls', 'max', 'mean_sqrt_len']
        if v not in valid_strategies:
            raise ValueError(f"Invalid pooling strategy: {v}. Valid options: {valid_strategies}")
        return v
    
    @field_validator('torch_dtype')
    @classmethod
    def validate_torch_dtype(cls, v):
        valid_dtypes = ['float16', 'float32', 'bfloat16']
        if v not in valid_dtypes:
            raise ValueError(f"Invalid torch_dtype: {v}. Valid options: {valid_dtypes}")
        return v


class HuggingFaceEmbedding(EmbeddingProvider):

    
    config: HuggingFaceEmbeddingConfig
    
    def __init__(self, config: Optional[HuggingFaceEmbeddingConfig] = None, **kwargs):
        if not _TRANSFORMERS_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="transformers",
                install_command='pip install transformers',
                feature_name="transformers provider"
            )

        if config is None:
            config = HuggingFaceEmbeddingConfig(**kwargs)

        super().__init__(config=config)
        
        self._setup_device()
        self._setup_authentication()
        
        if self.config.use_local and not self.config.use_api:
            self._setup_local_model()
        
        self._model_info: Optional[Dict[str, Any]] = None
        
        if self.config.use_api:
            self._setup_api_session()
    
    def _setup_device(self):
        """Setup compute device for local models."""
        if self.config.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
            
        info_log(f"HuggingFace embedding using device: {self.device}", context="HuggingFaceEmbedding")
    
    def _setup_authentication(self):
        """Setup HuggingFace authentication."""
        hf_token = self.config.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        if hf_token:
            try:
                login(token=hf_token)
                from upsonic.utils.printing import success_log
                success_log("Successfully authenticated with HuggingFace Hub", "HuggingFaceProvider")
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"HuggingFace authentication failed: {e}", "HuggingFaceProvider")
    
    def _setup_local_model(self):
        """Setup local model and tokenizer with optional quantization."""
        
        try:
            info_log(f"Loading HuggingFace model: {self.config.model_name}", context="HuggingFaceEmbedding")
            
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "cache_dir": self.config.cache_dir,
                "force_download": self.config.force_download
            }
            
            
            torch_dtype = getattr(torch, self.config.torch_dtype)
            
            if self.config.enable_quantization:
                if self.config.quantization_bits == 8:
                    debug_log("Configuring 8-bit quantization using bitsandbytes...", context="HuggingFaceEmbedding")
                    model_kwargs['load_in_8bit'] = True
                    model_kwargs['device_map'] = "auto"

                elif self.config.quantization_bits == 4:
                    debug_log("Configuring 4-bit quantization using bitsandbytes...", context="HuggingFaceEmbedding")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    model_kwargs['quantization_config'] = quantization_config
                    model_kwargs['device_map'] = "auto"
                    torch_dtype = torch.bfloat16

            tokenizer_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['quantization_config', 'device_map', 'load_in_8bit']}
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                **tokenizer_kwargs
            )
            
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
                **model_kwargs
            )

            if 'device_map' not in model_kwargs:
                self.model = self.model.to(self.device)
                debug_log(f"Model moved to device: {self.device}", context="HuggingFaceEmbedding")

            self.model.eval()
            
            if self.config.enable_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            info_log("Model loaded successfully.", context="HuggingFaceEmbedding")
            
        except Exception as e:
            if "bitsandbytes" in str(e).lower():
                raise ConfigurationError(
                    "Quantization requires the 'bitsandbytes' and 'accelerate' libraries. Please install them with 'pip install bitsandbytes accelerate'",
                    error_code="DEPENDENCY_MISSING",
                    original_error=e
                )
            raise ConfigurationError(
                f"Failed to load HuggingFace model {self.config.model_name}: {str(e)}",
                error_code="MODEL_LOAD_ERROR",
                original_error=e
            )
    
    def _setup_api_session(self):
        """Setup InferenceClient for HuggingFace API calls."""
        
        hf_token = self.config.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        self._inference_client = InferenceClient(
            token=hf_token,
            timeout=getattr(self.config, 'timeout', 30)
        )

    
    @property
    def supported_modes(self) -> List[EmbeddingMode]:
        """HuggingFace supports all embedding modes."""
        return [EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY, EmbeddingMode.SYMMETRIC]
    
    @property 
    def pricing_info(self) -> Dict[str, float]:
        """Get HuggingFace pricing info (API usage)."""
        if self.config.use_local:
            return {
                "per_million_tokens": 0.0,
                "currency": "USD",
                "note": "Local model execution - no API costs"
            }
        else:
            return {
                "per_million_tokens": 0.01,
                "currency": "USD",
                "note": "HuggingFace Inference API pricing varies by model",
                "updated": "2024-01-01"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current HuggingFace model."""
        if self._model_info is None:
            info = {
                "model_name": self.config.model_name,
                "provider": "HuggingFace",
                "type": "embedding",
                "execution_mode": "API" if self.config.use_api else "Local",
                "device": self.device if hasattr(self, 'device') else "unknown"
            }
            
            if hasattr(self, 'model') and self.model:
                try:
                    info["parameters"] = sum(p.numel() for p in self.model.parameters())
                    info["hidden_size"] = getattr(self.model.config, 'hidden_size', 'unknown')
                    info["max_position_embeddings"] = getattr(self.model.config, 'max_position_embeddings', 'unknown')
                except:
                    pass
            
            if hasattr(self, 'tokenizer') and self.tokenizer:
                info["vocab_size"] = len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else 'unknown'
                info["model_max_length"] = getattr(self.tokenizer, 'model_max_length', 'unknown')
            
            self._model_info = info
        
        return self._model_info
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _cls_pooling(self, model_output, attention_mask):
        """Use CLS token for pooling."""
        return model_output[0][:, 0]
    
    def _max_pooling(self, model_output, attention_mask):
        """Apply max pooling."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]
    
    def _apply_pooling(self, model_output, attention_mask):
        """Apply the configured pooling strategy."""
        if self.config.pooling_strategy == "mean":
            return self._mean_pooling(model_output, attention_mask)
        elif self.config.pooling_strategy == "cls":
            return self._cls_pooling(model_output, attention_mask)
        elif self.config.pooling_strategy == "max":
            return self._max_pooling(model_output, attention_mask)
        else:
            return self._mean_pooling(model_output, attention_mask)
    
    async def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using local model."""
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            raise ModelConnectionError("Local model not initialized")
        
        try:
            max_length = self.config.max_seq_length or self.tokenizer.model_max_length
            if max_length > 512:
                max_length = 512
            
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
                sentence_embeddings = self._apply_pooling(model_output, encoded_input['attention_mask'])
                
                if self.config.normalize_embeddings:
                    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                
                embeddings = sentence_embeddings.cpu().numpy().tolist()
                
                return embeddings
                
        except Exception as e:
            raise ModelConnectionError(
                f"Local embedding failed: {str(e)}",
                error_code="LOCAL_EMBEDDING_ERROR",
                original_error=e
            )
    

    async def _embed_api(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using HuggingFace InferenceClient."""
        if not self._inference_client:
            raise ModelConnectionError("InferenceClient not initialized")
        
        try:
            def sync_api_call():
                return self._inference_client.feature_extraction(
                    text=texts,
                    model=self.config.model_name
                )

            embeddings_np = await asyncio.to_thread(sync_api_call)

            if isinstance(embeddings_np, np.ndarray):
                if embeddings_np.ndim == 1:
                    return [embeddings_np.tolist()]
                else:
                    return embeddings_np.tolist()
            
            if isinstance(embeddings_np, list):
                return embeddings_np
            
            return []
            
        except Exception as e:
            raise ModelConnectionError(f"Failed to get embeddings via API: {str(e)}", original_error=e)
    
    async def _embed_batch(self, texts: List[str], mode: EmbeddingMode = EmbeddingMode.DOCUMENT) -> List[List[float]]:
        """
        Embed a batch of texts using HuggingFace model or API.
        
        Args:
            texts: List of text strings to embed
            mode: Embedding mode
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self.config.use_api:
            return await self._embed_api(texts)
        else:
            return await self._embed_local(texts)
    
    async def validate_connection(self) -> bool:
        """Validate HuggingFace model or API connection."""
        try:
            test_result = await self._embed_batch(["test connection"], EmbeddingMode.QUERY)
            return len(test_result) > 0 and len(test_result[0]) > 0
            
        except Exception as e:
            debug_log(f"HuggingFace connection validation failed: {str(e)}", context="HuggingFaceEmbedding")
            return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for local models."""
        if not hasattr(self, 'model') or self.device == "cpu":
            return {"memory_usage": "N/A", "device": self.device}
        
        try:
            if self.device.startswith("cuda"):
                allocated = torch.cuda.memory_allocated(self.device)
                cached = torch.cuda.memory_reserved(self.device)
                
                return {
                    "device": self.device,
                    "allocated_memory_mb": allocated / 1024 / 1024,
                    "cached_memory_mb": cached / 1024 / 1024,
                    "total_memory_mb": torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024
                }
            else:
                return {"memory_usage": "Memory tracking not available for this device", "device": self.device}
                
        except Exception as e:
            return {"memory_usage": f"Error getting memory info: {e}", "device": self.device}

    async def close(self):
        """
        Clean up HuggingFace models, tokenizer, and API client.
        """
        if hasattr(self, 'model') and self.model:
            try:
                del self.model
                self.model = None
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Error closing HuggingFace model: {e}", "HuggingFaceProvider")
        
        if hasattr(self, 'tokenizer') and self.tokenizer:
            try:
                del self.tokenizer
                self.tokenizer = None
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Error closing HuggingFace tokenizer: {e}", "HuggingFaceProvider")
        
        if hasattr(self, '_inference_client') and self._inference_client:
            try:
                if hasattr(self._inference_client, 'aclose'):
                    await self._inference_client.aclose()
                elif hasattr(self._inference_client, 'close'):
                    if asyncio.iscoroutinefunction(self._inference_client.close):
                        await self._inference_client.close()
                    else:
                        self._inference_client.close()
                
                del self._inference_client
                self._inference_client = None
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Error closing HuggingFace InferenceClient: {e}", "HuggingFaceProvider")
        
        # Clear CUDA cache if using GPU
        if hasattr(self, 'device') and self.device.startswith('cuda'):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    from upsonic.utils.printing import success_log
                    success_log("CUDA cache cleared", "HuggingFaceProvider")
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Could not clear CUDA cache: {e}", "HuggingFaceProvider")
        
        await super().close()

    async def remove_local_cache(self) -> bool:
        """
        Remove HuggingFace model/tokenizer cache files from local storage.
        """
        from huggingface_hub import try_to_load_from_cache
        import shutil, os

        try:
            model_path = try_to_load_from_cache(
                repo_id=self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            if model_path and os.path.exists(model_path):
                root = os.path.dirname(model_path)
                shutil.rmtree(root, ignore_errors=True)
                from upsonic.utils.printing import success_log
                success_log(f"Local cache removed for {self.config.model_name} at {root}", "HuggingFaceProvider")
                return True
            else:
                from upsonic.utils.printing import info_log
                info_log(f"No local cache found for {self.config.model_name}", "HuggingFaceProvider")
                return False
        except Exception as e:
            from upsonic.utils.printing import error_log
            error_log(f"Error removing local cache for {self.config.model_name}: {e}", "HuggingFaceProvider")
            return False


def create_sentence_transformer_embedding(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs
) -> HuggingFaceEmbedding:
    """Create a sentence transformer embedding provider."""
    config = HuggingFaceEmbeddingConfig(
        model_name=model_name,
        use_local=True,
        pooling_strategy="mean",
        normalize_embeddings=True,
        **kwargs
    )
    return HuggingFaceEmbedding(config=config)


def create_mpnet_embedding(**kwargs) -> HuggingFaceEmbedding:
    """Create MPNet embedding provider (high quality)."""
    return create_sentence_transformer_embedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
        **kwargs
    )


def create_minilm_embedding(**kwargs) -> HuggingFaceEmbedding:
    """Create MiniLM embedding provider (fast and efficient)."""
    return create_sentence_transformer_embedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        **kwargs
    )


def create_huggingface_api_embedding(
    model_name: str,
    hf_token: Optional[str] = None,
    **kwargs
) -> HuggingFaceEmbedding:
    """Create HuggingFace API embedding provider."""
    config = HuggingFaceEmbeddingConfig(
        model_name=model_name,
        use_api=True,
        use_local=False,
        hf_token=hf_token,
        **kwargs
    )
    return HuggingFaceEmbedding(config=config)
