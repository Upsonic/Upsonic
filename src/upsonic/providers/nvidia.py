from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from upsonic.utils.package.exception import UserError
from upsonic.models import cached_async_http_client
from upsonic.profiles import ModelProfile
from upsonic.profiles.cohere import cohere_model_profile
from upsonic.profiles.deepseek import deepseek_model_profile
from upsonic.profiles.google import google_model_profile
from upsonic.profiles.meta import meta_model_profile
from upsonic.profiles.mistral import mistral_model_profile
from upsonic.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from upsonic.profiles.qwen import qwen_model_profile
from upsonic.profiles.nvidia import nvidia_model_profile
from upsonic.providers import Provider

try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    AsyncOpenAI = None
    _OPENAI_AVAILABLE = False



class NvidiaProvider(Provider[AsyncOpenAI]):
    """Provider for NVIDIA NIM API.
    
    NVIDIA NIM provides access to various models from different vendors through an OpenAI-compatible API.
    """

    @property
    def name(self) -> str:
        return 'nvidia'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Get the model profile for an NVIDIA NIM model.
        
        NVIDIA NIM hosts models from various vendors. This method maps model names
        to their appropriate vendor profiles based on the model name prefix.
        """
        # Map model prefixes to their respective profile functions
        # Based on NVIDIA NIM API documentation
        prefix_to_profile = {
            'meta/llama': meta_model_profile,
            'meta/codellama': meta_model_profile,
            'google/gemma': google_model_profile,
            'google/codegemma': google_model_profile,
            'google/shieldgemma': google_model_profile,
            'microsoft/phi': lambda model_name: ModelProfile(),  # Microsoft Phi models
            'mistralai/mistral': mistral_model_profile,
            'mistralai/mixtral': mistral_model_profile,
            'mistralai/codestral': mistral_model_profile,
            'mistralai/mathstral': mistral_model_profile,
            'deepseek-ai/deepseek': deepseek_model_profile,
            'qwen/qwen': qwen_model_profile,
            'ai21labs/jamba': lambda model_name: ModelProfile(),  # AI21 Jamba models
            'nvidia/llama': meta_model_profile,  # NVIDIA's Llama variants
            'nvidia/nemotron': nvidia_model_profile,  # NVIDIA's own models
            'ibm/granite': lambda model_name: ModelProfile(),  # IBM Granite models
            'snowflake/arctic': lambda model_name: ModelProfile(),  # Snowflake Arctic
            'upstage/solar': lambda model_name: ModelProfile(),  # Upstage Solar
            'databricks/dbrx': lambda model_name: ModelProfile(),  # Databricks DBRX
            'cohere/command': cohere_model_profile,  # Cohere Command models
        }

        profile = None
        model_name_lower = model_name.lower()
        
        for prefix, profile_func in prefix_to_profile.items():
            if model_name_lower.startswith(prefix):
                # Extract the model name part after the vendor prefix
                model_short_name = model_name[len(prefix.split('/')[0]) + 1:]
                profile = profile_func(model_short_name)
                break
        
        # Special handling for NVIDIA's own models
        if profile is None and model_name_lower.startswith('nvidia/'):
            profile = nvidia_model_profile(model_name)

        # As NvidiaProvider is always used with OpenAIChatModel (OpenAI-compatible API),
        # we need to ensure OpenAIJsonSchemaTransformer is used unless explicitly set
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new NVIDIA NIM provider.

        Args:
            base_url: The base url for the NVIDIA NIM requests. If not provided, defaults to
                'https://integrate.api.nvidia.com/v1'. Can also be set via `NVIDIA_BASE_URL` environment variable.
            api_key: The API key to use for authentication. If not provided, the `NVIDIA_API_KEY` or 
                `NGC_API_KEY` environment variable will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if not _OPENAI_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="openai",
                install_command='pip install openai',
                feature_name="NVIDIA provider"
            )

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            # Default to NVIDIA NIM base URL
            base_url = base_url or os.getenv('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
            
            # Try NVIDIA_API_KEY first, then NGC_API_KEY (NGC is NVIDIA's platform)
            api_key = api_key or os.getenv('NVIDIA_API_KEY') or os.getenv('NGC_API_KEY')
            
            if not api_key:
                raise UserError(
                    'Set the `NVIDIA_API_KEY` or `NGC_API_KEY` environment variable or pass it via '
                    '`NvidiaProvider(api_key=...)` to use the NVIDIA provider.'
                )

            if http_client is not None:
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='nvidia')
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

