from __future__ import annotations as _annotations

import os

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
from upsonic.providers import Provider

try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    AsyncOpenAI = None
    _OPENAI_AVAILABLE = False



class VLLMProvider(Provider[AsyncOpenAI]):
    """Provider for vLLM API - an OpenAI-compatible high-throughput serving engine for LLMs."""

    @property
    def name(self) -> str:
        return 'vllm'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Determine the appropriate model profile based on the model name.
        
        vLLM can serve various model families, so we detect the family from the model name
        and apply the appropriate profile configuration.
        """
        prefix_to_profile = {
            'llama': meta_model_profile,
            'meta-llama': meta_model_profile,
            'gemma': google_model_profile,
            'qwen': qwen_model_profile,
            'qwq': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'command': cohere_model_profile,
        }

        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            model_name_lower = model_name.lower()
            if model_name_lower.startswith(prefix):
                profile = profile_func(model_name)
                break

        # As VLLMProvider is always used with OpenAIChatModel (OpenAI-compatible API),
        # we need to use OpenAIJsonSchemaTransformer for proper JSON schema handling
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new vLLM provider.

        Args:
            base_url: The base url for the vLLM server. If not provided, the `VLLM_BASE_URL` environment variable
                will be used if available. Typically runs on http://localhost:8000/v1 by default.
            api_key: The API key to use for authentication, if not provided, the `VLLM_API_KEY` environment variable
                will be used if available. vLLM server may or may not require authentication depending on configuration.
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
                feature_name="vLLM provider"
            )

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            base_url = base_url or os.getenv('VLLM_BASE_URL')
            if not base_url:
                raise UserError(
                    'Set the `VLLM_BASE_URL` environment variable or pass it via `VLLMProvider(base_url=...)`'
                    'to use the vLLM provider. Example: http://localhost:8000/v1'
                )

            # vLLM server may not require authentication, but OpenAI client requires a non-empty API key
            # So we provide a placeholder if no key is configured
            api_key = api_key or os.getenv('VLLM_API_KEY') or 'api-key-not-set'

            if http_client is not None:
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='vllm')
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

