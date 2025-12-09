from __future__ import annotations as _annotations

from typing import overload
import os

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI

from upsonic.profiles import ModelProfile
from upsonic.models import cached_async_http_client
from upsonic.profiles.amazon import amazon_model_profile
from upsonic.profiles.anthropic import anthropic_model_profile
from upsonic.profiles.cohere import cohere_model_profile
from upsonic.profiles.deepseek import deepseek_model_profile
from upsonic.profiles.google import google_model_profile
from upsonic.profiles.grok import grok_model_profile
from upsonic.profiles.groq import groq_model_profile
from upsonic.profiles.meta import meta_model_profile
from upsonic.profiles.mistral import mistral_model_profile
from upsonic.profiles.moonshotai import moonshotai_model_profile
from upsonic.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from upsonic.profiles.qwen import qwen_model_profile
from upsonic.providers import Provider
from upsonic.utils.package.exception import UserError

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    from upsonic.utils.printing import import_error
    import_error(
        package_name="openai",
        install_command="pip install openai",
        feature_name="LiteLLM provider"
    )


class LiteLLMProvider(Provider[AsyncOpenAI]):
    """Provider for LiteLLM API."""

    @property
    def name(self) -> str:
        return 'litellm'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        # Map provider prefixes to their profile functions
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'openai': openai_model_profile,
            'google': google_model_profile,
            'mistralai': mistral_model_profile,
            'mistral': mistral_model_profile,
            'cohere': cohere_model_profile,
            'amazon': amazon_model_profile,
            'bedrock': amazon_model_profile,
            'meta-llama': meta_model_profile,
            'meta': meta_model_profile,
            'groq': groq_model_profile,
            'deepseek': deepseek_model_profile,
            'moonshotai': moonshotai_model_profile,
            'x-ai': grok_model_profile,
            'qwen': qwen_model_profile,
        }

        profile = None

        # Check if model name contains a provider prefix (e.g., "anthropic/claude-3")
        if '/' in model_name:
            provider_prefix, model_suffix = model_name.split('/', 1)
            if provider_prefix in provider_to_profile:
                profile = provider_to_profile[provider_prefix](model_suffix)

        # If no profile found, default to OpenAI profile
        if profile is None:
            profile = openai_model_profile(model_name)

        # As LiteLLMProvider is used with OpenAIModel, which uses OpenAIJsonSchemaTransformer,
        # we maintain that behavior
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: AsyncHTTPClient,
    ) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Initialize a LiteLLM provider.

        Args:
            api_key: API key for the model provider. If None, LiteLLM will try to get it from environment variables.
            base_url: Base URL for the model provider. Use this for custom endpoints or self-hosted models.
            openai_client: Pre-configured OpenAI client. If provided, other parameters are ignored.
            http_client: Custom HTTP client to use.
        """

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            base_url = base_url or os.getenv('LITELLM_BASE_URL')
            if not base_url:
                raise UserError(
                    'Set the `LITELLM_BASE_URL` environment variable or pass it via `LiteLLMProvider(base_url=...)`'
                    'to use the LiteLLM provider. Example: http://localhost:4000/v1'
                )
        api_key = api_key or os.getenv('LITELLM_API_KEY') or 'litellm-placeholder'
        # Create OpenAI client that will be used with LiteLLM's completion function
        # The actual API calls will be intercepted and routed through LiteLLM
        if http_client is not None:
            self._client = AsyncOpenAI(
                base_url=base_url, api_key=api_key, http_client=http_client
            )
        else:
            http_client = cached_async_http_client(provider='litellm')
            self._client = AsyncOpenAI(
                base_url=base_url, api_key=api_key, http_client=http_client
            )