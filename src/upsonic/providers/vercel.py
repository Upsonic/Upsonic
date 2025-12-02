from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from upsonic.profiles import ModelProfile
from upsonic.utils.package.exception import UserError
from upsonic.models import cached_async_http_client
from upsonic.profiles.amazon import amazon_model_profile
from upsonic.profiles.anthropic import anthropic_model_profile
from upsonic.profiles.cohere import cohere_model_profile
from upsonic.profiles.deepseek import deepseek_model_profile
from upsonic.profiles.google import google_model_profile
from upsonic.profiles.grok import grok_model_profile
from upsonic.profiles.mistral import mistral_model_profile
from upsonic.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from upsonic.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    from upsonic.utils.printing import import_error
    import_error(
        package_name="openai",
        install_command="pip install openai",
        feature_name="Vercel provider"
    )


class VercelProvider(Provider[AsyncOpenAI]):
    """Provider for Vercel AI Gateway API."""

    @property
    def name(self) -> str:
        return 'vercel'

    @property
    def base_url(self) -> str:
        return 'https://ai-gateway.vercel.sh/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'bedrock': amazon_model_profile,
            'cohere': cohere_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'openai': openai_model_profile,
            'vertex': google_model_profile,
            'xai': grok_model_profile,
        }

        profile = None

        try:
            provider, model_name = model_name.split('/', 1)
        except ValueError:
            raise UserError(f"Model name must be in 'provider/model' format, got: {model_name!r}")

        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)

        # As VercelProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
        ).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # Support Vercel AI Gateway's standard environment variables
        api_key = api_key or os.getenv('VERCEL_AI_GATEWAY_API_KEY') or os.getenv('VERCEL_OIDC_TOKEN')

        if not api_key and openai_client is None:
            raise UserError(
                'Set the `VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` environment variable '
                'or pass the API key via `VercelProvider(api_key=...)` to use the Vercel provider.'
            )

        default_headers = {'http-referer': 'https://upsonic.ai/', 'x-title': 'upsonic'}

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
        else:
            http_client = cached_async_http_client(provider='vercel')
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )