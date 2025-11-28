from __future__ import annotations as _annotations

import os
from typing import overload

from httpx import AsyncClient

from upsonic.profiles import ModelProfile
from upsonic.utils.package.exception import UserError
from upsonic.profiles.deepseek import deepseek_model_profile
from upsonic.profiles.google import google_model_profile
from upsonic.profiles.meta import meta_model_profile
from upsonic.profiles.mistral import mistral_model_profile
from upsonic.profiles.moonshotai import moonshotai_model_profile
from upsonic.profiles.qwen import qwen_model_profile

try:
    from huggingface_hub import AsyncInferenceClient
except ImportError:  # pragma: no cover
    from upsonic.utils.printing import import_error
    import_error(
        package_name="huggingface_hub",
        install_command="pip install huggingface_hub",
        feature_name="HuggingFace provider"
    )

from upsonic.providers import Provider


class HuggingFaceProvider(Provider[AsyncInferenceClient]):
    """Provider for Hugging Face."""

    @property
    def name(self) -> str:
        return 'huggingface'

    @property
    def base_url(self) -> str:
        return self.client.model  # type: ignore

    @property
    def client(self) -> AsyncInferenceClient:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'deepseek-ai': deepseek_model_profile,
            'google': google_model_profile,
            'qwen': qwen_model_profile,
            'meta-llama': meta_model_profile,
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
        }

        if '/' not in model_name:
            return None

        model_name = model_name.lower()
        provider, model_name = model_name.split('/', 1)
        if provider in provider_to_profile:
            return provider_to_profile[provider](model_name)

        return None

    @overload
    def __init__(self, *, base_url: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, provider_name: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, base_url: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, provider_name: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, api_key: str | None = None) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        hf_client: AsyncInferenceClient | None = None,
        http_client: AsyncClient | None = None,
        provider_name: str | None = None,
    ) -> None:
        """Create a new Hugging Face provider.

        Args:
            base_url: The base url for the Hugging Face requests.
            api_key: The API key to use for authentication, if not provided, the `HF_TOKEN` environment variable
                will be used if available.
            hf_client: An existing
                [`AsyncInferenceClient`](https://huggingface.co/docs/huggingface_hub/v0.29.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)
                client to use. If not provided, a new instance will be created.
            http_client: (currently ignored) An existing `httpx.AsyncClient` to use for making HTTP requests.
            provider_name : Name of the provider to use for inference. available providers can be found in the [HF Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
                defaults to "auto", which will select the first available provider for the model, the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
                If `base_url` is passed, then `provider_name` is not used.
        """
        api_key = api_key or os.getenv('HF_TOKEN')

        if api_key is None:
            raise UserError(
                'Set the `HF_TOKEN` environment variable or pass it via `HuggingFaceProvider(api_key=...)`'
                'to use the HuggingFace provider.'
            )

        if http_client is not None:
            raise ValueError('`http_client` is ignored for HuggingFace provider, please use `hf_client` instead.')

        if base_url is not None and provider_name is not None:
            raise ValueError('Cannot provide both `base_url` and `provider_name`.')

        if hf_client is None:
            self._client = AsyncInferenceClient(api_key=api_key, provider=provider_name, base_url=base_url)  # type: ignore
        else:
            self._client = hf_client