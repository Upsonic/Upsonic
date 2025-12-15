from __future__ import annotations as _annotations

from typing import Literal
from typing_extensions import overload

from upsonic.profiles import ModelProfileSpec
from upsonic.providers import Provider
from upsonic.models.settings import ModelSettings
from upsonic.models.openai import OpenAIChatModel, OpenAIModelName

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

__all__ = ('GrokModel',)


class GrokModel(OpenAIChatModel):
    """A model that uses the xAI Grok API.

    This class is a convenience wrapper around `OpenAIChatModel` that sets the provider to `'grok'` by default.
    All functionality is provided by the underlying `OpenAIChatModel` instance.
    """

    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
            'nebius',
            'ovhcloud',
            'gateway',
            'vllm',
            'nvidia',
        ]
        | Provider[AsyncOpenAI] = 'grok',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
            'nebius',
            'ovhcloud',
            'gateway',
            'vllm',
            'nvidia',
        ]
        | Provider[AsyncOpenAI] = 'grok',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Grok model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use. Defaults to `'grok'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        super().__init__(
            model_name=model_name,
            provider=provider,
            profile=profile,
            settings=settings,
        )
