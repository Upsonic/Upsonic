from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

from upsonic.messages import ModelMessage, ModelResponse
from upsonic.profiles import ModelProfile
from upsonic.providers import Provider
from upsonic.models.settings import ModelSettings
from upsonic.usage import RequestUsage
from upsonic.models import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model


# Attribute names whose writes on a WrapperModel are also forwarded to ``self.wrapped``.
# Bounded set tied to the per-model shaping contract on ``Model``; the guard prevents
# accidental forwarding of unrelated subclass-private state.
_PASSTHROUGH_ATTRS = frozenset({"_settings", "_profile"})


@dataclass(init=False)
class WrapperModel(Model):
    """Model which wraps another model.

    Does nothing on its own, used as a base class.
    """

    wrapped: Model
    """The underlying model being wrapped."""

    def __init__(self, wrapped: Model | KnownModelName):
        super().__init__()
        self.wrapped = infer_model(wrapped)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        return await self.wrapped.count_tokens(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters
        ) as response_stream:
            yield response_stream

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return self.wrapped.customize_request_parameters(model_request_parameters)  # pragma: no cover

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return self.wrapped.prepare_request(model_settings, model_request_parameters)

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def provider(self) -> Optional[Provider[Any]]:
        return getattr(self.wrapped, "provider", None)

    @property
    def system(self) -> str:
        return self.wrapped.system

    @cached_property
    def profile(self) -> ModelProfile:
        return self.wrapped.profile

    @property
    def settings(self) -> ModelSettings | None:
        """Get the settings from the wrapped model."""
        return self.wrapped.settings

    def __setattr__(self, name: str, value: Any) -> None:
        """Mirror every write onto self; additionally forward writes of
        ``_settings``/``_profile`` to ``self.wrapped`` so nested wrapper chains
        mirror the value at every level (terminating at the innermost non-wrapper
        primary). Subclasses needing wrapper-local state can bypass via
        ``object.__setattr__(self, name, value)``.
        """
        super().__setattr__(name, value)
        if name in _PASSTHROUGH_ATTRS and "wrapped" in self.__dict__:
            setattr(self.wrapped, name, value)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)