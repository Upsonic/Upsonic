"""Upsonic integrations with observability and prompt management platforms."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.integrations.tracing import TracingProvider as TracingProvider
    from upsonic.integrations.tracing import DefaultTracingProvider as DefaultTracingProvider
    from upsonic.integrations.langfuse import Langfuse as Langfuse
    from upsonic.integrations.promptlayer import PromptLayer as PromptLayer
    from upsonic.integrations.asqav import AsqavGovernance as AsqavGovernance


def __getattr__(name: str) -> Any:
    if name == "TracingProvider":
        from upsonic.integrations.tracing import TracingProvider
        return TracingProvider
    if name == "DefaultTracingProvider":
        from upsonic.integrations.tracing import DefaultTracingProvider
        return DefaultTracingProvider
    if name == "Langfuse":
        from upsonic.integrations.langfuse import Langfuse
        return Langfuse
    if name == "PromptLayer":
        from upsonic.integrations.promptlayer import PromptLayer
        return PromptLayer
    if name == "AsqavGovernance":
        from upsonic.integrations.asqav import AsqavGovernance
        return AsqavGovernance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TracingProvider", "DefaultTracingProvider", "Langfuse", "PromptLayer", "AsqavGovernance"]
