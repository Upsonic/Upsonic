"""Shared fixtures and helpers for stream_final_answer smoke tests.

Underscore prefix prevents pytest from collecting this as a test module.
Imported by every test_final_answer_marker_*.py file in this directory.
"""

from __future__ import annotations

import os
from typing import List

import pytest
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Provider availability gates
# ---------------------------------------------------------------------------


def has_anthropic() -> bool:
    """Anthropic SDK installed AND API key set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        from anthropic import AsyncAnthropic  # noqa: F401
        return True
    except ImportError:
        return False


def has_google() -> bool:
    """Google Gemini SDK installed AND API key set."""
    if not os.environ.get("GOOGLE_API_KEY"):
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Cached-HTTP-client workaround for Google
# ---------------------------------------------------------------------------
#
# Upsonic's _cached_async_http_client is @functools.cache-decorated at module
# level. Each test gets a fresh pytest-asyncio event loop, but the cache hands
# out the SAME httpx.AsyncClient across event loops. Google's SDK
# (BaseApiClient._async_request_once) is sensitive to closed event loops, so
# subsequent Google tests crash with "RuntimeError: Event loop is closed".
#
# Anthropic's SDK is more tolerant — it tends to retry transparently — so the
# bug surfaces almost exclusively for Google.
#
# Solution: clear the cache before each test that talks to Google.


def clear_cached_http_clients() -> None:
    """Clear the shared httpx.AsyncClient cache so the next test gets a
    fresh client bound to the current event loop.
    """
    try:
        from upsonic.models import _cached_async_http_client  # type: ignore
        _cached_async_http_client.cache_clear()
    except Exception:
        pass


@pytest.fixture(autouse=False)
def _fresh_http_client():
    """Opt-in fixture: clears the cached httpx client before AND after the
    test. Use only for Google tests (Anthropic doesn't need it and the cache
    sharing is actually beneficial for Anthropic sessions).
    """
    clear_cached_http_clients()
    yield
    clear_cached_http_clients()


# ---------------------------------------------------------------------------
# Agent / Chat / Task / Tools — shared imports and constructors
# ---------------------------------------------------------------------------


from upsonic.agent.agent import Agent
from upsonic.chat.chat import Chat
from upsonic.tasks.tasks import Task
from upsonic.tools import tool
from upsonic.tools.framework_tools import FINAL_ANSWER_MARKER_TOOL_NAME

__all__ = [
    "has_anthropic",
    "has_google",
    "clear_cached_http_clients",
    "_fresh_http_client",
    "Agent",
    "Chat",
    "Task",
    "tool",
    "FINAL_ANSWER_MARKER_TOOL_NAME",
    "AnswerSchema",
    "add",
    "multiply",
    "greet",
    "build_agent_anthropic",
    "build_agent_google",
    "build_agent_default",
    "collect_events",
    "skip_if_no_provider",
]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"


class AnswerSchema(BaseModel):
    """A short answer with a numeric result."""

    answer: str = Field(..., description="One-sentence explanation.")
    result: int = Field(..., description="The numeric answer.")


def build_agent_anthropic(**kwargs) -> Agent:
    """Construct an Anthropic-backed Agent. Skips the calling test if the SDK
    or API key is missing.
    """
    if not has_anthropic():
        pytest.skip("Anthropic SDK or key unavailable")
    return Agent(model="anthropic/claude-sonnet-4-6", **kwargs)


def build_agent_google(**kwargs) -> Agent:
    """Construct a Gemini-backed Agent. Skips the calling test if the SDK
    or API key is missing.
    """
    if not has_google():
        pytest.skip("Google Gemini SDK or key unavailable")
    return Agent(model="google-gla/gemini-2.5-flash", **kwargs)


def build_agent_default(**kwargs) -> Agent:
    """Prefer Anthropic; fall back to Google if only Google is available."""
    if has_anthropic():
        return build_agent_anthropic(**kwargs)
    return build_agent_google(**kwargs)


async def collect_events(agent: Agent, task: Task, **kwargs) -> List:
    """Drain the entire astream(events=True) generator into a flat list."""
    out = []
    async for event in agent.astream(task, events=True, **kwargs):
        out.append(event)
    return out


# Module-level guard: skip everything in this directory if no provider is
# available. Each test file re-applies this as `pytestmark`.
skip_if_no_provider = pytest.mark.skipif(
    not (has_anthropic() or has_google()),
    reason="stream_final_answer smoke tests need Anthropic or Google",
)
