"""Shared model selection for smoke tests.

Smoke tests hit real provider APIs, so the model is chosen from whichever
API key is available in the environment (loaded from .env by the root
conftest):

  * ``OPENAI_API_KEY``    -> ``openai-chat/gpt-5``
  * ``ANTHROPIC_API_KEY`` -> ``anthropic/claude-opus-4-6``

OpenAI takes precedence when both are set. Import ``smoke_model()`` and use it
wherever a smoke test needs a chat model, instead of hardcoding a provider/model
string.
"""
from __future__ import annotations

import contextlib
import os

OPENAI_SMOKE_MODEL = "openai-chat/gpt-5"
ANTHROPIC_SMOKE_MODEL = "anthropic/claude-opus-4-6"


@contextlib.contextmanager
def without_model_override():
    """Temporarily disable the global ``LLM_MODEL_KEY`` model override.

    A few tests must resolve a *specific* model (e.g. context-window math that
    depends on a known window, or model-override/span assertions). Those wrap
    their model construction in this context so the smoke-suite global override
    set in ``conftest`` doesn't swap the model out from under them.
    """
    saved = os.environ.pop("LLM_MODEL_KEY", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["LLM_MODEL_KEY"] = saved


def smoke_model() -> str:
    """Return the smoke-test chat model id based on available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        return OPENAI_SMOKE_MODEL
    if os.getenv("ANTHROPIC_API_KEY"):
        return ANTHROPIC_SMOKE_MODEL
    raise RuntimeError(
        "No OPENAI_API_KEY or ANTHROPIC_API_KEY found for smoke tests. "
        "Set one in your environment or .env."
    )
