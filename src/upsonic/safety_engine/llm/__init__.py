"""
LLM providers for AI Safety Engine - Upsonic Only
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .upsonic_llm import UpsonicLLMProvider

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    if name == "UpsonicLLMProvider":
        from .upsonic_llm import UpsonicLLMProvider
        return UpsonicLLMProvider
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "UpsonicLLMProvider", 
]