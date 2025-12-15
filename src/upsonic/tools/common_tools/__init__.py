"""Common tool implementations for Upsonic agents."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .financial_tools import YFinanceTools
    from .tavily import tavily_search_tool
    from .duckduckgo import duckduckgo_search_tool

def _get_common_tools():
    """Lazy import of common tool classes and functions."""
    from .financial_tools import YFinanceTools
    from .tavily import tavily_search_tool
    from .duckduckgo import duckduckgo_search_tool
    
    return {
        'YFinanceTools': YFinanceTools,
        'tavily_search_tool': tavily_search_tool,
        'duckduckgo_search_tool': duckduckgo_search_tool,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    common_tools = _get_common_tools()
    if name in common_tools:
        return common_tools[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    'YFinanceTools',
    'tavily_search_tool',
    'duckduckgo_search_tool',
]