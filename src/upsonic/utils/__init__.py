from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .async_utils import AsyncExecutionMixin
    from .printing import (
        print_price_id_summary, 
        call_end,
        get_estimated_cost,
        get_estimated_cost_from_usage,
        get_estimated_cost_from_run_result,
        get_estimated_cost_from_stream_result,
        get_estimated_cost_from_agent
    )

def _get_utils_classes():
    """Lazy import of utility classes and functions."""
    from .async_utils import AsyncExecutionMixin
    from .printing import (
        print_price_id_summary, 
        call_end,
        get_estimated_cost,
        get_estimated_cost_from_usage,
        get_estimated_cost_from_run_result,
        get_estimated_cost_from_stream_result,
        get_estimated_cost_from_agent
    )
    
    return {
        'AsyncExecutionMixin': AsyncExecutionMixin,
        'print_price_id_summary': print_price_id_summary,
        'call_end': call_end,
        'get_estimated_cost': get_estimated_cost,
        'get_estimated_cost_from_usage': get_estimated_cost_from_usage,
        'get_estimated_cost_from_run_result': get_estimated_cost_from_run_result,
        'get_estimated_cost_from_stream_result': get_estimated_cost_from_stream_result,
        'get_estimated_cost_from_agent': get_estimated_cost_from_agent,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    utils_classes = _get_utils_classes()
    if name in utils_classes:
        return utils_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "AsyncExecutionMixin",
    "print_price_id_summary",
    "call_end",
    "get_estimated_cost",
    "get_estimated_cost_from_usage",
    "get_estimated_cost_from_run_result",
    "get_estimated_cost_from_stream_result",
    "get_estimated_cost_from_agent",
]