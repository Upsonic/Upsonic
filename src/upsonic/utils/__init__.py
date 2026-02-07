from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .async_utils import AsyncExecutionMixin
    from .printing import (
        print_price_id_summary, 
        call_end,
        get_estimated_cost,
        get_estimated_cost_from_usage,
        get_estimated_cost_from_agent_run_output,
        get_estimated_cost_from_agent
    )
    from .usage import (
        calculate_cost,
        calculate_cost_from_usage,
        calculate_cost_from_run_output,
        calculate_cost_from_agent,
        get_model_name,
        get_model_context_window,
        format_cost,
    )

def _get_utils_classes():
    """Lazy import of utility classes and functions."""
    from .async_utils import AsyncExecutionMixin
    from .printing import (
        print_price_id_summary, 
        call_end,
        get_estimated_cost,
        get_estimated_cost_from_usage,
        get_estimated_cost_from_agent_run_output,
        get_estimated_cost_from_agent
    )
    from .usage import (
        calculate_cost,
        calculate_cost_from_usage,
        calculate_cost_from_run_output,
        calculate_cost_from_agent,
        get_model_name,
        get_model_context_window,
        format_cost,
    )
    
    return {
        'AsyncExecutionMixin': AsyncExecutionMixin,
        'print_price_id_summary': print_price_id_summary,
        'call_end': call_end,
        'get_estimated_cost': get_estimated_cost,
        'get_estimated_cost_from_usage': get_estimated_cost_from_usage,
        'get_estimated_cost_from_agent_run_output': get_estimated_cost_from_agent_run_output,
        'get_estimated_cost_from_agent': get_estimated_cost_from_agent,
        # Core cost calculation functions (return float)
        'calculate_cost': calculate_cost,
        'calculate_cost_from_usage': calculate_cost_from_usage,
        'calculate_cost_from_run_output': calculate_cost_from_run_output,
        'calculate_cost_from_agent': calculate_cost_from_agent,
        # Helper functions
        'get_model_name': get_model_name,
        'get_model_context_window': get_model_context_window,
        'format_cost': format_cost,
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
    # Display functions (return formatted strings)
    "print_price_id_summary",
    "call_end",
    "get_estimated_cost",
    "get_estimated_cost_from_usage",
    "get_estimated_cost_from_agent_run_output",
    "get_estimated_cost_from_agent",
    # Core cost calculation functions (return float)
    "calculate_cost",
    "calculate_cost_from_usage",
    "calculate_cost_from_run_output",
    "calculate_cost_from_agent",
    # Helper functions
    "get_model_name",
    "get_model_context_window",
    "format_cost",
]