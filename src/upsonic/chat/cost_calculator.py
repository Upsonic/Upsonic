"""Cost calculation utilities for Chat class.

This module provides cost calculation using the same logic as the main framework.
Cost data is derived from session usage stored in storage.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
    from upsonic.models import Model
    from upsonic.usage import RequestUsage, RunUsage
    from upsonic.run.agent.output import AgentRunOutput

from upsonic.utils.usage import (
    get_estimated_cost,
    get_estimated_cost_from_usage,
    get_estimated_cost_from_run_output,
    get_model_name,
)


class CostTracker:
    """
    Cost calculation utilities using framework cost calculation logic.
    
    This class provides static methods for cost calculation without
    maintaining local state. All cost data should come from session.usage
    in storage (via SessionManager).
    """
    
    @staticmethod
    def calculate_cost(
        input_tokens: int,
        output_tokens: int,
        model: Optional[Union["Model", str]] = None
    ) -> str:
        """
        Calculate cost using the same logic as printing.py.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model instance or identifier
            
        Returns:
            Formatted cost string (e.g., "~$0.0123")
        """
        return get_estimated_cost(input_tokens, output_tokens, model)
    
    @staticmethod
    def calculate_cost_from_usage(
        usage: "RequestUsage",
        model: Optional[Union["Model", str]] = None
    ) -> str:
        """
        Calculate cost from usage object using printing.py logic.
        
        Args:
            usage: RequestUsage object
            model: Model instance or identifier
            
        Returns:
            Formatted cost string
        """
        return get_estimated_cost_from_usage(usage, model)
    
    @staticmethod
    def calculate_cost_from_agent_run_output(
        agent_run_output: "AgentRunOutput",
        model: Optional[Union["Model", str]] = None
    ) -> str:
        """
        Calculate cost from AgentRunOutput using printing.py logic.
        
        Args:
            agent_run_output: AgentRunOutput object
            model: Model instance or identifier
            
        Returns:
            Formatted cost string
        """
        return get_estimated_cost_from_run_output(agent_run_output, model)
    
    @staticmethod
    def calculate_cost_from_run_usage(
        run_usage: "RunUsage",
        model: Optional[Union["Model", str]] = None
    ) -> str:
        """
        Calculate cost from RunUsage (session-level aggregated usage).
        
        Args:
            run_usage: RunUsage object from session.usage
            model: Model instance or identifier
            
        Returns:
            Formatted cost string
        """
        if run_usage is None:
            return "~$0.0000"
        
        input_tokens = run_usage.input_tokens or 0
        output_tokens = run_usage.output_tokens or 0
        
        return get_estimated_cost(input_tokens, output_tokens, model)
    
    @staticmethod
    def extract_cost_from_string(cost_string: str) -> float:
        """
        Extract numeric cost from formatted cost string.
        
        Args:
            cost_string: Cost string (e.g., "~$0.0123")
            
        Returns:
            Numeric cost value
        """
        try:
            if cost_string.startswith('~$'):
                return float(cost_string[2:])
            elif cost_string.startswith('$'):
                return float(cost_string[1:])
            else:
                return float(cost_string)
        except (ValueError, TypeError):
            return 0.0

    
    @staticmethod
    def get_name(model: Optional[Union["Model", str]]) -> str:
        """
        Extract model name from model provider.
        
        Args:
            model: Model instance or identifier
            
        Returns:
            Model name string
        """
        return get_model_name(model)
    
    @staticmethod
    def get_provider_name(model: Optional["Model"]) -> str:
        """
        Get provider name from model.
        
        Args:
            model: Model instance
            
        Returns:
            Provider name string
        """
        if not model:
            return "unknown"
        
        if hasattr(model, 'provider_name'):
            return model.provider_name
        elif hasattr(model, 'model_name'):
            model_name = model.model_name
            if isinstance(model_name, str):
                if '/' in model_name:
                    return model_name.split('/', 1)[0]
                elif model_name.startswith('gpt-'):
                    return 'openai'
                elif model_name.startswith('claude-'):
                    return 'anthropic'
                elif model_name.startswith('gemini-'):
                    return 'google'
        
        return "unknown"


def format_cost(cost: float, currency: str = "USD") -> str:
    """
    Format cost for display.
    
    Args:
        cost: Cost in USD
        currency: Currency symbol (unused, always USD)
        
    Returns:
        Formatted cost string
    """
    if cost < 0.0001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.5f}"
    else:
        return f"${cost:.4f}"


def format_tokens(tokens: int) -> str:
    """
    Format token count for display.
    
    Args:
        tokens: Number of tokens
        
    Returns:
        Formatted token string
    """
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1000000:
        return f"{tokens/1000:.1f}K"
    else:
        return f"{tokens/1000000:.1f}M"
