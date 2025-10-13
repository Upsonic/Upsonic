import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.models import Model
    from upsonic.usage import RequestUsage
    from upsonic.agent.run_result import RunResult, StreamRunResult

from upsonic.utils.printing import (
    get_estimated_cost,
    get_estimated_cost_from_usage,
    get_estimated_cost_from_run_result,
    get_estimated_cost_from_stream_result,
    _get_model_name,
    _get_model_pricing
)


@dataclass
class CostEntry:
    """Individual cost entry for tracking."""
    timestamp: float
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    model_name: str
    provider_name: str
    cost_string: str  # Formatted cost string from printing.py


@dataclass
class CostTracker:
    """
    Comprehensive cost tracker that integrates with printing.py functions.
    
    This class provides accurate cost tracking using the same calculation
    logic as the main framework, ensuring consistency across all components.
    """
    _cost_history: List[CostEntry] = field(default_factory=list)
    
    def add_usage(self, usage: "RequestUsage", model: Optional["Model"] = None) -> None:
        """Add usage information to the cost tracker."""
        if not usage:
            return

        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        
        # Use the same cost calculation as printing.py
        cost_string = get_estimated_cost_from_usage(usage, model)
        
        # Extract numeric cost from the formatted string
        estimated_cost = self._extract_cost_from_string(cost_string)
        
        model_name = self._get_model_name(model)
        provider_name = self._get_provider_name(model)

        self._cost_history.append(
            CostEntry(
                timestamp=time.time(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=estimated_cost,
                model_name=model_name,
                provider_name=provider_name,
                cost_string=cost_string
            )
        )
    
    def add_run_result(self, run_result: "RunResult", model: Optional["Model"] = None) -> None:
        """Add cost information from a RunResult object."""
        if not run_result:
            return
        
        # Use the same cost calculation as printing.py
        cost_string = get_estimated_cost_from_run_result(run_result, model)
        estimated_cost = self._extract_cost_from_string(cost_string)
        
        # Calculate total tokens from the run result
        total_input_tokens = 0
        total_output_tokens = 0
        
        if hasattr(run_result, 'all_messages'):
            messages = run_result.all_messages()
            for message in messages:
                if hasattr(message, 'usage') and message.usage and hasattr(message, 'kind') and message.kind == 'response':
                    usage = message.usage
                    total_input_tokens += usage.input_tokens or 0
                    total_output_tokens += usage.output_tokens or 0
        
        model_name = self._get_model_name(model)
        provider_name = self._get_provider_name(model)

        self._cost_history.append(
            CostEntry(
                timestamp=time.time(),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                estimated_cost=estimated_cost,
                model_name=model_name,
                provider_name=provider_name,
                cost_string=cost_string
            )
        )
    
    def add_stream_result(self, stream_result: "StreamRunResult", model: Optional["Model"] = None) -> None:
        """Add cost information from a StreamRunResult object."""
        if not stream_result:
            return
        
        # Use the same cost calculation as printing.py
        cost_string = get_estimated_cost_from_stream_result(stream_result, model)
        estimated_cost = self._extract_cost_from_string(cost_string)
        
        # Calculate total tokens from the stream result
        total_input_tokens = 0
        total_output_tokens = 0
        
        if hasattr(stream_result, 'all_messages'):
            messages = stream_result.all_messages()
            for message in messages:
                if hasattr(message, 'usage') and message.usage and hasattr(message, 'kind') and message.kind == 'response':
                    usage = message.usage
                    total_input_tokens += usage.input_tokens or 0
                    total_output_tokens += usage.output_tokens or 0
        
        model_name = self._get_model_name(model)
        provider_name = self._get_provider_name(model)

        self._cost_history.append(
            CostEntry(
                timestamp=time.time(),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                estimated_cost=estimated_cost,
                model_name=model_name,
                provider_name=provider_name,
                cost_string=cost_string
            )
        )
    
    def _extract_cost_from_string(self, cost_string: str) -> float:
        """Extract numeric cost from formatted cost string."""
        try:
            # Remove ~$ prefix and convert to float
            if cost_string.startswith('~$'):
                return float(cost_string[2:])
            elif cost_string.startswith('$'):
                return float(cost_string[1:])
            else:
                return float(cost_string)
        except (ValueError, TypeError):
            return 0.0
    
    def _get_model_name(self, model: Optional["Model"]) -> str:
        """Get model name using the same logic as printing.py."""
        return _get_model_name(model)
    
    def _get_provider_name(self, model: Optional["Model"]) -> str:
        """Get provider name from model."""
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
                else:
                    return 'unknown'
        
        return "unknown"
    
    def get_cost_history(self) -> List[Dict[str, Any]]:
        """Get detailed cost history as list of dictionaries."""
        return [
            {
                "timestamp": entry.timestamp,
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "estimated_cost": entry.estimated_cost,
                "model_name": entry.model_name,
                "provider_name": entry.provider_name,
                "cost_string": entry.cost_string
            }
            for entry in self._cost_history
        ]
    
    @property
    def input_tokens(self) -> int:
        """Total input tokens across all entries."""
        return sum(entry.input_tokens for entry in self._cost_history)
    
    @property
    def output_tokens(self) -> int:
        """Total output tokens across all entries."""
        return sum(entry.output_tokens for entry in self._cost_history)
    
    @property
    def total_cost(self) -> float:
        """Total cost across all entries."""
        return sum(entry.estimated_cost for entry in self._cost_history)
    
    def get_formatted_total_cost(self, model: Optional["Model"] = None) -> str:
        """Get formatted total cost string using printing.py logic."""
        return get_estimated_cost(self.input_tokens, self.output_tokens, model)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session cost summary."""
        return {
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "formatted_total_cost": self.get_formatted_total_cost(),
            "entry_count": len(self._cost_history),
            "cost_history": self.get_cost_history()
        }
    
    # Static methods for cost calculation (integrated from CostCalculator)
    @staticmethod
    def calculate_cost(input_tokens: int, output_tokens: int, model: Union["Model", str]) -> str:
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
    def calculate_cost_from_usage(usage: "RequestUsage", model: Union["Model", str]) -> str:
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
    def calculate_cost_from_run_result(run_result: "RunResult", model: Union["Model", str]) -> str:
        """
        Calculate cost from RunResult using printing.py logic.
        
        Args:
            run_result: RunResult object
            model: Model instance or identifier
            
        Returns:
            Formatted cost string
        """
        return get_estimated_cost_from_run_result(run_result, model)
    
    @staticmethod
    def calculate_cost_from_stream_result(stream_result: "StreamRunResult", model: Union["Model", str]) -> str:
        """
        Calculate cost from StreamRunResult using printing.py logic.
        
        Args:
            stream_result: StreamRunResult object
            model: Model instance or identifier
            
        Returns:
            Formatted cost string
        """
        return get_estimated_cost_from_stream_result(stream_result, model)
    
    @staticmethod
    def get_model_pricing(model_name: str) -> Optional[Dict[str, float]]:
        """
        Get pricing information for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with pricing information or None
        """
        return _get_model_pricing(model_name)
    
    @staticmethod
    def get_model_name(model: Union["Model", str]) -> str:
        """
        Extract model name from model provider.
        
        Args:
            model: Model instance or identifier
            
        Returns:
            Model name string
        """
        return _get_model_name(model)




def format_cost(cost: float, currency: str = "USD") -> str:
    """
    Format cost for display.
    
    Args:
        cost: Cost in USD
        currency: Currency symbol
        
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
