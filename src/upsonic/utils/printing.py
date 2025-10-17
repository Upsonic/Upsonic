from typing import Any, Dict, Literal, Optional, Union, TYPE_CHECKING
from decimal import Decimal

if TYPE_CHECKING:
    from upsonic.models import Model
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
import platform
from rich.markup import escape


console = Console()

def get_estimated_cost(input_tokens: int, output_tokens: int, model: Union["Model", str]) -> str:
    """
    Calculate estimated cost based on tokens and model provider.
    
    This function provides accurate cost estimation for both streaming and non-streaming
    agent executions by using comprehensive pricing data for all supported models.
    
    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens  
        model: Model instance or model name string
        
    Returns:
        Formatted cost string (e.g., "~$0.0123")
    """
    try:
        if input_tokens is None or output_tokens is None:
            return "~$0.0000"
        
        try:
            input_tokens = max(0, int(input_tokens))
            output_tokens = max(0, int(output_tokens))
        except (ValueError, TypeError):
            return "~$0.0000"
        
        try:
            from genai_prices import calculate_cost
            from upsonic.usage import RequestUsage
            
            usage = RequestUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            model_name = _get_model_name(model)
            cost = calculate_cost(usage, model_name)
            return f"~${cost:.4f}"
            
        except ImportError:
            pass
        except Exception:
            pass
        
        model_name = _get_model_name(model)
        pricing_data = _get_model_pricing(model_name)
        
        if not pricing_data:
            pricing_data = {
                'input_cost_per_1m': 0.50,
                'output_cost_per_1m': 1.50
            }
        
        input_cost = (input_tokens / 1_000_000) * pricing_data['input_cost_per_1m']
        output_cost = (output_tokens / 1_000_000) * pricing_data['output_cost_per_1m']
        total_cost = input_cost + output_cost
        
        if total_cost < 0.0001:
            return f"~${total_cost:.6f}"
        elif total_cost < 0.01:
            return f"~${total_cost:.5f}"
        else:
            return f"~${total_cost:.4f}"
        
    except Exception as e:
        console.print(f"[yellow]Warning: Cost calculation failed: {e}[/yellow]")
        return "~$0.0000"


def _get_model_name(model: Union["Model", str]) -> str:
    """Extract model name from model provider."""
    if isinstance(model, str):
        if '/' in model:
            return model.split('/', 1)[1]
        return model
    elif hasattr(model, 'model_name'):
        model_name = model.model_name
        # Handle case where model_name might be a coroutine (in tests)
        if hasattr(model_name, '__await__'):
            return "test-model"  # Default for async mocks
        return model_name
    else:
        return str(model)


def _get_model_pricing(model_name: str) -> Optional[Dict[str, float]]:
    """Get comprehensive pricing data for a model."""
    # Handle case where model_name might be a coroutine (in tests)
    if hasattr(model_name, '__await__'):
        model_name = "test-model"
    
    # Ensure model_name is a string
    model_name = str(model_name)
    
    pricing_map = {
        'gpt-4o': {'input_cost_per_1m': 2.50, 'output_cost_per_1m': 10.00},
        'gpt-4o-2024-05-13': {'input_cost_per_1m': 2.50, 'output_cost_per_1m': 10.00},
        'gpt-4o-2024-08-06': {'input_cost_per_1m': 2.50, 'output_cost_per_1m': 10.00},
        'gpt-4o-2024-11-20': {'input_cost_per_1m': 2.50, 'output_cost_per_1m': 10.00},
        'gpt-4o-mini': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.60},
        'gpt-4o-mini-2024-07-18': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.60},
        'gpt-4-turbo': {'input_cost_per_1m': 10.00, 'output_cost_per_1m': 30.00},
        'gpt-4-turbo-2024-04-09': {'input_cost_per_1m': 10.00, 'output_cost_per_1m': 30.00},
        'gpt-4': {'input_cost_per_1m': 30.00, 'output_cost_per_1m': 60.00},
        'gpt-4-0613': {'input_cost_per_1m': 30.00, 'output_cost_per_1m': 60.00},
        'gpt-4-32k': {'input_cost_per_1m': 60.00, 'output_cost_per_1m': 120.00},
        'gpt-4-32k-0613': {'input_cost_per_1m': 60.00, 'output_cost_per_1m': 120.00},
        'gpt-3.5-turbo': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 1.50},
        'gpt-3.5-turbo-1106': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 1.50},
        'gpt-3.5-turbo-16k': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 4.00},
        'gpt-3.5-turbo-16k-0613': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 4.00},
        'gpt-5': {'input_cost_per_1m': 5.00, 'output_cost_per_1m': 15.00},
        'gpt-5-2025-08-07': {'input_cost_per_1m': 5.00, 'output_cost_per_1m': 15.00},
        'gpt-5-mini': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 1.20},
        'gpt-5-mini-2025-08-07': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 1.20},
        'gpt-5-nano': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.40},
        'gpt-5-nano-2025-08-07': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.40},
        'gpt-4.1': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 12.00},
        'gpt-4.1-2025-04-14': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 12.00},
        'gpt-4.1-mini': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.80},
        'gpt-4.1-mini-2025-04-14': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.80},
        'gpt-4.1-nano': {'input_cost_per_1m': 0.08, 'output_cost_per_1m': 0.32},
        'gpt-4.1-nano-2025-04-14': {'input_cost_per_1m': 0.08, 'output_cost_per_1m': 0.32},
        'o1': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 60.00},
        'o1-2024-12-17': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 60.00},
        'o1-mini': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 12.00},
        'o1-mini-2024-09-12': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 12.00},
        'o1-preview': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 60.00},
        'o1-preview-2024-09-12': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 60.00},
        'o1-pro': {'input_cost_per_1m': 60.00, 'output_cost_per_1m': 180.00},
        'o1-pro-2025-03-19': {'input_cost_per_1m': 60.00, 'output_cost_per_1m': 180.00},
        'o3': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 80.00},
        'o3-2025-04-16': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 80.00},
        'o3-mini': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 16.00},
        'o3-mini-2025-01-31': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 16.00},
        'o3-pro': {'input_cost_per_1m': 80.00, 'output_cost_per_1m': 240.00},
        'o3-pro-2025-06-10': {'input_cost_per_1m': 80.00, 'output_cost_per_1m': 240.00},
        'o3-deep-research': {'input_cost_per_1m': 100.00, 'output_cost_per_1m': 300.00},
        'o3-deep-research-2025-06-26': {'input_cost_per_1m': 100.00, 'output_cost_per_1m': 300.00},
        'claude-3-5-sonnet-20241022': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
        'claude-3-5-sonnet-latest': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
        'claude-3-5-sonnet-20240620': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
        'claude-3-5-haiku-20241022': {'input_cost_per_1m': 0.80, 'output_cost_per_1m': 4.00},
        'claude-3-5-haiku-latest': {'input_cost_per_1m': 0.80, 'output_cost_per_1m': 4.00},
        'claude-3-7-sonnet-20250219': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
        'claude-3-7-sonnet-latest': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
        'claude-3-opus-20240229': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 75.00},
        'claude-3-opus-latest': {'input_cost_per_1m': 15.00, 'output_cost_per_1m': 75.00},
        'claude-3-haiku-20240307': {'input_cost_per_1m': 0.25, 'output_cost_per_1m': 1.25},
        'claude-4-opus-20250514': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 100.00},
        'claude-4-sonnet-20250514': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 20.00},
        'claude-opus-4-0': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 100.00},
        'claude-opus-4-1-20250805': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 100.00},
        'claude-opus-4-20250514': {'input_cost_per_1m': 20.00, 'output_cost_per_1m': 100.00},
        'claude-sonnet-4-0': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 20.00},
        'claude-sonnet-4-20250514': {'input_cost_per_1m': 4.00, 'output_cost_per_1m': 20.00},
        
        'gemini-2.0-flash': {'input_cost_per_1m': 0.075, 'output_cost_per_1m': 0.30},
        'gemini-2.0-flash-lite': {'input_cost_per_1m': 0.0375, 'output_cost_per_1m': 0.15},
        'gemini-2.5-flash': {'input_cost_per_1m': 0.075, 'output_cost_per_1m': 0.30},
        'gemini-2.5-flash-lite': {'input_cost_per_1m': 0.0375, 'output_cost_per_1m': 0.15},
        'gemini-2.5-pro': {'input_cost_per_1m': 1.25, 'output_cost_per_1m': 5.00},
        'gemini-1.5-pro': {'input_cost_per_1m': 1.25, 'output_cost_per_1m': 5.00},
        'gemini-1.5-flash': {'input_cost_per_1m': 0.075, 'output_cost_per_1m': 0.30},
        'gemini-1.0-pro': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 1.50},
        
        'llama-3.3-70b-versatile': {'input_cost_per_1m': 0.59, 'output_cost_per_1m': 0.79},
        'llama-3.1-8b-instant': {'input_cost_per_1m': 0.05, 'output_cost_per_1m': 0.05},
        'llama3-70b-8192': {'input_cost_per_1m': 0.59, 'output_cost_per_1m': 0.79},
        'llama3-8b-8192': {'input_cost_per_1m': 0.05, 'output_cost_per_1m': 0.05},
        'mixtral-8x7b-32768': {'input_cost_per_1m': 0.24, 'output_cost_per_1m': 0.24},
        'gemma2-9b-it': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
        
        'mistral-large-latest': {'input_cost_per_1m': 2.00, 'output_cost_per_1m': 6.00},
        'mistral-small-latest': {'input_cost_per_1m': 1.00, 'output_cost_per_1m': 3.00},
        'codestral-latest': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.20},
        
        'command': {'input_cost_per_1m': 1.00, 'output_cost_per_1m': 2.00},
        'command-light': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 0.30},
        'command-r': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 1.50},
        'command-r-plus': {'input_cost_per_1m': 3.00, 'output_cost_per_1m': 15.00},
        
        'deepseek-chat': {'input_cost_per_1m': 0.14, 'output_cost_per_1m': 0.28},
        'deepseek-reasoner': {'input_cost_per_1m': 0.55, 'output_cost_per_1m': 2.19},
        
        'grok-4': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
        'grok-4-0709': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
        'grok-3': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
        'grok-3-mini': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
        'grok-3-fast': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
        'grok-3-mini-fast': {'input_cost_per_1m': 0.01, 'output_cost_per_1m': 0.03},
        
        'moonshot-v1-8k': {'input_cost_per_1m': 0.012, 'output_cost_per_1m': 0.012},
        'moonshot-v1-32k': {'input_cost_per_1m': 0.024, 'output_cost_per_1m': 0.024},
        'moonshot-v1-128k': {'input_cost_per_1m': 0.06, 'output_cost_per_1m': 0.06},
        'kimi-latest': {'input_cost_per_1m': 0.012, 'output_cost_per_1m': 0.012},
        'kimi-thinking-preview': {'input_cost_per_1m': 0.012, 'output_cost_per_1m': 0.012},
        
        'gpt-oss-120b': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
        'llama3.1-8b': {'input_cost_per_1m': 0.05, 'output_cost_per_1m': 0.05},
        'llama-3.3-70b': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.20},
        'llama-4-scout-17b-16e-instruct': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.15},
        'llama-4-maverick-17b-128e-instruct': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.15},
        'qwen-3-235b-a22b-instruct-2507': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 0.30},
        'qwen-3-32b': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
        'qwen-3-coder-480b': {'input_cost_per_1m': 0.50, 'output_cost_per_1m': 0.50},
        'qwen-3-235b-a22b-thinking-2507': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 0.30},
        
        'Qwen/QwQ-32B': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
        'Qwen/Qwen2.5-72B-Instruct': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.20},
        'Qwen/Qwen3-235B-A22B': {'input_cost_per_1m': 0.30, 'output_cost_per_1m': 0.30},
        'Qwen/Qwen3-32B': {'input_cost_per_1m': 0.10, 'output_cost_per_1m': 0.10},
        'deepseek-ai/DeepSeek-R1': {'input_cost_per_1m': 0.55, 'output_cost_per_1m': 2.19},
        'meta-llama/Llama-3.3-70B-Instruct': {'input_cost_per_1m': 0.20, 'output_cost_per_1m': 0.20},
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.15},
        'meta-llama/Llama-4-Scout-17B-16E-Instruct': {'input_cost_per_1m': 0.15, 'output_cost_per_1m': 0.15},
        
        'test': {'input_cost_per_1m': 0.00, 'output_cost_per_1m': 0.00},
    }
    
    if model_name.startswith('bedrock:'):
        model_name = model_name.replace('bedrock:', '')
    
    provider_prefixes = ['anthropic:', 'google-gla:', 'google-vertex:', 'groq:', 'mistral:', 'cohere:', 'deepseek:', 'grok:', 'moonshotai:', 'cerebras:', 'huggingface:', 'heroku:']
    for prefix in provider_prefixes:
        if model_name.startswith(prefix):
            model_name = model_name.replace(prefix, '')
            break
    
    return pricing_map.get(model_name)


def get_estimated_cost_from_usage(usage: Union[Dict[str, int], Any], model: Union["Model", str]) -> str:
    """Calculate estimated cost from usage data."""
    try:
        if isinstance(usage, dict):
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        else:
            # RequestUsage objects have input_tokens and output_tokens attributes
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
        
        return get_estimated_cost(input_tokens, output_tokens, model)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Cost calculation from usage failed: {e}[/yellow]")
        return "~$0.0000"


def get_estimated_cost_from_run_result(run_result: Any, model: Union["Model", str]) -> str:
    """Calculate estimated cost from a RunResult object."""
    try:
        total_input_tokens = 0
        total_output_tokens = 0
        
        if hasattr(run_result, 'all_messages'):
            messages = run_result.all_messages()
            for message in messages:
                # Only ModelResponse objects have usage information
                if hasattr(message, 'usage') and message.usage and hasattr(message, 'kind') and message.kind == 'response':
                    usage = message.usage
                    total_input_tokens += usage.input_tokens
                    total_output_tokens += usage.output_tokens
        
        return get_estimated_cost(total_input_tokens, total_output_tokens, model)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Cost calculation from RunResult failed: {e}[/yellow]")
        return "~$0.0000"


def get_estimated_cost_from_stream_result(stream_result: Any, model: Union["Model", str]) -> str:
    """Calculate estimated cost from a StreamRunResult object."""
    try:
        total_input_tokens = 0
        total_output_tokens = 0
        
        if hasattr(stream_result, 'all_messages'):
            messages = stream_result.all_messages()
            for message in messages:
                # Only ModelResponse objects have usage information
                if hasattr(message, 'usage') and message.usage and hasattr(message, 'kind') and message.kind == 'response':
                    usage = message.usage
                    total_input_tokens += usage.input_tokens
                    total_output_tokens += usage.output_tokens
        
        return get_estimated_cost(total_input_tokens, total_output_tokens, model)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Cost calculation from StreamRunResult failed: {e}[/yellow]")
        return "~$0.0000"


def get_estimated_cost_from_agent(agent: Any, run_type: str = "last") -> str:
    """Calculate estimated cost from an Agent's run results."""
    try:
        if run_type in ["last", "non_stream"]:
            if hasattr(agent, 'get_run_result'):
                run_result = agent.get_run_result()
                if run_result and hasattr(run_result, 'all_messages') and run_result.all_messages():
                    return get_estimated_cost_from_run_result(run_result, agent.model)
        
        if run_type in ["last", "stream"]:
            if hasattr(agent, 'get_stream_run_result'):
                stream_result = agent.get_stream_run_result()
                if stream_result and hasattr(stream_result, 'all_messages') and stream_result.all_messages():
                    return get_estimated_cost_from_stream_result(stream_result, agent.model)
        
        return "~$0.0000"
        
    except Exception as e:
        console.print(f"[yellow]Warning: Cost calculation from Agent failed: {e}[/yellow]")
        return "~$0.0000"


price_id_summary = {}

def spacing():
    console.print("")


def escape_rich_markup(text):
    """Escape special characters in text to prevent Rich markup interpretation"""
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    return escape(text)


def connected_to_server(server_type: str, status: str, total_time: float = None):
    """
    Prints a 'Connected to Server' section for Upsonic, full width,
    with two columns: 
      - left column (labels) left-aligned
      - right column (values) left-aligned, positioned on the right half 
    """

    server_type = escape_rich_markup(server_type)

    if status.lower() == "established":
        status_text = "[green]✓ Established[/green]"
    elif status.lower() == "failed":
        status_text = "[red]✗ Failed[/red]"
    else:
        status_text = f"[cyan]… {escape_rich_markup(status)}[/cyan]"

    table = Table(show_header=False, expand=True, box=None)
    
    table.add_column("Label", justify="left", ratio=1)
    table.add_column("Value", justify="left", ratio=1)

    table.add_row("[bold]Server Type:[/bold]", f"[yellow]{server_type}[/yellow]")
    table.add_row("[bold]Connection Status:[/bold]", status_text)
    
    if total_time is not None:
        table.add_row("[bold]Total Time:[/bold]", f"[cyan]{total_time:.2f} seconds[/cyan]")

    table.width = 60

    panel = Panel(
        table, 
        title="[bold cyan]Upsonic - Server Connection[/bold cyan]",
        border_style="cyan",
        expand=True,  # panel takes the full terminal width
        width=70  # Adjust as preferred
    )

    console.print(panel)

    spacing()

def call_end(result: Any, model: Any, response_format: str, start_time: float, end_time: float, usage: dict, tool_usage: list, debug: bool = False, price_id: str = None):
    if tool_usage and len(tool_usage) > 0:
        tool_table = Table(show_header=True, expand=True, box=None)
        tool_table.width = 60
        
        tool_table.add_column("[bold]Tool Name[/bold]", justify="left")
        tool_table.add_column("[bold]Parameters[/bold]", justify="left")
        tool_table.add_column("[bold]Result[/bold]", justify="left")

        for tool in tool_usage:
            tool_name = escape_rich_markup(str(tool.get('tool_name', '')))
            params = escape_rich_markup(str(tool.get('params', '')))
            result_str = escape_rich_markup(str(tool.get('tool_result', '')))
            
            if len(params) > 50:
                params = params[:47] + "..."
            if len(result_str) > 50:
                result_str = result_str[:47] + "..."
                
            tool_table.add_row(
                f"[cyan]{tool_name}[/cyan]",
                f"[yellow]{params}[/yellow]",
                f"[green]{result_str}[/green]"
            )

        tool_panel = Panel(
            tool_table,
            title=f"[bold cyan]Tool Usage Summary ({len(tool_usage)} tools)[/bold cyan]",
            border_style="cyan",
            expand=True,
            width=70
        )

        console.print(tool_panel)
        spacing()

    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    display_model_name = escape_rich_markup(model.model_name)
    response_format = escape_rich_markup(response_format)
    price_id_display = escape_rich_markup(price_id) if price_id else None

    if price_id:
        estimated_cost = get_estimated_cost(usage['input_tokens'], usage['output_tokens'], model)
        if price_id not in price_id_summary:
            price_id_summary[price_id] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'estimated_cost': 0.0
            }
        price_id_summary[price_id]['input_tokens'] += usage['input_tokens']
        price_id_summary[price_id]['output_tokens'] += usage['output_tokens']
        try:
            cost_str = str(estimated_cost).replace('~', '').replace('$', '').strip()
            if isinstance(price_id_summary[price_id]['estimated_cost'], (float, int)):
                price_id_summary[price_id]['estimated_cost'] += float(cost_str)
            else:
                from decimal import Decimal
                price_id_summary[price_id]['estimated_cost'] = Decimal(str(price_id_summary[price_id]['estimated_cost'])) + Decimal(cost_str)
        except Exception as e:
            if debug:
                pass  # Error calculating cost

    result_str = str(result)
    if not debug:
        result_str = result_str[:370]
    if len(result_str) < len(str(result)):
        result_str += "..."

    table.add_row("[bold]Result:[/bold]", f"[green]{escape_rich_markup(result_str)}[/green]")
    panel = Panel(
        table,
        title="[bold white]Task Result[/bold white]",
        border_style="white",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()



def agent_end(result: Any, model: Any, response_format: str, start_time: float, end_time: float, usage: dict, tool_usage: list, tool_count: int, context_count: int, debug: bool = False, price_id:str = None):
    if tool_usage and len(tool_usage) > 0:
        tool_table = Table(show_header=True, expand=True, box=None)
        tool_table.width = 60
        
        tool_table.add_column("[bold]Tool Name[/bold]", justify="left")
        tool_table.add_column("[bold]Parameters[/bold]", justify="left")
        tool_table.add_column("[bold]Result[/bold]", justify="left")

        for tool in tool_usage:
            tool_name = escape_rich_markup(str(tool.get('tool_name', '')))
            params = escape_rich_markup(str(tool.get('params', '')))
            result_str = escape_rich_markup(str(tool.get('tool_result', '')))
            
            if len(params) > 50:
                params = params[:47] + "..."
            if len(result_str) > 50:
                result_str = result_str[:47] + "..."
                
            tool_table.add_row(
                f"[cyan]{tool_name}[/cyan]",
                f"[yellow]{params}[/yellow]",
                f"[green]{result_str}[/green]"
            )

        tool_panel = Panel(
            tool_table,
            title=f"[bold cyan]Tool Usage Summary ({len(tool_usage)} tools)[/bold cyan]",
            border_style="cyan",
            expand=True,
            width=70
        )

        console.print(tool_panel)
        spacing()

    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    display_model_name = escape_rich_markup(model.model_name)
    response_format = escape_rich_markup(response_format)
    price_id = escape_rich_markup(price_id) if price_id else None

    if price_id:
        estimated_cost = get_estimated_cost(usage['input_tokens'], usage['output_tokens'], model)
        if price_id not in price_id_summary:
            price_id_summary[price_id] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'estimated_cost': 0.0
            }
        price_id_summary[price_id]['input_tokens'] += usage['input_tokens']
        price_id_summary[price_id]['output_tokens'] += usage['output_tokens']
        try:
            cost_str = str(estimated_cost).replace('~', '').replace('$', '').strip()
            if isinstance(price_id_summary[price_id]['estimated_cost'], (float, int)):
                price_id_summary[price_id]['estimated_cost'] += float(cost_str)
            else:
                price_id_summary[price_id]['estimated_cost'] = Decimal(str(price_id_summary[price_id]['estimated_cost'])) + Decimal(cost_str)
        except Exception as e:
            console.print(f"[bold red]Warning: Could not parse cost value: {estimated_cost}. Error: {e}[/bold red]")

    table.add_row("[bold]LLM Model:[/bold]", f"{display_model_name}")
    table.add_row("")
    result_str = str(result)
    if not debug:
        result_str = result_str[:370]
    if len(result_str) < len(str(result)):
        result_str += "..."

    table.add_row("[bold]Result:[/bold]", f"[green]{escape_rich_markup(result_str)}[/green]")
    table.add_row("")
    table.add_row("[bold]Response Format:[/bold]", f"{response_format}")
    
    table.add_row("[bold]Tools:[/bold]", f"{tool_count} [bold]Context Used:[/bold]", f"{context_count}")
    table.add_row("[bold]Estimated Cost:[/bold]", f"{get_estimated_cost(usage['input_tokens'], usage['output_tokens'], model)}$")
    time_taken = end_time - start_time
    time_taken_str = f"{time_taken:.2f} seconds"
    table.add_row("[bold]Time Taken:[/bold]", f"{time_taken_str}")
    panel = Panel(
        table,
        title="[bold white]Upsonic - Agent Result[/bold white]",
        border_style="white",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def agent_total_cost(total_input_tokens: int, total_output_tokens: int, total_time: float, model: Any):
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    llm_model = escape_rich_markup(model.model_name)

    table.add_row("[bold]Estimated Cost:[/bold]", f"{get_estimated_cost(total_input_tokens, total_output_tokens, model)}$")
    table.add_row("[bold]Time Taken:[/bold]", f"{total_time:.2f} seconds")
    panel = Panel(
        table,
        title="[bold white]Upsonic - Agent Total Cost[/bold white]",
        border_style="white",
        expand=True,
        width=70
    )
    console.print(panel)
    spacing()

def print_price_id_summary(price_id: str, task) -> dict:
    """
    Get the summary of usage and costs for a specific price ID and print it in a formatted panel.
    
    Args:
        price_id (str): The price ID to look up
        task: The task object containing timing information
        
    Returns:
        dict: A dictionary containing the usage summary, or None if price_id not found
    """
    price_id_display = escape_rich_markup(price_id)
    task_display = escape_rich_markup(str(task))
    
    if price_id not in price_id_summary:
        console.print("[bold red]Price ID not found![/bold red]")
        return None
    
    summary = price_id_summary[price_id].copy()
    summary['estimated_cost'] = f"${summary['estimated_cost']:.4f}"

    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Price ID:[/bold]", f"[magenta]{price_id_display}[/magenta]")
    table.add_row("")  # Add spacing
    table.add_row("[bold]Input Tokens:[/bold]", f"[magenta]{summary['input_tokens']:,}[/magenta]")
    table.add_row("[bold]Output Tokens:[/bold]", f"[magenta]{summary['output_tokens']:,}[/magenta]")
    table.add_row("[bold]Total Estimated Cost:[/bold]", f"[magenta]{summary['estimated_cost']}[/magenta]")
    
    if task and hasattr(task, 'duration') and task.duration is not None:
        time_str = f"{task.duration:.2f} seconds"
        table.add_row("[bold]Time Taken:[/bold]", f"[magenta]{time_str}[/magenta]")

    panel = Panel(
        table,
        title="[bold magenta]Task Metrics[/bold magenta]",
        border_style="magenta",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()

    return summary

def agent_retry(retry_count: int, max_retries: int):
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Retry Status:[/bold]", f"[yellow]Attempt {retry_count + 1} of {max_retries + 1}[/yellow]")
    
    panel = Panel(
        table,
        title="[bold yellow]Upsonic - Agent Retry[/bold yellow]",
        border_style="yellow",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()

def call_retry(retry_count: int, max_retries: int):
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Retry Status:[/bold]", f"[yellow]Attempt {retry_count + 1} of {max_retries + 1}[/yellow]")
    
    panel = Panel(
        table,
        title="[bold yellow]Upsonic - Call Retry[/bold yellow]",
        border_style="yellow",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()

def get_price_id_total_cost(price_id: str):
    """
    Get the total cost for a specific price ID.
    
    Args:
        price_id (str): The price ID to get totals for
        
    Returns:
        dict: Dictionary containing input tokens, output tokens, and estimated cost for the price ID.
        None: If the price ID is not found.
    """
    if price_id not in price_id_summary:
        return None

    data = price_id_summary[price_id]
    return {
        'input_tokens': data['input_tokens'],
        'output_tokens': data['output_tokens'],
        'estimated_cost': float(data['estimated_cost'])
    }

def mcp_tool_operation(operation: str, result=None):
    """
    Prints a formatted panel for MCP tool operations.
    
    Args:
        operation: The operation being performed (e.g., "Adding", "Added", "Removing")
        result: The result of the operation, if available
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    operation_text = f"[bold cyan]{escape_rich_markup(operation)}[/bold cyan]"
    table.add_row(operation_text)
    
    if result:
        result_str = str(result)
        table.add_row("")  # Add spacing
        table.add_row(f"[green]{escape_rich_markup(result_str)}[/green]")
    
    panel = Panel(
        table,
        title="[bold cyan]Upsonic - MCP Tool Operation[/bold cyan]",
        border_style="cyan",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def error_message(error_type: str, detail: str, error_code: int = None):
    """
    Prints a formatted error panel for API and service errors.
    
    Args:
        error_type: The type of error (e.g., "API Key Error", "Call Error")
        detail: Detailed error message
        error_code: Optional HTTP status code
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    if error_code:
        table.add_row("[bold]Error Code:[/bold]", f"[red]{error_code}[/red]")
        table.add_row("")  # Add spacing
    
    table.add_row("[bold]Error Details:[/bold]")
    table.add_row(f"[red]{escape_rich_markup(detail)}[/red]")
    
    panel = Panel(
        table,
        title=f"[bold red]Upsonic - {escape_rich_markup(error_type)}[/bold red]",
        border_style="red",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def missing_dependencies(tool_name: str, missing_deps: list):
    """
    Prints a formatted panel with missing dependencies and installation instructions.
    
    Args:
        tool_name: Name of the tool with missing dependencies
        missing_deps: List of missing dependency names
    """
    if not missing_deps:
        return
    
    tool_name = escape_rich_markup(tool_name)
    missing_deps = [escape_rich_markup(dep) for dep in missing_deps]
    
    install_cmd = "pip install " + " ".join(missing_deps)
    
    deps_list = "\n".join([f"  • [bold white]{dep}[/bold white]" for dep in missing_deps])
    
    content = f"[bold red]Missing Dependencies for {tool_name}:[/bold red]\n\n{deps_list}\n\n[bold green]Installation Command:[/bold green]\n  {install_cmd}"
    
    panel = Panel(content, title="[bold yellow]⚠️ Dependencies Required[/bold yellow]", border_style="yellow", expand=False)
    console.print(panel)

def missing_api_key(tool_name: str, env_var_name: str, dotenv_support: bool = True):
    """
    Prints a formatted panel with information about a missing API key and how to set it.
    
    Args:
        tool_name: Name of the tool requiring the API key
        env_var_name: Name of the environment variable for the API key
        dotenv_support: Whether the tool supports loading from .env file
    """
    tool_name = escape_rich_markup(tool_name)
    env_var_name = escape_rich_markup(env_var_name)
    
    system = platform.system()
    
    if system == "Windows":
        env_instructions = f"setx {env_var_name} your_api_key_here"
        env_instructions_temp = f"set {env_var_name}=your_api_key_here"
        env_description = f"[bold green]Option 1: Set environment variable (Windows):[/bold green]\n  • Permanent (new sessions): {env_instructions}\n  • Current session only: {env_instructions_temp}"
    else:  # macOS or Linux
        env_instructions_export = f"export {env_var_name}=your_api_key_here"
        env_instructions_profile = f"echo 'export {env_var_name}=your_api_key_here' >> ~/.bashrc  # or ~/.zshrc"
        env_description = f"[bold green]Option 1: Set environment variable (macOS/Linux):[/bold green]\n  • Current session: {env_instructions_export}\n  • Permanent: {env_instructions_profile}"
    
    if dotenv_support:
        dotenv_instructions = f"Create a .env file in your project directory with:\n  {env_var_name}=your_api_key_here"
        content = f"[bold red]Missing API Key for {tool_name}[/bold red]\n\n[bold white]The {env_var_name} environment variable is not set.[/bold white]\n\n{env_description}\n\n[bold green]Option 2: Use a .env file:[/bold green]\n  {dotenv_instructions}"
    else:
        content = f"[bold red]Missing API Key for {tool_name}[/bold red]\n\n[bold white]The {env_var_name} environment variable is not set.[/bold white]\n\n{env_description}"
    
    panel = Panel(content, title="[bold yellow]🔑 API Key Required[/bold yellow]", border_style="yellow", expand=False)
    console.print(panel)

def tool_operation(operation: str, result=None):
    """
    Prints a formatted panel for regular tool operations.
    
    Args:
        operation: The operation being performed (e.g., "Adding", "Added", "Removing")
        result: The result of the operation, if available
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    operation_text = f"[bold magenta]{escape_rich_markup(operation)}[/bold magenta]"
    table.add_row(operation_text)
    
    if result:
        result_str = str(result)
        table.add_row("")  # Add spacing
        table.add_row(f"[green]{escape_rich_markup(result_str)}[/green]")
    
    panel = Panel(
        table,
        title="[bold magenta]Upsonic - Tool Operation[/bold magenta]",
        border_style="magenta",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def print_orchestrator_tool_step(tool_name: str, params: dict, result: Any):
    """
    Prints a formatted panel for a single tool step executed by the orchestrator.
    This creates the "Tool Usage Summary"-style block for intermediate steps.
    """
    tool_table = Table(show_header=True, expand=True, box=None)
    tool_table.width = 70

    tool_table.add_column("[bold]Tool Name[/bold]", justify="left")
    tool_table.add_column("[bold]Parameters[/bold]", justify="left")
    tool_table.add_column("[bold]Result[/bold]", justify="left")

    tool_name_str = escape_rich_markup(str(tool_name))
    params_str = escape_rich_markup(str(params))
    result_str = escape_rich_markup(str(result))
    
    if len(params_str) > 50:
        params_str = params_str[:47] + "..."
    if len(result_str) > 50:
        result_str = result_str[:47] + "..."
            
    tool_table.add_row(
        f"[cyan]{tool_name_str}[/cyan]",
        f"[yellow]{params_str}[/yellow]",
        f"[green]{result_str}[/green]"
    )

    tool_panel = Panel(
        tool_table,
        title=f"[bold cyan]Orchestrator - Tool Call Result[/bold cyan]",
        border_style="cyan",
        expand=True,
        width=70
    )

    console.print(tool_panel)
    spacing()


def policy_triggered(policy_name: str, check_type: str, action_taken: str, rule_output: Any):
    """
    Prints a formatted panel when a Safety Engine policy is triggered.
    """
    
    if "BLOCK" in action_taken.upper() or "DISALLOWED" in action_taken.upper():
        border_style = "bold red"
        title = f"[bold red]🛡️ Safety Policy Triggered: ACCESS DENIED[/bold red]"
    elif "REPLACE" in action_taken.upper() or "ANONYMIZE" in action_taken.upper():
        border_style = "bold yellow"
        title = f"[bold yellow]🛡️ Safety Policy Triggered: CONTENT MODIFIED[/bold yellow]"
    else:
        border_style = "bold green"
        title = f"[bold green]🛡️ Safety Policy Check: PASSED[/bold green]"

    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    policy_name_esc = escape_rich_markup(policy_name)
    check_type_esc = escape_rich_markup(check_type)
    action_taken_esc = escape_rich_markup(action_taken)
    details_esc = escape_rich_markup(rule_output.details)
    content_type_esc = escape_rich_markup(rule_output.content_type)
    
    table.add_row("[bold]Policy Name:[/bold]", f"[cyan]{policy_name_esc}[/cyan]")
    table.add_row("[bold]Check Point:[/bold]", f"[cyan]{check_type_esc}[/cyan]")
    table.add_row("")
    table.add_row("[bold]Action Taken:[/bold]", f"[{border_style.split(' ')[1]}]{action_taken_esc}[/]")
    table.add_row("[bold]Confidence:[/bold]", f"{rule_output.confidence:.2f}")
    table.add_row("[bold]Content Type:[/bold]", f"{content_type_esc}")
    table.add_row("[bold]Details:[/bold]", f"{details_esc}")

    if hasattr(rule_output, 'triggered_keywords') and rule_output.triggered_keywords:
        keywords_str = ", ".join(map(str, rule_output.triggered_keywords))
        if len(keywords_str) > 100:
            keywords_str = keywords_str[:97] + "..."
        keywords_esc = escape_rich_markup(keywords_str)
        table.add_row("[bold]Triggers:[/bold]", f"[yellow]{keywords_esc}[/yellow]")

    panel = Panel(
        table,
        title=title,
        border_style=border_style,
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def cache_hit(cache_method: Literal["vector_search", "llm_call"], similarity: Optional[float] = None, input_preview: Optional[str] = None) -> None:
    """
    Prints a formatted panel when a cache hit occurs.
    
    Args:
        cache_method: The cache method used ("vector_search" or "llm_call")
        similarity: Similarity score for vector search (optional)
        input_preview: Preview of the input text (optional)
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    cache_method_esc = escape_rich_markup(cache_method)
    input_preview_esc = escape_rich_markup(input_preview) if input_preview else "N/A"
    
    table.add_row("[bold]Cache Status:[/bold]", "[green]✓ HIT[/green]")
    table.add_row("[bold]Method:[/bold]", f"[cyan]{cache_method_esc}[/cyan]")
    
    if similarity is not None:
        similarity_pct = f"{similarity:.1%}"
        table.add_row("[bold]Similarity:[/bold]", f"[yellow]{similarity_pct}[/yellow]")
    
    table.add_row("")  # Add spacing
    table.add_row("[bold]Input Preview:[/bold]")
    if len(input_preview_esc) > 100:
        input_preview_esc = input_preview_esc[:97] + "..."
    table.add_row(f"[dim]{input_preview_esc}[/dim]")
    
    panel = Panel(
        table,
        title="[bold green]🚀 Cache Hit - Response Retrieved[/bold green]",
        border_style="green",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def cache_miss(cache_method: Literal["vector_search", "llm_call"], input_preview: Optional[str] = None) -> None:
    """
    Prints a formatted panel when a cache miss occurs.
    
    Args:
        cache_method: The cache method used ("vector_search" or "llm_call")
        input_preview: Preview of the input text (optional)
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    cache_method_esc = escape_rich_markup(cache_method)
    input_preview_esc = escape_rich_markup(input_preview) if input_preview else "N/A"
    
    table.add_row("[bold]Cache Status:[/bold]", "[yellow]✗ MISS[/yellow]")
    table.add_row("[bold]Method:[/bold]", f"[cyan]{cache_method_esc}[/cyan]")
    table.add_row("")  # Add spacing
    table.add_row("[bold]Input Preview:[/bold]")
    if len(input_preview_esc) > 100:
        input_preview_esc = input_preview_esc[:97] + "..."
    table.add_row(f"[dim]{input_preview_esc}[/dim]")
    table.add_row("")  # Add spacing
    table.add_row("[bold]Action:[/bold]", "[blue]Executing task and caching result[/blue]")
    
    panel = Panel(
        table,
        title="[bold yellow]💾 Cache Miss - Executing Task[/bold yellow]",
        border_style="yellow",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def cache_stored(cache_method: Literal["vector_search", "llm_call"], input_preview: Optional[str] = None, duration_minutes: Optional[int] = None) -> None:
    """
    Prints a formatted panel when a new cache entry is stored.
    
    Args:
        cache_method: The cache method used ("vector_search" or "llm_call")
        input_preview: Preview of the input text (optional)
        duration_minutes: Cache duration in minutes (optional)
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    cache_method_esc = escape_rich_markup(cache_method)
    input_preview_esc = escape_rich_markup(input_preview) if input_preview else "N/A"
    
    table.add_row("[bold]Cache Status:[/bold]", "[green]✓ STORED[/green]")
    table.add_row("[bold]Method:[/bold]", f"[cyan]{cache_method_esc}[/cyan]")
    
    if duration_minutes is not None:
        table.add_row("[bold]Duration:[/bold]", f"[blue]{duration_minutes} minutes[/blue]")
    
    table.add_row("")  # Add spacing
    table.add_row("[bold]Input Preview:[/bold]")
    if len(input_preview_esc) > 100:
        input_preview_esc = input_preview_esc[:97] + "..."
    table.add_row(f"[dim]{input_preview_esc}[/dim]")
    
    panel = Panel(
        table,
        title="[bold green]💾 Cache Entry Stored[/bold green]",
        border_style="green",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def cache_stats(stats: Dict[str, Any]) -> None:
    """
    Prints a formatted panel with cache statistics.
    
    Args:
        stats: Dictionary containing cache statistics
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    total_entries = stats.get("total_entries", 0)
    active_entries = stats.get("active_entries", 0)
    expired_entries = stats.get("expired_entries", 0)
    cache_method = escape_rich_markup(stats.get("cache_method", "unknown"))
    cache_threshold = stats.get("cache_threshold", 0.0)
    cache_duration = stats.get("cache_duration_minutes", 0)
    cache_hit = stats.get("cache_hit", False)
    
    table.add_row("[bold]Total Entries:[/bold]", f"[cyan]{total_entries}[/cyan]")
    table.add_row("[bold]Active Entries:[/bold]", f"[green]{active_entries}[/green]")
    table.add_row("[bold]Expired Entries:[/bold]", f"[red]{expired_entries}[/red]")
    table.add_row("")  # Add spacing
    table.add_row("[bold]Method:[/bold]", f"[yellow]{cache_method}[/yellow]")
    
    if cache_method == "vector_search":
        threshold_pct = f"{cache_threshold:.1%}"
        table.add_row("[bold]Threshold:[/bold]", f"[blue]{threshold_pct}[/blue]")
    
    table.add_row("[bold]Duration:[/bold]", f"[blue]{cache_duration} minutes[/blue]")
    table.add_row("")  # Add spacing
    table.add_row("[bold]Last Hit:[/bold]", "[green]✓ Yes[/green]" if cache_hit else "[red]✗ No[/red]")
    
    panel = Panel(
        table,
        title="[bold magenta]📊 Cache Statistics[/bold magenta]",
        border_style="magenta",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def cache_cleared() -> None:
    """
    Prints a formatted panel when cache is cleared.
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Cache Status:[/bold]", "[red]🗑️ CLEARED[/red]")
    table.add_row("")  # Add spacing
    table.add_row("[bold]Action:[/bold]", "[blue]All cache entries have been removed[/blue]")
    
    panel = Panel(
        table,
        title="[bold red]🗑️ Cache Cleared[/bold red]",
        border_style="red",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def cache_configuration(enable_cache: bool, cache_method: Literal["vector_search", "llm_call"], cache_threshold: Optional[float] = None, 
                       cache_duration_minutes: Optional[int] = None, embedding_provider: Optional[str] = None) -> None:
    """
    Prints a formatted panel showing cache configuration.
    
    Args:
        enable_cache: Whether cache is enabled
        cache_method: The cache method ("vector_search" or "llm_call")
        cache_threshold: Similarity threshold for vector search (optional)
        cache_duration_minutes: Cache duration in minutes (optional)
        embedding_provider: Name of embedding provider (optional)
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    cache_method_esc = escape_rich_markup(cache_method)
    embedding_provider_esc = escape_rich_markup(embedding_provider) if embedding_provider else "Auto-detected"
    
    table.add_row("[bold]Cache Enabled:[/bold]", "[green]✓ Yes[/green]" if enable_cache else "[red]✗ No[/red]")
    
    if enable_cache:
        table.add_row("[bold]Method:[/bold]", f"[cyan]{cache_method_esc}[/cyan]")
        
        if cache_method == "vector_search":
            if cache_threshold is not None:
                threshold_pct = f"{cache_threshold:.1%}"
                table.add_row("[bold]Threshold:[/bold]", f"[blue]{threshold_pct}[/blue]")
            table.add_row("[bold]Embedding Provider:[/bold]", f"[yellow]{embedding_provider_esc}[/yellow]")
        
        if cache_duration_minutes is not None:
            table.add_row("[bold]Duration:[/bold]", f"[blue]{cache_duration_minutes} minutes[/blue]")
    
    panel = Panel(
        table,
        title="[bold cyan]⚙️ Cache Configuration[/bold cyan]",
        border_style="cyan",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()

def agent_started(agent_name: str) -> None:
    """
    Prints a formatted panel when an agent starts to work.

    Args:
        agent_name: Name or ID of the agent that started working
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    agent_name_esc = escape_rich_markup(agent_name)

    table.add_row("[bold]Agent Status:[/bold]", "[green]🚀 Started to work[/green]")
    table.add_row("[bold]Agent Name:[/bold]", f"[cyan]{agent_name_esc}[/cyan]")

    panel = Panel(
        table,
        title="[bold green]🤖 Agent Started[/bold green]",
        border_style="green",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def info_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints an info log message.
    
    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)
    
    console.print(f"[blue][INFO][/blue] [{context_esc}] {message_esc}")


def warning_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints a warning log message.
    
    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)
    
    console.print(f"[yellow][WARNING][/yellow] [{context_esc}] {message_esc}")


def error_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints an error log message.
    
    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)
    
    console.print(f"[red][ERROR][/red] [{context_esc}] {message_esc}")


def debug_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints a debug log message.
    
    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)
    
    console.print(f"[dim][DEBUG][/dim] [{context_esc}] {message_esc}")
    
def import_error(package_name: str, install_command: str = None, feature_name: str = None) -> None:
    """
    Prints a formatted error panel for missing package imports.

    Args:
        package_name: Name of the missing package
        install_command: Command to install the package (e.g., "pip install package_name")
        feature_name: Optional name of the feature requiring this package
    """
    table = Table(show_header=False, expand=True, box=None)

    package_name_esc = escape_rich_markup(package_name)

    if feature_name:
        feature_name_esc = escape_rich_markup(feature_name)
        title = f"[bold red]📦 Missing Package for {feature_name_esc}[/bold red]"
        table.add_row("[bold]Feature:[/bold]", f"[cyan]{feature_name_esc}[/cyan]")
    else:
        title = "[bold red]📦 Missing Package[/bold red]"

    table.add_row("[bold]Package:[/bold]", f"[yellow]{package_name_esc}[/yellow]")
    table.add_row("")  # Add spacing

    if install_command:
        install_command_esc = escape_rich_markup(install_command)
        table.add_row("[bold]Install Command:[/bold]")
        table.add_row(f"[green]{install_command_esc}[/green]")
    else:
        table.add_row("[bold]Install Command:[/bold]")
        table.add_row(f"[green]pip install {package_name_esc}[/green]")

    panel = Panel(
        table,
        title=title,
        border_style="red",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()
    raise ImportError(f"Missing required package: {package_name}")


def success_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints a success log message.
    
    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)
    
    console.print(f"[green][SUCCESS][/green] [{context_esc}] {message_esc}")


def connection_info(provider: str, version: str = "unknown") -> None:
    """
    Log connection information for a provider.
    
    Args:
        provider: The provider name
        version: The provider version
    """
    provider_esc = escape_rich_markup(provider)
    version_esc = escape_rich_markup(version)
    
    console.print(f"[green][CONNECTED][/green] [{provider_esc}] version: {version_esc}")


def pipeline_started(total_steps: int) -> None:
    """
    Log pipeline execution start.

    Args:
        total_steps: Total number of steps in the pipeline
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Pipeline Status:[/bold]", "[blue]Starting[/blue]")
    table.add_row("[bold]Total Steps:[/bold]", f"[blue]{total_steps}[/blue]")

    panel = Panel(
        table,
        title="[bold blue]Pipeline Started[/bold blue]",
        border_style="blue",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def pipeline_step_started(step_name: str, step_description: str = None) -> None:
    """
    Log pipeline step execution start.

    Args:
        step_name: Name of the step
        step_description: Optional description of the step
    """
    step_name_esc = escape_rich_markup(step_name)
    step_description_esc = escape_rich_markup(step_description) if step_description else "Processing..."

    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Step:[/bold]", f"[cyan]{step_name_esc}[/cyan]")
    table.add_row("[bold]Description:[/bold]", f"{step_description_esc}")

    panel = Panel(
        table,
        title="[bold cyan]Step Started[/bold cyan]",
        border_style="cyan",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def pipeline_step_completed(step_name: str, status: str, execution_time: float, message: str = None) -> None:
    """
    Log pipeline step completion.

    Args:
        step_name: Name of the step
        status: Step status (SUCCESS, ERROR, PENDING)
        execution_time: Time taken to execute the step
        message: Optional message from the step
    """
    step_name_esc = escape_rich_markup(step_name)
    message_esc = escape_rich_markup(message) if message else "Completed"

    if status == "SUCCESS":
        status_color = "green"
        border_style = "green"
    elif status == "ERROR":
        status_color = "red"
        border_style = "red"
    elif status == "PENDING":
        status_color = "yellow"
        border_style = "yellow"
    else:
        status_color = "dim"
        border_style = "dim"

    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Step:[/bold]", f"[{status_color}]{step_name_esc}[/{status_color}]")
    table.add_row("[bold]Status:[/bold]", f"[{status_color}]{status}[/{status_color}]")
    table.add_row("[bold]Time:[/bold]", f"[{status_color}]{execution_time:.3f}s[/{status_color}]")
    if message:
        table.add_row("[bold]Message:[/bold]", f"{message_esc}")

    panel = Panel(
        table,
        title=f"[bold {status_color}]Step Completed[/bold {status_color}]",
        border_style=border_style,
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def pipeline_completed(executed_steps: int, total_steps: int, total_time: float) -> None:
    """
    Log pipeline completion.

    Args:
        executed_steps: Number of steps executed
        total_steps: Total number of steps
        total_time: Total execution time
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Pipeline Status:[/bold]", "[green]Completed[/green]")
    table.add_row("[bold]Steps Executed:[/bold]", f"[green]{executed_steps}/{total_steps}[/green]")
    table.add_row("[bold]Total Time:[/bold]", f"[green]{total_time:.3f}s[/green]")

    panel = Panel(
        table,
        title="[bold green]Pipeline Completed[/bold green]",
        border_style="green",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def pipeline_failed(error_message: str, executed_steps: int, total_steps: int, failed_step: str = None, step_time: float = None) -> None:
    """
    Log pipeline failure.

    Args:
        error_message: Error message
        executed_steps: Number of steps executed before failure
        total_steps: Total number of steps
        failed_step: Name of the step that failed
        step_time: Time taken by the failed step
    """
    error_esc = escape_rich_markup(error_message)
    failed_step_esc = escape_rich_markup(failed_step) if failed_step else "Unknown"

    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Pipeline Status:[/bold]", "[red]Failed[/red]")
    table.add_row("[bold]Failed Step:[/bold]", f"[red]{failed_step_esc}[/red]")
    table.add_row("[bold]Steps Executed:[/bold]", f"[red]{executed_steps}/{total_steps}[/red]")
    if step_time is not None:
        table.add_row("[bold]Step Time:[/bold]", f"[red]{step_time:.3f}s[/red]")
    table.add_row("[bold]Error:[/bold]", f"[red]{error_esc}[/red]")

    panel = Panel(
        table,
        title="[bold red]Pipeline Failed[/bold red]",
        border_style="red",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def pipeline_paused(step_name: str) -> None:
    """
    Log pipeline pause (e.g., for external execution).

    Args:
        step_name: Name of the step where pipeline paused
    """
    step_name_esc = escape_rich_markup(step_name)

    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Pipeline Status:[/bold]", "[yellow]Paused[/yellow]")
    table.add_row("[bold]Step:[/bold]", f"[yellow]{step_name_esc}[/yellow]")
    table.add_row("[bold]Reason:[/bold]", "[yellow]External execution[/yellow]")

    panel = Panel(
        table,
        title="[bold yellow]Pipeline Paused[/bold yellow]",
        border_style="yellow",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def compression_fallback(original_strategy: str, fallback_strategy: str, error: str) -> None:
    """
    Log compression strategy fallback.
    
    Args:
        original_strategy: Original compression strategy that failed
        fallback_strategy: Fallback strategy being used
        error: Error message from the original strategy
    """
    from rich.table import Table
    from rich.panel import Panel
    
    original_esc = escape_rich_markup(original_strategy)
    fallback_esc = escape_rich_markup(fallback_strategy)
    error_esc = escape_rich_markup(str(error))
    
    table = Table(show_header=False, box=None, expand=True)
    table.add_column(style="bold yellow", width=20)
    table.add_column(style="white")
    
    table.add_row("⚠️ STATUS", "[bold yellow]COMPRESSION FALLBACK[/bold yellow]")
    table.add_row("❌ ORIGINAL", f"[bold red]{original_esc}[/bold red]")
    table.add_row("✅ FALLBACK", f"[bold green]{fallback_esc}[/bold green]")
    table.add_row("💬 ERROR", f"[dim]{error_esc}[/dim]")
    table.add_row("🔄 ACTION", "[bold cyan]CONTINUING WITH FALLBACK[/bold cyan]")
    
    panel = Panel(
        table,
        title="[bold yellow]⚠️ COMPRESSION STRATEGY FALLBACK[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
        expand=True
    )
    console.print(panel)


def model_recommendation_summary(recommendation) -> None:
    """
    Log model recommendation summary.
    
    Args:
        recommendation: ModelRecommendation object
    """
    from rich.table import Table
    from rich.panel import Panel
    
    method_esc = escape_rich_markup(recommendation.selection_method)
    model_esc = escape_rich_markup(recommendation.model_name)
    reason_esc = escape_rich_markup(recommendation.reason)
    confidence_esc = escape_rich_markup(f"{recommendation.confidence_score:.2f}")
    
    # Create cost and speed tier bars
    cost_bar = "█" * recommendation.estimated_cost_tier + "░" * (10 - recommendation.estimated_cost_tier)
    speed_bar = "█" * recommendation.estimated_speed_tier + "░" * (10 - recommendation.estimated_speed_tier)
    
    table = Table(show_header=False, box=None, expand=True)
    table.add_column(style="bold blue", width=20)
    table.add_column(style="white")
    
    table.add_row("🤖 MODEL", f"[bold cyan]{model_esc}[/bold cyan]")
    table.add_row("🧠 METHOD", f"[bold]{method_esc}[/bold]")
    table.add_row("💭 REASON", reason_esc)
    table.add_row("🎯 CONFIDENCE", f"[bold green]{confidence_esc}[/bold green]")
    table.add_row("💰 COST", f"[bold]{recommendation.estimated_cost_tier}/10[/bold] [{cost_bar}]")
    table.add_row("⚡ SPEED", f"[bold]{recommendation.estimated_speed_tier}/10[/bold] [{speed_bar}]")
    
    if recommendation.alternative_models:
        alternatives = ", ".join(recommendation.alternative_models[:3])
        alternatives_esc = escape_rich_markup(alternatives)
        table.add_row("🔄 ALTERNATIVES", alternatives_esc)
    
    panel = Panel(
        table,
        title="[bold blue]🤖 MODEL RECOMMENDATION[/bold blue]",
        border_style="blue",
        padding=(1, 2),
        expand=True
    )
    console.print(panel)


def model_recommendation_error(error_message: str) -> None:
    """
    Log model recommendation error.
    
    Args:
        error_message: Error message
    """
    from rich.table import Table
    from rich.panel import Panel
    
    error_esc = escape_rich_markup(str(error_message))
    
    table = Table(show_header=False, box=None, expand=True)
    table.add_column(style="bold red", width=20)
    table.add_column(style="white")
    
    table.add_row("❌ STATUS", "[bold red]RECOMMENDATION FAILED[/bold red]")
    table.add_row("💬 ERROR", f"[red]{error_esc}[/red]")
    table.add_row("🔧 ACTION", "[bold yellow]USING DEFAULT MODEL[/bold yellow]")
    table.add_row("🔄 RECOVERY", "[bold green]CONTINUING EXECUTION[/bold green]")
    
    panel = Panel(
        table,
        title="[bold red]❌ MODEL RECOMMENDATION ERROR[/bold red]",
        border_style="red",
        padding=(1, 2),
        expand=True
    )
    console.print(panel)


def pipeline_timeline(step_results: dict, total_time: float, min_threshold: float = 0.001) -> None:
    """
    Print a timeline visualization of pipeline step execution times.

    Args:
        step_results: Dictionary of step names to their execution stats
        total_time: Total pipeline execution time
        min_threshold: Minimum time in seconds to show (default 0.001s = 1ms)
    """
    if not step_results:
        return

    # Sort steps by their execution time (descending)
    sorted_steps = sorted(
        step_results.items(),
        key=lambda x: x[1].get("execution_time", 0),
        reverse=True
    )

    # Filter steps above threshold
    significant_steps = [
        (name, info) for name, info in sorted_steps
        if info.get("execution_time", 0) >= min_threshold
    ]

    # Count filtered steps
    filtered_count = len(sorted_steps) - len(significant_steps)

    table = Table(show_header=True, expand=True, box=None)
    table.width = 60

    table.add_column("[bold]Step[/bold]", justify="left", style="cyan")
    table.add_column("[bold]Time[/bold]", justify="right", style="magenta")
    table.add_column("[bold]%[/bold]", justify="right", style="yellow")
    table.add_column("[bold]Bar[/bold]", justify="left", style="blue")

    # Add each significant step
    for step_name, step_info in significant_steps:
        step_name_esc = escape_rich_markup(step_name)
        exec_time = step_info.get("execution_time", 0)
        time_str = f"{exec_time:.3f}s"

        # Calculate percentage
        percentage = (exec_time / total_time * 100) if total_time > 0 else 0
        percentage_str = f"{percentage:.1f}%"

        # Create a visual bar (max 20 characters)
        bar_length = int(percentage / 5) if percentage > 0 else 0  # 5% = 1 char
        bar_length = min(bar_length, 20)  # Cap at 20 chars
        bar = "█" * bar_length

        table.add_row(
            step_name_esc,
            time_str,
            percentage_str,
            f"[blue]{bar}[/blue]"
        )

    # Add note about filtered steps
    if filtered_count > 0:
        table.add_row("")
        table.add_row(
            f"[dim]({filtered_count} steps < {min_threshold*1000:.0f}ms hidden)[/dim]",
            "",
            "",
            ""
        )

    # Add total row
    table.add_row("")
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold magenta]{total_time:.3f}s[/bold magenta]",
        "[bold yellow]100.0%[/bold yellow]",
        ""
    )

    panel = Panel(
        table,
        title="[bold blue]Pipeline Timeline[/bold blue]",
        border_style="blue",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()


def simple_output(message: str) -> None:
    """
    Simple output function for basic console printing.

    Args:
        message: Message to print
    """
    console.print(message)


def deep_agent_todo_completion_check(iteration: int, completed_count: int, total_count: int) -> None:
    """
    Print a formatted panel for Deep Agent todo completion check.
    
    Args:
        iteration: Current iteration number
        completed_count: Number of completed todos
        total_count: Total number of todos
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    completion_percentage = (completed_count / total_count * 100) if total_count > 0 else 0
    
    table.add_row("[bold]Todo Completion Check:[/bold]", f"[cyan]Iteration {iteration}[/cyan]")
    table.add_row("[bold]Completed:[/bold]", f"[green]{completed_count}/{total_count}[/green]")
    table.add_row("[bold]Progress:[/bold]", f"[yellow]{completion_percentage:.1f}%[/yellow]")
    table.add_row("[bold]Status:[/bold]", "[blue]Continuing to complete remaining todos...[/blue]")
    
    panel = Panel(
        table,
        title="[bold yellow]⚠️ Deep Agent - Todo Completion Check[/bold yellow]",
        border_style="yellow",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()


def deep_agent_all_todos_completed(total_count: int) -> None:
    """
    Print a formatted panel when all Deep Agent todos are completed.
    
    Args:
        total_count: Total number of todos that were completed
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Status:[/bold]", "[green]✅ All todos completed successfully![/green]")
    table.add_row("[bold]Total Completed:[/bold]", f"[green]{total_count}[/green]")
    table.add_row("[bold]Result:[/bold]", "[green]Deep Agent task finished[/green]")
    
    panel = Panel(
        table,
        title="[bold green]✅ Deep Agent - All Todos Completed[/bold green]",
        border_style="green",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()


def deep_agent_max_iterations_warning(max_iterations: int, incomplete_count: int) -> None:
    """
    Print a formatted panel when Deep Agent reaches maximum iterations with incomplete todos.
    
    Args:
        max_iterations: Maximum number of iterations allowed
        incomplete_count: Number of todos still incomplete
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Status:[/bold]", "[red]⚠️ WARNING: Maximum iterations reached[/red]")
    table.add_row("[bold]Max Iterations:[/bold]", f"[yellow]{max_iterations}[/yellow]")
    table.add_row("[bold]Incomplete Todos:[/bold]", f"[red]{incomplete_count}[/red]")
    table.add_row("[bold]Action:[/bold]", "[yellow]Stopping execution[/yellow]")
    
    panel = Panel(
        table,
        title="[bold red]⚠️ Deep Agent - Max Iterations Warning[/bold red]",
        border_style="red",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
