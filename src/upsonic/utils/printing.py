from typing import Any, Dict, Literal, Optional, Union, TYPE_CHECKING
from decimal import Decimal

if TYPE_CHECKING:
    from upsonic.models import Model
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
# Align and Text imports removed - not currently used
import platform
import sys
from rich.markup import escape

# Setup background logging (console disabled, only file/Sentry)
from upsonic.utils.logging_config import setup_logging, get_logger
setup_logging(enable_console=False)  # Console kapalı, Rich kullanıyoruz
_bg_logger = get_logger("upsonic.user")  # Background logger for Sentry/file
_sentry_logger = get_logger("upsonic.sentry")  # Sentry event logger (INFO+ -> Sentry)

# Initialize Console with Windows encoding compatibility
# Handle Unicode encoding errors gracefully on Windows
try:
    if platform.system() == "Windows":
        # On Windows, try to set UTF-8 encoding for stdout if possible
        try:
            # Python 3.7+ supports reconfigure
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            # Note: We don't try to wrap stdout buffer as it can break Rich Console
            # Rich handles encoding internally, we just configure stdout if supported
        except (AttributeError, OSError, ValueError):
            # If encoding setup fails, continue with default
            pass
    console = Console()
except (AttributeError, OSError, ValueError):  # noqa: BLE001
    # Fallback to default console if initialization fails
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

    # Sentry logging (kullanıcı model call sonucunu gördü)
    execution_time = end_time - start_time
    event_data = {
        "model": str(model.model_name),
        "response_format": str(response_format),
        "execution_time": execution_time,
        "input_tokens": str(usage.get('input_tokens', 0)),
        "output_tokens": str(usage.get('output_tokens', 0)),
        "estimated_cost": str(get_estimated_cost(usage.get('input_tokens', 0), usage.get('output_tokens', 0), model))
    }

    # Tool kullanıldıysa ekle
    if tool_usage and len(tool_usage) > 0:
        event_data["tools_used"] = len(tool_usage)
        event_data["tool_names"] = [t.get('tool_name', '') for t in tool_usage[:5]]  # İlk 5 tool

    # Sentry event olarak gönder (LoggingIntegration ile otomatik)
    _sentry_logger.info(
        "Model call: %s (%.2fs, %d tools)",
        model.model_name, execution_time, len(tool_usage) if tool_usage else 0,
        extra=event_data
    )




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

    # Sentry logging (kullanıcı agent sonucunu gördü)
    execution_time = end_time - start_time
    event_data = {
        "model": str(model.model_name),
        "response_format": response_format,
        "execution_time": execution_time,
        "tool_count": tool_count,
        "context_count": context_count,
        "input_tokens": usage.get('input_tokens', 0),
        "output_tokens": usage.get('output_tokens', 0),
    }

    # Tool kullanıldıysa ekle
    if tool_usage and len(tool_usage) > 0:
        event_data["tools_used"] = len(tool_usage)
        event_data["tool_names"] = [t.get('tool_name', '') for t in tool_usage[:5]]  # İlk 5 tool

    # Sentry event olarak gönder (LoggingIntegration ile otomatik)
    _sentry_logger.info(
        "Agent completed: %d tools, %d contexts, %.2fs",
        tool_count, context_count, execution_time,
        extra=event_data
    )


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

    # Sentry event olarak gönder (LoggingIntegration ile otomatik)
    _sentry_logger.info("Agent started: %s", agent_name, extra={"agent_name": agent_name})


def info_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints an info log message.

    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)

    # Rich console output (user görür)
    console.print(f"[blue][INFO][/blue] [{context_esc}] {message_esc}")

    # Background logging (Sentry/file'a gider)
    _bg_logger.info(f"[{context}] {message}")


def warning_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints a warning log message.

    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)

    # Rich console output (user görür)
    console.print(f"[yellow][WARNING][/yellow] [{context_esc}] {message_esc}")

    # Background logging (Sentry/file'a gider)
    _bg_logger.warning(f"[{context}] {message}")


def error_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints an error log message.

    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)

    # Rich console output (user görür)
    console.print(f"[red][ERROR][/red] [{context_esc}] {message_esc}")

    # Background logging (Sentry/file'a gider)
    # _bg_logger.error() zaten LoggingIntegration ile Sentry'e event olarak gider
    _bg_logger.error(f"[{context}] {message}")


def debug_log(message: str, context: str = "Upsonic") -> None:
    """
    Prints a debug log message.

    Args:
        message: The log message
        context: The context/module name
    """
    message_esc = escape_rich_markup(message)
    context_esc = escape_rich_markup(context)

    # Rich console output (user görür)
    console.print(f"[dim][DEBUG][/dim] [{context_esc}] {message_esc}")

    # Background logging (file'a gider, Sentry'e GİTMEZ - debug log)
    _bg_logger.debug(f"[{context}] {message}")

    # NOT: Debug loglar Sentry'e gönderilmez, sadece user-facing important loglar gider
    
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

    # Rich console output (user görür)
    console.print(f"[green][SUCCESS][/green] [{context_esc}] {message_esc}")

    # Background logging (Sentry/file'a gider)
    _bg_logger.info(f"[SUCCESS] [{context}] {message}")


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

    # Sentry event olarak gönder (LoggingIntegration ile otomatik)
    event_data = {"total_steps": total_steps}
    _sentry_logger.info("Pipeline started: %d steps", total_steps, extra=event_data)


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

    # Sentry event olarak gönder (LoggingIntegration ile otomatik)
    event_data = {
        "executed_steps": executed_steps,
        "total_steps": total_steps,
        "total_time": total_time,
        "status": "completed"
    }
    _sentry_logger.info(
        "Pipeline completed: %d/%d steps, %.3fs",
        executed_steps, total_steps, total_time,
        extra=event_data
    )


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

    # Sentry event olarak gönder (LoggingIntegration ile otomatik)
    event_data = {
        "error_message": error_message,
        "executed_steps": executed_steps,
        "total_steps": total_steps,
        "failed_step": failed_step,
        "step_time": step_time,
        "status": "failed"
    }
    _sentry_logger.error(
        "Pipeline failed: %s (step: %s)",
        error_message, failed_step,
        extra=event_data
    )


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
    
    # Use safe characters for Windows compatibility
    is_windows = platform.system() == "Windows"
    model_char = "🤖" if not is_windows else "[MODEL]"
    method_char = "🧠" if not is_windows else "[METHOD]"
    reason_char = "💭" if not is_windows else "[REASON]"
    confidence_char = "🎯" if not is_windows else ">>"
    cost_char = "💰" if not is_windows else "$$"
    speed_char = "⚡" if not is_windows else ">>"
    alternatives_char = "🔄" if not is_windows else "[ALT]"
    
    table = Table(show_header=False, box=None, expand=True)
    table.add_column(style="bold blue", width=20)
    table.add_column(style="white")
    
    table.add_row(f"{model_char} MODEL", f"[bold cyan]{model_esc}[/bold cyan]")
    table.add_row(f"{method_char} METHOD", f"[bold]{method_esc}[/bold]")
    table.add_row(f"{reason_char} REASON", reason_esc)
    table.add_row(f"{confidence_char} CONFIDENCE", f"[bold green]{confidence_esc}[/bold green]")
    table.add_row(f"{cost_char} COST", f"[bold]{recommendation.estimated_cost_tier}/10[/bold] [{cost_bar}]")
    table.add_row(f"{speed_char} SPEED", f"[bold]{recommendation.estimated_speed_tier}/10[/bold] [{speed_bar}]")
    
    if recommendation.alternative_models:
        alternatives = ", ".join(recommendation.alternative_models[:3])
        alternatives_esc = escape_rich_markup(alternatives)
        table.add_row(f"{alternatives_char} ALTERNATIVES", alternatives_esc)
    
    panel = Panel(
        table,
        title=f"[bold blue]{model_char} MODEL RECOMMENDATION[/bold blue]",
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


def direct_started(model_name: str, task_description: str, response_format: str = "str") -> None:
    """
    Print a formatted panel when Direct class starts execution.
    
    Args:
        model_name: Name of the model being used
        task_description: Description of the task
        response_format: Expected response format
    """
    model_name_esc = escape_rich_markup(model_name)
    response_format_esc = escape_rich_markup(response_format)
    
    # Truncate task description if too long
    task_preview = task_description[:150] + "..." if len(task_description) > 150 else task_description
    task_preview_esc = escape_rich_markup(task_preview)
    
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    # Use safe character for Windows compatibility
    lightning_char = "⚡" if platform.system() != "Windows" else "►"
    
    table.add_row("[bold]Status:[/bold]", f"[blue]{lightning_char} Direct Execution Started[/blue]")
    table.add_row("[bold]Model:[/bold]", f"[cyan]{model_name_esc}[/cyan]")
    table.add_row("[bold]Response Format:[/bold]", f"[yellow]{response_format_esc}[/yellow]")
    table.add_row("")  # Spacing
    table.add_row("[bold]Task:[/bold]")
    table.add_row(f"[dim]{task_preview_esc}[/dim]")
    
    panel = Panel(
        table,
        title=f"[bold blue]{lightning_char} Upsonic Direct - Execution Started[/bold blue]",
        border_style="blue",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    # Sentry logging
    _sentry_logger.info(
        "Direct execution started: %s",
        model_name,
        extra={
            "model": model_name,
            "response_format": response_format,
            "task_preview": task_description[:100]
        }
    )


def direct_completed(
    result: Any, 
    model: Any, 
    response_format: str, 
    start_time: float, 
    end_time: float, 
    usage: dict,
    debug: bool = False,
    task_description: str = None
) -> None:
    """
    Print a formatted panel when Direct class completes execution.
    Shows comprehensive metrics including cost, time, and token usage.
    
    Args:
        result: The result from Direct execution
        model: Model instance
        response_format: Response format used
        start_time: Start timestamp
        end_time: End timestamp
        usage: Dictionary with input_tokens and output_tokens
        debug: Whether to show full result
        task_description: Optional task description preview
    """
    execution_time = end_time - start_time
    
    display_model_name = escape_rich_markup(model.model_name)
    response_format_esc = escape_rich_markup(response_format)
    
    # Calculate cost
    estimated_cost = get_estimated_cost(
        usage.get('input_tokens', 0), 
        usage.get('output_tokens', 0), 
        model
    )
    
    # Format result
    result_str = str(result)
    if not debug:
        result_str = result_str[:370]
    if len(result_str) < len(str(result)):
        result_str += "..."
    result_esc = escape_rich_markup(result_str)
    
    # Create main table
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Status:[/bold]", "[green]✅ Execution Completed[/green]")
    table.add_row("[bold]Model:[/bold]", f"[cyan]{display_model_name}[/cyan]")
    table.add_row("[bold]Response Format:[/bold]", f"[yellow]{response_format_esc}[/yellow]")
    table.add_row("")  # Spacing
    
    # Show task preview if provided
    if task_description:
        task_preview = task_description[:100] + "..." if len(task_description) > 100 else task_description
        task_preview_esc = escape_rich_markup(task_preview)
        table.add_row("[bold]Task:[/bold]")
        table.add_row(f"[dim]{task_preview_esc}[/dim]")
        table.add_row("")  # Spacing
    
    # Show result
    table.add_row("[bold]Result:[/bold]")
    table.add_row(f"[green]{result_esc}[/green]")
    table.add_row("")  # Spacing
    
    # Performance metrics section
    table.add_row("[bold cyan]📊 Performance Metrics[/bold cyan]", "")
    table.add_row("├─ [bold]Execution Time:[/bold]", f"[magenta]{execution_time:.3f}s[/magenta]")
    table.add_row("├─ [bold]Input Tokens:[/bold]", f"[blue]{usage.get('input_tokens', 0):,}[/blue]")
    table.add_row("├─ [bold]Output Tokens:[/bold]", f"[blue]{usage.get('output_tokens', 0):,}[/blue]")
    table.add_row("└─ [bold]Estimated Cost:[/bold]", f"[yellow]{estimated_cost}[/yellow]")
    
    # Use safe character for Windows compatibility
    lightning_char = "⚡" if platform.system() != "Windows" else "►"
    
    panel = Panel(
        table,
        title=f"[bold green]{lightning_char} Upsonic Direct - Execution Complete[/bold green]",
        border_style="green",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    # Sentry logging
    _sentry_logger.info(
        "Direct execution completed: %s (%.2fs)",
        model.model_name, execution_time,
        extra={
            "model": str(model.model_name),
            "response_format": response_format,
            "execution_time": execution_time,
            "input_tokens": usage.get('input_tokens', 0),
            "output_tokens": usage.get('output_tokens', 0),
            "estimated_cost": str(estimated_cost)
        }
    )


def direct_error(
    error_message: str, 
    model_name: str = None,
    task_description: str = None,
    execution_time: float = None
) -> None:
    """
    Print a formatted panel when Direct class encounters an error.
    
    Args:
        error_message: The error message
        model_name: Optional model name
        task_description: Optional task description
        execution_time: Optional execution time before error
    """
    error_esc = escape_rich_markup(str(error_message))
    
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Status:[/bold]", "[red]❌ Execution Failed[/red]")
    
    if model_name:
        model_name_esc = escape_rich_markup(model_name)
        table.add_row("[bold]Model:[/bold]", f"[cyan]{model_name_esc}[/cyan]")
    
    if task_description:
        task_preview = task_description[:100] + "..." if len(task_description) > 100 else task_description
        task_preview_esc = escape_rich_markup(task_preview)
        table.add_row("[bold]Task:[/bold]", f"[dim]{task_preview_esc}[/dim]")
    
    table.add_row("")  # Spacing
    table.add_row("[bold]Error Details:[/bold]")
    table.add_row(f"[red]{error_esc}[/red]")
    
    if execution_time is not None:
        table.add_row("")  # Spacing
        table.add_row("[bold]Time Before Error:[/bold]", f"[yellow]{execution_time:.3f}s[/yellow]")
    
    # Use safe character for Windows compatibility
    lightning_char = "⚡" if platform.system() != "Windows" else "►"
    
    panel = Panel(
        table,
        title=f"[bold red]{lightning_char} Upsonic Direct - Execution Error[/bold red]",
        border_style="red",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    # Sentry logging
    _sentry_logger.error(
        "Direct execution failed: %s",
        error_message,
        extra={
            "error": str(error_message),
            "model": model_name,
            "execution_time": execution_time
        }
    )


def direct_metrics_summary(
    total_calls: int,
    total_time: float,
    total_input_tokens: int,
    total_output_tokens: int,
    total_cost: float,
    model_name: str,
    avg_time: float = None
) -> None:
    """
    Print a formatted panel with summary metrics for multiple Direct calls.
    
    Args:
        total_calls: Total number of Direct calls
        total_time: Total execution time
        total_input_tokens: Total input tokens used
        total_output_tokens: Total output tokens used
        total_cost: Total estimated cost
        model_name: Model name used
        avg_time: Optional average execution time per call
    """
    model_name_esc = escape_rich_markup(model_name)
    
    if avg_time is None:
        avg_time = total_time / total_calls if total_calls > 0 else 0
    
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold cyan]📊 Direct Execution Summary[/bold cyan]", "")
    table.add_row("├─ [bold]Model:[/bold]", f"[cyan]{model_name_esc}[/cyan]")
    table.add_row("├─ [bold]Total Calls:[/bold]", f"[blue]{total_calls}[/blue]")
    table.add_row("├─ [bold]Total Time:[/bold]", f"[magenta]{total_time:.3f}s[/magenta]")
    table.add_row("├─ [bold]Avg Time/Call:[/bold]", f"[magenta]{avg_time:.3f}s[/magenta]")
    table.add_row("")  # Spacing
    table.add_row("[bold yellow]💰 Token & Cost Metrics[/bold yellow]", "")
    table.add_row("├─ [bold]Input Tokens:[/bold]", f"[blue]{total_input_tokens:,}[/blue]")
    table.add_row("├─ [bold]Output Tokens:[/bold]", f"[blue]{total_output_tokens:,}[/blue]")
    table.add_row("├─ [bold]Total Tokens:[/bold]", f"[blue]{(total_input_tokens + total_output_tokens):,}[/blue]")
    table.add_row("└─ [bold]Total Cost:[/bold]", f"[yellow]~${total_cost:.4f}[/yellow]")
    
    # Use safe character for Windows compatibility
    lightning_char = "⚡" if platform.system() != "Windows" else "►"
    
    panel = Panel(
        table,
        title=f"[bold magenta]{lightning_char} Upsonic Direct - Session Summary[/bold magenta]",
        border_style="magenta",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()


def direct_configuration(
    model_name: str,
    settings: dict = None,
    provider: str = None
) -> None:
    """
    Print a formatted panel showing Direct configuration.
    
    Args:
        model_name: Model name
        settings: Optional model settings dictionary
        provider: Optional provider name
    """
    model_name_esc = escape_rich_markup(model_name)
    
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Configuration:[/bold]", "[cyan]⚙️ Direct Instance[/cyan]")
    table.add_row("[bold]Model:[/bold]", f"[cyan]{model_name_esc}[/cyan]")
    
    if provider:
        provider_esc = escape_rich_markup(provider)
        table.add_row("[bold]Provider:[/bold]", f"[yellow]{provider_esc}[/yellow]")
    
    if settings:
        table.add_row("")  # Spacing
        table.add_row("[bold]Settings:[/bold]", "")
        for key, value in settings.items():
            key_esc = escape_rich_markup(str(key))
            value_esc = escape_rich_markup(str(value))
            table.add_row(f"  ├─ [bold]{key_esc}:[/bold]", f"[blue]{value_esc}[/blue]")
    
    # Use safe character for Windows compatibility
    lightning_char = "⚡" if platform.system() != "Windows" else "►"
    
    panel = Panel(
        table,
        title=f"[bold cyan]{lightning_char} Upsonic Direct - Configuration[/bold cyan]",
        border_style="cyan",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()


# OCR-specific printing functions

def ocr_loading(provider_name: str, languages: list = None, extra_info: dict = None) -> None:
    """
    Print a formatted message when OCR provider is loading/initializing.
    
    Args:
        provider_name: Name of the OCR provider
        languages: List of languages to load
        extra_info: Optional dictionary with additional info (gpu, version, features, etc.)
    """
    provider_esc = escape_rich_markup(provider_name)
    
    # Simple one-line output for better UX
    lang_str = ", ".join(languages) if languages else "default"
    lang_esc = escape_rich_markup(lang_str)
    
    console.print(f"[blue]🔄 Initializing {provider_esc}[/blue] [dim](languages: {lang_esc})[/dim]")
    
    if extra_info:
        for key, value in extra_info.items():
            key_esc = escape_rich_markup(str(key))
            value_esc = escape_rich_markup(str(value))
            console.print(f"   [dim]• {key_esc}: {value_esc}[/dim]")
    
    # Background logging
    _bg_logger.info(f"[OCR] Initializing {provider_name} with languages: {lang_str}")


def ocr_initialized(provider_name: str) -> None:
    """
    Print a success message when OCR provider is initialized.
    
    Args:
        provider_name: Name of the OCR provider
    """
    provider_esc = escape_rich_markup(provider_name)
    console.print(f"   [green]✓ {provider_esc} initialized successfully[/green]")
    
    # Background logging
    _bg_logger.info(f"[OCR] {provider_name} initialized successfully")


def ocr_language_not_supported(
    provider_name: str, 
    unsupported_langs: list, 
    supported_langs: list = None,
    help_url: str = None
) -> None:
    """
    Print error message when requested language is not supported.
    
    Args:
        provider_name: Name of the OCR provider
        unsupported_langs: List of unsupported language codes
        supported_langs: Optional list of supported languages (shows sample)
        help_url: Optional URL for more information
    """
    provider_esc = escape_rich_markup(provider_name)
    unsupported_esc = escape_rich_markup(", ".join(unsupported_langs))
    
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Provider:[/bold]", f"[red]{provider_esc}[/red]")
    table.add_row("[bold]Unsupported Languages:[/bold]", f"[red]{unsupported_esc}[/red]")
    
    if supported_langs:
        # Show a sample of supported languages
        sample_size = min(30, len(supported_langs))
        sample_langs = ", ".join(supported_langs[:sample_size])
        if len(supported_langs) > sample_size:
            sample_langs += "..."
        sample_esc = escape_rich_markup(sample_langs)
        table.add_row("")
        table.add_row("[bold]Available Languages:[/bold]")
        table.add_row(f"[dim]{sample_esc}[/dim]")
    
    if help_url:
        help_url_esc = escape_rich_markup(help_url)
        table.add_row("")
        table.add_row("[bold]More Info:[/bold]", f"[blue]{help_url_esc}[/blue]")
    
    panel = Panel(
        table,
        title=f"[bold red]❌ OCR Language Not Supported[/bold red]",
        border_style="red",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    # Background logging
    _bg_logger.error(f"[OCR] {provider_name}: Unsupported languages: {', '.join(unsupported_langs)}")


def ocr_language_warning(provider_name: str, warning_langs: list, best_supported: list = None) -> None:
    """
    Print warning message when requested language has limited support.
    
    Args:
        provider_name: Name of the OCR provider
        warning_langs: List of languages with limited support
        best_supported: Optional list of best supported languages
    """
    provider_esc = escape_rich_markup(provider_name)
    warning_esc = escape_rich_markup(", ".join(warning_langs))
    
    console.print(f"[yellow]⚠️  Warning: {provider_esc}[/yellow] [dim]- Language(s) may have limited support: {warning_esc}[/dim]")
    
    if best_supported:
        best_esc = escape_rich_markup(", ".join(best_supported))
        console.print(f"   [dim]• Best supported: {best_esc}[/dim]")
    
    # Background logging
    _bg_logger.warning(f"[OCR] {provider_name}: Limited support for languages: {', '.join(warning_langs)}")


def tool_safety_check(tool_name: str, validation_type: str, status: str, details: Optional[str] = None, confidence: Optional[float] = None) -> None:
    """
    Prints a formatted panel for tool safety validation results.
    
    Args:
        tool_name: Name of the tool being validated
        validation_type: Type of validation ("Pre-Execution" or "Post-Execution")
        status: Validation status ("BLOCKED", "ALLOWED", "SAFE", "HARMFUL")
        details: Optional details about the validation
        confidence: Optional confidence score (0.0-1.0)
    """
    tool_name_esc = escape_rich_markup(tool_name)
    validation_type_esc = escape_rich_markup(validation_type)
    details_esc = escape_rich_markup(details) if details else ""
    
    # Determine styling based on status
    if status.upper() in ["BLOCKED", "HARMFUL", "MALICIOUS"]:
        border_style = "bold red"
        title = "[bold red]🛡️ Tool Safety: BLOCKED[/bold red]"
        status_display = f"[red]{status.upper()}[/red]"
    elif status.upper() in ["ALLOWED", "SAFE"]:
        border_style = "bold green"
        title = "[bold green]🛡️ Tool Safety: PASSED[/bold green]"
        status_display = f"[green]{status.upper()}[/green]"
    else:
        border_style = "yellow"
        title = "[yellow]🛡️ Tool Safety: WARNING[/yellow]"
        status_display = f"[yellow]{status.upper()}[/yellow]"
    
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Tool Name:[/bold]", f"[cyan]{tool_name_esc}[/cyan]")
    table.add_row("[bold]Validation Type:[/bold]", f"[cyan]{validation_type_esc}[/cyan]")
    table.add_row("")
    table.add_row("[bold]Status:[/bold]", status_display)
    
    if confidence is not None:
        table.add_row("[bold]Confidence:[/bold]", f"{confidence:.2f}")
    
    if details:
        if len(details_esc) > 150:
            details_esc = details_esc[:147] + "..."
        table.add_row("")
        table.add_row("[bold]Details:[/bold]")
        table.add_row(f"[dim]{details_esc}[/dim]")
    
    panel = Panel(
        table,
        title=title,
        border_style=border_style,
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    # Background logging for Sentry
    _sentry_logger.info(
        "Tool safety validation: %s (%s) - %s",
        tool_name, validation_type, status,
        extra={
            "tool_name": tool_name,
            "validation_type": validation_type,
            "status": status,
            "confidence": confidence,
            "details": details[:200] if details else None
        }
    )


def reflection_started(iteration: int, max_iterations: int) -> None:
    """
    Prints a formatted panel when reflection process starts.
    
    Args:
        iteration: Current reflection iteration (1-based)
        max_iterations: Maximum number of reflection iterations
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Reflection Status:[/bold]", "[cyan]🔄 Started[/cyan]")
    table.add_row("[bold]Iteration:[/bold]", f"[yellow]{iteration}/{max_iterations}[/yellow]")
    table.add_row("[bold]Process:[/bold]", "[green]Evaluating response quality[/green]")
    
    panel = Panel(
        table,
        title="[bold cyan]🔍 Reflection Process Started[/bold cyan]",
        border_style="cyan",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.info(f"Reflection started: iteration {iteration}/{max_iterations}")


def reflection_evaluation(
    iteration: int,
    overall_score: float,
    accuracy: float,
    completeness: float,
    relevance: float,
    clarity: float,
    action: str,
    confidence: float
) -> None:
    """
    Prints a formatted panel for reflection evaluation results.
    
    Args:
        iteration: Current reflection iteration
        overall_score: Overall evaluation score (0-1)
        accuracy: Accuracy score (0-1)
        completeness: Completeness score (0-1)
        relevance: Relevance score (0-1)
        clarity: Clarity score (0-1)
        action: Action taken (ACCEPT, REVISE, RETRY, CLARIFY)
        confidence: Confidence level (0-1)
    """
    # Determine border style based on score
    if overall_score >= 0.8:
        border_style = "green"
        status_emoji = "✅"
        status_text = "High Quality"
    elif overall_score >= 0.6:
        border_style = "yellow"
        status_emoji = "⚠️"
        status_text = "Moderate Quality"
    else:
        border_style = "red"
        status_emoji = "❌"
        status_text = "Needs Improvement"
    
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Iteration:[/bold]", f"[cyan]{iteration}[/cyan]")
    table.add_row("[bold]Overall Score:[/bold]", f"[{border_style}]{overall_score:.2f} - {status_emoji} {status_text}[/]")
    table.add_row("")
    table.add_row("[bold]Criteria Scores:[/bold]", "")
    table.add_row("  Accuracy:", f"[yellow]{accuracy:.2f}[/yellow]")
    table.add_row("  Completeness:", f"[yellow]{completeness:.2f}[/yellow]")
    table.add_row("  Relevance:", f"[yellow]{relevance:.2f}[/yellow]")
    table.add_row("  Clarity:", f"[yellow]{clarity:.2f}[/yellow]")
    table.add_row("")
    table.add_row("[bold]Action:[/bold]", f"[cyan]{action}[/cyan]")
    table.add_row("[bold]Confidence:[/bold]", f"[yellow]{confidence:.2f}[/yellow]")
    
    panel = Panel(
        table,
        title=f"[bold {border_style}]🔍 Reflection Evaluation - Iteration {iteration}[/bold {border_style}]",
        border_style=border_style,
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.info(
        f"Reflection evaluation: iteration {iteration}, score {overall_score:.2f}, action {action}"
    )


def reflection_improvement_started(iteration: int, feedback: str) -> None:
    """
    Prints a formatted panel when starting to improve response based on reflection.
    
    Args:
        iteration: Current reflection iteration
        feedback: Feedback from evaluator
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    feedback_esc = escape_rich_markup(feedback)
    if len(feedback_esc) > 200:
        feedback_esc = feedback_esc[:197] + "..."
    
    table.add_row("[bold]Iteration:[/bold]", f"[cyan]{iteration}[/cyan]")
    table.add_row("[bold]Status:[/bold]", "[yellow]🔄 Generating Improved Response[/yellow]")
    table.add_row("")
    table.add_row("[bold]Feedback:[/bold]", f"[dim]{feedback_esc}[/dim]")
    
    panel = Panel(
        table,
        title="[bold yellow]✨ Reflection Improvement Started[/bold yellow]",
        border_style="yellow",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.info(f"Reflection improvement started: iteration {iteration}")


def reflection_completed(
    final_score: float,
    total_iterations: int,
    termination_reason: str
) -> None:
    """
    Prints a formatted panel when reflection process completes.
    
    Args:
        final_score: Final evaluation score
        total_iterations: Total number of iterations performed
        termination_reason: Reason for termination (acceptance_threshold_met, max_iterations_reached, etc.)
    """
    # Determine border style based on final score
    if final_score >= 0.8:
        border_style = "green"
        status_emoji = "✅"
    elif final_score >= 0.6:
        border_style = "yellow"
        status_emoji = "⚠️"
    else:
        border_style = "red"
        status_emoji = "❌"
    
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    table.add_row("[bold]Status:[/bold]", f"[{border_style}]{status_emoji} Reflection Complete[/]")
    table.add_row("[bold]Final Score:[/bold]", f"[{border_style}]{final_score:.2f}[/]")
    table.add_row("[bold]Total Iterations:[/bold]", f"[cyan]{total_iterations}[/cyan]")
    table.add_row("[bold]Termination Reason:[/bold]", f"[dim]{termination_reason}[/dim]")
    
    panel = Panel(
        table,
        title=f"[bold {border_style}]🎯 Reflection Process Completed[/bold {border_style}]",
        border_style=border_style,
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.info(
        f"Reflection completed: score {final_score:.2f}, iterations {total_iterations}, reason {termination_reason}"
    )

def policy_feedback_generated(
    policy_type: Literal["user_policy", "agent_policy"],
    policy_name: str,
    feedback_message: str,
    retry_count: int,
    max_retries: int,
    violation_reason: Optional[str] = None
) -> None:
    """
    Prints a formatted panel when policy feedback is generated.
    
    Args:
        policy_type: Type of policy ("user_policy" or "agent_policy")
        policy_name: Name of the policy that triggered feedback
        feedback_message: The feedback message generated by LLM
        retry_count: Current retry attempt number
        max_retries: Maximum number of retries allowed
        violation_reason: Optional reason for the policy violation
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    policy_type_esc = escape_rich_markup(policy_type.replace("_", " ").title())
    policy_name_esc = escape_rich_markup(policy_name)
    
    # Truncate feedback message for display
    feedback_preview = feedback_message[:200] if len(feedback_message) > 200 else feedback_message
    feedback_preview_esc = escape_rich_markup(feedback_preview)
    if len(feedback_message) > 200:
        feedback_preview_esc += "..."
    
    table.add_row("[bold]Policy Type:[/bold]", f"[cyan]{policy_type_esc}[/cyan]")
    table.add_row("[bold]Policy Name:[/bold]", f"[cyan]{policy_name_esc}[/cyan]")
    table.add_row("[bold]Retry Attempt:[/bold]", f"[yellow]{retry_count}/{max_retries}[/yellow]")
    table.add_row("")
    
    if violation_reason:
        violation_esc = escape_rich_markup(violation_reason[:100])
        table.add_row("[bold]Violation:[/bold]", f"[red]{violation_esc}[/red]")
        table.add_row("")
    
    table.add_row("[bold]Feedback:[/bold]")
    table.add_row(f"[dim]{feedback_preview_esc}[/dim]")
    
    panel = Panel(
        table,
        title="[bold blue]💬 Policy Feedback Generated[/bold blue]",
        border_style="blue",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.info(
        f"Policy feedback generated: {policy_type}, policy={policy_name}, retry {retry_count}/{max_retries}"
    )
def policy_feedback_retry(
    policy_type: Literal["user_policy", "agent_policy"],
    retry_count: int,
    max_retries: int
) -> None:
    """
    Prints a formatted panel when the agent is retrying due to policy feedback.
    
    Args:
        policy_type: Type of policy ("user_policy" or "agent_policy")
        retry_count: Current retry attempt number
        max_retries: Maximum number of retries allowed
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    policy_type_esc = escape_rich_markup(policy_type.replace("_", " ").title())
    
    table.add_row("[bold]Status:[/bold]", f"[yellow]🔄 Retrying with Feedback[/yellow]")
    table.add_row("[bold]Policy Type:[/bold]", f"[cyan]{policy_type_esc}[/cyan]")
    table.add_row("[bold]Attempt:[/bold]", f"[yellow]{retry_count + 1} of {max_retries}[/yellow]")
    table.add_row("")
    table.add_row("[bold]Action:[/bold]", "[blue]Re-executing model with feedback injected[/blue]")
    
    panel = Panel(
        table,
        title="[bold yellow]🔄 Policy Feedback - Agent Retry[/bold yellow]",
        border_style="yellow",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.info(
        f"Policy feedback retry: {policy_type}, attempt {retry_count + 1}/{max_retries}"
    )
def policy_feedback_exhausted(
    policy_type: Literal["user_policy", "agent_policy"],
    policy_name: str,
    fallback_action: str,
    total_attempts: int
) -> None:
    """
    Prints a formatted panel when policy feedback loop is exhausted and fallback action is applied.
    
    Args:
        policy_type: Type of policy ("user_policy" or "agent_policy")
        policy_name: Name of the policy
        fallback_action: The fallback action being applied (BLOCK, RAISE, etc.)
        total_attempts: Total number of attempts made
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    policy_type_esc = escape_rich_markup(policy_type.replace("_", " ").title())
    policy_name_esc = escape_rich_markup(policy_name)
    fallback_esc = escape_rich_markup(fallback_action)
    
    table.add_row("[bold]Status:[/bold]", f"[red]⚠️ Feedback Loop Exhausted[/red]")
    table.add_row("[bold]Policy Type:[/bold]", f"[cyan]{policy_type_esc}[/cyan]")
    table.add_row("[bold]Policy Name:[/bold]", f"[cyan]{policy_name_esc}[/cyan]")
    table.add_row("[bold]Total Attempts:[/bold]", f"[yellow]{total_attempts}[/yellow]")
    table.add_row("")
    table.add_row("[bold]Fallback Action:[/bold]", f"[red]{fallback_esc}[/red]")
    
    panel = Panel(
        table,
        title="[bold red]⚠️ Policy Feedback - Fallback Applied[/bold red]",
        border_style="red",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.warning(
        f"Policy feedback exhausted: {policy_type}, policy={policy_name}, "
        f"attempts={total_attempts}, fallback={fallback_action}"
    )
def user_policy_feedback_returned(
    policy_name: str,
    feedback_message: str
) -> None:
    """
    Prints a formatted panel when user policy feedback is returned to the user.
    
    Args:
        policy_name: Name of the policy that triggered feedback
        feedback_message: The feedback message being returned to user
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    policy_name_esc = escape_rich_markup(policy_name)
    
    # Truncate feedback for display
    feedback_preview = feedback_message[:250] if len(feedback_message) > 250 else feedback_message
    feedback_preview_esc = escape_rich_markup(feedback_preview)
    if len(feedback_message) > 250:
        feedback_preview_esc += "..."
    
    table.add_row("[bold]Status:[/bold]", f"[green]✓ Feedback Returned to User[/green]")
    table.add_row("[bold]Policy:[/bold]", f"[cyan]{policy_name_esc}[/cyan]")
    table.add_row("")
    table.add_row("[bold]Message to User:[/bold]")
    table.add_row(f"[dim]{feedback_preview_esc}[/dim]")
    
    panel = Panel(
        table,
        title="[bold green]📢 User Policy Feedback[/bold green]",
        border_style="green",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.info(
        f"User policy feedback returned: policy={policy_name}"
    )
def agent_policy_feedback_success(
    policy_name: str,
    total_iterations: int
) -> None:
    """
    Prints a formatted panel when agent policy passes after feedback iterations.
    
    Args:
        policy_name: Name of the policy
        total_iterations: Number of iterations it took to pass
    """
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    policy_name_esc = escape_rich_markup(policy_name)
    
    table.add_row("[bold]Status:[/bold]", f"[green]✓ Policy Passed After Feedback[/green]")
    table.add_row("[bold]Policy:[/bold]", f"[cyan]{policy_name_esc}[/cyan]")
    table.add_row("[bold]Iterations:[/bold]", f"[yellow]{total_iterations}[/yellow]")
    
    panel = Panel(
        table,
        title="[bold green]✅ Agent Policy - Feedback Success[/bold green]",
        border_style="green",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.info(
        f"Agent policy passed after feedback: policy={policy_name}, iterations={total_iterations}"
    )

def planning_todo_list(todo_list: any, debug: bool = False) -> None:
    """
    Prints a formatted panel showing the planning todo list.
    
    Args:
        todo_list: TodoList object containing todos
        debug: Whether debug mode is enabled (only prints if True)
    """
    if not debug:
        return
    
    table = Table(show_header=True, expand=True, box=None)
    table.width = 60
    
    table.add_column("[bold]#[/bold]", justify="center", style="dim", width=4)
    table.add_column("[bold]Status[/bold]", justify="center", width=12)
    table.add_column("[bold]Task[/bold]", justify="left")
    
    # Get todos from the list
    todos = getattr(todo_list, 'todos', [])
    
    for i, todo in enumerate(todos, 1):
        # Get todo attributes
        content = getattr(todo, 'content', str(todo))
        status = getattr(todo, 'status', 'pending')
        todo_id = getattr(todo, 'id', str(i))
        
        # Escape content for Rich
        content_esc = escape_rich_markup(content)
        
        # Format status with color and icon
        if status == "completed":
            status_display = "[green]✓ Done[/green]"
            content_style = f"[dim]{content_esc}[/dim]"
        elif status == "in_progress":
            status_display = "[blue]◐ Active[/blue]"
            content_style = f"[cyan]{content_esc}[/cyan]"
        elif status == "cancelled":
            status_display = "[red]✗ Cancel[/red]"
            content_style = f"[dim strikethrough]{content_esc}[/dim strikethrough]"
        else:  # pending
            status_display = "[yellow]○ Pending[/yellow]"
            content_style = content_esc
        
        table.add_row(f"[dim]{todo_id}[/dim]", status_display, content_style)
    
    panel = Panel(
        table,
        title="[bold blue]📋 Planning Todo List[/bold blue]",
        border_style="blue",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.debug(f"Planning todo list: {len(todos)} items")
def planning_todo_update(
    todo_list: any,
    updated_count: int,
    added_count: int,
    status_counts: dict,
    debug: bool = False
) -> None:
    """
    Prints a formatted panel showing the planning todo list update.
    
    Args:
        todo_list: List of Todo objects (current state after update)
        updated_count: Number of todos that were updated
        added_count: Number of new todos added
        status_counts: Dictionary with status counts (completed, in_progress, pending, cancelled)
        debug: Whether debug mode is enabled (only prints if True)
    """
    if not debug:
        return
    
    # Create todo items table
    todo_table = Table(show_header=True, expand=True, box=None)
    todo_table.add_column("[bold]#[/bold]", justify="center", style="dim", width=4)
    todo_table.add_column("[bold]Status[/bold]", justify="center", width=12)
    todo_table.add_column("[bold]Task[/bold]", justify="left")
    
    # Add each todo
    for i, todo in enumerate(todo_list, 1):
        content = getattr(todo, 'content', str(todo))
        status = getattr(todo, 'status', 'pending')
        todo_id = getattr(todo, 'id', str(i))
        
        content_esc = escape_rich_markup(content)
        
        # Format status with color and icon
        if status == "completed":
            status_display = "[green]✓ Done[/green]"
            content_style = f"[dim]{content_esc}[/dim]"
        elif status == "in_progress":
            status_display = "[blue]◐ Active[/blue]"
            content_style = f"[cyan]{content_esc}[/cyan]"
        elif status == "cancelled":
            status_display = "[red]✗ Cancel[/red]"
            content_style = f"[dim strikethrough]{content_esc}[/dim strikethrough]"
        else:  # pending
            status_display = "[yellow]○ Pending[/yellow]"
            content_style = content_esc
        
        todo_table.add_row(f"[dim]{todo_id}[/dim]", status_display, content_style)
    
    # Add spacing
    todo_table.add_row("", "", "")
    
    # Progress summary
    total_todos = sum(status_counts.values())
    completed = status_counts.get("completed", 0)
    progress_pct = int((completed / total_todos) * 100) if total_todos > 0 else 0
    
    bar_length = 20
    filled = int(bar_length * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    # Update summary line
    update_info = []
    if updated_count > 0:
        update_info.append(f"[cyan]{updated_count} updated[/cyan]")
    if added_count > 0:
        update_info.append(f"[green]{added_count} added[/green]")
    update_text = " | ".join(update_info) if update_info else "[dim]No changes[/dim]"
    
    todo_table.add_row("", "[bold]Changes:[/bold]", update_text)
    todo_table.add_row("", "[bold]Progress:[/bold]", f"[green]{bar}[/green] {progress_pct}%")
    
    panel = Panel(
        todo_table,
        title="[bold magenta]📝 Plan Updated[/bold magenta]",
        border_style="magenta",
        expand=True,
        width=70
    )
    
    console.print(panel)
    spacing()
    
    _bg_logger.debug(f"Planning todo update: {updated_count} updated, {added_count} added")