from typing import Any, Dict, Literal, Optional, Union
from upsonic.models import Model
from decimal import Decimal
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
import platform
from rich.markup import escape


console = Console()

# TODO: Implement proper implementation of get_estimated_cost
# Temporary stub for get_estimated_cost until proper implementation
def get_estimated_cost(input_tokens: int, output_tokens: int, model_provider: Union[Model, str]) -> str:
    """Estimate cost based on tokens. Returns approximate cost string."""
    # Very rough estimates - should be replaced with actual pricing
    input_cost_per_1m = 0.50  # $0.50 per 1M input tokens (rough average)
    output_cost_per_1m = 1.50  # $1.50 per 1M output tokens (rough average)
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost
    
    return f"~{total_cost:.4f}"

# Global dictionary to store aggregated values by price_id
price_id_summary = {}

def spacing():
    console.print("")


def escape_rich_markup(text):
    """Escape special characters in text to prevent Rich markup interpretation"""
    if text is None:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Use Rich's built-in escape function
    return escape(text)


def connected_to_server(server_type: str, status: str, total_time: float = None):
    """
    Prints a 'Connected to Server' section for Upsonic, full width,
    with two columns: 
      - left column (labels) left-aligned
      - right column (values) left-aligned, positioned on the right half 
    """

    # Escape input text
    server_type = escape_rich_markup(server_type)

    # Determine color and symbol for the status
    if status.lower() == "established":
        status_text = "[green]✓ Established[/green]"
    elif status.lower() == "failed":
        status_text = "[red]✗ Failed[/red]"
    else:
        status_text = f"[cyan]… {escape_rich_markup(status)}[/cyan]"

    # Build a table that expands to full console width
    table = Table(show_header=False, expand=True, box=None)
    
    # We define 2 columns, each with ratio=1 so they evenly split the width
    # Both are left-aligned, but the second column will end up on the right half.
    table.add_column("Label", justify="left", ratio=1)
    table.add_column("Value", justify="left", ratio=1)

    # Rows: one for server type, one for status
    table.add_row("[bold]Server Type:[/bold]", f"[yellow]{server_type}[/yellow]")
    table.add_row("[bold]Connection Status:[/bold]", status_text)
    
    if total_time is not None:
        table.add_row("[bold]Total Time:[/bold]", f"[cyan]{total_time:.2f} seconds[/cyan]")

    table.width = 60

    # Wrap the table in a Panel that also expands full width
    panel = Panel(
        table, 
        title="[bold cyan]Upsonic - Server Connection[/bold cyan]",
        border_style="cyan",
        expand=True,  # panel takes the full terminal width
        width=70  # Adjust as preferred
    )

    # Print the panel (it will fill the entire width, with two columns inside)
    console.print(panel)

    spacing()

def call_end(result: Any, model_provider: Any, response_format: str, start_time: float, end_time: float, usage: dict, tool_usage: list, debug: bool = False, price_id: str = None):
    # First panel for tool usage if there are any tools used
    if tool_usage and len(tool_usage) > 0:
        tool_table = Table(show_header=True, expand=True, box=None)
        tool_table.width = 60
        
        # Add columns for the tool usage table
        tool_table.add_column("[bold]Tool Name[/bold]", justify="left")
        tool_table.add_column("[bold]Parameters[/bold]", justify="left")
        tool_table.add_column("[bold]Result[/bold]", justify="left")

        # Add each tool usage as a row
        for tool in tool_usage:
            tool_name = escape_rich_markup(str(tool.get('tool_name', '')))
            params = escape_rich_markup(str(tool.get('params', '')))
            result_str = escape_rich_markup(str(tool.get('tool_result', '')))
            
            # Truncate long strings
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

    # Second panel with main results
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    # Escape input values
    display_model_name = escape_rich_markup(model_provider.model_name)
    response_format = escape_rich_markup(response_format)
    price_id_display = escape_rich_markup(price_id) if price_id else None

    # Track values if price_id is provided
    if price_id:
        estimated_cost = get_estimated_cost(usage['input_tokens'], usage['output_tokens'], model_provider)
        if price_id not in price_id_summary:
            price_id_summary[price_id] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'estimated_cost': 0.0
            }
        price_id_summary[price_id]['input_tokens'] += usage['input_tokens']
        price_id_summary[price_id]['output_tokens'] += usage['output_tokens']
        # Extract the numeric value from the estimated_cost string
        # Handle the format from get_estimated_cost which returns "~0.0123"
        try:
            # Remove tilde and dollar sign (if present) and convert to Decimal
            cost_str = str(estimated_cost).replace('~', '').replace('$', '').strip()
            # The cost is already calculated as a number - use it directly
            if isinstance(price_id_summary[price_id]['estimated_cost'], (float, int)):
                price_id_summary[price_id]['estimated_cost'] += float(cost_str)
            else:
                # For Decimal objects or other types
                from decimal import Decimal
                price_id_summary[price_id]['estimated_cost'] = Decimal(str(price_id_summary[price_id]['estimated_cost'])) + Decimal(cost_str)
        except Exception as e:
            # If there's any error in cost calculation, log it but continue
            if debug:
                pass  # Error calculating cost

    result_str = str(result)
    # Limit result to 370 characters
    if not debug:
        result_str = result_str[:370]
    # Add ellipsis if result is truncated
    if len(result_str) < len(str(result)):
        result_str += "[bold white]...[/bold white]"

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



def agent_end(result: Any, model_provider: Any, response_format: str, start_time: float, end_time: float, usage: dict, tool_usage: list, tool_count: int, context_count: int, debug: bool = False, price_id:str = None):
    # First panel for tool usage if there are any tools used
    if tool_usage and len(tool_usage) > 0:
        tool_table = Table(show_header=True, expand=True, box=None)
        tool_table.width = 60
        
        # Add columns for the tool usage table
        tool_table.add_column("[bold]Tool Name[/bold]", justify="left")
        tool_table.add_column("[bold]Parameters[/bold]", justify="left")
        tool_table.add_column("[bold]Result[/bold]", justify="left")

        # Add each tool usage as a row
        for tool in tool_usage:
            tool_name = escape_rich_markup(str(tool.get('tool_name', '')))
            params = escape_rich_markup(str(tool.get('params', '')))
            result_str = escape_rich_markup(str(tool.get('tool_result', '')))
            
            # Truncate long strings
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

    # Main result panel
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    # Escape input values
    display_model_name = escape_rich_markup(model_provider.model_name)
    response_format = escape_rich_markup(response_format)
    price_id = escape_rich_markup(price_id) if price_id else None

    # Track values if price_id is provided
    if price_id:
        estimated_cost = get_estimated_cost(usage['input_tokens'], usage['output_tokens'], model_provider)
        if price_id not in price_id_summary:
            price_id_summary[price_id] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'estimated_cost': 0.0
            }
        price_id_summary[price_id]['input_tokens'] += usage['input_tokens']
        price_id_summary[price_id]['output_tokens'] += usage['output_tokens']
        # Extract the numeric value from the estimated_cost string
        # Handle the format from get_estimated_cost which returns "~0.0123"
        try:
            # Remove tilde and dollar sign (if present) and convert to Decimal
            cost_str = str(estimated_cost).replace('~', '').replace('$', '').strip()
            # The cost is already calculated as a number - use it directly
            if isinstance(price_id_summary[price_id]['estimated_cost'], (float, int)):
                price_id_summary[price_id]['estimated_cost'] += float(cost_str)
            else:
                # If it's stored as Decimal already
                price_id_summary[price_id]['estimated_cost'] = Decimal(str(price_id_summary[price_id]['estimated_cost'])) + Decimal(cost_str)
        except Exception as e:
            # Fallback in case of conversion errors
            console.print(f"[bold red]Warning: Could not parse cost value: {estimated_cost}. Error: {e}[/bold red]")

    table.add_row("[bold]LLM Model:[/bold]", f"{display_model_name}")
    # Add spacing
    table.add_row("")
    result_str = str(result)
    # Limit result to 370 characters
    if not debug:
        result_str = result_str[:370]
    # Add ellipsis if result is truncated
    if len(result_str) < len(str(result)):
        result_str += "[bold white]...[/bold white]"

    table.add_row("[bold]Result:[/bold]", f"[green]{escape_rich_markup(result_str)}[/green]")
    # Add spacing
    table.add_row("")
    table.add_row("[bold]Response Format:[/bold]", f"{response_format}")
    
    table.add_row("[bold]Tools:[/bold]", f"{tool_count} [bold]Context Used:[/bold]", f"{context_count}")
    table.add_row("[bold]Estimated Cost:[/bold]", f"{get_estimated_cost(usage['input_tokens'], usage['output_tokens'], model_provider)}$")
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


def agent_total_cost(total_input_tokens: int, total_output_tokens: int, total_time: float, model_provider: Any):
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    # Escape input values
    llm_model = escape_rich_markup(model_provider.model_name)

    table.add_row("[bold]Estimated Cost:[/bold]", f"{get_estimated_cost(total_input_tokens, total_output_tokens, model_provider)}$")
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
    # Escape input values
    price_id_display = escape_rich_markup(price_id)
    task_display = escape_rich_markup(str(task))
    
    if price_id not in price_id_summary:
        console.print("[bold red]Price ID not found![/bold red]")
        return None
    
    summary = price_id_summary[price_id].copy()
    # Format the estimated cost to include $ symbol
    summary['estimated_cost'] = f"${summary['estimated_cost']:.4f}"

    # Create a table for pretty printing
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Price ID:[/bold]", f"[magenta]{price_id_display}[/magenta]")
    table.add_row("")  # Add spacing
    table.add_row("[bold]Input Tokens:[/bold]", f"[magenta]{summary['input_tokens']:,}[/magenta]")
    table.add_row("[bold]Output Tokens:[/bold]", f"[magenta]{summary['output_tokens']:,}[/magenta]")
    table.add_row("[bold]Total Estimated Cost:[/bold]", f"[magenta]{summary['estimated_cost']}[/magenta]")
    
    # Add total time if available from task
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
    
    # Format the operation text
    operation_text = f"[bold cyan]{escape_rich_markup(operation)}[/bold cyan]"
    table.add_row(operation_text)
    
    # If there's a result, add it to the table
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
    
    # Add error code if provided
    if error_code:
        table.add_row("[bold]Error Code:[/bold]", f"[red]{error_code}[/red]")
        table.add_row("")  # Add spacing
    
    # Add error details
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
    
    # Escape input values
    tool_name = escape_rich_markup(tool_name)
    missing_deps = [escape_rich_markup(dep) for dep in missing_deps]
    
    # Create the installation command
    install_cmd = "pip install " + " ".join(missing_deps)
    
    # Create a bulleted list of dependencies
    deps_list = "\n".join([f"  • [bold white]{dep}[/bold white]" for dep in missing_deps])
    
    # Create the content with the dependencies list and installation command
    content = f"[bold red]Missing Dependencies for {tool_name}:[/bold red]\n\n{deps_list}\n\n[bold green]Installation Command:[/bold green]\n  {install_cmd}"
    
    # Create and print the panel
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
    # Escape input values
    tool_name = escape_rich_markup(tool_name)
    env_var_name = escape_rich_markup(env_var_name)
    
    # Determine the operating system
    system = platform.system()
    
    # Create OS-specific instructions for setting the API key
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
    
    # Create and print the panel
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
    
    # Format the operation text
    operation_text = f"[bold magenta]{escape_rich_markup(operation)}[/bold magenta]"
    table.add_row(operation_text)
    
    # If there's a result, add it to the table
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

    # Add columns for the tool usage table
    tool_table.add_column("[bold]Tool Name[/bold]", justify="left")
    tool_table.add_column("[bold]Parameters[/bold]", justify="left")
    tool_table.add_column("[bold]Result[/bold]", justify="left")

    # Escape and format the data for display
    tool_name_str = escape_rich_markup(str(tool_name))
    params_str = escape_rich_markup(str(params))
    result_str = escape_rich_markup(str(result))
    
    # Truncate long strings for clean display
    if len(params_str) > 50:
        params_str = params_str[:47] + "..."
    if len(result_str) > 50:
        result_str = result_str[:47] + "..."
            
    tool_table.add_row(
        f"[cyan]{tool_name_str}[/cyan]",
        f"[yellow]{params_str}[/yellow]",
        f"[green]{result_str}[/green]"
    )

    # Use a more descriptive title for clarity
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
    
    # Escape input values
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
    
    # Escape input values
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
    
    # Escape input values
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
    
    # Extract and escape values
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
    
    # Escape input values
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
    
    # Escape input values
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