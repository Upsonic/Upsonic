from typing import Any
from decimal import Decimal
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
from rich.markup import escape
from .price import get_estimated_cost
import platform


console = Console()

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

def call_end(result: Any, llm_model: str, response_format: str, start_time: float, end_time: float, usage: dict, tool_usage: list, debug: bool = False, price_id: str = None):
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
    llm_model = escape_rich_markup(llm_model)
    response_format = escape_rich_markup(response_format)
    price_id_display = escape_rich_markup(price_id) if price_id else None

    # Track values if price_id is provided
    if price_id:
        estimated_cost = get_estimated_cost(usage['input_tokens'], usage['output_tokens'], llm_model)
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
                print(f"Error calculating cost: {e}")

    table.add_row("[bold]LLM Model:[/bold]", f"{llm_model}")
    # Add spacing
    table.add_row("")

    from ..client.level_two.agent import SubTaskList, SearchResult, CompanyObjective, HumanObjective

    is_it_subtask = isinstance(result, SubTaskList)
    is_it_search = isinstance(result, SearchResult)
    is_it_company = isinstance(result, CompanyObjective)
    is_it_human = isinstance(result, HumanObjective)

    if is_it_subtask:
        # Print total task count
        table.add_row(f"[bold]Total Subtasks:[/bold]", f"[yellow]{len(result.sub_tasks)}[/yellow]")
        table.add_row("")
        # Print each task as well as bullet list
        for each in result.sub_tasks:
            table.add_row(f"[bold]Subtask:[/bold]", f"[green]{escape_rich_markup(each.description)}[/green]")
            table.add_row(f"[bold]Required Output:[/bold]", f"[green]{escape_rich_markup(each.required_output)}[/green]")
            table.add_row(f"[bold]Tools:[/bold]", f"[green]{escape_rich_markup(each.tools)}[/green]")
            table.add_row("")
    elif is_it_search:
        table.add_row("[bold]Has Customers:[/bold]", f"[green]{'Yes' if result.any_customers else 'No'}[/green]")
        table.add_row("")
        table.add_row("[bold]Products:[/bold]")
        for product in result.products:
            table.add_row("", f"[green]• {escape_rich_markup(product)}[/green]")
        table.add_row("")
        table.add_row("[bold]Services:[/bold]")
        for service in result.services:
            table.add_row("", f"[green]• {escape_rich_markup(service)}[/green]")
        table.add_row("")
        table.add_row("[bold]Potential Competitors:[/bold]")
        for competitor in result.potential_competitors:
            table.add_row("", f"[yellow]• {escape_rich_markup(competitor)}[/yellow]")
        table.add_row("")
    elif is_it_company:
        table.add_row("[bold]Company Objective:[/bold]", f"[blue]{escape_rich_markup(result.objective)}[/blue]")
        table.add_row("")
        table.add_row("[bold]Goals:[/bold]")
        for goal in result.goals:
            table.add_row("", f"[blue]• {escape_rich_markup(goal)}[/blue]")
        table.add_row("")
        table.add_row("[bold]State:[/bold]", f"[blue]{escape_rich_markup(result.state)}[/blue]")
        table.add_row("")
    elif is_it_human:
        table.add_row("[bold]Job Title:[/bold]", f"[magenta]{escape_rich_markup(result.job_title)}[/magenta]")
        table.add_row("")
        table.add_row("[bold]Job Description:[/bold]", f"[magenta]{escape_rich_markup(result.job_description)}[/magenta]")
        table.add_row("")
        table.add_row("[bold]Job Goals:[/bold]")
        for goal in result.job_goals:
            table.add_row("", f"[magenta]• {escape_rich_markup(goal)}[/magenta]")
        table.add_row("")
    else:
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

    table.add_row("[bold]Estimated Cost:[/bold]", f"{get_estimated_cost(usage['input_tokens'], usage['output_tokens'], llm_model)}$")
    time_taken = end_time - start_time
    time_taken_str = f"{time_taken:.2f} seconds"
    table.add_row("[bold]Time Taken:[/bold]", f"{time_taken_str}")
    panel = Panel(
        table,
        title="[bold white]Upsonic - Call Result[/bold white]",
        border_style="white",
        expand=True,
        width=70
    )

    console.print(panel)
    spacing()



def agent_end(result: Any, llm_model: str, response_format: str, start_time: float, end_time: float, usage: dict, tool_usage: list, tool_count: int, context_count: int, debug: bool = False, price_id:str = None):
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
    llm_model = escape_rich_markup(llm_model)
    response_format = escape_rich_markup(response_format)
    price_id = escape_rich_markup(price_id) if price_id else None

    # Track values if price_id is provided
    if price_id:
        estimated_cost = get_estimated_cost(usage['input_tokens'], usage['output_tokens'], llm_model)
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

    table.add_row("[bold]LLM Model:[/bold]", f"{llm_model}")
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
    table.add_row("[bold]Estimated Cost:[/bold]", f"{get_estimated_cost(usage['input_tokens'], usage['output_tokens'], llm_model)}$")
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


def agent_total_cost(total_input_tokens: int, total_output_tokens: int, total_time: float, llm_model: str):
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60
    
    # Escape input values
    llm_model = escape_rich_markup(llm_model)

    table.add_row("[bold]Estimated Cost:[/bold]", f"{get_estimated_cost(total_input_tokens, total_output_tokens, llm_model)}$")
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

    panel = Panel(
        table,
        title="[bold magenta]Upsonic - Price ID Summary[/bold magenta]",
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