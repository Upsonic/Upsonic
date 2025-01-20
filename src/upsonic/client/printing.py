from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from .price import get_estimated_cost

console = Console()

def spacing():
    console.print("")



def connected_to_server(server_type: str, status: str):
    """
    Prints a 'Connected to Server' section for Upsonic, full width,
    with two columns: 
      - left column (labels) left-aligned
      - right column (values) left-aligned, positioned on the right half 
    """

    # Determine color and symbol for the status
    if status.lower() == "established":
        status_text = "[green]✓ Established[/green]"
    elif status.lower() == "failed":
        status_text = "[red]✗ Failed[/red]"
    else:
        status_text = f"[cyan]… {status}[/cyan]"

    # Build a table that expands to full console width
    table = Table(show_header=False, expand=True, box=None)
    
    # We define 2 columns, each with ratio=1 so they evenly split the width
    # Both are left-aligned, but the second column will end up on the right half.
    table.add_column("Label", justify="left", ratio=1)
    table.add_column("Value", justify="left", ratio=1)

    # Rows: one for server type, one for status
    table.add_row("[bold]Server Type:[/bold]", f"[yellow]{server_type}[/yellow]")
    table.add_row("[bold]Connection Status:[/bold]", status_text)

    table.width = 60

    # Wrap the table in a Panel that also expands full width
    panel = Panel(
        table, 
        title="[bold cyan]Upsonic - Server Connection[/bold cyan]",
        border_style="cyan",
        expand=True,  # panel takes the full terminal width
        width=50  # Adjust as preferred
    )

    # Print the panel (it will fill the entire width, with two columns inside)
    console.print(panel)

    spacing()

def call_end(result: Any, llm_model: str, response_format: str, start_time: float, end_time: float, usage: dict, debug: bool = False):
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

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

    table.add_row("[bold]Result:[/bold]", f"[green]{result_str}[/green]")
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
        width=50
    )

    console.print(panel)
    spacing()



def agent_end(result: Any, llm_model: str, response_format: str, start_time: float, end_time: float, usage: dict, tool_count: int, context_count: int, debug: bool = False):
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

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

    table.add_row("[bold]Result:[/bold]", f"[green]{result_str}[/green]")
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
        width=50
    )

    console.print(panel)
    spacing()


def agent_total_cost(total_input_tokens: int, total_output_tokens: int, total_time: float, llm_model: str):
    table = Table(show_header=False, expand=True, box=None)
    table.width = 60

    table.add_row("[bold]Estimated Cost:[/bold]", f"{get_estimated_cost(total_input_tokens, total_output_tokens, llm_model)}$")
    table.add_row("[bold]Time Taken:[/bold]", f"{total_time:.2f} seconds")
    panel = Panel(
        table,
        title="[bold white]Upsonic - Agent Total Cost[/bold white]",
        border_style="white",
        expand=True,
        width=50
    )
    console.print(panel)
    spacing()