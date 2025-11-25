"""Beautiful printing utilities for the Upsonic CLI using Rich - Optimized for speed."""

# Lazy import cache for Rich components
_RICH_IMPORTS = None


def _get_rich_imports():
    """
    Lazy load Rich library components only when needed.
    
    This defers the loading of the Rich library until the first
    time a print function is called, significantly improving CLI startup time.
    """
    global _RICH_IMPORTS
    if _RICH_IMPORTS is None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt, Confirm
        from rich.markup import escape
        from rich.table import Table
        from rich import box
        
        _RICH_IMPORTS = {
            'Console': Console,
            'Panel': Panel,
            'Prompt': Prompt,
            'Confirm': Confirm,
            'escape': escape,
            'Table': Table,
            'box': box,
            'console': Console(),
        }
    return _RICH_IMPORTS


def _escape_rich_markup(text: str) -> str:
    """Escape text to prevent Rich markup interpretation."""
    rich = _get_rich_imports()
    return rich['escape'](str(text))


def prompt_agent_name() -> str:
    """Prompt user for agent name with styled input."""
    rich = _get_rich_imports()
    console = rich['console']
    Prompt = rich['Prompt']
    
    console.print()
    console.print("[bold cyan]🤖 Upsonic Agent Initialization[/bold cyan]")
    console.print()
    agent_name = Prompt.ask("[bold]Agent Name[/bold]", default="")
    return agent_name.strip()


def print_error(message: str) -> None:
    """Print an error message in a styled panel."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    panel = Panel(
        f"[bold red]{_escape_rich_markup(message)}[/bold red]",
        title="[bold red]❌ Error[/bold red]",
        border_style="red",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_success(message: str) -> None:
    """Print a success message in a styled panel."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    panel = Panel(
        f"[bold green]{_escape_rich_markup(message)}[/bold green]",
        title="[bold green]✅ Success[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_info(message: str) -> None:
    """Print an info message."""
    rich = _get_rich_imports()
    console = rich['console']
    console.print(f"[cyan]ℹ[/cyan] [bold]{_escape_rich_markup(message)}[/bold]")


def print_file_created(file_path: str) -> None:
    """Print a message indicating a file was created."""
    rich = _get_rich_imports()
    console = rich['console']
    console.print(f"[green]✓[/green] [bold]Created[/bold] [cyan]{_escape_rich_markup(str(file_path))}[/cyan]")


def confirm_overwrite(file_path: str) -> bool:
    """Ask user to confirm overwriting an existing file."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    Confirm = rich['Confirm']
    box = rich['box']
    
    console.print()
    panel = Panel(
        f"[yellow]⚠[/yellow]  [bold]{_escape_rich_markup(str(file_path))}[/bold] already exists.",
        title="[bold yellow]File Exists[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    )
    console.print(panel)
    return Confirm.ask("[bold]Overwrite?[/bold]", default=False)


def print_cancelled() -> None:
    """Print a cancellation message."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    panel = Panel(
        "[yellow]Operation cancelled by user.[/yellow]",
        title="[bold yellow]⚠ Cancelled[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_init_success(agent_name: str, files_created: list[str]) -> None:
    """Print a beautiful success message after initialization."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    Table = rich['Table']
    box = rich['box']
    
    console.print()
    
    # Create a table with the created files
    table = Table(show_header=True, box=box.ROUNDED, border_style="green")
    table.add_column("[bold]File[/bold]", style="cyan", no_wrap=True)
    table.add_column("[bold]Status[/bold]", style="green", justify="center")
    
    for file_path in files_created:
        table.add_row(
            _escape_rich_markup(str(file_path)),
            "[bold green]✓ Created[/bold green]"
        )
    
    # Print agent name
    console.print(f"[bold]Agent Name:[/bold] [cyan]{_escape_rich_markup(agent_name)}[/cyan]")
    console.print()
    
    # Print table in a panel
    panel = Panel(
        table,
        title="[bold green]🎉 Upsonic Agent Initialized Successfully![/bold green]",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_usage() -> None:
    """Print CLI usage information."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    Table = rich['Table']
    box = rich['box']
    
    console.print()
    
    table = Table(show_header=True, box=box.ROUNDED, border_style="cyan")
    table.add_column("[bold]Command[/bold]", style="cyan", no_wrap=True)
    table.add_column("[bold]Description[/bold]", style="white")
    
    table.add_row(
        "[bold]init[/bold]",
        "Initialize a new Upsonic agent project"
    )
    table.add_row(
        "[bold]add[/bold]",
        "Add a dependency to upsonic_configs.json"
    )
    table.add_row(
        "[bold]install[/bold]",
        "Install dependencies from upsonic_configs.json"
    )
    table.add_row(
        "[bold]run[/bold]",
        "Run the agent as a FastAPI server"
    )
    
    panel = Panel(
        table,
        title="[bold cyan]🚀 Upsonic CLI[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_unknown_command(command: str) -> None:
    """Print error for unknown command."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    panel = Panel(
        f"[bold red]Unknown command:[/bold red] [yellow]{_escape_rich_markup(command)}[/yellow]\n\n"
        "[bold]Available commands:[/bold] [cyan]init[/cyan], [cyan]add[/cyan], [cyan]install[/cyan], [cyan]run[/cyan]",
        title="[bold red]❌ Error[/bold red]",
        border_style="red",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_dependency_added(library: str, section: str) -> None:
    """Print success message when a dependency is added."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    panel = Panel(
        f"[bold green]✓ Added[/bold green] [cyan]{_escape_rich_markup(library)}[/cyan] to [bold]dependencies.{_escape_rich_markup(section)}[/bold]",
        title="[bold green]✅ Dependency Added[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_config_not_found() -> None:
    """Print error when upsonic_configs.json is not found."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    panel = Panel(
        "[bold red]upsonic_configs.json not found![/bold red]\n\n"
        "Please run [cyan]upsonic init[/cyan] first to create the configuration file.",
        title="[bold red]❌ Configuration Not Found[/bold red]",
        border_style="red",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_invalid_section(section: str, available_sections: list[str]) -> None:
    """Print error for invalid dependency section."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    sections_str = ", ".join([f"[cyan]{s}[/cyan]" for s in available_sections])
    console.print()
    panel = Panel(
        f"[bold red]Invalid section:[/bold red] [yellow]{_escape_rich_markup(section)}[/yellow]\n\n"
        f"[bold]Available sections:[/bold] {sections_str}",
        title="[bold red]❌ Invalid Section[/bold red]",
        border_style="red",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()

