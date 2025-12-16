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
        from rich.text import Text
        from rich.style import Style
        from rich import box
        
        _RICH_IMPORTS = {
            'Console': Console,
            'Panel': Panel,
            'Prompt': Prompt,
            'Confirm': Confirm,
            'escape': escape,
            'Table': Table,
            'Text': Text,
            'Style': Style,
            'box': box,
            'console': Console(force_terminal=True, color_system="auto"),
        }
    return _RICH_IMPORTS


def _escape_rich_markup(text: str) -> str:
    """Escape text to prevent Rich markup interpretation."""
    rich = _get_rich_imports()
    return rich['escape'](str(text))


def print_banner() -> None:
    """Print the Upsonic CLI banner with ASCII art."""
    # Use ANSI escape codes directly to ensure green color works
    GREEN_BOLD = "\033[1;32m"  # Bold green
    RESET = "\033[0m"  # Reset color
    BOLD = "\033[1m"  # Bold
    BLUE = "\033[34m"  # Blue
    
    print()
    print(f"{GREEN_BOLD}██╗   ██╗██████╗ ███████╗ ██████╗ ███╗   ██╗██╗ ██████╗{RESET}")
    print(f"{GREEN_BOLD}██║   ██║██╔══██╗██╔════╝██╔═══██╗████╗  ██║██║██╔════╝{RESET}")
    print(f"{GREEN_BOLD}██║   ██║██████╔╝███████╗██║   ██║██╔██╗ ██║██║██║{RESET}")
    print(f"{GREEN_BOLD}██║   ██║██╔═══╝ ╚════██║██║   ██║██║╚██╗██║██║██║{RESET}")
    print(f"{GREEN_BOLD}╚██████╔╝██║     ███████║╚██████╔╝██║ ╚████║██║╚██████╗{RESET}")
    print(f"{GREEN_BOLD} ╚═════╝ ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝ ╚═════╝{RESET}")
    print()
    # Center the CLI text - UPSONIC is 60 chars wide, CLI is 20 chars, so pad with 16 spaces on left
    print(f"{BOLD}                ██████╗██╗     ██╗{RESET}")
    print(f"{BOLD}               ██╔════╝██║     ██║{RESET}")
    print(f"{BOLD}               ██║     ██║     ██║{RESET}")
    print(f"{BOLD}               ██║     ██║     ██║{RESET}")
    print(f"{BOLD}               ╚██████╗███████╗██║{RESET}")
    print(f"{BOLD}                ╚═════╝╚══════╝╚═╝{RESET}")
    print()
    print(f"{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print()


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
    print_banner()
    
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    Table = rich['Table']
    box = rich['box']
    
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
    table.add_row(
        "[bold]zip[/bold]",
        "Create a zip file with the current directory context"
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
        "[bold]Available commands:[/bold] [cyan]init[/cyan], [cyan]add[/cyan], [cyan]remove[/cyan], [cyan]install[/cyan], [cyan]run[/cyan], [cyan]zip[/cyan]",
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


def print_dependency_removed(library: str, section: str) -> None:
    """Print success message when a dependency is removed."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    panel = Panel(
        f"[bold green]✓ Removed[/bold green] [cyan]{_escape_rich_markup(library)}[/cyan] from [bold]dependencies.{_escape_rich_markup(section)}[/bold]",
        title="[bold green]✅ Dependency Removed[/bold green]",
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


def print_help_general() -> None:
    """Print general help for Upsonic CLI."""
    print_banner()
    
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    Table = rich['Table']
    box = rich['box']
    
    table = Table(show_header=True, box=box.ROUNDED, border_style="cyan", show_lines=False)
    table.add_column("[bold]Command[/bold]", style="cyan", no_wrap=True, width=12)
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
        "[bold]remove[/bold]",
        "Remove a dependency from upsonic_configs.json"
    )
    table.add_row(
        "[bold]install[/bold]",
        "Install dependencies from upsonic_configs.json"
    )
    table.add_row(
        "[bold]run[/bold]",
        "Run the agent as a FastAPI server"
    )
    table.add_row(
        "[bold]zip[/bold]",
        "Create a zip file with the current directory context"
    )
    
    panel = Panel(
        table,
        title="[bold cyan]🚀 Upsonic CLI[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()
    console.print("[bold]Usage:[/bold] [cyan]upsonic <command> [options][/cyan]")
    console.print("[bold]Help:[/bold] [cyan]upsonic <command> --help[/cyan] or [cyan]upsonic <command> -h[/cyan]")
    console.print()


def print_help_init() -> None:
    """Print help for init command."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    
    help_text = (
        "[bold]Description:[/bold]\n"
        "Initialize a new Upsonic agent project in the current directory.\n"
        "This command will prompt you for an agent name and create the following files:\n\n"
        "  • [cyan]main.py[/cyan] - Main agent file with async main() function\n"
        "  • [cyan]upsonic_configs.json[/cyan] - Configuration file with agent settings\n\n"
        "[bold]Usage:[/bold]\n"
        "  [cyan]upsonic init[/cyan]\n\n"
        "[bold]What it does:[/bold]\n"
        "  • Prompts for an agent name\n"
        "  • Creates project structure with default templates\n"
        "  • Sets up configuration with default dependencies\n"
        "  • Configures input/output schemas in config file\n\n"
        "[bold]Note:[/bold] If files already exist, you will be prompted to overwrite them."
    )
    
    panel = Panel(
        help_text,
        title="[bold cyan]📦 upsonic init[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_help_add() -> None:
    """Print help for add command."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    
    help_text = (
        "[bold]Description:[/bold]\n"
        "Add a dependency to the upsonic_configs.json file.\n\n"
        "[bold]Usage:[/bold]\n"
        "  [cyan]upsonic add <library> <section>[/cyan]\n\n"
        "[bold]Arguments:[/bold]\n"
        "  [bold]library[/bold]  Library name with optional version specification\n"
        "              Example: [cyan]x_library==0.52.0[/cyan] or [cyan]requests>=2.28.0[/cyan]\n"
        "  [bold]section[/bold]  Dependency section name (api, streamlit, or development)\n\n"
        "[bold]Examples:[/bold]\n"
        "  [cyan]upsonic add requests==2.31.0 api[/cyan]\n"
        "  [cyan]upsonic add pandas streamlit[/cyan]\n"
        "  [cyan]upsonic add pytest development[/cyan]\n\n"
        "[bold]Available sections:[/bold]\n"
        "  • [cyan]api[/cyan] - API runtime dependencies\n"
        "  • [cyan]streamlit[/cyan] - Streamlit UI dependencies\n"
        "  • [cyan]development[/cyan] - Development tools and testing dependencies\n\n"
        "[bold]Note:[/bold] Requires upsonic_configs.json to exist. Run [cyan]upsonic init[/cyan] first."
    )
    
    panel = Panel(
        help_text,
        title="[bold cyan]➕ upsonic add[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_help_remove() -> None:
    """Print help for remove command."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    
    help_text = (
        "[bold]Description:[/bold]\n"
        "Remove a dependency from the upsonic_configs.json file.\n\n"
        "[bold]Usage:[/bold]\n"
        "  [cyan]upsonic remove <library> <section>[/cyan]\n\n"
        "[bold]Arguments:[/bold]\n"
        "  [bold]library[/bold]  Library name to remove (exact match or package name)\n"
        "              Example: [cyan]requests[/cyan] or [cyan]requests==2.31.0[/cyan]\n"
        "  [bold]section[/bold]  Dependency section name (api, streamlit, or development)\n\n"
        "[bold]Examples:[/bold]\n"
        "  [cyan]upsonic remove requests api[/cyan]          # Removes requests (any version)\n"
        "  [cyan]upsonic remove pandas streamlit[/cyan]      # Removes pandas\n\n"
        "[bold]Note:[/bold] Requires upsonic_configs.json to exist."
    )
    
    panel = Panel(
        help_text,
        title="[bold cyan]➖ upsonic remove[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_help_install() -> None:
    """Print help for install command."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    
    help_text = (
        "[bold]Description:[/bold]\n"
        "Install dependencies from upsonic_configs.json.\n"
        "Uses [cyan]uv[/cyan] (preferred) or falls back to [cyan]pip[/cyan].\n\n"
        "[bold]Usage:[/bold]\n"
        "  [cyan]upsonic install[/cyan]              # Install 'api' dependencies (default)\n"
        "  [cyan]upsonic install <section>[/cyan]    # Install specific section\n"
        "  [cyan]upsonic install all[/cyan]          # Install all dependencies\n\n"
        "[bold]Arguments:[/bold]\n"
        "  [bold]section[/bold]  (optional) Dependency section to install\n"
        "              Options: [cyan]api[/cyan], [cyan]streamlit[/cyan], [cyan]development[/cyan], [cyan]all[/cyan]\n"
        "              Default: [cyan]api[/cyan]\n\n"
        "[bold]Examples:[/bold]\n"
        "  [cyan]upsonic install[/cyan]              # Install API dependencies\n"
        "  [cyan]upsonic install api[/cyan]          # Same as above\n"
        "  [cyan]upsonic install streamlit[/cyan]    # Install Streamlit dependencies\n"
        "  [cyan]upsonic install development[/cyan]   # Install development dependencies\n"
        "  [cyan]upsonic install all[/cyan]           # Install all dependencies\n\n"
        "[bold]Note:[/bold] Requires upsonic_configs.json to exist. Run [cyan]upsonic init[/cyan] first."
    )
    
    panel = Panel(
        help_text,
        title="[bold cyan]📥 upsonic install[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_help_run() -> None:
    """Print help for run command."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    
    help_text = (
        "[bold]Description:[/bold]\n"
        "Run the agent as a FastAPI server with automatic OpenAPI documentation.\n"
        "Dynamically builds API schema from upsonic_configs.json input/output schemas.\n\n"
        "[bold]Usage:[/bold]\n"
        "  [cyan]upsonic run[/cyan]                          # Run on default host:port\n"
        "  [cyan]upsonic run --host <host>[/cyan]            # Specify host\n"
        "  [cyan]upsonic run --port <port>[/cyan]            # Specify port\n"
        "  [cyan]upsonic run --host <host> --port <port>[/cyan]  # Specify both\n\n"
        "[bold]Options:[/bold]\n"
        "  [bold]--host[/bold]  Server host address (default: [cyan]0.0.0.0[/cyan])\n"
        "  [bold]--port[/bold]  Server port number (default: [cyan]8000[/cyan])\n\n"
        "[bold]Examples:[/bold]\n"
        "  [cyan]upsonic run[/cyan]                          # http://localhost:8000\n"
        "  [cyan]upsonic run --port 3000[/cyan]              # http://localhost:3000\n"
        "  [cyan]upsonic run --host 127.0.0.1 --port 8080[/cyan]  # http://127.0.0.1:8080\n\n"
        "[bold]Features:[/bold]\n"
        "  • Automatic OpenAPI/Swagger documentation at [cyan]/docs[/cyan]\n"
        "  • Supports both [cyan]multipart/form-data[/cyan] and [cyan]application/json[/cyan]\n"
        "  • Dynamic schema generation from config\n"
        "  • Interactive API testing via Swagger UI\n\n"
        "[bold]Note:[/bold] Requires upsonic_configs.json and main.py to exist."
    )
    
    panel = Panel(
        help_text,
        title="[bold cyan]🚀 upsonic run[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_help_zip() -> None:
    """Print help for zip command."""
    rich = _get_rich_imports()
    console = rich['console']
    Panel = rich['Panel']
    box = rich['box']
    
    console.print()
    
    help_text = (
        "[bold]Description:[/bold]\n"
        "Create a zip archive containing all files in the current directory.\n"
        "Useful for backing up or sharing your Upsonic agent project.\n\n"
        "[bold]Usage:[/bold]\n"
        "  [cyan]upsonic zip[/cyan]                    # Auto-generate filename with timestamp\n"
        "  [cyan]upsonic zip <filename>[/cyan]         # Specify output filename\n\n"
        "[bold]Arguments:[/bold]\n"
        "  [bold]filename[/bold]  (optional) Name of the output zip file\n"
        "              If not provided, defaults to: [cyan]upsonic_context_YYYYMMDD_HHMMSS.zip[/cyan]\n"
        "              If provided without .zip extension, it will be added automatically\n\n"
        "[bold]Examples:[/bold]\n"
        "  [cyan]upsonic zip[/cyan]                           # upsonic_context_20240101_120000.zip\n"
        "  [cyan]upsonic zip my_agent[/cyan]                   # my_agent.zip\n"
        "  [cyan]upsonic zip backup.zip[/cyan]                 # backup.zip\n\n"
        "[bold]What it includes:[/bold]\n"
        "  • All files in the current directory and subdirectories\n"
        "  • Preserves directory structure\n"
        "  • Excludes the output zip file itself (if it exists)\n\n"
        "[bold]Output:[/bold]\n"
        "  Shows file count, total size, and archive size after creation."
    )
    
    panel = Panel(
        help_text,
        title="[bold cyan]📦 upsonic zip[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()

