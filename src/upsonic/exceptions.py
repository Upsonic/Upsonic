"""
Upsonic custom exceptions for better error handling and user experience.
"""

import platform
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape

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


class UpsonicError(Exception):
    """Base exception for all Upsonic-related errors."""
    pass


class APIKeyMissingError(UpsonicError):
    """
    Exception raised when an API key is missing for a model provider.
    
    This exception automatically displays a beautiful error panel with
    instructions on how to set the missing API key.
    """
    
    def __init__(self, provider_name: str, env_var_name: str, dotenv_support: bool = True):
        self.provider_name = provider_name
        self.env_var_name = env_var_name
        self.dotenv_support = dotenv_support
        
        # Display the cool error panel
        self._display_error_panel()
        
        # Set the exception message
        message = f"API key not found. Please provide it directly or set the {env_var_name} environment variable."
        super().__init__(message)
    
    def _display_error_panel(self):
        """Display a formatted panel with API key setup instructions."""
        # Escape input values
        tool_name = escape(self.provider_name)
        env_var_name = escape(self.env_var_name)
        
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
        
        if self.dotenv_support:
            dotenv_instructions = f"Create a .env file in your project directory with:\n  {env_var_name}=your_api_key_here"
            content = f"[bold red]Missing API Key for {tool_name}[/bold red]\n\n[bold white]The {env_var_name} environment variable is not set.[/bold white]\n\n{env_description}\n\n[bold green]Option 2: Use a .env file:[/bold green]\n  {dotenv_instructions}"
        else:
            content = f"[bold red]Missing API Key for {tool_name}[/bold red]\n\n[bold white]The {env_var_name} environment variable is not set.[/bold white]\n\n{env_description}"
        
        # Use safe character for Windows compatibility
        key_char = "🔑" if platform.system() != "Windows" else "[KEY]"
        
        # Create and print the panel
        panel = Panel(content, title=f"[bold yellow]{key_char} API Key Required[/bold yellow]", border_style="yellow", expand=False)
        console.print(panel)


class ConfigurationError(UpsonicError):
    """
    Exception raised when there's a configuration issue (e.g., missing Azure endpoints).
    
    This exception automatically displays a formatted error panel.
    """
    
    def __init__(self, error_type: str, detail: str, error_code: int = None):
        self.error_type = error_type
        self.detail = detail
        self.error_code = error_code
        
        # Display the cool error panel
        self._display_error_panel()
        
        # Set the exception message
        super().__init__(detail)
    
    def _display_error_panel(self):
        """Display a formatted error panel for configuration errors."""
        table = Table(show_header=False, expand=True, box=None)
        table.width = 60
        
        # Add error code if provided
        if self.error_code:
            table.add_row("[bold]Error Code:[/bold]", f"[red]{self.error_code}[/red]")
            table.add_row("")  # Add spacing
        
        # Add error details
        table.add_row("[bold]Error Details:[/bold]")
        table.add_row(f"[red]{escape(self.detail)}[/red]")
        
        panel = Panel(
            table,
            title=f"[bold red]Upsonic - {escape(self.error_type)}[/bold red]",
            border_style="red",
            expand=True,
            width=70
        )
        
        console.print(panel)


class ProviderError(UpsonicError):
    """Exception raised when there's an issue with a model provider."""
    pass


class ModelNotFoundError(ProviderError):
    """Exception raised when a requested model is not available."""
    pass


class RateLimitError(UpsonicError):
    """Exception raised when API rate limits are exceeded."""
    pass


class AuthenticationError(UpsonicError):
    """Exception raised when API authentication fails."""
    pass


class FileNotFoundError(UpsonicError):
    """Exception raised when a file specified in context cannot be accessed."""
    
    def __init__(self, file_path: str, reason: str = "File not found"):
        self.file_path = file_path
        self.reason = reason
        message = f"File not found: {file_path}. {reason}"
        super().__init__(message)