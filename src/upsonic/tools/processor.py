from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Callable, Any, Generator, Tuple

# Assuming the tool.py file is in the same directory.
from .tool import ToolConfig

# We leverage the existing rich console for beautiful user interaction.
from upsonic.utils.printing import console, spacing


class ToolValidationError(Exception):
    """Custom exception raised for invalid tool definitions."""
    pass


class ToolProcessor:
    """
    The internal engine for inspecting, validating, normalizing, and wrapping
    user-provided tools into a format the agent can execute.
    """

    def _validate_function(self, func: Callable):
        """
        Inspects a function to ensure it meets the requirements for a valid tool.
        A valid tool must have type hints for all parameters, a return type hint,
        and a non-empty docstring.

        Raises:
            ToolValidationError: If the function fails validation.
        """
        signature = inspect.signature(func)

        # 1. Validate parameter type hints
        for param_name, param in signature.parameters.items():
            if param.name in ('self', 'cls'):
                continue
            if param.annotation is inspect.Parameter.empty:
                raise ToolValidationError(
                    f"Tool '{func.__name__}' is missing a type hint for parameter '{param_name}'."
                )

        # 2. Validate return type hint
        if signature.return_annotation is inspect.Signature.empty:
            raise ToolValidationError(
                f"Tool '{func.__name__}' is missing a return type hint."
            )

        # 3. Validate docstring
        if not inspect.getdoc(func):
            raise ToolValidationError(
                f"Tool '{func.__name__}' is missing a docstring. The docstring is required to explain the tool's purpose to the LLM."
            )

    def normalize_and_process(self, task_tools: list) -> Generator[Tuple[Callable, ToolConfig], None, None]:
        """
        Processes a list of raw tools from a Task.

        This method iterates through functions and object methods, validates them,
        resolves their `ToolConfig` (either from a decorator or by applying a default),
        and yields a standardized tuple of (callable, config).

        Args:
            task_tools: The raw list of tools from `task.tools`.

        Yields:
            A tuple containing the callable function/method and its resolved ToolConfig.
        """
        if not task_tools:
            return

        for tool_item in task_tools:
            # Case 1: The tool is a standalone function
            if inspect.isfunction(tool_item):
                self._validate_function(tool_item)
                # Resolve config: use existing if decorator was used, else create a default one.
                config = getattr(tool_item, '_upsonic_tool_config', ToolConfig())
                yield (tool_item, config)

            # Case 2: The tool is a class instance (an object)
            elif not inspect.isfunction(tool_item) and hasattr(tool_item, '__class__'):
                # Find all public methods on the object
                for name, method in inspect.getmembers(tool_item, inspect.ismethod):
                    # Ignore private/dunder methods
                    if not name.startswith('_'):
                        self._validate_function(method)
                        # Resolve config for the method
                        config = getattr(method, '_upsonic_tool_config', ToolConfig())
                        yield (method, config)

    def generate_behavioral_wrapper(self, original_func: Callable, config: ToolConfig) -> Callable:
        """
        Dynamically generates and returns a new function that wraps the original tool.
        This new function contains all the behavioral logic (caching, confirmation, etc.)
        defined in the ToolConfig.
        (This is the method provided in the previous step, included here for completeness.)
        """
        @functools.wraps(original_func)
        def behavioral_wrapper(*args: Any, **kwargs: Any) -> Any:
            # --- 1. Pre-execution Hooks ---
            if config.tool_hooks and config.tool_hooks.before:
                config.tool_hooks.before(*args, **kwargs)

            # --- 2. User Confirmation Gate ---
            if config.requires_confirmation:
                console.print(f"[bold yellow]⚠️ Confirmation Required[/bold yellow]")
                console.print(f"About to execute tool: [cyan]{original_func.__name__}[/cyan]")
                console.print(f"With arguments: [dim]{args}, {kwargs}[/dim]")
                try:
                    confirm = input("Do you want to proceed? (y/n): ").lower().strip()
                except KeyboardInterrupt:
                    confirm = 'n'
                    print()
                if confirm not in ['y', 'yes']:
                    console.print("[bold red]Tool execution cancelled by user.[/bold red]")
                    spacing()
                    return "Tool execution was cancelled by the user."
                spacing()

            # --- 3. Interactive User Input ---
            if config.requires_user_input and config.user_input_fields:
                console.print(f"[bold blue]📝 User Input Required for {original_func.__name__}[/bold blue]")
                for field_name in config.user_input_fields:
                    try:
                        user_provided_value = input(f"Please provide a value for '{field_name}': ")
                        kwargs[field_name] = user_provided_value
                    except KeyboardInterrupt:
                        console.print("\n[bold red]Input cancelled by user.[/bold red]")
                        return "Tool execution was cancelled by the user during input."
                spacing()
            
            # --- 4. Cache Read Logic ---
            cache_file = None
            if config.cache_results:
                cache_dir_path = Path(config.cache_dir) if config.cache_dir else Path.home() / '.upsonic' / 'cache'
                arg_string = json.dumps((args, kwargs), sort_keys=True, default=str)
                call_signature = f"{original_func.__name__}:{arg_string}".encode('utf-8')
                cache_key = hashlib.sha256(call_signature).hexdigest()
                cache_file = cache_dir_path / f"{cache_key}.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                        is_expired = False
                        if config.cache_ttl is not None:
                            age = time.time() - cache_data.get('timestamp', 0)
                            if age > config.cache_ttl:
                                is_expired = True
                        if not is_expired:
                            console.print(f"[bold green]✓ Cache Hit[/bold green] for tool [cyan]{original_func.__name__}[/cyan]. Returning cached result.")
                            spacing()
                            return cache_data['result']
                        else:
                            cache_file.unlink()
                    except (json.JSONDecodeError, KeyError, OSError):
                        pass

            # --- 5. CORE TOOL EXECUTION ---
            try:
                result = original_func(*args, **kwargs)
            except Exception as e:
                console.print(f"[bold red]An error occurred while executing tool '{original_func.__name__}': {e}[/bold red]")
                raise

            # --- 6. Cache Write Logic ---
            if config.cache_results and cache_file:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_data_to_write = {'timestamp': time.time(), 'result': result}
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data_to_write, f, indent=2, default=str)
                except (TypeError, OSError) as e:
                    console.print(f"[yellow]Warning: Could not cache result for tool '{original_func.__name__}'. Reason: {e}[/yellow]")

            # --- 7. Post-execution Hooks ---
            if config.tool_hooks and config.tool_hooks.after:
                config.tool_hooks.after(result)

            # --- 8. Set Flow-Control Flags ---
            setattr(behavioral_wrapper, '_upsonic_stop_after_call', config.stop_after_tool_call)
            setattr(behavioral_wrapper, '_upsonic_show_result', config.show_result)
            
            return result
        return behavioral_wrapper