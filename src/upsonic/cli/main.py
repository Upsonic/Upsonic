"""Main CLI entry point for Upsonic - Optimized for speed."""

import sys
from typing import Optional


# Command dispatch table for O(1) lookup
_COMMAND_HANDLERS = {
    'init': lambda args: _handle_init(args),
    'add': lambda args: _handle_add(args),
    'remove': lambda args: _handle_remove(args),
    'install': lambda args: _handle_install(args),
    'run': lambda args: _handle_run(args),
    'zip': lambda args: _handle_zip(args),
}


def _handle_init(args: list[str]) -> int:
    """Handle 'init' command with lazy import."""
    # Check for help flag
    if len(args) >= 2 and args[1] in ("--help", "-h"):
        from upsonic.cli.printer import print_help_init
        print_help_init()
        return 0
    
    from upsonic.cli.commands import init_command
    return init_command()


def _handle_add(args: list[str]) -> int:
    """Handle 'add' command with lazy import."""
    # Check for help flag
    if len(args) >= 2 and args[1] in ("--help", "-h"):
        from upsonic.cli.printer import print_help_add
        print_help_add()
        return 0
    
    if len(args) < 3:
        from upsonic.cli.printer import print_error
        print_error("Usage: upsonic add <library> <section>\nExample: upsonic add x_library==0.52.0 api")
        return 1
    from upsonic.cli.commands import add_command
    return add_command(args[1], args[2])


def _handle_remove(args: list[str]) -> int:
    """Handle 'remove' command with lazy import."""
    # Check for help flag
    if len(args) >= 2 and args[1] in ("--help", "-h"):
        from upsonic.cli.printer import print_help_remove
        print_help_remove()
        return 0
    
    if len(args) < 3:
        from upsonic.cli.printer import print_error
        print_error("Usage: upsonic remove <library> <section>\nExample: upsonic remove requests api")
        return 1
    from upsonic.cli.commands import remove_command
    return remove_command(args[1], args[2])


def _handle_install(args: list[str]) -> int:
    """Handle 'install' command with lazy import."""
    # Check for help flag
    if len(args) >= 2 and args[1] in ("--help", "-h"):
        from upsonic.cli.printer import print_help_install
        print_help_install()
        return 0
    
    section = args[1] if len(args) >= 2 else None
    from upsonic.cli.commands import install_command
    return install_command(section)


def _handle_run(args: list[str]) -> int:
    """Handle 'run' command with lazy import and optimized arg parsing."""
    # Check for help flag
    if len(args) >= 2 and args[1] in ("--help", "-h"):
        from upsonic.cli.printer import print_help_run
        print_help_run()
        return 0
    
    host = "0.0.0.0"
    port = 8000
    
    # Fast argument parsing using iteration
    i = 1
    while i < len(args):
        arg = args[i]
        if arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif arg == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
                i += 2
            except ValueError:
                from upsonic.cli.printer import print_error
                print_error(f"Invalid port: {args[i + 1]}")
                return 1
        else:
            i += 1
    
    from upsonic.cli.commands import run_command
    return run_command(host=host, port=port)


def _handle_zip(args: list[str]) -> int:
    """Handle 'zip' command with lazy import."""
    # Check for help flag
    if len(args) >= 2 and args[1] in ("--help", "-h"):
        from upsonic.cli.printer import print_help_zip
        print_help_zip()
        return 0
    
    output_file = args[1] if len(args) >= 2 else None
    from upsonic.cli.commands import zip_command
    return zip_command(output_file)


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for the Upsonic CLI.
    
    Optimized with lazy imports for lightning-fast command execution.
    Only imports what's needed for the specific command being run.
    
    Args:
        args: Command line arguments. If None, uses sys.argv[1:].
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    if args is None:
        args = sys.argv[1:]
    
    # Fast path: no args - show usage
    if not args:
        from upsonic.cli.printer import print_usage
        print_usage()
        return 0
    
    # Check for general help flag
    if args[0] in ("--help", "-h"):
        from upsonic.cli.printer import print_help_general
        print_help_general()
        return 0
    
    command = args[0]
    
    # O(1) command dispatch using dict lookup
    handler = _COMMAND_HANDLERS.get(command)
    if handler:
        return handler(args)
    
    # Unknown command
    from upsonic.cli.printer import print_unknown_command
    print_unknown_command(command)
    return 1

