"""Main CLI entry point for Upsonic."""

import sys
from typing import Optional

from upsonic.cli.commands import init_command, add_command, run_command
from upsonic.cli.printer import print_usage, print_unknown_command, print_error


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for the Upsonic CLI.
    
    Args:
        args: Command line arguments. If None, uses sys.argv[1:].
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    if args is None:
        args = sys.argv[1:]
    
    if not args:
        print_usage()
        return 0
    
    command = args[0]
    
    if command == "init":
        return init_command()
    elif command == "add":
        if len(args) < 3:
            print_error("Usage: upsonic add <library> <section>\nExample: upsonic add x_library==0.52.0 api")
            return 1
        library = args[1]
        section = args[2]
        return add_command(library, section)
    elif command == "run":
        # Parse optional host and port arguments
        host = "0.0.0.0"
        port = 8000
        if len(args) >= 2:
            # Check if it's --host or --port
            i = 1
            while i < len(args):
                if args[i] == "--host" and i + 1 < len(args):
                    host = args[i + 1]
                    i += 2
                elif args[i] == "--port" and i + 1 < len(args):
                    try:
                        port = int(args[i + 1])
                        i += 2
                    except ValueError:
                        print_error(f"Invalid port: {args[i + 1]}")
                        return 1
                else:
                    i += 1
        return run_command(host=host, port=port)
    else:
        print_unknown_command(command)
        return 1

