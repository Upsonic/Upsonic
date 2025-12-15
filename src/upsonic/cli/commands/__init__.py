"""
CLI commands module.

This module exports all command functions for backward compatibility.
Commands are organized in separate folders under commands/.
"""

# Import all command functions to maintain backward compatibility
from upsonic.cli.commands.init.command import init_command
from upsonic.cli.commands.add.command import add_command
from upsonic.cli.commands.remove.command import remove_command
from upsonic.cli.commands.install.command import install_command
from upsonic.cli.commands.run.command import run_command
from upsonic.cli.commands.zip.command import zip_command

__all__ = [
    "init_command",
    "add_command",
    "remove_command",
    "install_command",
    "run_command",
    "zip_command",
]

