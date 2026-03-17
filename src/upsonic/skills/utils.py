"""Utility functions for the skills module."""

import os
import platform
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def is_safe_path(base_dir: Path, requested_path: str) -> bool:
    """Check if the requested path stays within the base directory.

    Prevents path traversal attacks (e.g. ``../../../etc/passwd``).

    Args:
        base_dir: The base directory that the path must stay within.
        requested_path: The user-provided path to validate.

    Returns:
        True if the path is safe (stays within *base_dir*), False otherwise.
    """
    try:
        full_path = (base_dir / requested_path).resolve()
        base_resolved = base_dir.resolve()
        return full_path.is_relative_to(base_resolved)
    except (ValueError, OSError):
        return False


def ensure_executable(file_path: Path) -> None:
    """Ensure a file has the executable bit set for the owner.

    Args:
        file_path: Path to the file to make executable.
    """
    current_mode = file_path.stat().st_mode
    if not (current_mode & stat.S_IXUSR):
        os.chmod(file_path, current_mode | stat.S_IXUSR)


def parse_shebang(script_path: Path) -> Optional[str]:
    """Parse the shebang line from a script to determine the interpreter.

    Handles common formats::

        #!/usr/bin/env python3  -> "python3"
        #!/usr/bin/python3      -> "python3"
        #!/bin/bash             -> "bash"
        #!/usr/bin/env -S node  -> "node"

    Returns:
        The interpreter name or ``None`` if no valid shebang is found.
    """
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
    except (OSError, UnicodeDecodeError):
        return None

    if not first_line.startswith("#!"):
        return None

    shebang = first_line[2:].strip()
    if not shebang:
        return None

    parts = shebang.split()

    # Handle /usr/bin/env style shebangs
    if Path(parts[0]).name == "env":
        for part in parts[1:]:
            if not part.startswith("-"):
                return part
        return None

    # Direct path shebangs like #!/bin/bash
    return Path(parts[0]).name


def get_interpreter_command(interpreter: str) -> List[str]:
    """Map an interpreter name to a platform-compatible command.

    Python interpreters are mapped to ``sys.executable`` so the current
    virtual-environment is respected.
    """
    if interpreter.lower() in ("python", "python3", "python2"):
        return [sys.executable]
    return [interpreter]


def _build_windows_command(script_path: Path, args: List[str]) -> List[str]:
    """Build the command list for executing a script on Windows.

    Windows does not process shebang lines, so we parse it ourselves
    and invoke the interpreter explicitly.
    """
    interpreter = parse_shebang(script_path)
    if interpreter:
        cmd_prefix = get_interpreter_command(interpreter)
        return [*cmd_prefix, str(script_path), *args]
    # Fallback: direct execution (may fail but gives a clear error)
    return [str(script_path), *args]


@dataclass
class ScriptResult:
    """Result of a script execution."""

    stdout: str
    stderr: str
    returncode: int


def run_script(
    script_path: Path,
    args: Optional[List[str]] = None,
    timeout: int = 30,
    cwd: Optional[Path] = None,
) -> ScriptResult:
    """Execute a script and return the result.

    On Unix the script is executed directly via its shebang.
    On Windows the shebang is parsed to find the interpreter.

    Args:
        script_path: Path to the script to execute.
        args: Optional arguments to pass to the script.
        timeout: Maximum execution time in seconds.
        cwd: Working directory for the script.

    Returns:
        A :class:`ScriptResult` with stdout, stderr and returncode.

    Raises:
        subprocess.TimeoutExpired: If the script exceeds *timeout*.
        FileNotFoundError: If script or interpreter is not found.
    """
    if platform.system() == "Windows":
        cmd = _build_windows_command(script_path, args or [])
    else:
        # On Unix, try shebang first; if missing, infer interpreter from extension
        interpreter = parse_shebang(script_path)
        if interpreter:
            cmd_prefix = get_interpreter_command(interpreter)
            cmd = [*cmd_prefix, str(script_path), *(args or [])]
        elif script_path.suffix in (".py",):
            cmd = [sys.executable, str(script_path), *(args or [])]
        elif script_path.suffix in (".sh", ".bash"):
            cmd = ["bash", str(script_path), *(args or [])]
        else:
            ensure_executable(script_path)
            cmd = [str(script_path), *(args or [])]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )

    return ScriptResult(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def read_file_safe(file_path: Path, encoding: str = "utf-8") -> str:
    """Read a file's contents safely.

    Raises:
        FileNotFoundError: If file doesn't exist.
        PermissionError: If file can't be read.
        UnicodeDecodeError: If file can't be decoded.
    """
    return file_path.read_text(encoding=encoding)
