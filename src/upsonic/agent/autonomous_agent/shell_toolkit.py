"""
Autonomous Agent Shell Toolkit - Command execution operations.

Provides production-ready tools for shell command execution with:
- Timeout support
- Working directory enforcement
- Environment variable management
- Output capturing and formatting
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from upsonic.tools import ToolKit, tool


class AutonomousShellToolKit(ToolKit):
    """
    Shell command execution toolkit for AutonomousAgent.
    
    Provides safe command execution with:
    - Timeout support
    - Working directory enforcement
    - Environment variable management
    - Output capturing and formatting
    - Security command blocking
    
    Features:
    - Workspace sandboxing (commands run in workspace directory)
    - Configurable timeouts to prevent hanging
    - Combined stdout/stderr capture
    - Exit code reporting
    - Output truncation for large results
    
    Usage:
        ```python
        from upsonic import AutonomousAgent
        
        agent = AutonomousAgent(workspace="/path/to/project")
        result = agent.do("Run the test suite")
        ```
    """
    
    def __init__(
        self,
        workspace: Path,
        default_timeout: int = 120,
        max_output_length: int = 10000,
        allowed_commands: Optional[List[str]] = None,
        blocked_commands: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize shell toolkit.
        
        Args:
            workspace: Working directory for command execution
            default_timeout: Default command timeout in seconds
            max_output_length: Maximum output length before truncation
            allowed_commands: If set, only these commands are allowed (whitelist)
            blocked_commands: Commands that are blocked (blacklist)
        """
        super().__init__()
        self.workspace: Path = Path(workspace).resolve()
        self.default_timeout: int = default_timeout
        self.max_output_length: int = max_output_length
        self.allowed_commands: Optional[List[str]] = allowed_commands
        self.blocked_commands: List[str] = blocked_commands or [
            "rm -rf /",
            "rm -rf /*",
            ":(){:|:&};:",
            "mkfs",
            "dd if=/dev/zero",
        ]
    
    def _validate_command(self, command: str) -> Optional[str]:
        """
        Validate command against security rules.
        
        Returns:
            Error message if command is blocked, None if allowed
        """
        cmd_lower = command.lower().strip()
        
        for blocked in self.blocked_commands:
            if blocked.lower() in cmd_lower:
                return f"Command blocked for security: contains '{blocked}'"
        
        if self.allowed_commands is not None:
            cmd_parts = command.split()
            if cmd_parts:
                base_cmd = cmd_parts[0]
                if base_cmd not in self.allowed_commands:
                    return f"Command not in allowed list: '{base_cmd}'"
        
        return None
    
    @tool
    def run_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = True,
    ) -> str:
        """
        Execute a shell command in the workspace directory.
        
        Commands are executed with the workspace as the working directory.
        Output is captured and returned along with the exit code.
        
        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (defaults to 120). Set to None for no timeout.
            env: Additional environment variables to set
            shell: If True, run through shell (allows pipes, etc.)
        
        Returns:
            Command output (stdout + stderr) and exit code
            
        Example:
            run_command("python --version")
            run_command("pip install -r requirements.txt")
            run_command("ls -la")
        """
        validation_error = self._validate_command(command)
        if validation_error:
            return f"Error: {validation_error}"
        
        effective_timeout = timeout if timeout is not None else self.default_timeout
        
        environment = os.environ.copy()
        if env:
            environment.update(env)
        
        try:
            result = subprocess.run(
                command,
                shell=shell,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=environment,
            )
            
            output_parts: List[str] = []
            
            if result.stdout:
                stdout = result.stdout
                if len(stdout) > self.max_output_length:
                    stdout = stdout[:self.max_output_length] + f"\n... [truncated, {len(result.stdout)} total chars]"
                output_parts.append(f"STDOUT:\n{stdout}")
            
            if result.stderr:
                stderr = result.stderr
                if len(stderr) > self.max_output_length:
                    stderr = stderr[:self.max_output_length] + f"\n... [truncated, {len(result.stderr)} total chars]"
                output_parts.append(f"STDERR:\n{stderr}")
            
            if not output_parts:
                output_parts.append("(no output)")
            
            exit_indicator = "✅" if result.returncode == 0 else "❌"
            output_parts.append(f"\n{exit_indicator} Exit code: {result.returncode}")
            
            return "\n".join(output_parts)
            
        except subprocess.TimeoutExpired:
            return f"❌ Error: Command timed out after {effective_timeout} seconds"
        except FileNotFoundError as e:
            return f"❌ Error: Command not found: {e}"
        except PermissionError:
            return "❌ Error: Permission denied"
        except Exception as e:
            return f"❌ Error running command: {str(e)}"
    
    @tool
    async def arun_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = True,
    ) -> str:
        """
        Async version of run_command.
        
        Execute a shell command asynchronously.
        
        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            env: Additional environment variables
            shell: Run through shell
        
        Returns:
            Command output and exit code
        """
        validation_error = self._validate_command(command)
        if validation_error:
            return f"Error: {validation_error}"
        
        effective_timeout = timeout if timeout is not None else self.default_timeout
        
        environment = os.environ.copy()
        if env:
            environment.update(env)
        
        try:
            if shell:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(self.workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=environment,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *command.split(),
                    cwd=str(self.workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=environment,
                )
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"❌ Error: Command timed out after {effective_timeout} seconds"
            
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            
            output_parts: List[str] = []
            
            if stdout:
                if len(stdout) > self.max_output_length:
                    stdout = stdout[:self.max_output_length] + "\n... [truncated]"
                output_parts.append(f"STDOUT:\n{stdout}")
            
            if stderr:
                if len(stderr) > self.max_output_length:
                    stderr = stderr[:self.max_output_length] + "\n... [truncated]"
                output_parts.append(f"STDERR:\n{stderr}")
            
            if not output_parts:
                output_parts.append("(no output)")
            
            exit_code = proc.returncode if proc.returncode is not None else -1
            exit_indicator = "✅" if exit_code == 0 else "❌"
            output_parts.append(f"\n{exit_indicator} Exit code: {exit_code}")
            
            return "\n".join(output_parts)
            
        except FileNotFoundError as e:
            return f"❌ Error: Command not found: {e}"
        except PermissionError:
            return "❌ Error: Permission denied"
        except Exception as e:
            return f"❌ Error running command: {str(e)}"
    
    @tool
    def run_python(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Execute Python code directly.
        
        Runs the provided Python code using the Python interpreter.
        Useful for quick computations, testing snippets, or running scripts.
        
        Args:
            code: Python code to execute
            timeout: Timeout in seconds
        
        Returns:
            Code output and exit status
            
        Example:
            run_python("print(2 + 2)")
            run_python("import sys; print(sys.version)")
        """
        escaped_code = code.replace("'", "'\"'\"'")
        command = f"python3 -c '{escaped_code}'"
        return self.run_command(command, timeout=timeout)
    
    @tool
    async def arun_python(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> str:
        """Async version of run_python."""
        escaped_code = code.replace("'", "'\"'\"'")
        command = f"python3 -c '{escaped_code}'"
        return await self.arun_command(command, timeout=timeout)
    
    @tool
    def check_command_exists(self, command: str) -> str:
        """
        Check if a command is available in the system.
        
        Args:
            command: Command name to check
        
        Returns:
            Information about the command availability
        """
        result = subprocess.run(
            f"which {command}",
            shell=True,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            return f"✅ Command '{command}' is available at: {result.stdout.strip()}"
        else:
            return f"❌ Command '{command}' is not available"
    
    @tool
    async def acheck_command_exists(self, command: str) -> str:
        """Async version of check_command_exists."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.check_command_exists(command)
        )
