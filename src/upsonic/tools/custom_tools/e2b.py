"""
E2B Code Interpreter Toolkit for Upsonic Framework.

This module provides E2B sandbox integration, allowing agents to:
- Execute Python, JavaScript, Java, R, and Bash code in isolated cloud sandboxes
- Upload and download files to/from the sandbox
- Run shell commands in the sandbox
- Install packages dynamically
- Manage sandbox lifecycle (timeout, pause, resume, kill)

Required Environment Variables:
-----------------------------
- E2B_API_KEY: E2B API key from https://e2b.dev

Example Usage:
    ```python
    from upsonic import Agent, Task
    from upsonic.tools.custom_tools.e2b import E2BTools

    tools = E2BTools(api_key="e2b-YOUR-API-KEY")
    agent = Agent("openai/gpt-4o", tools=[tools])
    task = Task("Calculate the first 20 Fibonacci numbers using Python")
    agent.print_do(task)
    ```
"""

import base64
import json
from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Optional

from upsonic.tools.base import ToolKit
from upsonic.tools.config import tool
from upsonic.utils.printing import error_log

try:
    from e2b_code_interpreter import Sandbox
    _E2B_AVAILABLE = True
except ImportError:
    Sandbox = None
    _E2B_AVAILABLE = False


class E2BTools(ToolKit):
    """E2B sandbox toolkit for code execution, file operations, and shell commands."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 300,
        sandbox_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the E2B toolkit.

        Args:
            api_key: E2B API key. Falls back to E2B_API_KEY env var.
            timeout: Sandbox timeout in seconds (default: 300). Max 86400 for Pro, 3600 for Hobby.
            sandbox_options: Additional options for Sandbox.create() (e.g. template, envs, metadata).
            **kwargs: ToolKit params (include_tools, exclude_tools, timeout, etc.).
        """
        super().__init__(**kwargs)

        if not _E2B_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="e2b-code-interpreter",
                install_command="pip install e2b-code-interpreter",
                feature_name="E2B tools",
            )

        self.api_key: str = api_key or getenv("E2B_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "E2B API key is required. Set E2B_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.sandbox_options: Dict[str, Any] = sandbox_options or {}
        self._timeout: int = timeout
        self._sandbox: Optional[Any] = None

    @property
    def sandbox(self) -> Any:
        """Lazy-create the sandbox on first access."""
        if self._sandbox is None:
            self._sandbox = Sandbox.create(
                api_key=self.api_key,
                timeout=self._timeout,
                **self.sandbox_options,
            )
        return self._sandbox

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def ae2b_run_code(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
        envs: Optional[Dict[str, str]] = None,
    ) -> str:
        try:
            kwargs: Dict[str, Any] = {}
            if language != "python":
                kwargs["language"] = language
            if timeout is not None:
                kwargs["timeout"] = timeout
            if envs is not None:
                kwargs["envs"] = envs

            execution = self.sandbox.run_code(code, **kwargs)
            return self._format_execution(execution)
        except Exception as e:
            error_log(f"E2B run_code error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_run_command(self, command: str, timeout: Optional[int] = None) -> str:
        try:
            kwargs: Dict[str, Any] = {}
            if timeout is not None:
                kwargs["timeout"] = timeout

            result = self.sandbox.commands.run(command, **kwargs)
            return self._format_command_result(result)
        except Exception as e:
            error_log(f"E2B run_command error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_upload_file(self, local_path: str, sandbox_path: Optional[str] = None) -> str:
        try:
            dest = sandbox_path or f"/home/user/{Path(local_path).name}"
            with open(local_path, "rb") as f:
                info = self.sandbox.files.write(dest, f)
            return json.dumps({"path": info.path})
        except Exception as e:
            error_log(f"E2B sandbox_upload_file error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_download_file(self, sandbox_path: str, local_path: Optional[str] = None) -> str:
        try:
            dest = local_path or Path(sandbox_path).name
            content = self.sandbox.files.read(sandbox_path)
            if isinstance(content, str):
                with open(dest, "w") as f:
                    f.write(content)
            else:
                with open(dest, "wb") as f:
                    f.write(content)
            return json.dumps({"saved_to": str(dest)})
        except Exception as e:
            error_log(f"E2B sandbox_download_file error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_list_files(self, directory_path: str = "/home/user") -> str:
        try:
            entries = self.sandbox.files.list(directory_path)
            result = []
            for entry in entries:
                result.append({
                    "name": entry.name,
                    "type": str(entry.type),
                    "path": entry.path,
                })
            return json.dumps(result)
        except Exception as e:
            error_log(f"E2B sandbox_list_files error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_write_file(self, sandbox_path: str, content: str) -> str:
        try:
            info = self.sandbox.files.write(sandbox_path, content)
            return json.dumps({"path": info.path})
        except Exception as e:
            error_log(f"E2B sandbox_write_file error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_read_file(self, sandbox_path: str) -> str:
        try:
            content = self.sandbox.files.read(sandbox_path)
            if isinstance(content, bytes):
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return json.dumps({"error": f"Binary file ({len(content)} bytes). Use download_file instead."})
            return content
        except Exception as e:
            error_log(f"E2B sandbox_read_file error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_install_packages(self, packages: List[str], language: str = "python") -> str:
        try:
            if language == "python":
                cmd = f"pip install {' '.join(packages)}"
            elif language == "javascript":
                cmd = f"npm install {' '.join(packages)}"
            else:
                return json.dumps({"error": f"Unsupported language for package install: {language}"})

            result = self.sandbox.commands.run(cmd)
            return self._format_command_result(result)
        except Exception as e:
            error_log(f"E2B install_packages error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_get_sandbox_info(self) -> str:
        try:
            info = self.sandbox.get_info()
            return json.dumps({
                "sandbox_id": info.sandbox_id,
                "template_id": info.template_id,
                "started_at": str(info.started_at),
                "end_at": str(info.end_at),
            }, default=str)
        except Exception as e:
            error_log(f"E2B get_sandbox_info error: {e}")
            return json.dumps({"error": str(e)})

    async def ae2b_shutdown_sandbox(self) -> str:
        try:
            self.sandbox.kill()
            self._sandbox = None
            return json.dumps({"status": "success", "message": "Sandbox shut down."})
        except Exception as e:
            error_log(f"E2B shutdown_sandbox error: {e}")
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Tool methods (sync, exposed to LLM)
    # ------------------------------------------------------------------

    @tool
    def e2b_run_code(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
        envs: Optional[Dict[str, str]] = None,
    ) -> str:
        """Execute code in an isolated E2B cloud sandbox.

        Supports Python, JavaScript, Java, R, and Bash. The sandbox has internet
        access and a full Linux environment. Use e2b_install_packages first if you
        need additional libraries.

        Args:
            code: The source code to execute.
            language: Programming language (python, javascript, java, r, bash). Defaults to python.
            timeout: Execution timeout in seconds. Optional.
            envs: Environment variables to set for this execution. Optional.

        Returns:
            JSON string with execution results including output, logs, errors, and generated images.
        """
        try:
            kwargs: Dict[str, Any] = {}
            if language != "python":
                kwargs["language"] = language
            if timeout is not None:
                kwargs["timeout"] = timeout
            if envs is not None:
                kwargs["envs"] = envs

            execution = self.sandbox.run_code(code, **kwargs)
            return self._format_execution(execution)
        except Exception as e:
            error_log(f"E2B run_code error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_run_command(self, command: str, timeout: Optional[int] = None) -> str:
        """Run a shell command in the E2B sandbox.

        Useful for system operations, file manipulation, package installation,
        or running compiled programs.

        Args:
            command: The shell command to execute.
            timeout: Command timeout in seconds. Optional.

        Returns:
            JSON string with stdout, stderr, and exit code.
        """
        try:
            kwargs: Dict[str, Any] = {}
            if timeout is not None:
                kwargs["timeout"] = timeout

            result = self.sandbox.commands.run(command, **kwargs)
            return self._format_command_result(result)
        except Exception as e:
            error_log(f"E2B run_command error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_upload_file(self, local_path: str, sandbox_path: Optional[str] = None) -> str:
        """Upload a local file to the E2B sandbox.

        Args:
            local_path: Path to the file on the local system.
            sandbox_path: Destination path in the sandbox. Defaults to /home/user/<filename>.

        Returns:
            JSON string with the sandbox file path.
        """
        try:
            dest = sandbox_path or f"/home/user/{Path(local_path).name}"
            with open(local_path, "rb") as f:
                info = self.sandbox.files.write(dest, f)
            return json.dumps({"path": info.path})
        except Exception as e:
            error_log(f"E2B sandbox_upload_file error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_download_file(self, sandbox_path: str, local_path: Optional[str] = None) -> str:
        """Download a file from the E2B sandbox to the local system.

        Args:
            sandbox_path: Path to the file in the sandbox.
            local_path: Destination path on the local system. Defaults to the filename.

        Returns:
            JSON string with the local file path.
        """
        try:
            dest = local_path or Path(sandbox_path).name
            content = self.sandbox.files.read(sandbox_path)
            if isinstance(content, str):
                with open(dest, "w") as f:
                    f.write(content)
            else:
                with open(dest, "wb") as f:
                    f.write(content)
            return json.dumps({"saved_to": str(dest)})
        except Exception as e:
            error_log(f"E2B sandbox_download_file error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_list_files(self, directory_path: str = "/home/user") -> str:
        """List files and directories in the E2B sandbox.

        Args:
            directory_path: Path to list. Defaults to /home/user.

        Returns:
            JSON array of file entries with name, type, and path.
        """
        try:
            entries = self.sandbox.files.list(directory_path)
            result = []
            for entry in entries:
                result.append({
                    "name": entry.name,
                    "type": str(entry.type),
                    "path": entry.path,
                })
            return json.dumps(result)
        except Exception as e:
            error_log(f"E2B sandbox_list_files error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_write_file(self, sandbox_path: str, content: str) -> str:
        """Write text content to a file in the E2B sandbox.

        Args:
            sandbox_path: Destination path in the sandbox.
            content: Text content to write.

        Returns:
            JSON string with the file path.
        """
        try:
            info = self.sandbox.files.write(sandbox_path, content)
            return json.dumps({"path": info.path})
        except Exception as e:
            error_log(f"E2B sandbox_write_file error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_read_file(self, sandbox_path: str) -> str:
        """Read the content of a text file from the E2B sandbox.

        Args:
            sandbox_path: Path to the file in the sandbox.

        Returns:
            The file content as a string, or an error message for binary files.
        """
        try:
            content = self.sandbox.files.read(sandbox_path)
            if isinstance(content, bytes):
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return json.dumps({"error": f"Binary file ({len(content)} bytes). Use download_file instead."})
            return content
        except Exception as e:
            error_log(f"E2B sandbox_read_file error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_install_packages(self, packages: List[str], language: str = "python") -> str:
        """Install packages in the E2B sandbox environment.

        Args:
            packages: List of package names to install (e.g. ["pandas", "numpy"]).
            language: Package manager to use - "python" for pip, "javascript" for npm.

        Returns:
            JSON string with installation output.
        """
        try:
            if language == "python":
                cmd = f"pip install {' '.join(packages)}"
            elif language == "javascript":
                cmd = f"npm install {' '.join(packages)}"
            else:
                return json.dumps({"error": f"Unsupported language for package install: {language}"})

            result = self.sandbox.commands.run(cmd)
            return self._format_command_result(result)
        except Exception as e:
            error_log(f"E2B install_packages error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_get_sandbox_info(self) -> str:
        """Get current E2B sandbox status and metadata.

        Returns:
            JSON string with sandbox_id, template_id, started_at, and end_at.
        """
        try:
            info = self.sandbox.get_info()
            return json.dumps({
                "sandbox_id": info.sandbox_id,
                "template_id": info.template_id,
                "started_at": str(info.started_at),
                "end_at": str(info.end_at),
            }, default=str)
        except Exception as e:
            error_log(f"E2B get_sandbox_info error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def e2b_shutdown_sandbox(self) -> str:
        """Shut down the E2B sandbox and release resources.

        Returns:
            JSON string confirming shutdown.
        """
        try:
            self.sandbox.kill()
            self._sandbox = None
            return json.dumps({"status": "success", "message": "Sandbox shut down."})
        except Exception as e:
            error_log(f"E2B shutdown_sandbox error: {e}")
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_execution(self, execution: Any) -> str:
        """Format an E2B Execution object into a JSON string."""
        output: Dict[str, Any] = {}

        if execution.error:
            output["error"] = {
                "name": execution.error.name,
                "value": execution.error.value,
                "traceback": execution.error.traceback,
            }

        if execution.logs:
            logs: Dict[str, List[str]] = {}
            if execution.logs.stdout:
                logs["stdout"] = execution.logs.stdout
            if execution.logs.stderr:
                logs["stderr"] = execution.logs.stderr
            if logs:
                output["logs"] = logs

        results = []
        for result in execution.results:
            entry: Dict[str, Any] = {}
            if hasattr(result, "text") and result.text:
                entry["text"] = result.text
            if hasattr(result, "png") and result.png:
                entry["png"] = f"base64 PNG image ({len(result.png)} chars)"
                entry["png_data"] = result.png
            if hasattr(result, "html") and result.html:
                entry["html"] = result.html
            if hasattr(result, "markdown") and result.markdown:
                entry["markdown"] = result.markdown
            if hasattr(result, "chart") and result.chart:
                entry["chart"] = {
                    "type": str(result.chart.type) if hasattr(result.chart, "type") else "unknown",
                    "title": getattr(result.chart, "title", ""),
                }
            if entry:
                results.append(entry)

        if results:
            output["results"] = results

        if not output:
            output["message"] = "Code executed successfully with no output."

        return json.dumps(output, default=str)

    def _format_command_result(self, result: Any) -> str:
        """Format an E2B CommandResult into a JSON string."""
        output: Dict[str, Any] = {}
        if hasattr(result, "stdout") and result.stdout:
            output["stdout"] = result.stdout
        if hasattr(result, "stderr") and result.stderr:
            output["stderr"] = result.stderr
        if hasattr(result, "exit_code"):
            output["exit_code"] = result.exit_code
        if not output:
            output["message"] = "Command executed successfully with no output."
        return json.dumps(output)
