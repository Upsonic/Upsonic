"""
Daytona Sandbox Toolkit for Upsonic Framework.

This module provides Daytona sandbox integration, allowing agents to:
- Execute Python, TypeScript, and JavaScript code in isolated cloud sandboxes
- Run shell commands in the sandbox
- Create, read, write, list, and delete files in the sandbox
- Search and replace content within files
- Clone Git repositories and manage branches
- Install packages dynamically
- Manage sandbox lifecycle (start, stop, delete)

Required Environment Variables:
-----------------------------
- DAYTONA_API_KEY: Daytona API key from https://app.daytona.io
- DAYTONA_API_URL: (Optional) Daytona API URL, defaults to https://app.daytona.io/api

Example Usage:
    ```python
    from upsonic import Agent, Task
    from upsonic.tools.custom_tools.daytona import DaytonaTools

    tools = DaytonaTools(api_key="your-api-key")
    agent = Agent("openai/gpt-4o", tools=[tools])
    task = Task("Write and run a Python script that calculates prime numbers")
    agent.print_do(task)
    ```
"""

import json
from os import getenv
from typing import Any, Dict, List, Optional

from upsonic.tools.base import ToolKit
from upsonic.tools.config import tool
from upsonic.utils.printing import error_log

try:
    from daytona import (
        CodeLanguage,
        CreateSandboxFromSnapshotParams,
        Daytona,
        DaytonaConfig,
        Sandbox,
    )

    _DAYTONA_AVAILABLE = True
except ImportError:
    _DAYTONA_AVAILABLE = False


class DaytonaTools(ToolKit):
    """Daytona sandbox toolkit for code execution, file operations, shell commands, and git operations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        target: Optional[str] = None,
        organization_id: Optional[str] = None,
        sandbox_id: Optional[str] = None,
        sandbox_language: Optional[str] = None,
        os_user: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        auto_stop_interval: Optional[int] = 60,
        timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        """Initialize the Daytona toolkit.

        Args:
            api_key: Daytona API key. Falls back to DAYTONA_API_KEY env var.
            api_url: Daytona API URL. Falls back to DAYTONA_API_URL env var.
            target: Daytona target region. Falls back to DAYTONA_TARGET env var.
            organization_id: Organization ID for multi-org setups.
            sandbox_id: Existing sandbox ID to connect to instead of creating new.
            sandbox_language: Default language for code execution (python, typescript, javascript).
            os_user: OS user for sandbox commands.
            env_vars: Environment variables to set in the sandbox.
            labels: Labels to attach to the sandbox.
            auto_stop_interval: Minutes of inactivity before auto-stop (default: 60). 0 to disable.
            timeout: Sandbox creation/start timeout in seconds (default: 300).
            **kwargs: ToolKit params (include_tools, exclude_tools, etc.).
        """
        super().__init__(**kwargs)

        if not _DAYTONA_AVAILABLE:
            from upsonic.utils.printing import import_error

            import_error(
                package_name="daytona",
                install_command="pip install daytona",
                feature_name="Daytona tools",
            )

        self.api_key: str = api_key or getenv("DAYTONA_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Daytona API key is required. Set DAYTONA_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.api_url: Optional[str] = api_url or getenv("DAYTONA_API_URL")
        self._target: Optional[str] = target or getenv("DAYTONA_TARGET")
        self._organization_id: Optional[str] = organization_id
        self._sandbox_id: Optional[str] = sandbox_id
        self._os_user: Optional[str] = os_user
        self._env_vars: Optional[Dict[str, str]] = env_vars
        self._labels: Dict[str, str] = labels or {}
        self._auto_stop_interval: Optional[int] = auto_stop_interval
        self._timeout: int = timeout
        self._sandbox: Optional[Any] = None
        self._daytona: Optional[Any] = None

        # Map string language to CodeLanguage enum
        lang_map = {
            "python": CodeLanguage.PYTHON if _DAYTONA_AVAILABLE else None,
            "typescript": CodeLanguage.TYPESCRIPT if _DAYTONA_AVAILABLE else None,
            "javascript": CodeLanguage.JAVASCRIPT if _DAYTONA_AVAILABLE else None,
        }
        lang_str = (sandbox_language or "python").lower()
        self._sandbox_language = lang_map.get(lang_str, lang_map.get("python"))

    @property
    def daytona(self) -> Any:
        """Lazy-create the Daytona client on first access."""
        if self._daytona is None:
            config = DaytonaConfig(
                api_key=self.api_key,
                api_url=self.api_url,
                target=self._target,
                organization_id=self._organization_id,
            )
            self._daytona = Daytona(config)
        return self._daytona

    @property
    def sandbox(self) -> Any:
        """Lazy-create or connect to a sandbox on first access."""
        if self._sandbox is None:
            if self._sandbox_id:
                self._sandbox = self.daytona.get(self._sandbox_id)
                if self._sandbox.state != "started":
                    self.daytona.start(self._sandbox, timeout=self._timeout)
            else:
                labels = self._labels.copy()
                labels.setdefault("created_by", "upsonic_daytona_toolkit")
                params = CreateSandboxFromSnapshotParams(
                    language=self._sandbox_language,
                    os_user=self._os_user,
                    env_vars=self._env_vars,
                    labels=labels,
                    auto_stop_interval=self._auto_stop_interval,
                )
                self._sandbox = self.daytona.create(params, timeout=self._timeout)
        return self._sandbox

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def adaytona_run_code(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
    ) -> str:
        try:
            return self._execute_code(code, language, timeout)
        except Exception as e:
            error_log(f"Daytona run_code error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        try:
            return self._execute_command(command, cwd, timeout)
        except Exception as e:
            error_log(f"Daytona run_command error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_create_file(self, file_path: str, content: str) -> str:
        try:
            return self._write_file(file_path, content)
        except Exception as e:
            error_log(f"Daytona create_file error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_read_file(self, file_path: str) -> str:
        try:
            return self._read_file_content(file_path)
        except Exception as e:
            error_log(f"Daytona read_file error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_list_files(self, directory: str = "/home/daytona") -> str:
        try:
            return self._list_directory(directory)
        except Exception as e:
            error_log(f"Daytona list_files error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_delete_file(self, file_path: str) -> str:
        try:
            return self._delete_path(file_path)
        except Exception as e:
            error_log(f"Daytona delete_file error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_install_packages(self, packages: List[str], language: str = "python") -> str:
        try:
            return self._install_pkgs(packages, language)
        except Exception as e:
            error_log(f"Daytona install_packages error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_search_files(self, path: str, pattern: str) -> str:
        try:
            return self._search_in_files(path, pattern)
        except Exception as e:
            error_log(f"Daytona search_files error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_git_clone(
        self,
        repo_url: str,
        path: str = "/home/daytona/project",
        branch: Optional[str] = None,
    ) -> str:
        try:
            return self._git_clone_repo(repo_url, path, branch)
        except Exception as e:
            error_log(f"Daytona git_clone error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_get_sandbox_info(self) -> str:
        try:
            return self._get_info()
        except Exception as e:
            error_log(f"Daytona get_sandbox_info error: {e}")
            return json.dumps({"error": str(e)})

    async def adaytona_shutdown_sandbox(self) -> str:
        try:
            return self._shutdown()
        except Exception as e:
            error_log(f"Daytona shutdown_sandbox error: {e}")
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Tool methods (sync, exposed to LLM)
    # ------------------------------------------------------------------

    @tool
    def daytona_run_code(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
    ) -> str:
        """Execute code in an isolated Daytona cloud sandbox.

        Supports Python, TypeScript, and JavaScript. The sandbox has internet
        access and a full Linux environment. Use daytona_install_packages first if you
        need additional libraries.

        Args:
            code: The source code to execute.
            language: Programming language (python, typescript, javascript). Defaults to python.
            timeout: Execution timeout in seconds. Optional.

        Returns:
            JSON string with execution result including output and exit code.
        """
        try:
            return self._execute_code(code, language, timeout)
        except Exception as e:
            error_log(f"Daytona run_code error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Run a shell command in the Daytona sandbox.

        Useful for system operations, file manipulation, package installation,
        or running compiled programs.

        Args:
            command: The shell command to execute.
            cwd: Working directory for the command. Defaults to sandbox home.
            timeout: Command timeout in seconds. Optional.

        Returns:
            JSON string with command output and exit code.
        """
        try:
            return self._execute_command(command, cwd, timeout)
        except Exception as e:
            error_log(f"Daytona run_command error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_create_file(self, file_path: str, content: str) -> str:
        """Create or overwrite a file in the Daytona sandbox.

        Creates parent directories automatically if they don't exist.

        Args:
            file_path: Absolute path for the file in the sandbox.
            content: Text content to write to the file.

        Returns:
            JSON string confirming file creation with path.
        """
        try:
            return self._write_file(file_path, content)
        except Exception as e:
            error_log(f"Daytona create_file error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_read_file(self, file_path: str) -> str:
        """Read the content of a file from the Daytona sandbox.

        Args:
            file_path: Absolute path to the file in the sandbox.

        Returns:
            The file content as a string, or an error message.
        """
        try:
            return self._read_file_content(file_path)
        except Exception as e:
            error_log(f"Daytona read_file error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_list_files(self, directory: str = "/home/daytona") -> str:
        """List files and directories in the Daytona sandbox.

        Args:
            directory: Directory path to list. Defaults to /home/daytona.

        Returns:
            JSON array of file entries with name, type, size, and permissions.
        """
        try:
            return self._list_directory(directory)
        except Exception as e:
            error_log(f"Daytona list_files error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_delete_file(self, file_path: str) -> str:
        """Delete a file or directory from the Daytona sandbox.

        For directories, deletion is recursive.

        Args:
            file_path: Absolute path to the file or directory to delete.

        Returns:
            JSON string confirming deletion.
        """
        try:
            return self._delete_path(file_path)
        except Exception as e:
            error_log(f"Daytona delete_file error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_install_packages(self, packages: List[str], language: str = "python") -> str:
        """Install packages in the Daytona sandbox environment.

        Args:
            packages: List of package names to install (e.g. ["pandas", "numpy"]).
            language: Package manager - "python" for pip, "javascript"/"typescript" for npm.

        Returns:
            JSON string with installation output.
        """
        try:
            return self._install_pkgs(packages, language)
        except Exception as e:
            error_log(f"Daytona install_packages error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_search_files(self, path: str, pattern: str) -> str:
        """Search for content within files in the Daytona sandbox using grep-like pattern matching.

        Args:
            path: Directory path to search in.
            pattern: Search pattern (supports regex).

        Returns:
            JSON string with matching files and line content.
        """
        try:
            return self._search_in_files(path, pattern)
        except Exception as e:
            error_log(f"Daytona search_files error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_git_clone(
        self,
        repo_url: str,
        path: str = "/home/daytona/project",
        branch: Optional[str] = None,
    ) -> str:
        """Clone a Git repository into the Daytona sandbox.

        Args:
            repo_url: URL of the Git repository to clone.
            path: Destination path in the sandbox. Defaults to /home/daytona/project.
            branch: Branch to checkout. Optional, defaults to the repo's default branch.

        Returns:
            JSON string confirming the clone with path.
        """
        try:
            return self._git_clone_repo(repo_url, path, branch)
        except Exception as e:
            error_log(f"Daytona git_clone error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_get_sandbox_info(self) -> str:
        """Get current Daytona sandbox status and metadata.

        Returns:
            JSON string with sandbox_id, state, resources, and labels.
        """
        try:
            return self._get_info()
        except Exception as e:
            error_log(f"Daytona get_sandbox_info error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def daytona_shutdown_sandbox(self) -> str:
        """Stop and delete the Daytona sandbox, releasing all resources.

        Returns:
            JSON string confirming shutdown.
        """
        try:
            return self._shutdown()
        except Exception as e:
            error_log(f"Daytona shutdown_sandbox error: {e}")
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_code(self, code: str, language: str, timeout: Optional[int]) -> str:
        """Execute code using the sandbox code runner."""
        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT,
        }
        lang_enum = lang_map.get(language.lower())
        if lang_enum is None:
            return json.dumps({"error": f"Unsupported language: {language}. Use python, typescript, or javascript."})

        kwargs: Dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout

        response = self.sandbox.process.code_run(code, **kwargs)
        return self._format_exec_response(response)

    def _execute_command(self, command: str, cwd: Optional[str], timeout: Optional[int]) -> str:
        """Execute a shell command in the sandbox."""
        kwargs: Dict[str, Any] = {}
        if cwd is not None:
            kwargs["cwd"] = cwd
        if timeout is not None:
            kwargs["timeout"] = timeout

        response = self.sandbox.process.exec(command, **kwargs)
        return self._format_exec_response(response)

    def _write_file(self, file_path: str, content: str) -> str:
        """Write content to a file, creating parent dirs as needed."""
        # Ensure parent directory exists
        parent = "/".join(file_path.rsplit("/", 1)[:-1])
        if parent:
            self.sandbox.fs.create_folder(parent, mode="755")

        self.sandbox.fs.upload_file(content.encode("utf-8"), file_path)
        return json.dumps({"status": "success", "path": file_path})

    def _read_file_content(self, file_path: str) -> str:
        """Read file content from the sandbox."""
        content = self.sandbox.fs.download_file(file_path)
        if isinstance(content, bytes):
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return json.dumps(
                    {"error": f"Binary file ({len(content)} bytes). Cannot display as text."}
                )
        return content

    def _list_directory(self, directory: str) -> str:
        """List files in a directory."""
        entries = self.sandbox.fs.list_files(directory)
        result = []
        for entry in entries:
            item: Dict[str, Any] = {"name": entry.name, "is_dir": entry.is_dir}
            if hasattr(entry, "size"):
                item["size"] = entry.size
            if hasattr(entry, "permissions"):
                item["permissions"] = entry.permissions
            result.append(item)
        return json.dumps(result)

    def _delete_path(self, file_path: str) -> str:
        """Delete a file or directory."""
        # Check if it's a directory by trying to get info
        try:
            info = self.sandbox.fs.get_file_info(file_path)
            is_dir = info.is_dir if hasattr(info, "is_dir") else False
        except Exception:
            is_dir = False

        self.sandbox.fs.delete_file(file_path, recursive=is_dir)
        return json.dumps({"status": "success", "deleted": file_path})

    def _install_pkgs(self, packages: List[str], language: str) -> str:
        """Install packages using the appropriate package manager."""
        if language == "python":
            cmd = f"pip install {' '.join(packages)}"
        elif language in ("javascript", "typescript"):
            cmd = f"npm install {' '.join(packages)}"
        else:
            return json.dumps({"error": f"Unsupported language for package install: {language}"})

        response = self.sandbox.process.exec(cmd)
        return self._format_exec_response(response)

    def _search_in_files(self, path: str, pattern: str) -> str:
        """Search for content within files."""
        matches = self.sandbox.fs.find_files(path, pattern)
        result = []
        for match in matches:
            item: Dict[str, Any] = {}
            if hasattr(match, "file"):
                item["file"] = match.file
            if hasattr(match, "line"):
                item["line"] = match.line
            if hasattr(match, "content"):
                item["content"] = match.content
            result.append(item)
        return json.dumps(result)

    def _git_clone_repo(self, repo_url: str, path: str, branch: Optional[str]) -> str:
        """Clone a git repository into the sandbox."""
        kwargs: Dict[str, Any] = {}
        if branch is not None:
            kwargs["branch"] = branch

        self.sandbox.git.clone(repo_url, path, **kwargs)
        return json.dumps({"status": "success", "path": path, "repo": repo_url})

    def _get_info(self) -> str:
        """Get sandbox information."""
        sb = self.sandbox
        info: Dict[str, Any] = {
            "sandbox_id": sb.id,
            "state": str(sb.state) if hasattr(sb, "state") else "unknown",
        }
        if hasattr(sb, "cpu"):
            info["cpu"] = sb.cpu
        if hasattr(sb, "memory"):
            info["memory"] = sb.memory
        if hasattr(sb, "disk"):
            info["disk"] = sb.disk
        if hasattr(sb, "labels"):
            info["labels"] = sb.labels
        if hasattr(sb, "created_at"):
            info["created_at"] = str(sb.created_at)
        return json.dumps(info, default=str)

    def _shutdown(self) -> str:
        """Stop and delete the sandbox."""
        if self._sandbox is not None:
            try:
                self.daytona.delete(self._sandbox, timeout=self._timeout)
            except Exception:
                # If delete fails, try stop first then delete
                try:
                    self.daytona.stop(self._sandbox, timeout=self._timeout)
                    self.daytona.delete(self._sandbox, timeout=self._timeout)
                except Exception:
                    pass
            self._sandbox = None
        return json.dumps({"status": "success", "message": "Sandbox shut down and deleted."})

    def _format_exec_response(self, response: Any) -> str:
        """Format a Daytona ExecuteResponse into a JSON string."""
        output: Dict[str, Any] = {}
        if hasattr(response, "result") and response.result:
            output["output"] = response.result
        if hasattr(response, "exit_code"):
            output["exit_code"] = response.exit_code
        if not output:
            output["message"] = "Executed successfully with no output."
        return json.dumps(output, default=str)
