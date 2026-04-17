"""
Prebuilt Autonomous Agent.

A thin subclass of :class:`AutonomousAgent` that bootstraps its
``system_prompt.md`` and ``first_message.md`` from a remote git repository
subfolder, copies user inputs into the workspace, and renders the first
message from a ``str.format`` template.

The main entry points are:

- :meth:`PrebuiltAutonomousAgent.run` — run to completion and return result.
- :meth:`PrebuiltAutonomousAgent.run_async` — async variant of ``run``.
- :meth:`PrebuiltAutonomousAgent.run_stream` — yield text chunks live.
- :meth:`PrebuiltAutonomousAgent.run_stream_async` — async variant of stream.
- :meth:`PrebuiltAutonomousAgent.run_console` — pretty terminal output with
  tool calls, results, and streamed text.
"""
from __future__ import annotations

import asyncio
import shutil
import string
import subprocess
import tempfile
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Union,
    TYPE_CHECKING,
)

from upsonic.agent.autonomous_agent.autonomous_agent import AutonomousAgent
from upsonic.agent.autonomous_agent.filesystem_toolkit import AutonomousFilesystemToolKit
from upsonic.agent.autonomous_agent.shell_toolkit import AutonomousShellToolKit

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task


class PrebuiltAutonomousAgent(AutonomousAgent):
    """
    Autonomous agent whose system prompt and first message come from a git repo.

    Constructor stores the repo coordinates; every call to :meth:`run`,
    :meth:`run_stream`, or :meth:`run_console` performs a fresh shallow
    ``git sparse-checkout`` of ``agent_folder`` into ``workspace``, so the
    workspace always reflects the current state of the remote template.

    ``workspace`` can be set either at construction time or per-run. If omitted
    at both, a :class:`ValueError` is raised when the agent is invoked.

    Example:
        ```python
        agent = PrebuiltAutonomousAgent(
            model="anthropic/claude-sonnet-4-5",
            agent_repo="https://github.com/Upsonic/AutonomousAgents",
            agent_folder="applied_scientist",
        )

        agent.run_console(
            workspace="./ws",
            inputs=["example_1/"],
            research_paper="example_1/paper.pdf",
            current_notebook="example_1/baseline.ipynb",
        )
        ```
    """

    def __init__(
        self,
        *args: Any,
        agent_repo: Optional[str] = None,
        agent_folder: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent_repo: Optional[str] = agent_repo
        self.agent_folder: Optional[str] = agent_folder
        self._first_message_template: Optional[str] = None
        self._repo_system_prompt: Optional[str] = None

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _log(self, verbose: bool, message: str) -> None:
        """Emit a prefixed progress line when ``verbose`` is True."""
        if verbose:
            print(f"[PrebuiltAutonomousAgent] {message}")

    def _apply_workspace(self, workspace: Union[str, Path]) -> Path:
        """
        Point the agent at a new workspace, recreating its filesystem/shell
        toolkits so they sandbox to the new path and re-registering them with
        the underlying :class:`~upsonic.agent.agent.Agent` tool manager.
        """
        new_ws = Path(workspace).resolve()
        new_ws.mkdir(parents=True, exist_ok=True)

        to_remove: List[Any] = []
        if self.filesystem_toolkit is not None:
            to_remove.append(self.filesystem_toolkit)
            self.filesystem_toolkit = None
        if self.shell_toolkit is not None:
            to_remove.append(self.shell_toolkit)
            self.shell_toolkit = None
        if to_remove:
            self.remove_tools(to_remove)

        to_add: List[Any] = []
        fs = AutonomousFilesystemToolKit(workspace=new_ws)
        self.filesystem_toolkit = fs
        to_add.append(fs)

        sh = AutonomousShellToolKit(workspace=new_ws)
        self.shell_toolkit = sh
        to_add.append(sh)
        self.add_tools(to_add)

        self.autonomous_workspace = new_ws
        self.workspace = str(new_ws)
        self._workspace_greeting_executed = False
        self._workspace_agents_md_content = self._read_workspace_agents_md()
        return new_ws

    def _clone_repo_folder(
        self,
        repo_url: str,
        folder: str,
        destination: Path,
        verbose: bool,
    ) -> None:
        """
        Shallow sparse-clone ``repo_url`` and copy the contents of ``folder``
        into ``destination``. Falls back to a full shallow clone if sparse
        checkout fails.
        """
        if shutil.which("git") is None:
            raise RuntimeError(
                "git is required to fetch agent_repo but was not found on PATH."
            )

        folder = folder.strip().strip("/\\")
        if not folder:
            raise ValueError("agent_folder must be a non-empty path inside the repo.")

        self._log(verbose, f"Cloning {repo_url} (folder='{folder}') → '{destination}' ...")

        with tempfile.TemporaryDirectory(prefix="upsonic_prebuilt_repo_") as tmp_dir:
            clone_dir = Path(tmp_dir) / "repo"
            try:
                subprocess.run(
                    [
                        "git", "clone",
                        "--depth", "1",
                        "--filter=blob:none",
                        "--sparse",
                        repo_url,
                        str(clone_dir),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "sparse-checkout", "set", folder],
                    cwd=str(clone_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                if clone_dir.exists():
                    shutil.rmtree(clone_dir, ignore_errors=True)
                self._log(verbose, "Sparse-checkout failed; falling back to full shallow clone.")
                try:
                    subprocess.run(
                        ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as e:
                    stderr = e.stderr or ""
                    raise RuntimeError(
                        f"Failed to clone repo '{repo_url}': {stderr.strip() or e}"
                    ) from e

            source = clone_dir / folder
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Folder '{folder}' not found inside repo '{repo_url}'."
                )

            copied = 0
            for item in source.iterdir():
                target = destination / item.name
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                else:
                    if target.exists():
                        target.unlink()
                    shutil.copy2(item, target)
                copied += 1

            self._log(verbose, f"Copied {copied} item(s) from '{folder}' into workspace.")

    def _copy_inputs(
        self,
        inputs: List[str],
        destination: Path,
        verbose: bool,
    ) -> None:
        """
        Copy user-supplied files/directories into the workspace so the
        sandboxed agent can read them. Relative paths preserve their layout;
        absolute paths land at the workspace root by basename.
        """
        dest_root = destination.resolve()
        for raw in inputs:
            if not raw:
                continue
            src_arg = Path(raw)
            if src_arg.is_absolute():
                src = src_arg.resolve()
                rel_dest = Path(src.name)
            else:
                src = (Path.cwd() / src_arg).resolve()
                rel_dest = Path(*src_arg.parts) if src_arg.parts else Path(src.name)

            if not src.exists():
                raise FileNotFoundError(
                    f"input path does not exist: {raw!r} (resolved to {src})"
                )

            dest = (dest_root / rel_dest).resolve()
            try:
                dest.relative_to(dest_root)
            except ValueError as e:
                raise ValueError(
                    f"Input path {raw!r} would escape the workspace sandbox."
                ) from e

            if dest.exists():
                if dest.is_dir() and not dest.is_symlink():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            dest.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                shutil.copytree(src, dest)
                kind = "directory"
            else:
                shutil.copy2(src, dest)
                kind = "file"

            self._log(verbose, f"Copied input {kind} '{raw}' → workspace '{rel_dest}'.")

    def _load_repo_files(self, workspace: Path, verbose: bool) -> None:
        """Load ``system_prompt.md`` and ``first_message.md`` from the workspace."""
        sp = workspace / "system_prompt.md"
        fm = workspace / "first_message.md"

        if sp.exists():
            content = sp.read_text(encoding="utf-8")
            self._repo_system_prompt = content
            self.system_prompt = self._build_autonomous_system_prompt(
                user_system_prompt=content,
                enable_filesystem=self.filesystem_toolkit is not None,
                enable_shell=self.shell_toolkit is not None,
            )
            self._log(verbose, f"Loaded system_prompt.md ({len(content)} chars).")
        else:
            self._repo_system_prompt = None
            self._log(verbose, "No system_prompt.md found in repo folder.")

        if fm.exists():
            self._first_message_template = fm.read_text(encoding="utf-8")
            self._log(
                verbose,
                f"Loaded first_message.md ({len(self._first_message_template)} chars).",
            )
        else:
            self._first_message_template = None
            self._log(verbose, "No first_message.md found in repo folder.")

    @staticmethod
    def _extract_template_fields(template: str) -> Set[str]:
        """Return the set of named placeholder fields in a ``str.format`` template."""
        fields: Set[str] = set()
        for _, field_name, _, _ in string.Formatter().parse(template):
            if field_name:
                root = field_name.split(".", 1)[0].split("[", 1)[0]
                if root:
                    fields.add(root)
        return fields

    def _render_first_message(self, template: str, params: Dict[str, Any]) -> str:
        """
        Render ``template`` with ``params``. Raises :class:`ValueError` listing
        every missing placeholder so callers know which kwargs to pass.
        """
        required = self._extract_template_fields(template)
        missing = sorted(f for f in required if f not in params)
        if missing:
            raise ValueError(
                "run() is missing required template parameter(s) for "
                "first_message.md: "
                + ", ".join(missing)
                + ". Pass them as keyword arguments."
            )
        try:
            return template.format(**params)
        except KeyError as e:
            raise ValueError(
                f"run() missing template parameter {e} for first_message.md."
            ) from e

    def _bootstrap(
        self,
        workspace: Optional[str],
        inputs: Optional[List[str]],
        verbose: bool,
        template_params: Dict[str, Any],
    ) -> "Task":
        """
        Shared prep for every entry point: apply workspace, clone the repo
        subtree, copy user inputs, load prompts, render the first message,
        and return it as a :class:`Task`.
        """
        from upsonic.tasks.tasks import Task

        if workspace is not None:
            self._apply_workspace(workspace)
        elif self.autonomous_workspace is None:
            raise ValueError(
                "PrebuiltAutonomousAgent requires a workspace. Pass workspace=... "
                "or set it at construction."
            )

        assert self.autonomous_workspace is not None

        if self.agent_repo:
            if not self.agent_folder:
                raise ValueError(
                    "agent_repo was provided without agent_folder."
                )
            self._clone_repo_folder(
                repo_url=self.agent_repo,
                folder=self.agent_folder,
                destination=self.autonomous_workspace,
                verbose=verbose,
            )
            self._load_repo_files(self.autonomous_workspace, verbose=verbose)

        if inputs:
            self._copy_inputs(inputs, self.autonomous_workspace, verbose=verbose)

        if self._first_message_template is None:
            raise ValueError(
                "run() could not find first_message.md in the repo folder "
                f"'{self.agent_folder}'. It is required to start a run."
            )

        rendered = self._render_first_message(
            self._first_message_template, template_params
        )
        self._log(verbose, f"Rendered first message ({len(rendered)} chars).")
        return Task(description=rendered)

    # --------------------------------------------------------------------- #
    # Public API — run
    # --------------------------------------------------------------------- #

    async def run_async(
        self,
        *,
        workspace: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        verbose: bool = False,
        return_output: bool = False,
        timeout: Optional[float] = None,
        partial_on_timeout: bool = False,
        **template_params: Any,
    ) -> Any:
        """
        Async: bootstrap the agent from the repo and execute the rendered first
        message against :meth:`Agent.do_async`.
        """
        task = self._bootstrap(workspace, inputs, verbose, template_params)

        saved_print_param: Optional[bool] = self._print_param
        saved_print_attr: bool = self.print
        saved_show_tool_calls: bool = self.show_tool_calls
        saved_debug: bool = self.debug
        saved_debug_level: int = self.debug_level
        try:
            if verbose:
                self._print_param = True
                self.print = True
                self.show_tool_calls = True
                self.debug = True
                self.debug_level = max(self.debug_level, 2)
            result = await self.do_async(
                task,
                return_output=return_output,
                timeout=timeout,
                partial_on_timeout=partial_on_timeout,
            )
        finally:
            self._print_param = saved_print_param
            self.print = saved_print_attr
            self.show_tool_calls = saved_show_tool_calls
            self.debug = saved_debug
            self.debug_level = saved_debug_level

        self._log(verbose, "Run finished.")
        return result

    def run(
        self,
        *,
        workspace: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        verbose: bool = False,
        return_output: bool = False,
        timeout: Optional[float] = None,
        partial_on_timeout: bool = False,
        **template_params: Any,
    ) -> Any:
        """
        Sync wrapper around :meth:`run_async`.

        Clones the repo subtree fresh on every call, copies ``inputs`` into the
        workspace, applies the repo's ``system_prompt.md``, renders
        ``first_message.md`` with ``template_params``, and executes the run.

        Raises:
            ValueError: Missing template params or workspace.
            FileNotFoundError: Missing repo folder or input path.
            RuntimeError: Git is missing or clone failed.
        """
        from upsonic.agent.agent import _run_in_bg_loop
        return _run_in_bg_loop(
            self.run_async(
                workspace=workspace,
                inputs=inputs,
                verbose=verbose,
                return_output=return_output,
                timeout=timeout,
                partial_on_timeout=partial_on_timeout,
                **template_params,
            )
        )

    # --------------------------------------------------------------------- #
    # Public API — stream
    # --------------------------------------------------------------------- #

    async def run_stream_async(
        self,
        *,
        workspace: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        verbose: bool = False,
        events: bool = False,
        **template_params: Any,
    ) -> AsyncIterator[Any]:
        """
        Async streaming variant. Yields text chunks by default; set
        ``events=True`` to yield :class:`AgentStreamEvent` objects instead.
        """
        task = self._bootstrap(workspace, inputs, verbose, template_params)
        async for chunk in self.astream(task, events=events):
            yield chunk
        self._log(verbose, "Stream finished.")

    def run_stream(
        self,
        *,
        workspace: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        verbose: bool = False,
        events: bool = False,
        **template_params: Any,
    ) -> Iterator[Any]:
        """
        Sync streaming wrapper: yields text chunks (or events with
        ``events=True``) as the agent produces them.

        Example:
            ```python
            for chunk in agent.run_stream(workspace="./ws", **params):
                print(chunk, end="", flush=True)
            ```
        """
        import queue
        import threading
        from upsonic.agent.agent import _get_bg_loop

        task = self._bootstrap(workspace, inputs, verbose, template_params)

        result_queue: "queue.Queue[Any]" = queue.Queue()
        error_holder: List[BaseException] = []

        async def stream_to_queue() -> None:
            try:
                async for item in self.astream(task, events=events):
                    result_queue.put(item)
            except BaseException as exc:
                error_holder.append(exc)
            finally:
                result_queue.put(None)

        def run_async_stream() -> None:
            loop = _get_bg_loop()
            asyncio.run_coroutine_threadsafe(stream_to_queue(), loop).result()

        thread = threading.Thread(target=run_async_stream, daemon=True)
        thread.start()

        while True:
            item = result_queue.get()
            if item is None:
                if error_holder:
                    raise error_holder[0]
                break
            yield item
        self._log(verbose, "Stream finished.")

    # --------------------------------------------------------------------- #
    # Public API — run_console (pretty TTY)
    # --------------------------------------------------------------------- #

    def run_console(
        self,
        *,
        workspace: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        verbose: bool = False,
        preview_chars: int = 400,
        **template_params: Any,
    ) -> str:
        """
        Run the agent and render the model's output, tool calls, and tool
        results to the terminal live with color and formatting.

        Uses :mod:`rich` for colored panels and inline text so the assistant's
        response, each tool invocation, and each tool result are visually
        distinct as they arrive. Returns the concatenated assistant text once
        the run completes.
        """
        from rich.console import Console
        from rich.panel import Panel
        from upsonic.run.events.events import (
            TextDeltaEvent,
            ToolCallEvent,
            ToolResultEvent,
            RunCompletedEvent,
            RunCancelledEvent,
        )

        console = Console()
        task = self._bootstrap(workspace, inputs, verbose, template_params)

        console.print(
            Panel.fit(
                f"[bold]PrebuiltAutonomousAgent starting[/bold]\n"
                f"workspace: [cyan]{self.autonomous_workspace}[/cyan]\n"
                f"model: [cyan]{self.model_name}[/cyan]",
                border_style="blue",
            )
        )

        accumulated: List[str] = []
        in_text_block: bool = False

        def close_text_block() -> None:
            nonlocal in_text_block
            if in_text_block:
                console.print()
                in_text_block = False

        for event in self.stream(task, events=True):
            if isinstance(event, TextDeltaEvent):
                if not in_text_block:
                    console.print("[bold cyan]▸ Assistant[/bold cyan] ", end="")
                    in_text_block = True
                console.print(event.content, end="", highlight=False, markup=False)
                accumulated.append(event.content)

            elif isinstance(event, ToolCallEvent):
                close_text_block()
                args_preview = str(event.tool_args) if event.tool_args else ""
                if len(args_preview) > preview_chars:
                    args_preview = args_preview[:preview_chars] + "…"
                console.print(
                    f"[bold yellow]🔧 {event.tool_name}[/bold yellow] "
                    f"[dim]{args_preview}[/dim]"
                )

            elif isinstance(event, ToolResultEvent):
                close_text_block()
                preview = event.result_preview
                if preview is None:
                    preview = str(event.result) if event.result is not None else ""
                if len(preview) > preview_chars:
                    preview = preview[:preview_chars] + "…"
                if event.is_error:
                    console.print(
                        f"[bold red]✗ {event.tool_name}[/bold red] "
                        f"[red]{event.error_message or preview}[/red]"
                    )
                else:
                    time_str = (
                        f" [dim]({event.execution_time:.2f}s)[/dim]"
                        if event.execution_time is not None
                        else ""
                    )
                    console.print(
                        f"[bold green]✓ {event.tool_name}[/bold green]{time_str}"
                    )
                    if preview:
                        console.print(f"  [dim]{preview}[/dim]")

            elif isinstance(event, RunCancelledEvent):
                close_text_block()
                console.print("[bold red]⚠ Run cancelled[/bold red]")

            elif isinstance(event, RunCompletedEvent):
                close_text_block()

        close_text_block()
        console.print(
            Panel.fit(
                "[bold green]Run completed[/bold green]",
                border_style="green",
            )
        )
        return "".join(accumulated)

    def __repr__(self) -> str:
        ws = str(self.autonomous_workspace) if self.autonomous_workspace else None
        return (
            f"PrebuiltAutonomousAgent("
            f"model={self.model_name!r}, "
            f"workspace={ws!r}, "
            f"agent_repo={self.agent_repo!r}, "
            f"agent_folder={self.agent_folder!r}"
            f")"
        )
