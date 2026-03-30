"""
Smoke tests for DaytonaTools with Agent and AutonomousAgent.

Requires a valid DAYTONA_API_KEY environment variable.
Automatically skipped when the key is not set.

Coverage:
- DaytonaTools instantiation and lazy sandbox creation
- Direct method calls: daytona_run_code, daytona_run_command, daytona_create_file, daytona_read_file, daytona_list_files
- Agent + DaytonaTools: tool_calls on Task and AgentRunOutput tracking
- AutonomousAgent + DaytonaTools: local filesystem + remote sandbox execution
- Task.tool_calls populated with tool_name, params, tool_result
- AgentRunOutput.tools populated with ToolExecution entries
- AgentRunOutput.tool_call_count incremented

Run:
    uv run pytest tests/smoke_tests/tools/test_daytona_tool.py -v --tb=short -s
"""

import json
import os
from io import StringIO
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import pytest

DAYTONA_API_KEY: Optional[str] = os.getenv("DAYTONA_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not DAYTONA_API_KEY,
        reason="DAYTONA_API_KEY not set; skipping Daytona smoke tests",
    ),
    pytest.mark.timeout(180),
]

MODEL = "openai/gpt-4o-mini"


def _get_tool_call_names(tool_calls: List[Dict[str, Any]]) -> List[str]:
    """Extract tool_name values from task.tool_calls list."""
    return [tc.get("tool_name", "") for tc in tool_calls]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def daytona_tools():
    from upsonic.tools.custom_tools.daytona import DaytonaTools
    tools = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=300)
    yield tools
    # Cleanup: shutdown sandbox if it was created
    if tools._sandbox is not None:
        try:
            tools.daytona.delete(tools._sandbox, timeout=60)
        except Exception:
            pass


# ------------------------------------------------------------------
# Unit: DaytonaTools construction and lazy init
# ------------------------------------------------------------------

class TestDaytonaToolsConstruction:
    """Test DaytonaTools construction without creating a sandbox."""

    def test_init_with_api_key(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        tools = DaytonaTools(api_key=DAYTONA_API_KEY)
        assert tools.api_key == DAYTONA_API_KEY
        assert tools._sandbox is None, "Sandbox should not be created at init (lazy)"
        assert tools._daytona is None, "Daytona client should not be created at init (lazy)"

    def test_init_missing_api_key_raises(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        original = os.environ.get("DAYTONA_API_KEY")
        try:
            os.environ.pop("DAYTONA_API_KEY", None)
            with pytest.raises(ValueError, match="Daytona API key is required"):
                DaytonaTools(api_key=None)
        finally:
            if original:
                os.environ["DAYTONA_API_KEY"] = original

    def test_lazy_sandbox_creation(self, daytona_tools):
        """Accessing .sandbox should trigger creation."""
        sandbox = daytona_tools.sandbox
        assert sandbox is not None
        assert daytona_tools._sandbox is not None
        assert daytona_tools._daytona is not None

    def test_sandbox_language_default(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        from daytona import CodeLanguage
        tools = DaytonaTools(api_key=DAYTONA_API_KEY)
        assert tools._sandbox_language == CodeLanguage.PYTHON

    def test_sandbox_language_typescript(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        from daytona import CodeLanguage
        tools = DaytonaTools(api_key=DAYTONA_API_KEY, sandbox_language="typescript")
        assert tools._sandbox_language == CodeLanguage.TYPESCRIPT

    def test_init_custom_timeout(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        tools = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=600)
        assert tools._timeout == 600

    def test_init_env_vars(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        env = {"FOO": "bar", "BAZ": "qux"}
        tools = DaytonaTools(api_key=DAYTONA_API_KEY, env_vars=env)
        assert tools._env_vars == env

    def test_init_labels(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        labels = {"project": "test", "team": "core"}
        tools = DaytonaTools(api_key=DAYTONA_API_KEY, labels=labels)
        assert tools._labels == labels

    def test_init_auto_stop_interval(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        tools = DaytonaTools(api_key=DAYTONA_API_KEY, auto_stop_interval=120)
        assert tools._auto_stop_interval == 120

    def test_init_sandbox_id(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        tools = DaytonaTools(api_key=DAYTONA_API_KEY, sandbox_id="test-id-123")
        assert tools._sandbox_id == "test-id-123"


# ------------------------------------------------------------------
# Unit: Direct method calls on DaytonaTools
# ------------------------------------------------------------------

class TestDaytonaToolsDirect:
    """Test DaytonaTools methods directly (not via agent)."""

    def test_daytona_run_code_python(self, daytona_tools):
        result = daytona_tools.daytona_run_code("print(2 + 3)")
        print(f"daytona_run_code result: {result}")
        parsed = json.loads(result)
        assert "error" not in parsed, f"Unexpected error: {parsed}"
        assert "5" in parsed.get("output", ""), f"Expected '5' in output, got: {parsed}"

    def test_daytona_run_code_error(self, daytona_tools):
        result = daytona_tools.daytona_run_code("raise ValueError('test error')")
        print(f"daytona_run_code error result: {result}")
        parsed = json.loads(result)
        has_error = (
            parsed.get("exit_code", 0) != 0
            or "error" in parsed.get("output", "").lower()
            or "error" in parsed
        )
        assert has_error, f"Expected error indicator, got: {parsed}"

    def test_daytona_run_command(self, daytona_tools):
        result = daytona_tools.daytona_run_command("echo hello_daytona")
        print(f"daytona_run_command result: {result}")
        parsed = json.loads(result)
        assert "hello_daytona" in parsed.get("output", ""), f"Expected 'hello_daytona' in output, got: {parsed}"

    def test_daytona_run_command_exit_code(self, daytona_tools):
        result = daytona_tools.daytona_run_command("true")
        print(f"daytona_run_command exit_code result: {result}")
        parsed = json.loads(result)
        assert parsed.get("exit_code") == 0, f"Expected exit_code 0, got: {parsed}"

    def test_daytona_create_and_read_file(self, daytona_tools):
        write_result = daytona_tools.daytona_create_file("/home/daytona/test_daytona.txt", "hello from daytona")
        print(f"daytona_create_file result: {write_result}")
        parsed_write = json.loads(write_result)
        assert parsed_write.get("status") == "success", f"Write failed: {parsed_write}"

        read_result = daytona_tools.daytona_read_file("/home/daytona/test_daytona.txt")
        print(f"daytona_read_file result: {read_result}")
        assert "hello from daytona" in read_result, f"Expected content in read, got: {read_result}"

    def test_daytona_list_files(self, daytona_tools):
        result = daytona_tools.daytona_list_files("/home/daytona")
        print(f"daytona_list_files result: {result}")
        parsed = json.loads(result)
        assert isinstance(parsed, list), f"Expected list, got: {type(parsed)}"

    def test_daytona_delete_file(self, daytona_tools):
        daytona_tools.daytona_create_file("/home/daytona/to_delete.txt", "delete me")
        result = daytona_tools.daytona_delete_file("/home/daytona/to_delete.txt")
        print(f"daytona_delete_file result: {result}")
        parsed = json.loads(result)
        assert parsed.get("status") == "success", f"Delete failed: {parsed}"

    def test_daytona_install_packages(self, daytona_tools):
        result = daytona_tools.daytona_install_packages(["cowsay"])
        print(f"daytona_install_packages result: {result}")
        parsed = json.loads(result)
        assert "error" not in parsed, f"Install failed: {parsed}"

    def test_daytona_get_sandbox_info(self, daytona_tools):
        result = daytona_tools.daytona_get_sandbox_info()
        print(f"daytona_get_sandbox_info result: {result}")
        parsed = json.loads(result)
        assert "sandbox_id" in parsed, f"Expected sandbox_id, got: {parsed}"
        assert isinstance(parsed["sandbox_id"], str)
        assert len(parsed["sandbox_id"]) > 0

    def test_daytona_search_files(self, daytona_tools):
        """Create a file then search for content in it."""
        daytona_tools.daytona_create_file("/home/daytona/searchable.txt", "unique_search_token_xyz")
        result = daytona_tools.daytona_search_files("/home/daytona", "unique_search_token_xyz")
        print(f"daytona_search_files result: {result}")
        parsed = json.loads(result)
        assert isinstance(parsed, list), f"Expected list, got: {type(parsed)}"

    def test_daytona_run_code_unsupported_language(self, daytona_tools):
        """Unsupported language should return error."""
        result = daytona_tools.daytona_run_code("print(1)", language="ruby")
        print(f"daytona_run_code unsupported lang result: {result}")
        parsed = json.loads(result)
        assert "error" in parsed, f"Expected error for unsupported language, got: {parsed}"

    def test_daytona_install_packages_unsupported_language(self, daytona_tools):
        """Unsupported language for install should return error."""
        result = daytona_tools.daytona_install_packages(["pkg"], language="ruby")
        print(f"daytona_install_packages unsupported lang result: {result}")
        parsed = json.loads(result)
        assert "error" in parsed, f"Expected error for unsupported language, got: {parsed}"

    def test_daytona_run_command_with_cwd(self, daytona_tools):
        """Run command with explicit cwd parameter."""
        # Create a directory and a file in it
        daytona_tools.daytona_create_file("/home/daytona/testdir/hello.txt", "cwd test")
        result = daytona_tools.daytona_run_command("ls", cwd="/home/daytona/testdir")
        print(f"daytona_run_command with cwd result: {result}")
        parsed = json.loads(result)
        assert "hello.txt" in parsed.get("output", ""), f"Expected 'hello.txt' in output, got: {parsed}"

    def test_daytona_run_command_with_timeout(self, daytona_tools):
        """Run command with explicit timeout parameter."""
        result = daytona_tools.daytona_run_command("echo timeout_test", timeout=30)
        print(f"daytona_run_command with timeout result: {result}")
        parsed = json.loads(result)
        assert "timeout_test" in parsed.get("output", ""), f"Expected 'timeout_test' in output, got: {parsed}"

    def test_daytona_git_clone(self, daytona_tools):
        """Clone a public Git repository into the sandbox."""
        result = daytona_tools.daytona_git_clone(
            "https://github.com/octocat/Hello-World.git",
            path="/home/daytona/hello-world",
        )
        print(f"daytona_git_clone result: {result}")
        parsed = json.loads(result)
        assert parsed.get("status") == "success", f"Clone failed: {parsed}"
        assert parsed.get("path") == "/home/daytona/hello-world"
        assert "Hello-World" in parsed.get("repo", "")

        # Verify cloned content exists
        list_result = daytona_tools.daytona_list_files("/home/daytona/hello-world")
        print(f"daytona_list_files after clone: {list_result}")
        parsed_list = json.loads(list_result)
        assert isinstance(parsed_list, list)
        assert len(parsed_list) > 0, "Cloned repo should have files"

    def test_daytona_create_file_nested_dirs(self, daytona_tools):
        """Create file in nested dirs should auto-create parents."""
        result = daytona_tools.daytona_create_file(
            "/home/daytona/deep/nested/dir/file.txt", "nested content"
        )
        print(f"daytona_create_file nested result: {result}")
        parsed = json.loads(result)
        assert parsed.get("status") == "success", f"Create failed: {parsed}"

        read_result = daytona_tools.daytona_read_file("/home/daytona/deep/nested/dir/file.txt")
        print(f"daytona_read_file nested result: {read_result}")
        assert "nested content" in read_result

    def test_daytona_install_packages_javascript(self, daytona_tools):
        """Install npm packages with language='javascript'."""
        result = daytona_tools.daytona_install_packages(["cowsay"], language="javascript")
        print(f"daytona_install_packages javascript result: {result}")
        parsed = json.loads(result)
        # npm install may produce output or succeed
        assert "error" not in parsed, f"Unexpected error: {parsed}"


# ------------------------------------------------------------------
# Shutdown test (must run last, creates its own sandbox)
# ------------------------------------------------------------------

class TestDaytonaShutdown:
    """Test daytona_shutdown_sandbox - uses its own sandbox since it deletes it."""

    def test_daytona_shutdown_sandbox(self):
        from upsonic.tools.custom_tools.daytona import DaytonaTools
        tools = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=300)
        # Force sandbox creation
        _ = tools.sandbox
        assert tools._sandbox is not None, "Sandbox should be created"

        result = tools.daytona_shutdown_sandbox()
        print(f"daytona_shutdown_sandbox result: {result}")
        parsed = json.loads(result)
        assert parsed.get("status") == "success", f"Expected success, got: {parsed}"
        assert "shut down" in parsed.get("message", "").lower(), f"Expected shutdown message, got: {parsed}"
        assert tools._sandbox is None, "Sandbox should be None after shutdown"


# ------------------------------------------------------------------
# E2E: Agent + DaytonaTools
# ------------------------------------------------------------------

class TestAgentWithDaytona:
    """Test Agent using DaytonaTools -- verifies tool_calls on Task and AgentRunOutput."""

    @pytest.mark.asyncio
    async def test_agent_run_code_via_task(self):
        """Agent should call daytona_run_code and record it in task.tool_calls."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.daytona import DaytonaTools

        daytona = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=300)

        agent = Agent(model=MODEL, tools=[daytona])
        task = Task(
            description=(
                "Use the daytona_run_code tool to execute this Python code: print(7 * 8). "
                "Return the result number."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- Agent + Daytona terminal output ---\n{terminal_output}\n---")

        # Task.tool_calls should be populated
        assert len(task.tool_calls) > 0, f"task.tool_calls should not be empty, got: {task.tool_calls}"
        called_names = _get_tool_call_names(task.tool_calls)
        assert "daytona_run_code" in called_names, f"daytona_run_code should be in tool_calls, got: {called_names}"

        # Verify tool_call structure
        run_code_call = next(tc for tc in task.tool_calls if tc.get("tool_name") == "daytona_run_code")
        assert "params" in run_code_call, "Tool call should have 'params'"
        assert "tool_result" in run_code_call, "Tool call should have 'tool_result'"

        # AgentRunOutput should track tools
        run_output = agent.get_run_output()
        assert run_output is not None, "get_run_output() should not be None"
        assert run_output.tool_call_count > 0, f"tool_call_count should be > 0, got: {run_output.tool_call_count}"

        # Result should contain 56
        response_text = str(task.response) if task.response else terminal_output
        assert "56" in response_text, f"Response should contain '56', got: {response_text}"

        # Cleanup
        if daytona._sandbox is not None:
            try:
                daytona.daytona.delete(daytona._sandbox, timeout=60)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_agent_run_command_via_task(self):
        """Agent should call daytona_run_command and record it in task.tool_calls."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.daytona import DaytonaTools

        daytona = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=300)

        agent = Agent(model=MODEL, tools=[daytona])
        task = Task(
            description=(
                "Use the daytona_run_command tool to execute: echo UPSONIC_DAYTONA_TEST. "
                "Return the output."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- Agent + Daytona run_command output ---\n{terminal_output}\n---")

        assert len(task.tool_calls) > 0, "task.tool_calls should not be empty"
        called_names = _get_tool_call_names(task.tool_calls)
        assert "daytona_run_command" in called_names, f"daytona_run_command should be in tool_calls, got: {called_names}"

        run_output = agent.get_run_output()
        assert run_output is not None
        assert run_output.tool_call_count > 0

        if daytona._sandbox is not None:
            try:
                daytona.daytona.delete(daytona._sandbox, timeout=60)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_agent_install_and_run(self):
        """Agent should install a package then run code using it."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.daytona import DaytonaTools

        daytona = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=300)

        agent = Agent(model=MODEL, tools=[daytona])
        task = Task(
            description=(
                "First use daytona_install_packages to install the 'cowsay' package (language python), "
                "then use daytona_run_code to execute: import cowsay; print(cowsay.cow('Daytona works!')). "
                "Return the output."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- Agent install+run output ---\n{terminal_output}\n---")

        called_names = _get_tool_call_names(task.tool_calls)
        assert "daytona_install_packages" in called_names, f"daytona_install_packages should be in tool_calls, got: {called_names}"
        assert "daytona_run_code" in called_names, f"daytona_run_code should be in tool_calls, got: {called_names}"

        run_output = agent.get_run_output()
        assert run_output is not None
        assert run_output.tool_call_count >= 2, f"Expected >= 2 tool calls, got: {run_output.tool_call_count}"

        if daytona._sandbox is not None:
            try:
                daytona.daytona.delete(daytona._sandbox, timeout=60)
            except Exception:
                pass


# ------------------------------------------------------------------
# E2E: AutonomousAgent + DaytonaTools (no local shell)
# ------------------------------------------------------------------

class TestAutonomousAgentWithDaytona:
    """Test AutonomousAgent with local filesystem + remote Daytona execution."""

    @pytest.mark.asyncio
    async def test_autonomous_agent_daytona_code_execution(self, tmp_path):
        """AutonomousAgent should use Daytona for code execution, not local shell."""
        from upsonic import AutonomousAgent, Task
        from upsonic.tools.custom_tools.daytona import DaytonaTools

        daytona = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=300)

        agent = AutonomousAgent(
            model=MODEL,
            workspace=str(tmp_path),
            enable_filesystem=True,
            enable_shell=False,  # No local shell
            tools=[daytona],
        )

        task = Task(
            description=(
                "Use the daytona_run_code tool to execute this Python code: "
                "print(sum(range(1, 11))). Return only the result number."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- AutonomousAgent + Daytona output ---\n{terminal_output}\n---")

        # Should have used daytona_run_code from Daytona, not local shell
        assert len(task.tool_calls) > 0, "task.tool_calls should not be empty"
        called_names = _get_tool_call_names(task.tool_calls)
        assert "daytona_run_code" in called_names, f"daytona_run_code should be in tool_calls, got: {called_names}"

        # Local shell tools should NOT be called
        assert "run_python" not in called_names, "Local run_python should not be called"

        # Verify AgentRunOutput
        run_output = agent.get_run_output()
        assert run_output is not None
        assert run_output.tool_call_count > 0

        # Result should contain 55
        response_text = str(task.response) if task.response else terminal_output
        assert "55" in response_text, f"Response should contain '55', got: {response_text}"

        if daytona._sandbox is not None:
            try:
                daytona.daytona.delete(daytona._sandbox, timeout=60)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_autonomous_agent_filesystem_and_daytona(self, tmp_path):
        """AutonomousAgent should have both local filesystem and remote Daytona tools."""
        from upsonic import AutonomousAgent, Task
        from upsonic.tools.custom_tools.daytona import DaytonaTools

        daytona = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=300)

        agent = AutonomousAgent(
            model=MODEL,
            workspace=str(tmp_path),
            enable_filesystem=True,
            enable_shell=False,
            tools=[daytona],
        )

        task = Task(
            description=(
                "Do these steps:\n"
                "1. Use daytona_run_code to execute: print(3 ** 4)\n"
                "2. Use write_file (the local filesystem tool) to create a file "
                "called 'result.txt' in the workspace with the content '81'\n"
                "Return 'done' when finished."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- AutonomousAgent filesystem+Daytona output ---\n{terminal_output}\n---")

        called_names = _get_tool_call_names(task.tool_calls)
        assert "daytona_run_code" in called_names, f"daytona_run_code (Daytona) should be in tool_calls, got: {called_names}"
        assert "write_file" in called_names, f"write_file (local) should be in tool_calls, got: {called_names}"

        # Verify the file was written locally
        result_file = tmp_path / "result.txt"
        assert result_file.exists(), "result.txt should exist in the workspace"
        assert "81" in result_file.read_text(), "result.txt should contain '81'"

        run_output = agent.get_run_output()
        assert run_output is not None
        assert run_output.tool_call_count >= 2

        if daytona._sandbox is not None:
            try:
                daytona.daytona.delete(daytona._sandbox, timeout=60)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_autonomous_agent_no_local_shell(self, tmp_path):
        """With enable_shell=False, local shell tools should NOT be registered."""
        from upsonic import AutonomousAgent
        from upsonic.tools.custom_tools.daytona import DaytonaTools

        daytona = DaytonaTools(api_key=DAYTONA_API_KEY, timeout=300)

        agent = AutonomousAgent(
            model=MODEL,
            workspace=str(tmp_path),
            enable_filesystem=True,
            enable_shell=False,
            tools=[daytona],
        )

        # Shell toolkit should not exist
        assert agent.shell_toolkit is None, "shell_toolkit should be None when enable_shell=False"
        # Filesystem toolkit should exist
        assert agent.filesystem_toolkit is not None, "filesystem_toolkit should exist"

        if daytona._sandbox is not None:
            try:
                daytona.daytona.delete(daytona._sandbox, timeout=60)
            except Exception:
                pass
