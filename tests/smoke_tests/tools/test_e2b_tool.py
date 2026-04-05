"""
Smoke tests for E2BTools with Agent and AutonomousAgent.

Requires a valid E2B_API_KEY environment variable.
Automatically skipped when the key is not set.

Coverage:
- E2BTools instantiation and lazy sandbox creation
- Agent + E2BTools: e2b_run_code, e2b_run_command, e2b_install_packages
- AutonomousAgent + E2BTools: code execution routed to remote sandbox
- Task.tool_calls populated with tool_name, params, tool_result
- AgentRunOutput.tools populated with ToolExecution entries
- AgentRunOutput.tool_call_count incremented

Run:
    uv run pytest tests/smoke_tests/tools/test_e2b_tool.py -v --tb=short -s
"""

import json
import os
from io import StringIO
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import pytest

E2B_API_KEY: Optional[str] = os.getenv("E2B_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not E2B_API_KEY,
        reason="E2B_API_KEY not set; skipping E2B smoke tests",
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
def e2b_tools():
    from upsonic.tools.custom_tools.e2b import E2BTools
    tools = E2BTools(api_key=E2B_API_KEY, timeout=300)
    yield tools
    # Cleanup: shutdown sandbox if it was created
    if tools._sandbox is not None:
        try:
            tools.sandbox.kill()
        except Exception:
            pass


# ------------------------------------------------------------------
# Unit: E2BTools construction and lazy init
# ------------------------------------------------------------------

class TestE2BToolsConstruction:
    """Test E2BTools construction without creating a sandbox."""

    def test_init_with_api_key(self):
        from upsonic.tools.custom_tools.e2b import E2BTools
        tools = E2BTools(api_key=E2B_API_KEY)
        assert tools.api_key == E2B_API_KEY
        assert tools._sandbox is None, "Sandbox should not be created at init (lazy)"

    def test_init_missing_api_key_raises(self):
        from upsonic.tools.custom_tools.e2b import E2BTools
        original = os.environ.get("E2B_API_KEY")
        try:
            os.environ.pop("E2B_API_KEY", None)
            with pytest.raises(ValueError, match="E2B API key is required"):
                E2BTools(api_key=None)
        finally:
            if original:
                os.environ["E2B_API_KEY"] = original

    def test_init_custom_timeout(self):
        from upsonic.tools.custom_tools.e2b import E2BTools
        tools = E2BTools(api_key=E2B_API_KEY, timeout=600)
        assert tools._timeout == 600

    def test_init_sandbox_options(self):
        from upsonic.tools.custom_tools.e2b import E2BTools
        opts = {"envs": {"FOO": "bar"}, "metadata": {"project": "test"}}
        tools = E2BTools(api_key=E2B_API_KEY, sandbox_options=opts)
        assert tools.sandbox_options == opts

    def test_init_default_sandbox_options(self):
        from upsonic.tools.custom_tools.e2b import E2BTools
        tools = E2BTools(api_key=E2B_API_KEY)
        assert tools.sandbox_options == {}

    def test_lazy_sandbox_creation(self, e2b_tools):
        """Accessing .sandbox should trigger creation."""
        sandbox = e2b_tools.sandbox
        assert sandbox is not None
        assert e2b_tools._sandbox is not None


# ------------------------------------------------------------------
# Unit: Direct method calls on E2BTools
# ------------------------------------------------------------------

class TestE2BToolsDirect:
    """Test E2BTools methods directly (not via agent)."""

    def test_e2b_run_code_python(self, e2b_tools):
        result = e2b_tools.e2b_run_code("print(2 + 3)")
        print(f"e2b_run_code result: {result}")
        parsed = json.loads(result)
        assert "error" not in parsed or parsed.get("error") is None
        logs = parsed.get("logs", {})
        stdout = logs.get("stdout", [])
        assert any("5" in line for line in stdout), f"Expected '5' in stdout, got: {parsed}"

    def test_e2b_run_code_with_result(self, e2b_tools):
        result = e2b_tools.e2b_run_code("2 ** 10")
        print(f"e2b_run_code expression result: {result}")
        parsed = json.loads(result)
        results = parsed.get("results", [])
        assert len(results) > 0, f"Expected results, got: {parsed}"
        assert any("1024" in str(r) for r in results), f"Expected '1024' in results, got: {results}"

    def test_e2b_run_code_error(self, e2b_tools):
        result = e2b_tools.e2b_run_code("raise ValueError('test error')")
        print(f"e2b_run_code error result: {result}")
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["error"]["name"] == "ValueError"
        assert "test error" in parsed["error"]["value"]

    def test_e2b_run_code_javascript(self, e2b_tools):
        result = e2b_tools.e2b_run_code("console.log(42)", language="javascript")
        print(f"e2b_run_code javascript result: {result}")
        parsed = json.loads(result)
        assert "error" not in parsed or parsed.get("error") is None

    def test_e2b_run_command(self, e2b_tools):
        result = e2b_tools.e2b_run_command("echo hello_e2b")
        print(f"e2b_run_command result: {result}")
        parsed = json.loads(result)
        assert "hello_e2b" in parsed.get("stdout", ""), f"Expected 'hello_e2b' in stdout, got: {parsed}"

    def test_e2b_run_command_exit_code(self, e2b_tools):
        result = e2b_tools.e2b_run_command("exit 0")
        print(f"e2b_run_command exit_code result: {result}")
        parsed = json.loads(result)
        assert parsed.get("exit_code") == 0

    def test_e2b_write_and_read_file(self, e2b_tools):
        write_result = e2b_tools.e2b_write_file("/home/user/test_e2b.txt", "hello from e2b")
        print(f"e2b_write_file result: {write_result}")
        parsed_write = json.loads(write_result)
        assert "path" in parsed_write

        read_result = e2b_tools.e2b_read_file("/home/user/test_e2b.txt")
        print(f"e2b_read_file result: {read_result}")
        assert "hello from e2b" in read_result

    def test_e2b_list_files(self, e2b_tools):
        result = e2b_tools.e2b_list_files("/home/user")
        print(f"e2b_list_files result: {result}")
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_e2b_install_packages(self, e2b_tools):
        result = e2b_tools.e2b_install_packages(["cowsay"])
        print(f"e2b_install_packages result: {result}")
        parsed = json.loads(result)
        assert "error" not in parsed or parsed.get("error") is None

    def test_e2b_get_sandbox_info(self, e2b_tools):
        result = e2b_tools.e2b_get_sandbox_info()
        print(f"e2b_get_sandbox_info result: {result}")
        parsed = json.loads(result)
        assert "sandbox_id" in parsed
        assert isinstance(parsed["sandbox_id"], str)
        assert len(parsed["sandbox_id"]) > 0

    def test_e2b_run_code_no_output(self, e2b_tools):
        """Code that produces no output should return a success message."""
        result = e2b_tools.e2b_run_code("x = 1")
        print(f"e2b_run_code no output result: {result}")
        parsed = json.loads(result)
        assert "error" not in parsed or parsed.get("error") is None

    def test_e2b_run_command_with_stderr(self, e2b_tools):
        """Command that writes to stderr."""
        result = e2b_tools.e2b_run_command("echo err_msg >&2")
        print(f"e2b_run_command stderr result: {result}")
        parsed = json.loads(result)
        assert "err_msg" in parsed.get("stderr", ""), f"Expected 'err_msg' in stderr, got: {parsed}"

    def test_e2b_upload_file(self, e2b_tools, tmp_path):
        """Upload a local file to the sandbox."""
        local_file = tmp_path / "upload_test.txt"
        local_file.write_text("upload content from local")
        result = e2b_tools.e2b_upload_file(str(local_file), "/home/user/uploaded.txt")
        print(f"e2b_upload_file result: {result}")
        parsed = json.loads(result)
        assert "path" in parsed, f"Expected 'path' in result, got: {parsed}"
        assert parsed["path"] == "/home/user/uploaded.txt"

        # Verify the file was actually uploaded by reading it back
        read_result = e2b_tools.e2b_read_file("/home/user/uploaded.txt")
        print(f"e2b_read_file after upload: {read_result}")
        assert "upload content from local" in read_result

    def test_e2b_upload_file_default_path(self, e2b_tools, tmp_path):
        """Upload without specifying sandbox_path should default to /home/user/<filename>."""
        local_file = tmp_path / "default_upload.txt"
        local_file.write_text("default path test")
        result = e2b_tools.e2b_upload_file(str(local_file))
        print(f"e2b_upload_file default path result: {result}")
        parsed = json.loads(result)
        assert "path" in parsed
        assert "default_upload.txt" in parsed["path"]

    def test_e2b_download_file(self, e2b_tools, tmp_path):
        """Download a file from the sandbox to local."""
        # First write a file in the sandbox
        e2b_tools.e2b_write_file("/home/user/download_test.txt", "download content from sandbox")

        local_dest = str(tmp_path / "downloaded.txt")
        result = e2b_tools.e2b_download_file("/home/user/download_test.txt", local_dest)
        print(f"e2b_download_file result: {result}")
        parsed = json.loads(result)
        assert "saved_to" in parsed, f"Expected 'saved_to' in result, got: {parsed}"

        # Verify local file content
        with open(local_dest) as f:
            content = f.read()
        assert "download content from sandbox" in content, f"Expected content in downloaded file, got: {content}"

    def test_e2b_install_packages_javascript(self, e2b_tools):
        """Install npm packages with language='javascript'."""
        result = e2b_tools.e2b_install_packages(["cowsay"], language="javascript")
        print(f"e2b_install_packages javascript result: {result}")
        parsed = json.loads(result)
        # npm install may produce output or succeed silently
        assert "error" not in parsed or parsed.get("error") is None, f"Unexpected error: {parsed}"

    def test_e2b_install_packages_unsupported_language(self, e2b_tools):
        """Unsupported language should return error."""
        result = e2b_tools.e2b_install_packages(["pkg"], language="ruby")
        print(f"e2b_install_packages unsupported lang result: {result}")
        parsed = json.loads(result)
        assert "error" in parsed, f"Expected error for unsupported language, got: {parsed}"

    def test_e2b_run_command_with_timeout(self, e2b_tools):
        """Run command with explicit timeout parameter."""
        result = e2b_tools.e2b_run_command("echo timeout_test", timeout=30)
        print(f"e2b_run_command with timeout result: {result}")
        parsed = json.loads(result)
        assert "timeout_test" in parsed.get("stdout", ""), f"Expected 'timeout_test' in stdout, got: {parsed}"

    def test_e2b_run_code_with_envs(self, e2b_tools):
        """Run code with custom environment variables."""
        result = e2b_tools.e2b_run_code(
            "import os; print(os.environ.get('TEST_VAR', 'not set'))",
            envs={"TEST_VAR": "hello_env"},
        )
        print(f"e2b_run_code with envs result: {result}")
        parsed = json.loads(result)
        logs = parsed.get("logs", {})
        stdout = logs.get("stdout", [])
        assert any("hello_env" in line for line in stdout), f"Expected 'hello_env' in stdout, got: {parsed}"


# ------------------------------------------------------------------
# Shutdown test (must run last, creates its own sandbox)
# ------------------------------------------------------------------

class TestE2BShutdown:
    """Test e2b_shutdown_sandbox - uses its own sandbox since it kills it."""

    def test_e2b_shutdown_sandbox(self):
        from upsonic.tools.custom_tools.e2b import E2BTools
        tools = E2BTools(api_key=E2B_API_KEY, timeout=300)
        # Force sandbox creation
        _ = tools.sandbox
        assert tools._sandbox is not None, "Sandbox should be created"

        result = tools.e2b_shutdown_sandbox()
        print(f"e2b_shutdown_sandbox result: {result}")
        parsed = json.loads(result)
        assert parsed.get("status") == "success", f"Expected success, got: {parsed}"
        assert "shut down" in parsed.get("message", "").lower(), f"Expected shutdown message, got: {parsed}"
        assert tools._sandbox is None, "Sandbox should be None after shutdown"


# ------------------------------------------------------------------
# E2E: Agent + E2BTools
# ------------------------------------------------------------------

class TestAgentWithE2B:
    """Test Agent using E2BTools -- verifies tool_calls on Task and AgentRunOutput."""

    @pytest.mark.asyncio
    async def test_agent_run_code_via_task(self):
        """Agent should call e2b_run_code and record it in task.tool_calls."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.e2b import E2BTools

        e2b = E2BTools(api_key=E2B_API_KEY, timeout=300)

        agent = Agent(model=MODEL, tools=[e2b])
        task = Task(
            description=(
                "Use the e2b_run_code tool to execute this Python code: print(7 * 8). "
                "Return the result number."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- Agent + E2B terminal output ---\n{terminal_output}\n---")

        # Task.tool_calls should be populated
        assert len(task.tool_calls) > 0, f"task.tool_calls should not be empty, got: {task.tool_calls}"
        called_names = _get_tool_call_names(task.tool_calls)
        assert "e2b_run_code" in called_names, f"e2b_run_code should be in tool_calls, got: {called_names}"

        # Verify tool_call structure
        run_code_call = next(tc for tc in task.tool_calls if tc.get("tool_name") == "e2b_run_code")
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
        if e2b._sandbox is not None:
            try:
                e2b.sandbox.kill()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_agent_run_command_via_task(self):
        """Agent should call e2b_run_command and record it in task.tool_calls."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.e2b import E2BTools

        e2b = E2BTools(api_key=E2B_API_KEY, timeout=300)

        agent = Agent(model=MODEL, tools=[e2b])
        task = Task(
            description=(
                "Use the e2b_run_command tool to execute: echo UPSONIC_E2B_TEST. "
                "Return the output."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- Agent + E2B run_command output ---\n{terminal_output}\n---")

        assert len(task.tool_calls) > 0, "task.tool_calls should not be empty"
        called_names = _get_tool_call_names(task.tool_calls)
        assert "e2b_run_command" in called_names, f"e2b_run_command should be in tool_calls, got: {called_names}"

        run_output = agent.get_run_output()
        assert run_output is not None
        assert run_output.tool_call_count > 0

        if e2b._sandbox is not None:
            try:
                e2b.sandbox.kill()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_agent_install_and_run(self):
        """Agent should install a package then run code using it."""
        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.e2b import E2BTools

        e2b = E2BTools(api_key=E2B_API_KEY, timeout=300)

        agent = Agent(model=MODEL, tools=[e2b])
        task = Task(
            description=(
                "First use e2b_install_packages to install the 'cowsay' package (language python), "
                "then use e2b_run_code to execute: import cowsay; print(cowsay.cow('E2B works!')). "
                "Return the output."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- Agent install+run output ---\n{terminal_output}\n---")

        called_names = _get_tool_call_names(task.tool_calls)
        assert "e2b_install_packages" in called_names, f"e2b_install_packages should be in tool_calls, got: {called_names}"
        assert "e2b_run_code" in called_names, f"e2b_run_code should be in tool_calls, got: {called_names}"

        run_output = agent.get_run_output()
        assert run_output is not None
        assert run_output.tool_call_count >= 2, f"Expected >= 2 tool calls, got: {run_output.tool_call_count}"

        if e2b._sandbox is not None:
            try:
                e2b.sandbox.kill()
            except Exception:
                pass


# ------------------------------------------------------------------
# E2E: AutonomousAgent + E2BTools (no local shell)
# ------------------------------------------------------------------

class TestAutonomousAgentWithE2B:
    """Test AutonomousAgent with local filesystem + remote E2B execution."""

    @pytest.mark.asyncio
    async def test_autonomous_agent_e2b_code_execution(self, tmp_path):
        """AutonomousAgent should use E2B for code execution, not local shell."""
        from upsonic import AutonomousAgent, Task
        from upsonic.tools.custom_tools.e2b import E2BTools

        e2b = E2BTools(api_key=E2B_API_KEY, timeout=300)

        agent = AutonomousAgent(
            model=MODEL,
            workspace=str(tmp_path),
            enable_filesystem=True,
            enable_shell=False,  # No local shell
            tools=[e2b],
        )

        task = Task(
            description=(
                "Use the e2b_run_code tool to execute this Python code: "
                "print(sum(range(1, 11))). Return only the result number."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- AutonomousAgent + E2B output ---\n{terminal_output}\n---")

        # Should have used e2b_run_code from E2B, not local shell
        assert len(task.tool_calls) > 0, "task.tool_calls should not be empty"
        called_names = _get_tool_call_names(task.tool_calls)
        assert "e2b_run_code" in called_names, f"e2b_run_code should be in tool_calls, got: {called_names}"

        # Local shell tools should NOT be available
        assert "run_python" not in called_names, "Local run_python should not be called"

        # Verify AgentRunOutput
        run_output = agent.get_run_output()
        assert run_output is not None
        assert run_output.tool_call_count > 0

        # Result should contain 55
        response_text = str(task.response) if task.response else terminal_output
        assert "55" in response_text, f"Response should contain '55', got: {response_text}"

        if e2b._sandbox is not None:
            try:
                e2b.sandbox.kill()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_autonomous_agent_filesystem_and_e2b(self, tmp_path):
        """AutonomousAgent should have both local filesystem and remote E2B tools."""
        from upsonic import AutonomousAgent, Task
        from upsonic.tools.custom_tools.e2b import E2BTools

        e2b = E2BTools(api_key=E2B_API_KEY, timeout=300)

        agent = AutonomousAgent(
            model=MODEL,
            workspace=str(tmp_path),
            enable_filesystem=True,
            enable_shell=False,
            tools=[e2b],
        )

        task = Task(
            description=(
                "Do these steps:\n"
                "1. Use e2b_run_code to execute: print(3 ** 4)\n"
                "2. Use write_file (the local filesystem tool) to create a file "
                "called 'result.txt' in the workspace with the content '81'\n"
                "Return 'done' when finished."
            ),
        )

        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)

        terminal_output = output_buffer.getvalue()
        print(f"\n--- AutonomousAgent filesystem+E2B output ---\n{terminal_output}\n---")

        called_names = _get_tool_call_names(task.tool_calls)
        assert "e2b_run_code" in called_names, f"e2b_run_code (E2B) should be in tool_calls, got: {called_names}"
        assert "write_file" in called_names, f"write_file (local) should be in tool_calls, got: {called_names}"

        # Verify the file was written locally
        result_file = tmp_path / "result.txt"
        assert result_file.exists(), "result.txt should exist in the workspace"
        assert "81" in result_file.read_text(), "result.txt should contain '81'"

        run_output = agent.get_run_output()
        assert run_output is not None
        assert run_output.tool_call_count >= 2

        if e2b._sandbox is not None:
            try:
                e2b.sandbox.kill()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_autonomous_agent_no_local_shell(self, tmp_path):
        """With enable_shell=False, local shell tools should NOT be registered."""
        from upsonic import AutonomousAgent
        from upsonic.tools.custom_tools.e2b import E2BTools

        e2b = E2BTools(api_key=E2B_API_KEY, timeout=300)

        agent = AutonomousAgent(
            model=MODEL,
            workspace=str(tmp_path),
            enable_filesystem=True,
            enable_shell=False,
            tools=[e2b],
        )

        # Shell toolkit should not exist
        assert agent.shell_toolkit is None, "shell_toolkit should be None when enable_shell=False"
        # Filesystem toolkit should exist
        assert agent.filesystem_toolkit is not None, "filesystem_toolkit should exist"

        if e2b._sandbox is not None:
            try:
                e2b.sandbox.kill()
            except Exception:
                pass
