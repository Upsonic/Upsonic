"""
Comprehensive smoke tests for AutonomousAgent.

Tests cover:
- Class inheritance from Agent
- Default storage and memory setup
- Filesystem toolkit operations
- Shell toolkit operations
- Real LLM integration with tools
- Workspace sandboxing

All tests use REAL LLM requests - no mocking.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from upsonic import AutonomousAgent
from upsonic.agent import Agent
from upsonic.agent.autonomous_agent import (
    AutonomousFilesystemToolKit,
    AutonomousShellToolKit,
)
from upsonic.storage import InMemoryStorage, Memory

pytestmark = pytest.mark.timeout(180)




@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for tests."""
    workspace = tempfile.mkdtemp(prefix="autonomous_agent_test_")
    yield workspace
    # Cleanup after test
    if os.path.exists(workspace):
        shutil.rmtree(workspace)


@pytest.fixture
def sample_python_file(temp_workspace):
    """Create a sample Python file in the workspace."""
    file_path = Path(temp_workspace) / "sample.py"
    content = '''"""Sample Python module for testing."""

def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    print(hello("World"))
'''
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_json_file(temp_workspace):
    """Create a sample JSON file in the workspace."""
    file_path = Path(temp_workspace) / "config.json"
    content = '{"name": "test", "version": "1.0.0", "enabled": true}'
    file_path.write_text(content)
    return str(file_path)


# ---------------------------------------------------------------------------
# Test: Class Inheritance
# ---------------------------------------------------------------------------

class TestAutonomousAgentInheritance:
    """Tests verifying AutonomousAgent properly inherits from Agent."""
    
    def test_inherits_from_agent(self):
        """Verify AutonomousAgent is a subclass of Agent."""
        assert issubclass(AutonomousAgent, Agent)
    
    def test_mro_includes_agent(self):
        """Verify Method Resolution Order includes Agent and BaseAgent."""
        mro_names = [cls.__name__ for cls in AutonomousAgent.__mro__]
        assert "Agent" in mro_names
        assert "BaseAgent" in mro_names
    
    def test_instance_is_agent(self, temp_workspace):
        """Verify AutonomousAgent instance is also an Agent instance."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        assert isinstance(agent, Agent)
        assert isinstance(agent, AutonomousAgent)


# ---------------------------------------------------------------------------
# Test: Initialization and Default Storage/Memory
# ---------------------------------------------------------------------------

class TestAutonomousAgentInitialization:
    """Tests for AutonomousAgent initialization with default storage and memory."""
    
    def test_default_storage_created(self, temp_workspace):
        """Verify InMemoryStorage is created by default."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        assert agent.autonomous_storage is not None
        assert isinstance(agent.autonomous_storage, InMemoryStorage)
    
    def test_default_memory_created(self, temp_workspace):
        """Verify Memory is created by default with the storage."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        assert agent.autonomous_memory is not None
        assert isinstance(agent.autonomous_memory, Memory)
        assert agent.memory is not None
    
    def test_custom_storage_used(self, temp_workspace):
        """Verify custom storage is used when provided."""
        custom_storage = InMemoryStorage()
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
            storage=custom_storage,
        )
        assert agent.autonomous_storage is custom_storage
    
    def test_custom_memory_used(self, temp_workspace):
        """Verify custom memory is used when provided."""
        custom_storage = InMemoryStorage()
        custom_memory = Memory(storage=custom_storage)
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
            memory=custom_memory,
        )
        assert agent.memory is custom_memory
    
    def test_workspace_resolved(self, temp_workspace):
        """Verify workspace path is properly resolved."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        assert agent.autonomous_workspace == Path(temp_workspace).resolve()
    
    def test_default_workspace_is_cwd(self):
        """Verify default workspace is current working directory."""
        agent = AutonomousAgent(model="openai/gpt-4o-mini")
        assert agent.autonomous_workspace == Path.cwd().resolve()
    
    def test_toolkits_created(self, temp_workspace):
        """Verify filesystem and shell toolkits are created."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        assert agent.filesystem_toolkit is not None
        assert agent.shell_toolkit is not None
        assert isinstance(agent.filesystem_toolkit, AutonomousFilesystemToolKit)
        assert isinstance(agent.shell_toolkit, AutonomousShellToolKit)
    
    def test_toolkits_disabled(self, temp_workspace):
        """Verify toolkits can be disabled."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
            enable_filesystem=False,
            enable_shell=False,
        )
        assert agent.filesystem_toolkit is None
        assert agent.shell_toolkit is None
    
    def test_tools_registered(self, temp_workspace):
        """Verify toolkits are registered as tools."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        # The toolkits should be in the tools list
        assert len(agent.tools) >= 2
    
    def test_agent_properties(self, temp_workspace):
        """Verify agent properties are accessible."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
            name="TestAgent",
        )
        assert agent.name == "TestAgent"
        assert agent.agent_id is not None
        assert agent.model is not None


# ---------------------------------------------------------------------------
# Test: Filesystem Toolkit Operations (Direct)
# ---------------------------------------------------------------------------

class TestFilesystemToolkitDirect:
    """Tests for filesystem toolkit operations called directly."""
    
    def test_read_file(self, temp_workspace, sample_python_file):
        """Test reading a file directly."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.read_file("sample.py")
        
        assert "def hello" in result
        assert "def add" in result
        assert "1|" in result  # Line numbers
    
    def test_read_file_with_pagination(self, temp_workspace, sample_python_file):
        """Test reading a file with offset and limit."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.read_file("sample.py", offset=0, limit=5)
        
        assert "Sample Python module" in result
        assert "Showing lines 1-5" in result
    
    def test_write_file(self, temp_workspace):
        """Test writing a new file."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        content = "print('Hello, World!')"
        result = toolkit.write_file("new_file.py", content)
        
        assert "✅" in result
        assert "Successfully wrote" in result
        
        # Verify file exists
        file_path = Path(temp_workspace) / "new_file.py"
        assert file_path.exists()
        assert file_path.read_text() == content
    
    def test_edit_file_requires_read(self, temp_workspace, sample_python_file):
        """Test that edit_file requires prior read_file call."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        
        # Try to edit without reading first
        result = toolkit.edit_file(
            "sample.py",
            "def hello",
            "def greet"
        )
        
        assert "❌" in result
        assert "must call read_file" in result
    
    def test_edit_file_after_read(self, temp_workspace, sample_python_file):
        """Test editing a file after reading it."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        
        # Read first
        toolkit.read_file("sample.py")
        
        # Then edit
        result = toolkit.edit_file(
            "sample.py",
            "def hello(name: str) -> str:",
            "def greet(name: str) -> str:"
        )
        
        assert "✅" in result
        assert "Successfully edited" in result
        
        # Verify change
        content = (Path(temp_workspace) / "sample.py").read_text()
        assert "def greet" in content
        assert "def hello" not in content
    
    def test_list_files(self, temp_workspace, sample_python_file, sample_json_file):
        """Test listing directory contents."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.list_files(".")
        
        assert "sample.py" in result
        assert "config.json" in result
    
    def test_list_files_recursive(self, temp_workspace):
        """Test recursive directory listing."""
        # Create nested structure
        subdir = Path(temp_workspace) / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text("# Main")
        
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.list_files(".", recursive=True)
        
        assert "[DIR]" in result
        assert "src" in result
        assert "main.py" in result
    
    def test_search_files(self, temp_workspace, sample_python_file, sample_json_file):
        """Test searching files by pattern."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.search_files("*.py")
        
        assert "sample.py" in result
        assert "config.json" not in result
    
    def test_grep_files(self, temp_workspace, sample_python_file):
        """Test searching text within files."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.grep_files("def", file_pattern="*.py")
        
        assert "sample.py" in result
        assert "def hello" in result or "hello" in result
    
    def test_file_info(self, temp_workspace, sample_python_file):
        """Test getting file information."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.file_info("sample.py")
        
        assert "sample.py" in result
        assert "Type: File" in result
        assert "Size:" in result
    
    def test_create_directory(self, temp_workspace):
        """Test creating a directory."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.create_directory("new_dir/sub_dir")
        
        assert "✅" in result
        assert (Path(temp_workspace) / "new_dir" / "sub_dir").exists()
    
    def test_copy_file(self, temp_workspace, sample_python_file):
        """Test copying a file."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.copy_file("sample.py", "sample_backup.py")
        
        assert "✅" in result
        assert (Path(temp_workspace) / "sample_backup.py").exists()
    
    def test_move_file(self, temp_workspace, sample_python_file):
        """Test moving a file."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.move_file("sample.py", "moved_sample.py")
        
        assert "✅" in result
        assert not (Path(temp_workspace) / "sample.py").exists()
        assert (Path(temp_workspace) / "moved_sample.py").exists()
    
    def test_delete_file(self, temp_workspace, sample_python_file):
        """Test deleting a file."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = toolkit.delete_file("sample.py")
        
        assert "✅" in result
        assert not (Path(temp_workspace) / "sample.py").exists()
    
    def test_workspace_sandboxing(self, temp_workspace):
        """Test that paths outside workspace are blocked."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        
        # Try to read file outside workspace
        with pytest.raises(ValueError) as exc_info:
            toolkit._validate_path("/etc/passwd")
        
        assert "outside workspace" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: Shell Toolkit Operations (Direct)
# ---------------------------------------------------------------------------

class TestShellToolkitDirect:
    """Tests for shell toolkit operations called directly."""
    
    def test_run_echo_command(self, temp_workspace):
        """Test running a simple echo command."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = toolkit.run_command("echo 'Hello, World!'")
        
        assert "Hello, World!" in result
        assert "Exit code: 0" in result
    
    def test_run_pwd_command(self, temp_workspace):
        """Test running pwd command in workspace."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = toolkit.run_command("pwd")
        
        assert temp_workspace in result
        assert "Exit code: 0" in result
    
    def test_run_ls_command(self, temp_workspace, sample_python_file):
        """Test running ls command."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = toolkit.run_command("ls -la")
        
        assert "sample.py" in result
        assert "Exit code: 0" in result
    
    def test_run_python_version(self, temp_workspace):
        """Test running python --version."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = toolkit.run_command("python3 --version")
        
        assert "Python" in result
        assert "Exit code: 0" in result
    
    def test_run_python_code(self, temp_workspace):
        """Test running Python code directly."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = toolkit.run_python("print(2 + 2)")
        
        assert "4" in result
        assert "Exit code: 0" in result
    
    def test_check_command_exists(self, temp_workspace):
        """Test checking if a command exists."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        
        # Check for a command that should exist
        result = toolkit.check_command_exists("python3")
        assert "✅" in result or "is available" in result
    
    def test_check_command_not_exists(self, temp_workspace):
        """Test checking for a non-existent command."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = toolkit.check_command_exists("nonexistent_command_xyz123")
        
        assert "❌" in result or "not available" in result
    
    def test_command_timeout(self, temp_workspace):
        """Test command timeout handling."""
        toolkit = AutonomousShellToolKit(
            workspace=temp_workspace,
            default_timeout=1,
        )
        result = toolkit.run_command("sleep 5", timeout=1)
        
        assert "timed out" in result.lower()
    
    def test_blocked_command(self, temp_workspace):
        """Test that dangerous commands are blocked."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = toolkit.run_command("rm -rf /")
        
        assert "blocked" in result.lower() or "error" in result.lower()
    
    def test_environment_variables(self, temp_workspace):
        """Test passing environment variables."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = toolkit.run_command(
            "echo $MY_TEST_VAR",
            env={"MY_TEST_VAR": "test_value_123"}
        )
        
        assert "test_value_123" in result


# ---------------------------------------------------------------------------
# Test: Real LLM Integration
# ---------------------------------------------------------------------------

class TestAutonomousAgentWithLLM:
    """Tests for AutonomousAgent with real LLM requests."""
    
    def test_simple_task_without_tools(self, temp_workspace):
        """Test simple task that doesn't require tools."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.do("What is 2 + 2? Reply with just the number.")
        
        assert result is not None
        assert "4" in result
    
    def test_read_file_via_llm(self, temp_workspace, sample_python_file):
        """Test LLM using read_file tool."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.do("Read the sample.py file and tell me what functions are defined in it.")
        
        assert result is not None
        # The LLM should mention the functions
        assert "hello" in result.lower() or "add" in result.lower()
    
    def test_write_file_via_llm(self, temp_workspace):
        """Test LLM using write_file tool."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.do(
            "Create a file called 'greeting.txt' with the content 'Hello from LLM!'. "
            "Confirm when done."
        )
        
        assert result is not None
        # Verify file was created
        greeting_file = Path(temp_workspace) / "greeting.txt"
        assert greeting_file.exists()
        content = greeting_file.read_text()
        assert "Hello" in content or "hello" in content.lower()
    
    def test_list_files_via_llm(self, temp_workspace, sample_python_file, sample_json_file):
        """Test LLM using list_files tool."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.do("List all files in the current directory and tell me what file types you see.")
        
        assert result is not None
        # Should mention Python and/or JSON files
        assert "py" in result.lower() or "python" in result.lower() or "json" in result.lower()
    
    def test_run_command_via_llm(self, temp_workspace):
        """Test LLM using run_command tool."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.do("Run 'echo Hello from shell' and tell me what the output was.")
        
        assert result is not None
        assert "Hello" in result or "hello" in result.lower()
    
    def test_python_execution_via_llm(self, temp_workspace):
        """Test LLM executing Python code."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.do(
            "Use the run_python tool to calculate 15 * 7 and tell me the result."
        )
        
        assert result is not None
        assert "105" in result
    
    def test_edit_file_via_llm(self, temp_workspace, sample_python_file):
        """Test LLM using edit_file tool (requires read first)."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.do(
            "Read sample.py, then edit it to change the function 'hello' to 'greet'. "
            "Keep the same functionality. Confirm the change was made."
        )
        
        assert result is not None
        # Verify the change was made
        content = (Path(temp_workspace) / "sample.py").read_text()
        assert "greet" in content.lower()
    
    def test_complex_multi_tool_task(self, temp_workspace):
        """Test LLM using multiple tools in sequence."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.do(
            "1. Create a directory called 'project'\n"
            "2. Inside 'project', create a file called 'main.py' with a simple hello world program\n"
            "3. List the contents of the 'project' directory\n"
            "4. Tell me what you created"
        )
        
        assert result is not None
        # Verify directory and file were created
        project_dir = Path(temp_workspace) / "project"
        main_file = project_dir / "main.py"
        assert project_dir.exists()
        assert main_file.exists()


# ---------------------------------------------------------------------------
# Test: Async Operations
# ---------------------------------------------------------------------------

class TestAutonomousAgentAsync:
    """Tests for async operations."""
    
    @pytest.mark.asyncio
    async def test_async_read_file(self, temp_workspace, sample_python_file):
        """Test async read_file operation."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = await toolkit.aread_file("sample.py")
        
        assert "def hello" in result
    
    @pytest.mark.asyncio
    async def test_async_write_file(self, temp_workspace):
        """Test async write_file operation."""
        toolkit = AutonomousFilesystemToolKit(workspace=temp_workspace)
        result = await toolkit.awrite_file("async_test.txt", "Async content")
        
        assert "✅" in result
        assert (Path(temp_workspace) / "async_test.txt").exists()
    
    @pytest.mark.asyncio
    async def test_async_run_command(self, temp_workspace):
        """Test async run_command operation."""
        toolkit = AutonomousShellToolKit(workspace=temp_workspace)
        result = await toolkit.arun_command("echo 'Async test'")
        
        assert "Async test" in result
    
    @pytest.mark.asyncio
    async def test_async_agent_do(self, temp_workspace):
        """Test async agent.do_async operation with LLM."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = await agent.do_async("What is 3 + 3? Reply with just the number.")
        
        assert result is not None
        assert "6" in result


# ---------------------------------------------------------------------------
# Test: Agent Methods Inherited from Agent
# ---------------------------------------------------------------------------

class TestInheritedAgentMethods:
    """Tests verifying inherited Agent methods work correctly."""
    
    def test_print_do_method(self, temp_workspace):
        """Test print_do method inherited from Agent."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        result = agent.print_do("Say hello")
        
        assert result is not None
    
    def test_agent_id_property(self, temp_workspace):
        """Test agent_id property inherited from Agent."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        assert agent.agent_id is not None
        assert isinstance(agent.agent_id, str)
    
    def test_get_cache_stats(self, temp_workspace):
        """Test get_cache_stats method inherited from Agent."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        stats = agent.get_cache_stats()
        
        assert stats is not None
        assert isinstance(stats, dict)
    
    def test_add_tools(self, temp_workspace):
        """Test add_tools method inherited from Agent."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        initial_tool_count = len(agent.tools)
        
        def custom_tool(x: int) -> int:
            """Multiply by 2."""
            return x * 2
        
        agent.add_tools(custom_tool)
        assert len(agent.tools) > initial_tool_count
    
    def test_repr(self, temp_workspace):
        """Test __repr__ method."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
            name="TestAgent",
        )
        repr_str = repr(agent)
        
        assert "AutonomousAgent" in repr_str
        assert "TestAgent" in repr_str
        assert temp_workspace in repr_str


# ---------------------------------------------------------------------------
# Test: Memory Integration
# ---------------------------------------------------------------------------

class TestMemoryIntegration:
    """Tests for memory integration with AutonomousAgent."""
    
    def test_memory_session_persistence(self, temp_workspace):
        """Test that memory persists across tasks."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
            full_session_memory=True,
        )
        
        # First task
        agent.do("Remember this number: 42")
        
        # Second task should recall
        result = agent.do("What number did I ask you to remember?")
        
        assert result is not None
        assert "42" in result
    
    def test_session_id_accessible(self, temp_workspace):
        """Test that session_id is accessible."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
            session_id="test_session_123",
        )
        
        assert agent.session_id == "test_session_123"
    
    def test_user_id_accessible(self, temp_workspace):
        """Test that user_id is accessible."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
            user_id="test_user_456",
        )
        
        assert agent.user_id == "test_user_456"


# ---------------------------------------------------------------------------
# Test: Reset Tracking
# ---------------------------------------------------------------------------

class TestFilesystemTracking:
    """Tests for filesystem read tracking reset."""
    
    def test_reset_filesystem_tracking(self, temp_workspace, sample_python_file):
        """Test reset_filesystem_tracking method."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=temp_workspace,
        )
        
        # Read a file through the toolkit
        agent.filesystem_toolkit.read_file("sample.py")
        
        # Verify it's tracked (use resolve to handle macOS symlinks like /var -> /private/var)
        read_files = agent.filesystem_toolkit.get_read_files()
        assert len(read_files) > 0
        # Check that sample.py is in the tracked files (comparing resolved paths)
        expected_path = Path(temp_workspace).resolve() / "sample.py"
        tracked_paths = [Path(f).resolve() for f in read_files]
        assert expected_path in tracked_paths
        
        # Reset tracking
        agent.reset_filesystem_tracking()
        
        # Verify it's cleared
        assert len(agent.filesystem_toolkit.get_read_files()) == 0
