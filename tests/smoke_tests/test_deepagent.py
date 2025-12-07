"""
Test 9: DeepAgent test with enable_planning, enable_filesystem, subagents, filesystem_backend

Success criteria:
- DeepAgent properly creates todos and executes them all properly
- Files exist that we want to create as described in the Task description
- DeepAgent registers and calls subagents properly
- We use CompositeBackend properly
- We check attributes and logs for that
"""

import pytest
import os
import tempfile
import shutil
from upsonic import Agent, Task
from upsonic.agent.deepagent import DeepAgent
from upsonic.agent.deepagent.backends import StateBackend, MemoryBackend, CompositeBackend
from upsonic.storage.providers import InMemoryStorage, SqliteStorage
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(180)


@pytest.fixture
def temp_dir():
    """Create temporary directory for JSON storage."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def sqlite_db_file():
    """Create temporary SQLite database file."""
    db_file = tempfile.mktemp(suffix=".db")
    yield db_file
    if os.path.exists(db_file):
        os.remove(db_file)


@pytest.mark.asyncio
async def test_deepagent_planning_and_filesystem_state_backend():
    """Test DeepAgent with planning, filesystem, and StateBackend."""
    # Create DeepAgent with StateBackend
    backend = StateBackend()
    agent = DeepAgent(
        model="openai/gpt-4o",
        name="Test DeepAgent",
        enable_planning=True,
        enable_filesystem=True,
        enable_subagents=False,
        filesystem_backend=backend,
        debug=True
    )
    
    # Verify attributes
    assert agent.enable_planning is True, "enable_planning should be True"
    assert agent.enable_filesystem is True, "enable_filesystem should be True"
    assert agent.filesystem_backend is not None, "filesystem_backend should be set"
    assert isinstance(agent.filesystem_backend, StateBackend), "filesystem_backend should be StateBackend"
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Task that requires planning and file creation
        task = Task(
            description=(
                "Create a plan to write two files: "
                "1. /documents/plan.txt with content 'Project Plan' "
                "2. /documents/summary.txt with content 'Project Summary'. "
                "Execute the plan and create both files."
            )
        )
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify todos were created and executed
    todos = agent.get_current_plan()
    assert len(todos) >= 2, f"Should have at least 2 todos, got {len(todos)}"
    
    # Check that files were created
    file1_content = await backend.read("/documents/plan.txt")
    file2_content = await backend.read("/documents/summary.txt")
    
    assert "Project Plan" in file1_content or "plan" in file1_content.lower(), f"File 1 should contain plan content. Got: {file1_content}"
    assert "Project Summary" in file2_content or "summary" in file2_content.lower(), f"File 2 should contain summary content. Got: {file2_content}"
    
    # Verify logs contain planning and filesystem operations
    assert "write_todos" in output.lower() or "todo" in output.lower(), f"Should see todo/planning logs. Output: {output[:500]}"
    assert "write_file" in output.lower() or "file" in output.lower(), f"Should see file operation logs. Output: {output[:500]}"


@pytest.mark.asyncio
async def test_deepagent_with_subagents():
    """Test DeepAgent with subagents."""
    # Create subagents
    researcher = Agent(
        model="openai/gpt-4o",
        name="researcher",
        role="Research Specialist",
        goal="Conduct thorough research on topics",
        debug=True
    )
    
    writer = Agent(
        model="openai/gpt-4o",
        name="writer",
        role="Content Writer",
        goal="Write clear and engaging content",
        debug=True
    )
    
    # Create DeepAgent with subagents
    agent = DeepAgent(
        model="openai/gpt-4o",
        name="Test DeepAgent",
        enable_planning=True,
        enable_filesystem=True,
        enable_subagents=True,
        subagents=[researcher, writer],
        debug=True
    )
    
    # Verify attributes
    assert agent.enable_subagents is True, "enable_subagents should be True"
    assert len(agent.subagents) >= 2, f"Should have at least 2 subagents, got {len(agent.subagents)}"
    assert "researcher" in agent.get_subagent_names(), "researcher subagent should be registered"
    assert "writer" in agent.get_subagent_names(), "writer subagent should be registered"
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Task that requires subagent delegation
        task = Task(
            description=(
                "Use the researcher subagent to research Python programming basics, "
                "then use the writer subagent to write a summary of the research findings."
            )
        )
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify subagent calls in logs
    assert "researcher" in output.lower() or "task" in output.lower(), f"Should see subagent/researcher logs. Output: {output[:500]}"
    assert "writer" in output.lower() or "task" in output.lower(), f"Should see subagent/writer logs. Output: {output[:500]}"


@pytest.mark.asyncio
async def test_deepagent_with_memory_backend_sqlite(sqlite_db_file):
    """Test DeepAgent with MemoryBackend using SqliteStorage."""
    # Create storage and backend
    storage = SqliteStorage(
        db_file=sqlite_db_file,
        sessions_table_name="sessions",
        profiles_table_name="profiles"
    )
    await storage.connect_async()
    await storage.create_async()
    
    backend = MemoryBackend(storage)
    agent = DeepAgent(
        model="openai/gpt-4o",
        name="Test DeepAgent",
        enable_planning=True,
        enable_filesystem=True,
        enable_subagents=False,
        filesystem_backend=backend,
        debug=True
    )
    
    # Verify backend type
    assert isinstance(agent.filesystem_backend, MemoryBackend), "filesystem_backend should be MemoryBackend"
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Task that creates files
        task = Task(
            description=(
                "Create a file at /persistent/notes.txt with content 'Important Notes'. "
                "Then read the file to verify it was created."
            )
        )
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify file was created and persisted
    file_content = await backend.read("/persistent/notes.txt")
    assert "Important Notes" in file_content or "notes" in file_content.lower(), f"File should contain notes content. Got: {file_content}"
    
    # Verify logs
    assert "write_file" in output.lower() or "file" in output.lower(), f"Should see file operation logs. Output: {output[:500]}"
    
    # Cleanup
    await storage.disconnect_async()


@pytest.mark.asyncio
async def test_deepagent_with_composite_backend():
    """Test DeepAgent with CompositeBackend."""
    # Create multiple backends
    memory_storage = InMemoryStorage()
    await memory_storage.connect_async()
    await memory_storage.create_async()
    
    memory_backend = MemoryBackend(memory_storage)
    state_backend = StateBackend()
    
    # Create composite backend with routes
    composite_backend = CompositeBackend(
        default=state_backend,
        routes={
            "/persistent/": memory_backend,
            "/memories/": memory_backend,
            "/tmp/": state_backend
        }
    )
    
    agent = DeepAgent(
        model="openai/gpt-4o",
        name="Test DeepAgent",
        enable_planning=True,
        enable_filesystem=True,
        enable_subagents=False,
        filesystem_backend=composite_backend,
        debug=True
    )
    
    # Verify backend type
    assert isinstance(agent.filesystem_backend, CompositeBackend), "filesystem_backend should be CompositeBackend"
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Task that creates files in different paths (different backends)
        task = Task(
            description=(
                "Create two files: "
                "1. /persistent/important.txt with content 'Persistent Data' (should use MemoryBackend) "
                "2. /tmp/temp.txt with content 'Temporary Data' (should use StateBackend). "
                "Then read both files to verify they were created correctly."
            )
        )
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify files were created in correct backends
    persistent_content = await composite_backend.read("/persistent/important.txt")
    temp_content = await composite_backend.read("/tmp/temp.txt")
    
    assert "Persistent Data" in persistent_content or "persistent" in persistent_content.lower(), f"Persistent file should exist. Got: {persistent_content}"
    assert "Temporary Data" in temp_content or "temporary" in temp_content.lower(), f"Temp file should exist. Got: {temp_content}"
    
    # Verify logs show composite backend usage
    assert "write_file" in output.lower() or "file" in output.lower(), f"Should see file operation logs. Output: {output[:500]}"
    
    # Cleanup
    await memory_storage.disconnect_async()


@pytest.mark.asyncio
async def test_deepagent_full_features():
    """Test DeepAgent with all features enabled: planning, filesystem, subagents."""
    # Create subagent
    assistant = Agent(
        model="openai/gpt-4o",
        name="assistant",
        role="Assistant",
        goal="Help with various tasks",
        debug=True
    )
    
    # Create DeepAgent with all features
    backend = StateBackend()
    agent = DeepAgent(
        model="openai/gpt-4o",
        name="Test DeepAgent",
        enable_planning=True,
        enable_filesystem=True,
        enable_subagents=True,
        subagents=[assistant],
        filesystem_backend=backend,
        debug=True
    )
    
    # Verify all features are enabled
    assert agent.enable_planning is True, "enable_planning should be True"
    assert agent.enable_filesystem is True, "enable_filesystem should be True"
    assert agent.enable_subagents is True, "enable_subagents should be True"
    assert len(agent.subagents) >= 1, "Should have at least 1 subagent"
    
    # Capture stdout
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Complex task requiring planning, filesystem, and subagents
        task = Task(
            description=(
                "Create a plan to: "
                "1. Use the assistant subagent to generate a short story about AI "
                "2. Write the story to /documents/story.txt "
                "3. Create a summary file at /documents/summary.txt with a brief summary. "
                "Execute the plan completely."
            )
        )
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify todos were created
    todos = agent.get_current_plan()
    assert len(todos) >= 2, f"Should have at least 2 todos, got {len(todos)}"
    
    # Verify files were created
    story_content = await backend.read("/documents/story.txt")
    summary_content = await backend.read("/documents/summary.txt")
    
    assert len(story_content) > 0, "Story file should have content"
    assert len(summary_content) > 0, "Summary file should have content"
    
    # Verify logs contain all features
    assert "write_todos" in output.lower() or "todo" in output.lower(), f"Should see planning logs. Output: {output[:500]}"
    assert "write_file" in output.lower() or "file" in output.lower(), f"Should see filesystem logs. Output: {output[:500]}"
    assert "task" in output.lower() or "assistant" in output.lower(), f"Should see subagent logs. Output: {output[:500]}"

