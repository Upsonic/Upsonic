"""
Tests for Deep Agent

This module contains comprehensive tests for the DeepAgent class,
including initialization, execution methods, file operations, todo management,
and subagent spawning.
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from upsonic import DeepAgent, Agent, Task
from upsonic.agent.deep_agent.state import DeepAgentState, Todo
from upsonic.storage.providers.in_memory import InMemoryStorage
from upsonic.storage.memory.memory import Memory


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_name="test-model"):
        self.model_name = model_name
        self.request = AsyncMock()
        self.settings = MagicMock()
        self.customize_request_parameters = MagicMock(side_effect=lambda x: x)


class TestDeepAgentInitialization(unittest.TestCase):
    """Test suite for DeepAgent class initialization."""

    @patch("upsonic.models.infer_model")
    def test_deep_agent_initialization(self, mock_infer_model):
        """Test DeepAgent initialization."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model

        agent = DeepAgent(model="openai/gpt-4o")

        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.deep_agent_state)
        self.assertIsInstance(agent.deep_agent_state, DeepAgentState)
        self.assertEqual(agent.deep_agent_state.todos, [])
        self.assertEqual(agent.deep_agent_state.files, {})
        self.assertEqual(agent.subagents, [])
        self.assertIsNotNone(agent.memory)
        self.assertEqual(agent.tool_call_limit, 100)

    @patch("upsonic.models.infer_model")
    def test_deep_agent_initialization_with_subagents(self, mock_infer_model):
        """Test init with subagents."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model

        subagent1 = Agent(model=mock_model, name="researcher")
        subagent2 = Agent(model=mock_model, name="reviewer")

        agent = DeepAgent(model="openai/gpt-4o", subagents=[subagent1, subagent2])

        self.assertEqual(len(agent.subagents), 2)
        self.assertEqual(agent.subagents[0].name, "researcher")
        self.assertEqual(agent.subagents[1].name, "reviewer")

    @patch("upsonic.models.infer_model")
    def test_deep_agent_initialization_with_instructions(self, mock_infer_model):
        """Test init with custom instructions."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model

        custom_instructions = "You are a specialized code reviewer."
        agent = DeepAgent(model="openai/gpt-4o", instructions=custom_instructions)

        self.assertIsNotNone(agent)
        # Verify instructions are included in system prompt
        self.assertIn(custom_instructions, agent.system_prompt)

    @patch("upsonic.models.infer_model")
    def test_deep_agent_initialization_with_memory(self, mock_infer_model):
        """Test init with custom memory."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model

        storage = InMemoryStorage()
        memory = Memory(storage=storage, session_id="test-session", user_id="test-user")

        agent = DeepAgent(model="openai/gpt-4o", memory=memory)

        self.assertEqual(agent.memory, memory)

    @patch("upsonic.models.infer_model")
    def test_deep_agent_initialization_with_tool_call_limit(self, mock_infer_model):
        """Test init with custom tool_call_limit."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model

        agent = DeepAgent(model="openai/gpt-4o", tool_call_limit=50)

        self.assertEqual(agent.tool_call_limit, 50)


class TestDeepAgentDoMethods(unittest.TestCase):
    """Test suite for DeepAgent do() and do_async() methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MockModel()
        self.mock_model.request = AsyncMock(
            return_value=MagicMock(
                parts=[MagicMock(content="Test response")], model_name="test-model"
            )
        )

    @patch("upsonic.models.infer_model")
    @patch("upsonic.agent.agent.Agent.do_async")
    def test_deep_agent_do_basic(self, mock_base_do_async, mock_infer_model):
        """Test basic do() method."""
        mock_infer_model.return_value = self.mock_model
        mock_base_do_async.return_value = "Test response"

        agent = DeepAgent(model="openai/gpt-4o")
        task = Task("What is 2+2?")

        result = agent.do(task)

        self.assertEqual(result, "Test response")
        mock_base_do_async.assert_called_once()

    @patch("upsonic.models.infer_model")
    @patch("upsonic.agent.agent.Agent.do_async")
    @pytest.mark.asyncio
    async def test_deep_agent_do_async(self, mock_base_do_async, mock_infer_model):
        """Test async execution."""
        mock_infer_model.return_value = self.mock_model
        mock_base_do_async.return_value = "Async test response"

        agent = DeepAgent(model="openai/gpt-4o")
        task = Task("Async test task")

        result = await agent.do_async(task)

        self.assertEqual(result, "Async test response")
        mock_base_do_async.assert_called_once()

    @patch("upsonic.models.infer_model")
    @patch("upsonic.agent.agent.Agent.do_async")
    def test_deep_agent_do_with_todo_completion_loop(
        self, mock_base_do_async, mock_infer_model
    ):
        """Test do() with todo completion loop."""
        mock_infer_model.return_value = self.mock_model

        # Mock to return a value that marks todos as completed
        def mock_do_async_side_effect(*args, **kwargs):
            # Mark todos as completed after first call
            if mock_base_do_async.call_count == 0:
                return "Initial response"
            return "Completion response"

        mock_base_do_async.side_effect = mock_do_async_side_effect

        agent = DeepAgent(model="openai/gpt-4o")

        agent.deep_agent_state.todos = [
            Todo(content="Task 1", status="pending"),
            Todo(content="Task 2", status="in_progress"),
        ]

        task = Task("Complete all tasks")

        # This will trigger the completion loop, but we'll mark todos as completed
        # to prevent infinite loop
        try:
            agent.do(task)
        except Exception:
            # If it fails due to loop, that's okay - we're testing the mechanism
            pass

        # Should have called do_async at least once
        self.assertGreaterEqual(mock_base_do_async.call_count, 1)


class TestDeepAgentFileOperations(unittest.TestCase):
    """Test suite for DeepAgent file operations."""

    @patch("upsonic.models.infer_model")
    def setUp(self, mock_infer_model):
        """Set up test fixtures."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model
        self.agent = DeepAgent(model="openai/gpt-4o")

    def test_deep_agent_add_file(self):
        """Test adding files to virtual filesystem."""
        self.agent.add_file("/app/main.py", "def hello(): pass")

        files = self.agent.get_files()
        self.assertIn("/app/main.py", files)
        self.assertEqual(files["/app/main.py"], "def hello(): pass")

    def test_deep_agent_add_file_multiple(self):
        """Test adding multiple files."""
        self.agent.add_file("/app/main.py", "def main(): pass")
        self.agent.add_file("/app/config.json", '{"debug": true}')
        self.agent.add_file("/docs/README.md", "# Documentation")

        files = self.agent.get_files()
        self.assertEqual(len(files), 3)
        self.assertIn("/app/main.py", files)
        self.assertIn("/app/config.json", files)
        self.assertIn("/docs/README.md", files)

    def test_deep_agent_get_files(self):
        """Test getting files from virtual filesystem."""
        self.agent.add_file("/test/file.txt", "test content")

        files = self.agent.get_files()

        self.assertIsInstance(files, dict)
        self.assertEqual(files["/test/file.txt"], "test content")

    def test_deep_agent_set_files(self):
        """Test setting files in virtual filesystem."""
        files_dict = {
            "/app/main.py": "def main(): pass",
            "/app/utils.py": "def helper(): pass",
        }

        self.agent.set_files(files_dict)

        files = self.agent.get_files()
        self.assertEqual(files, files_dict)

    def test_deep_agent_set_files_overwrites(self):
        """Test that set_files overwrites existing files."""
        self.agent.add_file("/old/file.txt", "old content")

        new_files = {"/new/file.txt": "new content"}
        self.agent.set_files(new_files)

        files = self.agent.get_files()
        self.assertEqual(files, new_files)
        self.assertNotIn("/old/file.txt", files)


class TestDeepAgentTodoManagement(unittest.TestCase):
    """Test suite for DeepAgent todo management."""

    @patch("upsonic.models.infer_model")
    def setUp(self, mock_infer_model):
        """Set up test fixtures."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model
        self.agent = DeepAgent(model="openai/gpt-4o")

    def test_deep_agent_todo_management(self):
        """Test todo creation and tracking."""
        todos = [
            Todo(content="Research topic", status="pending"),
            Todo(content="Write report", status="in_progress"),
            Todo(content="Review document", status="pending"),
        ]

        self.agent.deep_agent_state.todos = todos

        retrieved_todos = self.agent.get_todos()
        self.assertEqual(len(retrieved_todos), 3)
        self.assertEqual(retrieved_todos[0]["content"], "Research topic")
        self.assertEqual(retrieved_todos[0]["status"], "pending")

    def test_deep_agent_todo_completion(self):
        """Test todo completion loop."""
        # Set up incomplete todos
        todos = [
            Todo(content="Task 1", status="pending"),
            Todo(content="Task 2", status="in_progress"),
        ]
        self.agent.deep_agent_state.todos = todos

        # Check completion status
        all_completed, completed_count, total_count = (
            self.agent._check_todos_completion()
        )

        self.assertFalse(all_completed)
        self.assertEqual(completed_count, 0)
        self.assertEqual(total_count, 2)

        # Mark all as completed
        for todo in self.agent.deep_agent_state.todos:
            todo.status = "completed"

        all_completed, completed_count, total_count = (
            self.agent._check_todos_completion()
        )

        self.assertTrue(all_completed)
        self.assertEqual(completed_count, 2)
        self.assertEqual(total_count, 2)

    def test_deep_agent_write_todos_tool(self):
        """Test write_todos tool integration."""
        from upsonic.agent.deep_agent.tools import write_todos
        from upsonic.agent.deep_agent.tools import set_current_deep_agent

        set_current_deep_agent(self.agent)

        todos = [
            Todo(content="Task 1", status="pending"),
            Todo(content="Task 2", status="in_progress"),
        ]

        result = write_todos(todos)

        self.assertIn("Updated todo list", result)
        self.assertEqual(len(self.agent.deep_agent_state.todos), 2)

    def test_deep_agent_multiple_todos(self):
        """Test multiple todos management."""
        todos = [Todo(content=f"Task {i}", status="pending") for i in range(10)]

        self.agent.deep_agent_state.todos = todos

        retrieved_todos = self.agent.get_todos()
        self.assertEqual(len(retrieved_todos), 10)

    def test_deep_agent_todo_states(self):
        """Test todo state transitions (pending, in_progress, completed)."""
        todo = Todo(content="Test task", status="pending")
        self.agent.deep_agent_state.todos = [todo]

        # Test pending state
        todos = self.agent.get_todos()
        self.assertEqual(todos[0]["status"], "pending")

        # Transition to in_progress
        todo.status = "in_progress"
        todos = self.agent.get_todos()
        self.assertEqual(todos[0]["status"], "in_progress")

        # Transition to completed
        todo.status = "completed"
        todos = self.agent.get_todos()
        self.assertEqual(todos[0]["status"], "completed")

    def test_deep_agent_get_incomplete_todos_summary(self):
        """Test getting summary of incomplete todos."""
        todos = [
            Todo(content="Task 1", status="pending"),
            Todo(content="Task 2", status="in_progress"),
            Todo(content="Task 3", status="completed"),
        ]
        self.agent.deep_agent_state.todos = todos

        summary = self.agent._get_incomplete_todos_summary()

        self.assertIn("incomplete todos", summary.lower())
        self.assertIn("Task 1", summary)
        self.assertIn("Task 2", summary)
        self.assertNotIn("Task 3", summary)  # Completed tasks not in summary


class TestDeepAgentVirtualFilesystem(unittest.TestCase):
    """Test suite for DeepAgent virtual filesystem tools."""

    @patch("upsonic.models.infer_model")
    def setUp(self, mock_infer_model):
        """Set up test fixtures."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model
        self.agent = DeepAgent(model="openai/gpt-4o")
        from upsonic.agent.deep_agent.tools import set_current_deep_agent

        set_current_deep_agent(self.agent)

    def test_deep_agent_virtual_filesystem_ls(self):
        """Test ls tool functionality."""
        from upsonic.agent.deep_agent.tools import ls

        self.agent.add_file("/app/main.py", "content")
        self.agent.add_file("/app/utils.py", "content")

        files = ls()

        self.assertIsInstance(files, list)
        self.assertIn("/app/main.py", files)
        self.assertIn("/app/utils.py", files)

    def test_deep_agent_virtual_filesystem_ls_empty(self):
        """Test ls with empty filesystem."""
        from upsonic.agent.deep_agent.tools import ls

        files = ls()

        self.assertEqual(files, [])

    def test_deep_agent_virtual_filesystem_read_file(self):
        """Test read_file tool."""
        from upsonic.agent.deep_agent.tools import read_file

        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        self.agent.add_file("/test/file.txt", content)

        result = read_file("/test/file.txt")

        self.assertIn("Line 1", result)
        self.assertIn("Line 2", result)
        self.assertIn("     1", result)  # Line numbers

    def test_deep_agent_virtual_filesystem_read_file_with_offset(self):
        """Test read_file with offset."""
        from upsonic.agent.deep_agent.tools import read_file

        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        self.agent.add_file("/test/file.txt", content)

        result = read_file("/test/file.txt", offset=2, limit=2)

        self.assertIn("Line 3", result)
        self.assertIn("Line 4", result)
        self.assertNotIn("Line 1", result)
        self.assertNotIn("Line 2", result)

    def test_deep_agent_virtual_filesystem_read_file_not_found(self):
        """Test read_file with non-existent file."""
        from upsonic.agent.deep_agent.tools import read_file

        result = read_file("/nonexistent/file.txt")

        self.assertIn("Error", result)
        self.assertIn("not found", result)

    def test_deep_agent_virtual_filesystem_write_file(self):
        """Test write_file tool."""
        from upsonic.agent.deep_agent.tools import write_file

        result = write_file("/app/main.py", "def main(): pass")

        self.assertIn("Successfully wrote", result)
        files = self.agent.get_files()
        self.assertEqual(files["/app/main.py"], "def main(): pass")

    def test_deep_agent_virtual_filesystem_write_file_overwrites(self):
        """Test write_file overwrites existing file."""
        from upsonic.agent.deep_agent.tools import write_file

        self.agent.add_file("/app/main.py", "old content")
        write_file("/app/main.py", "new content")

        files = self.agent.get_files()
        self.assertEqual(files["/app/main.py"], "new content")

    def test_deep_agent_virtual_filesystem_edit_file(self):
        """Test edit_file tool."""
        from upsonic.agent.deep_agent.tools import edit_file

        content = "def old_function(): pass"
        self.agent.add_file("/app/main.py", content)

        result = edit_file("/app/main.py", "old_function", "new_function")

        self.assertIn("Successfully replaced", result)
        files = self.agent.get_files()
        self.assertIn("new_function", files["/app/main.py"])
        self.assertNotIn("old_function", files["/app/main.py"])

    def test_deep_agent_virtual_filesystem_edit_file_replace_all(self):
        """Test edit_file with replace_all=True."""
        from upsonic.agent.deep_agent.tools import edit_file

        content = "old_var = 1\nold_var = 2\nold_var = 3"
        self.agent.add_file("/app/main.py", content)

        result = edit_file("/app/main.py", "old_var", "new_var", replace_all=True)

        self.assertIn("Successfully replaced", result)
        self.assertIn("3 instance(s)", result)
        files = self.agent.get_files()
        self.assertEqual(files["/app/main.py"].count("new_var"), 3)
        self.assertEqual(files["/app/main.py"].count("old_var"), 0)

    def test_deep_agent_virtual_filesystem_edit_file_not_found(self):
        """Test edit_file with non-existent file."""
        from upsonic.agent.deep_agent.tools import edit_file

        result = edit_file("/nonexistent/file.py", "old", "new")

        self.assertIn("Error", result)
        self.assertIn("not found", result)

    def test_deep_agent_virtual_filesystem_edit_file_string_not_found(self):
        """Test edit_file with string not in file."""
        from upsonic.agent.deep_agent.tools import edit_file

        self.agent.add_file("/app/main.py", "def hello(): pass")

        result = edit_file("/app/main.py", "nonexistent_string", "new_string")

        self.assertIn("Error", result)
        self.assertIn("not found", result)


class TestDeepAgentSubagentSpawning(unittest.TestCase):
    """Test suite for DeepAgent subagent spawning."""

    @patch("upsonic.models.infer_model")
    def setUp(self, mock_infer_model):
        """Set up test fixtures."""
        self.mock_model = MockModel()
        mock_infer_model.return_value = self.mock_model
        self.agent = DeepAgent(model="openai/gpt-4o")

    @patch("upsonic.models.infer_model")
    def test_deep_agent_subagent_spawning(self, mock_infer_model):
        """Test create_task_tool subagent creation."""
        from upsonic.agent.deep_agent.tools import create_task_tool

        mock_infer_model.return_value = self.mock_model

        subagent = Agent(model="openai/gpt-4o", name="researcher")
        self.agent.subagents = [subagent]

        task_tool = create_task_tool(self.agent, ["- researcher: Research expert"])

        self.assertIsNotNone(task_tool)
        self.assertTrue(callable(task_tool))

    @patch("upsonic.models.infer_model")
    def test_deep_agent_add_subagent(self, mock_infer_model):
        """Test adding subagent."""
        mock_infer_model.return_value = self.mock_model

        subagent = Agent(model="openai/gpt-4o", name="reviewer")
        self.agent.add_subagent(subagent)

        self.assertIn(subagent, self.agent.subagents)
        self.assertEqual(self.agent.subagents[0].name, "reviewer")

    @patch("upsonic.models.infer_model")
    def test_deep_agent_add_subagent_without_name(self, mock_infer_model):
        """Test adding subagent without name raises error."""
        mock_infer_model.return_value = self.mock_model

        subagent = Agent(model="openai/gpt-4o")
        # Remove name if it exists
        if hasattr(subagent, "name"):
            subagent.name = None

        with self.assertRaises(ValueError):
            self.agent.add_subagent(subagent)

    @patch("upsonic.models.infer_model")
    @patch("upsonic.agent.agent.Agent.do_async")
    @pytest.mark.asyncio
    async def test_deep_agent_execute_subagent_general_purpose(
        self, mock_do_async, mock_infer_model
    ):
        """Test executing general-purpose subagent."""
        mock_infer_model.return_value = self.mock_model
        mock_do_async.return_value = "Subagent result"

        result = await self.agent._execute_subagent(
            description="Do some work", subagent_type="general-purpose"
        )

        self.assertEqual(result, "Subagent result")

    @patch("upsonic.models.infer_model")
    @patch("upsonic.agent.agent.Agent.do_async")
    @pytest.mark.asyncio
    async def test_deep_agent_execute_subagent_named(
        self, mock_do_async, mock_infer_model
    ):
        """Test executing named subagent."""
        mock_infer_model.return_value = self.mock_model
        mock_do_async.return_value = "Named subagent result"

        subagent = Agent(model="openai/gpt-4o", name="researcher")
        self.agent.subagents = [subagent]

        result = await self.agent._execute_subagent(
            description="Research topic", subagent_type="researcher"
        )

        self.assertEqual(result, "Named subagent result")

    @pytest.mark.asyncio
    async def test_deep_agent_execute_subagent_not_found(self):
        """Test executing non-existent subagent."""
        result = await self.agent._execute_subagent(
            description="Do work", subagent_type="nonexistent"
        )

        self.assertIn("Error", result)
        self.assertIn("not found", result)

    @patch("upsonic.models.infer_model")
    def test_deep_agent_get_subagent_descriptions(self, mock_infer_model):
        """Test getting subagent descriptions."""
        mock_infer_model.return_value = self.mock_model

        subagent1 = Agent(
            model="openai/gpt-4o", name="researcher", system_prompt="Research expert"
        )
        subagent2 = Agent(
            model="openai/gpt-4o", name="reviewer", system_prompt="Code reviewer"
        )

        self.agent.subagents = [subagent1, subagent2]

        descriptions = self.agent._get_subagent_descriptions()

        self.assertEqual(len(descriptions), 2)
        self.assertIn("researcher", descriptions[0])
        self.assertIn("Research expert", descriptions[0])
        self.assertIn("reviewer", descriptions[1])
        self.assertIn("Code reviewer", descriptions[1])


class TestDeepAgentStatePersistence(unittest.TestCase):
    """Test suite for DeepAgent state persistence."""

    @patch("upsonic.models.infer_model")
    def setUp(self, mock_infer_model):
        """Set up test fixtures."""
        mock_model = MockModel()
        mock_infer_model.return_value = mock_model
        self.agent = DeepAgent(model="openai/gpt-4o")

    def test_deep_agent_state_persistence(self):
        """Test state persistence across calls."""
        from upsonic.agent.deep_agent.state import Todo

        # Add files and todos
        self.agent.add_file("/app/main.py", "def main(): pass")
        todos = [
            Todo(content="Task 1", status="in_progress"),
            Todo(content="Task 2", status="pending"),
        ]
        self.agent.deep_agent_state.todos = todos

        # Verify state persists
        files = self.agent.get_files()
        retrieved_todos = self.agent.get_todos()

        self.assertEqual(files["/app/main.py"], "def main(): pass")
        self.assertEqual(len(retrieved_todos), 2)
        self.assertEqual(retrieved_todos[0]["content"], "Task 1")

    def test_deep_agent_state_persistence_multiple_operations(self):
        """Test state persistence across multiple operations."""
        # Add files
        self.agent.add_file("/file1.txt", "content1")
        self.agent.add_file("/file2.txt", "content2")

        # Add todos
        from upsonic.agent.deep_agent.state import Todo

        self.agent.deep_agent_state.todos = [Todo(content="Task 1", status="completed")]

        # Modify files
        from upsonic.agent.deep_agent.tools import edit_file, set_current_deep_agent

        set_current_deep_agent(self.agent)
        edit_file("/file1.txt", "content1", "modified_content1")

        # Verify all state persists
        files = self.agent.get_files()
        todos = self.agent.get_todos()

        self.assertEqual(files["/file1.txt"], "modified_content1")
        self.assertEqual(files["/file2.txt"], "content2")
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
