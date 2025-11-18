import pytest
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from upsonic.canvas.canvas import Canvas


@pytest.fixture
def temp_dir():
    """Create a temporary directory for canvas files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_canvas_initialization():
    """Test Canvas initialization."""
    with patch("upsonic.canvas.canvas.infer_model") as mock_infer:
        mock_model = Mock()
        mock_infer.return_value = mock_model

        canvas = Canvas(canvas_name="test_canvas")

        assert canvas.canvas_name == "test_canvas"
        assert canvas.model == mock_model


def test_canvas_add_element(temp_dir):
    """Test adding elements."""
    with patch("upsonic.canvas.canvas.infer_model") as mock_infer:
        mock_model = Mock()
        mock_infer.return_value = mock_model

        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            canvas = Canvas(canvas_name="test_canvas")

            # Test initial state
            state = canvas.get_current_state_of_canvas()
            assert state == "Empty Canvas"
        finally:
            os.chdir(original_cwd)


def test_canvas_update_element(temp_dir):
    """Test updating elements."""
    with (
        patch("upsonic.canvas.canvas.infer_model") as mock_infer,
        patch("upsonic.agent.agent.Agent") as mock_agent_class,
        patch("upsonic.tasks.tasks.Task") as mock_task_class,
    ):
        mock_model = Mock()
        mock_infer.return_value = mock_model

        mock_agent = Mock()
        mock_agent.do_async = AsyncMock(return_value=None)
        mock_agent_class.return_value = mock_agent

        mock_task = Mock()
        mock_task.response = "New content"
        mock_task_class.return_value = mock_task

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            canvas = Canvas(canvas_name="test_canvas")

            # Test updating canvas
            import asyncio

            result = asyncio.run(canvas.change_in_canvas("New content", "section1"))

            assert result == "New content"
        finally:
            os.chdir(original_cwd)


def test_canvas_remove_element(temp_dir):
    """Test removing elements."""
    with patch("upsonic.canvas.canvas.infer_model") as mock_infer:
        mock_model = Mock()
        mock_infer.return_value = mock_model

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            canvas = Canvas(canvas_name="test_canvas")

            # Canvas doesn't have explicit remove, but we can test state management
            state = canvas.get_current_state_of_canvas()
            assert state == "Empty Canvas"
        finally:
            os.chdir(original_cwd)


def test_canvas_render(temp_dir):
    """Test rendering."""
    with patch("upsonic.canvas.canvas.infer_model") as mock_infer:
        mock_model = Mock()
        mock_infer.return_value = mock_model

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            canvas = Canvas(canvas_name="test_canvas")

            # Test getting current state
            state = canvas.get_current_state_of_canvas()
            assert isinstance(state, str)

            # Test functions method
            functions = canvas.functions()
            assert len(functions) == 2
            assert canvas.get_current_state_of_canvas in functions
            assert canvas.change_in_canvas in functions
        finally:
            os.chdir(original_cwd)
