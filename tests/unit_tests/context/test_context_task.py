from unittest.mock import Mock
from upsonic.context.task import turn_task_to_string


def test_context_task_initialization():
    """Test context task."""
    mock_task = Mock()
    mock_task.task_id = "task-123"
    mock_task.description = "Test task description"
    mock_task.attachments = ["file1.txt", "file2.txt"]
    mock_task.response = "Test response"

    # Test conversion to string
    result = turn_task_to_string(mock_task)

    assert isinstance(result, str)
    assert "task-123" in result
    assert "Test task description" in result
    assert "Test response" in result


def test_context_task_build_context():
    """Test context building."""
    mock_task = Mock()
    mock_task.task_id = "task-456"
    mock_task.description = "Build context task"
    mock_task.attachments = ["context_file.txt"]
    mock_task.response = "Context response"

    # Convert to string
    context_string = turn_task_to_string(mock_task)

    # Verify all fields are present
    assert "task-456" in context_string
    assert "Build context task" in context_string
    assert "Context response" in context_string
