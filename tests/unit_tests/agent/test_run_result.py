import pytest
from unittest.mock import Mock
from dataclasses import asdict, fields

from upsonic.agent.run_result import RunResult, AgentRunResult
from upsonic.messages.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    SystemPromptPart,
)


class TestAgentRunResultCreation:
    """Test suite for AgentRunResult creation and initialization."""

    def test_agent_run_result_creation(self):
        """Test AgentRunResult creation with basic output."""
        # Test with string output
        output = "Hello, world!"
        result = RunResult(output=output)

        assert result.output == output
        assert result._all_messages == []
        assert result._run_boundaries == []

        # Test with None output
        result_none = RunResult(output=None)
        assert result_none.output is None
        assert result_none._all_messages == []
        assert result_none._run_boundaries == []

        # Test with integer output
        result_int = RunResult(output=42)
        assert result_int.output == 42

        # Test with dict output
        result_dict = RunResult(output={"key": "value"})
        assert result_dict.output == {"key": "value"}

    def test_agent_run_result_creation_with_messages(self):
        """Test AgentRunResult creation with initial messages."""
        # Create mock messages matching real structure (SystemPromptPart + UserPromptPart)
        mock_request = ModelRequest(
            parts=[
                SystemPromptPart(content="You are a helpful agent."),
                UserPromptPart(content="Test prompt"),
            ],
            instructions=None,
        )
        mock_response = ModelResponse(
            parts=[TextPart(content="Test response")],
            model_name="test-model",
            usage=Mock(),
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )

        # Create result with messages added after creation
        result = RunResult(output="test output")
        result.add_message(mock_request)
        result.add_message(mock_response)

        assert len(result._all_messages) == 2
        assert result._all_messages[0] == mock_request
        assert result._all_messages[1] == mock_response
        # Verify request has both system and user parts
        assert len(result._all_messages[0].parts) == 2
        assert isinstance(result._all_messages[0].parts[0], SystemPromptPart)
        assert isinstance(result._all_messages[0].parts[1], UserPromptPart)

    def test_agent_run_result_type_alias(self):
        """Test that AgentRunResult is properly aliased to RunResult."""
        output = "test"
        result1 = RunResult(output=output)
        result2 = AgentRunResult(output=output)

        # Both should be instances of RunResult
        assert isinstance(result1, RunResult)
        assert isinstance(result2, RunResult)
        assert type(result1) is type(result2)


class TestAgentRunResultProperties:
    """Test suite for AgentRunResult properties and methods."""

    @pytest.fixture
    def run_result(self):
        """Create a basic RunResult instance for testing."""
        return RunResult(output="test output")

    @pytest.fixture
    def mock_messages(self):
        """Create mock messages for testing matching real structure."""
        request1 = ModelRequest(
            parts=[
                SystemPromptPart(content="You are a helpful agent."),
                UserPromptPart(content="First prompt"),
            ],
            instructions=None,
        )
        response1 = ModelResponse(
            parts=[TextPart(content="First response")],
            model_name="test-model",
            usage=Mock(),
            provider_name="test-provider",
            provider_response_id="test-id-1",
            provider_details={},
            finish_reason="stop",
        )
        request2 = ModelRequest(
            parts=[
                SystemPromptPart(content="You are a helpful agent."),
                UserPromptPart(content="Second prompt"),
            ],
            instructions=None,
        )
        response2 = ModelResponse(
            parts=[TextPart(content="Second response")],
            model_name="test-model",
            usage=Mock(),
            provider_name="test-provider",
            provider_response_id="test-id-2",
            provider_details={},
            finish_reason="stop",
        )
        return [request1, response1, request2, response2]

    def test_agent_run_result_properties(self, run_result):
        """Test all properties of AgentRunResult."""
        # Test output property
        assert run_result.output == "test output"

        # Test _all_messages property (should be empty initially)
        assert run_result._all_messages == []
        assert isinstance(run_result._all_messages, list)

        # Test _run_boundaries property (should be empty initially)
        assert run_result._run_boundaries == []
        assert isinstance(run_result._run_boundaries, list)

    def test_all_messages_method(self, run_result, mock_messages):
        """Test the all_messages() method."""
        # Initially should return empty list
        assert run_result.all_messages() == []

        # Add messages
        run_result.add_messages(mock_messages)

        # Should return all messages
        all_msgs = run_result.all_messages()
        assert len(all_msgs) == 4
        assert all_msgs == mock_messages

        # Should return a copy, not the original list
        assert all_msgs is not run_result._all_messages
        all_msgs.append("should not affect original")
        assert len(run_result._all_messages) == 4

    def test_new_messages_method(self, run_result, mock_messages):
        """Test the new_messages() method."""
        # Initially should return empty list
        assert run_result.new_messages() == []

        # Start first run (boundary at 0) and add messages
        run_result.start_new_run()  # First run starts at index 0
        run_result.add_messages(mock_messages[:2])  # First request + response
        new_msgs = run_result.new_messages()
        assert len(new_msgs) == 2
        assert new_msgs == mock_messages[:2]

        # Mark a new run and add more messages (second run)
        run_result.start_new_run()  # Second run starts at index 2
        run_result.add_messages(mock_messages[2:])  # Second request + response

        # new_messages() should return only the last run's messages
        new_msgs = run_result.new_messages()
        assert len(new_msgs) == 2
        assert new_msgs[0] == mock_messages[2]
        assert new_msgs[1] == mock_messages[3]

        # all_messages() should still return all messages
        assert len(run_result.all_messages()) == 4
        # Verify run boundaries are set correctly (like real usage: [0, 2])
        assert run_result._run_boundaries == [0, 2]

    def test_add_messages_method(self, run_result, mock_messages):
        """Test the add_messages() method."""
        # Add multiple messages at once
        run_result.add_messages(mock_messages)
        assert len(run_result._all_messages) == 4
        assert run_result._all_messages == mock_messages

        # Add more messages
        additional = [
            ModelRequest(
                parts=[UserPromptPart(content="Additional")], instructions=None
            )
        ]
        run_result.add_messages(additional)
        assert len(run_result._all_messages) == 5

    def test_add_message_method(self, run_result):
        """Test the add_message() method."""
        # Add single message (matching real structure)
        message = ModelRequest(
            parts=[
                SystemPromptPart(content="You are a helpful agent."),
                UserPromptPart(content="Single message"),
            ],
            instructions=None,
        )
        run_result.add_message(message)
        assert len(run_result._all_messages) == 1
        assert run_result._all_messages[0] == message

        # Add another message
        message2 = ModelResponse(
            parts=[TextPart(content="Response")],
            model_name="test-model",
            usage=Mock(),
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )
        run_result.add_message(message2)
        assert len(run_result._all_messages) == 2

    def test_start_new_run_method(self, run_result, mock_messages):
        """Test the start_new_run() method."""
        # Add some messages
        run_result.add_messages(mock_messages[:2])
        assert len(run_result._all_messages) == 2
        assert len(run_result._run_boundaries) == 0

        # Start a new run
        run_result.start_new_run()
        assert len(run_result._run_boundaries) == 1
        assert run_result._run_boundaries[0] == 2

        # Add more messages
        run_result.add_messages(mock_messages[2:])
        assert len(run_result._all_messages) == 4

        # Start another run
        run_result.start_new_run()
        assert len(run_result._run_boundaries) == 2
        assert run_result._run_boundaries[1] == 4

    def test_str_method(self, run_result):
        """Test the __str__() method."""
        # Test with string output
        assert str(run_result) == "test output"

        # Test with None output
        result_none = RunResult(output=None)
        assert str(result_none) == "None"

        # Test with integer output
        result_int = RunResult(output=42)
        assert str(result_int) == "42"

    def test_repr_method(self, run_result, mock_messages):
        """Test the __repr__() method."""
        # Test with no messages
        repr_str = repr(run_result)
        assert "RunResult" in repr_str
        assert "test output" in repr_str
        assert "messages_count=0" in repr_str

        # Test with messages
        run_result.add_messages(mock_messages)
        repr_str = repr(run_result)
        assert "messages_count=4" in repr_str


class TestAgentRunResultSerialization:
    """Test suite for AgentRunResult serialization."""

    def test_agent_run_result_serialization(self):
        """Test serialization of AgentRunResult using dataclasses.asdict()."""
        # Create a result with output (typically a string in real usage)
        output = "The capital of France is Paris."
        result = RunResult(output=output)

        # Add some messages (matching real structure)
        request = ModelRequest(
            parts=[
                SystemPromptPart(content="You are a helpful agent."),
                UserPromptPart(content="Test"),
            ],
            instructions=None,
        )
        response = ModelResponse(
            parts=[TextPart(content="Response")],
            model_name="test-model",
            usage=Mock(),
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop",
        )
        result.add_message(request)
        result.add_message(response)
        result.start_new_run()

        # Serialize using dataclasses.asdict()
        serialized = asdict(result)

        # Verify structure
        assert isinstance(serialized, dict)
        assert "output" in serialized
        assert "_all_messages" in serialized
        assert "_run_boundaries" in serialized

        # Verify output is preserved
        assert serialized["output"] == output

        # Verify run boundaries are preserved
        assert serialized["_run_boundaries"] == [2]

        # Messages are complex objects, verify they're in the dict
        assert len(serialized["_all_messages"]) == 2

    def test_serialization_with_empty_result(self):
        """Test serialization of empty RunResult."""
        result = RunResult(output=None)
        serialized = asdict(result)

        assert serialized["output"] is None
        assert serialized["_all_messages"] == []
        assert serialized["_run_boundaries"] == []

    def test_serialization_preserves_structure(self):
        """Test that serialization preserves the complete structure."""
        # Create a complex result
        output = {"status": "complete", "data": [1, 2, 3], "metadata": {"key": "value"}}
        result = RunResult(output=output)

        # Add multiple runs
        for i in range(3):
            request = ModelRequest(
                parts=[UserPromptPart(content=f"Request {i}")], instructions=None
            )
            result.add_message(request)
            result.start_new_run()

        # Serialize
        serialized = asdict(result)

        # Verify nested structure is preserved
        assert serialized["output"]["status"] == "complete"
        assert serialized["output"]["data"] == [1, 2, 3]
        assert serialized["output"]["metadata"]["key"] == "value"
        assert len(serialized["_run_boundaries"]) == 3

    def test_dataclass_fields(self):
        """Test that RunResult has the expected dataclass fields."""
        result = RunResult(output="test")
        field_names = [f.name for f in fields(result)]

        # Verify all expected fields exist
        assert "output" in field_names
        assert "_all_messages" in field_names
        assert "_run_boundaries" in field_names

        # Verify field types
        output_field = next(f for f in fields(result) if f.name == "output")
        messages_field = next(f for f in fields(result) if f.name == "_all_messages")
        boundaries_field = next(
            f for f in fields(result) if f.name == "_run_boundaries"
        )

        assert output_field.type is not None
        assert messages_field.type is not None
        assert boundaries_field.type is not None
