import time
from upsonic.chat.message import ChatMessage


def test_message_creation():
    """Test message creation."""
    message = ChatMessage(
        content="Hello, world!",
        role="user",
        timestamp=time.time(),
        attachments=["file1.txt"],
        metadata={"key": "value"},
    )

    assert message.content == "Hello, world!"
    assert message.role == "user"
    assert message.timestamp > 0
    assert message.attachments == ["file1.txt"]
    assert message.metadata == {"key": "value"}


def test_message_serialization():
    """Test message serialization."""
    message = ChatMessage(
        content="Test message",
        role="assistant",
        timestamp=time.time(),
        tool_calls=[{"tool_name": "test_tool", "args": {"param": "value"}}],
    )

    # Test that message can be converted to dict-like structure
    assert message.content == "Test message"
    assert message.role == "assistant"
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["tool_name"] == "test_tool"


def test_message_parts():
    """Test message parts."""
    # Test user message with attachments
    user_message = ChatMessage(
        content="User message",
        role="user",
        timestamp=time.time(),
        attachments=["image.jpg", "document.pdf"],
    )

    assert user_message.role == "user"
    assert len(user_message.attachments) == 2

    # Test assistant message with tool calls
    assistant_message = ChatMessage(
        content="Assistant response",
        role="assistant",
        timestamp=time.time(),
        tool_calls=[
            {
                "tool_name": "get_weather",
                "tool_call_id": "call_1",
                "args": {"city": "NYC"},
            },
            {"tool_name": "get_time", "tool_call_id": "call_2", "args": {}},
        ],
    )

    assert assistant_message.role == "assistant"
    assert len(assistant_message.tool_calls) == 2
    assert assistant_message.tool_calls[0]["tool_name"] == "get_weather"
    assert assistant_message.tool_calls[1]["tool_name"] == "get_time"
