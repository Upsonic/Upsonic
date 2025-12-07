"""
Test 28: Chat class testing
Success criteria: Chat methods work properly! We check attributes and results for that
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout

from upsonic import Agent, Chat, Task

pytestmark = pytest.mark.timeout(120)


@pytest.mark.asyncio
async def test_chat_basic_invoke():
    """Test basic Chat invoke method."""
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_1",
        user_id="test_user_1",
        agent=agent,
        debug=True
    )
    
    output_buffer = StringIO()
    try:
        with redirect_stdout(output_buffer):
            response = await chat.invoke("What is 2 + 2?")
        
        output = output_buffer.getvalue()
        
        # Verify response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert "4" in response, "Response should contain the answer"
        
        # Verify chat attributes
        assert len(chat.all_messages) >= 2, "Should have at least 2 messages (user + assistant)"
        assert chat.total_cost >= 0, "Total cost should be non-negative"
        assert chat.input_tokens > 0, "Should have input tokens"
        assert chat.output_tokens > 0, "Should have output tokens"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_chat_streaming():
    """Test Chat streaming invoke."""
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_2",
        user_id="test_user_2",
        agent=agent,
        debug=True
    )
    
    output_buffer = StringIO()
    accumulated_text = ""
    
    try:
        with redirect_stdout(output_buffer):
            # invoke with stream=True returns an async generator - need to await it
            stream_generator = await chat.invoke("Count from 1 to 3, one number per line.", stream=True)
            async for chunk in stream_generator:
                accumulated_text += chunk
                assert isinstance(chunk, str), "Stream chunks should be strings"
        
        output = output_buffer.getvalue()
        
        # Verify streaming worked
        assert accumulated_text is not None, "Should have accumulated text"
        assert len(accumulated_text) > 0, "Should have received text chunks"
        
        # Verify chat attributes after streaming
        assert len(chat.all_messages) >= 2, "Should have messages after streaming"
        assert chat.total_cost >= 0, "Total cost should be tracked"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_chat_conversation_history():
    """Test Chat conversation history management."""
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_3",
        user_id="test_user_3",
        agent=agent,
        debug=True
    )
    
    try:
        # First message
        response1 = await chat.invoke("My name is Alice.")
        assert len(chat.all_messages) >= 2, "Should have 2 messages after first invoke"
        
        # Second message (should remember context)
        response2 = await chat.invoke("What is my name?")
        assert len(chat.all_messages) >= 4, "Should have 4 messages after second invoke"
        assert "alice" in response2.lower(), "Should remember the name from previous message"
        
        # Verify message history
        all_messages = chat.all_messages
        assert len(all_messages) >= 4, "Should have at least 4 messages"
        
        # Check message roles
        user_messages = [msg for msg in all_messages if msg.role == "user"]
        assistant_messages = [msg for msg in all_messages if msg.role == "assistant"]
        assert len(user_messages) >= 2, "Should have at least 2 user messages"
        assert len(assistant_messages) >= 2, "Should have at least 2 assistant messages"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_chat_attributes():
    """Test Chat class attributes."""
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_4",
        user_id="test_user_4",
        agent=agent,
        debug=True
    )
    
    try:
        # Before any invocation
        assert chat.session_id == "test_session_4", "session_id should be set"
        assert chat.user_id == "test_user_4", "user_id should be set"
        assert chat.agent == agent, "agent should be set"
        assert len(chat.all_messages) == 0, "Should start with no messages"
        assert chat.total_cost == 0.0, "Should start with zero cost"
        assert chat.input_tokens == 0, "Should start with zero input tokens"
        assert chat.output_tokens == 0, "Should start with zero output tokens"
        
        # After invocation
        await chat.invoke("Hello")
        
        assert len(chat.all_messages) > 0, "Should have messages after invoke"
        assert chat.total_cost >= 0, "Total cost should be tracked"
        assert chat.input_tokens > 0, "Should have input tokens"
        assert chat.output_tokens > 0, "Should have output tokens"
        assert chat.session_duration >= 0, "Session duration should be tracked"
        assert chat.last_activity >= 0, "Last activity should be tracked"
        
        # Test get_recent_messages
        recent = chat.get_recent_messages(count=5)
        assert isinstance(recent, list), "get_recent_messages should return a list"
        assert len(recent) <= 5, "Should return at most 5 messages"
        
        # Test get_session_metrics
        metrics = chat.get_session_metrics()
        assert metrics is not None, "Session metrics should not be None"
        assert hasattr(metrics, 'message_count'), "Metrics should have message_count"
        assert hasattr(metrics, 'total_cost'), "Metrics should have total_cost"
        
        # Test get_session_summary
        summary = chat.get_session_summary()
        assert isinstance(summary, str), "Session summary should be a string"
        assert len(summary) > 0, "Session summary should not be empty"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_chat_with_task():
    """Test Chat with Task object instead of string."""
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_5",
        user_id="test_user_5",
        agent=agent,
        debug=True
    )
    
    try:
        task = Task(description="What is the capital of France?")
        response = await chat.invoke(task)
        
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert "paris" in response.lower(), "Response should mention Paris"
        
        assert len(chat.all_messages) >= 2, "Should have messages after invoke"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_chat_cost_tracking():
    """Test Chat cost tracking."""
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_6",
        user_id="test_user_6",
        agent=agent,
        debug=True
    )
    
    try:
        initial_cost = chat.total_cost
        assert initial_cost == 0.0, "Should start with zero cost"
        
        # Make a call
        await chat.invoke("Hello")
        
        cost_after_one = chat.total_cost
        assert cost_after_one > initial_cost, "Cost should increase after invoke"
        
        # Make another call
        await chat.invoke("What is 1 + 1?")
        
        cost_after_two = chat.total_cost
        assert cost_after_two > cost_after_one, "Cost should increase after second invoke"
        
        # Test get_cost_history
        cost_history = chat.get_cost_history()
        assert isinstance(cost_history, list), "Cost history should be a list"
        assert len(cost_history) > 0, "Should have cost history entries"
        
    finally:
        pass  # Agent cleanup handled automatically

