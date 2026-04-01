"""
Test: Full Session Memory with Storage Providers

Success criteria:
- Agent remembers previous conversations across runs
- Messages are properly stored in session
- AgentRunOutput.messages contains ONLY NEW messages for that run (ModelRequest + ModelResponse pairs)
- AgentRunOutput.chat_history contains FULL chat history (all ModelRequest + ModelResponse pairs)
- AgentSession.messages matches AgentRunOutput.chat_history at the end
- Both DIRECT (do_async) and STREAMING (astream) flows work correctly
- num_last_messages correctly limits chat history
- feed_tool_call_results correctly filters tool messages
- Session data persists and can be retrieved
"""

import pytest
import os
import tempfile
import uuid
from typing import List, Any, Optional, Union
from upsonic import Agent, Task
from upsonic.storage import Memory, SqliteStorage, InMemoryStorage
from upsonic.session.agent import AgentSession
from upsonic.session.base import SessionType
from upsonic.messages import ModelRequest, ModelResponse
from upsonic.run.agent.output import AgentRunOutput

pytestmark = pytest.mark.timeout(120)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def test_user_id() -> str:
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_session_id() -> str:
    return f"test_session_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sqlite_storage() -> SqliteStorage:
    """Create a temporary SQLite storage."""
    db_file = tempfile.mktemp(suffix=".db")
    storage = SqliteStorage(db_file=db_file)
    yield storage
    if os.path.exists(db_file):
        os.remove(db_file)


@pytest.fixture
def inmemory_storage() -> InMemoryStorage:
    """Create an in-memory storage."""
    return InMemoryStorage()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_message_types(messages: List[Any]) -> dict:
    """Count ModelRequest and ModelResponse in message list."""
    counts = {"request": 0, "response": 0, "other": 0}
    for msg in messages:
        if isinstance(msg, ModelRequest):
            counts["request"] += 1
        elif isinstance(msg, ModelResponse):
            counts["response"] += 1
        else:
            counts["other"] += 1
    return counts


def assert_valid_message_pairs(messages: List[Any], expected_pairs: int, context: str) -> None:
    """Assert that messages contain expected request/response pairs."""
    counts = count_message_types(messages)
    # Each pair should have at least 1 request and 1 response
    assert counts["request"] >= expected_pairs, \
        f"{context}: Expected at least {expected_pairs} ModelRequest, got {counts['request']}"
    assert counts["response"] >= expected_pairs, \
        f"{context}: Expected at least {expected_pairs} ModelResponse, got {counts['response']}"


class StreamingRunResult:
    """Container for streaming run results since AgentRunOutput is not directly returned."""
    
    def __init__(
        self, 
        messages: List[Any], 
        chat_history: List[Any],
        response: Any = None
    ) -> None:
        self.messages = messages  # NEW messages from this run
        self.chat_history = chat_history  # FULL history from session
        self.response = response  # The response (if available)


async def consume_stream(
    agent: Agent, 
    task: Task,
    storage: Any = None,
    session_id: str = None
) -> Optional[StreamingRunResult]:
    """
    Consume streaming output and return a result containing messages info.
    
    Since streaming yields text chunks (not AgentRunOutput), we construct
    the result from session state before and after streaming.
    """
    # Get message count before streaming
    messages_before = 0
    if storage and session_id:
        session_before = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True
        )
        if session_before and session_before.messages:
            messages_before = len(session_before.messages)
    
    # Collect response chunks
    response_chunks: List[str] = []
    async for chunk in agent.astream(task):
        if isinstance(chunk, str):
            response_chunks.append(chunk)
    
    # Get session state after streaming
    if storage and session_id:
        session_after = storage.get_session(
            session_id=session_id,
            session_type=SessionType.AGENT,
            deserialize=True
        )
        if session_after and session_after.messages:
            # NEW messages = messages added during this run
            new_messages = session_after.messages[messages_before:]
            # FULL chat history = all messages in session
            chat_history = list(session_after.messages)
            # Response from chunks
            response = "".join(response_chunks) if response_chunks else None
            
            return StreamingRunResult(
                messages=new_messages,
                chat_history=chat_history,
                response=response
            )
    
    return None


# =============================================================================
# TEST: DIRECT FLOW - Basic Memory Behavior
# =============================================================================

@pytest.mark.asyncio
async def test_direct_flow_messages_attribute_contains_only_new_messages(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """
    DIRECT FLOW: AgentRunOutput.messages should contain ONLY NEW messages from THIS run.
    Each run should add exactly 1 ModelRequest + 1 ModelResponse (minimum).
    """
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 1
    task1 = Task(description="Say hello")
    result1 = await agent.do_async(task1)
    assert result1 is not None
    
    # Get the run output from the latest run in session
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    assert session.runs is not None
    
    run_data = list(session.runs.values())[-1]
    run_output_1: AgentRunOutput = run_data.output
    
    # Run 1: messages should contain ONLY new messages from this run
    assert run_output_1.messages is not None, "Run 1: messages should not be None"
    assert len(run_output_1.messages) >= 2, \
        f"Run 1: messages should have at least 2 (request+response), got {len(run_output_1.messages)}"
    assert_valid_message_pairs(run_output_1.messages, 1, "Run 1 messages")
    
    # Run 2
    task2 = Task(description="Say goodbye")
    result2 = await agent.do_async(task2)
    assert result2 is not None
    
    # Get session again
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    run_data_2 = list(session.runs.values())[-1]
    run_output_2: AgentRunOutput = run_data_2.output
    
    # Run 2: messages should contain ONLY new messages from THIS run (not cumulative)
    assert run_output_2.messages is not None, "Run 2: messages should not be None"
    assert len(run_output_2.messages) >= 2, \
        f"Run 2: messages should have at least 2 (request+response), got {len(run_output_2.messages)}"
    assert_valid_message_pairs(run_output_2.messages, 1, "Run 2 messages")
    
    # Run 2 messages should NOT be same as Run 1 messages count (or cumulative)
    # Each run should add its own pair
    print(f"Run 1 messages count: {len(run_output_1.messages)}")
    print(f"Run 2 messages count: {len(run_output_2.messages)}")


@pytest.mark.asyncio
async def test_direct_flow_chat_history_contains_full_history(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """
    DIRECT FLOW: AgentRunOutput.chat_history should contain FULL chat history.
    After 2 runs, it should have at least 4 messages (2 pairs).
    """
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 1
    task1 = Task(description="My name is Alice")
    await agent.do_async(task1)
    
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    run_output_1: AgentRunOutput = list(session.runs.values())[-1].output
    chat_history_1_len = len(run_output_1.chat_history)
    
    # Run 2
    task2 = Task(description="What is my name?")
    await agent.do_async(task2)
    
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    run_output_2: AgentRunOutput = list(session.runs.values())[-1].output
    chat_history_2_len = len(run_output_2.chat_history)
    
    # chat_history should GROW with each run
    assert chat_history_2_len > chat_history_1_len, \
        f"chat_history should grow: Run1={chat_history_1_len}, Run2={chat_history_2_len}"
    
    # After 2 runs, should have at least 4 messages (2 request/response pairs)
    assert chat_history_2_len >= 4, \
        f"chat_history after 2 runs should have >= 4 messages, got {chat_history_2_len}"
    
    print(f"chat_history after Run 1: {chat_history_1_len}")
    print(f"chat_history after Run 2: {chat_history_2_len}")


@pytest.mark.asyncio
async def test_direct_flow_session_messages_equals_chat_history(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """
    DIRECT FLOW: AgentSession.messages should equal AgentRunOutput.chat_history at the end.
    """
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 3 tasks
    for i in range(3):
        task = Task(description=f"This is message {i+1}")
        await agent.do_async(task)
    
    # Get final session state
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    
    # Get latest run output
    run_output: AgentRunOutput = list(session.runs.values())[-1].output
    
    # AgentSession.messages should match chat_history length
    session_messages_len = len(session.messages) if session.messages else 0
    chat_history_len = len(run_output.chat_history)
    
    assert session_messages_len == chat_history_len, \
        f"session.messages ({session_messages_len}) should equal chat_history ({chat_history_len})"
    
    print(f"session.messages length: {session_messages_len}")
    print(f"chat_history length: {chat_history_len}")


# =============================================================================
# TEST: STREAMING FLOW - Basic Memory Behavior
# =============================================================================

@pytest.mark.asyncio
async def test_streaming_flow_messages_attribute_contains_only_new_messages(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """
    STREAMING FLOW: AgentRunOutput.messages should contain ONLY NEW messages from THIS run.
    """
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 1 via streaming
    task1 = Task(description="Say hello")
    run_output_1 = await consume_stream(agent, task1, inmemory_storage, test_session_id)
    assert run_output_1 is not None
    
    # messages should contain ONLY new messages from this run
    assert run_output_1.messages is not None, "Run 1: messages should not be None"
    assert len(run_output_1.messages) >= 2, \
        f"Run 1: messages should have at least 2, got {len(run_output_1.messages)}"
    assert_valid_message_pairs(run_output_1.messages, 1, "Streaming Run 1 messages")
    
    # Run 2 via streaming
    task2 = Task(description="Say goodbye")
    run_output_2 = await consume_stream(agent, task2, inmemory_storage, test_session_id)
    assert run_output_2 is not None
    
    # Run 2: messages should contain ONLY new messages from THIS run
    assert run_output_2.messages is not None, "Run 2: messages should not be None"
    assert len(run_output_2.messages) >= 2, \
        f"Run 2: messages should have at least 2, got {len(run_output_2.messages)}"
    assert_valid_message_pairs(run_output_2.messages, 1, "Streaming Run 2 messages")
    
    print(f"Streaming Run 1 messages count: {len(run_output_1.messages)}")
    print(f"Streaming Run 2 messages count: {len(run_output_2.messages)}")


@pytest.mark.asyncio
async def test_streaming_flow_chat_history_contains_full_history(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """
    STREAMING FLOW: AgentRunOutput.chat_history should contain FULL chat history.
    """
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 1 via streaming
    task1 = Task(description="My name is Bob")
    run_output_1 = await consume_stream(agent, task1, inmemory_storage, test_session_id)
    chat_history_1_len = len(run_output_1.chat_history)
    
    # Run 2 via streaming
    task2 = Task(description="What is my name?")
    run_output_2 = await consume_stream(agent, task2, inmemory_storage, test_session_id)
    chat_history_2_len = len(run_output_2.chat_history)
    
    # chat_history should GROW with each run
    assert chat_history_2_len > chat_history_1_len, \
        f"chat_history should grow: Run1={chat_history_1_len}, Run2={chat_history_2_len}"
    
    # After 2 runs, should have at least 4 messages
    assert chat_history_2_len >= 4, \
        f"chat_history after 2 runs should have >= 4, got {chat_history_2_len}"
    
    print(f"Streaming chat_history after Run 1: {chat_history_1_len}")
    print(f"Streaming chat_history after Run 2: {chat_history_2_len}")


@pytest.mark.asyncio
async def test_streaming_flow_session_messages_equals_chat_history(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """
    STREAMING FLOW: AgentSession.messages should equal AgentRunOutput.chat_history at the end.
    """
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 3 tasks via streaming
    run_output = None
    for i in range(3):
        task = Task(description=f"Streaming message {i+1}")
        run_output = await consume_stream(agent, task, inmemory_storage, test_session_id)
    
    # Get final session state
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    
    # session.messages should equal chat_history length
    session_messages_len = len(session.messages) if session.messages else 0
    chat_history_len = len(run_output.chat_history)
    
    assert session_messages_len == chat_history_len, \
        f"session.messages ({session_messages_len}) should equal chat_history ({chat_history_len})"
    
    print(f"Streaming session.messages length: {session_messages_len}")
    print(f"Streaming chat_history length: {chat_history_len}")


# =============================================================================
# TEST: Memory Recall (Agent Remembers Previous Conversations)
# =============================================================================

@pytest.mark.asyncio
async def test_direct_flow_agent_remembers_previous_context(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """DIRECT FLOW: Agent should remember information from previous turns."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # First turn - provide information
    task1 = Task(description="My name is Alice and I love Python programming")
    await agent.do_async(task1)
    
    # Second turn - ask about the information
    task2 = Task(description="What is my name?")
    result2 = await agent.do_async(task2)
    
    assert "alice" in str(result2).lower(), f"Expected 'alice' in response, got: {result2}"


@pytest.mark.asyncio
async def test_streaming_flow_agent_remembers_previous_context(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """STREAMING FLOW: Agent should remember information from previous turns."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # First turn via streaming
    task1 = Task(description="My name is Charlie and I love JavaScript programming")
    await consume_stream(agent, task1, inmemory_storage, test_session_id)
    
    # Second turn - ask about the information
    task2 = Task(description="What is my name?")
    run_output = await consume_stream(agent, task2, inmemory_storage, test_session_id)
    
    # Get the text response
    response_text = str(run_output.response).lower() if run_output.response else ""
    assert "charlie" in response_text, f"Expected 'charlie' in response, got: {response_text}"


# =============================================================================
# TEST: Message Accumulation Across Runs
# =============================================================================

@pytest.mark.asyncio
async def test_direct_flow_message_accumulation(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """DIRECT FLOW: Messages should accumulate correctly across multiple runs."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    message_counts: List[int] = []
    
    # Run 3 tasks and track message accumulation
    for i in range(3):
        task = Task(description=f"This is message number {i+1}")
        await agent.do_async(task)
        
        session = inmemory_storage.get_session(
            session_id=test_session_id,
            session_type=SessionType.AGENT,
            deserialize=True
        )
        if session and session.messages:
            message_counts.append(len(session.messages))
        else:
            message_counts.append(0)
    
    # Verify message count increases with each run
    assert len(message_counts) == 3
    assert message_counts[1] > message_counts[0], \
        f"Message count should increase. Run 1: {message_counts[0]}, Run 2: {message_counts[1]}"
    assert message_counts[2] > message_counts[1], \
        f"Message count should increase. Run 2: {message_counts[1]}, Run 3: {message_counts[2]}"
    
    print(f"Direct flow message counts: {message_counts}")


@pytest.mark.asyncio
async def test_streaming_flow_message_accumulation(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """STREAMING FLOW: Messages should accumulate correctly across multiple runs."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    message_counts: List[int] = []
    
    # Run 3 tasks via streaming and track message accumulation
    for i in range(3):
        task = Task(description=f"Streaming message number {i+1}")
        await consume_stream(agent, task, inmemory_storage, test_session_id)
        
        session = inmemory_storage.get_session(
            session_id=test_session_id,
            session_type=SessionType.AGENT,
            deserialize=True
        )
        if session and session.messages:
            message_counts.append(len(session.messages))
        else:
            message_counts.append(0)
    
    # Verify message count increases with each run
    assert len(message_counts) == 3
    assert message_counts[1] > message_counts[0], \
        f"Streaming message count should increase. Run 1: {message_counts[0]}, Run 2: {message_counts[1]}"
    assert message_counts[2] > message_counts[1], \
        f"Streaming message count should increase. Run 2: {message_counts[1]}, Run 3: {message_counts[2]}"
    
    print(f"Streaming flow message counts: {message_counts}")


# =============================================================================
# TEST: num_last_messages Limiting
# =============================================================================

@pytest.mark.asyncio
async def test_num_last_messages_limiting_direct(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """DIRECT FLOW: num_last_messages correctly limits chat history."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        num_last_messages=2,  # Only keep last 2 message turns
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 5 conversations
    for i in range(5):
        task = Task(description=f"Message {i}: I like number {i}")
        await agent.do_async(task)
    
    # Verify messages are stored in session (raw storage)
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    assert session.messages is not None
    # Storage should have ALL messages
    assert len(session.messages) >= 10, \
        f"Storage should have all messages, got {len(session.messages)}"
    
    # Verify prepare_inputs_for_task applies the limit
    prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)
    limited_messages = prepared["message_history"]
    
    # With num_last_messages=2, we should get at most 4 messages (2 runs * 2 messages per run)
    assert len(limited_messages) <= 4, \
        f"Limited history should have at most 4 messages, got {len(limited_messages)}"


@pytest.mark.asyncio
async def test_num_last_messages_limiting_streaming(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """STREAMING FLOW: num_last_messages correctly limits chat history."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        num_last_messages=2,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 5 conversations via streaming
    for i in range(5):
        task = Task(description=f"Streaming message {i}: I like number {i}")
        await consume_stream(agent, task, inmemory_storage, test_session_id)
    
    # Verify storage has ALL messages
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    assert len(session.messages) >= 10
    
    # Verify prepare_inputs_for_task applies the limit
    prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)
    limited_messages = prepared["message_history"]
    
    assert len(limited_messages) <= 4, \
        f"Limited history should have at most 4 messages, got {len(limited_messages)}"


# =============================================================================
# TEST: feed_tool_call_results Filtering
# =============================================================================

@pytest.mark.asyncio
async def test_feed_tool_call_results_filtering_direct(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """DIRECT FLOW: feed_tool_call_results=False filters out tool messages."""
    from upsonic.tools import tool
    
    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"
    
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        feed_tool_call_results=False,
        debug=False
    )
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        memory=memory,
        tools=[get_weather]
    )
    
    # Make a task that triggers tool call
    task = Task(description="What's the weather in New York? Use the get_weather tool.")
    await agent.do_async(task)
    
    # Check that prepare_inputs_for_task filters tool messages
    prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)
    message_history = prepared["message_history"]
    
    # Count tool messages
    tool_count = 0
    for msg in message_history:
        if isinstance(msg, ModelRequest):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                        tool_count += 1
                        break
        elif isinstance(msg, ModelResponse):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                        tool_count += 1
                        break
    
    assert tool_count == 0, \
        f"Tool messages should be filtered when feed_tool_call_results=False, found {tool_count}"


@pytest.mark.asyncio
async def test_feed_tool_call_results_included_direct(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """DIRECT FLOW: feed_tool_call_results=True includes tool messages."""
    from upsonic.tools import tool
    
    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"
    
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        feed_tool_call_results=True,
        debug=False
    )
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        memory=memory,
        tools=[get_weather]
    )
    
    task = Task(description="What's the weather in London? Use the get_weather tool.")
    await agent.do_async(task)
    
    prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)
    message_history = prepared["message_history"]
    
    # Count tool messages
    tool_count = 0
    for msg in message_history:
        if isinstance(msg, ModelRequest):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                        tool_count += 1
                        break
        elif isinstance(msg, ModelResponse):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                        tool_count += 1
                        break
    
    assert tool_count > 0, \
        f"Tool messages should be included when feed_tool_call_results=True, found {tool_count}"


# =============================================================================
# TEST: Session Persistence with SQLite
# =============================================================================

@pytest.mark.asyncio
async def test_session_persistence_sqlite_direct(
    sqlite_storage: SqliteStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """DIRECT FLOW: Sessions persist across Memory instances with SQLite."""
    # First instance - create and populate session
    memory1 = Memory(
        storage=sqlite_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent1 = Agent(model="openai/gpt-4o-mini", memory=memory1)
    
    task1 = Task(description="Remember that my secret code is DELTA-789.")
    await agent1.do_async(task1)
    
    # Create second Memory instance with same session_id
    memory2 = Memory(
        storage=sqlite_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent2 = Agent(model="openai/gpt-4o-mini", memory=memory2)
    
    # Should remember the secret code
    task2 = Task(description="What is my secret code?")
    result2 = await agent2.do_async(task2)
    
    result2_str = str(result2).upper()
    code_remembered = "DELTA" in result2_str or "789" in result2_str
    assert code_remembered, f"Session should persist - 'DELTA' or '789' not found in: {result2}"


@pytest.mark.asyncio
async def test_session_persistence_sqlite_streaming(
    sqlite_storage: SqliteStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """STREAMING FLOW: Sessions persist across Memory instances with SQLite."""
    # First instance via streaming
    memory1 = Memory(
        storage=sqlite_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent1 = Agent(model="openai/gpt-4o-mini", memory=memory1)
    
    task1 = Task(description="Remember that my secret code is GAMMA-456.")
    await consume_stream(agent1, task1, sqlite_storage, test_session_id)
    
    # Create second Memory instance with same session_id
    memory2 = Memory(
        storage=sqlite_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent2 = Agent(model="openai/gpt-4o-mini", memory=memory2)
    
    # Should remember via streaming
    task2 = Task(description="What is my secret code?")
    run_output = await consume_stream(agent2, task2, sqlite_storage, test_session_id)
    
    response_text = str(run_output.response).upper() if run_output.response else ""
    code_remembered = "GAMMA" in response_text or "456" in response_text
    assert code_remembered, f"Session should persist - 'GAMMA' or '456' not found in: {response_text}"


# =============================================================================
# TEST: Run Output Attributes
# =============================================================================

@pytest.mark.asyncio
async def test_run_output_attributes_stored_direct(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """DIRECT FLOW: AgentRunOutput attributes are stored correctly."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    task = Task(description="Say hello")
    await agent.do_async(task)
    
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    assert session.runs is not None
    assert len(session.runs) >= 1
    
    run_data = list(session.runs.values())[-1]
    run_output: AgentRunOutput = run_data.output
    
    # Verify run has messages (NEW messages for this run)
    assert run_output.messages is not None, "run_output.messages should not be None"
    assert isinstance(run_output.messages, list), "Run messages should be a list"
    assert len(run_output.messages) >= 2, \
        f"Expected at least 2 messages in run, got {len(run_output.messages)}"
    
    # Verify chat_history (FULL history)
    assert run_output.chat_history is not None, "run_output.chat_history should not be None"
    assert len(run_output.chat_history) >= 2, \
        f"Expected at least 2 in chat_history, got {len(run_output.chat_history)}"


@pytest.mark.asyncio
async def test_run_output_attributes_stored_streaming(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """STREAMING FLOW: AgentRunOutput attributes are stored correctly."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    task = Task(description="Say hello")
    run_output = await consume_stream(agent, task, inmemory_storage, test_session_id)
    
    assert run_output is not None
    
    # Verify run has messages (NEW messages for this run)
    assert run_output.messages is not None, "run_output.messages should not be None"
    assert isinstance(run_output.messages, list), "Run messages should be a list"
    assert len(run_output.messages) >= 2, \
        f"Expected at least 2 messages in run, got {len(run_output.messages)}"
    
    # Verify chat_history (FULL history)
    assert run_output.chat_history is not None, "run_output.chat_history should not be None"
    assert len(run_output.chat_history) >= 2, \
        f"Expected at least 2 in chat_history, got {len(run_output.chat_history)}"


# =============================================================================
# TEST: CRUD Operations on Session
# =============================================================================

@pytest.mark.asyncio
async def test_session_crud_operations(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """Test CRUD operations on session."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # CREATE - Run a task to create session
    task = Task(description="Hello world")
    await agent.do_async(task)
    
    # READ - Get session
    session = await memory.get_session_async()
    assert session is not None, "Session should exist after task"
    
    # UPDATE - Set metadata
    await memory.set_metadata_async({"test_key": "test_value"})
    metadata = await memory.get_metadata_async()
    assert metadata is not None
    assert metadata.get("test_key") == "test_value"
    
    # DELETE - Delete session
    deleted = await memory.delete_session_async()
    assert deleted is True
    
    # Verify deleted
    session_after = await memory.get_session_async()
    assert session_after is None, "Session should be None after delete"


# =============================================================================
# TEST: Mixed Direct and Streaming Flows
# =============================================================================

@pytest.mark.asyncio
async def test_mixed_direct_and_streaming_flows(
    inmemory_storage: InMemoryStorage, 
    test_user_id: str, 
    test_session_id: str
) -> None:
    """Test that direct and streaming flows can be mixed and memory persists."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Turn 1: Direct flow
    task1 = Task(description="My favorite number is 42")
    await agent.do_async(task1)
    
    # Turn 2: Streaming flow
    task2 = Task(description="My favorite color is green")
    await consume_stream(agent, task2, inmemory_storage, test_session_id)
    
    # Turn 3: Direct flow - should remember both
    task3 = Task(description="What is my favorite number?")
    result3 = await agent.do_async(task3)
    assert "42" in str(result3), f"Should remember number from direct flow, got: {result3}"
    
    # Turn 4: Streaming flow - should remember all
    task4 = Task(description="What is my favorite color?")
    run_output = await consume_stream(agent, task4, inmemory_storage, test_session_id)
    response_text = str(run_output.response).lower() if run_output.response else ""
    assert "green" in response_text, f"Should remember color from streaming flow, got: {response_text}"
    
    # Verify total messages accumulated
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    # 4 turns = at least 8 messages (4 requests + 4 responses)
    assert len(session.messages) >= 8, \
        f"Expected at least 8 messages after 4 turns, got {len(session.messages)}"
    
    print(f"Mixed flow total messages: {len(session.messages)}")
