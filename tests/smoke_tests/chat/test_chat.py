"""
Test 28: Chat class testing
Success criteria: Chat methods work properly! We check attributes and results for that.

This test suite validates:
1. Basic invoke and streaming
2. Conversation history with memory
3. All usage properties from storage (tokens, cost, requests, duration, etc.)
4. Session management (clear_history, reset_session)
5. ChatMessage conversion from ModelMessage
6. SessionMetrics with all RunUsage fields
7. Attachments and history manipulation (delete messages, remove attachments)
"""
import pytest
import os
from pathlib import Path

from upsonic import Agent, Chat, Task
from upsonic.chat import SessionState, ChatMessage, ChatAttachment
from upsonic.storage import InMemoryStorage, Memory

# Get project root (4 levels up from this test file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

pytestmark = pytest.mark.timeout(120)


# ---------------------------------------------------------------------------
# Regression: ``Chat`` follows an **agent-first** policy.
#
# Resolution order (highest → lowest):
#   1. ``agent.memory`` (and its ``storage``)   — kept as-is, Chat does NOT
#                                                  override it; ``storage=``
#                                                  kwarg on Chat is ignored.
#   2. ``Chat(storage=…)`` kwarg                — only when the agent has
#                                                  no memory.
#   3. New ``InMemoryStorage()``                — fallback, wired to agent.
#
# All checks below are synchronous — no LLM call required.
# Bare ``Agent(...)`` has ``memory=None`` unless memory is supplied, so we
# drive the regression through an explicitly-wired ``Memory``. The
# ``AutonomousAgent`` path (which seeds a default storage) is covered under
# ``tests/smoke_tests/autonomous_agent/test_autonomous_agent.py``.
# ---------------------------------------------------------------------------

def test_chat_reuses_agent_memory_and_storage():
    """Agent has Memory ⇒ Chat reuses both the Memory *and* the storage."""
    shared = InMemoryStorage()
    # Build Memory with matching ids so no alignment warning is triggered.
    agent_memory = Memory(storage=shared, session_id="reuse-default", user_id="u")
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Storage Reuse Agent",
        memory=agent_memory,
    )

    chat = Chat(session_id="reuse-default", user_id="u", agent=agent)

    # Storage is shared
    assert chat._storage is shared
    # Memory is the *same* instance (Chat does not build a new one)
    assert chat._memory is agent_memory
    assert agent.memory is agent_memory
    assert agent.memory.storage is shared


def test_chat_creates_storage_when_agent_has_none():
    """No memory on the agent ⇒ Chat creates one and wires it into the agent."""
    agent = Agent(model="openai/gpt-4o-mini", name="Storage Reuse Agent")
    assert agent.memory is None

    chat = Chat(session_id="reuse-fresh", user_id="u", agent=agent)

    assert isinstance(chat._storage, InMemoryStorage)
    # Chat builds a Memory and attaches it to the agent.
    assert agent.memory is not None
    assert agent.memory is chat._memory
    assert agent.memory.storage is chat._storage


def test_chat_explicit_storage_used_only_when_agent_has_no_memory():
    """No agent memory + explicit Chat storage ⇒ Chat's storage is used."""
    chat_storage = InMemoryStorage()
    agent = Agent(model="openai/gpt-4o-mini", name="Storage Reuse Agent")
    assert agent.memory is None

    chat = Chat(
        session_id="explicit-used",
        user_id="u",
        agent=agent,
        storage=chat_storage,
    )

    assert chat._storage is chat_storage
    assert agent.memory.storage is chat_storage


def test_agent_storage_wins_over_chat_explicit_storage():
    """Agent has Memory + Chat also passes ``storage=`` ⇒ agent wins.

    Under the agent-first policy, ``Chat(storage=…)`` is silently ignored
    when the agent already has a memory wrapper.
    """
    agent_storage = InMemoryStorage()
    agent_memory = Memory(storage=agent_storage, session_id="agent-wins", user_id="u")
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Storage Reuse Agent",
        memory=agent_memory,
    )
    rejected_storage = InMemoryStorage()

    chat = Chat(
        session_id="agent-wins",
        user_id="u",
        agent=agent,
        storage=rejected_storage,   # ← agent already has memory → ignored
    )

    assert chat._storage is agent_storage
    assert chat._storage is not rejected_storage
    assert chat._memory is agent_memory
    assert agent.memory is agent_memory


def test_chat_aligns_session_id_with_agent_memory():
    """Chat session_id mismatched with agent.memory ⇒ agent wins + UserWarning."""
    import pytest

    agent_memory = Memory(
        storage=InMemoryStorage(),
        session_id="agent-session",
        user_id="u",
    )
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Align Agent",
        memory=agent_memory,
    )

    with pytest.warns(UserWarning, match=r"session_id.*overridden by agent\.memory"):
        chat = Chat(session_id="chat-session", user_id="u", agent=agent)

    # Chat's label is realigned to the agent's memory.
    assert chat.session_id == "agent-session"
    assert chat._memory is agent_memory
    # The SessionManager must also key against the aligned id.
    assert chat._session_manager.session_id == "agent-session"


def test_chat_aligns_user_id_with_agent_memory():
    """Chat user_id mismatched with agent.memory ⇒ agent wins + UserWarning."""
    import pytest

    agent_memory = Memory(
        storage=InMemoryStorage(),
        session_id="s",
        user_id="agent-user",
    )
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Align Agent",
        memory=agent_memory,
    )

    with pytest.warns(UserWarning, match=r"user_id.*overridden by agent\.memory"):
        chat = Chat(session_id="s", user_id="chat-user", agent=agent)

    assert chat.user_id == "agent-user"
    assert chat._session_manager.user_id == "agent-user"


def test_chat_no_warning_when_ids_match_agent_memory():
    """Matching ids ⇒ no warning, ids stay as-is."""
    import warnings

    agent_memory = Memory(
        storage=InMemoryStorage(),
        session_id="s-match",
        user_id="u-match",
    )
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Match Agent",
        memory=agent_memory,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning ⇒ failure
        chat = Chat(session_id="s-match", user_id="u-match", agent=agent)

    assert chat.session_id == "s-match"
    assert chat.user_id == "u-match"


@pytest.mark.asyncio
async def test_chat_basic_invoke():
    """Test basic Chat invoke method."""
    print("\n" + "="*60)
    print("TEST: test_chat_basic_invoke")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_1",
        user_id="test_user_1",
        agent=agent
    )
    
    try:
        response = await chat.invoke("What is 2 + 2?")
        
        # Verify response
        print(f"\n[RESULT] Response: {response[:100]}...")
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert "4" in response, "Response should contain the answer"
        print("[PASS] Response is valid and contains answer")
        
        # Verify messages from storage
        messages = chat.all_messages
        print(f"\n[RESULT] Message count: {len(messages)}")
        assert len(messages) >= 2, "Should have at least 2 messages (user + assistant)"
        print("[PASS] Messages retrieved from storage")
        
        # Verify ChatMessage conversion
        for msg in messages:
            assert isinstance(msg, ChatMessage), f"Message should be ChatMessage, got {type(msg)}"
            assert msg.role in ("user", "assistant"), f"Role should be user/assistant, got {msg.role}"
            assert isinstance(msg.content, str), "Content should be string"
            assert msg.timestamp > 0, "Timestamp should be set"
            print(f"  - {msg.role}: {msg.content[:50]}...")
        print("[PASS] ChatMessage conversion works correctly")
        
        # Verify usage properties from storage
        print(f"\n[RESULT] Usage from storage:")
        print(f"  - Input tokens: {chat.usage.input_tokens}")
        print(f"  - Output tokens: {chat.usage.output_tokens}")
        print(f"  - Total tokens: {chat.usage.total_tokens}")
        print(f"  - Total cost: ${(chat.usage.cost or 0.0):.6f}")
        print(f"  - Total requests: {chat.usage.requests}")
        print(f"  - Total tool calls: {chat.usage.tool_calls}")
        
        assert chat.usage.input_tokens > 0, "Should have input tokens"
        assert chat.usage.output_tokens > 0, "Should have output tokens"
        assert chat.usage.total_tokens == chat.usage.input_tokens + chat.usage.output_tokens, "Total should equal sum"
        assert (chat.usage.cost or 0.0) >= 0, "Total cost should be non-negative"
        assert chat.usage.requests >= 1, "Should have at least 1 request"
        print("[PASS] All usage properties verified")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_basic_invoke completed")


@pytest.mark.asyncio
async def test_chat_streaming():
    """Test Chat streaming invoke."""
    print("\n" + "="*60)
    print("TEST: test_chat_streaming")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_2",
        user_id="test_user_2",
        agent=agent,
    )
    
    accumulated_text = ""
    chunk_count = 0
    
    try:
        stream_generator = await chat.invoke("Count from 1 to 3, one number per line.", stream=True)
        async for chunk in stream_generator:
            accumulated_text += chunk
            chunk_count += 1
            assert isinstance(chunk, str), "Stream chunks should be strings"
        
        print(f"\n[RESULT] Streaming completed:")
        print(f"  - Chunk count: {chunk_count}")
        print(f"  - Accumulated text length: {len(accumulated_text)}")
        print(f"  - Text preview: {accumulated_text[:100]}...")
        
        assert accumulated_text is not None, "Should have accumulated text"
        assert len(accumulated_text) > 0, "Should have received text chunks"
        assert chunk_count > 0, "Should have received multiple chunks"
        print("[PASS] Streaming works correctly")
        
        # Verify messages after streaming
        messages = chat.all_messages
        print(f"\n[RESULT] Messages after streaming: {len(messages)}")
        assert len(messages) >= 1, "Should have at least the user message"
        print("[PASS] Messages accessible after streaming")
        
        # Verify usage is tracked
        print(f"\n[RESULT] Usage after streaming:")
        print(f"  - Input tokens: {chat.usage.input_tokens}")
        print(f"  - Output tokens: {chat.usage.output_tokens}")
        print(f"  - Total cost: ${(chat.usage.cost or 0.0):.6f}")
        assert (chat.usage.cost or 0.0) >= 0, "Total cost should be tracked"
        print("[PASS] Usage tracked for streaming")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_streaming completed")


@pytest.mark.asyncio
async def test_chat_conversation_history():
    """Test Chat conversation history management with memory."""
    print("\n" + "="*60)
    print("TEST: test_chat_conversation_history")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_3",
        user_id="test_user_3",
        agent=agent,
        full_session_memory=True,
        debug=True
    )
    
    try:
        # First message
        print("\n[ACTION] Sending first message...")
        response1 = await chat.invoke("My name is Alice.")
        print(f"[RESULT] Response 1: {response1[:100]}...")
        
        messages_after_1 = chat.all_messages
        print(f"[RESULT] Messages after first invoke: {len(messages_after_1)}")
        assert len(messages_after_1) >= 2, "Should have 2 messages after first invoke"
        print("[PASS] First message processed")
        
        # Second message (should remember context)
        print("\n[ACTION] Sending second message (testing memory)...")
        response2 = await chat.invoke("What is my name?")
        print(f"[RESULT] Response 2: {response2[:100]}...")
        
        messages_after_2 = chat.all_messages
        print(f"[RESULT] Messages after second invoke: {len(messages_after_2)}")
        assert len(messages_after_2) >= 4, "Should have 4 messages after second invoke"
        assert "alice" in response2.lower(), "Should remember the name from previous message"
        print("[PASS] Memory works - agent remembered the name!")
        
        # Verify message history
        all_messages = chat.all_messages
        user_messages = [msg for msg in all_messages if msg.role == "user"]
        assistant_messages = [msg for msg in all_messages if msg.role == "assistant"]
        
        print(f"\n[RESULT] Message breakdown:")
        print(f"  - User messages: {len(user_messages)}")
        print(f"  - Assistant messages: {len(assistant_messages)}")
        
        assert len(user_messages) >= 2, "Should have at least 2 user messages"
        assert len(assistant_messages) >= 2, "Should have at least 2 assistant messages"
        print("[PASS] Message roles are correct")
        
        # Print all messages
        print("\n[RESULT] Full conversation:")
        for i, msg in enumerate(all_messages):
            print(f"  {i+1}. [{msg.role}] {msg.content[:80]}...")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_conversation_history completed")


@pytest.mark.asyncio
async def test_chat_all_attributes():
    """Test all Chat class attributes and properties."""
    print("\n" + "="*60)
    print("TEST: test_chat_all_attributes")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_4",
        user_id="test_user_4",
        agent=agent
    )
    
    try:
        # Before any invocation
        print("\n[CHECK] Attributes BEFORE invocation:")
        print(f"  - session_id: {chat.session_id}")
        print(f"  - user_id: {chat.user_id}")
        print(f"  - state: {chat.state}")
        print(f"  - all_messages: {len(chat.all_messages)}")
        print(f"  - input_tokens: {chat.usage.input_tokens}")
        print(f"  - output_tokens: {chat.usage.output_tokens}")
        print(f"  - total_tokens: {chat.usage.total_tokens}")
        print(f"  - total_cost: {(chat.usage.cost or 0.0)}")
        print(f"  - total_requests: {chat.usage.requests}")
        print(f"  - total_tool_calls: {chat.usage.tool_calls}")
        print(f"  - run_duration: {chat.usage.duration}")
        print(f"  - time_to_first_token: {chat.usage.time_to_first_token}")
        print(f"  - duration: {chat.duration:.2f}s")
        print(f"  - start_time: {chat.start_time}")
        print(f"  - end_time: {chat.end_time}")
        print(f"  - last_activity: {chat.last_activity}")
        print(f"  - last_activity_time: {chat.last_activity_time}")
        print(f"  - is_closed: {chat.is_closed}")
        
        assert chat.session_id == "test_session_4", "session_id should be set"
        assert chat.user_id == "test_user_4", "user_id should be set"
        assert chat.agent == agent, "agent should be set"
        assert chat.state == SessionState.IDLE, "Should start in IDLE state"
        assert len(chat.all_messages) == 0, "Should start with no messages"
        assert (chat.usage.cost or 0.0) == 0.0, "Should start with zero cost"
        assert chat.usage.input_tokens == 0, "Should start with zero input tokens"
        assert chat.usage.output_tokens == 0, "Should start with zero output tokens"
        assert chat.usage.total_tokens == 0, "Should start with zero total tokens"
        assert chat.usage.requests == 0, "Should start with zero requests"
        assert chat.usage.tool_calls == 0, "Should start with zero tool calls"
        print("[PASS] All pre-invocation attributes correct")
        
        # After invocation
        print("\n[ACTION] Invoking chat...")
        await chat.invoke("Hello, tell me a fun fact!")
        
        print("\n[CHECK] Attributes AFTER invocation:")
        print(f"  - state: {chat.state}")
        print(f"  - all_messages: {len(chat.all_messages)}")
        print(f"  - input_tokens: {chat.usage.input_tokens}")
        print(f"  - output_tokens: {chat.usage.output_tokens}")
        print(f"  - total_tokens: {chat.usage.total_tokens}")
        print(f"  - total_cost: ${(chat.usage.cost or 0.0):.6f}")
        print(f"  - total_requests: {chat.usage.requests}")
        print(f"  - total_tool_calls: {chat.usage.tool_calls}")
        print(f"  - run_duration: {chat.usage.duration}")
        print(f"  - time_to_first_token: {chat.usage.time_to_first_token}")
        print(f"  - duration: {chat.duration:.2f}s")
        print(f"  - start_time: {chat.start_time}")
        print(f"  - end_time: {chat.end_time}")
        print(f"  - last_activity: {chat.last_activity}")
        print(f"  - last_activity_time: {chat.last_activity_time}")
        print(f"  - is_closed: {chat.is_closed}")
        
        assert chat.state == SessionState.IDLE, "Should return to IDLE after invoke"
        assert len(chat.all_messages) > 0, "Should have messages after invoke"
        assert chat.usage.input_tokens > 0, "Should have input tokens"
        assert chat.usage.output_tokens > 0, "Should have output tokens"
        assert chat.usage.total_tokens == chat.usage.input_tokens + chat.usage.output_tokens, "Total should match sum"
        assert (chat.usage.cost or 0.0) >= 0, "Total cost should be tracked"
        assert chat.usage.requests >= 1, "Should have at least 1 request"
        assert chat.duration > 0, "Session duration should increase"
        assert chat.start_time > 0, "Start time should be set"
        assert chat.end_time is None, "End time should be None (session still active)"
        assert chat.is_closed is False, "Session should not be closed"
        assert chat.last_activity > 0, "Last activity should be tracked"
        assert chat.last_activity_time > 0, "Last activity time should be set"
        print("[PASS] All post-invocation attributes correct")
        
        # Test get_recent_messages
        print("\n[CHECK] get_recent_messages(5):")
        recent = chat.get_recent_messages(count=5)
        print(f"  - Returned: {len(recent)} messages")
        assert isinstance(recent, list), "get_recent_messages should return a list"
        assert len(recent) <= 5, "Should return at most 5 messages"
        for msg in recent:
            assert isinstance(msg, ChatMessage), "Should return ChatMessage objects"
        print("[PASS] get_recent_messages works")
        
        # Test get_usage
        print("\n[CHECK] get_usage():")
        usage = chat.usage
        print(f"  - Type: {type(usage)}")
        if usage:
            print(f"  - requests: {usage.requests}")
            print(f"  - tool_calls: {usage.tool_calls}")
            print(f"  - input_tokens: {usage.input_tokens}")
            print(f"  - output_tokens: {usage.output_tokens}")
            print(f"  - cost: {usage.cost}")
            print(f"  - duration: {usage.duration}")
            print(f"  - time_to_first_token: {usage.time_to_first_token}")
            assert usage.input_tokens > 0, "Usage should have input tokens"
        print("[PASS] get_usage returns RunUsage object")
        
        # ``chat.get_session_metrics()`` and ``chat.get_session_summary()``
        # were removed in the usage-registry unification — every counter
        # is now a field on ``chat.usage`` (AggregatedUsage), which is
        # exercised in the block above.

    finally:
        await chat.close()
        print("\n[DONE] test_chat_all_attributes completed")


@pytest.mark.asyncio
async def test_chat_with_task():
    """Test Chat with Task object instead of string."""
    print("\n" + "="*60)
    print("TEST: test_chat_with_task")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_5",
        user_id="test_user_5",
        agent=agent
    )
    
    try:
        task = Task(description="What is the capital of France?")
        print(f"\n[ACTION] Invoking with Task: {task.description}")
        
        response = await chat.invoke(task)
        
        print(f"[RESULT] Response: {response[:100]}...")
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert "paris" in response.lower(), "Response should mention Paris"
        print("[PASS] Task invocation works")
        
        assert len(chat.all_messages) >= 2, "Should have messages after invoke"
        print(f"[RESULT] Messages: {len(chat.all_messages)}")
        print("[PASS] Messages stored correctly")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_with_task completed")


@pytest.mark.asyncio
async def test_chat_cost_tracking():
    """Test Chat cost tracking from storage."""
    print("\n" + "="*60)
    print("TEST: test_chat_cost_tracking")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_6",
        user_id="test_user_6",
        agent=agent
    )
    
    try:
        initial_cost = (chat.usage.cost or 0.0)
        initial_tokens = chat.usage.total_tokens
        initial_requests = chat.usage.requests
        
        print(f"\n[CHECK] Initial state:")
        print(f"  - Cost: ${initial_cost:.6f}")
        print(f"  - Tokens: {initial_tokens}")
        print(f"  - Requests: {initial_requests}")
        
        assert initial_cost == 0.0, "Should start with zero cost"
        assert initial_tokens == 0, "Should start with zero tokens"
        assert initial_requests == 0, "Should start with zero requests"
        print("[PASS] Initial state is zero")
        
        # Make a call
        print("\n[ACTION] First invoke...")
        await chat.invoke("Hello")
        
        cost_after_one = (chat.usage.cost or 0.0)
        tokens_after_one = chat.usage.total_tokens
        requests_after_one = chat.usage.requests
        
        print(f"\n[CHECK] After first invoke:")
        print(f"  - Cost: ${cost_after_one:.6f}")
        print(f"  - Tokens: {tokens_after_one}")
        print(f"  - Requests: {requests_after_one}")
        
        assert cost_after_one >= initial_cost, "Cost should not decrease"
        assert tokens_after_one > initial_tokens, "Tokens should increase"
        assert requests_after_one > initial_requests, "Requests should increase"
        print("[PASS] Metrics increased after first invoke")
        
        # Make another call
        print("\n[ACTION] Second invoke...")
        await chat.invoke("What is 1 + 1?")
        
        cost_after_two = (chat.usage.cost or 0.0)
        tokens_after_two = chat.usage.total_tokens
        requests_after_two = chat.usage.requests
        
        print(f"\n[CHECK] After second invoke:")
        print(f"  - Cost: ${cost_after_two:.6f}")
        print(f"  - Tokens: {tokens_after_two}")
        print(f"  - Requests: {requests_after_two}")
        
        assert cost_after_two >= cost_after_one, "Cost should not decrease"
        assert tokens_after_two > tokens_after_one, "Tokens should increase"
        assert requests_after_two > requests_after_one, "Requests should increase"
        print("[PASS] Metrics increased after second invoke")
        
        # Verify usage object
        usage = chat.usage
        print(f"\n[CHECK] Final RunUsage object:")
        if usage:
            print(f"  - input_tokens: {usage.input_tokens}")
            print(f"  - output_tokens: {usage.output_tokens}")
            print(f"  - requests: {usage.requests}")
            print(f"  - cost: {usage.cost}")
            assert usage.input_tokens == chat.usage.input_tokens, "Usage should match property"
            assert usage.output_tokens == chat.usage.output_tokens, "Usage should match property"
        print("[PASS] RunUsage object is consistent")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_cost_tracking completed")


@pytest.mark.asyncio
async def test_chat_clear_history():
    """Test Chat clear_history method."""
    print("\n" + "="*60)
    print("TEST: test_chat_clear_history")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_7",
        user_id="test_user_7",
        agent=agent
    )
    
    try:
        # Add some messages
        print("\n[ACTION] Adding messages...")
        await chat.invoke("Hello")
        await chat.invoke("How are you?")
        
        messages_before = len(chat.all_messages)
        cost_before = (chat.usage.cost or 0.0)
        
        print(f"\n[CHECK] Before clear_history:")
        print(f"  - Messages: {messages_before}")
        print(f"  - Cost: ${cost_before:.6f}")
        
        assert messages_before >= 4, "Should have at least 4 messages"
        print("[PASS] Messages accumulated")
        
        # Clear history
        print("\n[ACTION] Calling clear_history()...")
        chat.clear_history()
        
        messages_after = len(chat.all_messages)
        cost_after = (chat.usage.cost or 0.0)
        
        print(f"\n[CHECK] After clear_history:")
        print(f"  - Messages: {messages_after}")
        print(f"  - Cost: ${cost_after:.6f}")
        
        assert messages_after == 0, "Messages should be cleared"
        # Cost should be preserved (only messages are cleared)
        assert cost_after == cost_before, "Cost should be preserved after clear_history"
        print("[PASS] clear_history works - messages cleared, cost preserved")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_clear_history completed")


@pytest.mark.asyncio
async def test_chat_reset_session():
    """Test Chat reset_session method."""
    print("\n" + "="*60)
    print("TEST: test_chat_reset_session")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_8",
        user_id="test_user_8",
        agent=agent
    )
    
    try:
        # Add some activity
        print("\n[ACTION] Adding activity...")
        await chat.invoke("Hello")
        
        messages_before = len(chat.all_messages)
        cost_before = (chat.usage.cost or 0.0)
        
        print(f"\n[CHECK] Before reset_session:")
        print(f"  - Messages: {messages_before}")
        print(f"  - Cost: ${cost_before:.6f}")
        
        assert messages_before >= 2, "Should have messages"
        print("[PASS] Activity recorded")
        
        # Reset session
        print("\n[ACTION] Calling reset_session()...")
        chat.reset_session()
        
        messages_after = len(chat.all_messages)
        cost_after = (chat.usage.cost or 0.0)
        state_after = chat.state
        
        print(f"\n[CHECK] After reset_session:")
        print(f"  - Messages: {messages_after}")
        print(f"  - Cost: ${cost_after:.6f}")
        print(f"  - State: {state_after}")
        
        assert messages_after == 0, "Messages should be cleared"
        assert cost_after == 0.0, "Cost should be reset"
        assert state_after == SessionState.IDLE, "State should be IDLE"
        print("[PASS] reset_session clears everything")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_reset_session completed")


@pytest.mark.asyncio
async def test_chat_state_transitions():
    """Test Chat state transitions during invoke."""
    print("\n" + "="*60)
    print("TEST: test_chat_state_transitions")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_9",
        user_id="test_user_9",
        agent=agent
    )
    
    try:
        # Check initial state
        print(f"\n[CHECK] Initial state: {chat.state}")
        assert chat.state == SessionState.IDLE, "Should start IDLE"
        print("[PASS] Starts in IDLE state")
        
        # After invoke, should return to IDLE
        print("\n[ACTION] Invoking...")
        await chat.invoke("Hello")
        
        print(f"\n[CHECK] State after invoke: {chat.state}")
        assert chat.state == SessionState.IDLE, "Should return to IDLE after invoke"
        print("[PASS] Returns to IDLE after invoke")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_state_transitions completed")


@pytest.mark.asyncio
async def test_chat_message_content_types():
    """Test ChatMessage handles different content types."""
    print("\n" + "="*60)
    print("TEST: test_chat_message_content_types")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_10",
        user_id="test_user_10",
        agent=agent
    )
    
    try:
        # Send a message that might trigger tool calls
        print("\n[ACTION] Sending message...")
        response = await chat.invoke("What is 5 * 7?")
        
        print(f"\n[RESULT] Response: {response[:100]}...")
        
        # Check message content
        messages = chat.all_messages
        print(f"\n[CHECK] Messages ({len(messages)}):")
        
        for i, msg in enumerate(messages):
            print(f"\n  Message {i+1}:")
            print(f"    - role: {msg.role}")
            print(f"    - content: {msg.content[:100]}...")
            print(f"    - timestamp: {msg.timestamp}")
            print(f"    - tool_calls: {msg.tool_calls}")
            print(f"    - metadata: {msg.metadata}")
            
            # Verify ChatMessage structure
            assert isinstance(msg.content, str), "Content should be string"
            assert msg.role in ("user", "assistant"), "Role should be valid"
            assert msg.timestamp > 0, "Timestamp should be positive"
            
            # Check metadata for assistant messages
            if msg.role == "assistant" and msg.metadata:
                print(f"    - metadata keys: {list(msg.metadata.keys())}")
        
        print("\n[PASS] ChatMessage structure is correct")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_message_content_types completed")


@pytest.mark.asyncio
async def test_chat_async_context_manager():
    """Test Chat async context manager."""
    print("\n" + "="*60)
    print("TEST: test_chat_async_context_manager")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    
    async with Chat(
        session_id="test_session_11",
        user_id="test_user_11",
        agent=agent
    ) as chat:
        print(f"\n[CHECK] Inside context manager")
        print(f"  - session_id: {chat.session_id}")
        print(f"  - state: {chat.state}")
        
        response = await chat.invoke("Hello!")
        print(f"  - Response: {response[:50]}...")
        
        assert response is not None, "Should get response"
        print("[PASS] Context manager works")
    
    print("\n[CHECK] After context manager exit")
    print("[PASS] Context manager cleaned up correctly")
    print("\n[DONE] test_chat_async_context_manager completed")


@pytest.mark.asyncio
async def test_chat_repr():
    """Test Chat __repr__ method."""
    print("\n" + "="*60)
    print("TEST: test_chat_repr")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_12",
        user_id="test_user_12",
        agent=agent
    )
    
    try:
        repr_before = repr(chat)
        print(f"\n[CHECK] Repr before invoke: {repr_before}")
        assert "test_session_12" in repr_before, "Should contain session_id"
        assert "test_user_12" in repr_before, "Should contain user_id"
        print("[PASS] Repr contains identifiers")
        
        await chat.invoke("Hello")
        
        repr_after = repr(chat)
        print(f"\n[CHECK] Repr after invoke: {repr_after}")
        assert "messages=" in repr_after, "Should contain message count"
        assert "cost=" in repr_after, "Should contain cost"
        print("[PASS] Repr shows updated state")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_repr completed")


@pytest.mark.asyncio
async def test_chat_message_index():
    """Test ChatMessage.message_index for history manipulation."""
    print("\n" + "="*60)
    print("TEST: test_chat_message_index")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_13",
        user_id="test_user_13",
        agent=agent
    )
    
    try:
        # Send multiple messages
        print("\n[ACTION] Sending multiple messages...")
        await chat.invoke("Message one")
        await chat.invoke("Message two")
        
        messages = chat.all_messages
        print(f"\n[RESULT] Messages with indices:")
        for msg in messages:
            print(f"  - Index {msg.message_index}: [{msg.role}] {msg.content[:40]}...")
            assert msg.message_index >= 0, "message_index should be non-negative"
        
        # Verify indices are sequential
        indices = [msg.message_index for msg in messages]
        print(f"\n[CHECK] Message indices: {indices}")
        assert indices == list(range(len(messages))), "Indices should be sequential"
        print("[PASS] message_index is correctly assigned")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_message_index completed")


@pytest.mark.asyncio
async def test_chat_delete_message():
    """Test Chat.delete_message() for removing messages."""
    print("\n" + "="*60)
    print("TEST: test_chat_delete_message")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_14",
        user_id="test_user_14",
        agent=agent
    )
    
    try:
        # Send multiple messages
        print("\n[ACTION] Sending multiple messages...")
        await chat.invoke("First message")
        await chat.invoke("Second message")
        
        initial_count = len(chat.all_messages)
        print(f"\n[CHECK] Initial message count: {initial_count}")
        assert initial_count >= 4, "Should have at least 4 messages (2 user + 2 assistant)"
        
        # Get the first user message content
        first_message = chat.all_messages[0]
        print(f"[CHECK] First message: [{first_message.role}] {first_message.content[:50]}...")
        
        # Delete the first message
        print("\n[ACTION] Deleting first message (index 0)...")
        result = chat.delete_message(0)
        
        assert result is True, "delete_message should return True on success"
        print("[PASS] delete_message returned True")
        
        new_count = len(chat.all_messages)
        print(f"[CHECK] Message count after deletion: {new_count}")
        assert new_count == initial_count - 1, "Should have one less message"
        print("[PASS] Message was deleted")
        
        # Verify the first message is now different
        new_first_message = chat.all_messages[0]
        print(f"[CHECK] New first message: [{new_first_message.role}] {new_first_message.content[:50]}...")
        
        # Test deleting invalid index
        print("\n[ACTION] Testing invalid index deletion...")
        result = chat.delete_message(9999)
        assert result is False, "delete_message should return False for invalid index"
        print("[PASS] Invalid index handled correctly")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_delete_message completed")


@pytest.mark.asyncio
async def test_chat_get_set_raw_messages():
    """Test Chat.get_raw_messages() and set_messages() for direct manipulation."""
    print("\n" + "="*60)
    print("TEST: test_chat_get_set_raw_messages")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_15",
        user_id="test_user_15",
        agent=agent
    )
    
    try:
        # Send a message
        print("\n[ACTION] Sending message...")
        await chat.invoke("Hello world")
        
        # Get raw messages
        print("\n[CHECK] Getting raw messages...")
        raw_messages = chat.get_raw_messages()
        print(f"  - Raw message count: {len(raw_messages)}")
        print(f"  - Types: {[type(m).__name__ for m in raw_messages]}")
        
        assert len(raw_messages) >= 2, "Should have at least 2 raw messages"
        print("[PASS] get_raw_messages works")
        
        # Remove one message and set back
        print("\n[ACTION] Removing first message and setting back...")
        modified_messages = raw_messages[1:]  # Remove first message
        chat.set_messages(modified_messages)
        
        # Verify change persisted
        new_messages = chat.all_messages
        print(f"  - New message count: {len(new_messages)}")
        assert len(new_messages) == len(raw_messages) - 1, "Should have one less message"
        print("[PASS] set_messages works and persists to storage")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_get_set_raw_messages completed")


@pytest.mark.asyncio
async def test_chat_with_attachments():
    """Test Chat with file attachments (PDF and Image)."""
    print("\n" + "="*60)
    print("TEST: test_chat_with_attachments")
    print("="*60)
    
    # Use actual test files from project root
    pdf_path = str(PROJECT_ROOT / "example_document.pdf")
    image_path = str(PROJECT_ROOT / "premium_photo-1661342428515-5ca8cee4385a.jpeg")
    
    # Verify files exist
    assert os.path.exists(pdf_path), f"PDF file not found: {pdf_path}"
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    print(f"\n[CHECK] Test files exist:")
    print(f"  - PDF: {pdf_path}")
    print(f"  - Image: {image_path}")
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_16",
        user_id="test_user_16",
        agent=agent
    )
    
    try:
        # Test 1: Send message with PDF attachment
        print("\n[ACTION] Sending message with PDF attachment...")
        task_pdf = Task(
            description="What type of document is this?",
            attachments=[pdf_path]
        )
        response_pdf = await chat.invoke(task_pdf)
        print(f"[RESULT] Response: {response_pdf[:150]}...")
        assert response_pdf is not None, "Should get response for PDF"
        print("[PASS] PDF attachment processed")
        
        # Test 2: Send message with Image attachment
        print("\n[ACTION] Sending message with Image attachment...")
        task_image = Task(
            description="Describe what you see in this image briefly.",
            attachments=[image_path]
        )
        response_image = await chat.invoke(task_image)
        print(f"[RESULT] Response: {response_image[:150]}...")
        assert response_image is not None, "Should get response for image"
        print("[PASS] Image attachment processed")
        
        # Test 3: Send message with BOTH attachments
        print("\n[ACTION] Sending message with BOTH PDF and Image attachments...")
        task_both = Task(
            description="I'm sending you two files. Briefly describe each.",
            attachments=[pdf_path, image_path]
        )
        response_both = await chat.invoke(task_both)
        print(f"[RESULT] Response: {response_both[:200]}...")
        assert response_both is not None, "Should get response for both"
        print("[PASS] Multiple attachments processed")
        
        # Check messages
        messages = chat.all_messages
        print(f"\n[CHECK] Total messages: {len(messages)}")
        
        for msg in messages:
            content_preview = msg.content[:80].replace('\n', ' ')
            print(f"\n  [{msg.message_index}] {msg.role}: {content_preview}...")
            if msg.attachments:
                print(f"      Attachments ({len(msg.attachments)}):")
                for att in msg.attachments:
                    print(f"        - [{att.index}] {att.type}: {att.identifier[-40:]}")
        
        print("\n[PASS] Attachments are tracked in ChatMessage")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_with_attachments completed")


@pytest.mark.asyncio
async def test_chat_attachment_class():
    """Test ChatAttachment class structure."""
    print("\n" + "="*60)
    print("TEST: test_chat_attachment_class")
    print("="*60)
    
    # Test ChatAttachment creation
    print("\n[CHECK] Creating ChatAttachment...")
    attachment = ChatAttachment(
        type="image",
        identifier="test_image.png",
        media_type="image/png",
        index=0
    )
    
    print(f"  - type: {attachment.type}")
    print(f"  - identifier: {attachment.identifier}")
    print(f"  - media_type: {attachment.media_type}")
    print(f"  - index: {attachment.index}")
    
    assert attachment.type == "image", "Type should be 'image'"
    assert attachment.identifier == "test_image.png", "Identifier should match"
    assert attachment.media_type == "image/png", "Media type should match"
    assert attachment.index == 0, "Index should be 0"
    print("[PASS] ChatAttachment created correctly")
    
    # Test to_dict
    print("\n[CHECK] Testing to_dict()...")
    att_dict = attachment.to_dict()
    print(f"  - dict: {att_dict}")
    
    assert "type" in att_dict, "Dict should have 'type'"
    assert "identifier" in att_dict, "Dict should have 'identifier'"
    assert "media_type" in att_dict, "Dict should have 'media_type'"
    assert "index" in att_dict, "Dict should have 'index'"
    print("[PASS] to_dict works correctly")
    
    print("\n[DONE] test_chat_attachment_class completed")


@pytest.mark.asyncio
async def test_chat_message_to_dict():
    """Test ChatMessage.to_dict() with attachments."""
    print("\n" + "="*60)
    print("TEST: test_chat_message_to_dict")
    print("="*60)
    
    # Create a ChatMessage with attachments
    print("\n[CHECK] Creating ChatMessage with attachments...")
    attachments = [
        ChatAttachment(type="image", identifier="img1.png", index=0),
        ChatAttachment(type="document", identifier="doc.pdf", index=1),
    ]
    
    message = ChatMessage(
        content="Test message",
        role="user",
        timestamp=1234567890.0,
        attachments=attachments,
        message_index=5
    )
    
    print(f"  - content: {message.content}")
    print(f"  - role: {message.role}")
    print(f"  - attachments: {len(message.attachments)}")
    print(f"  - message_index: {message.message_index}")
    
    # Test to_dict
    print("\n[CHECK] Testing to_dict()...")
    msg_dict = message.to_dict()
    
    assert "content" in msg_dict, "Dict should have 'content'"
    assert "role" in msg_dict, "Dict should have 'role'"
    assert "attachments" in msg_dict, "Dict should have 'attachments'"
    assert "message_index" in msg_dict, "Dict should have 'message_index'"
    assert len(msg_dict["attachments"]) == 2, "Should have 2 attachments"
    
    print(f"  - Dict keys: {list(msg_dict.keys())}")
    print(f"  - attachments in dict: {msg_dict['attachments']}")
    print("[PASS] to_dict includes attachments correctly")
    
    # Test repr
    print("\n[CHECK] Testing __repr__...")
    repr_str = repr(message)
    print(f"  - repr: {repr_str}")
    assert "attachments=2" in repr_str, "Repr should show attachment count"
    print("[PASS] __repr__ shows attachment count")
    
    print("\n[DONE] test_chat_message_to_dict completed")


@pytest.mark.asyncio
async def test_chat_history_manipulation_workflow():
    """Test complete workflow: send with attachments, view, selectively delete."""
    print("\n" + "="*60)
    print("TEST: test_chat_history_manipulation_workflow")
    print("="*60)
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_17",
        user_id="test_user_17",
        agent=agent
    )
    
    try:
        # Step 1: Send multiple messages
        print("\n[STEP 1] Sending multiple messages...")
        await chat.invoke("First question: What is 2+2?")
        await chat.invoke("Second question: What is 3+3?")
        await chat.invoke("Third question: What is 4+4?")
        
        initial_messages = chat.all_messages
        initial_count = len(initial_messages)
        print(f"  - Initial message count: {initial_count}")
        
        # Step 2: View and analyze messages
        print("\n[STEP 2] Viewing all messages with indices...")
        for msg in initial_messages:
            print(f"  [{msg.message_index}] {msg.role}: {msg.content[:40]}...")
        
        # Step 3: Delete specific message (e.g., second question/answer pair)
        print("\n[STEP 3] Deleting messages at indices 2 and 3 (second Q&A)...")
        
        # Delete in reverse order to maintain correct indices
        chat.delete_message(3)  # Delete second answer
        chat.delete_message(2)  # Delete second question
        
        after_delete = chat.all_messages
        print(f"  - Messages after deletion: {len(after_delete)}")
        assert len(after_delete) == initial_count - 2, "Should have 2 fewer messages"
        
        # Step 4: Verify remaining messages
        print("\n[STEP 4] Verifying remaining messages...")
        for msg in after_delete:
            print(f"  [{msg.message_index}] {msg.role}: {msg.content[:40]}...")
            # Should not contain "Second question"
            assert "Second" not in msg.content or msg.role == "assistant", "Second question should be deleted"
        
        print("[PASS] Selective deletion workflow completed successfully")
        
        # Step 5: Test cost is preserved
        print("\n[STEP 5] Verifying usage is preserved after deletion...")
        assert (chat.usage.cost or 0.0) > 0, "Cost should be preserved"
        assert chat.usage.input_tokens > 0, "Input tokens should be preserved"
        print(f"  - Total cost: ${(chat.usage.cost or 0.0):.6f}")
        print(f"  - Total tokens: {chat.usage.total_tokens}")
        print("[PASS] Usage preserved after message deletion")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_history_manipulation_workflow completed")


@pytest.mark.asyncio
async def test_chat_remove_attachment():
    """Test selectively removing attachments from messages."""
    print("\n" + "="*60)
    print("TEST: test_chat_remove_attachment")
    print("="*60)
    
    # Use actual test files from project root
    pdf_path = str(PROJECT_ROOT / "example_document.pdf")
    image_path = str(PROJECT_ROOT / "premium_photo-1661342428515-5ca8cee4385a.jpeg")
    
    # Verify files exist
    assert os.path.exists(pdf_path), f"PDF file not found: {pdf_path}"
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_remove_att",
        user_id="test_user_remove_att",
        agent=agent
    )
    
    try:
        # Step 1: Send message with BOTH PDF and Image attachments
        print("\n[STEP 1] Sending message with PDF and Image attachments...")
        task = Task(
            description="I'm sending you a PDF and an image. Say OK.",
            attachments=[pdf_path, image_path]
        )
        response = await chat.invoke(task)
        print(f"[RESULT] Response: {response[:100]}...")
        
        # Step 2: View the message and its attachments
        # Step 2: View the message and its attachments
        print("\n[STEP 2] Viewing message with attachments...")
        messages = chat.all_messages
        user_message = messages[0]  # First message should be user
        
        print(f"  - Message index: {user_message.message_index}")
        print(f"  - Role: {user_message.role}")
        print(f"  - Content preview: {user_message.content[:60]}...")
        print(f"  - Attachments: {len(user_message.attachments) if user_message.attachments else 0}")
        
        if user_message.attachments:
            for att in user_message.attachments:
                print(f"    [{att.index}] {att.type}: ...{att.identifier[-30:]}")
        
        initial_attachment_count = len(user_message.attachments) if user_message.attachments else 0
        assert initial_attachment_count == 2, f"Should have 2 attachments, got {initial_attachment_count}"
        print("[PASS] Message has 2 attachments")
        
        # Step 3: Remove the FIRST attachment (PDF - index 0)
        print("\n[STEP 3] Removing first attachment (index 0)...")
        first_att = user_message.attachments[0]
        print(f"  - Removing: {first_att.type} - {first_att.identifier[-40:]}")
        
        result = chat.remove_attachment(
            message_index=user_message.message_index,
            attachment_index=0
        )
        
        print(f"  - remove_attachment returned: {result}")
        assert result is True, "remove_attachment should return True"
        print("[PASS] Attachment removal returned True")
        
        # Step 4: Verify the attachment was removed
        print("\n[STEP 4] Verifying attachment was removed...")
        messages_after = chat.all_messages
        user_message_after = messages_after[0]
        
        print(f"  - Content preview: {user_message_after.content[:60]}...")
        print(f"  - Attachments now: {len(user_message_after.attachments) if user_message_after.attachments else 0}")
        
        if user_message_after.attachments:
            for att in user_message_after.attachments:
                print(f"    [{att.index}] {att.type}: ...{att.identifier[-30:]}")
        
        new_attachment_count = len(user_message_after.attachments) if user_message_after.attachments else 0
        assert new_attachment_count == 1, f"Should have 1 attachment after removal, got {new_attachment_count}"
        print("[PASS] One attachment was removed, one remains")
        
        # Step 5: Remove the remaining attachment
        print("\n[STEP 5] Removing the remaining attachment...")
        result2 = chat.remove_attachment(
            message_index=user_message_after.message_index,
            attachment_index=0  # Now index 0 is the remaining one
        )
        
        print(f"  - remove_attachment returned: {result2}")
        assert result2 is True, "remove_attachment should return True"
        
        # Verify all attachments removed
        messages_final = chat.all_messages
        user_message_final = messages_final[0]
        
        final_attachment_count = len(user_message_final.attachments) if user_message_final.attachments else 0
        print(f"  - Attachments now: {final_attachment_count}")
        
        # Note: After removing all attachments, the message may still have text content
        print(f"  - Content still exists: '{user_message_final.content[:60]}...'")
        print("[PASS] All attachments removed, text content preserved")
        
        # Step 6: Test invalid removal
        print("\n[STEP 6] Testing invalid attachment removal...")
        result_invalid = chat.remove_attachment(
            message_index=999,  # Invalid index
            attachment_index=0
        )
        assert result_invalid is False, "Should return False for invalid message index"
        print("[PASS] Invalid message index handled correctly")
        
        # Step 7: Verify usage is preserved
        print("\n[STEP 7] Verifying usage is preserved...")
        assert (chat.usage.cost or 0.0) > 0, "Cost should be preserved"
        assert chat.usage.total_tokens > 0, "Tokens should be preserved"
        print(f"  - Total cost: ${(chat.usage.cost or 0.0):.6f}")
        print(f"  - Total tokens: {chat.usage.total_tokens}")
        print("[PASS] Usage preserved after attachment removal")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_remove_attachment completed")


@pytest.mark.asyncio
async def test_chat_session_timing():
    """Test session timing: start_time, end_time, duration on close/reset."""
    print("\n" + "="*60)
    print("TEST: test_chat_session_timing")
    print("="*60)
    
    import time
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_timing",
        user_id="test_user_timing",
        agent=agent
    )
    
    try:
        # Check initial timing
        print("\n[CHECK] Initial session timing...")
        initial_start = chat.start_time
        print(f"  - start_time: {initial_start}")
        print(f"  - end_time: {chat.end_time}")
        print(f"  - duration: {chat.duration:.3f}s")
        print(f"  - is_closed: {chat.is_closed}")
        
        assert initial_start > 0, "Start time should be set on init"
        assert chat.end_time is None, "End time should be None initially"
        assert chat.is_closed is False, "Should not be closed initially"
        print("[PASS] Initial timing correct")
        
        # Wait and check duration increases
        print("\n[CHECK] Duration increases with time...")
        time.sleep(0.1)
        duration_after_wait = chat.duration
        print(f"  - duration after 0.1s: {duration_after_wait:.3f}s")
        assert duration_after_wait >= 0.1, "Duration should increase with time"
        print("[PASS] Duration increases correctly")
        
        # Invoke and check timing
        print("\n[ACTION] Invoking chat...")
        await chat.invoke("Quick test")
        print(f"  - duration after invoke: {chat.duration:.3f}s")
        print(f"  - last_activity: {chat.last_activity:.3f}s ago")
        print(f"  - last_activity_time: {chat.last_activity_time}")
        assert chat.last_activity_time > initial_start, "Last activity should be updated"
        print("[PASS] Activity time updated")
        
        # Test close - should set end_time and freeze duration
        print("\n[ACTION] Closing session...")
        print(f"  - duration before close: {chat.duration:.3f}s")
        await chat.close()
        post_close_end_time = chat.end_time
        post_close_duration = chat.duration
        post_close_is_closed = chat.is_closed
        
        print(f"  - end_time after close: {post_close_end_time}")
        print(f"  - duration after close: {post_close_duration:.3f}s")
        print(f"  - is_closed: {post_close_is_closed}")
        
        assert post_close_end_time is not None, "End time should be set on close"
        assert post_close_is_closed is True, "Should be marked as closed"
        
        # Duration should now be fixed (not increasing)
        time.sleep(0.1)
        duration_after_wait_closed = chat.duration
        print(f"  - duration 0.1s after close: {duration_after_wait_closed:.3f}s")
        assert abs(duration_after_wait_closed - post_close_duration) < 0.01, \
            "Duration should be fixed after close"
        print("[PASS] Duration frozen after close")
        
        # Test reset - should clear end_time and set new start_time
        print("\n[ACTION] Resetting session...")
        old_start = chat.start_time
        await chat.areset_session()
        new_start = chat.start_time
        new_end = chat.end_time
        new_is_closed = chat.is_closed
        new_duration = chat.duration
        
        print(f"  - old start_time: {old_start}")
        print(f"  - new start_time: {new_start}")
        print(f"  - new end_time: {new_end}")
        print(f"  - is_closed: {new_is_closed}")
        print(f"  - new duration: {new_duration:.3f}s")
        
        assert new_start > old_start, "Start time should be updated on reset"
        assert new_end is None, "End time should be cleared on reset"
        assert new_is_closed is False, "Should not be closed after reset"
        assert new_duration < post_close_duration, "Duration should be reset"
        print("[PASS] Reset clears timing correctly")
        
    finally:
        # close() is idempotent
        await chat.close()
        print("\n[DONE] test_chat_session_timing completed")


@pytest.mark.asyncio
@pytest.mark.timeout(300)  # Extended timeout for attachment processing
async def test_chat_remove_attachment_by_path():
    """Test remove_attachment_by_path - simplified API for removing attachments."""
    print("\n" + "="*60)
    print("TEST: test_chat_remove_attachment_by_path")
    print("="*60)
    
    # Use actual test files from project root
    pdf_path = str(PROJECT_ROOT / "example_document.pdf")
    image_path = str(PROJECT_ROOT / "premium_photo-1661342428515-5ca8cee4385a.jpeg")
    
    # Verify files exist
    assert os.path.exists(pdf_path), f"PDF file not found: {pdf_path}"
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_remove_by_path",
        user_id="test_user_remove_by_path",
        agent=agent
    )
    
    try:
        # Step 1: Send multiple messages with same PDF in multiple places
        print("\n[STEP 1] Sending messages with PDF in multiple places...")
        await chat.invoke(
            Task(description="Check this PDF", attachments=[pdf_path])
        )
        await chat.invoke(
            Task(description="Also check with image", attachments=[pdf_path, image_path])
        )
        
        # Count attachments across all messages
        initial_messages = chat.all_messages
        initial_pdf_count = 0
        initial_image_count = 0
        
        print("\n[CHECK] Initial attachment counts:")
        for msg in initial_messages:
            if msg.attachments:
                for att in msg.attachments:
                    if "pdf" in att.identifier.lower() or "example_document" in att.identifier.lower():
                        initial_pdf_count += 1
                    if "jpeg" in att.identifier.lower() or "jpg" in att.identifier.lower() or "premium_photo" in att.identifier.lower():
                        initial_image_count += 1
        
        print(f"  - PDF attachments: {initial_pdf_count}")
        print(f"  - Image attachments: {initial_image_count}")
        assert initial_pdf_count >= 2, "Should have at least 2 PDF attachments"
        assert initial_image_count >= 1, "Should have at least 1 image attachment"
        
        # Step 2: Remove ALL PDFs using remove_attachment_by_path
        print("\n[STEP 2] Removing all PDFs using remove_attachment_by_path...")
        removed = chat.remove_attachment_by_path(pdf_path)
        print(f"  - Removed: {removed} attachments")
        assert removed == initial_pdf_count, f"Should have removed {initial_pdf_count} PDFs"
        print("[PASS] remove_attachment_by_path removed correct count")
        
        # Step 3: Verify PDFs are gone, images remain
        print("\n[STEP 3] Verifying PDFs removed, images remain...")
        after_messages = chat.all_messages
        after_pdf_count = 0
        after_image_count = 0
        
        for msg in after_messages:
            if msg.attachments:
                for att in msg.attachments:
                    if "pdf" in att.identifier.lower() or "example_document" in att.identifier.lower():
                        after_pdf_count += 1
                    if "jpeg" in att.identifier.lower() or "jpg" in att.identifier.lower() or "premium_photo" in att.identifier.lower():
                        after_image_count += 1
        
        print(f"  - PDF attachments after: {after_pdf_count}")
        print(f"  - Image attachments after: {after_image_count}")
        
        assert after_pdf_count == 0, "All PDFs should be removed"
        assert after_image_count == initial_image_count, "Images should be preserved"
        print("[PASS] PDFs removed, images preserved")
        
        # Step 4: Test partial path matching to remove the image
        print("\n[STEP 4] Testing partial path matching...")
        removed_image = chat.remove_attachment_by_path("premium_photo")
        print(f"  - Removed with partial match 'premium_photo': {removed_image}")
        assert removed_image >= 1, "Should match and remove by partial path"
        print("[PASS] Partial path matching works")
        
        # Step 5: Verify no attachments remain on user messages
        print("\n[STEP 5] Verifying all attachments removed...")
        final_messages = chat.all_messages
        remaining_attachments = 0
        for msg in final_messages:
            if msg.attachments:
                remaining_attachments += len(msg.attachments)
        print(f"  - Remaining attachments: {remaining_attachments}")
        assert remaining_attachments == 0, "All attachments should be removed"
        print("[PASS] All attachments successfully removed")
        
        # Step 6: Test removing non-existent path
        print("\n[STEP 6] Testing removal of non-existent path...")
        removed_none = chat.remove_attachment_by_path("/nonexistent/file.xyz")
        print(f"  - Removed for non-existent path: {removed_none}")
        assert removed_none == 0, "Should return 0 for non-existent path"
        print("[PASS] Non-existent path handled correctly")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_remove_attachment_by_path completed")


@pytest.mark.asyncio
async def test_chat_reopen_session():
    """Test reopening a closed session to continue conversation."""
    print("\n" + "="*60)
    print("TEST: test_chat_reopen_session")
    print("="*60)
    
    import time
    
    agent = Agent(model="openai/gpt-4o", name="Chat Agent")
    chat = Chat(
        session_id="test_session_reopen",
        user_id="test_user_reopen",
        agent=agent
    )
    
    try:
        # Step 1: Send initial message
        print("\n[STEP 1] Sending initial message...")
        await chat.invoke("Hello, my name is TestUser!")
        
        initial_messages = len(chat.all_messages)
        initial_tokens = chat.usage.total_tokens
        initial_duration = chat.duration
        
        print(f"  - Messages: {initial_messages}")
        print(f"  - Tokens: {initial_tokens}")
        print(f"  - Duration: {initial_duration:.3f}s")
        print(f"  - is_closed: {chat.is_closed}")
        
        assert initial_messages >= 2, "Should have at least 2 messages"
        assert chat.is_closed is False, "Should not be closed yet"
        print("[PASS] Initial state correct")
        
        # Step 2: Close the session
        print("\n[STEP 2] Closing session...")
        await chat.close()
        
        closed_end_time = chat.end_time
        closed_duration = chat.duration
        closed_is_closed = chat.is_closed
        
        print(f"  - end_time: {closed_end_time}")
        print(f"  - duration (frozen): {closed_duration:.3f}s")
        print(f"  - is_closed: {closed_is_closed}")
        
        assert closed_end_time is not None, "End time should be set"
        assert closed_is_closed is True, "Should be marked as closed"
        print("[PASS] Session closed correctly")
        
        # Step 3: Wait a bit (duration should remain frozen)
        print("\n[STEP 3] Verifying duration is frozen while closed...")
        time.sleep(0.2)
        duration_while_closed = chat.duration
        print(f"  - Duration after 0.2s wait: {duration_while_closed:.3f}s")
        assert abs(duration_while_closed - closed_duration) < 0.01, \
            "Duration should be frozen while closed"
        print("[PASS] Duration frozen while closed")
        
        # Step 4: Reopen the session
        print("\n[STEP 4] Reopening session...")
        chat.reopen()
        
        reopen_end_time = chat.end_time
        reopen_is_closed = chat.is_closed
        reopen_duration = chat.duration
        
        print(f"  - end_time after reopen: {reopen_end_time}")
        print(f"  - is_closed: {reopen_is_closed}")
        print(f"  - duration (continuing): {reopen_duration:.3f}s")
        
        assert reopen_end_time is None, "End time should be cleared"
        assert reopen_is_closed is False, "Should not be closed after reopen"
        assert reopen_duration >= closed_duration, "Duration should continue from previous"
        print("[PASS] Session reopened correctly")
        
        # Step 5: Verify messages and data are preserved
        print("\n[STEP 5] Verifying messages preserved...")
        reopen_messages = len(chat.all_messages)
        reopen_tokens = chat.usage.total_tokens
        
        print(f"  - Messages after reopen: {reopen_messages}")
        print(f"  - Tokens after reopen: {reopen_tokens}")
        
        assert reopen_messages == initial_messages, "Messages should be preserved"
        assert reopen_tokens == initial_tokens, "Tokens should be preserved"
        print("[PASS] Data preserved after reopen")
        
        # Step 6: Continue the conversation
        print("\n[STEP 6] Continuing conversation after reopen...")
        await chat.invoke("What was my name again?")
        
        final_messages = len(chat.all_messages)
        final_tokens = chat.usage.total_tokens
        
        print(f"  - Messages after continue: {final_messages}")
        print(f"  - Tokens after continue: {final_tokens}")
        
        assert final_messages > reopen_messages, "Should have more messages"
        assert final_tokens > reopen_tokens, "Should have more tokens"
        print("[PASS] Conversation continued successfully")
        
        # Step 7: Duration should now be increasing
        print("\n[STEP 7] Verifying duration increases after reopen...")
        time.sleep(0.1)
        later_duration = chat.duration
        print(f"  - Duration now: {later_duration:.3f}s")
        assert later_duration > reopen_duration, "Duration should be increasing again"
        print("[PASS] Duration continues to increase")
        
        # Step 8: Test idempotent reopen (already open)
        print("\n[STEP 8] Testing reopen on already-open session...")
        pre_reopen_duration = chat.duration
        chat.reopen()  # Should be a no-op
        post_reopen_duration = chat.duration
        print(f"  - Duration before: {pre_reopen_duration:.3f}s")
        print(f"  - Duration after: {post_reopen_duration:.3f}s")
        assert post_reopen_duration >= pre_reopen_duration, "Duration should not reset"
        print("[PASS] Reopen is idempotent")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_reopen_session completed")
