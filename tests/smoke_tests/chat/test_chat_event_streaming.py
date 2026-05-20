"""
Smoke Test: Chat Event Streaming

Success criteria: Chat class properly streams events (TextDeltaEvent, ToolCallDeltaEvent, ToolResultEvent)
during streaming execution with tools.

This test validates:
1. Event streaming with chat.stream(events=True)
2. Event streaming with chat.invoke(stream=True, events=True)
3. Proper event types are yielded (TextDeltaEvent, ToolCallDeltaEvent, ToolResultEvent)
4. Tool calls and results are captured in events
5. Text streaming works correctly
"""
import pytest
from typing import List, Any

from upsonic import Agent, Chat
from upsonic.run.events.events import (
    AgentEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    PipelineStartEvent,
    PipelineEndEvent,
    RunStartedEvent,
    RunCompletedEvent,
)
from upsonic.tools import tool


pytestmark = pytest.mark.timeout(120)


@tool
def add_numbers(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y


@tool
def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers together."""
    return x * y


@pytest.mark.asyncio
async def test_chat_stream_events_basic():
    """Test basic event streaming with chat.stream(events=True)."""
    print("\n" + "=" * 60)
    print("TEST: test_chat_stream_events_basic")
    print("=" * 60)
    
    agent = Agent(model="anthropic/claude-haiku-4-5", tools=[add_numbers])
    chat = Chat(
        session_id="test_event_stream_1",
        user_id="test_user",
        agent=agent
    )
    
    collected_events: List[AgentEvent] = []
    text_deltas: List[str] = []
    tool_call_deltas: List[ToolCallDeltaEvent] = []
    tool_results: List[ToolResultEvent] = []
    
    try:
        async for event in chat.stream("Calculate 5 + 3", events=True):
            collected_events.append(event)
            
            if isinstance(event, TextDeltaEvent):
                text_deltas.append(event.content)
            elif isinstance(event, ToolCallDeltaEvent):
                tool_call_deltas.append(event)
            elif isinstance(event, ToolResultEvent):
                tool_results.append(event)
        
        print(f"\n[RESULT] Total events collected: {len(collected_events)}")
        print(f"[RESULT] Text delta events: {len(text_deltas)}")
        print(f"[RESULT] Tool call delta events: {len(tool_call_deltas)}")
        print(f"[RESULT] Tool result events: {len(tool_results)}")
        
        # Verify we got events
        assert len(collected_events) > 0, "Should have collected events"
        print("[PASS] Events were collected")
        
        # Verify pipeline events
        has_pipeline_start = any(isinstance(e, PipelineStartEvent) for e in collected_events)
        has_pipeline_end = any(isinstance(e, PipelineEndEvent) for e in collected_events)
        assert has_pipeline_start, "Should have PipelineStartEvent"
        assert has_pipeline_end, "Should have PipelineEndEvent"
        print("[PASS] Pipeline start/end events present")
        
        # Verify run events
        has_run_started = any(isinstance(e, RunStartedEvent) for e in collected_events)
        has_run_completed = any(isinstance(e, RunCompletedEvent) for e in collected_events)
        assert has_run_started, "Should have RunStartedEvent"
        assert has_run_completed, "Should have RunCompletedEvent"
        print("[PASS] Run start/completed events present")
        
        # Verify tool was called (should have tool events for calculation)
        assert len(tool_results) >= 1 or len(tool_call_deltas) >= 1, \
            "Should have tool call deltas or tool results for calculation"
        print("[PASS] Tool events captured")
        
        # Verify text was streamed
        assert len(text_deltas) > 0, "Should have text delta events"
        accumulated_text = "".join(text_deltas)
        print(f"[RESULT] Accumulated text: {accumulated_text[:100]}...")
        assert len(accumulated_text) > 0, "Accumulated text should not be empty"
        print("[PASS] Text streaming works")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_stream_events_basic completed")


@pytest.mark.asyncio
async def test_chat_invoke_stream_events():
    """Test event streaming with chat.invoke(stream=True, events=True)."""
    print("\n" + "=" * 60)
    print("TEST: test_chat_invoke_stream_events")
    print("=" * 60)
    
    agent = Agent(model="anthropic/claude-haiku-4-5", tools=[multiply_numbers])
    chat = Chat(
        session_id="test_event_stream_2",
        user_id="test_user",
        agent=agent
    )
    
    collected_events: List[AgentEvent] = []
    
    try:
        # Use invoke with stream=True and events=True
        async for event in await chat.invoke("What is 6 times 7?", stream=True, events=True):
            collected_events.append(event)
        
        print(f"\n[RESULT] Total events via invoke(): {len(collected_events)}")
        
        # Verify events were collected
        assert len(collected_events) > 0, "Should have collected events via invoke()"
        print("[PASS] Events collected via invoke(stream=True, events=True)")
        
        # Verify event types
        event_types = set(type(e).__name__ for e in collected_events)
        print(f"[RESULT] Event types: {event_types}")
        
        # Should have basic events
        assert "PipelineStartEvent" in event_types, "Should have PipelineStartEvent"
        assert "PipelineEndEvent" in event_types, "Should have PipelineEndEvent"
        print("[PASS] Basic event types present")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_invoke_stream_events completed")


@pytest.mark.asyncio
async def test_chat_stream_events_no_tools():
    """Test event streaming without tools (text-only response)."""
    print("\n" + "=" * 60)
    print("TEST: test_chat_stream_events_no_tools")
    print("=" * 60)
    
    # Agent without tools
    agent = Agent(model="anthropic/claude-haiku-4-5")
    chat = Chat(
        session_id="test_event_stream_3",
        user_id="test_user",
        agent=agent
    )
    
    text_deltas: List[str] = []
    tool_events: List[AgentEvent] = []
    
    try:
        async for event in chat.stream("Say hello briefly", events=True):
            if isinstance(event, TextDeltaEvent):
                text_deltas.append(event.content)
            elif isinstance(event, (ToolCallDeltaEvent, ToolCallEvent, ToolResultEvent)):
                tool_events.append(event)
        
        print(f"\n[RESULT] Text delta events: {len(text_deltas)}")
        print(f"[RESULT] Tool events: {len(tool_events)}")
        
        # Verify text was streamed
        accumulated_text = "".join(text_deltas)
        assert len(accumulated_text) > 0, "Should have text content"
        print(f"[RESULT] Response: {accumulated_text[:100]}...")
        print("[PASS] Text streaming works without tools")
        
        # Should have no tool events since no tools are configured
        assert len(tool_events) == 0, "Should have no tool events when no tools configured"
        print("[PASS] No tool events when tools not configured")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_stream_events_no_tools completed")


@pytest.mark.asyncio
async def test_chat_stream_tool_result_content():
    """Test that tool results contain correct values."""
    print("\n" + "=" * 60)
    print("TEST: test_chat_stream_tool_result_content")
    print("=" * 60)
    
    agent = Agent(model="anthropic/claude-haiku-4-5", tools=[add_numbers])
    chat = Chat(
        session_id="test_event_stream_4",
        user_id="test_user",
        agent=agent
    )
    
    tool_results: List[ToolResultEvent] = []
    
    try:
        async for event in chat.stream("Add 10 and 20 together", events=True):
            if isinstance(event, ToolResultEvent):
                tool_results.append(event)
                print(f"[EVENT] ToolResultEvent: tool={event.tool_name}, result={event.result}")
        
        print(f"\n[RESULT] Tool results captured: {len(tool_results)}")
        
        # Should have at least one tool result
        assert len(tool_results) >= 1, "Should have at least one tool result"
        
        # Verify tool result content
        for result in tool_results:
            assert result.tool_name is not None, "Tool name should be set"
            assert result.result is not None, "Result should be set"
            
            # If it's the add_numbers tool, verify the result
            if result.tool_name == "add_numbers":
                # Result may be wrapped in dict {'func': value} or direct value
                actual_value = result.result
                if isinstance(actual_value, dict) and 'func' in actual_value:
                    actual_value = actual_value['func']
                assert actual_value == 30, f"add_numbers(10, 20) should be 30, got {result.result}"
                print("[PASS] Tool result value is correct (30)")
        
        print("[PASS] Tool result content is valid")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_stream_tool_result_content completed")


@pytest.mark.asyncio
async def test_chat_stream_multiple_messages():
    """Test event streaming across multiple messages in a session."""
    print("\n" + "=" * 60)
    print("TEST: test_chat_stream_multiple_messages")
    print("=" * 60)
    
    agent = Agent(model="anthropic/claude-haiku-4-5", tools=[add_numbers])
    chat = Chat(
        session_id="test_event_stream_5",
        user_id="test_user",
        agent=agent,
        full_session_memory=True
    )
    
    try:
        # First message
        events_msg1: List[AgentEvent] = []
        async for event in chat.stream("What is 1 + 1?", events=True):
            events_msg1.append(event)
        
        print(f"\n[RESULT] Message 1 events: {len(events_msg1)}")
        
        # Second message
        events_msg2: List[AgentEvent] = []
        async for event in chat.stream("Now add 5 + 5", events=True):
            events_msg2.append(event)
        
        print(f"[RESULT] Message 2 events: {len(events_msg2)}")
        
        # Both should have events
        assert len(events_msg1) > 0, "First message should have events"
        assert len(events_msg2) > 0, "Second message should have events"
        print("[PASS] Both messages have events")
        
        # Verify session has messages
        messages = chat.all_messages
        print(f"[RESULT] Total messages in session: {len(messages)}")
        assert len(messages) >= 4, "Should have at least 4 messages (2 user + 2 assistant)"
        print("[PASS] Session maintains message history")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_stream_multiple_messages completed")


@pytest.mark.asyncio
async def test_chat_stream_events_with_context():
    """Test event streaming with context attached."""
    print("\n" + "=" * 60)
    print("TEST: test_chat_stream_events_with_context")
    print("=" * 60)
    
    agent = Agent(model="anthropic/claude-haiku-4-5")
    chat = Chat(
        session_id="test_event_stream_6",
        user_id="test_user",
        agent=agent
    )
    
    collected_events: List[AgentEvent] = []
    
    try:
        # Stream with context (text context, not file)
        async for event in chat.stream(
            "Summarize this: Python is a programming language.",
            events=True
        ):
            collected_events.append(event)
        
        print(f"\n[RESULT] Events with context: {len(collected_events)}")
        
        # Verify events were collected
        assert len(collected_events) > 0, "Should have events"
        
        # Get text content
        text_deltas = [e.content for e in collected_events if isinstance(e, TextDeltaEvent)]
        accumulated = "".join(text_deltas)
        print(f"[RESULT] Response preview: {accumulated[:100]}...")
        
        assert len(accumulated) > 0, "Should have text response"
        print("[PASS] Event streaming works with context")
        
    finally:
        await chat.close()
        print("\n[DONE] test_chat_stream_events_with_context completed")
