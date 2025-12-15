"""
Comprehensive Test Suite for Agent Event Streaming

This test file validates the comprehensive event streaming feature across
all pipeline steps. Run with: pytest tests/test_stream_events.py -v

Requirements:
- Set OPENAI_API_KEY environment variable
- pip install pytest pytest-asyncio
"""

import asyncio
import os
import pytest
from typing import List, Dict, Any

from upsonic import Agent, Task
from upsonic.tools import tool

# Import all event classes for testing
from upsonic.agent.events import (
    # Base
    AgentEvent,
    
    # Pipeline events
    PipelineStartEvent,
    PipelineEndEvent,
    
    # Step events
    StepStartEvent,
    StepEndEvent,
    
    # Step-specific events
    AgentInitializedEvent,
    CacheCheckEvent,
    CacheHitEvent,
    CacheMissEvent,
    PolicyCheckEvent,
    ModelSelectedEvent,
    ToolsConfiguredEvent,
    MessagesBuiltEvent,
    ModelRequestStartEvent,
    ModelResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
    ExternalToolPauseEvent,
    ReflectionEvent,
    MemoryUpdateEvent,
    ReliabilityEvent,
    CacheStoredEvent,
    ExecutionCompleteEvent,
    
    # LLM stream events (as Agent events)
    TextDeltaEvent,
    TextCompleteEvent,
    ThinkingDeltaEvent,
    ToolCallDeltaEvent,
    FinalOutputEvent,
)


# =============================================================================
# Test Tools
# =============================================================================

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 25°C"


@tool
def search_database(query: str) -> List[Dict[str, Any]]:
    """Search the database for matching records."""
    return [
        {"id": 1, "name": "Result 1", "match": query},
        {"id": 2, "name": "Result 2", "match": query},
    ]


# =============================================================================
# Helper Functions
# =============================================================================

def collect_events_by_type(events: List[AgentEvent]) -> Dict[str, List[AgentEvent]]:
    """Organize events by their type name."""
    by_type: Dict[str, List[AgentEvent]] = {}
    for event in events:
        type_name = type(event).__name__
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(event)
    return by_type


def print_event_summary(events: List[AgentEvent]) -> None:
    """Print a summary of collected events."""
    by_type = collect_events_by_type(events)
    print("\n" + "=" * 60)
    print("EVENT SUMMARY")
    print("=" * 60)
    for type_name, type_events in sorted(by_type.items()):
        print(f"  {type_name}: {len(type_events)}")
    print(f"  TOTAL: {len(events)}")
    print("=" * 60)


# =============================================================================
# Test: Basic Text Streaming
# =============================================================================

@pytest.mark.asyncio
async def test_basic_text_streaming():
    """Test basic text streaming with event collection."""
    print("\n" + "=" * 60)
    print("TEST: Basic Text Streaming")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Say 'Hello World' and nothing else.")
    
    collected_events: List[AgentEvent] = []
    accumulated_text = ""
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            collected_events.append(event)
            
            # Print event info
            if isinstance(event, PipelineStartEvent):
                print(f"[PIPELINE START] Steps: {event.total_steps}")
            elif isinstance(event, StepStartEvent):
                print(f"  [STEP START] {event.step_name} ({event.step_index + 1}/{event.total_steps})")
            elif isinstance(event, StepEndEvent):
                print(f"  [STEP END] {event.step_name} - {event.status} ({event.execution_time:.3f}s)")
            elif isinstance(event, TextDeltaEvent):
                accumulated_text += event.content
                print(f"[TEXT] {event.content}", end="", flush=True)
            elif isinstance(event, TextCompleteEvent):
                print(f"\n[TEXT COMPLETE] Full: {event.content[:50]}...")
            elif isinstance(event, FinalOutputEvent):
                print(f"[FINAL OUTPUT] Type: {event.output_type}")
            elif isinstance(event, PipelineEndEvent):
                print(f"[PIPELINE END] Status: {event.status}, Duration: {event.total_duration:.2f}s")
    
    print_event_summary(collected_events)
    
    # Assertions
    by_type = collect_events_by_type(collected_events)
    
    assert "PipelineStartEvent" in by_type, "Should have PipelineStartEvent"
    assert "PipelineEndEvent" in by_type, "Should have PipelineEndEvent"
    assert "StepStartEvent" in by_type, "Should have StepStartEvent"
    assert "StepEndEvent" in by_type, "Should have StepEndEvent"
    assert "TextDeltaEvent" in by_type, "Should have TextDeltaEvent"
    assert "FinalOutputEvent" in by_type, "Should have FinalOutputEvent"
    
    # Verify pipeline events
    pipeline_start = by_type["PipelineStartEvent"][0]
    assert pipeline_start.total_steps > 0
    assert pipeline_start.is_streaming is True
    
    pipeline_end = by_type["PipelineEndEvent"][0]
    assert pipeline_end.status == "success"
    assert pipeline_end.total_duration > 0
    
    print(f"\n✅ Test passed! Collected {len(collected_events)} events")
    print(f"   Accumulated text: {accumulated_text[:100]}...")


# =============================================================================
# Test: Tool Calling Events
# =============================================================================

@pytest.mark.asyncio
async def test_tool_calling_events():
    """Test that tool call events are properly emitted."""
    print("\n" + "=" * 60)
    print("TEST: Tool Calling Events")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", tools=[add_numbers, multiply_numbers])
    task = Task("What is 5 + 3? Use the add_numbers tool.")
    
    collected_events: List[AgentEvent] = []
    tool_calls_seen: List[ToolCallEvent] = []
    tool_results_seen: List[ToolResultEvent] = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            collected_events.append(event)
            
            if isinstance(event, ToolCallEvent):
                tool_calls_seen.append(event)
                print(f"[TOOL CALL] {event.tool_name}({event.tool_args}) - ID: {event.tool_call_id}")
            elif isinstance(event, ToolResultEvent):
                tool_results_seen.append(event)
                print(f"[TOOL RESULT] {event.tool_name} -> {event.result_preview}")
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
            elif isinstance(event, ModelRequestStartEvent):
                print(f"\n[MODEL REQUEST] Model: {event.model_name}, Has tools: {event.has_tools}")
    
    print_event_summary(collected_events)
    
    # Assertions
    by_type = collect_events_by_type(collected_events)
    
    assert "ToolCallEvent" in by_type, "Should have ToolCallEvent"
    assert "ToolResultEvent" in by_type, "Should have ToolResultEvent"
    
    # Verify tool call details
    assert len(tool_calls_seen) > 0, "Should have at least one tool call"
    tool_call = tool_calls_seen[0]
    assert tool_call.tool_name == "add_numbers"
    assert "a" in tool_call.tool_args
    assert "b" in tool_call.tool_args
    
    # Verify tool result
    assert len(tool_results_seen) > 0, "Should have at least one tool result"
    tool_result = tool_results_seen[0]
    assert tool_result.tool_name == "add_numbers"
    # Tool results are wrapped in {'func': value}
    expected_result = tool_result.result
    if isinstance(expected_result, dict) and 'func' in expected_result:
        assert expected_result['func'] == 8  # 5 + 3
    else:
        assert expected_result == 8
    
    print(f"\n✅ Test passed! Tool calls: {len(tool_calls_seen)}, Results: {len(tool_results_seen)}")


# =============================================================================
# Test: Multiple Tool Calls
# =============================================================================

@pytest.mark.asyncio
async def test_multiple_tool_calls():
    """Test multiple sequential tool calls."""
    print("\n" + "=" * 60)
    print("TEST: Multiple Tool Calls")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", tools=[add_numbers, multiply_numbers])
    task = Task("First add 10 + 5, then multiply the result by 2. Show your work.")
    
    collected_events: List[AgentEvent] = []
    tool_calls: List[ToolCallEvent] = []
    tool_results: List[ToolResultEvent] = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            collected_events.append(event)
            
            if isinstance(event, ToolCallEvent):
                tool_calls.append(event)
                print(f"[TOOL CALL #{len(tool_calls)}] {event.tool_name}({event.tool_args})")
            elif isinstance(event, ToolResultEvent):
                tool_results.append(event)
                print(f"[TOOL RESULT #{len(tool_results)}] {event.tool_name} -> {event.result}")
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
    
    print_event_summary(collected_events)
    
    # Should have at least 2 tool calls (add then multiply)
    assert len(tool_calls) >= 2, f"Expected at least 2 tool calls, got {len(tool_calls)}"
    assert len(tool_results) >= 2, f"Expected at least 2 tool results, got {len(tool_results)}"
    
    print(f"\n✅ Test passed! Total tool calls: {len(tool_calls)}")


# =============================================================================
# Test: Cache Events
# =============================================================================

@pytest.mark.asyncio
async def test_cache_events():
    """Test cache-related events (miss on first, hit on second)."""
    print("\n" + "=" * 60)
    print("TEST: Cache Events")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini")
    
    # First request - should be cache miss
    task1 = Task("What is 2+2?", enable_cache=True, cache_method="vector_search")
    
    cache_events_1: List[AgentEvent] = []
    
    print("\n--- First request (should be cache miss) ---")
    async with agent.stream(task1) as result:
        async for event in result.stream_events():
            if isinstance(event, (CacheCheckEvent, CacheHitEvent, CacheMissEvent, CacheStoredEvent)):
                cache_events_1.append(event)
                print(f"[CACHE] {type(event).__name__}: {event}")
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
    
    # Second request with same input - should be cache hit
    task2 = Task("What is 2+2?", enable_cache=True, cache_method="vector_search")
    
    cache_events_2: List[AgentEvent] = []
    
    print("\n\n--- Second request (should be cache hit) ---")
    async with agent.stream(task2) as result:
        async for event in result.stream_events():
            if isinstance(event, (CacheCheckEvent, CacheHitEvent, CacheMissEvent, CacheStoredEvent)):
                cache_events_2.append(event)
                print(f"[CACHE] {type(event).__name__}")
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
    
    # Assertions for first request (cache miss)
    by_type_1 = collect_events_by_type(cache_events_1)
    assert "CacheMissEvent" in by_type_1, "First request should have CacheMissEvent"
    assert "CacheStoredEvent" in by_type_1, "First request should store in cache"
    
    # Assertions for second request (cache hit)
    by_type_2 = collect_events_by_type(cache_events_2)
    assert "CacheHitEvent" in by_type_2, "Second request should have CacheHitEvent"
    
    print(f"\n\n✅ Test passed! Cache miss then hit confirmed")


# =============================================================================
# Test: Step Events Order
# =============================================================================

@pytest.mark.asyncio
async def test_step_events_order():
    """Test that step events occur in correct order."""
    print("\n" + "=" * 60)
    print("TEST: Step Events Order")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Hi")
    
    step_events: List[AgentEvent] = []
    step_order: List[str] = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            if isinstance(event, StepStartEvent):
                step_events.append(event)
                step_order.append(f"START:{event.step_name}")
                print(f"[{event.step_index + 1}] START: {event.step_name}")
            elif isinstance(event, StepEndEvent):
                step_events.append(event)
                step_order.append(f"END:{event.step_name}")
                print(f"    END: {event.step_name} ({event.status})")
    
    print(f"\nStep order: {len(step_order)} events")
    
    # Verify each START has a matching END
    starts = [s for s in step_order if s.startswith("START:")]
    ends = [s for s in step_order if s.startswith("END:")]
    
    assert len(starts) == len(ends), "Each START should have an END"
    
    # Verify order: START should come before END for each step
    for start in starts:
        step_name = start.replace("START:", "")
        end = f"END:{step_name}"
        start_idx = step_order.index(start)
        end_idx = step_order.index(end)
        assert start_idx < end_idx, f"{step_name} START should come before END"
    
    print(f"\n✅ Test passed! {len(starts)} steps executed in correct order")


# =============================================================================
# Test: All Event Types Present
# =============================================================================

@pytest.mark.asyncio
async def test_all_event_types_present():
    """Test that all expected event types are emitted in a complex scenario."""
    print("\n" + "=" * 60)
    print("TEST: All Event Types Present")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", tools=[get_weather])
    task = Task("What's the weather in Tokyo?")
    
    collected_events: List[AgentEvent] = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            collected_events.append(event)
    
    by_type = collect_events_by_type(collected_events)
    print_event_summary(collected_events)
    
    # Required events for any execution
    required_events = [
        "PipelineStartEvent",
        "PipelineEndEvent",
        "StepStartEvent",
        "StepEndEvent",
        "AgentInitializedEvent",
        "ModelSelectedEvent",
        "ToolsConfiguredEvent",
        "MessagesBuiltEvent",
        "ModelRequestStartEvent",
        "FinalOutputEvent",
        "ExecutionCompleteEvent",
    ]
    
    # Tool-specific events
    tool_events = [
        "ToolCallEvent",
        "ToolResultEvent",
    ]
    
    missing = []
    for event_type in required_events:
        if event_type not in by_type:
            missing.append(event_type)
        else:
            print(f"  ✓ {event_type}")
    
    # Check tool events
    has_tool_call = "ToolCallEvent" in by_type
    has_tool_result = "ToolResultEvent" in by_type
    
    if has_tool_call:
        print(f"  ✓ ToolCallEvent")
    if has_tool_result:
        print(f"  ✓ ToolResultEvent")
    
    if missing:
        print(f"\n❌ Missing events: {missing}")
        pytest.fail(f"Missing required events: {missing}")
    
    print(f"\n✅ Test passed! All {len(required_events)} required event types present")


# =============================================================================
# Test: Event Attributes
# =============================================================================

@pytest.mark.asyncio
async def test_event_attributes():
    """Test that events have proper attributes set."""
    print("\n" + "=" * 60)
    print("TEST: Event Attributes")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", tools=[add_numbers])
    task = Task("Add 100 and 200 using the tool")
    
    collected_events: List[AgentEvent] = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            collected_events.append(event)
    
    by_type = collect_events_by_type(collected_events)
    
    # Test PipelineStartEvent attributes
    if "PipelineStartEvent" in by_type:
        event = by_type["PipelineStartEvent"][0]
        assert hasattr(event, 'total_steps')
        assert hasattr(event, 'is_streaming')
        assert hasattr(event, 'timestamp')
        assert hasattr(event, 'event_id')
        assert event.is_streaming is True
        print(f"  ✓ PipelineStartEvent: total_steps={event.total_steps}, is_streaming={event.is_streaming}")
    
    # Test StepStartEvent attributes
    if "StepStartEvent" in by_type:
        event = by_type["StepStartEvent"][0]
        assert hasattr(event, 'step_name')
        assert hasattr(event, 'step_index')
        assert hasattr(event, 'step_description')
        assert isinstance(event.step_name, str)
        assert isinstance(event.step_index, int)
        print(f"  ✓ StepStartEvent: step_name={event.step_name}, step_index={event.step_index}")
    
    # Test StepEndEvent attributes
    if "StepEndEvent" in by_type:
        event = by_type["StepEndEvent"][0]
        assert hasattr(event, 'step_name')
        assert hasattr(event, 'status')
        assert hasattr(event, 'execution_time')
        assert event.execution_time >= 0
        print(f"  ✓ StepEndEvent: step_name={event.step_name}, status={event.status}")
    
    # Test ModelSelectedEvent attributes
    if "ModelSelectedEvent" in by_type:
        event = by_type["ModelSelectedEvent"][0]
        assert hasattr(event, 'model_name')
        assert "gpt-4o-mini" in event.model_name.lower()
        print(f"  ✓ ModelSelectedEvent: model_name={event.model_name}")
    
    # Test ToolCallEvent attributes
    if "ToolCallEvent" in by_type:
        event = by_type["ToolCallEvent"][0]
        assert hasattr(event, 'tool_name')
        assert hasattr(event, 'tool_call_id')
        assert hasattr(event, 'tool_args')
        assert event.tool_name == "add_numbers"
        assert isinstance(event.tool_args, dict)
        print(f"  ✓ ToolCallEvent: tool_name={event.tool_name}, args={event.tool_args}")
    
    # Test ToolResultEvent attributes
    if "ToolResultEvent" in by_type:
        event = by_type["ToolResultEvent"][0]
        assert hasattr(event, 'tool_name')
        assert hasattr(event, 'tool_call_id')
        assert hasattr(event, 'result')
        # Tool results are wrapped in {'func': value}
        expected_result = event.result
        if isinstance(expected_result, dict) and 'func' in expected_result:
            assert expected_result['func'] == 300  # 100 + 200
        else:
            assert expected_result == 300
        print(f"  ✓ ToolResultEvent: tool_name={event.tool_name}, result={event.result}")
    
    # Test TextDeltaEvent attributes
    if "TextDeltaEvent" in by_type:
        event = by_type["TextDeltaEvent"][0]
        assert hasattr(event, 'content')
        assert isinstance(event.content, str)
        print(f"  ✓ TextDeltaEvent: content='{event.content[:20]}...'")
    
    # Test FinalOutputEvent attributes
    if "FinalOutputEvent" in by_type:
        event = by_type["FinalOutputEvent"][0]
        assert hasattr(event, 'output')
        assert hasattr(event, 'output_type')
        print(f"  ✓ FinalOutputEvent: output_type={event.output_type}")
    
    print(f"\n✅ Test passed! All event attributes verified")


# =============================================================================
# Test: Stream Result Methods
# =============================================================================

@pytest.mark.asyncio
async def test_stream_result_methods():
    """Test StreamRunResult helper methods."""
    print("\n" + "=" * 60)
    print("TEST: Stream Result Methods")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", tools=[add_numbers])
    task = Task("What is 7 + 8? Use the tool.")
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            if isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
        
        print("\n")
        
        # Test helper methods
        all_events = result.get_streaming_events()
        print(f"  get_streaming_events(): {len(all_events)} events")
        
        text_events = result.get_text_events()
        print(f"  get_text_events(): {len(text_events)} events")
        
        tool_events = result.get_tool_events()
        print(f"  get_tool_events(): {len(tool_events)} events")
        
        step_events = result.get_step_events()
        print(f"  get_step_events(): {len(step_events)} events")
        
        pipeline_events = result.get_pipeline_events()
        print(f"  get_pipeline_events(): {len(pipeline_events)} events")
        
        stats = result.get_streaming_stats()
        print(f"  get_streaming_stats(): {stats}")
        
        metrics = result.get_performance_metrics()
        print(f"  get_performance_metrics(): total_duration={metrics.get('total_duration', 'N/A'):.3f}s")
        
        accumulated = result.get_accumulated_text()
        print(f"  get_accumulated_text(): {len(accumulated)} chars")
        
        final = result.get_final_output()
        print(f"  get_final_output(): {str(final)[:50]}...")
    
    # Assertions
    assert len(all_events) > 0, "Should have events"
    assert len(text_events) > 0, "Should have text events"
    assert len(tool_events) > 0, "Should have tool events (add_numbers was called)"
    assert len(step_events) > 0, "Should have step events"
    assert len(pipeline_events) == 2, "Should have start and end pipeline events"
    assert stats["is_complete"] is True
    assert metrics["total_duration"] is not None and metrics["total_duration"] > 0
    assert len(accumulated) > 0
    assert final is not None
    
    print(f"\n✅ Test passed! All StreamRunResult methods work correctly")


# =============================================================================
# Test: Synchronous Streaming
# =============================================================================

def test_sync_streaming():
    """Test synchronous streaming with stream_events_sync()."""
    print("\n" + "=" * 60)
    print("TEST: Synchronous Streaming")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Say hello")
    
    collected_events: List[AgentEvent] = []
    
    result = agent.stream(task)
    for event in result.stream_events_sync():
        collected_events.append(event)
        if isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    print("\n")
    print_event_summary(collected_events)
    
    by_type = collect_events_by_type(collected_events)
    
    assert "PipelineStartEvent" in by_type
    assert "PipelineEndEvent" in by_type
    assert "TextDeltaEvent" in by_type
    
    print(f"\n✅ Test passed! Sync streaming collected {len(collected_events)} events")


# =============================================================================
# Test: Debug Mode Events
# =============================================================================

@pytest.mark.asyncio
async def test_debug_mode():
    """Test that debug mode provides additional visibility."""
    print("\n" + "=" * 60)
    print("TEST: Debug Mode")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", debug=True)
    task = Task("Hi")
    
    collected_events: List[AgentEvent] = []
    
    async with agent.stream(task, debug=True) as result:
        async for event in result.stream_events():
            collected_events.append(event)
    
    by_type = collect_events_by_type(collected_events)
    print_event_summary(collected_events)
    
    # Debug mode should still emit all required events
    assert "PipelineStartEvent" in by_type
    assert "PipelineEndEvent" in by_type
    
    print(f"\n✅ Test passed! Debug mode collected {len(collected_events)} events")


# =============================================================================
# Test: Error Handling in Events
# =============================================================================

@pytest.mark.asyncio
async def test_event_type_property():
    """Test that event_type property works correctly."""
    print("\n" + "=" * 60)
    print("TEST: Event Type Property")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Hi")
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            # Every event should have event_type property
            assert hasattr(event, 'event_type')
            event_type = event.event_type
            assert isinstance(event_type, str)
            assert event_type == type(event).__name__
            print(f"  Event: {event_type}")
    
    print(f"\n✅ Test passed! All events have correct event_type property")


# =============================================================================
# Test: Complex Multi-Tool Scenario
# =============================================================================

@pytest.mark.asyncio
async def test_complex_multi_tool():
    """Test complex scenario with multiple tools and interactions."""
    print("\n" + "=" * 60)
    print("TEST: Complex Multi-Tool Scenario")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", tools=[add_numbers, multiply_numbers, get_weather])
    task = Task(
        "I need to know: 1) What's 15 + 27? 2) What's 6 * 7? 3) What's the weather in Paris? "
        "Answer all three questions."
    )
    
    collected_events: List[AgentEvent] = []
    tool_calls: List[str] = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            collected_events.append(event)
            
            if isinstance(event, ToolCallEvent):
                tool_calls.append(event.tool_name)
                print(f"[TOOL] {event.tool_name}({event.tool_args})")
            elif isinstance(event, ToolResultEvent):
                print(f"  -> {event.result}")
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
    
    print("\n")
    print_event_summary(collected_events)
    
    print(f"\nTools called: {tool_calls}")
    
    # Should have called multiple different tools
    unique_tools = set(tool_calls)
    assert len(unique_tools) >= 2, f"Expected at least 2 different tools, got {unique_tools}"
    
    print(f"\n✅ Test passed! {len(tool_calls)} tool calls, {len(unique_tools)} unique tools")


# =============================================================================
# Main Runner
# =============================================================================

@pytest.mark.asyncio
async def test_memory_with_event_streaming():
    """
    Comprehensive test for event streaming with memory-enabled agent.
    
    This test validates:
    1. Memory is properly initialized and attached to the agent
    2. Events are correctly streamed during execution with memory
    3. Memory-related events (if any) are emitted
    4. Agent correctly uses memory context in responses
    5. Session and user profile memory work correctly
    6. Event streaming continues to work properly with memory overhead
    """
    print("\n" + "=" * 60)
    print("TEST: Memory Integration with Event Streaming")
    print("=" * 60)
    
    from upsonic.storage import InMemoryStorage, Memory
    
    # Create a storage provider for in-memory persistence
    storage = InMemoryStorage(max_sessions=100, max_profiles=50)
    
    # Create a Memory instance with all memory types enabled
    memory = Memory(
        storage=storage,
        session_id="test_session_stream_events_001",
        user_id="test_user_streaming_456",
        full_session_memory=True,      # Remember complete conversations
        summary_memory=True,           # Maintain conversation summaries  
        user_analysis_memory=True,     # Build user profiles
        dynamic_user_profile=True,     # Automatically adapt profile schema
        num_last_messages=10,          # Limit context to last 10 messages
        debug=False                    # Disable debug logging for cleaner test output
    )
    
    # Create an Agent with Memory and tools for comprehensive testing
    personal_assistant = Agent(
        "openai/gpt-4o-mini",
        tools=[add_numbers, multiply_numbers, get_weather],
        memory=memory
    )
    
    # =========================================================================
    # TASK 1: Building User Profile with detailed information
    # =========================================================================
    print("\n--- Task 1: Building User Profile ---")
    
    task1 = Task(
        description="""
        Hello! I want to introduce myself and share some preferences with you.
        
        My name is Alexander Thompson, but please call me 'Alex' in all our future conversations.
        I am a software engineer based in San Francisco, California, working primarily with Python
        and machine learning technologies. I have been coding for about 8 years now.
        
        Here are my key interests and preferences:
        1. I am deeply interested in artificial intelligence, especially deep learning and transformers
        2. I prefer concise, technical explanations over simplified analogies
        3. My favorite programming languages are Python and Rust
        4. I enjoy reading research papers from arXiv, particularly on NLP and computer vision
        5. I typically work late hours (after 8 PM) and appreciate quick, efficient responses
        
        Please acknowledge this information and confirm you've noted my preferences.
        Also, what is 42 + 58? Use the add_numbers tool to calculate this.
        """
    )
    
    task1_events: List[AgentEvent] = []
    task1_text = ""
    
    async with personal_assistant.stream(task1) as result:
        async for event in result.stream_events():
            task1_events.append(event)
            
            if isinstance(event, StepStartEvent):
                print(f"  [STEP] {event.step_name}")
            elif isinstance(event, ToolCallEvent):
                print(f"  [TOOL CALL] {event.tool_name}({event.tool_args})")
            elif isinstance(event, ToolResultEvent):
                print(f"  [TOOL RESULT] {event.tool_name} -> {event.result}")
            elif isinstance(event, TextDeltaEvent):
                task1_text += event.content
                print(event.content, end="", flush=True)
    
    print(f"\n  Total events: {len(task1_events)}")
    
    # Verify Task 1 events
    task1_by_type = collect_events_by_type(task1_events)
    assert "PipelineStartEvent" in task1_by_type, "Task 1 should have PipelineStartEvent"
    assert "PipelineEndEvent" in task1_by_type, "Task 1 should have PipelineEndEvent"
    assert "ToolCallEvent" in task1_by_type, "Task 1 should have ToolCallEvent for add_numbers"
    assert "ToolResultEvent" in task1_by_type, "Task 1 should have ToolResultEvent"
    
    # Verify the tool was called correctly
    tool_call = task1_by_type["ToolCallEvent"][0]
    assert tool_call.tool_name == "add_numbers"
    
    # =========================================================================
    # TASK 2: Testing Memory Recall - Agent should remember user info
    # =========================================================================
    print("\n\n--- Task 2: Testing Memory Recall ---")
    
    task2 = Task(
        description="""
        I have a few questions to test if you remember our previous conversation:
        
        1. What is my name and what do I prefer to be called?
        2. What city am I based in?
        3. What are my main programming interests?
        4. What time do I typically work?
        
        Please answer each question based on what I told you earlier.
        Be specific and reference the exact details I shared.
        """
    )
    
    task2_events: List[AgentEvent] = []
    task2_text = ""
    
    async with personal_assistant.stream(task2) as result:
        async for event in result.stream_events():
            task2_events.append(event)
            
            if isinstance(event, TextDeltaEvent):
                task2_text += event.content
                print(event.content, end="", flush=True)
    
    print(f"\n  Total events: {len(task2_events)}")
    
    # Verify Task 2 basic events
    task2_by_type = collect_events_by_type(task2_events)
    assert "PipelineStartEvent" in task2_by_type
    assert "PipelineEndEvent" in task2_by_type
    assert "TextDeltaEvent" in task2_by_type, "Should have text streaming"
    
    # Check if response mentions remembered information
    response_lower = task2_text.lower()
    assert "alex" in response_lower, "Agent should remember the user's preferred name 'Alex'"
    
    # =========================================================================
    # TASK 3: Using Memory for Personalized Recommendations with Tools
    # =========================================================================
    print("\n\n--- Task 3: Personalized Recommendations with Tools ---")
    
    task3 = Task(
        description="""
        Based on everything you know about me from our conversation, I need your help with two things:
        
        1. First, calculate some numbers for a project I'm working on:
           - Add 1500 + 2500 (my monthly cloud computing budget items)
           - Multiply 40 by 52 (my weekly learning hours times weeks per year)
        
        2. Then, based on my interests in machine learning and my preference for technical content,
           suggest a personalized 3-month learning roadmap for advancing my AI skills.
           Remember that I already have 8 years of coding experience and prefer research papers.
        
        3. Also check the weather in San Francisco (where I'm based) so I know if it's good
           weather for working from a coffee shop today.
        
        Format your response clearly with sections for each part.
        """
    )
    
    task3_events: List[AgentEvent] = []
    task3_text = ""
    tool_calls_seen: List[str] = []
    
    async with personal_assistant.stream(task3) as result:
        async for event in result.stream_events():
            task3_events.append(event)
            
            if isinstance(event, StepStartEvent):
                print(f"  [STEP] {event.step_name}")
            elif isinstance(event, ToolCallEvent):
                tool_calls_seen.append(event.tool_name)
                print(f"  [TOOL CALL] {event.tool_name}({event.tool_args})")
            elif isinstance(event, ToolResultEvent):
                print(f"  [TOOL RESULT] {event.tool_name} -> {event.result}")
            elif isinstance(event, TextDeltaEvent):
                task3_text += event.content
    
    print(f"\n  Response preview: {task3_text[:200]}...")
    print(f"  Total events: {len(task3_events)}")
    print(f"  Tools used: {tool_calls_seen}")
    
    # Verify Task 3 events and tool usage
    task3_by_type = collect_events_by_type(task3_events)
    assert "ToolCallEvent" in task3_by_type, "Task 3 should use tools"
    assert "ToolResultEvent" in task3_by_type, "Task 3 should have tool results"
    
    # Should have called multiple tools
    assert len(tool_calls_seen) >= 2, f"Should call at least 2 tools, got {len(tool_calls_seen)}"
    assert "add_numbers" in tool_calls_seen, "Should use add_numbers"
    
    # =========================================================================
    # Verify Memory Statistics
    # =========================================================================
    print("\n\n--- Memory Statistics ---")
    print(f"  Session ID: {memory.session_id}")
    print(f"  User ID: {memory.user_id}")
    print(f"  Full Session Memory Enabled: {memory.full_session_memory_enabled}")
    print(f"  Summary Memory Enabled: {memory.summary_memory_enabled}")
    print(f"  User Analysis Memory Enabled: {memory.user_analysis_memory_enabled}")
    
    # Verify memory configuration
    assert memory.session_id == "test_session_stream_events_001"
    assert memory.user_id == "test_user_streaming_456"
    assert memory.full_session_memory_enabled is True
    assert memory.summary_memory_enabled is True
    assert memory.user_analysis_memory_enabled is True
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Task 1 Events: {len(task1_events)}")
    print(f"  Task 2 Events: {len(task2_events)}")
    print(f"  Task 3 Events: {len(task3_events)}")
    print(f"  Total Events Across All Tasks: {len(task1_events) + len(task2_events) + len(task3_events)}")
    
    # Final event type breakdown
    all_events = task1_events + task2_events + task3_events
    all_by_type = collect_events_by_type(all_events)
    
    print("\n  Event Types Across All Tasks:")
    for event_type, events in sorted(all_by_type.items()):
        print(f"    {event_type}: {len(events)}")
    
    print("\n✅ Memory integration with event streaming test PASSED!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVENT STREAMING TEST SUITE")
    print("=" * 70)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("   Please set it before running tests:")
        print("   export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
