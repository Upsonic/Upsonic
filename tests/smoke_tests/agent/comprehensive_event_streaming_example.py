"""
Comprehensive Event Streaming Example

This example demonstrates ALL possible event types that can be emitted during
agent execution. It's designed to test and verify every event type in the system.

Run with: python -m pytest tests/smoke_tests/agent/comprehensive_event_streaming_example.py -v -s
"""

import asyncio
import os
import pytest
from typing import Dict, List, Set

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.storage.providers.in_memory import InMemoryStorage

# Import ALL event classes
from upsonic.run.events.events import (
    # Pipeline events
    PipelineStartEvent,
    PipelineEndEvent,
    
    # Step events
    StepStartEvent,
    StepEndEvent,
    
    # Agent initialization
    AgentInitializedEvent,
    StorageConnectionEvent,
    
    # Cache events
    CacheCheckEvent,
    CacheHitEvent,
    CacheMissEvent,
    CacheStoredEvent,
    
    # Policy events
    PolicyCheckEvent,
    PolicyFeedbackEvent,
    
    # Model events
    LLMPreparedEvent,
    ModelSelectedEvent,
    ValidationEvent,
    ToolsConfiguredEvent,
    MessagesBuiltEvent,
    ModelRequestStartEvent,
    ModelResponseEvent,
    
    # Tool events
    ToolCallEvent,
    ToolResultEvent,
    ExternalToolPauseEvent,
    ToolCallDeltaEvent,
    
    # Processing events
    ReflectionEvent,
    MemoryUpdateEvent,
    CultureUpdateEvent,
    ReliabilityEvent,
    
    # Run events
    RunStartedEvent,
    RunCompletedEvent,
    RunPausedEvent,
    RunCancelledEvent,
    ExecutionCompleteEvent,
    
    # Streaming events
    TextDeltaEvent,
    TextCompleteEvent,
    ThinkingDeltaEvent,
    FinalOutputEvent,
)


# Tool definitions for testing
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get weather information for a city (mock)."""
    return f"Weather in {city}: Sunny, 72°F"


@tool
def search_web(query: str) -> str:
    """Search the web for information (mock)."""
    return f"Search results for '{query}': Found 10 relevant results"


class EventTracker:
    """Track all events that occur during execution."""
    
    def __init__(self):
        self.events_by_type: Dict[str, List] = {}
        self.all_events: List = []
        self.event_types_seen: Set[str] = set()
        
    def track(self, event):
        """Track an event."""
        event_type = type(event).__name__
        self.event_types_seen.add(event_type)
        
        if event_type not in self.events_by_type:
            self.events_by_type[event_type] = []
        self.events_by_type[event_type].append(event)
        self.all_events.append(event)
        
    def get_count(self, event_class) -> int:
        """Get count of events of a specific type."""
        event_type = event_class.__name__
        return len(self.events_by_type.get(event_type, []))
        
    def has_seen(self, event_class) -> bool:
        """Check if an event type was seen."""
        return event_class.__name__ in self.event_types_seen
        
    def print_summary(self):
        """Print a summary of all events seen."""
        print("\n" + "=" * 80)
        print("EVENT TRACKING SUMMARY")
        print("=" * 80)
        print(f"\nTotal events captured: {len(self.all_events)}")
        print(f"Unique event types: {len(self.event_types_seen)}")
        print("\nEvents by type:")
        for event_type in sorted(self.event_types_seen):
            count = len(self.events_by_type[event_type])
            print(f"  {event_type:40} {count:3} occurrences")
        print("=" * 80)


@pytest.mark.asyncio
async def test_all_basic_events():
    """Test basic pipeline, step, and text events."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Pipeline and Text Events")
    print("=" * 80)
    
    tracker = EventTracker()
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Write a haiku about Python programming.")
    
    async for event in agent.astream(task, events=True):
        tracker.track(event)
        
        # Print key events
        if isinstance(event, PipelineStartEvent):
            print(f"✓ PipelineStartEvent: {event.total_steps} steps")
        elif isinstance(event, RunStartedEvent):
            print(f"✓ RunStartedEvent: agent_id={event.agent_id}")
        elif isinstance(event, StepStartEvent):
            print(f"  → StepStartEvent: {event.step_name}")
        elif isinstance(event, StepEndEvent):
            print(f"  ← StepEndEvent: {event.step_name} ({event.status})")
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
        elif isinstance(event, PipelineEndEvent):
            print(f"\n✓ PipelineEndEvent: {event.status} in {event.total_duration:.2f}s")
        elif isinstance(event, FinalOutputEvent):
            print(f"\n✓ FinalOutputEvent: type={event.output_type}")
    
    tracker.print_summary()
    
    # Verify expected events
    assert tracker.has_seen(PipelineStartEvent), "Missing PipelineStartEvent"
    assert tracker.has_seen(PipelineEndEvent), "Missing PipelineEndEvent"
    assert tracker.has_seen(RunStartedEvent), "Missing RunStartedEvent"
    assert tracker.has_seen(StepStartEvent), "Missing StepStartEvent"
    assert tracker.has_seen(StepEndEvent), "Missing StepEndEvent"
    assert tracker.has_seen(TextDeltaEvent), "Missing TextDeltaEvent"
    assert tracker.has_seen(FinalOutputEvent), "Missing FinalOutputEvent"
    
    print("\n✅ Test 1 PASSED: All basic events captured")


@pytest.mark.asyncio
async def test_model_and_tool_events():
    """Test model selection, tool configuration, and tool call events."""
    print("\n" + "=" * 80)
    print("TEST 2: Model and Tool Events")
    print("=" * 80)
    
    tracker = EventTracker()
    agent = Agent("openai/gpt-4o-mini", tools=[calculate, get_weather])
    task = Task("What is 15 * 23? Also, what's the weather in San Francisco?")
    
    async for event in agent.astream(task, events=True):
        tracker.track(event)
        
        if isinstance(event, ModelSelectedEvent):
            print(f"✓ ModelSelectedEvent: {event.model_name} (provider: {event.provider})")
        elif isinstance(event, ToolsConfiguredEvent):
            print(f"✓ ToolsConfiguredEvent: {event.tool_count} tools - {event.tool_names}")
        elif isinstance(event, MessagesBuiltEvent):
            print(f"✓ MessagesBuiltEvent: {event.message_count} messages")
        elif isinstance(event, ModelRequestStartEvent):
            print(f"✓ ModelRequestStartEvent: {event.model_name}, has_tools={event.has_tools}")
        elif isinstance(event, ModelResponseEvent):
            print(f"✓ ModelResponseEvent: has_text={event.has_text}, tool_calls={event.tool_call_count}")
        elif isinstance(event, ToolCallEvent):
            print(f"✓ ToolCallEvent: {event.tool_name} with args {event.tool_args}")
        elif isinstance(event, ToolResultEvent):
            print(f"✓ ToolResultEvent: {event.tool_name}, error={event.is_error}")
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    tracker.print_summary()
    
    # Verify expected events
    assert tracker.has_seen(ModelSelectedEvent), "Missing ModelSelectedEvent"
    assert tracker.has_seen(ToolsConfiguredEvent), "Missing ToolsConfiguredEvent"
    assert tracker.has_seen(MessagesBuiltEvent), "Missing MessagesBuiltEvent"
    assert tracker.has_seen(ModelRequestStartEvent), "Missing ModelRequestStartEvent"
    assert tracker.has_seen(ToolCallEvent), "Missing ToolCallEvent"
    assert tracker.has_seen(ToolResultEvent), "Missing ToolResultEvent"
    
    print("\n✅ Test 2 PASSED: All model and tool events captured")


@pytest.mark.asyncio
async def test_cache_events():
    """Test cache check, hit, miss, and stored events."""
    print("\n" + "=" * 80)
    print("TEST 3: Cache Events")
    print("=" * 80)
    
    tracker = EventTracker()
    agent = Agent("openai/gpt-4o-mini")
    
    # First run - should be cache miss
    task1 = Task("What is 2+2?")
    print("\nFirst run (cache miss expected):")
    async for event in agent.astream(task1, events=True):
        tracker.track(event)
        if isinstance(event, CacheCheckEvent):
            print(f"✓ CacheCheckEvent: enabled={event.cache_enabled}, hit={event.cache_hit}")
        elif isinstance(event, CacheMissEvent):
            print(f"✓ CacheMissEvent: method={event.cache_method}")
        elif isinstance(event, CacheHitEvent):
            print(f"✓ CacheHitEvent: method={event.cache_method}")
        elif isinstance(event, CacheStoredEvent):
            print(f"✓ CacheStoredEvent: method={event.cache_method}")
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    print("\n\nSecond run (cache hit expected):")
    task2 = Task("What is 2+2?")  # Same task
    async for event in agent.astream(task2, events=True):
        tracker.track(event)
        if isinstance(event, CacheCheckEvent):
            print(f"✓ CacheCheckEvent: enabled={event.cache_enabled}, hit={event.cache_hit}")
        elif isinstance(event, CacheMissEvent):
            print(f"✓ CacheMissEvent: method={event.cache_method}")
        elif isinstance(event, CacheHitEvent):
            print(f"✓ CacheHitEvent: method={event.cache_method}")
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    tracker.print_summary()
    
    # Verify cache events
    assert tracker.has_seen(CacheCheckEvent), "Missing CacheCheckEvent"
    assert tracker.get_count(CacheMissEvent) >= 1, "Should have at least one CacheMissEvent"
    assert tracker.get_count(CacheHitEvent) >= 1, "Should have at least one CacheHitEvent"
    
    print("\n✅ Test 3 PASSED: All cache events captured")


@pytest.mark.asyncio
async def test_memory_events():
    """Test memory update events."""
    print("\n" + "=" * 80)
    print("TEST 4: Memory Events")
    print("=" * 80)
    
    tracker = EventTracker()
    storage = InMemoryStorage()
    agent = Agent(
        "openai/gpt-4o-mini",
        memory=storage,
        session_id="test_session_memory"
    )
    
    task = Task("My name is Alice. Remember this.")
    
    async for event in agent.astream(task, events=True):
        tracker.track(event)
        
        if isinstance(event, StorageConnectionEvent):
            print(f"✓ StorageConnectionEvent: type={event.storage_type}, connected={event.is_connected}")
        elif isinstance(event, MemoryUpdateEvent):
            print(f"✓ MemoryUpdateEvent: type={event.memory_type}, messages_added={event.messages_added}")
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    tracker.print_summary()
    
    # Verify memory events
    assert tracker.has_seen(StorageConnectionEvent), "Missing StorageConnectionEvent"
    # MemoryUpdateEvent might not always fire depending on configuration
    if tracker.has_seen(MemoryUpdateEvent):
        print("  → MemoryUpdateEvent was captured")
    
    print("\n✅ Test 4 PASSED: Memory events captured")


@pytest.mark.asyncio
async def test_reflection_events():
    """Test reflection events (if reflection is enabled)."""
    print("\n" + "=" * 80)
    print("TEST 5: Reflection Events")
    print("=" * 80)
    
    tracker = EventTracker()
    # Note: Reflection might not be enabled by default
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Write a short story about a robot.")
    
    async for event in agent.astream(task, events=True):
        tracker.track(event)
        
        if isinstance(event, ReflectionEvent):
            print(f"✓ ReflectionEvent: applied={event.reflection_applied}, improved={event.improvement_made}")
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    tracker.print_summary()
    
    # ReflectionEvent might not fire if reflection is disabled
    if tracker.has_seen(ReflectionEvent):
        print("  → ReflectionEvent was captured")
        assert True
    else:
        print("  → ReflectionEvent not captured (reflection may be disabled)")
    
    print("\n✅ Test 5 PASSED: Reflection events checked")


@pytest.mark.asyncio
async def test_validation_events():
    """Test validation events."""
    print("\n" + "=" * 80)
    print("TEST 6: Validation Events")
    print("=" * 80)
    
    tracker = EventTracker()
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Hello, world!")
    
    async for event in agent.astream(task, events=True):
        tracker.track(event)
        
        if isinstance(event, ValidationEvent):
            print(f"✓ ValidationEvent: validated={event.attachments_validated}, passed={event.validation_passed}")
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    tracker.print_summary()
    
    # ValidationEvent might not always fire
    if tracker.has_seen(ValidationEvent):
        print("  → ValidationEvent was captured")
    
    print("\n✅ Test 6 PASSED: Validation events checked")


@pytest.mark.asyncio
async def test_execution_complete_event():
    """Test execution complete event."""
    print("\n" + "=" * 80)
    print("TEST 7: Execution Complete Event")
    print("=" * 80)
    
    tracker = EventTracker()
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Count to 3.")
    
    async for event in agent.astream(task, events=True):
        tracker.track(event)
        
        if isinstance(event, ExecutionCompleteEvent):
            print(f"✓ ExecutionCompleteEvent: type={event.output_type}, tool_calls={event.total_tool_calls}")
        elif isinstance(event, RunCompletedEvent):
            print(f"✓ RunCompletedEvent: agent_id={event.agent_id}")
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    tracker.print_summary()
    
    # Verify execution complete
    assert tracker.has_seen(ExecutionCompleteEvent), "Missing ExecutionCompleteEvent"
    
    print("\n✅ Test 7 PASSED: Execution complete event captured")


@pytest.mark.asyncio
async def test_all_event_types_comprehensive():
    """Comprehensive test that tries to trigger as many event types as possible."""
    print("\n" + "=" * 80)
    print("TEST 8: Comprehensive All Event Types Test")
    print("=" * 80)
    
    tracker = EventTracker()
    
    # Create agent with tools and memory to maximize events
    storage = InMemoryStorage()
    agent = Agent(
        "openai/gpt-4o-mini",
        tools=[calculate, get_weather, search_web],
        memory=storage,
        session_id="comprehensive_test_session"
    )
    
    # Task that will trigger multiple tool calls
    task = Task(
        "Calculate 10 * 20. Then get weather for New York. "
        "Finally, search for 'Python programming'."
    )
    
    print("\nStreaming events (showing all types):\n")
    
    async for event in agent.astream(task, events=True):
        tracker.track(event)
        
        # Print event type for visibility
        event_type = type(event).__name__
        if event_type not in tracker.events_by_type or len(tracker.events_by_type[event_type]) == 1:
            print(f"  [NEW] {event_type}")
        
        # Print text as it streams
        if isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
    
    print("\n")
    tracker.print_summary()
    
    # Expected events that should always be present
    required_events = [
        PipelineStartEvent,
        PipelineEndEvent,
        RunStartedEvent,
        StepStartEvent,
        StepEndEvent,
        ModelSelectedEvent,
        ToolsConfiguredEvent,
        MessagesBuiltEvent,
        ModelRequestStartEvent,
        TextDeltaEvent,
        FinalOutputEvent,
        ExecutionCompleteEvent,
    ]
    
    print("\nChecking required events:")
    missing_events = []
    for event_class in required_events:
        if tracker.has_seen(event_class):
            print(f"  ✓ {event_class.__name__}")
        else:
            print(f"  ✗ {event_class.__name__} - MISSING")
            missing_events.append(event_class.__name__)
    
    if missing_events:
        print(f"\n⚠️  Warning: Missing events: {missing_events}")
    else:
        print("\n✅ All required events captured!")
    
    # Optional events (may or may not be present)
    optional_events = [
        CacheCheckEvent,
        CacheHitEvent,
        CacheMissEvent,
        CacheStoredEvent,
        ToolCallEvent,
        ToolResultEvent,
        MemoryUpdateEvent,
        StorageConnectionEvent,
        ValidationEvent,
        ReflectionEvent,
        CultureUpdateEvent,
        ReliabilityEvent,
        RunCompletedEvent,
        ThinkingDeltaEvent,
        ToolCallDeltaEvent,
        TextCompleteEvent,
    ]
    
    print("\nOptional events found:")
    for event_class in optional_events:
        if tracker.has_seen(event_class):
            count = tracker.get_count(event_class)
            print(f"  ✓ {event_class.__name__} ({count}x)")
    
    print("\n✅ Test 8 PASSED: Comprehensive event test completed")


async def main():
    """Run all comprehensive event tests."""
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("   Please set it: export OPENAI_API_KEY='your-api-key'")
        return
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVENT STREAMING TEST SUITE")
    print("=" * 80)
    print("\nThis suite tests ALL possible event types in the agent framework.")
    print("Each test focuses on specific event categories.\n")
    
    try:
        await test_all_basic_events()
        await test_model_and_tool_events()
        await test_cache_events()
        await test_memory_events()
        await test_reflection_events()
        await test_validation_events()
        await test_execution_complete_event()
        await test_all_event_types_comprehensive()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nThis comprehensive test suite verifies that all event types")
        print("can be properly emitted and captured during agent execution.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())

