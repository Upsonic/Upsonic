"""
Test file for the new Run Models Architecture.

This file tests:
1. RunInput creation and serialization
2. RunStatus and RunEvent enums
3. Event classes creation and to_dict()
4. RunResult with new fields
5. StreamRunResult with full_stream=True
"""

import asyncio
from upsonic import Agent, Task
from upsonic.agent import (
    # Run Input
    RunInput,
    # Run Status and Events
    RunStatus,
    RunEvent,
    BaseAgentRunEvent,
    RunStartedEvent,
    RunContentEvent,
    RunCompletedEvent,
    RunErrorEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
    # Run Results
    RunResult,
    StreamRunResult,
)


def test_run_input():
    """Test RunInput creation and serialization."""
    print("\n" + "="*60)
    print("TEST 1: RunInput")
    print("="*60)
    
    # Create from string
    input1 = RunInput(input_content="What is 2 + 2?")
    print(f"✓ Created RunInput from string: {input1.input_content_string()}")
    
    # Create from Task
    task = Task("Calculate the factorial of 5")
    input2 = RunInput.from_task(task)
    print(f"✓ Created RunInput from Task: {input2.input_content_string()[:50]}...")
    
    # Serialize to dict
    input_dict = input1.to_dict()
    print(f"✓ Serialized to dict: {list(input_dict.keys())}")
    
    # Deserialize from dict
    input3 = RunInput.from_dict(input_dict)
    print(f"✓ Deserialized from dict: {input3.input_content_string()}")
    
    print("\n✅ RunInput tests passed!")


def test_run_status_and_events():
    """Test RunStatus and RunEvent enums."""
    print("\n" + "="*60)
    print("TEST 2: RunStatus and RunEvent Enums")
    print("="*60)
    
    # Test RunStatus values
    print("RunStatus values:")
    for status in RunStatus:
        print(f"  - {status.name}: {status.value}")
    
    # Test RunEvent values
    print("\nRunEvent values (first 5):")
    for i, event in enumerate(RunEvent):
        if i >= 5:
            print(f"  ... and {len(list(RunEvent)) - 5} more")
            break
        print(f"  - {event.name}: {event.value}")
    
    print("\n✅ Enum tests passed!")


def test_event_classes():
    """Test event class creation and serialization."""
    print("\n" + "="*60)
    print("TEST 3: Event Classes")
    print("="*60)
    
    # RunStartedEvent
    started = RunStartedEvent(
        agent_id="agent-123",
        agent_name="TestAgent",
        run_id="run-456",
        model="openai/gpt-4o",
        model_provider="openai"
    )
    print(f"✓ RunStartedEvent: event={started.event}")
    print(f"  Dict keys: {list(started.to_dict().keys())}")
    
    # ToolCallStartedEvent (with args)
    tool_start = ToolCallStartedEvent(
        agent_id="agent-123",
        agent_name="TestAgent",
        run_id="run-456",
        tool_name="calculator",
        tool_call_id="call-789",
        tool_args={"x": 5, "y": 3, "operation": "add"}
    )
    print(f"✓ ToolCallStartedEvent: tool={tool_start.tool_name}, args={tool_start.tool_args}")
    
    # ToolCallCompletedEvent (with result)
    tool_complete = ToolCallCompletedEvent(
        agent_id="agent-123",
        agent_name="TestAgent",
        run_id="run-456",
        tool_name="calculator",
        tool_call_id="call-789",
        tool_result=8,
        duration_ms=150
    )
    print(f"✓ ToolCallCompletedEvent: result={tool_complete.tool_result}, duration={tool_complete.duration_ms}ms")
    
    # RunContentEvent
    content = RunContentEvent(
        agent_id="agent-123",
        agent_name="TestAgent",
        run_id="run-456",
        content="The answer is ",
        content_type="str"
    )
    print(f"✓ RunContentEvent: content='{content.content}'")
    
    # RunCompletedEvent
    completed = RunCompletedEvent(
        agent_id="agent-123",
        agent_name="TestAgent",
        run_id="run-456",
        content="The answer is 8",
        duration_ms=2500,
        usage={"input_tokens": 50, "output_tokens": 20}
    )
    print(f"✓ RunCompletedEvent: duration={completed.duration_ms}ms")
    
    print("\n✅ Event class tests passed!")


def test_run_result_with_new_fields():
    """Test RunResult with new fields."""
    print("\n" + "="*60)
    print("TEST 4: RunResult with New Fields")
    print("="*60)
    
    # Create RunResult
    result = RunResult(output="Test output")
    print(f"✓ Created RunResult")
    print(f"  run_id: {result.run_id[:8]}...")
    print(f"  status: {result.status}")
    print(f"  agent_id: '{result.agent_id}' (empty by default)")
    
    # Set agent info
    result.agent_id = "my-agent-123"
    result.agent_name = "MyTestAgent"
    result.input = RunInput(input_content="What is 2+2?")
    print(f"✓ Set agent_id, agent_name, and input")
    
    # Add events
    result.add_event(RunStartedEvent(
        agent_id=result.agent_id,
        agent_name=result.agent_name,
        run_id=result.run_id
    ))
    result.add_event(RunCompletedEvent(
        agent_id=result.agent_id,
        agent_name=result.agent_name,
        run_id=result.run_id,
        content="Test output"
    ))
    print(f"✓ Added {len(result.get_events())} events")
    
    # Serialize to dict
    result_dict = result.to_dict()
    print(f"✓ Serialized to dict with keys: {list(result_dict.keys())}")
    
    print("\n✅ RunResult tests passed!")


async def test_streaming_text_only():
    """Test standard text-only streaming (backward compatible)."""
    print("\n" + "="*60)
    print("TEST 5: Standard Text Streaming (full_stream=False)")
    print("="*60)
    
    agent = Agent("openai/gpt-4o", name="TextStreamAgent")
    task = Task("Say 'Hello World' and nothing else.")
    
    print("Streaming text output:")
    print("-" * 40)
    
    async with agent.stream(task) as result:
        print(f"✓ StreamRunResult created with run_id: {result.run_id[:8]}...")
        print(f"  agent_id: {result.agent_id[:8]}...")
        print(f"  agent_name: {result.agent_name}")
        print(f"  status: {result.status}")
        print()
        print("Output: ", end="")
        
        async for text_chunk in result.stream_output():  # Default: full_stream=False
            print(text_chunk, end="", flush=True)
        
        print()
        print("-" * 40)
        print(f"✓ Final status: {result.status}")
        print(f"✓ Agent events captured: {len(result.get_agent_events())}")
    
    print("\n✅ Text streaming test passed!")


async def test_streaming_full_events():
    """Test full event streaming with full_stream=True."""
    print("\n" + "="*60)
    print("TEST 6: Full Event Streaming (full_stream=True)")
    print("="*60)
    
    # Define a simple tool
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y
    
    agent = Agent("openai/gpt-4o", name="FullStreamAgent", tools=[add_numbers])
    task = Task("Use the add_numbers tool to add 5 and 3, then tell me the result.")
    
    print("Streaming all events:")
    print("-" * 40)
    
    event_counts = {}
    
    async with agent.stream(task) as result:
        async for event in result.stream_output(full_stream=True):
            event_type = event.event
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event_type == "RunStarted":
                print(f"🚀 Run started (run_id: {event.run_id[:8]}...)")
            elif event_type == "ToolCallStarted":
                print(f"🔧 Tool call: {event.tool_name}")
                print(f"   Args: {event.tool_args}")
            elif event_type == "ToolCallCompleted":
                print(f"✅ Tool result: {event.tool_result}")
            elif event_type == "RunContent":
                # Just print content without newline
                print(event.content, end="", flush=True)
            elif event_type == "RunCompleted":
                print(f"\n🏁 Completed! (duration: {event.duration_ms}ms)")
    
    print("-" * 40)
    print(f"Event summary: {event_counts}")
    print(f"Total agent events stored: {len(result.get_agent_events())}")
    
    print("\n✅ Full event streaming test passed!")


async def test_streaming_comparison():
    """Compare text-only vs full event streaming."""
    print("\n" + "="*60)
    print("TEST 7: Streaming Mode Comparison")
    print("="*60)
    
    agent = Agent("openai/gpt-4o", name="ComparisonAgent")
    task = Task("Count from 1 to 3.")
    
    # Test 1: Text only
    print("\nMode 1: stream_output() [text only]")
    print("-" * 30)
    async with agent.stream(task) as result:
        async for chunk in result.stream_output():
            print(f"  Received: {repr(chunk)[:50]}")
        print(f"  Events stored: {len(result.get_agent_events())}")
    
    # Test 2: Full stream
    print("\nMode 2: stream_output(full_stream=True)")
    print("-" * 30)
    agent2 = Agent("openai/gpt-4o", name="ComparisonAgent2")
    async with agent2.stream(task) as result:
        async for event in result.stream_output(full_stream=True):
            print(f"  Event: {event.event}")
        print(f"  Events stored: {len(result.get_agent_events())}")
    
    print("\n✅ Comparison test passed!")


def run_sync_tests():
    """Run synchronous tests."""
    test_run_input()
    test_run_status_and_events()
    test_event_classes()
    test_run_result_with_new_fields()


async def run_async_tests():
    """Run async streaming tests."""
    await test_streaming_text_only()
    await test_streaming_full_events()
    # await test_streaming_comparison()  # Uncomment to compare modes


def main():
    """Run all tests."""
    print("="*60)
    print("RUN MODELS ARCHITECTURE - TEST SUITE")
    print("="*60)
    
    # Run sync tests
    run_sync_tests()
    
    # Run async tests
    print("\n" + "="*60)
    print("ASYNC STREAMING TESTS")
    print("="*60)
    asyncio.run(run_async_tests())
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
