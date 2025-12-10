"""
Test to verify that tool events have proper arguments and results.
"""

import asyncio
from upsonic import Agent, Task


async def test_tool_events():
    """Test that tool events have proper args and results."""
    print("=" * 50)
    print("TOOL EVENT VERIFICATION TEST")
    print("=" * 50)
    
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    agent = Agent("openai/gpt-4o", name="ToolEventAgent", tools=[add])
    task = Task("Use the add tool to add 15 and 25. Report the result.")
    
    # Execute non-streaming to capture events
    output = await agent.do_async(task)
    result = agent.get_run_result()
    
    print(f"\nOutput: {output}")
    print(f"\nEvents captured: {len(result.get_events())}")
    
    print("\nDetailed Event List:")
    print("-" * 50)
    
    for i, event in enumerate(result.get_events()):
        print(f"\n{i+1}. {event.event}")
        if event.event == "ToolCallStarted":
            print(f"   Tool Name: {event.tool_name}")
            print(f"   Tool Args: {event.tool_args}")
            print(f"   Tool Call ID: {event.tool_call_id}")
        elif event.event == "ToolCallCompleted":
            print(f"   Tool Name: {event.tool_name}")
            print(f"   Tool Result: {event.tool_result}")
            print(f"   Duration: {event.duration_ms}ms")
            if hasattr(event, 'error') and event.error:
                print(f"   Error: {event.error}")
    
    print("\n" + "=" * 50)
    
    # Check for ToolCallStarted and ToolCallCompleted
    tool_started = [e for e in result.get_events() if e.event == "ToolCallStarted"]
    tool_completed = [e for e in result.get_events() if e.event == "ToolCallCompleted"]
    
    if tool_started:
        print(f"\n✅ ToolCallStarted events: {len(tool_started)}")
        for e in tool_started:
            if e.tool_args:
                print(f"   ✅ Has args: {e.tool_args}")
            else:
                print(f"   ❌ Missing args!")
    else:
        print("❌ No ToolCallStarted events!")
    
    if tool_completed:
        print(f"\n✅ ToolCallCompleted events: {len(tool_completed)}")
        for e in tool_completed:
            print(f"   Result: {e.tool_result}")
            print(f"   Duration: {e.duration_ms}ms")
    else:
        print("❌ No ToolCallCompleted events!")
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_tool_events())
