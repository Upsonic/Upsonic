"""
Test streaming with full_stream and verify stored events.
"""

import asyncio
from upsonic import Agent, Task


async def test_streaming_with_stored_events():
    """Test streaming and check stored events after."""
    print("=" * 60)
    print("STREAMING TEST WITH STORED EVENT VERIFICATION")
    print("=" * 60)
    
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers together."""
        return x * y
    
    agent = Agent("openai/gpt-4o", name="StreamTestAgent", tools=[multiply])
    task = Task("Use multiply to calculate 12 times 5, then tell me the result. AND THEN MAKE A LONG JOKE")
    
    event_count_during_stream = 0
    
    async with agent.stream(task) as result:
        print("\n-- STREAMING EVENTS (full_stream=True) --")
        async for event in result.stream_output(full_stream=True):
            event_count_during_stream += 1
            if event.event == "RunStarted":
                print(f"🚀 Started")
            elif event.event == "ToolCallStarted":
                print(f"🔧 [STREAM] Tool: {event.tool_name}({event.tool_args})")
            elif event.event == "ToolCallCompleted":
                print(f"✅ [STREAM] Result: {event.tool_result}")
            elif event.event == "RunContent":
                print(event.content, end="", flush=True)
            elif event.event == "RunCompleted":
                print(f"\n🏁 Done")
        
        print()
        print(f"\n-- EVENTS DURING STREAM: {event_count_during_stream} --")
        
        # Check stored events
        stored_events = result.get_agent_events()
        print(f"\n-- STORED EVENTS: {len(stored_events)} --")
        
        # Look for ToolCallStarted and ToolCallCompleted with args
        tool_start_events = [e for e in stored_events if e.event == "ToolCallStarted"]
        tool_complete_events = [e for e in stored_events if e.event == "ToolCallCompleted"]
        
        print(f"\nToolCallStarted events: {len(tool_start_events)}")
        for e in tool_start_events:
            print(f"  - Tool: {e.tool_name}, Args: {e.tool_args}, CallID: {e.tool_call_id}")
        
        print(f"\nToolCallCompleted events: {len(tool_complete_events)}")
        for e in tool_complete_events:
            print(f"  - Tool: {e.tool_name}, Result: {e.tool_result}, Duration: {e.duration_ms}ms")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_streaming_with_stored_events())
