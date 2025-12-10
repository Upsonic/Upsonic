"""
Focused test to verify tool events are yielded in real-time with proper args.
"""

import asyncio
from upsonic import Agent, Task


async def test_realtime_tool_events():
    """Test that tool events are yielded in real-time with proper args."""
    print("=" * 60)
    print("REAL-TIME TOOL EVENT VERIFICATION")
    print("=" * 60)
    
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    agent = Agent("openai/gpt-4o", name="RealtimeTest", tools=[add])
    task = Task("Use add tool to add 10 and 20. Just say the result.")
    
    tool_events_during_stream = []
    
    async with agent.stream(task) as result:
        print("\n-- LIVE STREAMING --")
        async for event in result.stream_output(full_stream=True):
            if event.event == "RunStarted":
                print(f"🚀 Started")
            elif event.event == "ToolCallStarted":
                print(f"🔧 [LIVE] Tool: {event.tool_name}({event.tool_args})")
                tool_events_during_stream.append(("started", event.tool_name, event.tool_args))
            elif event.event == "ToolCallCompleted":
                print(f"✅ [LIVE] Result: {event.tool_result}, Duration: {event.duration_ms}ms")
                tool_events_during_stream.append(("completed", event.tool_name, event.tool_result))
            elif event.event == "RunContent":
                print(event.content, end="", flush=True)
            elif event.event == "RunCompleted":
                print(f"\n🏁 Done")
    
    print("\n" + "=" * 60)
    print(f"Tool events captured DURING streaming: {len(tool_events_during_stream)}")
    for status, name, data in tool_events_during_stream:
        print(f"  - {status}: {name} -> {data}")
    
    # Verify the events had proper data
    has_proper_args = any(
        args is not None and args != {} 
        for status, name, args in tool_events_during_stream 
        if status == "started"
    )
    has_proper_result = any(
        result is not None 
        for status, name, result in tool_events_during_stream 
        if status == "completed"
    )
    
    print()
    if has_proper_args:
        print("✅ ToolCallStarted had proper args!")
    else:
        print("❌ ToolCallStarted missing args!")
    
    if has_proper_result:
        print("✅ ToolCallCompleted had proper result!")
    else:
        print("❌ ToolCallCompleted missing result!")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_realtime_tool_events())
