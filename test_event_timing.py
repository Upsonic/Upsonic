"""
Debug test to understand event timing.
"""

import asyncio
from upsonic import Agent, Task


async def test_event_timing():
    """Debug event timing issues."""
    print("=" * 60)
    print("EVENT TIMING DEBUG")
    print("=" * 60)
    
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        print(f"    [DEBUG] add() called with x={x}, y={y}")
        return x + y
    
    agent = Agent("openai/gpt-4o", name="TimingTest", tools=[add])
    task = Task("Use add tool to add 5 and 7. Just report the result number.")
    
    print("\n-- LIVE EVENTS --")
    event_types_live = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_output(full_stream=True):
            event_types_live.append(event.event)
            if event.event == "ToolCallStarted":
                print(f"[LIVE] ToolCallStarted: {event.tool_name}({event.tool_args})")
            elif event.event == "ToolCallCompleted":
                print(f"[LIVE] ToolCallCompleted: {event.tool_result}, {event.duration_ms}ms")
            elif event.event in ("RunStarted", "RunCompleted"):
                print(f"[LIVE] {event.event}")
            elif event.event == "RunContent":
                pass  # Skip content for clarity
        
        # Check stored events
        stored = result.get_agent_events()
        print(f"\n-- STORED EVENTS: {len(stored)} --")
        for e in stored:
            if e.event == "ToolCallStarted":
                print(f"[STORED] ToolCallStarted: {e.tool_name}({e.tool_args})")
            elif e.event == "ToolCallCompleted":
                print(f"[STORED] ToolCallCompleted: {e.tool_result}, {e.duration_ms}ms")
    
    print(f"\nLive event types: {event_types_live}")


if __name__ == "__main__":
    asyncio.run(test_event_timing())
