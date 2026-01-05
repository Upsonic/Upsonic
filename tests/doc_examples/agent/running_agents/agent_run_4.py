import asyncio
from upsonic import Agent, Task
from upsonic.run.events.events import (
    PipelineStartEvent,
    PipelineEndEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from upsonic.tools import tool

@tool
def calculate(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

async def main():
    agent = Agent("openai/gpt-4o")
    task = Task("Calculate 5 + 3", tools=[calculate])
    
    async for event in agent.astream(task, events=True):
        if isinstance(event, PipelineStartEvent):
            print(f"🚀 Starting pipeline with {event.total_steps} steps")
        
        elif isinstance(event, ToolCallEvent):
            print(f"\n🔧 Calling: {event.tool_name}({event.tool_args})")
        
        elif isinstance(event, ToolResultEvent):
            status = "❌" if event.is_error else "✅"
            print(f"\n{status} Result: {event.result_preview}")
        
        elif isinstance(event, TextDeltaEvent):
            print("\nText Delta Event: ", event.content, end='', flush=True)
        
        elif isinstance(event, PipelineEndEvent):
            print(f"\n✅ Completed in {event.total_duration:.2f}s")

asyncio.run(main())