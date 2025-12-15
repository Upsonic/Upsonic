"""
Stream Events Example

This example demonstrates how to use the comprehensive event streaming feature
to get full visibility into agent execution.

Run with: python examples/stream_events_example.py
"""

import asyncio
import os

from upsonic import Agent, Task, tool

# Import event classes for type checking
from upsonic import (
    # Pipeline events
    PipelineStartEvent,
    PipelineEndEvent,
    
    # Step events
    StepStartEvent,
    StepEndEvent,
    
    # Step-specific events
    ToolsConfiguredEvent,
    ToolCallEvent,
    ToolResultEvent,
    FinalOutputEvent,
    
    # Text streaming events
    TextDeltaEvent,
    TextCompleteEvent,
)



@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"




async def basic_streaming_example():
    """Basic example showing all events during streaming."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Event Streaming")
    print("=" * 70)
    
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Write a haiku about programming.")
    
    print("\nStreaming events:\n")
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            # Handle different event types
            if isinstance(event, PipelineStartEvent):
                print(f"🚀 Pipeline started with {event.total_steps} steps")
                
            elif isinstance(event, StepStartEvent):
                print(f"  ⏳ [{event.step_index + 1}/{event.total_steps}] {event.step_name}...")
                
            elif isinstance(event, StepEndEvent):
                status_icon = "✅" if event.status == "success" else "❌"
                print(f"  {status_icon} {event.step_name} completed in {event.execution_time:.3f}s")
                
            elif isinstance(event, TextDeltaEvent):
                # Print text as it streams
                print(event.content, end="", flush=True)
                
            elif isinstance(event, TextCompleteEvent):
                print()  # New line after text complete
                
            elif isinstance(event, FinalOutputEvent):
                print(f"\n📦 Final output type: {event.output_type}")
                
            elif isinstance(event, PipelineEndEvent):
                print(f"\n🏁 Pipeline completed in {event.total_duration:.2f}s (status: {event.status})")
        
        # Access final output
        print(f"\nFinal output: {result.get_final_output()}")


async def tool_call_example():
    """Example showing tool call events."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Tool Call Monitoring")
    print("=" * 70)
    
    agent = Agent("openai/gpt-4o-mini", tools=[calculate, get_current_time])
    task = Task("What is 123 * 456? Also, what time is it right now?")
    
    print("\nMonitoring tool calls:\n")
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            if isinstance(event, ToolsConfiguredEvent):
                print(f"🔧 Tools configured: {event.tool_names}")
                
            elif isinstance(event, ToolCallEvent):
                print(f"\n🔨 Tool Call: {event.tool_name}")
                print(f"   Arguments: {event.tool_args}")
                print(f"   Call ID: {event.tool_call_id}")
                
            elif isinstance(event, ToolResultEvent):
                print(f"   📤 Result: {event.result}")
                if event.is_error:
                    print(f"   ⚠️ Error occurred!")
                    
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
        
        print("\n")
        
        # Get tool event summary
        tool_events = result.get_tool_events()
        print(f"\nTotal tool events: {len(tool_events)}")


async def performance_monitoring_example():
    """Example showing how to monitor performance using events."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Performance Monitoring")
    print("=" * 70)
    
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Explain quantum computing in one sentence.")
    
    step_times = {}
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            if isinstance(event, StepStartEvent):
                # Could track start times here if needed
                pass
                
            elif isinstance(event, StepEndEvent):
                step_times[event.step_name] = event.execution_time
                
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
        
        print("\n")
        
        # Print performance metrics
        metrics = result.get_performance_metrics()
        stats = result.get_streaming_stats()
        
        print("\n📊 Performance Metrics:")
        print(f"   Total duration: {metrics.get('total_duration', 0):.3f}s")
        print(f"   Time to first token: {metrics.get('time_to_first_token', 0):.3f}s")
        print(f"   Characters/second: {metrics.get('characters_per_second', 0):.1f}")
        
        print("\n📈 Step Execution Times:")
        for step_name, exec_time in sorted(step_times.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(exec_time * 50) if exec_time < 2 else "█" * 100
            print(f"   {step_name:30} {exec_time:.4f}s {bar}")
        
        print(f"\n📋 Event Statistics:")
        print(f"   Total events: {stats['total_events']}")
        print(f"   Text events: {stats['text_events']}")
        print(f"   Tool events: {stats['tool_events']}")
        print(f"   Step events: {stats['step_events']}")


async def event_filtering_example():
    """Example showing how to filter for specific events."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Event Filtering (Text Only)")
    print("=" * 70)
    
    agent = Agent("openai/gpt-4o-mini")
    task = Task("Count from 1 to 5 slowly.")
    
    print("\nFiltering for text events only:\n")
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            # Only process text events
            if isinstance(event, TextDeltaEvent):
                print(event.content, end="", flush=True)
            elif isinstance(event, TextCompleteEvent):
                print(f"\n\n📝 Complete text ({len(event.full_text)} chars)")


async def custom_handler_example():
    """Example showing a custom event handler pattern."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Custom Event Handler")
    print("=" * 70)
    
    # Define a custom event handler
    class EventHandler:
        def __init__(self):
            self.events = []
            self.text_buffer = ""
            
        async def handle(self, event):
            self.events.append(event)
            
            if isinstance(event, PipelineStartEvent):
                print(f"[Handler] Pipeline starting...")
            elif isinstance(event, ToolCallEvent):
                print(f"[Handler] Tool called: {event.tool_name}")
            elif isinstance(event, TextDeltaEvent):
                self.text_buffer += event.content
            elif isinstance(event, PipelineEndEvent):
                print(f"[Handler] Pipeline complete!")
                print(f"[Handler] Processed {len(self.events)} events")
        
        def get_text(self):
            return self.text_buffer
    
    agent = Agent("openai/gpt-4o-mini", tools=[calculate])
    task = Task("What is 99 + 1?")
    
    handler = EventHandler()
    
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            await handler.handle(event)
    
    print(f"\nFinal text: {handler.get_text()}")


async def main():
    """Run all examples."""
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("   Please set it: export OPENAI_API_KEY='your-api-key'")
        return
    
    print("\n" + "=" * 70)
    print("AGENT EVENT STREAMING EXAMPLES")
    print("=" * 70)
    
    # Run examples
    await basic_streaming_example()
    await tool_call_example()
    await performance_monitoring_example()
    await event_filtering_example()
    await custom_handler_example()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
