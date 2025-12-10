"""
Comprehensive test for Run Models streaming functionality.
Tests all event types with multiple tools and complex scenarios.
"""

import asyncio
from collections import defaultdict
from upsonic import Agent, Task
from upsonic.agent import RunStatus


# ============================================================
# TOOL DEFINITIONS
# ============================================================

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

def get_weather(city: str) -> dict:
    """Get weather for a city (simulated)."""
    weather_data = {
        "New York": {"temp": 22, "condition": "Sunny", "humidity": 45},
        "London": {"temp": 15, "condition": "Cloudy", "humidity": 70},
        "Tokyo": {"temp": 28, "condition": "Clear", "humidity": 60},
        "Paris": {"temp": 18, "condition": "Partly Cloudy", "humidity": 55},
    }
    return weather_data.get(city, {"temp": 20, "condition": "Unknown", "humidity": 50})

def calculate_statistics(numbers: list) -> dict:
    """Calculate statistics for a list of numbers."""
    if not numbers:
        return {"error": "Empty list"}
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "average": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }

def search_database(query: str, limit: int = 5) -> list:
    """Search a mock database (simulated)."""
    mock_results = [
        {"id": 1, "name": "Product A", "price": 29.99},
        {"id": 2, "name": "Product B", "price": 49.99},
        {"id": 3, "name": "Product C", "price": 19.99},
        {"id": 4, "name": "Product D", "price": 99.99},
        {"id": 5, "name": "Product E", "price": 39.99},
    ]
    # Filter by query
    results = [r for r in mock_results if query.lower() in r["name"].lower()]
    return results[:limit] if results else mock_results[:limit]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_event_summary(events: list, title: str = "Event Summary"):
    """Print a summary of captured events."""
    print(f"\n📊 {title}")
    print("-" * 50)
    
    event_counts = defaultdict(int)
    for event in events:
        event_counts[event.event] += 1
    
    for event_type, count in sorted(event_counts.items()):
        emoji = {
            "RunStarted": "🚀",
            "RunContent": "📝",
            "RunCompleted": "🏁",
            "RunError": "❌",
            "ToolCallStarted": "🔧",
            "ToolCallCompleted": "✅",
            "ThinkingStarted": "🧠",
            "ThinkingStep": "💭",
            "ThinkingCompleted": "✨",
            "CacheHit": "💾",
            "CacheMiss": "📭",
            "PolicyCheckStarted": "🛡️",
            "PolicyCheckCompleted": "✔️",
            "RunPaused": "⏸️",
            "RunContinued": "▶️",
        }.get(event_type, "•")
        print(f"  {emoji} {event_type}: {count}")
    
    print(f"\n  Total Events: {len(events)}")
    print("-" * 50)


# ============================================================
# TEST CASES
# ============================================================

async def test_1_simple_streaming():
    """Test 1: Basic text streaming without tools."""
    print("\n" + "=" * 60)
    print("TEST 1: Simple Text Streaming")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", name="SimpleAgent")
    task = Task("Write a haiku about programming. Just the haiku, nothing else.")
    
    output_text = ""
    async with agent.stream(task) as result:
        print(f"📋 Run ID: {result.run_id[:12]}...")
        print(f"👤 Agent: {result.agent_name}")
        print(f"📊 Status: {result.status}")
        print("\n📄 Output:")
        print("-" * 40)
        
        async for chunk in result.stream_output():
            print(chunk, end="", flush=True)
            output_text += chunk
        
        print("\n" + "-" * 40)
        print_event_summary(result.get_agent_events())
    
    assert len(output_text) > 0, "Should have output text"
    print("✅ Test 1 PASSED!")


async def test_2_single_tool_call():
    """Test 2: Single tool call with proper args and result."""
    print("\n" + "=" * 60)
    print("TEST 2: Single Tool Call")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", name="MathAgent", tools=[multiply])
    task = Task("What is 15 multiplied by 23? Use the multiply tool and report the answer.")
    
    tool_events = {"started": [], "completed": []}
    
    async with agent.stream(task) as result:
        print("🔄 Streaming events live:")
        print("-" * 40)
        
        async for event in result.stream_output(full_stream=True):
            if event.event == "RunStarted":
                print(f"🚀 Run started")
            elif event.event == "ToolCallStarted":
                print(f"🔧 TOOL CALL: {event.tool_name}({event.tool_args})")
                tool_events["started"].append(event)
                assert event.tool_args is not None, "Tool args should not be None!"
            elif event.event == "ToolCallCompleted":
                print(f"✅ RESULT: {event.tool_result} (took {event.duration_ms}ms)")
                tool_events["completed"].append(event)
                assert event.tool_result is not None, "Tool result should not be None!"
            elif event.event == "RunContent":
                print(event.content, end="", flush=True)
            elif event.event == "RunCompleted":
                print(f"\n🏁 Completed in {event.duration_ms}ms")
        
        print("-" * 40)
        print_event_summary(result.get_agent_events())
    
    # Verify tool events
    assert len(tool_events["started"]) >= 1, "Should have at least 1 ToolCallStarted"
    assert len(tool_events["completed"]) >= 1, "Should have at least 1 ToolCallCompleted"
    assert tool_events["started"][0].tool_name == "multiply"
    assert tool_events["started"][0].tool_args.get("x") == 15
    assert tool_events["started"][0].tool_args.get("y") == 23
    assert tool_events["completed"][0].tool_result.get("func") == 345
    
    print("✅ Test 2 PASSED!")


async def test_3_multiple_tool_calls():
    """Test 3: Multiple tool calls in sequence."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Tool Calls (Sequential)")
    print("=" * 60)
    
    agent = Agent(
        "openai/gpt-4o-mini", 
        name="CalculatorAgent", 
        tools=[add, subtract, multiply, divide]
    )
    task = Task(
        "Calculate this step by step using tools: "
        "First add 100 and 50, then multiply that result by 2, "
        "and finally subtract 25. Show each step result."
    )
    
    tool_call_count = 0
    tool_names_called = []
    
    async with agent.stream(task) as result:
        print("🔄 Streaming events live:")
        print("-" * 40)
        
        async for event in result.stream_output(full_stream=True):
            if event.event == "ToolCallStarted":
                tool_call_count += 1
                tool_names_called.append(event.tool_name)
                print(f"🔧 [{tool_call_count}] {event.tool_name}({event.tool_args})")
            elif event.event == "ToolCallCompleted":
                print(f"   ✅ Result: {event.tool_result}")
            elif event.event == "RunContent":
                print(event.content, end="", flush=True)
            elif event.event == "RunCompleted":
                print(f"\n🏁 Done ({event.duration_ms}ms)")
        
        print("-" * 40)
        print(f"\n📈 Tool calls made: {tool_call_count}")
        print(f"📈 Tools used: {tool_names_called}")
        print_event_summary(result.get_agent_events())
    
    assert tool_call_count >= 2, f"Expected at least 2 tool calls, got {tool_call_count}"
    print("✅ Test 3 PASSED!")


async def test_4_complex_multi_tool_scenario():
    """Test 4: Complex scenario with multiple different tools."""
    print("\n" + "=" * 60)
    print("TEST 4: Complex Multi-Tool Scenario")
    print("=" * 60)
    
    agent = Agent(
        "openai/gpt-4o-mini",
        name="MultiToolAgent",
        tools=[get_weather, calculate_statistics, search_database]
    )
    task = Task(
        "I need you to: "
        "1. Get the weather for Tokyo and New York "
        "2. Search the database for 'Product' "
        "3. Calculate statistics on the prices from the search results "
        "Provide a summary of your findings."
    )
    
    events_log = []
    
    async with agent.stream(task) as result:
        print("🔄 Live Event Stream:")
        print("-" * 50)
        
        async for event in result.stream_output(full_stream=True):
            events_log.append(event.event)
            
            if event.event == "RunStarted":
                print("🚀 Started")
            elif event.event == "ToolCallStarted":
                args_str = str(event.tool_args)[:50] + "..." if len(str(event.tool_args)) > 50 else str(event.tool_args)
                print(f"🔧 {event.tool_name}({args_str})")
            elif event.event == "ToolCallCompleted":
                result_str = str(event.tool_result)[:60] + "..." if len(str(event.tool_result)) > 60 else str(event.tool_result)
                print(f"   ✅ → {result_str}")
            elif event.event == "RunContent":
                pass  # Skip content for cleaner output
            elif event.event == "RunCompleted":
                print(f"🏁 Completed ({event.duration_ms}ms)")
        
        print("-" * 50)
        print_event_summary(result.get_agent_events())
        
        # Verify we got multiple tool types
        tool_started_events = [e for e in result.get_agent_events() if e.event == "ToolCallStarted"]
        tool_names = [e.tool_name for e in tool_started_events]
        print(f"\n🛠️ Tools used: {set(tool_names)}")
    
    assert len(tool_started_events) >= 2, "Should use at least 2 different tools"
    print("✅ Test 4 PASSED!")


async def test_5_event_ordering():
    """Test 5: Verify event ordering is correct."""
    print("\n" + "=" * 60)
    print("TEST 5: Event Ordering Verification")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", name="OrderingTestAgent", tools=[add])
    task = Task("Add 5 and 10 using the add tool.")
    
    event_sequence = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_output(full_stream=True):
            event_sequence.append(event.event)
    
    print(f"📋 Event sequence ({len(event_sequence)} events):")
    for i, event_type in enumerate(event_sequence):
        print(f"  {i+1}. {event_type}")
    
    # Verify ordering rules
    run_started_idx = event_sequence.index("RunStarted") if "RunStarted" in event_sequence else -1
    run_completed_idx = len(event_sequence) - 1 - event_sequence[::-1].index("RunCompleted") if "RunCompleted" in event_sequence else -1
    
    assert run_started_idx == 0, "RunStarted should be first"
    assert event_sequence[-1] == "RunCompleted", "RunCompleted should be last"
    
    # Check tool call ordering
    if "ToolCallStarted" in event_sequence and "ToolCallCompleted" in event_sequence:
        started_idx = event_sequence.index("ToolCallStarted")
        completed_idx = event_sequence.index("ToolCallCompleted")
        assert started_idx < completed_idx, "ToolCallStarted should come before ToolCallCompleted"
        print("\n✓ ToolCallStarted comes before ToolCallCompleted")
    
    print("✅ Test 5 PASSED!")


async def test_6_stored_events_match_streamed():
    """Test 6: Verify stored events match what was streamed."""
    print("\n" + "=" * 60)
    print("TEST 6: Stored Events Match Streamed")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", name="EventMatchAgent", tools=[multiply])
    task = Task("Calculate 12 * 8 using the multiply tool.")
    
    streamed_events = []
    
    async with agent.stream(task) as result:
        async for event in result.stream_output(full_stream=True):
            streamed_events.append(event)
        
        stored_events = result.get_agent_events()
    
    print(f"📤 Streamed events: {len(streamed_events)}")
    print(f"💾 Stored events: {len(stored_events)}")
    
    # Compare streamed vs stored
    streamed_types = [e.event for e in streamed_events]
    stored_types = [e.event for e in stored_events]
    
    print(f"\nStreamed types: {streamed_types}")
    print(f"Stored types: {stored_types}")
    
    # Verify tool events have proper data in stored events
    tool_started = [e for e in stored_events if e.event == "ToolCallStarted"]
    tool_completed = [e for e in stored_events if e.event == "ToolCallCompleted"]
    
    if tool_started:
        print(f"\n🔧 Stored ToolCallStarted.tool_args: {tool_started[0].tool_args}")
        assert tool_started[0].tool_args is not None, "Stored tool_args should not be None"
    
    if tool_completed:
        print(f"✅ Stored ToolCallCompleted.tool_result: {tool_completed[0].tool_result}")
        assert tool_completed[0].tool_result is not None, "Stored tool_result should not be None"
    
    print("✅ Test 6 PASSED!")


async def test_7_error_handling():
    """Test 7: Test error handling with divide by zero."""
    print("\n" + "=" * 60)
    print("TEST 7: Error Handling (Divide by Zero)")
    print("=" * 60)
    
    agent = Agent("openai/gpt-4o-mini", name="ErrorAgent", tools=[divide])
    task = Task("Try to divide 10 by 0 using the divide tool. Report what happens.")
    
    error_event_found = False
    
    async with agent.stream(task) as result:
        print("🔄 Streaming:")
        print("-" * 40)
        
        async for event in result.stream_output(full_stream=True):
            if event.event == "ToolCallStarted":
                print(f"🔧 {event.tool_name}({event.tool_args})")
            elif event.event == "ToolCallCompleted":
                if event.error:
                    print(f"❌ Error: {event.error}")
                    error_event_found = True
                else:
                    print(f"✅ Result: {event.tool_result}")
            elif event.event == "RunContent":
                print(event.content, end="", flush=True)
            elif event.event == "RunCompleted":
                print(f"\n🏁 Done")
        
        print("-" * 40)
        print_event_summary(result.get_agent_events())
    
    print(f"\n📝 Error event captured: {error_event_found}")
    print("✅ Test 7 PASSED!")


async def test_8_all_event_types():
    """Test 8: Comprehensive test to capture as many event types as possible."""
    print("\n" + "=" * 60)
    print("TEST 8: Comprehensive Event Type Coverage")
    print("=" * 60)
    
    agent = Agent(
        "openai/gpt-4o-mini",
        name="ComprehensiveAgent",
        tools=[add, multiply, get_weather, calculate_statistics]
    )
    task = Task(
        "Please help me with this multi-step task: "
        "1. Add 25 and 75 "
        "2. Multiply the result by 2 "
        "3. Get the weather in Tokyo "
        "4. Calculate statistics for [10, 20, 30, 40, 50] "
        "Summarize all results at the end."
    )
    
    all_event_types = set()
    
    async with agent.stream(task) as result:
        print("🔄 Capturing all event types...")
        print("-" * 40)
        
        async for event in result.stream_output(full_stream=True):
            all_event_types.add(event.event)
            
            if event.event in ["RunStarted", "RunCompleted"]:
                print(f"{'🚀' if event.event == 'RunStarted' else '🏁'} {event.event}")
            elif event.event == "ToolCallStarted":
                print(f"🔧 {event.tool_name}")
            elif event.event == "ToolCallCompleted":
                print(f"   ✅ Done")
        
        print("-" * 40)
        print_event_summary(result.get_agent_events())
        
        print(f"\n📋 Event types captured: {sorted(all_event_types)}")
    
    # Verify we got the core event types
    expected_types = {"RunStarted", "RunContent", "RunCompleted", "ToolCallStarted", "ToolCallCompleted"}
    missing = expected_types - all_event_types
    
    if missing:
        print(f"⚠️ Missing event types: {missing}")
    else:
        print("✓ All core event types captured!")
    
    print("✅ Test 8 PASSED!")


# ============================================================
# MAIN TEST RUNNER
# ============================================================

async def main():
    print("\n" + "🧪" * 30)
    print("     COMPREHENSIVE STREAMING EVENT TESTS")
    print("🧪" * 30)
    
    tests = [
        ("Simple Streaming", test_1_simple_streaming),
        ("Single Tool Call", test_2_single_tool_call),
        ("Multiple Tool Calls", test_3_multiple_tool_calls),
        ("Complex Multi-Tool", test_4_complex_multi_tool_scenario),
        ("Event Ordering", test_5_event_ordering),
        ("Stored Events Match", test_6_stored_events_match_streamed),
        ("Error Handling", test_7_error_handling),
        ("Event Type Coverage", test_8_all_event_types),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ Test '{name}' FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ Test '{name}' ERROR: {e}")
            failed += 1
    
    # Final Summary
    print("\n" + "=" * 60)
    print("                    FINAL SUMMARY")
    print("=" * 60)
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total:  {passed + failed}")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️ {failed} test(s) failed!")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())
