"""
Test file for Graph with DecisionFunc and DecisionLLM branching
Tests the tree display with complex branching scenarios

Usage:
    uv run test_debug_graph_branching.py
"""

import asyncio
from upsonic import Agent, Task, Graph
from upsonic.graph.graph import DecisionFunc, DecisionLLM


async def test_example_1_decision_func():
    """Test Example 1: DecisionFunc with branching."""
    print("\n" + "=" * 100)
    print("EXAMPLE 1: DecisionFunc with Branching - Debug Level 2")
    print("=" * 100)
    
    # Create an agent and graph
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Test Agent",
        debug=True,
        debug_level=2
    )
    graph = Graph(
        default_agent=agent,
        debug=True,
        debug_level=2
    )
    
    # Create tasks
    country_task = Task("What's an interesting country in Central Asia?")
    geography_task = Task("What is the geography of this country?")
    culture_task = Task("What is the culture of this country?")
    mountain_task = Task("What is the most popular mountain in this country?")
    final_task = Task("Summarize your findings")
    
    # Define a decision function
    def has_mountains(output):
        return "mountain" in output.lower()
    
    # Create a decision node
    decision = DecisionFunc("Has mountains?", has_mountains)
    
    # Add tasks with conditional branching to the graph
    graph.add(country_task >> geography_task >> decision.if_true(mountain_task).if_false(culture_task) >> final_task)
    
    print("\n[INFO] Running graph with DecisionFunc branching and debug_level=2...\n")
    result = await graph.run_async()
    
    print(f"\n✅ Final Output: {graph.get_output()}")
    print("\n" + "=" * 100)


async def test_example_2_decision_llm():
    """Test Example 2: DecisionLLM with branching."""
    print("\n" + "=" * 100)
    print("EXAMPLE 2: DecisionLLM with Branching - Debug Level 2")
    print("=" * 100)
    
    # Create an agent and graph
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Test Agent",
        debug=True,
        debug_level=2
    )
    graph = Graph(
        default_agent=agent,
        debug=True,
        debug_level=2
    )
    
    # Create tasks
    country_task = Task("What's an interesting country which has the biggest mountains?")
    geography_task = Task("What is the geography of this country?")
    culture_task = Task("What is the culture of this country?")
    mountain_task = Task("What is the most popular mountain in this country?")
    
    # Create a decision node using LLM
    decision = DecisionLLM("Has the biggest trains?")
    
    # Add tasks with conditional branching to the graph
    graph.add(country_task >> geography_task >> decision.if_true(mountain_task).if_false(culture_task))
    
    print("\n[INFO] Running graph with DecisionLLM branching and debug_level=2...\n")
    result = await graph.run_async()
    
    print(f"\n✅ Final Output: {graph.get_output()}")
    print("\n" + "=" * 100)


async def test_sequential_tasks():
    """Test simple sequential tasks (no branching)."""
    print("\n" + "=" * 100)
    print("TEST: Sequential Tasks (No Branching) - Debug Level 2")
    print("=" * 100)
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Test Agent",
        debug=True,
        debug_level=2
    )
    graph = Graph(
        default_agent=agent,
        debug=True,
        debug_level=2
    )
    
    graph.add(Task("What is 5 + 3?"))
    graph.add(Task("What is 10 - 2?"))
    graph.add(Task("What is 4 * 2?"))
    
    print("\n[INFO] Running sequential tasks with debug_level=2...\n")
    result = await graph.run_async()
    
    print(f"\n✅ Final Output: {graph.get_output()}")
    print("\n" + "=" * 100)


async def main():
    """Run all branching tests."""
    print("\n" + "=" * 100)
    print("GRAPH BRANCHING TESTS WITH TREE DISPLAY")
    print("=" * 100)
    print("\nThis test suite verifies:")
    print("  • Tree display with DecisionFunc branching")
    print("  • Tree display with DecisionLLM branching")
    print("  • Sequential task ordering (first to last)")
    print("  • Pruned branch display")
    print("  • Real-time tree updates")
    print("\n" + "=" * 100)
    
    # Test 1: Sequential tasks (simple case)
    await test_sequential_tasks()
    
    # Test 2: Example 1 - DecisionFunc
    await test_example_1_decision_func()
    
    # Test 3: Example 2 - DecisionLLM
    await test_example_2_decision_llm()
    
    print("\n" + "=" * 100)
    print("ALL BRANCHING TESTS COMPLETED!")
    print("=" * 100)
    print("\n💡 Verify:")
    print("   - Nodes appear in first-to-last order (not reversed)")
    print("   - Decision nodes are displayed correctly")
    print("   - Branches (true/false) are shown in the tree")
    print("   - Pruned branches are marked as 'Pruned'")
    print("   - Tree updates after each node completion")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
