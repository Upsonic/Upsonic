"""
Simple test file for Graph debug levels (1 and 2)

Usage:
    python test_debug_graph.py

This will run both debug level 1 and 2 to show the difference.
"""

import asyncio
from upsonic import Agent, Graph, Task


async def main():
    print("\n" + "=" * 80)
    print("GRAPH DEBUG LEVEL TESTING")
    print("=" * 80)
    
    # ============================================
    # DEBUG LEVEL 1 - Standard Debug
    # ============================================
    print("\n" + "=" * 80)
    print("DEBUG LEVEL 1: Standard Debug Output")
    print("=" * 80)
    
    agent1 = Agent(
        model="openai/gpt-4o-mini",
        name="Graph Agent",
        debug=True,
        debug_level=1
    )
    
    graph1 = Graph(
        default_agent=agent1,
        debug=True,
        debug_level=1  # Standard debug
    )
    
    graph1.add(Task("What is 5 + 3?"))
    graph1.add(Task("What is 10 - 2?"))
    
    result1 = await graph1.run_async()
    print(f"\n✅ Result: {result1}")
    
    # ============================================
    # DEBUG LEVEL 2 - Comprehensive Detailed Debug
    # ============================================
    print("\n\n" + "=" * 80)
    print("DEBUG LEVEL 2: Comprehensive Detailed Debug Output")
    print("=" * 80)
    
    agent2 = Agent(
        model="openai/gpt-4o-mini",
        name="Graph Agent",
        debug=True,
        debug_level=2
    )
    
    graph2 = Graph(
        default_agent=agent2,
        debug=True,
        debug_level=2  # Detailed debug - shows EVERYTHING
    )
    
    graph2.add(Task("What is 5 + 3?"))
    graph2.add(Task("What is 10 - 2?"))
    
    result2 = await graph2.run_async()
    print(f"\n✅ Result: {result2}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print("\n💡 Notice the difference:")
    print("   - Level 1: Shows standard graph debug information")
    print("   - Level 2: Shows comprehensive details including:")
    print("     * 🌳 Real-time tree-based graph structure display")
    print("     * Node execution details with full context")
    print("     * Task execution start/end with timing")
    print("     * State propagation details")
    print("     * Decision node evaluation (if used)")
    print("     * Parallel execution coordination (if enabled)")
    print("     * Tree updates after each node completion")
    print("     * And much more!")


if __name__ == "__main__":
    asyncio.run(main())
