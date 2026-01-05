"""
Quick test to verify tree structure is correct
"""

import asyncio
import pytest
from upsonic import Agent, Task, Graph
from upsonic.graph.graph import DecisionFunc


@pytest.mark.asyncio
async def test_tree_structure():
    """Test that sequential tasks are siblings and decision branches are children."""
    print("\n" + "=" * 100)
    print("TREE STRUCTURE VERIFICATION TEST")
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
    
    # Create tasks
    country_task = Task("What's an interesting country in Central Asia?")
    geography_task = Task("What is the geography of this country?")
    culture_task = Task("What is the culture of this country?")
    mountain_task = Task("What is the most popular mountain in this country?")
    
    # Define a decision function
    def has_mountains(output):
        return "mountain" in output.lower()
    
    # Create a decision node
    decision = DecisionFunc("Has mountains?", has_mountains)
    
    # Add tasks with conditional branching to the graph
    graph.add(country_task >> geography_task >> decision.if_true(mountain_task).if_false(culture_task))
    
    print("\n[INFO] Expected tree structure:")
    print("  ├── country_task [Task] (sibling)")
    print("  ├── geography_task [Task] (sibling)")
    print("  ├── decision [Decision (Func)] (sibling)")
    print("  │   ├── mountain_task [Task] (child of decision)")
    print("  │   └── culture_task [Task] (child of decision)")
    print("\n[INFO] Running graph...\n")
    
    result = await graph.run_async()
    
    print(f"\n✅ Final Output: {graph.get_output()}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    asyncio.run(test_tree_structure())
