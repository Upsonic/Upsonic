"""
Simple test file for Team debug levels (1 and 2)

Usage:
    python test_debug_team.py

This will run both debug level 1 and 2 to show the difference.
"""

import asyncio
from upsonic import Agent, Team, Task


async def main():
    print("\n" + "=" * 80)
    print("TEAM DEBUG LEVEL TESTING")
    print("=" * 80)
    
    # ============================================
    # DEBUG LEVEL 1 - Standard Debug
    # ============================================
    print("\n" + "=" * 80)
    print("DEBUG LEVEL 1: Standard Debug Output")
    print("=" * 80)
    
    math_agent1 = Agent(
        model="openai/gpt-4o-mini",
        name="Math Agent",
        debug=True,
        debug_level=1
    )
    
    team1 = Team(
        agents=[math_agent1],
        mode="sequential",
        debug=True,
        debug_level=1  # Standard debug
    )
    
    result1 = await team1.multi_agent_async([math_agent1], Task("What is 10 + 15?"))
    print(f"\n✅ Result: {result1}")
    
    # ============================================
    # DEBUG LEVEL 2 - Comprehensive Detailed Debug
    # ============================================
    print("\n\n" + "=" * 80)
    print("DEBUG LEVEL 2: Comprehensive Detailed Debug Output")
    print("=" * 80)
    
    math_agent2 = Agent(
        model="openai/gpt-4o-mini",
        name="Math Agent",
        debug=True,
        debug_level=2
    )
    
    team2 = Team(
        agents=[math_agent2],
        mode="sequential",
        debug=True,
        debug_level=2  # Detailed debug - shows EVERYTHING
    )
    
    result2 = await team2.multi_agent_async([math_agent2], Task("What is 10 + 15?"))
    print(f"\n✅ Result: {result2}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print("\n💡 Notice the difference:")
    print("   - Level 1: Shows standard team debug information")
    print("   - Level 2: Shows comprehensive details including:")
    print("     * Task assignment details")
    print("     * Agent selection reasoning")
    print("     * Context sharing information")
    print("     * Task execution completion details")
    print("     * Result combination process")
    print("     * And much more!")


if __name__ == "__main__":
    asyncio.run(main())
