"""
Simple test file for Agent debug levels (1 and 2)

Usage:
    python test_debug_agent.py

This will run both debug level 1 and 2 to show the difference.
"""

import asyncio
from upsonic import Agent, Task


async def main():
    print("\n" + "=" * 80)
    print("AGENT DEBUG LEVEL TESTING")
    print("=" * 80)
    
    # ============================================
    # DEBUG LEVEL 1 - Standard Debug
    # ============================================
    print("\n" + "=" * 80)
    print("DEBUG LEVEL 1: Standard Debug Output")
    print("=" * 80)
    
    agent1 = Agent(
        model="openai/gpt-4o-mini",
        name="Test Agent",
        debug=True,
        debug_level=1  # Standard debug
    )
    
    result1 = await agent1.do_async("What is 2 + 2?")
    print(f"\n✅ Result: {result1}")
    
    # ============================================
    # DEBUG LEVEL 2 - Comprehensive Detailed Debug
    # ============================================
    print("\n\n" + "=" * 80)
    print("DEBUG LEVEL 2: Comprehensive Detailed Debug Output")
    print("=" * 80)
    
    agent2 = Agent(
        model="openai/gpt-4o-mini",
        name="Test Agent",
        debug=True,
        debug_level=2  # Detailed debug - shows EVERYTHING
    )
    
    result2 = await agent2.do_async("What is 2 + 2?")
    print(f"\n✅ Result: {result2}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print("\n💡 Notice the difference:")
    print("   - Level 1: Shows standard debug information")
    print("   - Level 2: Shows comprehensive details including:")
    print("     * Full pipeline step execution details")
    print("     * Complete model request/response information")
    print("     * Detailed tool execution (if tools are used)")
    print("     * Memory operations details")
    print("     * Cache operations details")
    print("     * Policy check details")
    print("     * And much more!")


if __name__ == "__main__":
    asyncio.run(main())
