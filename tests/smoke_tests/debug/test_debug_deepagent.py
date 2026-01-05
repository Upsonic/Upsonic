"""
Simple test file for DeepAgent debug levels (1 and 2)

Usage:
    python test_debug_deepagent.py

This will run both debug level 1 and 2 to show the difference.
"""

import asyncio
from upsonic import Task
from upsonic.agent.deepagent import DeepAgent


async def main():
    print("\n" + "=" * 80)
    print("DEEPAGENT DEBUG LEVEL TESTING")
    print("=" * 80)
    
    # ============================================
    # DEBUG LEVEL 1 - Standard Debug
    # ============================================
    print("\n" + "=" * 80)
    print("DEBUG LEVEL 1: Standard Debug Output")
    print("=" * 80)
    
    agent1 = DeepAgent(
        model="openai/gpt-4o-mini",
        name="Test DeepAgent",
        debug=True,
        debug_level=1  # Standard debug
    )
    
    result1 = await agent1.do_async(Task("What is 2 + 2?"))
    print(f"\n✅ Result: {result1}")
    
    # ============================================
    # DEBUG LEVEL 2 - Comprehensive Detailed Debug
    # ============================================
    print("\n\n" + "=" * 80)
    print("DEBUG LEVEL 2: Comprehensive Detailed Debug Output")
    print("=" * 80)
    
    agent2 = DeepAgent(
        model="openai/gpt-4o-mini",
        name="Test DeepAgent",
        debug=True,
        debug_level=2  # Detailed debug - shows EVERYTHING
    )
    
    result2 = await agent2.do_async(Task("What is 2 + 2?"))
    print(f"\n✅ Result: {result2}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print("\n💡 Notice the difference:")
    print("   - Level 1: Shows standard DeepAgent debug information")
    print("   - Level 2: Shows comprehensive details including:")
    print("     * All Agent-level details (inherited)")
    print("     * Planning tool execution details")
    print("     * Filesystem operations (if enabled)")
    print("     * Subagent coordination details")
    print("     * And much more!")


if __name__ == "__main__":
    asyncio.run(main())
