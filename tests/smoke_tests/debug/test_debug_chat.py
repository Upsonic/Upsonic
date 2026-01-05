"""
Simple test file for Chat debug levels (1 and 2)

Usage:
    python test_debug_chat.py

This will run both debug level 1 and 2 to show the difference.
"""

import asyncio
from upsonic import Agent, Chat


async def main():
    print("\n" + "=" * 80)
    print("CHAT DEBUG LEVEL TESTING")
    print("=" * 80)
    
    # ============================================
    # DEBUG LEVEL 1 - Standard Debug
    # ============================================
    print("\n" + "=" * 80)
    print("DEBUG LEVEL 1: Standard Debug Output")
    print("=" * 80)
    
    agent1 = Agent(model="openai/gpt-4o-mini")
    chat1 = Chat(
        session_id="test_session_1",
        user_id="test_user_1",
        agent=agent1,
        debug=True,
        debug_level=1  # Standard debug
    )
    
    result1 = await chat1.invoke("Hello! What is 2 + 2?")
    print(f"\n✅ Result: {result1}")
    print(f"📊 Total messages: {len(chat1.all_messages)}")
    print(f"💰 Total cost: ${chat1.total_cost:.4f}")
    
    # ============================================
    # DEBUG LEVEL 2 - Comprehensive Detailed Debug
    # ============================================
    print("\n\n" + "=" * 80)
    print("DEBUG LEVEL 2: Comprehensive Detailed Debug Output")
    print("=" * 80)
    
    agent2 = Agent(model="openai/gpt-4o-mini")
    chat2 = Chat(
        session_id="test_session_2",
        user_id="test_user_2",
        agent=agent2,
        debug=True,
        debug_level=2  # Detailed debug - shows EVERYTHING
    )
    
    result2 = await chat2.invoke("Hello! What is 2 + 2?")
    print(f"\n✅ Result: {result2}")
    print(f"📊 Total messages: {len(chat2.all_messages)}")
    print(f"💰 Total cost: ${chat2.total_cost:.4f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print("\n💡 Notice the difference:")
    print("   - Level 1: Shows standard chat debug information")
    print("   - Level 2: Shows comprehensive details including:")
    print("     * Session initialization details")
    print("     * Invocation start/end with full context")
    print("     * Token usage breakdown")
    print("     * Message processing details")
    print("     * Retry attempts (if any)")
    print("     * And much more!")


if __name__ == "__main__":
    asyncio.run(main())
