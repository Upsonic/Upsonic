import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        max_concurrent_invocations=2
    )

    print(f"Max concurrent invocations: 2")
    print(f"Initial session state: {chat.state.value}\n")

    # Demonstrate 2 concurrent invocations (within limit)
    print("Starting 2 concurrent invocations...")
    tasks = [
        chat.invoke("What is 2+2?"),
        chat.invoke("What is the capital of France?")
    ]
    
    responses = await asyncio.gather(*tasks)
    
    print("\nResponses received:")
    for i, response in enumerate(responses, 1):
        print(f"{i}. {response}")
    
    # Show metrics
    print(f"\nTotal cost: ${chat.total_cost}")
    print(f"Input tokens: {chat.input_tokens}")
    print(f"Output tokens: {chat.output_tokens}")
    print(f"Messages in history: {len(chat.all_messages)}")
    print(f"Final session state: {chat.state.value}")
    
    print("\n✓ Successfully handled 2 concurrent invocations!")
    
    # Test that 3rd concurrent invocation raises RuntimeError
    print("\n--- Testing limit enforcement (3rd concurrent invocation) ---")
    try:
        # Start 2 invocations that will take some time
        task1 = asyncio.create_task(chat.invoke("Count to 5 slowly"))
        task2 = asyncio.create_task(chat.invoke("Count to 5 slowly"))
        
        # Give them a moment to start
        await asyncio.sleep(0.1)
        
        # Try a 3rd one while the first two are still running
        print("Attempting 3rd concurrent invocation...")
        await chat.invoke("This should fail")
        print("✗ ERROR: 3rd invocation was allowed (should have raised RuntimeError)")
    except RuntimeError as e:
        print(f"✓ Limit enforced correctly! RuntimeError raised: {e}")
        # Wait for the first two to complete
        await asyncio.gather(task1, task2, return_exceptions=True)
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())