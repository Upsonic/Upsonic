import asyncio
from upsonic import Agent, Task, Chat

async def main():
    # Create agent
    agent = Agent("openai/gpt-4o")
    
    # Create chat session
    chat = Chat(
        session_id="example_session",
        user_id="example_user",
        agent=agent
    )
    
    # Send messages
    response1 = await chat.invoke("Hello, my name is Alice")
    print(f"Assistant: {response1}")
    
    response2 = await chat.invoke("What's my name?")
    print(f"Assistant: {response2}")
    
    # Access history
    print(f"\nTotal messages: {len(chat.all_messages)}")
    print(f"Total cost: ${chat.total_cost:.4f}")
    
    # Clean up
    await chat.close()

if __name__ == "__main__":
    asyncio.run(main())