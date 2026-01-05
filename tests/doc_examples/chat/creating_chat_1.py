import asyncio
from upsonic import Agent, Chat


async def main():
    # Create agent
    agent = Agent("openai/gpt-4o")

    # Create chat with minimal configuration
    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent
    )

    # Send a message and get response
    response = await chat.invoke("Hello! How are you?")
    print(f"Response: {response}")
    print(f"\nTotal cost: ${chat.total_cost}")
    print(f"Messages in history: {len(chat.all_messages)}")


if __name__ == "__main__":
    asyncio.run(main())