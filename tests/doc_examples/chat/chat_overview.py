import asyncio
from upsonic import Agent, Task, Chat


async def main():
    # Create agent
    agent = Agent("openai/gpt-4o")

    # Create chat session
    chat = Chat(
        session_id="user123_session1",
        user_id="user123",
        agent=agent
    )

    # Send a message
    response = await chat.invoke("Hello, how are you?")
    print(response)

    # Access metrics
    print(f"Total cost: ${chat.total_cost}")
    print(f"Messages: {len(chat.all_messages)}")


if __name__ == "__main__":
    asyncio.run(main())