import asyncio
from upsonic import Agent, Task, Chat
from upsonic.storage.providers import InMemoryStorage


async def main():
    # Setup storage (in-memory, no dependencies required)
    storage = InMemoryStorage()

    # Create agent
    agent = Agent("openai/gpt-4o")

    # Create chat with configuration
    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        storage=storage,
        full_session_memory=True,
        summary_memory=True,
        user_analysis_memory=True,
        num_last_messages=50,
        retry_attempts=3,
        retry_delay=1.0
    )

    # Use chat
    response = await chat.invoke("Hello!")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())