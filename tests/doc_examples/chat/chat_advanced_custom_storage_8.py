import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import InMemoryStorage


async def main():
    # In-memory storage with LRU cache limits
    storage = InMemoryStorage(
        max_sessions=1000,
    )

    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        storage=storage
    )

    # Invoke chat
    response = await chat.invoke("Hello")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())