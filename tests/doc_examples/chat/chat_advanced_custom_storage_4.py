import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import RedisStorage


async def main():
    # Redis storage with TTL
    storage = RedisStorage(
        prefix="chat",
        host="localhost",
        port=6379,
        expire=3600  # 1 hour TTL
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