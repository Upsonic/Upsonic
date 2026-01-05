import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import MongoStorage


async def main():
    # MongoDB storage
    storage = MongoStorage(
        db_url="mongodb://localhost:27017",
        database_name="chat_db",
        sessions_collection_name="sessions",
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