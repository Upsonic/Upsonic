import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import SqliteStorage


async def main():
    # Setup persistent SQLite storage
    storage = SqliteStorage(
        db_file="chat.db",
        agent_sessions_table_name="sessions",
    )

    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        storage=storage
    )

    response = await chat.invoke("Hello")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())