import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import PostgresStorage


async def main():
    # PostgreSQL storage
    storage = PostgresStorage(
        agent_sessions_table_name="sessions",
        db_url="postgresql://user:pass@localhost:5432/dbname",
        schema="public"
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