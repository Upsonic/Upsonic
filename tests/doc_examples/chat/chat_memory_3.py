import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import SqliteStorage


async def main():
    # Setup persistent storage
    storage = SqliteStorage(
        db_file="chat.db",
        agent_sessions_table_name="sessions",
    )
    
    # Create agent
    agent = Agent("openai/gpt-4o")

    # First session - store information
    chat1 = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        storage=storage,
        full_session_memory=True,
        summary_memory=True,
        user_analysis_memory=True,
    )
    
    response1 = await chat1.invoke("Remember: I prefer dark mode")
    print(f"Response 1: {response1}")
    
    # Second session - retrieve information from memory
    chat2 = Chat(
        session_id="session2",
        user_id="user1",
        agent=agent,
        storage=storage,
        full_session_memory=True,
        summary_memory=True,
        user_analysis_memory=True,
    )
    
    response2 = await chat2.invoke("What's my preference?")
    print(f"Response 2: {response2}")


if __name__ == "__main__":
    asyncio.run(main())