import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import SqliteStorage

async def main():
    storage = SqliteStorage(
        db_file="chat.db",
        agent_sessions_table_name="sessions"
    )
    agent = Agent("openai/gpt-4o")
    
    # First session - enable user_analysis_memory to create user profile
    async with Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        storage=storage,
        full_session_memory=True,
        user_analysis_memory=True  # Enable to create user profile from conversation
    ) as chat1:
        await chat1.invoke("I love AI")
    
    # Second session (same user, different session)
    async with Chat(
        session_id="session2", # Session is different from previous session
        user_id="user1", # User is the same as in previous user session. Agent will remember the user profile from previous user session.
        agent=agent,
        storage=storage,
        full_session_memory=True,
        user_analysis_memory=True,  # Enable to load user profile from previous user session.
        debug=True  # Enable debug to trace profile loading
    ) as chat2:
        response = await chat2.invoke("What's my favorite topic?")
        print(response)  # Agent remembers from previous session

if __name__ == "__main__":
    asyncio.run(main())