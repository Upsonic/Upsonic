import asyncio
from upsonic import Agent, Task, Chat
from upsonic.storage.providers import SqliteStorage

async def main():
    storage = SqliteStorage(
        db_file="chat.db",
        agent_sessions_table_name="sessions",
    )
    agent = Agent("openai/gpt-4o")
    
    async with Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        storage=storage,
        full_session_memory=True,
        summary_memory=True
    ) as chat:
        # Stream responses
        async for chunk in chat.stream("Tell me a story about AI"):
            print(chunk, end="", flush=True)
        print()
        
        # Send follow-up
        response = await chat.invoke("Continue the story")
        print(f"\nAssistant: {response}")
        
        # Access session summary
        summary = chat.get_session_summary()
        print(f"\n{summary}")

if __name__ == "__main__":
    asyncio.run(main())