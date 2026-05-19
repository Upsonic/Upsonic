import asyncio
from upsonic import Agent, Task, Chat
from upsonic.storage.providers import SqliteStorage
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    preferences: dict

async def main():
    # Setup persistent storage
    storage = SqliteStorage(
        db_file="chat.db",
        agent_sessions_table_name="sessions",
    )
    
    # Create agent
    agent = Agent("openai/gpt-4o")
    
    # Create chat with advanced configuration
    chat = Chat(
        session_id="complex_session",
        user_id="user123",
        agent=agent,
        storage=storage,
        full_session_memory=True,
        summary_memory=True,
        user_analysis_memory=True,
        user_profile_schema=UserProfile,
        num_last_messages=50,
        retry_attempts=3,
        retry_delay=1.0
    )
    
    # Have a conversation
    await chat.invoke("My name is Bob and I love Python")
    await chat.invoke("What's my name and what do I love?")
    
    # Access metrics — chat.duration / len(chat.all_messages) for
    # runtime, chat.usage for everything else (registry view).
    print(f"Session duration: {chat.duration:.1f}s")
    print(f"Messages:         {len(chat.all_messages)}")
    print(f"Total cost (USD): {chat.usage.cost}")
    print(f"Input tokens:     {chat.usage.input_tokens}")
    print(f"Output tokens:    {chat.usage.output_tokens}")
    print(f"Requests:         {chat.usage.requests}")
    
    # Clean up
    await chat.close()

if __name__ == "__main__":
    asyncio.run(main())