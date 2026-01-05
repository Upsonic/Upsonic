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
    
    # Access metrics
    metrics = chat.get_session_metrics()
    print(f"Session duration: {metrics.duration:.1f}s")
    print(f"Messages: {metrics.message_count}")
    print(f"Total cost: ${chat.total_cost:.4f}")
    print(f"Avg response time: {metrics.average_response_time:.2f}s")
    
    # Get cost history
    cost_history = chat.get_cost_history()
    for entry in cost_history:
        print(f"Cost: ${entry['estimated_cost']:.4f}, "
              f"Tokens: {entry['input_tokens']} in, {entry['output_tokens']} out")
    
    # Clean up
    await chat.close()

if __name__ == "__main__":
    asyncio.run(main())