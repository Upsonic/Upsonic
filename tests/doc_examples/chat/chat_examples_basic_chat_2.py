import asyncio
from upsonic import Agent, Task, Chat

async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)
    
    # Use Task objects
    task1 = Task(description="Explain quantum computing")
    response1 = await chat.invoke(task1)
    print(response1)
    
    task2 = Task(description="Give me a summary")
    response2 = await chat.invoke(task2)
    print(response2)
    
    await chat.close()

if __name__ == "__main__":
    asyncio.run(main())