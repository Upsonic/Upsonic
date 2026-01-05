import asyncio
from upsonic import Agent, Task, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    # Send string message
    response = await chat.invoke("What is 2+2?")
    print(response)

    # Send Task object
    task = Task(description="Explain quantum computing")
    response = await chat.invoke(task)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())