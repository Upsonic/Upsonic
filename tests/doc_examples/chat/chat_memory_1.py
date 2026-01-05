import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        full_session_memory=True
    )

    # Chat automatically uses memory
    response1 = await chat.invoke("My name is Alice")
    print(response1)
    
    response2 = await chat.invoke("What's my name?")  # Agent remembers from previous message
    print(response2)


if __name__ == "__main__":
    asyncio.run(main())