import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    async with Chat(session_id="session1", user_id="user1", agent=agent) as chat1:
        await chat1.invoke("Hello")

    async with Chat(session_id="session2", user_id="user1", agent=agent) as chat2:
        await chat2.invoke("Hi")
        # Each session manages its own resources


if __name__ == "__main__":
    asyncio.run(main())