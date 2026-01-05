import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    
    async with Chat(session_id="session1", user_id="user1", agent=agent) as chat:
        response = await chat.invoke("Hello")
        print(response)
        # Storage connection automatically closed on exit


if __name__ == "__main__":
    asyncio.run(main())