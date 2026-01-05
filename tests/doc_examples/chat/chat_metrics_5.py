import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    await chat.invoke("Hello")

    # Session timing
    print(f"Session duration: {chat.session_duration:.1f}s")
    print(f"Last activity: {chat.last_activity:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())