import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    await chat.invoke("Hello")
    await chat.invoke("How are you?")

    # Get human-readable summary
    summary = chat.get_session_summary()
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())