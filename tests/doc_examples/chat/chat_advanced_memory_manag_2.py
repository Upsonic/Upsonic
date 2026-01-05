import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        full_session_memory=True,
        summary_memory=True  # Automatically summarize long conversations
    )

    # Invoke chat
    response = await chat.invoke("Explain Deep Learning")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())