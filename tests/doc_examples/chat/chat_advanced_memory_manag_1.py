import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        num_last_messages=20  # Only keep last 20 messages in context
    )

    # Invoke chat
    response = await chat.invoke("What is Machine Learning?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())