import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        full_session_memory=True,
        num_last_messages=20  # Keep last 20 request-response pairs
    )

    # Invoke chat
    response = await chat.invoke("What are Large Language Models?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())