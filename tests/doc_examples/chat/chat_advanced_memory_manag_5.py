import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        dynamic_user_profile=True  # Generate profile schema automatically
    )

    # Invoke chat
    response = await chat.invoke("What is Reinforcement Learning?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())