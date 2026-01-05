import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        user_analysis_memory=True,
        user_memory_mode="update"  # Incrementally update profiles
    )

    # Invoke chat
    response = await chat.invoke("Describe Computer Vision")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())