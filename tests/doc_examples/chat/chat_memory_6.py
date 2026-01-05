import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        user_analysis_memory=True,
        user_memory_mode="update"  # or "replace"
    )

    # Invoke chat
    response = await chat.invoke("Explain Generative AI")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())