import asyncio
from upsonic import Agent, Task, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    # Send message with file context
    response = await chat.invoke(
        "Analyze this document",
        context=["document.pdf", "data.csv"]
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())