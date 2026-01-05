import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        feed_tool_call_results=True  # Include tool execution results in memory
    )

    # Invoke chat
    response = await chat.invoke("What is NLP?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())