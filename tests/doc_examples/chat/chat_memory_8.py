import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        full_session_memory=True,
        feed_tool_call_results=True  # Include tool execution results
    )

    # Invoke chat
    response = await chat.invoke("Define Supervised Learning")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())