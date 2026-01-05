import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        full_session_memory=True,
        summary_memory=True,
        user_analysis_memory=True
    )

    # Send a message and get response
    response = await chat.invoke("Hello! How are you?")
    print(f"Response: {response}")
    print(f"\nTotal cost: ${chat.total_cost}")
    print(f"Input tokens: {chat.input_tokens}")
    print(f"Output tokens: {chat.output_tokens}")
    print(f"Messages in history: {len(chat.all_messages)}")


if __name__ == "__main__":
    asyncio.run(main())