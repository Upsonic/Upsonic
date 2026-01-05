import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    await chat.invoke("Hello")
    await chat.invoke("How are you?")

    # Access cost metrics
    print(f"Total cost: ${chat.total_cost:.4f}")
    print(f"Input tokens: {chat.input_tokens}")
    print(f"Output tokens: {chat.output_tokens}")


if __name__ == "__main__":
    asyncio.run(main())