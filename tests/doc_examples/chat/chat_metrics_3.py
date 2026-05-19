import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    await chat.invoke("Hello")

    # Read session metrics directly from chat + the unified usage view.
    print(f"Duration:        {chat.duration:.1f}s")
    print(f"Messages:        {len(chat.all_messages)}")
    print(f"Input tokens:    {chat.usage.input_tokens}")
    print(f"Output tokens:   {chat.usage.output_tokens}")
    print(f"Requests:        {chat.usage.requests}")
    print(f"Cost (USD):      {chat.usage.cost}")


if __name__ == "__main__":
    asyncio.run(main())
