import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    await chat.invoke("Hello")
    await chat.invoke("How are you?")

    # Human-readable summary — just format the unified usage view.
    u = chat.usage
    print(
        f"Chat {chat.session_id} ({chat.user_id})\n"
        f"  Duration:      {chat.duration:.1f}s\n"
        f"  Messages:      {len(chat.all_messages)}\n"
        f"  Tokens (in):   {u.input_tokens}\n"
        f"  Tokens (out):  {u.output_tokens}\n"
        f"  Cost (USD):    {u.cost}"
    )


if __name__ == "__main__":
    asyncio.run(main())
