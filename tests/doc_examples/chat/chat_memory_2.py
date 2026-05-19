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
        user_analysis_memory=True,
        num_last_messages=50,
        feed_tool_call_results=True
    )

    await chat.invoke("Hi! My name is Alice and I love Python programming.")
    await chat.invoke("What's my name and what do I love?")
    await chat.invoke("Can you help me write a Python function to calculate fibonacci numbers?")
    await chat.invoke("Can you also help me optimize it?")

    print(f"Total cost: ${(chat.usage.cost or 0.0):.4f} | Tokens: {chat.usage.input_tokens + chat.usage.output_tokens:,} | Messages: {len(chat.all_messages)}")


if __name__ == "__main__":
    asyncio.run(main())