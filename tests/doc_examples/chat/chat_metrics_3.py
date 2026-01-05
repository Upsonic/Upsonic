import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    await chat.invoke("Hello")

    # Get comprehensive session metrics
    metrics = chat.get_session_metrics()
    print(f"Duration: {metrics.duration:.1f}s")
    print(f"Messages: {metrics.message_count}")
    print(f"Avg response time: {metrics.average_response_time:.2f}s")
    print(f"Messages/min: {metrics.messages_per_minute:.1f}")


if __name__ == "__main__":
    asyncio.run(main())