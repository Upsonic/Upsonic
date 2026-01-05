import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    # Send messages
    await chat.invoke("Hello")
    await chat.invoke("How are you?")

    # Access all messages
    messages = chat.all_messages
    for msg in messages:
        print(f"{msg.role}: {msg.content}")

    # Get recent messages
    recent = chat.get_recent_messages(count=5)
    return recent


if __name__ == "__main__":
    recent = asyncio.run(main())
    print("--------------------------------")
    print(recent)