import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import JSONStorage


async def main():
    # JSON file-based storage
    storage = JSONStorage(
        directory_path="./chat_data",
        pretty_print=True
    )

    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        storage=storage
    )

    # Invoke chat
    response = await chat.invoke("Hello")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())