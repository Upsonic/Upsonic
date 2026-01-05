import asyncio
from upsonic import Agent, Chat

async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    try:
        response = await chat.invoke("Hello")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        # Reset session if needed
        chat.reset_session()

if __name__ == "__main__":
    asyncio.run(main())