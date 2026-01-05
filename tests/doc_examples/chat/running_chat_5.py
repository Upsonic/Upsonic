import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    # Check session state
    print(chat.state)  # IDLE, AWAITING_RESPONSE, STREAMING, or ERROR

    # Clear history
    chat.clear_history()

    # Reset session
    chat.reset_session()

    # Close session
    await chat.close()


if __name__ == "__main__":
    asyncio.run(main())