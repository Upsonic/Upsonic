import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    # Check initial session state
    print(f"Initial session state: {chat.state.value}")

    # Send a message
    response = await chat.invoke("Hello! How are you?")
    print(f"\nResponse: {response}")
    print(f"Session state after invoke: {chat.state.value}")

    # Check session state
    if chat.state.value == "error":
        # Handle error state
        print("\nError state detected! Resetting session...")
        chat.reset_session()
        print(f"Session state after reset: {chat.state.value}")
    else:
        print(f"\nSession is healthy. State: {chat.state.value}")


if __name__ == "__main__":
    asyncio.run(main())