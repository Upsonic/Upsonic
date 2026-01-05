import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    await chat.invoke("Hello")

    # Get detailed cost history
    cost_history = chat.get_cost_history()
    for entry in cost_history:
        print(f"Time: {entry['timestamp']}")
        print(f"Cost: ${entry['estimated_cost']:.4f}")
        print(f"Model: {entry['model_name']}")


if __name__ == "__main__":
    asyncio.run(main())