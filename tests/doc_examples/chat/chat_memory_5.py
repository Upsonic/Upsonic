import asyncio
from upsonic import Agent, Chat


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        dynamic_user_profile=True,  # Automatically generate schema fields
        user_analysis_memory=True  # Required for dynamic profiles
    )

    # The schema will be automatically generated based on user conversations
    # Fields are extracted as the user mentions information
    
    # Invoke chat
    response = await chat.invoke("Hello my name is Bob")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())