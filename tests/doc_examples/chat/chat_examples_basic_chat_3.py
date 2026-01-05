import asyncio
from upsonic import Agent, Chat

async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)
    
    # Stream response
    print("Assistant: ", end="")
    stream = await chat.invoke("Tell me a story", stream=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()
    
    await chat.close()

if __name__ == "__main__":
    asyncio.run(main())