import asyncio
from upsonic import Agent, Chat

async def main():
    agent = Agent("openai/gpt-4o")
    chat = Chat(session_id="session1", user_id="user1", agent=agent)

    # Stream using invoke with stream=True
    # Note: invoke is async, so we need to await it to get the AsyncIterator
    stream_iterator = await chat.invoke("Tell me a story", stream=True)
    async for chunk in stream_iterator:
        print(chunk, end='', flush=True)
    
    print()  # New line after first stream
    
    # Or use dedicated stream method
    async for chunk in chat.stream("Tell me a story"):
        print(chunk, end='', flush=True)

if __name__ == "__main__":
    asyncio.run(main())