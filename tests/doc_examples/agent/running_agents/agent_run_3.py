import asyncio
from upsonic import Agent, Task


async def main():
    # Create agent and task
    agent = Agent("openai/gpt-4o")
    task = Task("Write a short poem about coding")
    
    # Stream the output
    async for text_chunk in agent.astream(task):
        print(text_chunk, end='', flush=True)
    print()  # New line after streaming


if __name__ == "__main__":
    asyncio.run(main())