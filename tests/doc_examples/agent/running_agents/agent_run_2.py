from upsonic import Agent, Task
import asyncio

async def main():
    # Create agent
    agent = Agent("openai/gpt-4o")
    
    # Execute asynchronously (accepts Task or string)
    result = await agent.do_async("Explain quantum computing in simple terms")
    print(result)

# Run async function
asyncio.run(main())