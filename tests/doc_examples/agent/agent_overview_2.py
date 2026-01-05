import asyncio
from upsonic.agent.deepagent import DeepAgent
from upsonic import Task

async def main():
    # Create a deep agent
    agent = DeepAgent(model="openai/gpt-4o")
    
    # Create a complex task
    task = Task(
        description="Research Python web frameworks and create a comparison report. Save findings to /research/frameworks.txt and create /reports/comparison.txt"
    )
    
    # Execute - agent will automatically create todos and manage the workflow
    result = await agent.do_async(task)
    print(result)
    
    # Check files created
    files = await agent.filesystem_backend.glob("/**/*.txt")
    print(f"Files created: {files}")

asyncio.run(main())