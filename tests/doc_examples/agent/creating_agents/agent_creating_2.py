from upsonic import Agent, Task
from upsonic.tools import tool
import asyncio
async def main():
    @tool
    def search_web(query: str) -> str:
        """Search the web for information about a topic."""
        # Simulated web search - replace with actual implementation
        return f"Search results for '{query}': Found relevant information about the topic."

    agent = Agent(
        model="openai/gpt-4o",
        instructions="Write a report on the topic. Output only the report.",
        tools=[search_web]  # Tools can be added during initialization
    )

    # You can also add tools after initialization
    # agent.add_tools([another_tool])

    # do() and do_async() methods accept both Task objects and strings
    task = Task(description="Trending startups and products.")
    result = agent.do(task)
    # Or: result = agent.do("Trending startups and products.")

    # Print the response
    print(result)

    ################ STREAM RESPONSE #################
    async for text_chunk in agent.astream(task):
        print(text_chunk, end='', flush=True)
    print()  # New line after streaming

if __name__ == "__main__":
    asyncio.run(main())