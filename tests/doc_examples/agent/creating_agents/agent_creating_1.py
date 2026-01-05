from upsonic import Agent, Task
from upsonic.tools import tool

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

task = Task(description="Trending startups and products.")
agent.print_do(task)