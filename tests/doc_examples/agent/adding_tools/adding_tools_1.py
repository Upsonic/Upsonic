from upsonic import Agent
from upsonic.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

agent = Agent(
    model="openai/gpt-4o",
    tools=[web_search]
)