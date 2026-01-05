from upsonic import Agent
from upsonic.tools import tool

@tool
def calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

# Create agent
agent = Agent("openai/gpt-4o")

# Add tools dynamically
agent.add_tools([calculator])

# Remove tools (by name or object)
agent.remove_tools([calculator])
# Or: agent.remove_tools(["calculator"])