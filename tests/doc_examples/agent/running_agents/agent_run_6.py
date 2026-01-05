from upsonic import Agent
from upsonic.tools import tool

@tool
def calculate(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

@tool
def another_tool() -> str:
    """Another tool."""
    return "Another tool"

# Add tools during initialization
agent = Agent("openai/gpt-4o", tools=[calculate])

# Or add tools after initialization
agent.add_tools([another_tool])

# Get all registered tool definitions
tool_defs = agent.get_tool_defs()

print(tool_defs)