from upsonic import Agent
from upsonic.tools import tool

@tool
def calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

agent = Agent("openai/gpt-4o")
agent.add_tools([calculator])

# Access registered tools
print("\nRegistered tools:", agent.registered_agent_tools)  # Dict mapping tool names to wrapped tools

# Get tool definitions
tool_defs = agent.get_tool_defs()  # List[ToolDefinition]

print("\nTool definitions:", tool_defs)

agent.remove_tools([calculator])

# Access registered tools
print("\nRegistered tools:", agent.registered_agent_tools)  # Dict mapping tool names to wrapped tools

# Get tool definitions
tool_defs = agent.get_tool_defs()  # List[ToolDefinition]

print("\nTool definitions:", tool_defs)