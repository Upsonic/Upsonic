"""
Smoke test for ask_other_team_members feature in Team class.

Tests that when ask_other_team_members=True, other agents are added as tools to tasks.
"""

import pytest
from upsonic import Agent, Task, Team
from upsonic.tools import tool

pytestmark = pytest.mark.timeout(120)


@tool
def simple_calculator(operation: str, a: float, b: float) -> str:
    """Perform a simple calculation."""
    if operation == "add":
        return str(a + b)
    elif operation == "multiply":
        return str(a * b)
    else:
        return "Unknown operation"


@pytest.mark.asyncio
async def test_ask_other_team_members():
    """Test that ask_other_team_members adds other agents as tools to tasks."""
    researcher = Agent(
        model="openai/gpt-4o",
        name="Researcher",
        role="Research Specialist",
        goal="Find accurate information and data"
    )
    
    calculator_agent = Agent(
        model="openai/gpt-4o",
        name="Calculator",
        role="Math Specialist",
        goal="Perform calculations",
        tools=[simple_calculator]
    )
    
    # Create a task
    task = Task(
        description="What is 2 + 2?"
    )
    
    # Create team with ask_other_team_members=True and pass tasks during initialization
    # When ask_other_team_members=True, add_tool() is called in __init__
    # This adds agents to self.tasks (tasks passed during initialization)
    team = Team(
        agents=[researcher, calculator_agent],
        tasks=[task],  # Pass tasks during initialization so add_tool() can modify them
        mode="sequential",
        ask_other_team_members=True
    )
    
    # Verify that agents were added to task.tools
    assert hasattr(task, 'tools'), "Task should have tools attribute"
    assert task.tools is not None, "Task tools should not be None"
    assert isinstance(task.tools, list), "Task tools should be a list"
    
    # Verify that the agents are in the task tools
    agent_names_in_tools = [agent.get_agent_id() for agent in task.tools if isinstance(agent, Agent)]
    assert len(agent_names_in_tools) >= 1, f"At least one agent should be in task tools. Found: {agent_names_in_tools}"
    
    # Verify that both agents are in the tools (since ask_other_team_members adds all agents)
    assert len(agent_names_in_tools) == 2, f"Both agents should be in task tools. Found: {agent_names_in_tools}"
    
    # Verify the specific agents are present
    agent_ids = [agent.get_agent_id() for agent in [researcher, calculator_agent]]
    for agent_id in agent_ids:
        assert agent_id in agent_names_in_tools, f"Agent {agent_id} should be in task tools"

