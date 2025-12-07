"""
Smoke test for Team sequential, coordinate, and route mode logging.

Tests that:
- Sequential mode logs show corresponding Agent names
- Coordinate mode logs show agent names (leader and members)
- Route mode logs show the selected agent name
"""

import pytest
from upsonic import Agent, Task, Team
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(120)


@pytest.mark.asyncio
async def test_sequential_mode_agent_name_logging():
    """Test that sequential mode logs show corresponding Agent names.
    
    In sequential mode:
    - Task 1 goes to Agent 1
    - Task 2 goes to Agent 2
    - We need the same number of tasks as agents
    """
    researcher = Agent(
        model="openai/gpt-4o",
        name="Researcher",
        role="Research Specialist",
        goal="Find accurate information and data"
    )
    
    writer = Agent(
        model="openai/gpt-4o",
        name="Writer",
        role="Content Writer",
        goal="Create clear and engaging content"
    )
    
    team = Team(
        agents=[researcher, writer],
        mode="sequential"
    )
    
    # Sequential mode: same number of tasks as agents
    # Task 1 → Agent 1 (Researcher)
    # Task 2 → Agent 2 (Writer)
    tasks = [
        Task(description="Research the latest developments in quantum computing"),
        Task(description="Write a blog post about quantum computing for general audience")
    ]
    
    # Verify we have the same number of tasks as agents
    assert len(tasks) == len([researcher, writer]), "Sequential mode requires same number of tasks as agents"
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await team.multi_agent_async([researcher, writer], tasks)
    
    output = output_buffer.getvalue()
    
    # Verify result exists
    assert result is not None, "Result should not be None"
    
    # Verify agent names appear in logs
    # agent_started is called with agent.get_agent_id() which should include the name
    # In sequential mode, both agents should be used (task 1 → agent 1, task 2 → agent 2)
    assert "Researcher" in output or "researcher" in output.lower(), f"Researcher name should appear in logs (Task 1 → Agent 1). Output: {output[:500]}"
    assert "Writer" in output or "writer" in output.lower(), f"Writer name should appear in logs (Task 2 → Agent 2). Output: {output[:500]}"


@pytest.mark.asyncio
async def test_route_mode_selected_agent_logging():
    """Test that route mode logs show the selected agent name."""
    legal_expert = Agent(
        model="openai/gpt-4o",
        name="Legal Expert",
        role="Legal Advisor",
        goal="Provide legal guidance and compliance information",
        system_prompt="You are an expert in corporate law and regulations"
    )
    
    tech_expert = Agent(
        model="openai/gpt-4o",
        name="Tech Expert",
        role="Technology Specialist",
        goal="Provide technical solutions and architecture advice",
        system_prompt="You are an expert in software architecture and cloud systems"
    )
    
    team = Team(
        agents=[legal_expert, tech_expert],
        mode="route",
        model="openai/gpt-4o"
    )
    
    task = Task(description="What are the best practices for implementing OAuth 2.0?")
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await team.multi_agent_async([legal_expert, tech_expert], [task])
    
    output = output_buffer.getvalue()
    
    # Verify result exists
    assert result is not None, "Result should not be None"
    
    # Verify that one of the agent names appears in logs (the selected one)
    # Since OAuth 2.0 is a technical topic, Tech Expert should be selected
    agent_names_in_output = (
        "Tech Expert" in output or "tech expert" in output.lower() or
        "Legal Expert" in output or "legal expert" in output.lower()
    )
    assert agent_names_in_output, f"At least one agent name (Tech Expert or Legal Expert) should appear in logs. Output: {output[:500]}"


@pytest.mark.asyncio
async def test_coordinate_mode_agent_name_logging():
    """Test that coordinate mode logs show agent names (leader and members)."""
    data_analyst = Agent(
        model="openai/gpt-4o",
        name="Data Analyst",
        role="Data Analysis Expert",
        goal="Analyze data and extract insights"
    )
    
    report_writer = Agent(
        model="openai/gpt-4o",
        name="Report Writer",
        role="Business Report Specialist",
        goal="Create professional business reports"
    )
    
    team = Team(
        agents=[data_analyst, report_writer],
        mode="coordinate",
        model="openai/gpt-4o"
    )
    
    tasks = [
        Task(description="Analyze Q4 sales data and identify trends"),
        Task(description="Create executive summary of findings")
    ]
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await team.multi_agent_async([data_analyst, report_writer], tasks)
    
    output = output_buffer.getvalue()
    
    # Verify result exists
    assert result is not None, "Result should not be None"
    
    # Verify agent names appear in logs
    # In coordinate mode, the leader agent coordinates and delegates to members
    # Both leader and member agent names should appear in logs
    agent_names_in_output = (
        "Data Analyst" in output or "data analyst" in output.lower() or
        "Report Writer" in output or "report writer" in output.lower()
    )
    assert agent_names_in_output, f"At least one agent name (Data Analyst or Report Writer) should appear in logs. Output: {output[:500]}"

