"""
Smoke test for Structured Team Output.

Tests that Team objects with response_format return structured output for all modes:
- sequential
- coordinate  
- route
"""

import pytest
from pydantic import BaseModel
from upsonic import Agent, Task, Team
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(120)


class TeamReport(BaseModel):
    """Structured output model for team results."""
    summary: str
    findings: list[str]
    conclusion: str


@pytest.mark.asyncio
async def test_structured_team_output_sequential():
    """Test structured output for Team in sequential mode."""
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
        mode="sequential",
        response_format=TeamReport
    )
    
    tasks = [
        Task(description="Research the latest developments in quantum computing"),
        Task(description="Write a summary report about quantum computing")
    ]
    
    result = await team.multi_agent_async([researcher, writer], tasks)
    
    # Verify result is structured
    assert result is not None, "Result should not be None"
    assert isinstance(result, TeamReport), f"Result should be TeamReport instance, got {type(result)}"
    
    # Verify all required fields
    assert isinstance(result.summary, str), "summary should be a string"
    assert isinstance(result.findings, list), "findings should be a list"
    assert all(isinstance(f, str) for f in result.findings), "all findings should be strings"
    assert isinstance(result.conclusion, str), "conclusion should be a string"


@pytest.mark.asyncio
async def test_structured_team_output_coordinate():
    """Test structured output for Team in coordinate mode."""
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
        model="openai/gpt-4o",
        response_format=TeamReport
    )
    
    tasks = [
        Task(description="Analyze Q4 sales data and identify trends"),
        Task(description="Create executive summary of findings")
    ]
    
    result = await team.multi_agent_async([data_analyst, report_writer], tasks)
    
    # Verify result is structured
    assert result is not None, "Result should not be None"
    assert isinstance(result, TeamReport), f"Result should be TeamReport instance, got {type(result)}"
    
    # Verify all required fields
    assert isinstance(result.summary, str), "summary should be a string"
    assert isinstance(result.findings, list), "findings should be a list"
    assert all(isinstance(f, str) for f in result.findings), "all findings should be strings"
    assert isinstance(result.conclusion, str), "conclusion should be a string"


@pytest.mark.asyncio
async def test_structured_team_output_route():
    """Test structured output for Team in route mode."""
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
        model="openai/gpt-4o",
        response_format=TeamReport
    )
    
    task = Task(description="What are the best practices for implementing OAuth 2.0?")
    
    result = await team.multi_agent_async([legal_expert, tech_expert], [task])
    
    # Verify result is structured
    assert result is not None, "Result should not be None"
    assert isinstance(result, TeamReport), f"Result should be TeamReport instance, got {type(result)}"
    
    # Verify all required fields
    assert isinstance(result.summary, str), "summary should be a string"
    assert isinstance(result.findings, list), "findings should be a list"
    assert all(isinstance(f, str) for f in result.findings), "all findings should be strings"
    assert isinstance(result.conclusion, str), "conclusion should be a string"

