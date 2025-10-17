import pytest
from upsonic import Agent, Task, Team
from pydantic import BaseModel

#t
class AnalysisResult(BaseModel):
	summary: str
	confidence: float
	recommendations: list[str]


class TestTeam:
	"""Test team functionality"""
def test_team_without_mode(capsys):
    """Test basic team functionality WITHOUT mode paramete. Verifies if we can match task to agent"""
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
        agents=[researcher, writer]
    )
    
    tasks = [
        Task(description="Research the latest developments in quantum computing"),
        Task(description="Write a blog post about quantum computing for general audience")
    ]
    
    result = team.print_do(tasks)
    
    captured = capsys.readouterr()
    output = captured.out
    
    # Input assertions
    assert len(team.agents) == 2
    assert len(tasks) == 2

    assert "Agent Started" in output or "Agent Status" in output
    assert "Task Result" in output or "Result:" in output
    
    agent_started_count = output.count("Agent Started")
    assert agent_started_count == 5, f"Expected 5 agents executed (2 user-defined + 3 coordinators), got {agent_started_count}"
    
    assert "Researcher" in output
    assert "Writer" in output
    
    assert "selected_agent" in output
    assert '"Researcher"' in output or "'Researcher'" in output
    assert '"Writer"' in output or "'Writer'" in output
    
    assert result is not None
    assert isinstance(result, str)
    


def test_team_with_mode(capsys):
    """Test team functionality WITH coordinate mode."""
    researcher = Agent(
        model="openai/gpt-4o",
        name="Researcher",
        role="Information Gatherer",
        goal="Find comprehensive and accurate information"
    )
    
    analyst = Agent(
        model="openai/gpt-4o",
        name="Analyst",
        role="Data Analyst",
        goal="Extract insights and identify patterns"
    )
    
    writer = Agent(
        model="openai/gpt-4o",
        name="Writer",
        role="Report Writer",
        goal="Create professional reports"
    )
    
    team = Team(
        agents=[researcher, analyst, writer],
        mode="coordinate",
        model="openai/gpt-4o"
    )
    
    tasks = [
        Task(description="Research market trends for electric vehicles"),
        Task(description="Analyze competitive landscape"),
        Task(description="Write market entry strategy report")
    ]
    
    result = team.print_do(tasks)
    
    captured = capsys.readouterr()
    output = captured.out
    
    assert len(team.agents) == 3
    assert len(tasks) == 3

    assert "Agent Started" in output or "Agent Status" in output
    assert "Task Result" in output or "Result:" in output
    
    agent_started_count = output.count("Agent Started")
    assert agent_started_count == 4, f"Expected 4 agents executed (1 coordinator + 3 user-defined), got {agent_started_count}"
    
    assert "Researcher" in output
    assert "Analyst" in output
    assert "Writer" in output
    
    assert "delegate_task" in output
    assert "Tool Usage Summary" in output
    assert "3 tools" in output  

    assert result is not None
    assert isinstance(result, str)
    
def test_team_with_response_format(capsys):
    """Test team functionality with response format using Pydantic model."""
    class AnalysisResult(BaseModel):
        summary: str
        confidence: float
        recommendations: list[str]

    
    analyst = Agent(
        model="openai/gpt-4o",
        name="Analyst",
        role="Data Analyst",
        goal="Analyze data and provide structured insights"
    )
    summarizer = Agent(
        model="openai/gpt-4o",
        name="Summarizer",
        role="Summarize Data",
        goal="Summarize the data and provide structured insights"
    )

    
    team = Team(
        agents=[analyst, summarizer],
        response_format=AnalysisResult
    )
    
    tasks = [
        Task(
            description="Analyze the electric vehicle market data and provide structured results",
        ),
		Task(
            description="Summarize the data and provide structured insights",
        )
    ]
    
    result = team.print_do(tasks)
    
    captured = capsys.readouterr()
    output = captured.out
    
    assert len(team.agents) == 2
    assert len(tasks) == 2

    assert "Agent Started" in output or "Agent Status" in output
    assert "Task Result" in output or "Result:" in output
    assert "Task Metrics" in output or "Total Estimated Cost" in output
    
    assert result is not None
    
    assert hasattr(result, 'summary')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'recommendations')

    
    assert isinstance(result.summary, str)
    assert isinstance(result.confidence, float)
    assert isinstance(result.recommendations, list)

    
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.summary) > 0
    assert len(result.recommendations) > 0


def test_agent_name_verification_with_mock():
    """Test agent names by verifying agent properties directly"""
    
    # Create agents with specific names
    writer = Agent(model="openai/gpt-4o", name="Writer")
    editor = Agent(model="openai/gpt-4o", name="Editor")
    
    # Verify agent names are set correctly
    assert writer.name == "Writer", f"Expected writer name to be 'Writer', got '{writer.name}'"
    assert editor.name == "Editor", f"Expected editor name to be 'Editor', got '{editor.name}'"
    
    # Create team with agents
    team = Team(agents=[writer, editor], mode="sequential")
    
    # Verify agents are in the team
    assert len(team.agents) == 2, f"Expected 2 agents in team, got {len(team.agents)}"
    assert team.agents[0].name == "Writer", f"Expected first agent to be 'Writer', got '{team.agents[0].name}'"
    assert team.agents[1].name == "Editor", f"Expected second agent to be 'Editor', got '{team.agents[1].name}'"
    
    # Create tasks and explicitly assign agents
    task1 = Task(description="Write a product description")
    task1.agent = writer  # Force this task to use writer
    
    task2 = Task(description="Edit and polish the description")
    task2.agent = editor  # Force this task to use editor
    
    # Verify task-agent assignments
    assert task1.agent == writer, "Task1 should be assigned to writer"
    assert task2.agent == editor, "Task2 should be assigned to editor"
    
    # Execute team
    result = team.do([task1, task2])
    
    # Verify the result is not None
    assert result is not None, "Team execution should return a result"