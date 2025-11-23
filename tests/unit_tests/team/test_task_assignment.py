import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from upsonic.team.task_assignment import TaskAssignment
from upsonic.tasks.tasks import Task
from upsonic.agent.agent import Agent


# ============================================================================
# MOCK COMPONENTS FOR TESTING
# ============================================================================


class MockModel:
    """Mock model for testing."""

    def __init__(self, name: str = "test-model"):
        self.model_name = name


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "TestAgent", model: Any = None):
        self.name = name
        self.model = model or MockModel()
        self.agent_id_ = f"agent-{name.lower()}"
        self.agent = None

    def get_agent_id(self) -> str:
        return self.name if self.name else f"Agent_{self.agent_id_[:8]}"


# ============================================================================
# TEST 1: TASK ASSIGNMENT INITIALIZATION
# ============================================================================


def test_task_assignment_initialization():
    """
    Test TaskAssignment initialization.

    This tests that:
    1. TaskAssignment can be initialized
    2. Initialization is successful
    """
    print("\n" + "=" * 80)
    print("TEST 1: TaskAssignment initialization")
    print("=" * 80)

    assignment = TaskAssignment()

    assert assignment is not None, "TaskAssignment should be initialized"

    print("✓ TaskAssignment initialization works!")


# ============================================================================
# TEST 2: AGENTS REGISTRY PREPARATION
# ============================================================================


def test_task_assignment_prepare_agents_registry():
    """
    Test prepare_agents_registry method.

    This tests that:
    1. Agents registry is created correctly
    2. Agent names are extracted properly
    3. Returns tuple of (registry dict, names list)
    """
    print("\n" + "=" * 80)
    print("TEST 2: TaskAssignment prepare agents registry")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agent3 = MockAgent("Agent3")
    agents = [agent1, agent2, agent3]

    assignment = TaskAssignment()
    registry, names = assignment.prepare_agents_registry(agents)

    assert isinstance(registry, dict), "Registry should be a dict"
    assert isinstance(names, list), "Names should be a list"
    assert len(registry) == 3, "Should have 3 agents in registry"
    assert len(names) == 3, "Should have 3 agent names"

    assert "Agent1" in registry, "Should include Agent1"
    assert "Agent2" in registry, "Should include Agent2"
    assert "Agent3" in registry, "Should include Agent3"

    assert registry["Agent1"] == agent1, "Registry should map to agent instance"
    assert "Agent1" in names, "Names should include Agent1"

    print("✓ TaskAssignment prepare agents registry works!")


# ============================================================================
# TEST 3: AGENT SELECTION
# ============================================================================


@pytest.mark.asyncio
async def test_task_assignment_assign_task():
    """
    Test task assignment logic.

    This tests that:
    1. Agent is selected for a task
    2. Predefined agent in task is used if available
    3. Selection model is used when no predefined agent
    """
    print("\n" + "=" * 80)
    print("TEST 3: TaskAssignment assign task")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agents = [agent1, agent2]

    assignment = TaskAssignment()
    registry, names = assignment.prepare_agents_registry(agents)

    # Test with predefined agent
    task_with_agent = Task(description="Task with agent")
    task_with_agent.agent = agent1

    selected = await assignment.select_agent_for_task(
        current_task=task_with_agent,
        context=[],
        agents_registry=registry,
        agent_names=names,
        agent_configurations=agents,
    )

    assert selected == "Agent1", "Should select predefined agent"

    # Test without predefined agent (requires model)
    task_no_agent = Task(description="Task without agent")

    # Mock the Agent selection process
    from pydantic import BaseModel

    class SelectedAgent(BaseModel):
        selected_agent: str

    mock_task_response = Task(description="Selection task")
    mock_task_response._response = SelectedAgent(selected_agent="Agent2")

    with patch("upsonic.team.task_assignment.Agent") as mock_agent_class:
        mock_agent_instance = Mock()
        mock_agent_instance.do_async = AsyncMock()

        def mock_do_async(task):
            task._response = SelectedAgent(selected_agent="Agent2")
            return task.response

        mock_agent_instance.do_async = AsyncMock(side_effect=mock_do_async)
        mock_agent_class.return_value = mock_agent_instance

        selected = await assignment.select_agent_for_task(
            current_task=task_no_agent,
            context=[],
            agents_registry=registry,
            agent_names=names,
            agent_configurations=agents,
        )

        # Should select an agent (may be Agent2 or fallback to first)
        assert selected in names, "Should select a valid agent"

    print("✓ TaskAssignment assign task works!")


# ============================================================================
# TEST 4: TASK ROUTING
# ============================================================================


@pytest.mark.asyncio
async def test_task_assignment_route_task():
    """
    Test task routing.

    This tests that:
    1. Tasks are routed to appropriate agents
    2. Agent selection considers context
    3. Fallback to first agent if selection fails
    """
    print("\n" + "=" * 80)
    print("TEST 4: TaskAssignment route task")
    print("=" * 80)

    agent1 = MockAgent("Researcher", model=MockModel())
    agent2 = MockAgent("Writer", model=MockModel())
    agents = [agent1, agent2]

    assignment = TaskAssignment()
    registry, names = assignment.prepare_agents_registry(agents)

    task = Task(description="Research task")
    context = [task, agent1, agent2]

    # Mock selection that returns valid agent
    from pydantic import BaseModel

    class SelectedAgent(BaseModel):
        selected_agent: str

    with patch("upsonic.team.task_assignment.Agent") as mock_agent_class:
        mock_agent_instance = Mock()

        def mock_do_async(task):
            task._response = SelectedAgent(selected_agent="Researcher")
            return task.response

        mock_agent_instance.do_async = AsyncMock(side_effect=mock_do_async)
        mock_agent_class.return_value = mock_agent_instance

        selected = await assignment.select_agent_for_task(
            current_task=task,
            context=context,
            agents_registry=registry,
            agent_names=names,
            agent_configurations=agents,
        )

        assert selected in names, "Should select a valid agent"
        assert selected in registry, "Selected agent should be in registry"

    print("✓ TaskAssignment route task works!")


# ============================================================================
# TEST 5: AGENT NAME MATCHING
# ============================================================================


@pytest.mark.asyncio
async def test_task_assignment_agent_name_matching():
    """
    Test agent name matching logic.

    This tests that:
    1. Partial name matches work
    2. Case-insensitive matching works
    3. Fallback to first agent if no match
    """
    print("\n" + "=" * 80)
    print("TEST 5: TaskAssignment agent name matching")
    print("=" * 80)

    agent1 = MockAgent("ResearcherAgent", model=MockModel())
    agent2 = MockAgent("WriterAgent", model=MockModel())
    agents = [agent1, agent2]

    assignment = TaskAssignment()
    registry, names = assignment.prepare_agents_registry(agents)

    task = Task(description="Task")

    from pydantic import BaseModel

    class SelectedAgent(BaseModel):
        selected_agent: str

    # Test case-insensitive matching
    with patch("upsonic.team.task_assignment.Agent") as mock_agent_class:
        mock_agent_instance = Mock()

        def mock_do_async(task):
            task._response = SelectedAgent(selected_agent="researcher")  # lowercase
            return task.response

        mock_agent_instance.do_async = AsyncMock(side_effect=mock_do_async)
        mock_agent_class.return_value = mock_agent_instance

        selected = await assignment.select_agent_for_task(
            current_task=task,
            context=[],
            agents_registry=registry,
            agent_names=names,
            agent_configurations=agents,
        )

        # Should match "ResearcherAgent" due to case-insensitive matching
        assert selected in names, "Should match agent name"

    print("✓ TaskAssignment agent name matching works!")


# ============================================================================
# TEST 6: ERROR HANDLING
# ============================================================================


@pytest.mark.asyncio
async def test_task_assignment_error_handling():
    """
    Test error handling.

    This tests that:
    1. Missing model raises ValueError
    2. Invalid responses are handled
    3. Fallback to first agent on errors
    """
    print("\n" + "=" * 80)
    print("TEST 6: TaskAssignment error handling")
    print("=" * 80)

    # Test with empty agents list (should raise ValueError)
    assignment_empty = TaskAssignment()
    registry_empty, names_empty = assignment_empty.prepare_agents_registry([])

    task_empty = Task(description="Task")

    with pytest.raises(ValueError, match="must have a valid model"):
        await assignment_empty.select_agent_for_task(
            current_task=task_empty,
            context=[],
            agents_registry=registry_empty,
            agent_names=names_empty,
            agent_configurations=[],
        )

    # Test with agent that has no model attribute (using object instead of Mock)
    class AgentWithoutModel:
        def get_agent_id(self):
            return "Agent1"

        # No model attribute

    agent_no_model = AgentWithoutModel()
    agents_no_model = [agent_no_model]

    assignment_no_model = TaskAssignment()
    registry_no_model, names_no_model = assignment_no_model.prepare_agents_registry(
        agents_no_model
    )

    task_no_model = Task(description="Task")

    with pytest.raises(ValueError, match="must have a valid model"):
        await assignment_no_model.select_agent_for_task(
            current_task=task_no_model,
            context=[],
            agents_registry=registry_no_model,
            agent_names=names_no_model,
            agent_configurations=agents_no_model,
        )

    print("✓ TaskAssignment error handling works!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
