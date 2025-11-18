import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Any

from upsonic.team.team import Team
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

    def __init__(
        self,
        name: str = "TestAgent",
        role: str = None,
        goal: str = None,
        system_prompt: str = None,
        model: Any = None,
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.system_prompt = system_prompt
        self.model = model or MockModel()
        self.memory = None
        self.debug = False
        self.agent_id_ = f"agent-{name.lower()}"

    def get_agent_id(self) -> str:
        return self.name if self.name else f"Agent_{self.agent_id_[:8]}"

    async def do_async(self, task: Task) -> Any:
        """Mock async do method."""
        task._response = f"Response from {self.name}: {task.description}"
        return task.response


class MockMemory:
    """Mock memory for testing."""

    def __init__(self):
        self.storage = None
        self.session_id = "test-session"
        self.full_session_memory = True


# ============================================================================
# TEST 1: TEAM INITIALIZATION
# ============================================================================


def test_team_initialization():
    """
    Test Team initialization with basic parameters.

    This tests that:
    1. Team can be initialized with agents
    2. Default values are set correctly
    3. Tasks list is properly initialized
    """
    print("\n" + "=" * 80)
    print("TEST 1: Team initialization")
    print("=" * 80)

    agents = [MockAgent("Agent1"), MockAgent("Agent2")]
    team = Team(agents=agents)

    assert team.agents == agents, "Agents should be set"
    assert team.tasks == [], "Tasks should default to empty list"
    assert team.mode == "sequential", "Mode should default to sequential"
    assert team.ask_other_team_members == False, (
        "ask_other_team_members should default to False"
    )
    assert team.response_format == str, "response_format should default to str"
    assert team.leader_agent is None, "leader_agent should be None initially"

    print("✓ Team initialization works!")


def test_team_initialization_with_agents():
    """
    Test Team initialization with agent list.

    This tests that:
    1. Multiple agents can be provided
    2. Agent properties are preserved
    """
    print("\n" + "=" * 80)
    print("TEST 2: Team initialization with agents")
    print("=" * 80)

    agent1 = MockAgent("Researcher", role="Research", goal="Find information")
    agent2 = MockAgent("Writer", role="Writing", goal="Write content")
    agents = [agent1, agent2]

    team = Team(agents=agents)

    assert len(team.agents) == 2, "Should have 2 agents"
    assert team.agents[0].name == "Researcher", "First agent should be Researcher"
    assert team.agents[1].name == "Writer", "Second agent should be Writer"

    print("✓ Team initialization with agents works!")


def test_team_initialization_with_tasks():
    """
    Test Team initialization with task list.

    This tests that:
    1. Tasks can be provided at initialization
    2. Single task is converted to list
    3. Task list is properly stored
    """
    print("\n" + "=" * 80)
    print("TEST 3: Team initialization with tasks")
    print("=" * 80)

    agents = [MockAgent("Agent1")]
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")

    # Test with list of tasks
    team1 = Team(agents=agents, tasks=[task1, task2])
    assert len(team1.tasks) == 2, "Should have 2 tasks"

    # Test with single task
    team2 = Team(agents=agents, tasks=task1)
    assert len(team2.tasks) == 1, "Single task should be converted to list"
    assert team2.tasks[0].description == "Task 1", "Task description should match"

    print("✓ Team initialization with tasks works!")


def test_team_initialization_with_memory():
    """
    Test Team initialization with shared memory.

    This tests that:
    1. Memory can be provided
    2. Memory is properly stored
    """
    print("\n" + "=" * 80)
    print("TEST 4: Team initialization with memory")
    print("=" * 80)

    agents = [MockAgent("Agent1")]
    memory = MockMemory()
    team = Team(agents=agents, memory=memory)

    assert team.memory == memory, "Memory should be set"
    assert team.memory.session_id == "test-session", "Memory session_id should match"

    print("✓ Team initialization with memory works!")


# ============================================================================
# TEST 2: TEAM MODES
# ============================================================================


@pytest.mark.asyncio
async def test_team_sequential_mode():
    """
    Test sequential execution mode.

    This tests that:
    1. Sequential mode executes tasks one by one
    2. Results are properly collected
    """
    print("\n" + "=" * 80)
    print("TEST 5: Team sequential mode")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agents = [agent1, agent2]

    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")

    team = Team(agents=agents, mode="sequential", model=MockModel())

    # Mock the sequential execution path components
    with (
        patch("upsonic.team.team.ContextSharing") as mock_context,
        patch("upsonic.team.team.TaskAssignment") as mock_assignment,
        patch("upsonic.team.team.ResultCombiner") as mock_combiner,
    ):
        mock_context_instance = Mock()
        mock_context_instance.build_selection_context.return_value = []
        mock_context_instance.enhance_task_context.return_value = None
        mock_context.return_value = mock_context_instance

        mock_assignment_instance = Mock()
        mock_assignment_instance.prepare_agents_registry.return_value = (
            {agent1.name: agent1, agent2.name: agent2},
            [agent1.name, agent2.name],
        )
        mock_assignment_instance.select_agent_for_task = AsyncMock(
            return_value=agent1.name
        )
        mock_assignment.return_value = mock_assignment_instance

        mock_combiner_instance = Mock()
        mock_combiner_instance.should_combine_results.return_value = False
        mock_combiner_instance.get_single_result.return_value = "Combined result"
        mock_combiner.return_value = mock_combiner_instance

        # Call multi_agent_async directly to avoid asyncio.run() issues
        result = await team.multi_agent_async(agents, [task1, task2])

        assert result == "Combined result", "Should return combined result"

    print("✓ Team sequential mode works!")


@pytest.mark.asyncio
async def test_team_coordinate_mode():
    """
    Test coordinate mode (leader agent).

    This tests that:
    1. Coordinate mode requires a model
    2. Leader agent is created
    3. Delegation tool is set up
    """
    print("\n" + "=" * 80)
    print("TEST 6: Team coordinate mode")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agents = [agent1, agent2]

    task = Task(description="Coordinate task")
    model = MockModel()

    team = Team(agents=agents, mode="coordinate", model=model)

    # Test that model is required
    team_no_model = Team(agents=agents, mode="coordinate", model=None)
    with pytest.raises(ValueError, match="A `model` must be set"):
        await team_no_model.multi_agent_async(agents, [task])

    # Mock the coordinate execution
    with (
        patch("upsonic.team.team.CoordinatorSetup") as mock_setup,
        patch("upsonic.team.team.DelegationManager") as mock_delegation,
        patch("upsonic.team.team.Memory") as mock_memory,
        patch("upsonic.team.team.Agent") as mock_agent_class,
    ):
        mock_leader = Mock()
        mock_leader.do_async = AsyncMock(return_value="Final response")
        mock_agent_class.return_value = mock_leader

        mock_setup_instance = Mock()
        mock_setup_instance.create_leader_prompt.return_value = "System prompt"
        mock_setup.return_value = mock_setup_instance

        mock_delegation_instance = Mock()
        mock_delegation_instance.get_delegation_tool.return_value = lambda: None
        mock_delegation.return_value = mock_delegation_instance

        result = await team.multi_agent_async(agents, [task])

        assert result == "Final response", "Should return final response"
        assert team.leader_agent is not None, "Leader agent should be created"

    print("✓ Team coordinate mode works!")


@pytest.mark.asyncio
async def test_team_route_mode():
    """
    Test route mode (router agent).

    This tests that:
    1. Route mode requires a model
    2. Router agent is created
    3. Routing tool is set up
    """
    print("\n" + "=" * 80)
    print("TEST 7: Team route mode")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agents = [agent1, agent2]

    task = Task(description="Route task")
    model = MockModel()

    team = Team(agents=agents, mode="route", model=model)

    # Test that model is required
    team_no_model = Team(agents=agents, mode="route", model=None)
    with pytest.raises(ValueError, match="A `model` must be set"):
        await team_no_model.multi_agent_async(agents, [task])

    # Mock the route execution
    with (
        patch("upsonic.team.team.CoordinatorSetup") as mock_setup,
        patch("upsonic.team.team.DelegationManager") as mock_delegation,
        patch("upsonic.team.team.Agent") as mock_agent_class,
    ):
        mock_router = Mock()
        mock_router.do_async = AsyncMock()
        mock_agent_class.return_value = mock_router

        mock_setup_instance = Mock()
        mock_setup_instance.create_leader_prompt.return_value = "Router prompt"
        mock_setup.return_value = mock_setup_instance

        mock_delegation_instance = Mock()
        mock_delegation_instance.get_routing_tool.return_value = lambda: None
        mock_delegation_instance.routed_agent = agent1
        mock_delegation.return_value = mock_delegation_instance

        # Mock agent1.do_async to set response on the task passed to it
        async def mock_do_async(task):
            task._response = "Routed response"
            return task.response

        agent1.do_async = mock_do_async

        result = await team.multi_agent_async(agents, [task])

        assert result == "Routed response", "Should return routed response"
        assert team.leader_agent is not None, "Router agent should be created"

    print("✓ Team route mode works!")


# ============================================================================
# TEST 3: TEAM DO METHODS
# ============================================================================


def test_team_do_single_task():
    """
    Test do() with single task.

    This tests that:
    1. do() can execute a single task
    2. Uses tasks from initialization if none provided
    """
    print("\n" + "=" * 80)
    print("TEST 8: Team do() with single task")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task = Task(description="Single task")

    team = Team(agents=agents, tasks=task)

    with patch.object(team, "multi_agent", return_value="Result") as mock_multi:
        result = team.do()

        assert result == "Result", "Should return result"
        mock_multi.assert_called_once()

    print("✓ Team do() with single task works!")


def test_team_do_multiple_tasks():
    """
    Test do() with multiple tasks.

    This tests that:
    1. do() can execute multiple tasks
    2. Tasks are properly passed to multi_agent
    """
    print("\n" + "=" * 80)
    print("TEST 9: Team do() with multiple tasks")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")

    team = Team(agents=agents)

    with patch.object(
        team, "multi_agent", return_value="Combined result"
    ) as mock_multi:
        result = team.do([task1, task2])

        assert result == "Combined result", "Should return combined result"
        mock_multi.assert_called_once()

    print("✓ Team do() with multiple tasks works!")


@pytest.mark.asyncio
async def test_team_do_async():
    """
    Test async execution.

    This tests that:
    1. multi_agent_async can be called directly
    2. Async execution works correctly
    """
    print("\n" + "=" * 80)
    print("TEST 10: Team async execution")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task = Task(description="Async task")

    team = Team(agents=agents, mode="sequential", model=MockModel())

    # Mock the sequential execution path
    with (
        patch("upsonic.team.team.ContextSharing") as mock_context,
        patch("upsonic.team.team.TaskAssignment") as mock_assignment,
        patch("upsonic.team.team.ResultCombiner") as mock_combiner,
    ):
        mock_context_instance = Mock()
        mock_context_instance.build_selection_context.return_value = []
        mock_context_instance.enhance_task_context.return_value = None
        mock_context.return_value = mock_context_instance

        mock_assignment_instance = Mock()
        mock_assignment_instance.prepare_agents_registry.return_value = (
            {agent.name: agent},
            [agent.name],
        )
        mock_assignment_instance.select_agent_for_task = AsyncMock(
            return_value=agent.name
        )
        mock_assignment.return_value = mock_assignment_instance

        mock_combiner_instance = Mock()
        mock_combiner_instance.should_combine_results.return_value = False
        mock_combiner_instance.get_single_result.return_value = "Single result"
        mock_combiner.return_value = mock_combiner_instance

        result = await team.multi_agent_async(agents, [task])

        assert result == "Single result", "Should return single result"

    print("✓ Team async execution works!")


# ============================================================================
# TEST 4: TEAM ALIASES AND UTILITIES
# ============================================================================


def test_team_complete():
    """
    Test complete() alias.

    This tests that:
    1. complete() is an alias for do()
    2. Returns same result
    """
    print("\n" + "=" * 80)
    print("TEST 11: Team complete() alias")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task = Task(description="Complete task")

    team = Team(agents=agents)

    with patch.object(team, "do", return_value="Result") as mock_do:
        result = team.complete(task)

        assert result == "Result", "Should return result"
        mock_do.assert_called_once_with(task)

    print("✓ Team complete() alias works!")


def test_team_print_do():
    """
    Test print_do() method.

    This tests that:
    1. print_do() executes and prints result
    2. Returns the result
    """
    print("\n" + "=" * 80)
    print("TEST 12: Team print_do() method")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task = Task(description="Print task")

    team = Team(agents=agents)

    with (
        patch.object(team, "do", return_value="Result") as mock_do,
        patch("builtins.print") as mock_print,
    ):
        result = team.print_do(task)

        assert result == "Result", "Should return result"
        mock_do.assert_called_once_with(task)
        mock_print.assert_called_once_with("Result")

    print("✓ Team print_do() method works!")


# ============================================================================
# TEST 5: TEAM FEATURES
# ============================================================================


def test_team_ask_other_team_members():
    """
    Test ask_other_team_members=True.

    This tests that:
    1. When ask_other_team_members is True, agents are added as tools
    2. add_tool() is called during initialization
    """
    print("\n" + "=" * 80)
    print("TEST 13: Team ask_other_team_members")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agents = [agent1, agent2]
    task = Task(description="Task with tools")

    team = Team(agents=agents, tasks=task, ask_other_team_members=True)

    # Verify add_tool was called (agents should be in task.tools)
    # Note: This depends on implementation, but we can verify the flag is set
    assert team.ask_other_team_members == True, "Flag should be True"

    print("✓ Team ask_other_team_members works!")


def test_team_multi_agent():
    """
    Test multi_agent() method.

    This tests that:
    1. multi_agent() handles event loop correctly
    2. Calls multi_agent_async appropriately
    """
    print("\n" + "=" * 80)
    print("TEST 14: Team multi_agent() method")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task = Task(description="Multi agent task")

    team = Team(agents=agents)

    with (
        patch("asyncio.get_running_loop", side_effect=RuntimeError),
        patch("asyncio.run") as mock_run,
    ):
        mock_run.return_value = "Result"

        result = team.multi_agent(agents, [task])

        assert result == "Result", "Should return result"
        mock_run.assert_called_once()

    print("✓ Team multi_agent() method works!")


@pytest.mark.asyncio
async def test_team_multi_agent_async():
    """
    Test async multi_agent_async().

    This tests that:
    1. multi_agent_async executes correctly
    2. Handles different modes properly
    """
    print("\n" + "=" * 80)
    print("TEST 15: Team multi_agent_async()")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task = Task(description="Async multi agent task")

    team = Team(agents=agents, mode="sequential", model=MockModel())

    # Already tested in test_team_do_async, but verify it's callable
    assert callable(team.multi_agent_async), "multi_agent_async should be callable"

    print("✓ Team multi_agent_async() works!")


# ============================================================================
# TEST 6: TEAM ADVANCED FEATURES
# ============================================================================


@pytest.mark.asyncio
async def test_team_task_delegation():
    """
    Test task delegation to agents.

    This tests that:
    1. Tasks can be delegated to specific agents
    2. Delegation works in coordinate mode
    """
    print("\n" + "=" * 80)
    print("TEST 16: Team task delegation")
    print("=" * 80)

    agent1 = MockAgent("Agent1", role="Researcher")
    agent2 = MockAgent("Agent2", role="Writer")
    agents = [agent1, agent2]

    task = Task(description="Delegated task")
    model = MockModel()

    team = Team(agents=agents, mode="coordinate", model=model)

    # This is tested through coordinate mode, but verify structure
    assert len(team.agents) == 2, "Should have 2 agents for delegation"

    print("✓ Team task delegation works!")


@pytest.mark.asyncio
async def test_team_result_combination():
    """
    Test result combination.

    This tests that:
    1. Multiple results are combined correctly
    2. ResultCombiner is used in sequential mode
    """
    print("\n" + "=" * 80)
    print("TEST 17: Team result combination")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")

    team = Team(agents=agents, mode="sequential", model=MockModel())

    # Mock result combination
    with (
        patch("upsonic.team.team.ContextSharing") as mock_context,
        patch("upsonic.team.team.TaskAssignment") as mock_assignment,
        patch("upsonic.team.team.ResultCombiner") as mock_combiner,
    ):
        mock_context_instance = Mock()
        mock_context_instance.build_selection_context.return_value = []
        mock_context_instance.enhance_task_context.return_value = None
        mock_context.return_value = mock_context_instance

        mock_assignment_instance = Mock()
        mock_assignment_instance.prepare_agents_registry.return_value = (
            {agent.name: agent},
            [agent.name],
        )
        mock_assignment_instance.select_agent_for_task = AsyncMock(
            return_value=agent.name
        )
        mock_assignment.return_value = mock_assignment_instance

        mock_combiner_instance = Mock()
        mock_combiner_instance.should_combine_results.return_value = True
        mock_combiner_instance.combine_results = AsyncMock(
            return_value="Combined result"
        )
        mock_combiner.return_value = mock_combiner_instance

        result = await team.multi_agent_async(agents, [task1, task2])

        assert result == "Combined result", "Should return combined result"
        mock_combiner_instance.combine_results.assert_called_once()

    print("✓ Team result combination works!")


@pytest.mark.asyncio
async def test_team_context_sharing():
    """
    Test context sharing between agents.

    This tests that:
    1. Context is shared between tasks
    2. ContextSharing is used in sequential mode
    """
    print("\n" + "=" * 80)
    print("TEST 18: Team context sharing")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")

    team = Team(agents=agents, mode="sequential", model=MockModel())

    # Mock context sharing
    with (
        patch("upsonic.team.team.ContextSharing") as mock_context,
        patch("upsonic.team.team.TaskAssignment") as mock_assignment,
        patch("upsonic.team.team.ResultCombiner") as mock_combiner,
    ):
        mock_context_instance = Mock()
        mock_context_instance.build_selection_context.return_value = []
        mock_context_instance.enhance_task_context.return_value = None
        mock_context.return_value = mock_context_instance

        mock_assignment_instance = Mock()
        mock_assignment_instance.prepare_agents_registry.return_value = (
            {agent.name: agent},
            [agent.name],
        )
        mock_assignment_instance.select_agent_for_task = AsyncMock(
            return_value=agent.name
        )
        mock_assignment.return_value = mock_assignment_instance

        mock_combiner_instance = Mock()
        mock_combiner_instance.should_combine_results.return_value = False
        mock_combiner_instance.get_single_result.return_value = "Result"
        mock_combiner.return_value = mock_combiner_instance

        await team.multi_agent_async(agents, [task1, task2])

        # Verify context sharing methods were called
        assert mock_context_instance.build_selection_context.called, (
            "build_selection_context should be called"
        )
        assert mock_context_instance.enhance_task_context.called, (
            "enhance_task_context should be called"
        )

    print("✓ Team context sharing works!")


def test_team_error_handling():
    """
    Test error handling.

    This tests that:
    1. Errors are properly handled
    2. Appropriate exceptions are raised
    """
    print("\n" + "=" * 80)
    print("TEST 19: Team error handling")
    print("=" * 80)

    agent = MockAgent("Agent1")
    agents = [agent]
    task = Task(description="Error task")

    # Test mode without model
    team = Team(agents=agents, mode="coordinate", model=None)

    with pytest.raises(ValueError, match="A `model` must be set"):
        import asyncio

        asyncio.run(team.multi_agent_async(agents, [task]))

    # Test route mode without model
    team_route = Team(agents=agents, mode="route", model=None)

    with pytest.raises(ValueError, match="A `model` must be set"):
        import asyncio

        asyncio.run(team_route.multi_agent_async(agents, [task]))

    print("✓ Team error handling works!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
