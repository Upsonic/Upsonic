import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Callable, Any

from upsonic.team.delegation_manager import DelegationManager
from upsonic.tasks.tasks import Task
from upsonic.agent.agent import Agent


# ============================================================================
# MOCK COMPONENTS FOR TESTING
# ============================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "TestAgent", role: str = None):
        self.name = name
        self.role = role
        self.agent_id_ = f"agent-{name.lower()}"
        self.memory = None

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


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str = "test_tool"):
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        return f"Tool {self.__name__} executed"


# ============================================================================
# TEST 1: DELEGATION MANAGER INITIALIZATION
# ============================================================================


def test_delegation_manager_initialization():
    """
    Test DelegationManager initialization.

    This tests that:
    1. DelegationManager can be initialized with members and tool mapping
    2. All attributes are set correctly
    3. routed_agent is None initially
    """
    print("\n" + "=" * 80)
    print("TEST 1: DelegationManager initialization")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    members = [agent1, agent2]

    tool1 = MockTool("tool1")
    tool2 = MockTool("tool2")
    tool_mapping = {"tool1": tool1, "tool2": tool2}

    manager = DelegationManager(members=members, tool_mapping=tool_mapping)

    assert manager.members == members, "Members should be set"
    assert manager.tool_mapping == tool_mapping, "Tool mapping should be set"
    assert manager.routed_agent is None, "routed_agent should be None initially"

    print("✓ DelegationManager initialization works!")


# ============================================================================
# TEST 2: DELEGATION TOOL CREATION
# ============================================================================


@pytest.mark.asyncio
async def test_delegation_manager_get_delegation_tool():
    """
    Test delegation tool creation.

    This tests that:
    1. get_delegation_tool returns a callable function
    2. Delegation tool can delegate tasks to agents
    3. Tool mapping is used correctly
    """
    print("\n" + "=" * 80)
    print("TEST 2: DelegationManager get delegation tool")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    members = [agent1, agent2]

    tool1 = MockTool("get_data")
    tool_mapping = {"get_data": tool1}

    manager = DelegationManager(members=members, tool_mapping=tool_mapping)
    memory = MockMemory()

    delegation_tool = manager.get_delegation_tool(memory)

    assert callable(delegation_tool), "Should return a callable"

    # Test delegation
    result = await delegation_tool(
        member_id="Agent1",
        description="Test task",
        tools=["get_data"],
        context="Some context",
        attachments=["file.txt"],
    )

    assert "Response from Agent1" in result, "Should return agent response"
    assert "Test task" in result, "Should include task description"

    # Test with invalid member ID
    result_error = await delegation_tool(
        member_id="InvalidAgent", description="Test task"
    )

    assert "not found" in result_error.lower(), "Should return error for invalid agent"

    print("✓ DelegationManager get delegation tool works!")


# ============================================================================
# TEST 3: TASK DELEGATION
# ============================================================================


@pytest.mark.asyncio
async def test_delegation_manager_delegate_task():
    """
    Test task delegation.

    This tests that:
    1. Tasks are properly delegated to agents
    2. Memory is temporarily assigned
    3. Tools are correctly mapped and passed
    """
    print("\n" + "=" * 80)
    print("TEST 3: DelegationManager delegate task")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    members = [agent1, agent2]

    tool1 = MockTool("process_data")
    tool_mapping = {"process_data": tool1}

    manager = DelegationManager(members=members, tool_mapping=tool_mapping)
    memory = MockMemory()

    delegation_tool = manager.get_delegation_tool(memory)

    # Test delegation with tools
    original_memory_agent1 = agent1.memory
    result = await delegation_tool(
        member_id="Agent1",
        description="Process the data",
        tools=["process_data"],
        context={"data": "test"},
        attachments=["data.txt"],
    )

    # Verify memory was temporarily assigned
    assert agent1.memory == original_memory_agent1, "Memory should be restored"
    assert "Response from Agent1" in result, "Should return agent response"

    # Test delegation without tools
    result_no_tools = await delegation_tool(
        member_id="Agent2", description="Simple task"
    )

    assert "Response from Agent2" in result_no_tools, "Should work without tools"

    print("✓ DelegationManager delegate task works!")


# ============================================================================
# TEST 4: TOOL MAPPING
# ============================================================================


def test_delegation_manager_tool_mapping():
    """
    Test tool mapping.

    This tests that:
    1. Tools are correctly mapped by name
    2. Tool mapping is used when delegating tasks
    3. Missing tools are handled gracefully
    """
    print("\n" + "=" * 80)
    print("TEST 4: DelegationManager tool mapping")
    print("=" * 80)

    agent = MockAgent("Agent1")
    members = [agent]

    tool1 = MockTool("tool1")
    tool2 = MockTool("tool2")
    tool_mapping = {"tool1": tool1, "tool2": tool2}

    manager = DelegationManager(members=members, tool_mapping=tool_mapping)

    assert "tool1" in manager.tool_mapping, "Should have tool1 in mapping"
    assert "tool2" in manager.tool_mapping, "Should have tool2 in mapping"
    assert manager.tool_mapping["tool1"] == tool1, "Tool mapping should be correct"

    # Test with empty tool mapping
    manager_empty = DelegationManager(members=members, tool_mapping={})
    assert manager_empty.tool_mapping == {}, "Should handle empty tool mapping"

    print("✓ DelegationManager tool mapping works!")


# ============================================================================
# TEST 5: ROUTING TOOL
# ============================================================================


@pytest.mark.asyncio
async def test_delegation_manager_get_routing_tool():
    """
    Test routing tool creation.

    This tests that:
    1. get_routing_tool returns a callable function
    2. Routing tool selects an agent
    3. routed_agent is set correctly
    """
    print("\n" + "=" * 80)
    print("TEST 5: DelegationManager get routing tool")
    print("=" * 80)

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    members = [agent1, agent2]

    manager = DelegationManager(members=members, tool_mapping={})

    routing_tool = manager.get_routing_tool()

    assert callable(routing_tool), "Should return a callable"

    # Test routing to valid agent
    result = await routing_tool(member_id="Agent1")

    assert manager.routed_agent == agent1, "routed_agent should be set"
    assert "successfully routed" in result.lower(), "Should return success message"

    # Test routing to invalid agent
    manager2 = DelegationManager(members=members, tool_mapping={})
    routing_tool2 = manager2.get_routing_tool()

    result_error = await routing_tool2(member_id="InvalidAgent")

    assert manager2.routed_agent is None, (
        "routed_agent should not be set for invalid agent"
    )
    assert "invalid" in result_error.lower() or "not found" in result_error.lower(), (
        "Should return error"
    )

    print("✓ DelegationManager get routing tool works!")


# ============================================================================
# TEST 6: ERROR HANDLING
# ============================================================================


@pytest.mark.asyncio
async def test_delegation_manager_error_handling():
    """
    Test error handling in delegation.

    This tests that:
    1. Errors during task execution are caught
    2. Error messages are returned
    3. Memory is restored even on error
    """
    print("\n" + "=" * 80)
    print("TEST 6: DelegationManager error handling")
    print("=" * 80)

    agent = MockAgent("Agent1")

    # Create agent that raises error
    async def failing_do_async(task):
        raise Exception("Task execution failed")

    agent.do_async = failing_do_async
    members = [agent]

    manager = DelegationManager(members=members, tool_mapping={})
    memory = MockMemory()

    delegation_tool = manager.get_delegation_tool(memory)

    original_memory = agent.memory
    result = await delegation_tool(member_id="Agent1", description="Failing task")

    # Verify memory was restored even on error
    assert agent.memory == original_memory, "Memory should be restored even on error"
    assert "error occurred" in result.lower(), "Should return error message"

    print("✓ DelegationManager error handling works!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
