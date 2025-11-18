import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Any

from upsonic.team.context_sharing import ContextSharing
from upsonic.tasks.tasks import Task
from upsonic.agent.agent import Agent


# ============================================================================
# MOCK COMPONENTS FOR TESTING
# ============================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "TestAgent"):
        self.name = name
        self.agent_id_ = f"agent-{name.lower()}"

    def get_agent_id(self) -> str:
        return self.name if self.name else f"Agent_{self.agent_id_[:8]}"


# ============================================================================
# TEST 1: CONTEXT SHARING INITIALIZATION
# ============================================================================


def test_context_sharing_initialization():
    """
    Test ContextSharing initialization.

    This tests that:
    1. ContextSharing can be instantiated
    2. Methods are static and callable
    """
    print("\n" + "=" * 80)
    print("TEST 1: ContextSharing initialization")
    print("=" * 80)

    context_sharing = ContextSharing()

    assert context_sharing is not None, "ContextSharing should be instantiated"
    assert hasattr(context_sharing, "enhance_task_context"), (
        "Should have enhance_task_context method"
    )
    assert hasattr(context_sharing, "build_selection_context"), (
        "Should have build_selection_context method"
    )

    print("✓ ContextSharing initialization works!")


# ============================================================================
# TEST 2: ENHANCE TASK CONTEXT
# ============================================================================


def test_context_sharing_share_context():
    """
    Test context sharing.

    This tests that:
    1. Task context is enhanced with other tasks
    2. Agent configurations are added to context
    3. Current task is excluded from context
    """
    print("\n" + "=" * 80)
    print("TEST 2: ContextSharing share context")
    print("=" * 80)

    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")
    task3 = Task(description="Task 3")
    all_tasks = [task1, task2, task3]

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agent_configurations = [agent1, agent2]

    completed_results = []

    # Test enhancing task2 (index 1)
    ContextSharing.enhance_task_context(
        current_task=task2,
        all_tasks=all_tasks,
        task_index=1,
        agent_configurations=agent_configurations,
        completed_results=completed_results,
    )

    assert task2.context is not None, "Context should be set"
    assert isinstance(task2.context, list), "Context should be a list"

    # Verify other tasks are in context (task1 and task3, not task2)
    context_descriptions = [
        item.description if isinstance(item, Task) else str(item)
        for item in task2.context
    ]
    assert "Task 1" in context_descriptions or any(
        "Task 1" in str(item) for item in task2.context
    ), "Should include Task 1"
    assert "Task 3" in context_descriptions or any(
        "Task 3" in str(item) for item in task2.context
    ), "Should include Task 3"
    assert "Task 2" not in [
        item.description for item in task2.context if isinstance(item, Task)
    ], "Should not include current task"

    # Verify agents are in context
    assert len(task2.context) > 2, "Should have multiple context items"

    print("✓ ContextSharing share context works!")


# ============================================================================
# TEST 3: BUILD SELECTION CONTEXT
# ============================================================================


def test_context_sharing_build_selection_context():
    """
    Test build_selection_context method.

    This tests that:
    1. Selection context includes current task
    2. Other tasks are included
    3. Agent configurations are included
    """
    print("\n" + "=" * 80)
    print("TEST 3: ContextSharing build selection context")
    print("=" * 80)

    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")
    task3 = Task(description="Task 3")
    all_tasks = [task1, task2, task3]

    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agent_configurations = [agent1, agent2]

    completed_results = []

    # Build selection context for task2 (index 1)
    context = ContextSharing.build_selection_context(
        current_task=task2,
        all_tasks=all_tasks,
        task_index=1,
        agent_configurations=agent_configurations,
        completed_results=completed_results,
    )

    assert isinstance(context, list), "Context should be a list"
    assert len(context) > 0, "Context should not be empty"

    # Verify current task is first
    assert context[0] == task2, "Current task should be first"

    # Verify other tasks are included
    assert task1 in context, "Should include Task 1"
    assert task3 in context, "Should include Task 3"

    # Verify agents are included
    assert agent1 in context, "Should include Agent1"
    assert agent2 in context, "Should include Agent2"

    print("✓ ContextSharing build selection context works!")


# ============================================================================
# TEST 4: SHARED MEMORY ACCESS
# ============================================================================


def test_context_sharing_shared_memory():
    """
    Test shared memory access.

    This tests that:
    1. Completed results are tracked
    2. Context includes previous results
    3. Memory is shared across tasks
    """
    print("\n" + "=" * 80)
    print("TEST 4: ContextSharing shared memory")
    print("=" * 80)

    task1 = Task(description="Task 1")
    task1._response = "Result 1"
    task2 = Task(description="Task 2")
    task3 = Task(description="Task 3")
    all_tasks = [task1, task2, task3]

    agent1 = MockAgent("Agent1")
    agent_configurations = [agent1]

    # First task completed
    completed_results = [task1]

    # Enhance task2 with completed results
    ContextSharing.enhance_task_context(
        current_task=task2,
        all_tasks=all_tasks,
        task_index=1,
        agent_configurations=agent_configurations,
        completed_results=completed_results,
    )

    # Verify task1 is in context (as it's another task)
    context_has_task1 = any(
        isinstance(item, Task) and item.description == "Task 1"
        for item in task2.context
    )
    assert context_has_task1, "Should include completed Task 1 in context"

    # Build selection context with completed results
    selection_context = ContextSharing.build_selection_context(
        current_task=task3,
        all_tasks=all_tasks,
        task_index=2,
        agent_configurations=agent_configurations,
        completed_results=completed_results,
    )

    # Verify completed results context is available
    assert len(selection_context) > 0, "Selection context should include items"

    print("✓ ContextSharing shared memory works!")


# ============================================================================
# TEST 5: CONTEXT INITIALIZATION
# ============================================================================


def test_context_sharing_context_initialization():
    """
    Test context initialization handling.

    This tests that:
    1. None context is initialized as empty list
    2. Non-list context is converted to list
    3. Existing list context is preserved
    """
    print("\n" + "=" * 80)
    print("TEST 5: ContextSharing context initialization")
    print("=" * 80)

    # Test with None context
    task_none = Task(description="Task with None context")
    task_none.context = None

    ContextSharing.enhance_task_context(
        current_task=task_none,
        all_tasks=[task_none],
        task_index=0,
        agent_configurations=[],
        completed_results=[],
    )

    assert isinstance(task_none.context, list), "None context should become list"

    # Test with non-list context
    task_string = Task(description="Task with string context")
    task_string.context = "String context"

    ContextSharing.enhance_task_context(
        current_task=task_string,
        all_tasks=[task_string],
        task_index=0,
        agent_configurations=[],
        completed_results=[],
    )

    assert isinstance(task_string.context, list), "String context should become list"
    assert "String context" in task_string.context, (
        "Original context should be preserved"
    )

    # Test with existing list context
    task_list = Task(description="Task with list context")
    task_list.context = ["Existing context"]

    ContextSharing.enhance_task_context(
        current_task=task_list,
        all_tasks=[task_list],
        task_index=0,
        agent_configurations=[],
        completed_results=[],
    )

    assert isinstance(task_list.context, list), "List context should remain list"
    assert "Existing context" in task_list.context, (
        "Existing context should be preserved"
    )

    print("✓ ContextSharing context initialization works!")


# ============================================================================
# TEST 6: MULTIPLE TASKS CONTEXT
# ============================================================================


def test_context_sharing_multiple_tasks():
    """
    Test context sharing with multiple tasks.

    This tests that:
    1. All other tasks are included in context
    2. Current task is excluded
    3. Order is preserved
    """
    print("\n" + "=" * 80)
    print("TEST 6: ContextSharing multiple tasks")
    print("=" * 80)

    tasks = [Task(description=f"Task {i}") for i in range(5)]
    agents = [MockAgent(f"Agent{i}") for i in range(3)]

    # Enhance middle task (index 2)
    ContextSharing.enhance_task_context(
        current_task=tasks[2],
        all_tasks=tasks,
        task_index=2,
        agent_configurations=agents,
        completed_results=[],
    )

    # Verify all other tasks are in context
    context_tasks = [item for item in tasks[2].context if isinstance(item, Task)]
    assert len(context_tasks) == 4, "Should include 4 other tasks"

    # Verify current task is not in context
    assert tasks[2] not in context_tasks, "Current task should not be in context"

    # Verify all agents are in context
    context_agents = [item for item in tasks[2].context if isinstance(item, MockAgent)]
    assert len(context_agents) == 3, "Should include all 3 agents"

    print("✓ ContextSharing multiple tasks works!")


# ============================================================================
# TEST 7: EMPTY CONTEXT HANDLING
# ============================================================================


def test_context_sharing_empty_context():
    """
    Test empty context handling.

    This tests that:
    1. Empty task list is handled
    2. Empty agent list is handled
    3. Empty completed results are handled
    """
    print("\n" + "=" * 80)
    print("TEST 7: ContextSharing empty context")
    print("=" * 80)

    task = Task(description="Single task")

    # Test with empty lists
    ContextSharing.enhance_task_context(
        current_task=task,
        all_tasks=[task],
        task_index=0,
        agent_configurations=[],
        completed_results=[],
    )

    assert isinstance(task.context, list), "Context should be a list"
    # Context should only have agents (empty in this case) and other tasks (none)
    assert len(task.context) == 0, "Context should be empty for single task"

    # Build selection context with empty lists
    selection_context = ContextSharing.build_selection_context(
        current_task=task,
        all_tasks=[task],
        task_index=0,
        agent_configurations=[],
        completed_results=[],
    )

    assert isinstance(selection_context, list), "Selection context should be a list"
    assert len(selection_context) == 1, "Should include at least current task"
    assert selection_context[0] == task, "Should include current task"

    print("✓ ContextSharing empty context works!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
