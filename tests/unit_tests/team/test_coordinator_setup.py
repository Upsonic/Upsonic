import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from upsonic.team.coordinator_setup import CoordinatorSetup
from upsonic.tasks.tasks import Task
from upsonic.agent.agent import Agent


# ============================================================================
# MOCK COMPONENTS FOR TESTING
# ============================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        name: str = "TestAgent",
        role: str = None,
        goal: str = None,
        system_prompt: str = None,
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.system_prompt = system_prompt
        self.agent_id_ = f"agent-{name.lower()}"

    def get_agent_id(self) -> str:
        return self.name if self.name else f"Agent_{self.agent_id_[:8]}"


class MockTool:
    """Mock tool for testing."""

    def __init__(
        self, name: str = "test_tool", docstring: str = "Test tool description"
    ):
        self.__name__ = name
        self.__doc__ = docstring


# ============================================================================
# TEST 1: COORDINATOR SETUP INITIALIZATION
# ============================================================================


def test_coordinator_setup_initialization():
    """
    Test CoordinatorSetup initialization.

    This tests that:
    1. CoordinatorSetup can be initialized with members and tasks
    2. Mode is properly stored
    3. All attributes are set correctly
    """
    print("\n" + "=" * 80)
    print("TEST 1: CoordinatorSetup initialization")
    print("=" * 80)

    agent1 = MockAgent("Agent1", role="Researcher", goal="Research tasks")
    agent2 = MockAgent("Agent2", role="Writer", goal="Write content")
    members = [agent1, agent2]

    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")
    tasks = [task1, task2]

    # Test coordinate mode
    setup = CoordinatorSetup(members=members, tasks=tasks, mode="coordinate")

    assert setup.members == members, "Members should be set"
    assert setup.tasks == tasks, "Tasks should be set"
    assert setup.mode == "coordinate", "Mode should be coordinate"

    # Test route mode
    setup_route = CoordinatorSetup(members=members, tasks=tasks, mode="route")
    assert setup_route.mode == "route", "Mode should be route"

    print("✓ CoordinatorSetup initialization works!")


# ============================================================================
# TEST 2: AGENT MANIFEST FORMATTING
# ============================================================================


def test_coordinator_setup_format_agent_manifest():
    """
    Test agent manifest formatting.

    This tests that:
    1. Agent manifest is formatted correctly
    2. Includes agent ID, role, goal, and system prompt
    3. Handles missing information gracefully
    """
    print("\n" + "=" * 80)
    print("TEST 2: CoordinatorSetup format agent manifest")
    print("=" * 80)

    # Test with full agent info
    agent1 = MockAgent(
        "Researcher",
        role="Research Expert",
        goal="Find information",
        system_prompt="You are a research expert",
    )
    agent2 = MockAgent(
        "Writer",
        role="Content Writer",
        goal="Write articles",
        system_prompt="You are a content writer",
    )
    members = [agent1, agent2]

    setup = CoordinatorSetup(members=members, tasks=[], mode="coordinate")
    manifest = setup._format_agent_manifest()

    assert "Researcher" in manifest, "Should include agent name"
    assert "Research Expert" in manifest, "Should include role"
    assert "Find information" in manifest, "Should include goal"
    assert "You are a research expert" in manifest, "Should include system prompt"

    # Test with minimal agent info
    agent_minimal = MockAgent("MinimalAgent")
    setup_minimal = CoordinatorSetup(
        members=[agent_minimal], tasks=[], mode="coordinate"
    )
    manifest_minimal = setup_minimal._format_agent_manifest()

    assert "MinimalAgent" in manifest_minimal, "Should include agent name"
    assert "No specific role defined" in manifest_minimal, "Should handle missing role"

    # Test with empty members
    setup_empty = CoordinatorSetup(members=[], tasks=[], mode="coordinate")
    manifest_empty = setup_empty._format_agent_manifest()

    assert manifest_empty == "No team members are available.", (
        "Should handle empty members"
    )

    print("✓ CoordinatorSetup format agent manifest works!")


# ============================================================================
# TEST 3: TASKS MANIFEST FORMATTING
# ============================================================================


def test_coordinator_setup_format_tasks_manifest():
    """
    Test tasks manifest formatting.

    This tests that:
    1. Tasks manifest is formatted correctly
    2. Includes description, tools, context, and attachments
    3. Handles missing information gracefully
    """
    print("\n" + "=" * 80)
    print("TEST 3: CoordinatorSetup format tasks manifest")
    print("=" * 80)

    # Test with full task info
    tool1 = MockTool("get_data", "Fetches data from API")
    tool2 = MockTool("process_data", "Processes the data")

    task1 = Task(
        description="Fetch and process data",
        tools=[tool1, tool2],
        context=["context1", "context2"],
        attachments=["file1.txt", "file2.txt"],
    )

    setup = CoordinatorSetup(members=[], tasks=[task1], mode="coordinate")
    manifest = setup._format_tasks_manifest()

    assert "Fetch and process data" in manifest, "Should include description"
    assert "get_data" in manifest, "Should include tool names"
    assert "Fetches data from API" in manifest, "Should include tool descriptions"
    assert "context1" in manifest, "Should include context"
    assert "file1.txt" in manifest, "Should include attachments"

    # Test with minimal task info
    task_minimal = Task(description="Simple task")
    setup_minimal = CoordinatorSetup(
        members=[], tasks=[task_minimal], mode="coordinate"
    )
    manifest_minimal = setup_minimal._format_tasks_manifest()

    assert "Simple task" in manifest_minimal, "Should include description"
    assert "None</Tools>" in manifest_minimal, "Should handle missing tools"

    # Test with empty tasks
    setup_empty = CoordinatorSetup(members=[], tasks=[], mode="coordinate")
    manifest_empty = setup_empty._format_tasks_manifest()

    assert "No initial tasks provided" in manifest_empty, "Should handle empty tasks"

    # Test with multiple tasks
    task2 = Task(description="Second task", tools=[tool1])
    setup_multiple = CoordinatorSetup(
        members=[], tasks=[task1, task2], mode="coordinate"
    )
    manifest_multiple = setup_multiple._format_tasks_manifest()

    assert manifest_multiple.count("<Task index=") == 2, "Should include both tasks"

    print("✓ CoordinatorSetup format tasks manifest works!")


# ============================================================================
# TEST 4: PROMPT CREATION
# ============================================================================


def test_coordinator_setup_create_coordinate_prompt():
    """
    Test coordinate prompt creation.

    This tests that:
    1. Coordinate prompt is created correctly
    2. Includes agent manifest and tasks manifest
    3. Contains delegation instructions
    """
    print("\n" + "=" * 80)
    print("TEST 4: CoordinatorSetup create coordinate prompt")
    print("=" * 80)

    agent = MockAgent("Agent1", role="Researcher", goal="Research")
    task = Task(description="Research task")

    setup = CoordinatorSetup(members=[agent], tasks=[task], mode="coordinate")
    prompt = setup.create_leader_prompt()

    assert "Strategic Coordinator" in prompt, "Should identify as Strategic Coordinator"
    assert "Agent1" in prompt, "Should include agent manifest"
    assert "Research task" in prompt, "Should include tasks manifest"
    assert "delegate_task" in prompt, "Should mention delegate_task tool"
    assert "TEAM ROSTER" in prompt, "Should include team roster section"
    assert "MISSION OBJECTIVES" in prompt, "Should include mission objectives section"

    print("✓ CoordinatorSetup create coordinate prompt works!")


def test_coordinator_setup_create_route_prompt():
    """
    Test route prompt creation.

    This tests that:
    1. Route prompt is created correctly
    2. Includes agent manifest and tasks manifest
    3. Contains routing instructions
    """
    print("\n" + "=" * 80)
    print("TEST 5: CoordinatorSetup create route prompt")
    print("=" * 80)

    agent = MockAgent("Agent1", role="Researcher", goal="Research")
    task = Task(description="Route task")

    setup = CoordinatorSetup(members=[agent], tasks=[task], mode="route")
    prompt = setup.create_leader_prompt()

    assert "AI Router" in prompt, "Should identify as AI Router"
    assert "Agent1" in prompt, "Should include agent manifest"
    assert "Route task" in prompt, "Should include tasks manifest"
    assert "route_request_to_member" in prompt, "Should mention routing tool"
    assert "TEAM ROSTER" in prompt, "Should include team roster section"
    assert "MISSION OBJECTIVES" in prompt, "Should include mission objectives section"

    print("✓ CoordinatorSetup create route prompt works!")


# ============================================================================
# TEST 5: TOOL SUMMARIZATION
# ============================================================================


def test_coordinator_setup_summarize_tool():
    """
    Test tool summarization.

    This tests that:
    1. Tools are summarized correctly from docstrings
    2. Handles tools without docstrings
    3. Handles unnamed tools
    """
    print("\n" + "=" * 80)
    print("TEST 6: CoordinatorSetup summarize tool")
    print("=" * 80)

    setup = CoordinatorSetup(members=[], tasks=[], mode="coordinate")

    # Test with docstring
    tool_with_doc = MockTool("test_tool", "This tool does something useful")
    summary = setup._summarize_tool(tool_with_doc)

    assert "test_tool" in summary, "Should include tool name"
    assert "This tool does something useful" in summary, "Should include docstring"

    # Test without docstring
    tool_no_doc = MockTool("no_doc_tool", None)
    tool_no_doc.__doc__ = None
    summary_no_doc = setup._summarize_tool(tool_no_doc)

    assert "no_doc_tool" in summary_no_doc, "Should include tool name"
    assert "No description available" in summary_no_doc, (
        "Should handle missing docstring"
    )

    # Test with unnamed tool
    tool_unnamed = Mock()
    tool_unnamed.__name__ = "Unnamed Tool"
    tool_unnamed.__doc__ = "Some description"
    summary_unnamed = setup._summarize_tool(tool_unnamed)

    assert "Unnamed Tool" in summary_unnamed, "Should handle unnamed tool"

    print("✓ CoordinatorSetup summarize tool works!")


# ============================================================================
# TEST 6: CONTEXT SERIALIZATION
# ============================================================================


def test_coordinator_setup_serialize_context():
    """
    Test context item serialization.

    This tests that:
    1. Different context types are serialized correctly
    2. Handles strings, Tasks, and other objects
    """
    print("\n" + "=" * 80)
    print("TEST 7: CoordinatorSetup serialize context")
    print("=" * 80)

    setup = CoordinatorSetup(members=[], tasks=[], mode="coordinate")

    # Test string context
    context_str = "Simple string context"
    serialized = setup._serialize_context_item(context_str)
    assert serialized == context_str, "Should return string as-is"

    # Test Task context
    task_context = Task(description="Context task")
    serialized_task = setup._serialize_context_item(task_context)
    assert "Context task" in serialized_task, "Should serialize Task description"
    assert "Reference to another task" in serialized_task, (
        "Should indicate Task reference"
    )

    # Test other object
    class CustomObject:
        def __str__(self):
            return "Custom object"

    custom_obj = CustomObject()
    serialized_obj = setup._serialize_context_item(custom_obj)
    assert "Custom object" in serialized_obj, "Should serialize using str()"

    print("✓ CoordinatorSetup serialize context works!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
