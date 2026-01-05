"""
Test: MCP tool_name_prefix feature for avoiding tool name collisions

Success criteria:
- MCPHandler with tool_name_prefix registers tools with prefixed names
- MultiMCPHandler with tool_name_prefix registers tools with indexed prefixes
- MultiMCPHandler with tool_name_prefixes registers tools with specific prefixes
- Legacy config class with tool_name_prefix attribute works
- Prefixed tools can be added/removed via Agent and Task
- Tools from multiple MCP servers with same names don't collide
- Original tool names are preserved and used when calling MCP server
"""

import pytest
from io import StringIO
from contextlib import redirect_stdout

from upsonic import Agent, Task
from upsonic.tools.mcp import MCPHandler, MultiMCPHandler, MCPTool

pytestmark = pytest.mark.timeout(180)


# ============================================================
# MCPHandler with tool_name_prefix tests
# ============================================================

@pytest.mark.asyncio
async def test_mcp_handler_with_prefix_agent_init():
    """Test MCPHandler with tool_name_prefix in Agent initialization."""
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_prefix_init.db",
        tool_name_prefix="db1",
        timeout_seconds=30
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="MCP Prefix Test Agent",
        tools=[handler],
        debug=True
    )
    
    # Verify tools are registered with prefix
    tool_names = list(agent.registered_agent_tools.keys())
    assert len(tool_names) > 0, "Should have registered MCP tools"
    
    # All tools should have db1_ prefix
    for name in tool_names:
        assert name.startswith("db1_"), f"Tool '{name}' should have 'db1_' prefix"
    
    # Verify specific tools
    assert "db1_read_query" in agent.registered_agent_tools, "db1_read_query should be registered"
    assert "db1_write_query" in agent.registered_agent_tools, "db1_write_query should be registered"
    assert "db1_create_table" in agent.registered_agent_tools, "db1_create_table should be registered"
    
    # Verify handler is in agent.tools
    assert handler in agent.tools, "Handler should be in agent.tools"


@pytest.mark.asyncio
async def test_mcp_handler_with_prefix_agent_add_tools():
    """Test adding MCPHandler with tool_name_prefix via agent.add_tools()."""
    agent = Agent(
        model="openai/gpt-4o",
        name="MCP Prefix Add Test Agent",
        debug=True
    )
    
    # Initially no tools
    assert len(agent.registered_agent_tools) == 0, "Agent should start with no tools"
    
    # Create and add handler with prefix
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_prefix_add.db",
        tool_name_prefix="users_db",
        timeout_seconds=30
    )
    
    agent.add_tools(handler)
    
    # Verify prefixed tools are registered
    prefixed_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("users_db_")]
    assert len(prefixed_tools) > 0, "Should have users_db_ prefixed tools"
    assert "users_db_read_query" in agent.registered_agent_tools, "users_db_read_query should be registered"
    
    # Verify handler is tracked
    assert handler in agent.tools, "Handler should be in agent.tools"


@pytest.mark.asyncio
async def test_mcp_handler_with_prefix_agent_remove_tools():
    """Test removing MCPHandler with tool_name_prefix via agent.remove_tools()."""
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_prefix_remove.db",
        tool_name_prefix="removable_db",
        timeout_seconds=30
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="MCP Prefix Remove Test Agent",
        tools=[handler],
        debug=True
    )
    
    # Verify tools are registered
    prefixed_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("removable_db_")]
    assert len(prefixed_tools) > 0, "Should have prefixed tools initially"
    initial_count = len(prefixed_tools)
    
    # Remove entire handler by object
    agent.remove_tools(handler)
    
    # Verify all tools removed
    remaining_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("removable_db_")]
    assert len(remaining_tools) == 0, "All prefixed tools should be removed"
    assert handler not in agent.tools, "Handler should be removed from agent.tools"


@pytest.mark.asyncio
async def test_mcp_handler_remove_individual_prefixed_tools():
    """Test removing individual prefixed tools by name."""
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_prefix_individual.db",
        tool_name_prefix="partial_db",
        timeout_seconds=30
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="MCP Prefix Individual Remove Agent",
        tools=[handler],
        debug=True
    )
    
    # Get initial count
    initial_tools = list(agent.registered_agent_tools.keys())
    assert len(initial_tools) > 2, "Should have multiple tools"
    
    # Remove one tool by prefixed name
    agent.remove_tools("partial_db_read_query")
    
    # Verify only that tool is removed
    assert "partial_db_read_query" not in agent.registered_agent_tools, "partial_db_read_query should be removed"
    assert "partial_db_write_query" in agent.registered_agent_tools, "partial_db_write_query should remain"
    assert len(agent.registered_agent_tools) == len(initial_tools) - 1, "Should have one less tool"
    
    # Handler should still be in agent.tools (1:many relationship)
    assert handler in agent.tools, "Handler should still be in agent.tools"


# ============================================================
# MultiMCPHandler with tool_name_prefix tests
# ============================================================

@pytest.mark.asyncio
async def test_multi_mcp_handler_with_single_prefix():
    """Test MultiMCPHandler with single tool_name_prefix (auto-indexed)."""
    multi_handler = MultiMCPHandler(
        commands=[
            "uvx mcp-server-sqlite --db-path /tmp/test_multi_single1.db",
            "uvx mcp-server-sqlite --db-path /tmp/test_multi_single2.db",
        ],
        tool_name_prefix="db",  # Will become db_0_* and db_1_*
        timeout_seconds=60
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Multi MCP Single Prefix Agent",
        tools=[multi_handler],
        debug=True
    )
    
    # Verify tools from both servers with indexed prefixes
    db_0_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("db_0_")]
    db_1_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("db_1_")]
    
    assert len(db_0_tools) > 0, "Should have db_0_ prefixed tools from first server"
    assert len(db_1_tools) > 0, "Should have db_1_ prefixed tools from second server"
    
    # Both should have same set of tools (same MCP server type)
    assert len(db_0_tools) == len(db_1_tools), "Both servers should provide same number of tools"
    
    # Verify specific tools exist
    assert "db_0_read_query" in agent.registered_agent_tools, "db_0_read_query should exist"
    assert "db_1_read_query" in agent.registered_agent_tools, "db_1_read_query should exist"


@pytest.mark.asyncio
async def test_multi_mcp_handler_with_prefixes_list():
    """Test MultiMCPHandler with tool_name_prefixes list."""
    multi_handler = MultiMCPHandler(
        commands=[
            "uvx mcp-server-sqlite --db-path /tmp/test_multi_list1.db",
            "uvx mcp-server-sqlite --db-path /tmp/test_multi_list2.db",
        ],
        tool_name_prefixes=["users_db", "posts_db"],  # Specific prefixes
        timeout_seconds=60
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Multi MCP Prefixes List Agent",
        tools=[multi_handler],
        debug=True
    )
    
    # Verify tools from both servers with specific prefixes
    users_db_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("users_db_")]
    posts_db_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("posts_db_")]
    
    assert len(users_db_tools) > 0, "Should have users_db_ prefixed tools"
    assert len(posts_db_tools) > 0, "Should have posts_db_ prefixed tools"
    
    # Verify no collision - both should have separate tools
    assert "users_db_read_query" in agent.registered_agent_tools, "users_db_read_query should exist"
    assert "posts_db_read_query" in agent.registered_agent_tools, "posts_db_read_query should exist"
    
    # These should be different tools, not the same
    users_tool = agent.registered_agent_tools["users_db_read_query"]
    posts_tool = agent.registered_agent_tools["posts_db_read_query"]
    assert users_tool is not posts_tool, "Tools should be different objects"


@pytest.mark.asyncio
async def test_multi_mcp_handler_add_remove():
    """Test adding and removing MultiMCPHandler with prefixes."""
    agent = Agent(
        model="openai/gpt-4o",
        name="Multi MCP Add Remove Agent",
        debug=True
    )
    
    # Initially no tools
    assert len(agent.registered_agent_tools) == 0, "Agent should start with no tools"
    
    # Create and add multi-handler
    multi_handler = MultiMCPHandler(
        commands=[
            "uvx mcp-server-sqlite --db-path /tmp/test_multi_addrem1.db",
            "uvx mcp-server-sqlite --db-path /tmp/test_multi_addrem2.db",
        ],
        tool_name_prefixes=["first_db", "second_db"],
        timeout_seconds=60
    )
    
    agent.add_tools(multi_handler)
    
    # Verify tools are registered
    first_db_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("first_db_")]
    second_db_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("second_db_")]
    
    assert len(first_db_tools) > 0, "Should have first_db_ tools"
    assert len(second_db_tools) > 0, "Should have second_db_ tools"
    
    # Remove entire handler
    agent.remove_tools(multi_handler)
    
    # Verify all tools removed
    remaining_first = [name for name in agent.registered_agent_tools.keys() if name.startswith("first_db_")]
    remaining_second = [name for name in agent.registered_agent_tools.keys() if name.startswith("second_db_")]
    
    assert len(remaining_first) == 0, "All first_db_ tools should be removed"
    assert len(remaining_second) == 0, "All second_db_ tools should be removed"
    assert multi_handler not in agent.tools, "MultiMCPHandler should be removed"


# ============================================================
# Legacy config class with tool_name_prefix tests
# ============================================================

@pytest.mark.asyncio
async def test_legacy_config_class_with_prefix():
    """Test legacy MCP config class with tool_name_prefix attribute."""
    
    class LegacyDatabaseMCP:
        command = "uvx"
        args = ["mcp-server-sqlite", "--db-path", "/tmp/test_legacy_prefix.db"]
        tool_name_prefix = "legacy_db"
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Legacy Config Prefix Agent",
        tools=[LegacyDatabaseMCP],
        debug=True
    )
    
    # Verify tools are registered with prefix
    legacy_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("legacy_db_")]
    assert len(legacy_tools) > 0, "Should have legacy_db_ prefixed tools"
    
    # Verify specific tools
    assert "legacy_db_read_query" in agent.registered_agent_tools, "legacy_db_read_query should be registered"
    assert "legacy_db_write_query" in agent.registered_agent_tools, "legacy_db_write_query should be registered"


# ============================================================
# Task with prefixed MCP handlers tests
# ============================================================

@pytest.mark.asyncio
async def test_task_with_prefixed_mcp_handler():
    """Test Task with prefixed MCPHandler."""
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_task_prefix.db",
        tool_name_prefix="task_db",
        timeout_seconds=30
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Task MCP Prefix Agent",
        debug=True
    )
    
    task = Task(
        description="List all tables in the database using task_db_list_tables tool.",
        tools=[handler]
    )
    
    # Execute task to trigger registration
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    # Verify task has registered prefixed tools
    task_db_tools = [name for name in task.registered_task_tools.keys() if name.startswith("task_db_")]
    assert len(task_db_tools) > 0, "Should have task_db_ prefixed tools"
    assert "task_db_list_tables" in task.registered_task_tools, "task_db_list_tables should be registered"


@pytest.mark.asyncio
async def test_task_add_prefixed_mcp_handler():
    """Test adding prefixed MCPHandler to Task via add_tools."""
    agent = Agent(
        model="openai/gpt-4o",
        name="Task Add MCP Prefix Agent",
        debug=True
    )
    
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_task_add_prefix.db",
        tool_name_prefix="added_db",
        timeout_seconds=30
    )
    
    task = Task(
        description="List tables using added_db_list_tables."
    )
    
    # Add handler to task
    task.add_tools(handler)
    
    # Verify handler is in task.tools
    assert handler in task.tools, "Handler should be in task.tools"
    
    # Execute task to trigger registration
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    # Verify prefixed tools registered
    added_db_tools = [name for name in task.registered_task_tools.keys() if name.startswith("added_db_")]
    assert len(added_db_tools) > 0, "Should have added_db_ prefixed tools"


@pytest.mark.asyncio
async def test_task_remove_prefixed_mcp_handler():
    """Test removing prefixed MCPHandler from Task via remove_tools."""
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_task_remove_prefix.db",
        tool_name_prefix="removable_task_db",
        timeout_seconds=30
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Task Remove MCP Prefix Agent",
        debug=True
    )
    
    task = Task(
        description="List tables.",
        tools=[handler]
    )
    
    # Execute task to trigger registration
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    # Verify tools registered
    initial_tools = [name for name in task.registered_task_tools.keys() if name.startswith("removable_task_db_")]
    assert len(initial_tools) > 0, "Should have prefixed tools initially"
    
    # Remove handler from task
    task.remove_tools(handler, agent)
    
    # Verify tools removed
    remaining_tools = [name for name in task.registered_task_tools.keys() if name.startswith("removable_task_db_")]
    assert len(remaining_tools) == 0, "All prefixed tools should be removed"
    assert handler not in task.tools, "Handler should be removed from task.tools"


# ============================================================
# MCPTool internal behavior tests
# ============================================================

@pytest.mark.asyncio
async def test_mcp_tool_stores_original_name():
    """Test that MCPTool stores original name for MCP server calls."""
    from unittest.mock import MagicMock
    
    # Create mock handler and tool_info
    mock_handler = MagicMock()
    mock_handler.server_name = "test_server"
    mock_handler.connection_type = "stdio"
    mock_handler.transport = "stdio"
    
    mock_tool_info = MagicMock()
    mock_tool_info.name = "read_query"
    mock_tool_info.description = "Execute a query"
    mock_tool_info.inputSchema = {"type": "object", "properties": {}}
    
    # Create MCPTool with prefix
    tool = MCPTool(mock_handler, mock_tool_info, tool_name_prefix="test_db")
    
    # Verify names
    assert tool.name == "test_db_read_query", "Prefixed name should be 'test_db_read_query'"
    assert tool.original_name == "read_query", "Original name should be 'read_query'"
    
    # Verify metadata
    assert tool.metadata.custom.get("mcp_original_name") == "read_query", "Metadata should store original name"
    assert tool.metadata.custom.get("mcp_tool_name_prefix") == "test_db", "Metadata should store prefix"


@pytest.mark.asyncio
async def test_mcp_tool_without_prefix():
    """Test that MCPTool without prefix uses original name."""
    from unittest.mock import MagicMock
    
    # Create mock handler and tool_info
    mock_handler = MagicMock()
    mock_handler.server_name = "test_server"
    mock_handler.connection_type = "stdio"
    mock_handler.transport = "stdio"
    
    mock_tool_info = MagicMock()
    mock_tool_info.name = "write_query"
    mock_tool_info.description = "Write to database"
    mock_tool_info.inputSchema = {"type": "object", "properties": {}}
    
    # Create MCPTool without prefix
    tool = MCPTool(mock_handler, mock_tool_info, tool_name_prefix=None)
    
    # Verify names
    assert tool.name == "write_query", "Name should be 'write_query' (no prefix)"
    assert tool.original_name == "write_query", "Original name should be 'write_query'"
    
    # Verify metadata doesn't have prefix
    assert tool.metadata.custom.get("mcp_tool_name_prefix") is None, "Metadata should not have prefix"


# ============================================================
# No collision verification tests
# ============================================================

@pytest.mark.asyncio
async def test_no_tool_collision_with_prefixes():
    """Test that prefixed tools from same MCP server type don't collide."""
    handler1 = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_no_collision1.db",
        tool_name_prefix="db_one",
        timeout_seconds=30
    )
    
    handler2 = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_no_collision2.db",
        tool_name_prefix="db_two",
        timeout_seconds=30
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="No Collision Test Agent",
        tools=[handler1, handler2],
        debug=True
    )
    
    # Verify both sets of tools exist
    db_one_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("db_one_")]
    db_two_tools = [name for name in agent.registered_agent_tools.keys() if name.startswith("db_two_")]
    
    assert len(db_one_tools) > 0, "Should have db_one_ tools"
    assert len(db_two_tools) > 0, "Should have db_two_ tools"
    
    # Both should have same number of tools
    assert len(db_one_tools) == len(db_two_tools), "Both handlers should provide same number of tools"
    
    # Total should be sum of both (no collision)
    total_tools = len(agent.registered_agent_tools)
    expected_total = len(db_one_tools) + len(db_two_tools)
    assert total_tools == expected_total, f"Total tools ({total_tools}) should equal sum ({expected_total})"
    
    # Verify specific tools from each handler
    assert "db_one_read_query" in agent.registered_agent_tools, "db_one_read_query should exist"
    assert "db_two_read_query" in agent.registered_agent_tools, "db_two_read_query should exist"
    
    # Verify they are different tool objects
    tool1 = agent.registered_agent_tools["db_one_read_query"]
    tool2 = agent.registered_agent_tools["db_two_read_query"]
    assert tool1 is not tool2, "Should be different tool objects"
    assert tool1.handler is not tool2.handler, "Should have different handlers"


@pytest.mark.asyncio
async def test_collision_without_prefixes():
    """Test that tools without prefixes would collide (expected behavior)."""
    handler1 = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_collision1.db",
        timeout_seconds=30
        # No prefix!
    )
    
    handler2 = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_collision2.db",
        timeout_seconds=30
        # No prefix!
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Collision Test Agent",
        tools=[handler1, handler2],
        debug=True
    )
    
    # Both handlers provide same tool names, so second overwrites first
    # This is expected behavior when no prefix is used
    
    # Should only have 6 tools (SQLite MCP provides 6 tools)
    # The second handler's tools overwrite the first handler's tools
    assert len(agent.registered_agent_tools) <= 6, "Should have at most 6 tools (collision expected)"
    
    # read_query should exist but only one version
    assert "read_query" in agent.registered_agent_tools, "read_query should exist"


# ============================================================
# Handler info tests
# ============================================================

@pytest.mark.asyncio
async def test_mcp_handler_get_info_with_prefix():
    """Test that MCPHandler.get_info() includes prefix information."""
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_info_prefix.db",
        tool_name_prefix="info_db",
        timeout_seconds=30
    )
    
    # Connect to populate tools
    agent = Agent(
        model="openai/gpt-4o",
        name="Handler Info Test Agent",
        tools=[handler],
        debug=True
    )
    
    # Get handler info
    info = handler.get_info()
    
    # Verify info includes prefix
    assert info.get("tool_name_prefix") == "info_db", "Info should include tool_name_prefix"
    assert "original_tool_names" in info, "Info should include original_tool_names when prefix is used"
    
    # Verify tools are prefixed
    tools_list = info.get("tools", [])
    for tool_name in tools_list:
        assert tool_name.startswith("info_db_"), f"Tool '{tool_name}' should have prefix"
    
    # Verify original names are preserved
    original_names = info.get("original_tool_names", [])
    assert len(original_names) > 0, "Should have original tool names"
    assert "read_query" in original_names, "Original name 'read_query' should be preserved"


@pytest.mark.asyncio
async def test_multi_mcp_handler_get_server_info():
    """Test that MultiMCPHandler.get_server_info() includes prefix information."""
    multi_handler = MultiMCPHandler(
        commands=[
            "uvx mcp-server-sqlite --db-path /tmp/test_server_info1.db",
            "uvx mcp-server-sqlite --db-path /tmp/test_server_info2.db",
        ],
        tool_name_prefixes=["alpha_db", "beta_db"],
        timeout_seconds=60
    )
    
    # Connect to populate tools
    agent = Agent(
        model="openai/gpt-4o",
        name="Multi Handler Info Test Agent",
        tools=[multi_handler],
        debug=True
    )
    
    # Get server info
    server_info = multi_handler.get_server_info()
    
    assert len(server_info) == 2, "Should have info for 2 servers"
    
    # First server should have alpha_db prefix
    assert server_info[0].get("tool_name_prefix") == "alpha_db", "First server should have alpha_db prefix"
    assert "original_tool_names" in server_info[0], "Should include original tool names"
    
    # Second server should have beta_db prefix
    assert server_info[1].get("tool_name_prefix") == "beta_db", "Second server should have beta_db prefix"
    assert "original_tool_names" in server_info[1], "Should include original tool names"


# ============================================================
# ToolProcessor tracking tests
# ============================================================

@pytest.mark.asyncio
async def test_mcp_handler_to_tools_tracking_with_prefix():
    """Test that ToolProcessor correctly tracks prefixed tool names."""
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_tracking_prefix.db",
        tool_name_prefix="tracked_db",
        timeout_seconds=30
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Tracking Test Agent",
        tools=[handler],
        debug=True
    )
    
    processor = agent.tool_manager.processor
    handler_id = id(handler)
    
    # Verify handler is tracked
    assert handler_id in processor.mcp_handler_to_tools, "Handler should be tracked"
    
    # Verify tracked tool names are prefixed
    tracked_tools = processor.mcp_handler_to_tools[handler_id]
    assert len(tracked_tools) > 0, "Should have tracked tools"
    
    for tool_name in tracked_tools:
        assert tool_name.startswith("tracked_db_"), f"Tracked tool '{tool_name}' should have prefix"


@pytest.mark.asyncio
async def test_cleanup_on_prefixed_handler_removal():
    """Test that processor tracking is cleaned up when prefixed handler is removed."""
    handler = MCPHandler(
        command="uvx mcp-server-sqlite --db-path /tmp/test_cleanup_prefix.db",
        tool_name_prefix="cleanup_db",
        timeout_seconds=30
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Cleanup Test Agent",
        tools=[handler],
        debug=True
    )
    
    processor = agent.tool_manager.processor
    handler_id = id(handler)
    
    # Verify handler is tracked initially
    assert handler_id in processor.mcp_handler_to_tools, "Handler should be tracked initially"
    assert handler in processor.mcp_handlers, "Handler should be in mcp_handlers list"
    
    # Remove handler
    agent.remove_tools(handler)
    
    # Verify tracking is cleaned up
    assert handler_id not in processor.mcp_handler_to_tools, "Handler tracking should be cleaned up"
    assert handler not in processor.mcp_handlers, "Handler should be removed from mcp_handlers list"


# ============================================================
# Full execution test
# ============================================================

@pytest.mark.asyncio
async def test_prefixed_tools_execution():
    """Test that prefixed tools can be executed correctly by the agent."""
    multi_handler = MultiMCPHandler(
        commands=[
            "uvx mcp-server-sqlite --db-path /tmp/test_exec_users.db",
            "uvx mcp-server-sqlite --db-path /tmp/test_exec_posts.db",
        ],
        tool_name_prefixes=["users_db", "posts_db"],
        timeout_seconds=60
    )
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Multi-DB Execution Agent",
        tool_call_limit=15,
        debug=True
    )
    
    task = Task(
        description="""
        You have access to TWO separate SQLite databases with PREFIXED tool names:
        - Database 1 (users_db): Tools are prefixed with "users_db_" 
        - Database 2 (posts_db): Tools are prefixed with "posts_db_"
        
        Step 1: Using users_db_create_table, create a 'test_users' table with id (integer primary key) and name (text).
        Step 2: Using posts_db_create_table, create a 'test_posts' table with id (integer primary key) and title (text).
        Step 3: Using users_db_list_tables, show tables in first database.
        Step 4: Using posts_db_list_tables, show tables in second database.
        """,
        tools=[multi_handler]
    )
    
    # Execute task
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    
    # Verify prefixed tools were called (check output logs)
    assert "users_db_" in output or "posts_db_" in output, "Should see prefixed tool calls in output"
    
    # Verify tool calls
    assert len(task.tool_calls) > 0, "Should have tool calls"
    
    # Check that prefixed tool names were used
    tool_names_called = [call.get("tool_name", "") for call in task.tool_calls]
    users_db_calls = [name for name in tool_names_called if name.startswith("users_db_")]
    posts_db_calls = [name for name in tool_names_called if name.startswith("posts_db_")]
    
    assert len(users_db_calls) > 0 or len(posts_db_calls) > 0, "Should have calls to prefixed tools"


# ============================================================
# Prefix validation tests
# ============================================================

@pytest.mark.asyncio
async def test_multi_mcp_handler_prefixes_length_validation():
    """Test that tool_name_prefixes length must match server count."""
    # This should work - 2 commands, 2 prefixes
    valid_handler = MultiMCPHandler(
        commands=[
            "uvx mcp-server-sqlite --db-path /tmp/test_valid1.db",
            "uvx mcp-server-sqlite --db-path /tmp/test_valid2.db",
        ],
        tool_name_prefixes=["valid1", "valid2"],
        timeout_seconds=30
    )
    
    # This should fail during connect - 2 commands, 3 prefixes
    invalid_handler = MultiMCPHandler(
        commands=[
            "uvx mcp-server-sqlite --db-path /tmp/test_invalid1.db",
            "uvx mcp-server-sqlite --db-path /tmp/test_invalid2.db",
        ],
        tool_name_prefixes=["prefix1", "prefix2", "prefix3"],  # Wrong length!
        timeout_seconds=30
    )
    
    # Attempting to connect should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        await invalid_handler.connect()
    
    assert "tool_name_prefixes length" in str(exc_info.value), "Should mention prefix length mismatch"

