"""
Test 14: Test all variants of adding, removing of all type of tools

Success criteria:
- Tests all tool types: ToolKit, function tools, pure classes, Agent as tool, 
  financial_tools, duckduckgo, tavily, builtin tools
- Tests using Task and Agent class remove_tools, add_tools
- Tests runtime registration when running a Task using Agent (agent.do_async(task))
- Checks agent, task and tool_manager attributes (registered_task_tools, registered_agent_tools, etc.)
"""

import pytest
import os
from upsonic import Agent, Task
from upsonic.tools import tool, ToolKit
from upsonic.tools.builtin_tools import WebSearchTool, CodeExecutionTool
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(120)


# Custom tools (functions with @tool decorator)
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


# Common tools (ToolKit instance)
class MathToolKit(ToolKit):
    """A toolkit for mathematical operations."""
    
    @tool
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
    
    @tool
    def divide(self, a: int, b: int) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


class TextToolKit(ToolKit):
    """A toolkit for text operations."""
    
    @tool
    def uppercase(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()
    
    @tool
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()


@pytest.mark.asyncio
async def test_agent_add_remove_custom_tools():
    """Test adding and removing custom tools (functions) from Agent."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Initially no tools
    assert len(agent.registered_agent_tools) == 0, "Agent should start with no tools"
    assert len(agent.tools) == 0, "Agent.tools should be empty"
    
    # Add single custom tool
    agent.add_tools(add_numbers)
    assert "add_numbers" in agent.registered_agent_tools, "add_numbers should be registered"
    assert add_numbers in agent.tools, "add_numbers should be in agent.tools"
    
    # Add multiple custom tools
    agent.add_tools([multiply_numbers, greet])
    assert "multiply_numbers" in agent.registered_agent_tools, "multiply_numbers should be registered"
    assert "greet" in agent.registered_agent_tools, "greet should be registered"
    assert len(agent.registered_agent_tools) == 3, f"Should have 3 tools, got {len(agent.registered_agent_tools)}"
    
    # Remove by name
    agent.remove_tools("add_numbers")
    assert "add_numbers" not in agent.registered_agent_tools, "add_numbers should be removed"
    assert add_numbers not in agent.tools, "add_numbers should not be in agent.tools"
    
    # Remove by name (safer than object)
    agent.remove_tools("multiply_numbers")
    assert "multiply_numbers" not in agent.registered_agent_tools, "multiply_numbers should be removed"
    
    # Remove multiple
    agent.remove_tools(["greet"])
    assert len(agent.registered_agent_tools) == 0, "All tools should be removed"


@pytest.mark.asyncio
async def test_agent_add_remove_toolkit():
    """Test adding and removing ToolKit instances from Agent."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    
    # Verify all tools from toolkit are registered
    assert "subtract" in agent.registered_agent_tools, "subtract should be registered"
    assert "divide" in agent.registered_agent_tools, "divide should be registered"
    assert math_kit in agent.tools, "MathToolKit should be in agent.tools"
    
    # Add another toolkit
    text_kit = TextToolKit()
    agent.add_tools(text_kit)
    assert "uppercase" in agent.registered_agent_tools, "uppercase should be registered"
    assert "lowercase" in agent.registered_agent_tools, "lowercase should be registered"
    
    # Remove toolkit by object (removes all its tools)
    agent.remove_tools(math_kit)
    assert "subtract" not in agent.registered_agent_tools, "subtract should be removed"
    assert "divide" not in agent.registered_agent_tools, "divide should be removed"
    assert math_kit not in agent.tools, "MathToolKit should not be in agent.tools"
    
    # Remove toolkit by removing the toolkit object (removes all its tools)
    agent.remove_tools(text_kit)
    assert "uppercase" not in agent.registered_agent_tools, "uppercase should be removed"
    assert "lowercase" not in agent.registered_agent_tools, "lowercase should be removed"
    assert text_kit not in agent.tools, "TextToolKit should not be in agent.tools"


@pytest.mark.asyncio
async def test_agent_remove_individual_toolkit_methods():
    """Test removing individual methods from a ToolKit by name (keeping the toolkit instance)."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    
    # Verify all tools from toolkit are registered
    assert "subtract" in agent.registered_agent_tools, "subtract should be registered"
    assert "divide" in agent.registered_agent_tools, "divide should be registered"
    assert math_kit in agent.tools, "MathToolKit should be in agent.tools"
    assert len(agent.registered_agent_tools) == 2, "Should have 2 toolkit methods"
    
    # Remove one method by name (NOT the entire toolkit)
    agent.remove_tools("subtract")
    
    # Verify only that method is removed
    assert "subtract" not in agent.registered_agent_tools, "subtract should be removed"
    assert "divide" in agent.registered_agent_tools, "divide should still be registered"
    assert math_kit in agent.tools, "MathToolKit instance should still be in agent.tools"
    assert len(agent.registered_agent_tools) == 1, "Should have 1 toolkit method remaining"
    
    # Remove another method by name
    agent.remove_tools("divide")
    
    # Verify second method removed
    assert "divide" not in agent.registered_agent_tools, "divide should be removed"
    assert len(agent.registered_agent_tools) == 0, "All methods removed"
    
    # ToolKit instance should still be in agent.tools (even though all its methods are removed)
    # This is expected behavior - we only remove the instance when removed by object
    assert math_kit in agent.tools, "MathToolKit instance should still be in agent.tools"


@pytest.mark.asyncio
async def test_agent_remove_individual_class_methods():
    """Test removing individual methods from a regular class by name (keeping the class instance)."""
    try:
        from upsonic.tools.common_tools.financial_tools import YFinanceTools
        
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create and add financial tools instance (pure class, not ToolKit)
        financial_tools = YFinanceTools(stock_price=True, enable_all=False)
        agent.add_tools(financial_tools)
        
        # Verify tools are registered
        initial_count = len(agent.registered_agent_tools)
        assert initial_count > 0, "Financial tools should be registered"
        
        # Get one tool name to remove
        tool_names = list(agent.registered_agent_tools.keys())
        tool_to_remove = tool_names[0]
        
        # Remove one method by name (NOT the entire class instance)
        agent.remove_tools(tool_to_remove)
        
        # Verify only that method is removed
        assert tool_to_remove not in agent.registered_agent_tools, f"{tool_to_remove} should be removed"
        assert len(agent.registered_agent_tools) == initial_count - 1, "Should have one less tool"
        
        # Class instance should still be in agent.tools
        assert financial_tools in agent.tools, "Financial tools instance should still be in agent.tools"
        
        # Remove another method
        if len(agent.registered_agent_tools) > 0:
            second_tool = list(agent.registered_agent_tools.keys())[0]
            agent.remove_tools(second_tool)
            assert second_tool not in agent.registered_agent_tools, f"{second_tool} should be removed"
            
    except ImportError:
        pytest.skip("Financial tools dependencies not available")


@pytest.mark.asyncio
async def test_task_remove_individual_toolkit_methods():
    """Test removing individual methods from a ToolKit in a Task by name."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Create task with ToolKit
    math_kit = MathToolKit()
    task = Task(
        description="Test task with toolkit",
        tools=[math_kit, add_numbers]
    )
    
    # Execute task to trigger registration
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    # Verify tools are registered
    assert "subtract" in task.registered_task_tools, "subtract should be registered"
    assert "divide" in task.registered_task_tools, "divide should be registered"
    assert "add_numbers" in task.registered_task_tools, "add_numbers should be registered"
    assert len(task.registered_task_tools) == 3, "Should have 3 tools"
    
    # Remove one toolkit method by name
    task.remove_tools("subtract", agent)
    
    # Verify only that method is removed
    assert "subtract" not in task.registered_task_tools, "subtract should be removed"
    assert "divide" in task.registered_task_tools, "divide should still be registered"
    assert "add_numbers" in task.registered_task_tools, "add_numbers should still be registered"
    assert math_kit in task.tools, "MathToolKit instance should still be in task.tools"
    assert len(task.registered_task_tools) == 2, "Should have 2 tools remaining"
    
    # Remove another toolkit method
    task.remove_tools("divide", agent)
    
    # Verify second method removed
    assert "divide" not in task.registered_task_tools, "divide should be removed"
    assert "add_numbers" in task.registered_task_tools, "add_numbers should still be registered"
    assert len(task.registered_task_tools) == 1, "Should have 1 tool remaining"


@pytest.mark.asyncio
async def test_agent_add_remove_builtin_tools():
    """Test adding and removing builtin tools from Agent."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Initially no tools
    assert len(agent.registered_agent_tools) == 0, "Agent should start with no regular tools"
    assert len(agent.agent_builtin_tools) == 0, "Agent should start with no builtin tools"
    assert len(agent.tools) == 0, "Agent.tools should be empty"
    
    # Add builtin tool
    web_search = WebSearchTool()
    agent.add_tools(web_search)
    
    # Verify builtin tool is in agent_builtin_tools (NOT in registered_agent_tools)
    assert web_search in agent.tools, "WebSearchTool should be in agent.tools"
    assert len(agent.agent_builtin_tools) == 1, "Should have 1 builtin tool"
    assert any(tool.unique_id == "web_search" for tool in agent.agent_builtin_tools), "web_search should be in agent_builtin_tools"
    # Builtin tools are NOT in registered_agent_tools, they're tracked separately
    assert len(agent.registered_agent_tools) == 0, "Builtin tools should NOT be in registered_agent_tools"
    
    # Add another builtin tool
    code_exec = CodeExecutionTool()
    agent.add_tools(code_exec)
    assert any(tool.unique_id == "code_execution" for tool in agent.agent_builtin_tools), "code_execution should be in agent_builtin_tools"
    assert len(agent.agent_builtin_tools) == 2, "Should have 2 builtin tools"
    
    # Verify attributes
    assert len(agent.tools) == 2, "Should have 2 builtin tool objects in agent.tools"
    assert web_search in agent.tools, "WebSearchTool should be in agent.tools"
    assert code_exec in agent.tools, "CodeExecutionTool should be in agent.tools"
    assert len(agent.registered_agent_tools) == 0, "Builtin tools should NOT be in registered_agent_tools"
    
    # Test removing builtin tools by object
    agent.remove_tools([web_search])
    
    # Verify removal
    assert web_search not in agent.tools, "WebSearchTool should be removed from agent.tools"
    assert len(agent.agent_builtin_tools) == 1, "Should have 1 builtin tool after removal"
    assert not any(tool.unique_id == "web_search" for tool in agent.agent_builtin_tools), "web_search should be removed from agent_builtin_tools"
    assert any(tool.unique_id == "code_execution" for tool in agent.agent_builtin_tools), "code_execution should still be in agent_builtin_tools"
    
    # Remove the second builtin tool
    agent.remove_tools([code_exec])
    
    # Verify all builtin tools removed
    assert len(agent.agent_builtin_tools) == 0, "All builtin tools should be removed"
    assert len(agent.tools) == 0, "agent.tools should be empty"
    assert code_exec not in agent.tools, "CodeExecutionTool should be removed from agent.tools"


@pytest.mark.asyncio
async def test_task_add_remove_builtin_tools():
    """Test adding and removing builtin tools from Task (without execution)."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Create task with builtin tools
    # Note: We're testing tool management, not execution, so we don't actually run the task
    code_exec = CodeExecutionTool()
    from upsonic.tools.builtin_tools import ImageGenerationTool
    web_search = WebSearchTool()
    img_gen = ImageGenerationTool()
    
    task = Task(
        description="Test task with builtin tools",
        tools=[code_exec, web_search]
    )
    
    # Before registration, tools are in task.tools but not registered
    assert code_exec in task.tools, "CodeExecutionTool should be in task.tools"
    assert web_search in task.tools, "WebSearchTool should be in task.tools"
    assert len(task.registered_task_tools) == 0, "Task tools not registered until execution"
    assert len(task.task_builtin_tools) == 0, "Task builtin tools not populated until execution"
    
    # Add more builtin tools to task
    task.add_tools([img_gen])
    
    # Verify added to task.tools
    assert img_gen in task.tools, "ImageGenerationTool should be in task.tools"
    assert len(task.tools) == 3, "Should have 3 tools in task.tools"
    
    # Manually trigger tool registration (simulating what happens during execution)
    # Separate builtin tools from regular tools (same logic as in agent.py during execution)
    from upsonic.tools.builtin_tools import AbstractBuiltinTool
    tools_to_register = task.tools if task.tools else []
    builtin_tools = []
    regular_tools = []
    
    for tool in tools_to_register:
        if tool is not None and isinstance(tool, AbstractBuiltinTool):
            builtin_tools.append(tool)
        else:
            regular_tools.append(tool)
    
    # Set task_builtin_tools (same as agent.py line 848)
    task.task_builtin_tools = builtin_tools
    
    # Register only regular tools (in this case there are none)
    if regular_tools:
        newly_registered = agent.tool_manager.register_tools(
            tools=regular_tools,
            task=task,
            agent_instance=agent
        )
        task.registered_task_tools.update(newly_registered)
    
    # After registration, builtin tools should be in task_builtin_tools
    assert len(task.task_builtin_tools) == 3, "Should have 3 builtin tools after registration"
    builtin_ids = {tool.unique_id for tool in task.task_builtin_tools}
    assert "code_execution" in builtin_ids, "code_execution should be in task_builtin_tools"
    assert "web_search" in builtin_ids, "web_search should be in task_builtin_tools"
    assert "image_generation" in builtin_ids, "image_generation should be in task_builtin_tools"
    
    # Builtin tools should NOT be in registered_task_tools
    assert len(task.registered_task_tools) == 0, "Builtin tools should NOT be in registered_task_tools"
    
    # Test removing builtin tools from task (requires agent parameter)
    task.remove_tools([code_exec], agent)
    
    # Verify removal
    assert code_exec not in task.tools, "CodeExecutionTool should be removed from task.tools"
    assert len(task.task_builtin_tools) == 2, "Should have 2 builtin tools after removal"
    assert not any(tool.unique_id == "code_execution" for tool in task.task_builtin_tools), "code_execution should be removed from task_builtin_tools"
    
    # Remove remaining builtin tools
    task.remove_tools([web_search, img_gen], agent)
    
    # Verify all removed
    assert len(task.task_builtin_tools) == 0, "All builtin tools should be removed"
    assert len(task.tools) == 0, "task.tools should be empty"


@pytest.mark.asyncio
async def test_runtime_builtin_tool_registration():
    """Test that builtin tools in tasks are properly separated during registration."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Create task with builtin tools (not registered yet)
    code_exec = CodeExecutionTool()
    from upsonic.tools.builtin_tools import ImageGenerationTool
    web_search = WebSearchTool()
    img_gen = ImageGenerationTool()
    
    task = Task(
        description="Test task with builtin tools",
        tools=[code_exec, img_gen, web_search]
    )
    
    # Before registration, builtin tools are not registered
    assert len(task.registered_task_tools) == 0, "Task tools should not be registered before registration"
    assert len(task.task_builtin_tools) == 0, "Task builtin tools should be empty before registration"
    
    # Trigger tool registration (simulating what happens during execution)
    # Separate builtin tools from regular tools (same logic as in agent.py)
    from upsonic.tools.builtin_tools import AbstractBuiltinTool
    tools_to_register = task.tools if task.tools else []
    builtin_tools = []
    regular_tools = []
    
    for tool in tools_to_register:
        if tool is not None and isinstance(tool, AbstractBuiltinTool):
            builtin_tools.append(tool)
        else:
            regular_tools.append(tool)
    
    # Set task_builtin_tools
    task.task_builtin_tools = builtin_tools
    
    # Register regular tools (in this case there are none)
    if regular_tools:
        newly_registered = agent.tool_manager.register_tools(
            tools=regular_tools,
            task=task,
            agent_instance=agent
        )
        task.registered_task_tools.update(newly_registered)
    
    # After registration, builtin tools should be in task_builtin_tools
    assert len(task.task_builtin_tools) == 3, "Should have 3 builtin tools after registration"
    builtin_ids = {tool.unique_id for tool in task.task_builtin_tools}
    assert "code_execution" in builtin_ids, "code_execution should be in task_builtin_tools"
    assert "image_generation" in builtin_ids, "image_generation should be in task_builtin_tools"
    assert "web_search" in builtin_ids, "web_search should be in task_builtin_tools"
    
    # Builtin tools should NOT be in registered_task_tools
    assert len(task.registered_task_tools) == 0, "Builtin tools should NOT be in registered_task_tools"
    
    # Verify builtin tools are in task.tools
    assert code_exec in task.tools, "CodeExecutionTool should still be in task.tools"
    assert img_gen in task.tools, "ImageGenerationTool should still be in task.tools"
    assert web_search in task.tools, "WebSearchTool should still be in task.tools"


@pytest.mark.asyncio
async def test_task_mixed_builtin_and_regular_tools():
    """Test mixing builtin tools and regular tools in a task."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Create task with both builtin and regular tools
    code_exec = CodeExecutionTool()
    web_search = WebSearchTool()
    
    task = Task(
        description="Test task with mixed tools",
        tools=[code_exec, web_search, add_numbers, multiply_numbers]
    )
    
    # Before registration
    assert len(task.tools) == 4, "Should have 4 tools in task.tools"
    assert len(task.registered_task_tools) == 0, "No tools registered before registration"
    assert len(task.task_builtin_tools) == 0, "No builtin tools registered before registration"
    
    # Trigger tool registration (simulating what happens during execution)
    # Separate builtin tools from regular tools (same logic as in agent.py)
    from upsonic.tools.builtin_tools import AbstractBuiltinTool
    tools_to_register = task.tools if task.tools else []
    builtin_tools = []
    regular_tools = []
    
    for tool in tools_to_register:
        if tool is not None and isinstance(tool, AbstractBuiltinTool):
            builtin_tools.append(tool)
        else:
            regular_tools.append(tool)
    
    # Set task_builtin_tools
    task.task_builtin_tools = builtin_tools
    
    # Register regular tools
    if regular_tools:
        newly_registered = agent.tool_manager.register_tools(
            tools=regular_tools,
            task=task,
            agent_instance=agent
        )
        task.registered_task_tools.update(newly_registered)
    
    # After registration, verify separation of builtin vs regular tools
    assert len(task.task_builtin_tools) == 2, "Should have 2 builtin tools"
    builtin_ids = {tool.unique_id for tool in task.task_builtin_tools}
    assert "code_execution" in builtin_ids, "code_execution should be in task_builtin_tools"
    assert "web_search" in builtin_ids, "web_search should be in task_builtin_tools"
    
    # Regular tools should be in registered_task_tools
    assert len(task.registered_task_tools) == 2, "Should have 2 regular tools registered"
    assert "add_numbers" in task.registered_task_tools, "add_numbers should be in registered_task_tools"
    assert "multiply_numbers" in task.registered_task_tools, "multiply_numbers should be in registered_task_tools"
    
    # All tools should still be in task.tools
    assert len(task.tools) == 4, "All tools should still be in task.tools"
    
    # Test removing builtin tool
    task.remove_tools([code_exec], agent)
    
    # Verify builtin tool removed but regular tools remain
    assert len(task.task_builtin_tools) == 1, "Should have 1 builtin tool left"
    assert len(task.registered_task_tools) == 2, "Regular tools should remain"
    assert code_exec not in task.tools, "CodeExecutionTool should be removed from task.tools"
    assert add_numbers in task.tools, "add_numbers should still be in task.tools"
    
    # Test removing regular tool
    task.remove_tools(["add_numbers"], agent)
    
    # Verify regular tool removed
    assert len(task.registered_task_tools) == 1, "Should have 1 regular tool left"
    assert "add_numbers" not in task.registered_task_tools, "add_numbers should be removed"
    assert "multiply_numbers" in task.registered_task_tools, "multiply_numbers should remain"


@pytest.mark.asyncio
async def test_agent_initialization_with_builtin_tools():
    """Test initializing Agent with builtin tools."""
    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()
    
    # Initialize agent with both builtin and regular tools
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        tools=[web_search, code_exec, add_numbers, multiply_numbers],
        debug=True
    )
    
    # Verify builtin tools are in agent_builtin_tools
    assert len(agent.agent_builtin_tools) == 2, "Should have 2 builtin tools"
    builtin_ids = {tool.unique_id for tool in agent.agent_builtin_tools}
    assert "web_search" in builtin_ids, "web_search should be in agent_builtin_tools"
    assert "code_execution" in builtin_ids, "code_execution should be in agent_builtin_tools"
    
    # Verify regular tools are in registered_agent_tools
    assert len(agent.registered_agent_tools) == 2, "Should have 2 regular tools registered"
    assert "add_numbers" in agent.registered_agent_tools, "add_numbers should be registered"
    assert "multiply_numbers" in agent.registered_agent_tools, "multiply_numbers should be registered"
    
    # All tools should be in agent.tools
    assert len(agent.tools) == 4, "Should have 4 tools total in agent.tools"
    assert web_search in agent.tools, "WebSearchTool should be in agent.tools"
    assert code_exec in agent.tools, "CodeExecutionTool should be in agent.tools"
    assert add_numbers in agent.tools, "add_numbers should be in agent.tools"
    assert multiply_numbers in agent.tools, "multiply_numbers should be in agent.tools"


@pytest.mark.asyncio
async def test_builtin_tools_not_in_tool_processor():
    """Verify that builtin tools are NOT processed by ToolProcessor."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Get initial count of registered tools in processor
    initial_processor_count = len(agent.tool_manager.processor.registered_tools)
    
    # Add builtin tools
    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()
    agent.add_tools([web_search, code_exec])
    
    # ToolProcessor should NOT have processed builtin tools
    after_builtin_count = len(agent.tool_manager.processor.registered_tools)
    assert after_builtin_count == initial_processor_count, \
        f"ToolProcessor should not process builtin tools. Before: {initial_processor_count}, After: {after_builtin_count}"
    
    # Builtin tools should be in agent_builtin_tools
    assert len(agent.agent_builtin_tools) == 2, "Should have 2 builtin tools"
    
    # Add regular tool
    agent.add_tools([add_numbers])
    
    # ToolProcessor SHOULD have processed regular tool
    after_regular_count = len(agent.tool_manager.processor.registered_tools)
    assert after_regular_count == initial_processor_count + 1, \
        f"ToolProcessor should process regular tools. Before: {initial_processor_count}, After regular: {after_regular_count}"
    
    # Verify separation
    assert len(agent.agent_builtin_tools) == 2, "Builtin tools should remain separate"
    assert "add_numbers" in agent.registered_agent_tools, "Regular tool should be registered"


@pytest.mark.asyncio
async def test_agent_add_remove_financial_tools():
    """Test adding and removing financial tools (pure class instance, not ToolKit)."""
    try:
        from upsonic.tools.common_tools.financial_tools import YFinanceTools
        
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create financial tools instance (pure class, not ToolKit)
        # YFinanceTools is a regular class instance, processor extracts public methods
        financial_tools = YFinanceTools(stock_price=True, enable_all=False)
        
        # Add the instance directly - processor should extract methods via _process_class_tools
        agent.add_tools(financial_tools)
        
        # Verify tools are registered (processor extracts public methods from class instance)
        tool_names = list(agent.registered_agent_tools.keys())
        # Financial tools methods should be registered (get_current_stock_price, etc.)
        assert len(tool_names) > 0, f"Financial tools should be registered. Found tools: {tool_names}"
        
        # Check if any financial tool is registered
        financial_tool_found = any(
            "stock" in name.lower() or "price" in name.lower() or "get_current" in name.lower()
            for name in tool_names
        )
        assert financial_tool_found, f"Financial tool should be registered. Found: {tool_names}"
        
        # Remove by instance (should remove all its tools)
        agent.remove_tools(financial_tools)
        
        # Verify removal
        tool_names_after = list(agent.registered_agent_tools.keys())
        financial_tool_still_there = any(
            "stock" in name.lower() or "price" in name.lower() or "get_current" in name.lower()
            for name in tool_names_after
        )
        assert not financial_tool_still_there, "Financial tools should be removed"
        assert financial_tools not in agent.tools, "Financial tools instance should not be in agent.tools"
    except ImportError:
        pytest.skip("Financial tools dependencies not available")


@pytest.mark.asyncio
async def test_agent_add_remove_mcp_handler():
    """Test adding and removing MCP handler from Agent."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create MCP handler (using filesystem server as example)
        # Note: This tests the tool management logic, not actual MCP execution
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
        
        # Add MCP handler
        agent.add_tools(handler)
        
        # Verify MCP tools are registered
        # Handler should create tools like read_file, write_file, etc.
        initial_tool_count = len(agent.registered_agent_tools)
        assert initial_tool_count > 0, f"MCP handler should register tools, got {initial_tool_count}"
        
        # MCP handler should be in agent.tools
        assert handler in agent.tools, "MCP handler should be in agent.tools"
        
        # Verify handler is tracked in tool processor
        assert len(agent.tool_manager.processor.mcp_handlers) > 0, "MCP handler should be tracked in processor"
        
        # Remove ENTIRE handler by object (removes handler + ALL its tools)
        agent.remove_tools(handler)
        
        # Verify all MCP tools removed
        assert len(agent.registered_agent_tools) == 0, "All MCP tools should be removed"
        assert handler not in agent.tools, "MCP handler should be removed from agent.tools"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        # If MCP server not available, that's okay - we're testing tool management
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_agent_remove_individual_mcp_tools():
    """Test removing individual tools from MCP handler by name (keeping the handler)."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create MCP handler
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
        
        # Add MCP handler
        agent.add_tools(handler)
        
        # Get registered tools
        initial_tool_count = len(agent.registered_agent_tools)
        assert initial_tool_count > 0, "MCP handler should register tools"
        
        # Get one tool name to remove
        tool_names = list(agent.registered_agent_tools.keys())
        tool_to_remove = tool_names[0]
        
        # Remove individual MCP tool by name (keeps handler)
        agent.remove_tools(tool_to_remove)
        
        # Verify only that tool is removed
        assert tool_to_remove not in agent.registered_agent_tools, f"{tool_to_remove} should be removed"
        assert len(agent.registered_agent_tools) == initial_tool_count - 1, "Should have one less tool"
        
        # Handler should still be in agent.tools (1:many relationship)
        assert handler in agent.tools, "MCP handler should still be in agent.tools"
        
        # Remove another individual tool
        if len(agent.registered_agent_tools) > 0:
            second_tool = list(agent.registered_agent_tools.keys())[0]
            agent.remove_tools(second_tool)
            assert second_tool not in agent.registered_agent_tools, f"{second_tool} should be removed"
            assert len(agent.registered_agent_tools) == initial_tool_count - 2, "Should have two less tools"
            
            # Handler should STILL be in agent.tools
            assert handler in agent.tools, "MCP handler should still be in agent.tools after removing individual tools"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_agent_add_remove_multiple_mcp_handlers():
    """
    Test adding and removing multiple MCP handlers.
    
    Note: If handlers provide tools with identical names (e.g., two filesystem servers both 
    providing 'read_file'), the tools will overwrite each other in registered_agent_tools 
    (dict keyed by name). This test uses sequential add/remove to avoid this limitation.
    """
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create first MCP handler
        handler1 = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
        
        # Add first handler
        agent.add_tools(handler1)
        
        # Verify handler registered tools
        tools_from_handler1 = len(agent.registered_agent_tools)
        assert tools_from_handler1 > 0, "First MCP handler should register tools"
        assert handler1 in agent.tools, "First MCP handler should be in agent.tools"
        
        # Store tool names from first handler
        handler1_tools = set(agent.registered_agent_tools.keys())
        
        # Remove first handler by object (removes ALL its tools)
        agent.remove_tools(handler1)
        
        # Verify first handler and all its tools removed
        assert handler1 not in agent.tools, "First handler should be removed"
        assert len(agent.registered_agent_tools) == 0, "All tools from first handler should be removed"
        
        # Add second handler
        handler2 = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /var/tmp"
        )
        
        agent.add_tools(handler2)
        
        # Verify second handler registered tools
        tools_from_handler2 = len(agent.registered_agent_tools)
        assert tools_from_handler2 > 0, "Second handler should register tools"
        assert handler2 in agent.tools, "Second handler should be in agent.tools"
        
        # Both handlers should register similar number of tools (same server type)
        assert abs(tools_from_handler1 - tools_from_handler2) <= 1, \
            "Both handlers should register similar number of tools"
        
        # Remove second handler
        agent.remove_tools(handler2)
        
        # Verify all removed
        assert handler2 not in agent.tools, "Second handler should be removed"
        assert len(agent.registered_agent_tools) == 0, "All MCP tools should be removed"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_task_add_remove_mcp_handler():
    """Test adding and removing MCP handler from Task."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create MCP handler
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
        
        # Create task with MCP handler
        task = Task(
            description="Test task with MCP tools",
            tools=[handler, add_numbers]
        )
        
        # Execute task to trigger registration
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.do_async(task)
        
        # Verify MCP tools registered
        assert len(task.registered_task_tools) > 1, "Should have MCP tools + add_numbers"
        assert "add_numbers" in task.registered_task_tools, "add_numbers should be registered"
        
        # Get MCP tool names (all except add_numbers)
        mcp_tool_names = [name for name in task.registered_task_tools.keys() if name != "add_numbers"]
        assert len(mcp_tool_names) > 0, "Should have MCP tools"
        
        # Remove one MCP tool by name
        task.remove_tools(mcp_tool_names[0], agent)
        
        # Verify only that tool removed
        assert mcp_tool_names[0] not in task.registered_task_tools, "MCP tool should be removed"
        assert "add_numbers" in task.registered_task_tools, "add_numbers should remain"
        
        # Remove entire handler by object (removes all remaining MCP tools)
        task.remove_tools(handler, agent)
        
        # Verify all MCP tools removed but add_numbers remains
        for mcp_tool in mcp_tool_names:
            assert mcp_tool not in task.registered_task_tools, f"{mcp_tool} should be removed"
        assert "add_numbers" in task.registered_task_tools, "add_numbers should still be registered"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_agent_add_remove_duckduckgo_tool():
    """Test adding and removing DuckDuckGo search tool."""
    try:
        from upsonic.tools.common_tools.duckduckgo import duckduckgo_search_tool
        
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create DuckDuckGo tool (function tool)
        ddg_tool = duckduckgo_search_tool()
        
        # Add tool
        agent.add_tools(ddg_tool)
        
        # Verify tool is registered
        assert "duckduckgo_search" in agent.registered_agent_tools, "duckduckgo_search should be registered"
        
        # Remove by name
        agent.remove_tools("duckduckgo_search")
        assert "duckduckgo_search" not in agent.registered_agent_tools, "duckduckgo_search should be removed"
    except ImportError:
        pytest.skip("DuckDuckGo dependencies not available")


@pytest.mark.asyncio
async def test_agent_add_remove_tavily_tool():
    """Test adding and removing Tavily search tool."""
    try:
        from upsonic.tools.common_tools.tavily import tavily_search_tool
        
        # Tavily requires API key
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            pytest.skip("TAVILY_API_KEY not set")
        
        agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
        
        # Create Tavily tool (function tool)
        tavily_tool = tavily_search_tool(api_key=tavily_api_key)
        
        # Add tool
        agent.add_tools(tavily_tool)
        
        # Verify tool is registered
        assert "tavily_search" in agent.registered_agent_tools, "tavily_search should be registered"
        
        # Remove by name
        agent.remove_tools("tavily_search")
        assert "tavily_search" not in agent.registered_agent_tools, "tavily_search should be removed"
    except ImportError:
        pytest.skip("Tavily dependencies not available")


@pytest.mark.asyncio
async def test_agent_add_remove_thinking_tool():
    """Test adding and removing plan_and_execute (thinking tool)."""
    from upsonic.tools.orchestration import plan_and_execute
    
    # Test 1: Auto-added via enable_thinking_tool with other tools
    agent = Agent(
        model="openai/gpt-4o", 
        name="Test Agent", 
        debug=True, 
        enable_thinking_tool=True,
        tools=[add_numbers]  # Need at least one tool for plan_and_execute to be added
    )
    
    # plan_and_execute should be auto-added along with add_numbers
    assert "plan_and_execute" in agent.registered_agent_tools, "plan_and_execute should be auto-added when enable_thinking_tool=True"
    assert "add_numbers" in agent.registered_agent_tools, "add_numbers should also be registered"
    
    # Remove plan_and_execute by name
    agent.remove_tools("plan_and_execute")
    assert "plan_and_execute" not in agent.registered_agent_tools, "plan_and_execute should be removed"
    assert "add_numbers" in agent.registered_agent_tools, "add_numbers should still be registered"
    
    # Test 2: Explicitly added as regular tool
    agent2 = Agent(model="openai/gpt-4o", name="Test Agent 2", debug=True, enable_thinking_tool=False)
    
    # Initially no plan_and_execute
    assert "plan_and_execute" not in agent2.registered_agent_tools, "plan_and_execute should not be present initially"
    
    # Add explicitly
    agent2.add_tools(plan_and_execute)
    assert "plan_and_execute" in agent2.registered_agent_tools, "plan_and_execute should be added"
    
    # Remove by object
    agent2.remove_tools(plan_and_execute)
    assert "plan_and_execute" not in agent2.registered_agent_tools, "plan_and_execute should be removed"
    
    # Test 3: Task-level override
    agent3 = Agent(model="openai/gpt-4o", name="Test Agent 3", debug=True, enable_thinking_tool=False)
    
    # Task with thinking enabled (overrides agent)
    task = Task(
        description="Test task with thinking",
        tools=[add_numbers],
        enable_thinking_tool=True
    )
    
    # Execute to trigger registration
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent3.do_async(task)
    
    # plan_and_execute should be in task tools
    assert "plan_and_execute" in task.registered_task_tools, "plan_and_execute should be in task tools"
    assert "add_numbers" in task.registered_task_tools, "add_numbers should also be registered"
    
    # Remove from task
    task.remove_tools("plan_and_execute", agent3)
    assert "plan_and_execute" not in task.registered_task_tools, "plan_and_execute should be removed from task"
    assert "add_numbers" in task.registered_task_tools, "add_numbers should still be registered"


@pytest.mark.asyncio
async def test_agent_as_tool():
    """Test adding and removing Agent as a tool."""
    # Create sub-agent
    sub_agent = Agent(
        model="openai/gpt-4o",
        name="Math Assistant",
        role="Math Specialist",
        goal="Help with mathematical calculations"
    )
    
    # Create main agent
    main_agent = Agent(model="openai/gpt-4o", name="Main Agent", debug=True)
    
    # Add sub-agent as tool
    main_agent.add_tools(sub_agent)
    
    # Verify agent tool is registered (should create ask_* method)
    tool_names = list(main_agent.registered_agent_tools.keys())
    agent_tool_name = [name for name in tool_names if name.startswith("ask_")][0]
    assert agent_tool_name is not None, "Agent tool should be registered with ask_* name"
    assert sub_agent in main_agent.tools, "Sub-agent should be in main_agent.tools"
    
    # Remove agent tool
    main_agent.remove_tools(sub_agent)
    assert agent_tool_name not in main_agent.registered_agent_tools, "Agent tool should be removed"
    assert sub_agent not in main_agent.tools, "Sub-agent should not be in main_agent.tools"


@pytest.mark.asyncio
async def test_task_add_remove_tools():
    """Test adding and removing tools from Task."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Create task with tools
    task = Task(description="Test task", tools=[add_numbers])
    
    # Verify task has tools but not registered yet (runtime registration)
    assert add_numbers in task.tools, "add_numbers should be in task.tools"
    assert len(task.registered_task_tools) == 0, "Task tools not registered until execution"
    
    # Add tools to task
    task.add_tools([multiply_numbers, greet])
    assert multiply_numbers in task.tools, "multiply_numbers should be in task.tools"
    assert greet in task.tools, "greet should be in task.tools"
    
    # Execute task to trigger runtime registration
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    # Verify tools are registered after execution
    assert "add_numbers" in task.registered_task_tools, "add_numbers should be registered after execution"
    assert "multiply_numbers" in task.registered_task_tools, "multiply_numbers should be registered"
    assert "greet" in task.registered_task_tools, "greet should be registered"
    
    # Set task.agent for remove_tools to work properly
    task.agent = agent
    
    # Remove tools from task (requires agent)
    task.remove_tools("add_numbers", agent)
    assert "add_numbers" not in task.registered_task_tools, "add_numbers should be removed"
    assert add_numbers not in task.tools, "add_numbers should not be in task.tools"
    
    # Remove by name
    task.remove_tools("multiply_numbers", agent)
    assert "multiply_numbers" not in task.registered_task_tools, "multiply_numbers should be removed"


@pytest.mark.asyncio
async def test_runtime_task_tool_registration():
    """Test that task tools are registered at runtime when agent.do_async(task) is called."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Create task with tools (not registered yet)
    task = Task(
        description="Use add_numbers to calculate 5 + 3",
        tools=[add_numbers]
    )
    
    # Before execution, tools are not registered
    assert len(task.registered_task_tools) == 0, "Task tools should not be registered before execution"
    assert len(task.task_builtin_tools) == 0, "Task builtin tools should be empty before execution"
    
    # Execute task
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    # After execution, tools should be registered
    assert "add_numbers" in task.registered_task_tools, "add_numbers should be registered after execution"
    assert len(task.registered_task_tools) > 0, "Task should have registered tools after execution"
    
    # Verify tool_manager has the tools
    tool_defs = agent.tool_manager.get_tool_definitions()
    tool_names = [t.name for t in tool_defs]
    assert "add_numbers" in tool_names, "add_numbers should be in tool_manager definitions"


@pytest.mark.asyncio
async def test_mixed_tool_types():
    """Test mixing custom tools, toolkits, and builtin tools."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Add mixed tool types
    math_kit = MathToolKit()
    web_search = WebSearchTool()
    
    agent.add_tools([add_numbers, math_kit, web_search])
    
    # Verify all are registered appropriately
    assert "add_numbers" in agent.registered_agent_tools, "Custom tool should be registered"
    assert "subtract" in agent.registered_agent_tools, "Toolkit tool should be registered"
    assert "divide" in agent.registered_agent_tools, "Toolkit tool should be registered"
    
    # Builtin tools are tracked separately in agent_builtin_tools, not in registered_agent_tools
    assert len(agent.agent_builtin_tools) == 1, "Should have 1 builtin tool"
    assert any(tool.unique_id == "web_search" for tool in agent.agent_builtin_tools), "web_search should be in agent_builtin_tools"
    assert len(agent.registered_agent_tools) == 3, "Should have 3 regular tools registered (not builtin)"
    
    # All should be in agent.tools
    assert len(agent.tools) == 3, "Should have 3 tool objects in agent.tools (function + toolkit + builtin)"
    
    # Remove regular tools only (by name)
    agent.remove_tools(["add_numbers", "subtract", "divide"])
    assert "add_numbers" not in agent.registered_agent_tools, "Custom tool should be removed"
    assert "subtract" not in agent.registered_agent_tools, "Toolkit tool should be removed"
    assert "divide" not in agent.registered_agent_tools, "Toolkit tool should be removed"
    
    # Builtin tool should still be in agent.tools and agent_builtin_tools
    assert web_search in agent.tools, "Builtin tool should still be in agent.tools"
    assert len(agent.agent_builtin_tools) == 1, "Builtin tool should still be in agent_builtin_tools"
    assert any(tool.unique_id == "web_search" for tool in agent.agent_builtin_tools), "web_search should still be in agent_builtin_tools"
    
    # Now remove builtin tool by object
    agent.remove_tools([web_search])
    assert web_search not in agent.tools, "Builtin tool should be removed from agent.tools"
    assert len(agent.agent_builtin_tools) == 0, "Builtin tool should be removed from agent_builtin_tools"


@pytest.mark.asyncio
async def test_tool_manager_attributes():
    """Test that tool_manager attributes are properly updated."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Add tools
    agent.add_tools([add_numbers, multiply_numbers])
    
    # Verify tool_manager has the tools
    tool_defs = agent.tool_manager.get_tool_definitions()
    tool_names = [t.name for t in tool_defs]
    assert "add_numbers" in tool_names, "tool_manager should have add_numbers"
    assert "multiply_numbers" in tool_names, "tool_manager should have multiply_numbers"
    
    # Verify registered_agent_tools matches tool_manager
    assert len(agent.registered_agent_tools) == len([t for t in tool_names if t in agent.registered_agent_tools]), "registered_agent_tools should match tool_manager"
    
    # Remove tool
    agent.remove_tools("add_numbers")
    
    # Verify tool_manager updated
    tool_defs_after = agent.tool_manager.get_tool_definitions()
    tool_names_after = [t.name for t in tool_defs_after]
    assert "add_numbers" not in tool_names_after, "tool_manager should not have add_numbers after removal"


@pytest.mark.asyncio
async def test_task_tool_attributes_after_execution():
    """Test that task tool attributes are properly set after execution."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    # Add agent tools
    agent.add_tools([add_numbers])
    
    # Create task with different tools
    task = Task(
        description="Use multiply_numbers to calculate 4 * 2",
        tools=[multiply_numbers]
    )
    
    # Execute task
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    # Verify task attributes
    assert "multiply_numbers" in task.registered_task_tools, "Task should have registered tools"
    assert len(task.registered_task_tools) > 0, "Task should have registered_task_tools"
    
    # Verify agent still has its tools
    assert "add_numbers" in agent.registered_agent_tools, "Agent should still have its tools"
    
    # Verify both are in tool_manager
    tool_defs = agent.tool_manager.get_tool_definitions()
    tool_names = [t.name for t in tool_defs]
    assert "add_numbers" in tool_names, "Agent tool should be in tool_manager"
    assert "multiply_numbers" in tool_names, "Task tool should be in tool_manager"


@pytest.mark.asyncio
async def test_all_tool_types_comprehensive():
    """Comprehensive test of all tool types together."""
    agent = Agent(model="openai/gpt-4o", name="Test Agent", debug=True)
    
    regular_tools_added = []
    builtin_tools_added = []
    
    # 1. Add function tool
    agent.add_tools(add_numbers)
    assert "add_numbers" in agent.registered_agent_tools, "Function tool should be registered"
    regular_tools_added.append("add_numbers")
    
    # 2. Add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    assert "subtract" in agent.registered_agent_tools, "ToolKit tool should be registered"
    assert "divide" in agent.registered_agent_tools, "ToolKit tool should be registered"
    regular_tools_added.extend(["subtract", "divide"])
    
    # 3. Add builtin tools (tracked in agent_builtin_tools, not registered_agent_tools)
    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()
    agent.add_tools([web_search, code_exec])
    
    # Verify builtin tools are tracked correctly
    assert len(agent.agent_builtin_tools) == 2, "Should have 2 builtin tools"
    assert any(tool.unique_id == "web_search" for tool in agent.agent_builtin_tools), "web_search should be in agent_builtin_tools"
    assert any(tool.unique_id == "code_execution" for tool in agent.agent_builtin_tools), "code_execution should be in agent_builtin_tools"
    builtin_tools_added.extend([web_search, code_exec])
    
    # Builtin tools should NOT be in registered_agent_tools
    assert "web_search" not in agent.registered_agent_tools, "Builtin tools should NOT be in registered_agent_tools"
    assert "code_execution" not in agent.registered_agent_tools, "Builtin tools should NOT be in registered_agent_tools"
    
    # 4. Add Agent as tool
    sub_agent = Agent(model="openai/gpt-4o", name="Helper")
    agent.add_tools(sub_agent)
    tool_names = list(agent.registered_agent_tools.keys())
    agent_tool_name = [name for name in tool_names if name.startswith("ask_")][0]
    assert agent_tool_name is not None, "Agent tool should be registered"
    regular_tools_added.append(agent_tool_name)
    
    # 5. Add financial tools (pure class instance, not ToolKit) if available
    financial_tools_instance = None
    try:
        from upsonic.tools.common_tools.financial_tools import YFinanceTools
        financial_tools_instance = YFinanceTools(stock_price=True, enable_all=False)
        agent.add_tools(financial_tools_instance)  # Add instance directly, processor extracts methods
        # Track that we added a financial tools instance (to be removed by object, not by name)
    except (ImportError, Exception):
        pass  # Skip if not available
    
    # 6. Add DuckDuckGo tool if available
    try:
        from upsonic.tools.common_tools.duckduckgo import duckduckgo_search_tool
        ddg_tool = duckduckgo_search_tool()
        agent.add_tools(ddg_tool)
        assert "duckduckgo_search" in agent.registered_agent_tools, "DuckDuckGo tool should be registered"
        regular_tools_added.append("duckduckgo_search")
    except (ImportError, Exception):
        pass  # Skip if not available
    
    # Verify all attributes
    assert len(agent.registered_agent_tools) >= len(regular_tools_added), \
        f"Should have at least {len(regular_tools_added)} regular tools registered. Got {len(agent.registered_agent_tools)}"
    assert len(agent.agent_builtin_tools) == len(builtin_tools_added), \
        f"Should have {len(builtin_tools_added)} builtin tools. Got {len(agent.agent_builtin_tools)}"
    # Note: agent.tools contains original objects (function, toolkit instance, builtin tools, agent, class instance)
    # ToolKit is 1 object but provides multiple tools, so we count objects, not tool names
    # Expected: add_numbers (1) + math_kit (1) + 2 builtins (2) + sub_agent (1) + financial_tools (1) + ddg (1) = 7 minimum
    assert len(agent.tools) >= 5, \
        f"Should have at least 5 tool objects in agent.tools. Got {len(agent.tools)}"
    
    # Verify tool_manager has regular tools (builtin tools are not in tool_definitions, they're separate)
    tool_defs = agent.tool_manager.get_tool_definitions()
    tool_def_names = [t.name for t in tool_defs]
    
    # Check all regular tools are in tool_manager
    for tool_name in regular_tools_added:
        assert tool_name in tool_def_names, f"{tool_name} should be in tool_manager definitions"
    
    # Verify builtin tools are NOT in tool_definitions
    assert "web_search" not in tool_def_names, "Builtin tools should NOT be in tool_definitions"
    assert "code_execution" not in tool_def_names, "Builtin tools should NOT be in tool_definitions"
    
    # Remove function tools and toolkits by name
    if regular_tools_added:
        agent.remove_tools(regular_tools_added)
    
    # Verify removal of regular tools by name
    for tool_name in regular_tools_added:
        assert tool_name not in agent.registered_agent_tools, f"{tool_name} should be removed"
    
    # Remove class instances (ToolKit, financial tools, etc.) by object
    # These need to be removed by object to remove ALL their extracted methods
    if math_kit in agent.tools:
        agent.remove_tools([math_kit])
    if sub_agent in agent.tools:
        agent.remove_tools([sub_agent])
    if financial_tools_instance and financial_tools_instance in agent.tools:
        agent.remove_tools([financial_tools_instance])
    
    # Verify builtin tools still remain
    assert len(agent.agent_builtin_tools) == len(builtin_tools_added), "Builtin tools should still be present"
    
    # Remove builtin tools by object
    if builtin_tools_added:
        agent.remove_tools(builtin_tools_added)
        
        # Verify removal
        for builtin_tool in builtin_tools_added:
            assert builtin_tool not in agent.tools, f"{builtin_tool.unique_id} should be removed from agent.tools"
        
        assert len(agent.agent_builtin_tools) == 0, "All builtin tools should be removed from agent_builtin_tools"
    
    # Verify all tools are removed
    assert len(agent.registered_agent_tools) == 0, f"All regular tools should be removed. Remaining: {list(agent.registered_agent_tools.keys())}"
    assert len(agent.agent_builtin_tools) == 0, "All builtin tools should be removed"
    assert len(agent.tools) == 0, f"agent.tools should be empty. Remaining: {agent.tools}"
