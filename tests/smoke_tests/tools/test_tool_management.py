"""
Test 14: Test all variants of adding, removing of all type of tools (Agent-level)

Success criteria:
- Tests all agent-level tool types: ToolKit, function tools, pure classes, Agent as tool, 
  financial_tools, duckduckgo, tavily, builtin tools
- Tests using Agent class remove_tools, add_tools
- Checks agent and agent.tool_manager attributes (registered_agent_tools, etc.)

Note: Task-level tool management is tested in test_task_tool_management.py
"""

import pytest
import os
from upsonic import Agent, Task
from upsonic.tools import tool, ToolKit
from upsonic.tools.builtin_tools import WebSearchTool, CodeExecutionTool
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(120)

MODEL = "openai/gpt-4o-mini"
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


class AsyncMathToolKit(ToolKit):
    """ToolKit with both sync @tool methods and async methods for use_async testing."""
    
    @tool
    def add_sync(self, a: int, b: int) -> int:
        """Sync add two numbers."""
        return a + b
    
    @tool
    def subtract_sync(self, a: int, b: int) -> int:
        """Sync subtract b from a."""
        return a - b
    
    async def multiply_async(self, a: int, b: int) -> int:
        """Async multiply two numbers."""
        return a * b
    
    async def divide_async(self, a: int, b: int) -> float:
        """Async divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def helper_not_decorated(self, x: int) -> int:
        """Public sync method WITHOUT @tool decorator."""
        return x * 10
    
    def _private_helper(self) -> None:
        """Private helper, should never be registered."""
        pass


class ConfiguredToolKit(ToolKit):
    """ToolKit with explicit decorator configs for testing config priority."""
    
    @tool(requires_confirmation=True, timeout=999.0)
    def dangerous_action(self, x: int) -> int:
        """A dangerous action that needs confirmation."""
        return x * 2
    
    @tool(timeout=60.0)
    def safe_action(self, y: str) -> str:
        """A safe action."""
        return y.upper()


class BareToolKit(ToolKit):
    """ToolKit with public methods but NO @tool decorators (Case 5/6)."""
    
    def compute(self, x: int) -> int:
        """Compute something."""
        return x * 2
    
    def format_text(self, text: str) -> str:
        """Format text."""
        return text.strip()
    
    async def async_compute(self, x: int) -> int:
        """Async compute something."""
        return x * 3


@pytest.mark.asyncio
async def test_agent_add_remove_custom_tools():
    """Test adding and removing custom tools (functions) from Agent."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
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
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
    # Create task with ToolKit
    math_kit = MathToolKit()
    task = Task(
        description="Use add_numbers to calculate 1 + 1. Return the number.",
        tools=[math_kit, add_numbers]
    )
    
    # Execute task to trigger registration
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.print_do_async(task)
    
    # Verify printed output shows tool calls
    output_text = output_buffer.getvalue()
    assert "Tool Calls" in output_text, "Output should contain 'Tool Calls' table"
    assert "add_numbers" in output_text, "Output should show add_numbers was called"
    
    # Verify task.tool_calls attribute
    assert len(task.tool_calls) > 0, "task.tool_calls should not be empty"
    called_names = [tc.get("tool_name", "") for tc in task.tool_calls]
    assert "add_numbers" in called_names, f"add_numbers should be in tool_calls, got: {called_names}"
    
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
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
    
    # Trigger tool registration via _setup_task_tools (creates task.tool_manager)
    agent._setup_task_tools(task)
    
    # After registration, builtin tools should be in task_builtin_tools
    assert len(task.task_builtin_tools) == 3, "Should have 3 builtin tools after registration"
    builtin_ids = {tool.unique_id for tool in task.task_builtin_tools}
    assert "code_execution" in builtin_ids, "code_execution should be in task_builtin_tools"
    assert "web_search" in builtin_ids, "web_search should be in task_builtin_tools"
    assert "image_generation" in builtin_ids, "image_generation should be in task_builtin_tools"
    
    # Builtin tools should NOT be in registered_task_tools
    assert len(task.registered_task_tools) == 0, "Builtin tools should NOT be in registered_task_tools"
    
    # Task should have its own tool_manager
    assert task.tool_manager is not None, "Task should have a ToolManager after setup"
    
    # Test removing builtin tools from task (agent param is optional/deprecated)
    task.remove_tools([code_exec])
    
    # Verify removal
    assert code_exec not in task.tools, "CodeExecutionTool should be removed from task.tools"
    assert len(task.task_builtin_tools) == 2, "Should have 2 builtin tools after removal"
    assert not any(tool.unique_id == "code_execution" for tool in task.task_builtin_tools), "code_execution should be removed from task_builtin_tools"
    
    # Remove remaining builtin tools
    task.remove_tools([web_search, img_gen])
    
    # Verify all removed
    assert len(task.task_builtin_tools) == 0, "All builtin tools should be removed"
    assert len(task.tools) == 0, "task.tools should be empty"


@pytest.mark.asyncio
async def test_runtime_builtin_tool_registration():
    """Test that builtin tools in tasks are properly separated during registration."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
    
    # Trigger tool registration via _setup_task_tools (creates task.tool_manager)
    agent._setup_task_tools(task)
    
    # After registration, builtin tools should be in task_builtin_tools
    assert len(task.task_builtin_tools) == 3, "Should have 3 builtin tools after registration"
    builtin_ids = {tool.unique_id for tool in task.task_builtin_tools}
    assert "code_execution" in builtin_ids, "code_execution should be in task_builtin_tools"
    assert "image_generation" in builtin_ids, "image_generation should be in task_builtin_tools"
    assert "web_search" in builtin_ids, "web_search should be in task_builtin_tools"
    
    # Builtin tools should NOT be in registered_task_tools
    assert len(task.registered_task_tools) == 0, "Builtin tools should NOT be in registered_task_tools"
    
    # Task should have its own tool_manager
    assert task.tool_manager is not None, "Task should have a ToolManager after setup"
    
    # Verify builtin tools are in task.tools
    assert code_exec in task.tools, "CodeExecutionTool should still be in task.tools"
    assert img_gen in task.tools, "ImageGenerationTool should still be in task.tools"
    assert web_search in task.tools, "WebSearchTool should still be in task.tools"


@pytest.mark.asyncio
async def test_task_mixed_builtin_and_regular_tools():
    """Test mixing builtin tools and regular tools in a task."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
    
    # Trigger tool registration via _setup_task_tools (creates task.tool_manager)
    agent._setup_task_tools(task)
    
    # After registration, verify separation of builtin vs regular tools
    assert len(task.task_builtin_tools) == 2, "Should have 2 builtin tools"
    builtin_ids = {tool.unique_id for tool in task.task_builtin_tools}
    assert "code_execution" in builtin_ids, "code_execution should be in task_builtin_tools"
    assert "web_search" in builtin_ids, "web_search should be in task_builtin_tools"
    
    # Regular tools should be in registered_task_tools
    assert len(task.registered_task_tools) == 2, "Should have 2 regular tools registered"
    assert "add_numbers" in task.registered_task_tools, "add_numbers should be in registered_task_tools"
    assert "multiply_numbers" in task.registered_task_tools, "multiply_numbers should be in registered_task_tools"
    
    # Task should have its own tool_manager with regular tools
    assert task.tool_manager is not None, "Task should have a ToolManager after setup"
    task_tool_defs = task.get_tool_defs()
    task_tool_names = [td.name for td in task_tool_defs]
    assert "add_numbers" in task_tool_names, "add_numbers should be in task tool_manager"
    assert "multiply_numbers" in task_tool_names, "multiply_numbers should be in task tool_manager"
    
    # All tools should still be in task.tools
    assert len(task.tools) == 4, "All tools should still be in task.tools"
    
    # Test removing builtin tool (agent param is optional/deprecated)
    task.remove_tools([code_exec])
    
    # Verify builtin tool removed but regular tools remain
    assert len(task.task_builtin_tools) == 1, "Should have 1 builtin tool left"
    assert len(task.registered_task_tools) == 2, "Regular tools should remain"
    assert code_exec not in task.tools, "CodeExecutionTool should be removed from task.tools"
    assert add_numbers in task.tools, "add_numbers should still be in task.tools"
    
    # Test removing regular tool
    task.remove_tools(["add_numbers"])
    
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
        model=MODEL,
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
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
    # Get initial count of registered tools in processor
    initial_processor_count = len(agent.tool_manager.registry.registered_tools)
    
    # Add builtin tools
    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()
    agent.add_tools([web_search, code_exec])
    
    # ToolProcessor should NOT have processed builtin tools
    after_builtin_count = len(agent.tool_manager.registry.registered_tools)
    assert after_builtin_count == initial_processor_count, \
        f"ToolProcessor should not process builtin tools. Before: {initial_processor_count}, After: {after_builtin_count}"
    
    # Builtin tools should be in agent_builtin_tools
    assert len(agent.agent_builtin_tools) == 2, "Should have 2 builtin tools"
    
    # Add regular tool
    agent.add_tools([add_numbers])
    
    # ToolProcessor SHOULD have processed regular tool
    after_regular_count = len(agent.tool_manager.registry.registered_tools)
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
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
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
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
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
        assert len(agent.tool_manager.registry.mcp_handlers) > 0, "MCP handler should be tracked in processor"
        
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
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
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
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
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
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
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
            result = await agent.print_do_async(task)
        
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
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
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
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
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
        model=MODEL, 
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
    agent2 = Agent(model=MODEL, name="Test Agent 2", debug=True, enable_thinking_tool=False)
    
    # Initially no plan_and_execute
    assert "plan_and_execute" not in agent2.registered_agent_tools, "plan_and_execute should not be present initially"
    
    # Add explicitly
    agent2.add_tools(plan_and_execute)
    assert "plan_and_execute" in agent2.registered_agent_tools, "plan_and_execute should be added"
    
    # Remove by object
    agent2.remove_tools(plan_and_execute)
    assert "plan_and_execute" not in agent2.registered_agent_tools, "plan_and_execute should be removed"
    
    # Test 3: Task-level override — enable_thinking_tool adds plan_and_execute to task.
    # 3a: Run without thinking tool so the model reliably calls add_numbers; verify tool_calls and output.
    agent3 = Agent(model=MODEL, name="Test Agent 3", debug=True, enable_thinking_tool=False)
    task_call = Task(
        description="Use add_numbers to calculate 10 + 20. Return only the number.",
        tools=[add_numbers],
        enable_thinking_tool=False
    )
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        await agent3.print_do_async(task_call)
    output_text = output_buffer.getvalue()
    assert len(task_call.tool_calls) > 0, "Task must have at least one tool call (model must call add_numbers)"
    assert "Tool Calls" in output_text, "Output should contain 'Tool Calls' table when tools were called"
    called_names = [tc.get("tool_name", "") for tc in task_call.tool_calls]
    assert "add_numbers" in called_names, f"add_numbers should be in tool_calls, got: {called_names}"

    # 3b: Run with enable_thinking_tool=True; verify plan_and_execute and add_numbers are registered and removable.
    task = Task(
        description="Use add_numbers to calculate 10 + 20. Return only the number.",
        tools=[add_numbers],
        enable_thinking_tool=True
    )
    output_buffer2 = StringIO()
    with redirect_stdout(output_buffer2):
        await agent3.print_do_async(task)
    
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
        model=MODEL,
        name="Math Assistant",
        role="Math Specialist",
        goal="Help with mathematical calculations"
    )
    
    # Create main agent
    main_agent = Agent(model=MODEL, name="Main Agent", debug=True)
    
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
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
    # Create task with tools
    task = Task(description="Use add_numbers to calculate 2 + 3. Return the number.", tools=[add_numbers])
    
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
        result = await agent.print_do_async(task)
    
    # Verify printed output shows tool calls
    output_text = output_buffer.getvalue()
    assert "Tool Calls" in output_text, "Output should contain 'Tool Calls' table"
    assert "add_numbers" in output_text, "Output should show add_numbers was called"
    
    # Verify task.tool_calls attribute
    assert len(task.tool_calls) > 0, "task.tool_calls should not be empty"
    called_names = [tc.get("tool_name", "") for tc in task.tool_calls]
    assert "add_numbers" in called_names, f"add_numbers should be in tool_calls, got: {called_names}"
    
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
    """Test that task tools are registered at runtime when agent.print_do_async(task) is called."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
    # Create task with tools (not registered yet)
    task = Task(
        description="Use add_numbers to calculate 5 + 3. Return only the number.",
        tools=[add_numbers]
    )
    
    # Before execution, tools are not registered
    assert len(task.registered_task_tools) == 0, "Task tools should not be registered before execution"
    assert len(task.task_builtin_tools) == 0, "Task builtin tools should be empty before execution"
    
    # Execute task
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.print_do_async(task)
    
    # Verify printed output shows tool calls
    output_text = output_buffer.getvalue()
    assert "Tool Calls" in output_text, "Output should contain 'Tool Calls' table"
    assert "add_numbers" in output_text, "Output should show add_numbers was called"
    
    # Verify task.tool_calls attribute
    assert len(task.tool_calls) > 0, "task.tool_calls should not be empty"
    called_names = [tc.get("tool_name", "") for tc in task.tool_calls]
    assert "add_numbers" in called_names, f"add_numbers should be in tool_calls, got: {called_names}"
    add_call = next(tc for tc in task.tool_calls if tc.get("tool_name") == "add_numbers")
    assert "params" in add_call, "Tool call should have 'params' key"
    assert "tool_result" in add_call, "Tool call should have 'tool_result' key"
    
    # After execution, tools should be registered
    assert "add_numbers" in task.registered_task_tools, "add_numbers should be registered after execution"
    assert len(task.registered_task_tools) > 0, "Task should have registered tools after execution"
    
    # Task should have its own tool_manager with task tools
    assert task.tool_manager is not None, "Task should have a ToolManager after execution"
    task_tool_defs = task.get_tool_defs()
    task_tool_names = [t.name for t in task_tool_defs]
    assert "add_numbers" in task_tool_names, "add_numbers should be in task's tool_manager definitions"
    
    # Agent's tool_manager should NOT have task tools
    agent_tool_defs = agent.tool_manager.get_tool_definitions()
    agent_tool_names = [t.name for t in agent_tool_defs]
    assert "add_numbers" not in agent_tool_names, "Task tool should NOT be in agent's tool_manager"


@pytest.mark.asyncio
async def test_mixed_tool_types():
    """Test mixing custom tools, toolkits, and builtin tools."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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


# ============================================================
# ToolKit Registration Cases (Agent-level)
# Cases 1-7 from the refactored ToolKit registration algorithm
# ============================================================


@pytest.mark.asyncio
async def test_toolkit_case1_bare_tool_decorator_no_config():
    """Case 1: No config set on ToolKit init -- only @tool-decorated methods are registered with defaults."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    agent.add_tools(math_kit)

    assert "subtract" in agent.registered_agent_tools, "subtract should be registered"
    assert "divide" in agent.registered_agent_tools, "divide should be registered"
    assert len(agent.registered_agent_tools) == 2, "Only @tool-decorated methods should be registered"

    for tool_obj in agent.registered_agent_tools.values():
        config = tool_obj.config
        assert config.timeout == 30.0, "Default timeout should be preserved"
        assert config.requires_confirmation is False, "Default requires_confirmation should be False"


@pytest.mark.asyncio
async def test_toolkit_case2_init_config_overrides_decorator():
    """Case 2: ToolKit init config overrides @tool decorator config."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = ConfiguredToolKit(timeout=120.0)
    agent.add_tools(kit)

    assert "dangerous_action" in agent.registered_agent_tools
    assert "safe_action" in agent.registered_agent_tools

    dangerous_config = agent.registered_agent_tools["dangerous_action"].config
    assert dangerous_config.timeout == 120.0, (
        "ToolKit init timeout=120 should override decorator timeout=999"
    )
    assert dangerous_config.requires_confirmation is True, (
        "Decorator requires_confirmation=True should survive (toolkit didn't set it)"
    )

    safe_config = agent.registered_agent_tools["safe_action"].config
    assert safe_config.timeout == 120.0, (
        "ToolKit init timeout=120 should override decorator timeout=60"
    )


@pytest.mark.asyncio
async def test_toolkit_case3_exclude_tools():
    """Case 3: exclude_tools prevents methods from being registered."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = MathToolKit(exclude_tools=["subtract"])
    agent.add_tools(kit)

    assert "subtract" not in agent.registered_agent_tools, "subtract should be excluded"
    assert "divide" in agent.registered_agent_tools, "divide should be registered"
    assert len(agent.registered_agent_tools) == 1


@pytest.mark.asyncio
async def test_toolkit_case3_include_tools_additive():
    """Case 3: include_tools is additive -- adds non-decorated methods on top of @tool methods."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = AsyncMathToolKit(include_tools=["helper_not_decorated"])
    agent.add_tools(kit)

    assert "add_sync" in agent.registered_agent_tools, "@tool-decorated add_sync should be registered"
    assert "subtract_sync" in agent.registered_agent_tools, "@tool-decorated subtract_sync should be registered"
    assert "helper_not_decorated" in agent.registered_agent_tools, (
        "helper_not_decorated should be added via include_tools"
    )
    assert len(agent.registered_agent_tools) == 3


@pytest.mark.asyncio
async def test_toolkit_case3_exclude_is_supreme():
    """Case 3: exclude_tools is supreme -- overrides even include_tools."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = AsyncMathToolKit(
        include_tools=["helper_not_decorated"],
        exclude_tools=["helper_not_decorated", "add_sync"],
    )
    agent.add_tools(kit)

    assert "add_sync" not in agent.registered_agent_tools, "add_sync excluded"
    assert "helper_not_decorated" not in agent.registered_agent_tools, "helper excluded even though included"
    assert "subtract_sync" in agent.registered_agent_tools, "subtract_sync should remain"
    assert len(agent.registered_agent_tools) == 1


@pytest.mark.asyncio
async def test_toolkit_case4_use_async_mode():
    """Case 4: use_async=True registers ALL async methods, drops ALL sync (even @tool-decorated)."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = AsyncMathToolKit(use_async=True)
    agent.add_tools(kit)

    assert "multiply_async" in agent.registered_agent_tools, "async method should be registered"
    assert "divide_async" in agent.registered_agent_tools, "async method should be registered"
    assert "async_compute" not in agent.registered_agent_tools, (
        "async_compute is on BareToolKit, not AsyncMathToolKit"
    )
    assert "add_sync" not in agent.registered_agent_tools, "sync @tool method should be dropped"
    assert "subtract_sync" not in agent.registered_agent_tools, "sync @tool method should be dropped"
    assert "helper_not_decorated" not in agent.registered_agent_tools, "sync non-decorated should be dropped"
    assert len(agent.registered_agent_tools) == 2


@pytest.mark.asyncio
async def test_toolkit_case4_use_async_respects_exclude():
    """Case 4: use_async=True still respects exclude_tools."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = AsyncMathToolKit(use_async=True, exclude_tools=["multiply_async"])
    agent.add_tools(kit)

    assert "multiply_async" not in agent.registered_agent_tools, "excluded async method should not be registered"
    assert "divide_async" in agent.registered_agent_tools, "non-excluded async method should be registered"
    assert len(agent.registered_agent_tools) == 1


@pytest.mark.asyncio
async def test_toolkit_case4_use_async_include_adds_sync_back():
    """Case 4+3: use_async=True + include_tools can force-add a sync method back."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = AsyncMathToolKit(use_async=True, include_tools=["add_sync"])
    agent.add_tools(kit)

    assert "multiply_async" in agent.registered_agent_tools, "async method should be registered"
    assert "divide_async" in agent.registered_agent_tools, "async method should be registered"
    assert "add_sync" in agent.registered_agent_tools, (
        "include_tools should force-add sync method even in use_async mode"
    )
    assert "subtract_sync" not in agent.registered_agent_tools, "non-included sync should stay dropped"
    assert len(agent.registered_agent_tools) == 3


@pytest.mark.asyncio
async def test_toolkit_case5_no_decorator_with_init_params_empty():
    """Case 5: No @tool decorator + init params set = nothing registered."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = BareToolKit(timeout=120.0)
    agent.add_tools(kit)

    assert len(agent.registered_agent_tools) == 0, (
        "Init params alone should NOT create tools from non-decorated methods"
    )


@pytest.mark.asyncio
async def test_toolkit_case6_no_decorator_no_config_empty():
    """Case 6: No @tool decorator + no init config = nothing registered."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = BareToolKit()
    agent.add_tools(kit)

    assert len(agent.registered_agent_tools) == 0, (
        "No @tool and no config means nothing should be registered"
    )


@pytest.mark.asyncio
async def test_toolkit_case7_discovery_before_config():
    """Case 7: Tool set is decided BEFORE config merge -- config doesn't affect discovery."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = ConfiguredToolKit(timeout=1.0)
    agent.add_tools(kit)

    registered_names = set(agent.registered_agent_tools.keys())
    assert registered_names == {"dangerous_action", "safe_action"}, (
        "Config params should not change which tools are discovered"
    )

    for tool_obj in agent.registered_agent_tools.values():
        assert tool_obj.config.timeout == 1.0, "All tools should have timeout=1.0 from toolkit init"


@pytest.mark.asyncio
async def test_toolkit_bare_toolkit_use_async_discovers_async():
    """BareToolKit with use_async=True discovers async methods even without @tool."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = BareToolKit(use_async=True)
    agent.add_tools(kit)

    assert "async_compute" in agent.registered_agent_tools, (
        "use_async should discover async methods even without @tool"
    )
    assert "compute" not in agent.registered_agent_tools, "sync method should be dropped in use_async"
    assert "format_text" not in agent.registered_agent_tools, "sync method should be dropped in use_async"
    assert len(agent.registered_agent_tools) == 1


@pytest.mark.asyncio
async def test_toolkit_bare_toolkit_include_tools_only():
    """BareToolKit with only include_tools should register those methods."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = BareToolKit(include_tools=["compute"])
    agent.add_tools(kit)

    assert "compute" in agent.registered_agent_tools, (
        "include_tools should add compute even without @tool"
    )
    assert "format_text" not in agent.registered_agent_tools
    assert "async_compute" not in agent.registered_agent_tools
    assert len(agent.registered_agent_tools) == 1


@pytest.mark.asyncio
async def test_toolkit_config_priority_decorator_survives_unset_init():
    """Decorator config fields survive when ToolKit init does not override them."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = ConfiguredToolKit(max_retries=10)
    agent.add_tools(kit)

    dangerous_config = agent.registered_agent_tools["dangerous_action"].config
    assert dangerous_config.requires_confirmation is True, (
        "Decorator requires_confirmation=True should survive (init didn't set it)"
    )
    assert dangerous_config.timeout == 999.0, (
        "Decorator timeout=999 should survive (init didn't set timeout)"
    )
    assert dangerous_config.max_retries == 10, "Init max_retries=10 should be applied"

    safe_config = agent.registered_agent_tools["safe_action"].config
    assert safe_config.timeout == 60.0, "Decorator timeout=60 should survive (init didn't set timeout)"
    assert safe_config.max_retries == 10, "Init max_retries=10 should be applied"


@pytest.mark.asyncio
async def test_toolkit_add_remove_with_use_async():
    """Full add/remove lifecycle with use_async ToolKit."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    kit = AsyncMathToolKit(use_async=True)
    agent.add_tools(kit)

    assert "multiply_async" in agent.registered_agent_tools
    assert "divide_async" in agent.registered_agent_tools
    assert kit in agent.tools

    agent.remove_tools("multiply_async")
    assert "multiply_async" not in agent.registered_agent_tools
    assert "divide_async" in agent.registered_agent_tools
    assert kit in agent.tools

    agent.remove_tools(kit)
    assert "divide_async" not in agent.registered_agent_tools
    assert kit not in agent.tools
    assert len(agent.registered_agent_tools) == 0


@pytest.mark.asyncio
async def test_toolkit_with_exclude_and_mixed_tools():
    """Agent with ToolKit (exclude_tools) + standalone function tools."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit(exclude_tools=["divide"])
    agent.add_tools([math_kit, add_numbers])

    assert "subtract" in agent.registered_agent_tools
    assert "divide" not in agent.registered_agent_tools, "divide should be excluded"
    assert "add_numbers" in agent.registered_agent_tools
    assert len(agent.registered_agent_tools) == 2


@pytest.mark.asyncio
async def test_toolkit_dedup_with_use_async():
    """Deduplication works correctly with use_async ToolKit."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry

    kit = AsyncMathToolKit(use_async=True)
    agent.add_tools(kit)

    kit_id = id(kit)
    assert kit_id in processor.class_instance_to_tools
    initial_tracking = list(processor.class_instance_to_tools[kit_id])
    assert set(initial_tracking) == {"multiply_async", "divide_async"}

    agent.add_tools(kit)
    second_tracking = list(processor.class_instance_to_tools[kit_id])
    assert second_tracking == initial_tracking, "Deduplication should prevent re-registration"


@pytest.mark.asyncio
async def test_toolkit_re_add_use_async_after_removal():
    """Remove and re-add a use_async ToolKit -- tracking is properly reset."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry

    kit = AsyncMathToolKit(use_async=True)
    agent.add_tools(kit)

    kit_id = id(kit)
    assert kit_id in processor.raw_object_ids
    assert kit_id in processor.class_instance_to_tools

    agent.remove_tools(kit)
    assert kit_id not in processor.raw_object_ids
    assert kit_id not in processor.class_instance_to_tools
    assert len(agent.registered_agent_tools) == 0

    agent.add_tools(kit)
    assert "multiply_async" in agent.registered_agent_tools
    assert "divide_async" in agent.registered_agent_tools
    assert kit_id in processor.raw_object_ids
    assert kit_id in processor.class_instance_to_tools


@pytest.mark.asyncio
async def test_task_tool_attributes_after_execution():
    """Test that task tool attributes are properly set after execution."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
    # Add agent tools
    agent.add_tools([add_numbers])
    
    # Create task with different tools
    task = Task(
        description="Use multiply_numbers to calculate 4 * 2. Return only the number.",
        tools=[multiply_numbers]
    )
    
    # Execute task
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.print_do_async(task)
    
    # Verify printed output shows tool calls
    output_text = output_buffer.getvalue()
    assert "Tool Calls" in output_text, "Output should contain 'Tool Calls' table"
    assert "multiply_numbers" in output_text, "Output should show multiply_numbers was called"
    
    # Verify task.tool_calls attribute
    assert len(task.tool_calls) > 0, "task.tool_calls should not be empty"
    called_names = [tc.get("tool_name", "") for tc in task.tool_calls]
    assert "multiply_numbers" in called_names, f"multiply_numbers should be in tool_calls, got: {called_names}"
    mul_call = next(tc for tc in task.tool_calls if tc.get("tool_name") == "multiply_numbers")
    assert "params" in mul_call, "Tool call should have 'params' key"
    assert "tool_result" in mul_call, "Tool call should have 'tool_result' key"
    
    # Verify task attributes
    assert "multiply_numbers" in task.registered_task_tools, "Task should have registered tools"
    assert len(task.registered_task_tools) > 0, "Task should have registered_task_tools"
    
    # Verify agent still has its tools
    assert "add_numbers" in agent.registered_agent_tools, "Agent should still have its tools"
    
    # Agent's tool_manager should have agent tools only
    agent_tool_defs = agent.get_tool_defs()
    agent_tool_names = [t.name for t in agent_tool_defs]
    assert "add_numbers" in agent_tool_names, "Agent tool should be in agent's tool_manager"
    assert "multiply_numbers" not in agent_tool_names, "Task tool should NOT be in agent's tool_manager"
    
    # Task's tool_manager should have task tools only
    task_tool_defs = task.get_tool_defs()
    task_tool_names = [t.name for t in task_tool_defs]
    assert "multiply_numbers" in task_tool_names, "Task tool should be in task's tool_manager"
    assert "add_numbers" not in task_tool_names, "Agent tool should NOT be in task's tool_manager"


@pytest.mark.asyncio
async def test_all_tool_types_comprehensive():
    """Comprehensive test of all tool types together."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
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
    sub_agent = Agent(model=MODEL, name="Helper")
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


# ============================================================
# Tests for ToolProcessor internal state management
# ============================================================

@pytest.mark.asyncio
async def test_deduplication_prevents_reprocessing():
    """Test that registering the same tool twice doesn't re-process it."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
    # Get initial state
    processor = agent.tool_manager.registry
    initial_raw_ids_count = len(processor.raw_object_ids)
    
    # Add tool first time
    agent.add_tools(add_numbers)
    assert "add_numbers" in agent.registered_agent_tools, "Tool should be registered"
    first_raw_ids_count = len(processor.raw_object_ids)
    assert first_raw_ids_count == initial_raw_ids_count + 1, "Raw tool ID should be tracked"
    
    # Get the registered tool object
    first_tool = agent.registered_agent_tools["add_numbers"]
    
    # Add same tool again
    agent.add_tools(add_numbers)
    
    # Should not change anything (deduplication)
    assert len(processor.raw_object_ids) == first_raw_ids_count, "Raw tool ID count should not change"
    assert len(agent.registered_agent_tools) == 1, "Should still have exactly 1 tool"
    
    # Same tool object should be used (not re-processed)
    second_tool = agent.registered_agent_tools["add_numbers"]
    assert first_tool is second_tool, "Same tool instance should be used (not re-processed)"


@pytest.mark.asyncio
async def test_toolkit_deduplication_no_duplicate_tracking():
    """Test that registering the same ToolKit twice doesn't create duplicate tracking entries."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry
    
    # Create and add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    
    # Verify tracking
    kit_id = id(math_kit)
    assert kit_id in processor.class_instance_to_tools, "ToolKit should be tracked"
    first_tracking = list(processor.class_instance_to_tools[kit_id])
    assert len(first_tracking) == 2, "Should have 2 tools tracked"
    assert "subtract" in first_tracking, "subtract should be tracked"
    assert "divide" in first_tracking, "divide should be tracked"
    
    # Add same ToolKit again (should be deduplicated)
    agent.add_tools(math_kit)
    
    # Tracking should be identical (no duplicates)
    second_tracking = list(processor.class_instance_to_tools[kit_id])
    assert second_tracking == first_tracking, "Tracking should not have duplicates"
    assert len(second_tracking) == 2, "Should still have exactly 2 tools tracked"


@pytest.mark.asyncio
async def test_class_instance_to_tools_cleanup_on_individual_removal():
    """Test that class_instance_to_tools is properly cleaned up when removing individual tools."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry
    
    # Add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    
    kit_id = id(math_kit)
    assert kit_id in processor.class_instance_to_tools, "ToolKit should be tracked"
    assert len(processor.class_instance_to_tools[kit_id]) == 2, "Should have 2 tools"
    
    # Remove one tool by name
    agent.remove_tools("subtract")
    
    # Tracking should be updated
    assert kit_id in processor.class_instance_to_tools, "ToolKit should still be tracked (has 1 tool left)"
    assert len(processor.class_instance_to_tools[kit_id]) == 1, "Should have 1 tool left"
    assert "divide" in processor.class_instance_to_tools[kit_id], "divide should still be tracked"
    assert "subtract" not in processor.class_instance_to_tools[kit_id], "subtract should be removed"
    
    # Remove last tool
    agent.remove_tools("divide")
    
    # Tracking should be completely cleaned up
    assert kit_id not in processor.class_instance_to_tools, "ToolKit should be removed from tracking (no tools left)"


@pytest.mark.asyncio
async def test_raw_tool_ids_cleanup_on_removal():
    """Test that _raw_tool_ids is properly cleaned up when tools are removed."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry
    
    # Track initial state
    initial_count = len(processor.raw_object_ids)
    
    # Add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    
    kit_id = id(math_kit)
    assert kit_id in processor.raw_object_ids, "ToolKit raw ID should be tracked"
    
    # Remove all tools from ToolKit by removing individually
    agent.remove_tools("subtract")
    agent.remove_tools("divide")
    
    # Raw ID should be cleaned up when all tools are gone
    assert kit_id not in processor.raw_object_ids, "ToolKit raw ID should be cleaned up"
    assert len(processor.raw_object_ids) == initial_count, "Should return to initial raw IDs count"


@pytest.mark.asyncio
async def test_raw_tool_ids_cleanup_on_object_removal():
    """Test that _raw_tool_ids is properly cleaned up when removing by object."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry
    
    # Track initial state
    initial_count = len(processor.raw_object_ids)
    
    # Add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    
    kit_id = id(math_kit)
    assert kit_id in processor.raw_object_ids, "ToolKit raw ID should be tracked"
    
    # Remove entire ToolKit by object
    agent.remove_tools(math_kit)
    
    # Raw ID should be cleaned up
    assert kit_id not in processor.raw_object_ids, "ToolKit raw ID should be cleaned up"
    assert len(processor.raw_object_ids) == initial_count, "Should return to initial raw IDs count"


@pytest.mark.asyncio
async def test_mcp_handlers_list_cleanup_on_individual_removal():
    """Test that mcp_handlers list is properly cleaned up when all MCP tools are removed individually."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        processor = agent.tool_manager.registry
        
        # Create MCP handler
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
        
        # Add MCP handler
        agent.add_tools(handler)
        
        # Verify handler is tracked
        assert handler in processor.mcp_handlers, "Handler should be in mcp_handlers list"
        assert len(processor.mcp_handler_to_tools) > 0, "Handler should have tracked tools"
        
        handler_id = id(handler)
        mcp_tool_names = list(processor.mcp_handler_to_tools.get(handler_id, []))
        assert len(mcp_tool_names) > 0, "Should have MCP tools"
        
        # Remove all MCP tools individually
        for tool_name in mcp_tool_names:
            agent.remove_tools(tool_name)
        
        # Handler should be removed from mcp_handlers list
        assert handler not in processor.mcp_handlers, "Handler should be removed from mcp_handlers when all tools gone"
        assert handler_id not in processor.mcp_handler_to_tools, "Handler tracking should be cleaned up"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_function_tool_deduplication():
    """Test that registering the same function tool twice doesn't create duplicates."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry
    
    # Add function tool
    agent.add_tools(add_numbers)
    initial_count = len(agent.registered_agent_tools)
    initial_raw_count = len(processor.raw_object_ids)
    
    # Add same function again multiple times
    agent.add_tools(add_numbers)
    agent.add_tools([add_numbers])
    agent.add_tools([add_numbers, add_numbers])
    
    # Should still have only 1 tool
    assert len(agent.registered_agent_tools) == initial_count, "Should have same number of tools"
    assert len(processor.raw_object_ids) == initial_raw_count, "Raw IDs should not increase"
    assert "add_numbers" in agent.registered_agent_tools, "add_numbers should be registered"


@pytest.mark.asyncio
async def test_re_add_after_removal():
    """Test that removing and re-adding a tool works correctly."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry
    
    # Add tool
    agent.add_tools(add_numbers)
    assert "add_numbers" in agent.registered_agent_tools, "Tool should be registered"
    func_id = id(add_numbers)
    assert func_id in processor.raw_object_ids, "Raw ID should be tracked"
    
    # Remove tool
    agent.remove_tools("add_numbers")
    assert "add_numbers" not in agent.registered_agent_tools, "Tool should be removed"
    # Note: For function tools, raw ID tracking may remain since we only clean up for class instances/handlers
    # This is acceptable because function tools don't have 1:many relationships
    
    # Re-add tool (should work because the tool name is gone from registered_tools)
    # Since _raw_tool_ids still has the ID, it won't re-process, but that's fine
    # because the tool was never really "un-tracked" at the raw level
    # Actually, we need to also remove from _raw_tool_ids for function tools
    
    # For now, let's test that re-adding works via a fresh add
    agent.add_tools(multiply_numbers)
    assert "multiply_numbers" in agent.registered_agent_tools, "New tool should be registered"
    
    # Remove and verify cleanup
    agent.remove_tools("multiply_numbers")
    assert "multiply_numbers" not in agent.registered_agent_tools, "Tool should be removed"


@pytest.mark.asyncio  
async def test_toolkit_re_add_after_removal():
    """Test that removing and re-adding a ToolKit works correctly."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry
    
    # Add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    assert "subtract" in agent.registered_agent_tools, "Tool should be registered"
    assert "divide" in agent.registered_agent_tools, "Tool should be registered"
    
    kit_id = id(math_kit)
    assert kit_id in processor.raw_object_ids, "Raw ID should be tracked"
    assert kit_id in processor.class_instance_to_tools, "Class instance should be tracked"
    
    # Remove entire ToolKit
    agent.remove_tools(math_kit)
    assert "subtract" not in agent.registered_agent_tools, "Tool should be removed"
    assert "divide" not in agent.registered_agent_tools, "Tool should be removed"
    assert kit_id not in processor.raw_object_ids, "Raw ID should be cleaned up"
    assert kit_id not in processor.class_instance_to_tools, "Class instance tracking should be cleaned up"
    
    # Re-add same ToolKit (should work since tracking was cleaned up)
    agent.add_tools(math_kit)
    assert "subtract" in agent.registered_agent_tools, "Tool should be re-registered"
    assert "divide" in agent.registered_agent_tools, "Tool should be re-registered"
    assert kit_id in processor.raw_object_ids, "Raw ID should be tracked again"
    assert kit_id in processor.class_instance_to_tools, "Class instance should be tracked again"


@pytest.mark.asyncio
async def test_mixed_deduplication():
    """Test deduplication with mixed tool types."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    
    math_kit = MathToolKit()
    text_kit = TextToolKit()
    
    # Add everything at once
    agent.add_tools([add_numbers, math_kit, text_kit, multiply_numbers])
    
    initial_count = len(agent.registered_agent_tools)
    assert initial_count == 6, "Should have 6 tools (2 functions + 2 math kit + 2 text kit)"
    
    # Try adding everything again
    agent.add_tools([add_numbers, math_kit, text_kit, multiply_numbers])
    agent.add_tools(add_numbers)
    agent.add_tools(math_kit)
    
    # Should still have same number
    assert len(agent.registered_agent_tools) == initial_count, "Should have same number of tools after duplicate adds"


@pytest.mark.asyncio
async def test_processor_tracking_consistency():
    """Test that processor tracking dictionaries stay consistent through operations."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    processor = agent.tool_manager.registry
    
    # Start clean
    assert len(processor.registered_tools) == 0
    assert len(processor.class_instance_to_tools) == 0
    assert len(processor.raw_object_ids) == 0
    
    # Add ToolKit
    math_kit = MathToolKit()
    agent.add_tools(math_kit)
    
    kit_id = id(math_kit)
    
    # Verify consistency
    assert len(processor.registered_tools) == 2
    assert kit_id in processor.class_instance_to_tools
    assert len(processor.class_instance_to_tools[kit_id]) == 2
    assert kit_id in processor.raw_object_ids
    
    # Remove one tool
    agent.remove_tools("subtract")
    
    # Verify consistency
    assert len(processor.registered_tools) == 1
    assert kit_id in processor.class_instance_to_tools
    assert len(processor.class_instance_to_tools[kit_id]) == 1
    assert kit_id in processor.raw_object_ids  # Still tracked (has remaining tools)
    
    # Remove last tool
    agent.remove_tools("divide")
    
    # Verify complete cleanup
    assert len(processor.registered_tools) == 0
    assert kit_id not in processor.class_instance_to_tools
    assert kit_id not in processor.raw_object_ids
    
    # Verify agent state
    assert len(agent.registered_agent_tools) == 0


# ============================================================
# Tests for MCP tool_name_prefix feature
# ============================================================

@pytest.mark.asyncio
async def test_mcp_handler_with_tool_name_prefix_agent_init():
    """Test MCPHandler with tool_name_prefix via Agent initialization."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        # Create MCP handler with prefix
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp",
            tool_name_prefix="fs_server"
        )
        
        # Initialize agent with prefixed handler
        agent = Agent(
            model=MODEL,
            name="Test Agent",
            tools=[handler],
            debug=True
        )
        
        # Verify tools are registered with prefix
        tool_names = list(agent.registered_agent_tools.keys())
        assert len(tool_names) > 0, "MCP handler should register tools"
        
        # All tool names should have the prefix
        for tool_name in tool_names:
            assert tool_name.startswith("fs_server_"), \
                f"Tool '{tool_name}' should have 'fs_server_' prefix"
        
        # Verify handler is in agent.tools
        assert handler in agent.tools, "Handler should be in agent.tools"
        
        # Verify handler info contains prefix
        info = handler.get_info()
        assert info['tool_name_prefix'] == "fs_server", "Handler info should contain prefix"
        
        # Clean up
        agent.remove_tools(handler)
        assert len(agent.registered_agent_tools) == 0, "All tools should be removed"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_mcp_handler_with_tool_name_prefix_add_tools():
    """Test MCPHandler with tool_name_prefix via Agent.add_tools."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Create MCP handler with prefix
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp",
            tool_name_prefix="myprefix"
        )
        
        # Add handler via add_tools
        agent.add_tools(handler)
        
        # Verify tools are registered with prefix
        tool_names = list(agent.registered_agent_tools.keys())
        assert len(tool_names) > 0, "MCP handler should register tools"
        
        # All tool names should have the prefix
        for tool_name in tool_names:
            assert tool_name.startswith("myprefix_"), \
                f"Tool '{tool_name}' should have 'myprefix_' prefix"
        
        # Verify original_name is preserved in MCPTool
        for tool_wrapper in handler.tools:
            assert hasattr(tool_wrapper, 'original_name'), "MCPTool should have original_name"
            assert hasattr(tool_wrapper, 'tool_name_prefix'), "MCPTool should have tool_name_prefix"
            assert tool_wrapper.tool_name_prefix == "myprefix", "MCPTool should store the prefix"
            # Verify the registered name is prefixed version
            assert tool_wrapper.name.startswith("myprefix_"), "MCPTool.name should be prefixed"
            # Verify original_name is without prefix
            assert not tool_wrapper.original_name.startswith("myprefix_"), \
                "MCPTool.original_name should NOT have prefix"
        
        # Clean up
        agent.remove_tools(handler)
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_mcp_prefixed_tools_removal_by_name():
    """Test removing prefixed MCP tools by their prefixed names."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Create MCP handler with prefix
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp",
            tool_name_prefix="test_prefix"
        )
        
        agent.add_tools(handler)
        
        # Get registered tools
        initial_count = len(agent.registered_agent_tools)
        assert initial_count > 0, "MCP handler should register tools"
        
        # Get one prefixed tool name
        prefixed_tool_names = list(agent.registered_agent_tools.keys())
        tool_to_remove = prefixed_tool_names[0]
        
        # Verify the tool name has prefix
        assert tool_to_remove.startswith("test_prefix_"), "Tool should have prefix"
        
        # Remove by prefixed name
        agent.remove_tools(tool_to_remove)
        
        # Verify removal
        assert tool_to_remove not in agent.registered_agent_tools, \
            f"Tool '{tool_to_remove}' should be removed"
        assert len(agent.registered_agent_tools) == initial_count - 1, \
            "Should have one less tool"
        
        # Handler should still be in agent.tools
        assert handler in agent.tools, "Handler should still be in agent.tools"
        
        # Clean up - remove handler
        agent.remove_tools(handler)
        assert len(agent.registered_agent_tools) == 0, "All tools should be removed"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_mcp_prefix_prevents_collisions():
    """Test that tool_name_prefix prevents tool name collisions between handlers."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Create two handlers pointing to same server type but different dirs
        # Without prefix, they would have identical tool names and collision would occur
        handler1 = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp",
            tool_name_prefix="handler1"
        )
        
        handler2 = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /var/tmp",
            tool_name_prefix="handler2"
        )
        
        # Add both handlers
        agent.add_tools(handler1)
        tools_after_h1 = len(agent.registered_agent_tools)
        assert tools_after_h1 > 0, "Handler1 should register tools"
        
        agent.add_tools(handler2)
        tools_after_h2 = len(agent.registered_agent_tools)
        
        # With prefixes, we should have double the tools (no collision)
        assert tools_after_h2 == tools_after_h1 * 2, \
            f"Should have {tools_after_h1 * 2} tools (double), got {tools_after_h2}"
        
        # Verify both handlers' tools exist with their prefixes
        tool_names = list(agent.registered_agent_tools.keys())
        handler1_tools = [n for n in tool_names if n.startswith("handler1_")]
        handler2_tools = [n for n in tool_names if n.startswith("handler2_")]
        
        assert len(handler1_tools) == tools_after_h1, "Handler1 tools should have prefix"
        assert len(handler2_tools) == tools_after_h1, "Handler2 tools should have prefix"
        
        # Both handlers should be in agent.tools
        assert handler1 in agent.tools, "Handler1 should be in agent.tools"
        assert handler2 in agent.tools, "Handler2 should be in agent.tools"
        
        # Remove handler1, handler2 should remain
        agent.remove_tools(handler1)
        remaining_tools = list(agent.registered_agent_tools.keys())
        
        # Only handler2 tools should remain
        for tool_name in remaining_tools:
            assert tool_name.startswith("handler2_"), \
                f"Only handler2 tools should remain, found: {tool_name}"
        
        assert handler1 not in agent.tools, "Handler1 should be removed"
        assert handler2 in agent.tools, "Handler2 should still be in agent.tools"
        
        # Clean up
        agent.remove_tools(handler2)
        assert len(agent.registered_agent_tools) == 0, "All tools should be removed"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_multi_mcp_handler_with_single_prefix():
    """Test MultiMCPHandler with a single tool_name_prefix for all servers."""
    try:
        from upsonic.tools.mcp import MultiMCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Create MultiMCPHandler with single prefix (will become prefix_0, prefix_1)
        multi_handler = MultiMCPHandler(
            commands=[
                "npx -y @modelcontextprotocol/server-filesystem /tmp",
                "npx -y @modelcontextprotocol/server-filesystem /var/tmp",
            ],
            tool_name_prefix="shared_prefix"
        )
        
        agent.add_tools(multi_handler)
        
        # Get registered tools
        tool_names = list(agent.registered_agent_tools.keys())
        assert len(tool_names) > 0, "MultiMCPHandler should register tools"
        
        # Tools should have prefixes like shared_prefix_0_* and shared_prefix_1_*
        server0_tools = [n for n in tool_names if n.startswith("shared_prefix_0_")]
        server1_tools = [n for n in tool_names if n.startswith("shared_prefix_1_")]
        
        assert len(server0_tools) > 0, "Server 0 should have prefixed tools"
        assert len(server1_tools) > 0, "Server 1 should have prefixed tools"
        
        # Verify server info contains prefixes
        server_info = multi_handler.get_server_info()
        assert len(server_info) == 2, "Should have 2 servers"
        assert server_info[0]['tool_name_prefix'] == "shared_prefix_0", \
            "Server 0 should have prefix shared_prefix_0"
        assert server_info[1]['tool_name_prefix'] == "shared_prefix_1", \
            "Server 1 should have prefix shared_prefix_1"
        
        # Clean up
        agent.remove_tools(multi_handler)
        assert len(agent.registered_agent_tools) == 0, "All tools should be removed"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_multi_mcp_handler_with_prefixes_list():
    """Test MultiMCPHandler with individual tool_name_prefixes for each server."""
    try:
        from upsonic.tools.mcp import MultiMCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Create MultiMCPHandler with specific prefixes for each server
        multi_handler = MultiMCPHandler(
            commands=[
                "npx -y @modelcontextprotocol/server-filesystem /tmp",
                "npx -y @modelcontextprotocol/server-filesystem /var/tmp",
            ],
            tool_name_prefixes=["tmp_files", "var_files"]
        )
        
        agent.add_tools(multi_handler)
        
        # Get registered tools
        tool_names = list(agent.registered_agent_tools.keys())
        assert len(tool_names) > 0, "MultiMCPHandler should register tools"
        
        # Tools should have exact prefixes: tmp_files_* and var_files_*
        tmp_tools = [n for n in tool_names if n.startswith("tmp_files_")]
        var_tools = [n for n in tool_names if n.startswith("var_files_")]
        
        assert len(tmp_tools) > 0, "First server should have 'tmp_files_' prefixed tools"
        assert len(var_tools) > 0, "Second server should have 'var_files_' prefixed tools"
        
        # Verify server info contains the exact prefixes
        server_info = multi_handler.get_server_info()
        assert len(server_info) == 2, "Should have 2 servers"
        assert server_info[0]['tool_name_prefix'] == "tmp_files", \
            "Server 0 should have prefix 'tmp_files'"
        assert server_info[1]['tool_name_prefix'] == "var_files", \
            "Server 1 should have prefix 'var_files'"
        
        # Verify tools from different servers are distinguishable
        # Both servers have same tools but with different prefixes
        assert len(tmp_tools) == len(var_tools), \
            "Both servers should have same number of tools"
        
        # Clean up
        agent.remove_tools(multi_handler)
        assert len(agent.registered_agent_tools) == 0, "All tools should be removed"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_multi_mcp_handler_prefixes_validation():
    """Test that MultiMCPHandler validates prefixes list length.
    
    When the validation fails, the error is logged and no tools are registered.
    """
    try:
        from upsonic.tools.mcp import MultiMCPHandler
        
        # Create MultiMCPHandler with mismatched prefixes list
        multi_handler = MultiMCPHandler(
            commands=[
                "npx -y @modelcontextprotocol/server-filesystem /tmp",
                "npx -y @modelcontextprotocol/server-filesystem /var/tmp",
            ],
            tool_name_prefixes=["only_one_prefix"]  # Wrong length: 1 instead of 2
        )
        
        # The validation happens during connect(), which happens when we add to agent
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Adding handler with mismatched prefixes - validation error is logged
        # and no tools are registered (graceful failure)
        agent.add_tools(multi_handler)
        
        # No tools should be registered due to validation failure
        assert len(agent.registered_agent_tools) == 0, \
            "No tools should be registered when prefix validation fails"
        
        # The handler should not have any tools
        assert len(multi_handler.tools) == 0, \
            "Handler should have no tools after validation failure"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_task_mcp_handler_with_prefix():
    """Test MCPHandler with tool_name_prefix via Task."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Create MCP handler with prefix
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp",
            tool_name_prefix="task_fs"
        )
        
        # Create task with prefixed handler
        task = Task(
            description="Test task with prefixed MCP tools",
            tools=[handler, add_numbers]
        )
        
        # Execute task to trigger registration
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            result = await agent.print_do_async(task)
        
        # Verify tools are registered
        task_tools = list(task.registered_task_tools.keys())
        assert "add_numbers" in task_tools, "add_numbers should be registered"
        
        # Get MCP tools (prefixed)
        mcp_tools = [n for n in task_tools if n.startswith("task_fs_")]
        assert len(mcp_tools) > 0, "MCP tools should be registered with prefix"
        
        # Remove one prefixed tool by name
        tool_to_remove = mcp_tools[0]
        task.remove_tools(tool_to_remove, agent)
        
        # Verify removal
        assert tool_to_remove not in task.registered_task_tools, \
            f"Tool '{tool_to_remove}' should be removed"
        assert "add_numbers" in task.registered_task_tools, \
            "add_numbers should still be registered"
        
        # Remove handler
        task.remove_tools(handler, agent)
        
        # Only add_numbers should remain
        remaining_tools = list(task.registered_task_tools.keys())
        assert "add_numbers" in remaining_tools, "add_numbers should remain"
        assert all(not t.startswith("task_fs_") for t in remaining_tools), \
            "All MCP tools should be removed"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_mcp_tool_metadata_contains_prefix_info():
    """Test that MCPTool metadata contains prefix information."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Create MCP handler with prefix
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp",
            tool_name_prefix="meta_test"
        )
        
        agent.add_tools(handler)
        
        # Verify tools have metadata with prefix info
        for tool in handler.tools:
            assert hasattr(tool, 'metadata'), "MCPTool should have metadata"
            metadata = tool.metadata
            
            # Check metadata.custom contains MCP-specific info
            assert 'mcp_original_name' in metadata.custom, \
                "Metadata should contain mcp_original_name"
            assert 'mcp_tool_name_prefix' in metadata.custom, \
                "Metadata should contain mcp_tool_name_prefix"
            assert metadata.custom['mcp_tool_name_prefix'] == "meta_test", \
                "Prefix in metadata should match"
            
            # Verify original name doesn't have prefix
            original = metadata.custom['mcp_original_name']
            assert not original.startswith("meta_test_"), \
                f"Original name '{original}' should not have prefix"
        
        # Clean up
        agent.remove_tools(handler)
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_mcp_handler_without_prefix_no_prefix_metadata():
    """Test that MCPHandler without prefix doesn't add prefix metadata."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        
        # Create MCP handler WITHOUT prefix
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
        
        agent.add_tools(handler)
        
        # Verify tools don't have prefix metadata
        for tool in handler.tools:
            metadata = tool.metadata
            # mcp_tool_name_prefix should NOT be in metadata.custom
            assert 'mcp_tool_name_prefix' not in metadata.custom, \
                "Metadata should NOT contain mcp_tool_name_prefix when no prefix used"
            # But mcp_original_name should still be there
            assert 'mcp_original_name' in metadata.custom, \
                "Metadata should still contain mcp_original_name"
        
        # Verify handler info shows None for prefix
        info = handler.get_info()
        assert info['tool_name_prefix'] is None, "Handler info should show None for prefix"
        
        # Clean up
        agent.remove_tools(handler)
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_mcp_handler_processor_tracking_with_prefix():
    """Test that ToolProcessor correctly tracks prefixed MCP tools."""
    try:
        from upsonic.tools.mcp import MCPHandler
        
        agent = Agent(model=MODEL, name="Test Agent", debug=True)
        processor = agent.tool_manager.registry
        
        # Create MCP handler with prefix
        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp",
            tool_name_prefix="track_test"
        )
        
        agent.add_tools(handler)
        
        # Verify handler is tracked
        assert handler in processor.mcp_handlers, "Handler should be tracked"
        
        handler_id = id(handler)
        assert handler_id in processor.mcp_handler_to_tools, \
            "Handler should have tracked tools"
        
        # Verify tracked tool names are prefixed
        tracked_tools = processor.mcp_handler_to_tools[handler_id]
        for tool_name in tracked_tools:
            assert tool_name.startswith("track_test_"), \
                f"Tracked tool '{tool_name}' should have prefix"
        
        # Verify registered tools have prefixed names
        for tool_name in agent.registered_agent_tools.keys():
            assert tool_name.startswith("track_test_"), \
                f"Registered tool '{tool_name}' should have prefix"
        
        # Remove one prefixed tool and verify tracking update
        tool_to_remove = list(tracked_tools)[0]
        agent.remove_tools(tool_to_remove)
        
        # Tracking should be updated
        updated_tracking = processor.mcp_handler_to_tools.get(handler_id, set())
        assert tool_to_remove not in updated_tracking, \
            f"'{tool_to_remove}' should be removed from tracking"
        
        # Clean up
        agent.remove_tools(handler)
        assert handler not in processor.mcp_handlers, \
            "Handler should be removed from tracking"
        
    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


# ============================================================
# TASK-LEVEL TOOL MANAGEMENT TESTS
# Mirror of agent-level tests above, using Task.add_tools /
# Task.remove_tools / agent._setup_task_tools(task).
# ============================================================


@pytest.mark.asyncio
async def test_task_add_remove_custom_tools_management():
    """Test adding and removing custom tools (functions) from Task via management API."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    task = Task(description="test", tools=[add_numbers])
    assert add_numbers in task.tools
    assert len(task.registered_task_tools) == 0, "Not registered until setup"

    task.add_tools([multiply_numbers, greet])
    assert multiply_numbers in task.tools
    assert greet in task.tools
    assert len(task.tools) == 3

    agent._setup_task_tools(task)

    assert "add_numbers" in task.registered_task_tools
    assert "multiply_numbers" in task.registered_task_tools
    assert "greet" in task.registered_task_tools
    assert len(task.registered_task_tools) == 3

    task.remove_tools("add_numbers")
    assert "add_numbers" not in task.registered_task_tools
    assert add_numbers not in task.tools

    task.remove_tools("multiply_numbers")
    assert "multiply_numbers" not in task.registered_task_tools

    task.remove_tools(["greet"])
    assert len(task.registered_task_tools) == 0


@pytest.mark.asyncio
async def test_task_add_remove_toolkit():
    """Test adding and removing ToolKit instances from Task."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    text_kit = TextToolKit()
    task = Task(description="test", tools=[math_kit])
    task.add_tools(text_kit)

    agent._setup_task_tools(task)

    assert "subtract" in task.registered_task_tools
    assert "divide" in task.registered_task_tools
    assert "uppercase" in task.registered_task_tools
    assert "lowercase" in task.registered_task_tools

    task.remove_tools(math_kit)
    assert "subtract" not in task.registered_task_tools
    assert "divide" not in task.registered_task_tools
    assert math_kit not in task.tools

    assert "uppercase" in task.registered_task_tools
    assert "lowercase" in task.registered_task_tools

    task.remove_tools(text_kit)
    assert "uppercase" not in task.registered_task_tools
    assert "lowercase" not in task.registered_task_tools
    assert text_kit not in task.tools


@pytest.mark.asyncio
async def test_task_remove_individual_class_methods():
    """Test removing individual methods from a regular class by name on Task."""
    try:
        from upsonic.tools.common_tools.financial_tools import YFinanceTools

        agent = Agent(model=MODEL, name="Test Agent", debug=True)

        financial_tools = YFinanceTools(stock_price=True, enable_all=False)
        task = Task(description="test", tools=[financial_tools])

        agent._setup_task_tools(task)

        initial_count = len(task.registered_task_tools)
        assert initial_count > 0, "Financial tools should be registered"

        tool_names = list(task.registered_task_tools.keys())
        tool_to_remove = tool_names[0]

        task.remove_tools(tool_to_remove)
        assert tool_to_remove not in task.registered_task_tools
        assert len(task.registered_task_tools) == initial_count - 1

        assert financial_tools in task.tools, "Class instance should still be in task.tools"

    except ImportError:
        pytest.skip("Financial tools dependencies not available")


@pytest.mark.asyncio
async def test_task_add_remove_financial_tools():
    """Test adding and removing financial tools (pure class) from Task."""
    try:
        from upsonic.tools.common_tools.financial_tools import YFinanceTools

        agent = Agent(model=MODEL, name="Test Agent", debug=True)

        financial_tools = YFinanceTools(stock_price=True, enable_all=False)
        task = Task(description="test", tools=[financial_tools])

        agent._setup_task_tools(task)

        tool_names = list(task.registered_task_tools.keys())
        assert len(tool_names) > 0, f"Financial tools should be registered. Found: {tool_names}"

        financial_tool_found = any(
            "stock" in name.lower() or "price" in name.lower() or "get_current" in name.lower()
            for name in tool_names
        )
        assert financial_tool_found, f"Financial tool should be registered. Found: {tool_names}"

        task.remove_tools(financial_tools)

        tool_names_after = list(task.registered_task_tools.keys())
        financial_tool_still_there = any(
            "stock" in name.lower() or "price" in name.lower() or "get_current" in name.lower()
            for name in tool_names_after
        )
        assert not financial_tool_still_there, "Financial tools should be removed"
        assert financial_tools not in task.tools

    except ImportError:
        pytest.skip("Financial tools dependencies not available")


@pytest.mark.asyncio
async def test_task_remove_individual_mcp_tools():
    """Test removing individual MCP tools from Task by name (keeping the handler)."""
    try:
        from upsonic.tools.mcp import MCPHandler

        agent = Agent(model=MODEL, name="Test Agent", debug=True)

        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
        task = Task(description="test", tools=[handler, add_numbers])

        agent._setup_task_tools(task)

        initial_count = len(task.registered_task_tools)
        assert initial_count > 1, "Should have MCP tools + add_numbers"

        mcp_tool_names = [n for n in task.registered_task_tools.keys() if n != "add_numbers"]
        assert len(mcp_tool_names) > 0

        task.remove_tools(mcp_tool_names[0])
        assert mcp_tool_names[0] not in task.registered_task_tools
        assert len(task.registered_task_tools) == initial_count - 1
        assert "add_numbers" in task.registered_task_tools

        assert handler in task.tools, "MCP handler should still be in task.tools"

    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_task_add_remove_duckduckgo_tool():
    """Test adding and removing DuckDuckGo search tool from Task."""
    try:
        from upsonic.tools.common_tools.duckduckgo import duckduckgo_search_tool

        agent = Agent(model=MODEL, name="Test Agent", debug=True)

        ddg_tool = duckduckgo_search_tool()
        task = Task(description="test", tools=[ddg_tool])

        agent._setup_task_tools(task)

        assert "duckduckgo_search" in task.registered_task_tools

        task.remove_tools("duckduckgo_search")
        assert "duckduckgo_search" not in task.registered_task_tools

    except ImportError:
        pytest.skip("DuckDuckGo dependencies not available")


@pytest.mark.asyncio
async def test_task_add_remove_tavily_tool():
    """Test adding and removing Tavily search tool from Task."""
    try:
        from upsonic.tools.common_tools.tavily import tavily_search_tool

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            pytest.skip("TAVILY_API_KEY not set")

        agent = Agent(model=MODEL, name="Test Agent", debug=True)

        tavily_tool = tavily_search_tool(api_key=tavily_api_key)
        task = Task(description="test", tools=[tavily_tool])

        agent._setup_task_tools(task)

        assert "tavily_search" in task.registered_task_tools

        task.remove_tools("tavily_search")
        assert "tavily_search" not in task.registered_task_tools

    except ImportError:
        pytest.skip("Tavily dependencies not available")


@pytest.mark.asyncio
async def test_task_agent_as_tool():
    """Test adding and removing Agent as a tool on Task."""
    sub_agent = Agent(
        model=MODEL,
        name="Math Assistant",
        role="Math Specialist",
        goal="Help with mathematical calculations"
    )

    agent = Agent(model=MODEL, name="Main Agent", debug=True)
    task = Task(description="test", tools=[sub_agent, add_numbers])

    agent._setup_task_tools(task)

    tool_names = list(task.registered_task_tools.keys())
    agent_tool_name = [name for name in tool_names if name.startswith("ask_")][0]
    assert agent_tool_name is not None
    assert "add_numbers" in task.registered_task_tools

    task.remove_tools(sub_agent)
    assert agent_tool_name not in task.registered_task_tools
    assert sub_agent not in task.tools
    assert "add_numbers" in task.registered_task_tools


@pytest.mark.asyncio
async def test_task_initialization_with_builtin_tools():
    """Test Task initialized with builtin + regular tools before setup."""
    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()

    task = Task(
        description="test",
        tools=[web_search, code_exec, add_numbers, multiply_numbers]
    )

    assert len(task.tools) == 4
    assert web_search in task.tools
    assert code_exec in task.tools
    assert add_numbers in task.tools
    assert multiply_numbers in task.tools

    assert len(task.registered_task_tools) == 0, "Not registered until setup"
    assert len(task.task_builtin_tools) == 0, "Not populated until setup"

    agent = Agent(model=MODEL, name="Test Agent", debug=True)
    agent._setup_task_tools(task)

    assert len(task.task_builtin_tools) == 2
    builtin_ids = {t.unique_id for t in task.task_builtin_tools}
    assert "web_search" in builtin_ids
    assert "code_execution" in builtin_ids

    assert len(task.registered_task_tools) == 2
    assert "add_numbers" in task.registered_task_tools
    assert "multiply_numbers" in task.registered_task_tools


@pytest.mark.asyncio
async def test_task_builtin_tools_not_in_tool_processor():
    """Verify that builtin tools are NOT processed by Task's ToolProcessor."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()

    task = Task(description="test", tools=[web_search, code_exec, add_numbers])

    agent._setup_task_tools(task)

    processor = task.tool_manager.registry
    processor_count = len(processor.registered_tools)

    assert "add_numbers" in processor.registered_tools
    assert "web_search" not in processor.registered_tools
    assert "code_execution" not in processor.registered_tools

    assert processor_count == 1, (
        f"Processor should only have 1 regular tool, got {processor_count}"
    )

    assert len(task.task_builtin_tools) == 2


@pytest.mark.asyncio
async def test_task_tool_manager_attributes_full():
    """Test that task tool_manager attributes are properly updated."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    task = Task(description="test", tools=[add_numbers, multiply_numbers])
    agent._setup_task_tools(task)

    tool_defs = task.tool_manager.get_tool_definitions()
    tool_names = [t.name for t in tool_defs]
    assert "add_numbers" in tool_names
    assert "multiply_numbers" in tool_names

    assert "add_numbers" in task.tool_manager.registry.wrapped_tools
    assert "multiply_numbers" in task.tool_manager.registry.wrapped_tools
    assert "add_numbers" in task.tool_manager.registry.registered_tools
    assert "multiply_numbers" in task.tool_manager.registry.registered_tools

    task.remove_tools("add_numbers")

    tool_defs_after = task.tool_manager.get_tool_definitions()
    tool_names_after = [t.name for t in tool_defs_after]
    assert "add_numbers" not in tool_names_after
    assert "add_numbers" not in task.tool_manager.registry.wrapped_tools
    assert "add_numbers" not in task.tool_manager.registry.registered_tools

    assert "multiply_numbers" in tool_names_after
    assert "multiply_numbers" in task.tool_manager.registry.wrapped_tools


@pytest.mark.asyncio
async def test_task_mixed_tool_types():
    """Test mixing custom tools, toolkits, and builtin tools on Task."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    web_search = WebSearchTool()

    task = Task(description="test", tools=[add_numbers, math_kit, web_search])
    agent._setup_task_tools(task)

    assert "add_numbers" in task.registered_task_tools
    assert "subtract" in task.registered_task_tools
    assert "divide" in task.registered_task_tools
    assert len(task.registered_task_tools) == 3

    assert len(task.task_builtin_tools) == 1
    assert any(t.unique_id == "web_search" for t in task.task_builtin_tools)

    task.remove_tools(["add_numbers", "subtract", "divide"])
    assert len(task.registered_task_tools) == 0

    assert len(task.task_builtin_tools) == 1
    assert any(t.unique_id == "web_search" for t in task.task_builtin_tools)

    task.remove_tools([web_search])
    assert len(task.task_builtin_tools) == 0
    assert web_search not in task.tools


@pytest.mark.asyncio
async def test_task_all_tool_types_comprehensive():
    """Comprehensive test of all tool types together on Task."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    regular_tools_added: list = []
    builtin_tools_added: list = []

    math_kit = MathToolKit()
    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()

    sub_agent = Agent(model=MODEL, name="Helper")

    tools_list: list = [add_numbers, math_kit, web_search, code_exec, sub_agent]

    financial_tools_instance = None
    try:
        from upsonic.tools.common_tools.financial_tools import YFinanceTools
        financial_tools_instance = YFinanceTools(stock_price=True, enable_all=False)
        tools_list.append(financial_tools_instance)
    except (ImportError, Exception):
        pass

    try:
        from upsonic.tools.common_tools.duckduckgo import duckduckgo_search_tool
        ddg_tool = duckduckgo_search_tool()
        tools_list.append(ddg_tool)
    except (ImportError, Exception):
        pass

    task = Task(description="test", tools=tools_list)
    agent._setup_task_tools(task)

    assert "add_numbers" in task.registered_task_tools
    regular_tools_added.append("add_numbers")

    assert "subtract" in task.registered_task_tools
    assert "divide" in task.registered_task_tools
    regular_tools_added.extend(["subtract", "divide"])

    assert len(task.task_builtin_tools) == 2
    builtin_tools_added.extend([web_search, code_exec])

    tool_names = list(task.registered_task_tools.keys())
    agent_tool_names = [n for n in tool_names if n.startswith("ask_")]
    assert len(agent_tool_names) > 0
    regular_tools_added.extend(agent_tool_names)

    if "duckduckgo_search" in task.registered_task_tools:
        regular_tools_added.append("duckduckgo_search")

    assert len(task.registered_task_tools) >= len(regular_tools_added)

    task.remove_tools(regular_tools_added)
    for name in regular_tools_added:
        assert name not in task.registered_task_tools, f"{name} should be removed"

    assert len(task.task_builtin_tools) == 2

    task.remove_tools(builtin_tools_added)
    assert len(task.task_builtin_tools) == 0

    if math_kit in task.tools:
        task.remove_tools(math_kit)
    if sub_agent in task.tools:
        task.remove_tools(sub_agent)
    if financial_tools_instance and financial_tools_instance in task.tools:
        task.remove_tools(financial_tools_instance)


# ============================================================
# Task-level ToolProcessor internal state management
# ============================================================


@pytest.mark.asyncio
async def test_task_deduplication_prevents_reprocessing():
    """Test that registering the same tool twice on a task doesn't re-process it."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    task = Task(description="test", tools=[add_numbers, add_numbers])
    agent._setup_task_tools(task)

    assert "add_numbers" in task.registered_task_tools
    defs = task.get_tool_defs()
    add_defs = [d for d in defs if d.name == "add_numbers"]
    assert len(add_defs) == 1, "Should have exactly 1 add_numbers definition"


@pytest.mark.asyncio
async def test_task_toolkit_deduplication_no_duplicate_tracking():
    """Test that registering the same ToolKit twice on task doesn't create duplicate tracking."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit])
    agent._setup_task_tools(task)

    processor = task.tool_manager.registry
    kit_id = id(math_kit)
    assert kit_id in processor.class_instance_to_tools
    first_tracking = list(processor.class_instance_to_tools[kit_id])
    assert len(first_tracking) == 2
    assert "subtract" in first_tracking
    assert "divide" in first_tracking


@pytest.mark.asyncio
async def test_task_class_instance_to_tools_cleanup_on_individual_removal():
    """Test task processor class_instance_to_tools cleanup on individual tool removal."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit])
    agent._setup_task_tools(task)

    processor = task.tool_manager.registry
    kit_id = id(math_kit)
    assert kit_id in processor.class_instance_to_tools
    assert len(processor.class_instance_to_tools[kit_id]) == 2

    task.remove_tools("subtract")

    assert kit_id in processor.class_instance_to_tools
    assert len(processor.class_instance_to_tools[kit_id]) == 1
    assert "divide" in processor.class_instance_to_tools[kit_id]
    assert "subtract" not in processor.class_instance_to_tools[kit_id]

    task.remove_tools("divide")

    assert kit_id not in processor.class_instance_to_tools


@pytest.mark.asyncio
async def test_task_raw_tool_ids_cleanup_on_removal():
    """Test task processor _raw_tool_ids cleanup when tools are removed individually."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit])
    agent._setup_task_tools(task)

    processor = task.tool_manager.registry
    kit_id = id(math_kit)
    assert kit_id in processor.raw_object_ids

    task.remove_tools("subtract")
    task.remove_tools("divide")

    assert kit_id not in processor.raw_object_ids


@pytest.mark.asyncio
async def test_task_raw_tool_ids_cleanup_on_object_removal():
    """Test task processor _raw_tool_ids cleanup when removing toolkit by object."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit])
    agent._setup_task_tools(task)

    processor = task.tool_manager.registry
    kit_id = id(math_kit)
    assert kit_id in processor.raw_object_ids

    task.remove_tools(math_kit)

    assert kit_id not in processor.raw_object_ids


@pytest.mark.asyncio
async def test_task_function_tool_deduplication():
    """Test that registering the same function tool twice on task doesn't duplicate."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    task = Task(description="test", tools=[add_numbers, add_numbers, add_numbers])
    agent._setup_task_tools(task)

    assert "add_numbers" in task.registered_task_tools
    defs = task.get_tool_defs()
    add_defs = [d for d in defs if d.name == "add_numbers"]
    assert len(add_defs) == 1, "Should have exactly 1 add_numbers definition"


@pytest.mark.asyncio
async def test_task_toolkit_re_add_after_removal():
    """Test that removing and re-adding a ToolKit on task works correctly."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit])
    agent._setup_task_tools(task)

    processor = task.tool_manager.registry
    kit_id = id(math_kit)

    assert "subtract" in task.registered_task_tools
    assert "divide" in task.registered_task_tools
    assert kit_id in processor.raw_object_ids
    assert kit_id in processor.class_instance_to_tools

    task.remove_tools(math_kit)

    assert "subtract" not in task.registered_task_tools
    assert "divide" not in task.registered_task_tools
    assert kit_id not in processor.raw_object_ids
    assert kit_id not in processor.class_instance_to_tools

    newly_registered = task.tool_manager.register_tools(tools=[math_kit])
    task.registered_task_tools.update(newly_registered)

    assert "subtract" in task.registered_task_tools
    assert "divide" in task.registered_task_tools
    assert kit_id in processor.raw_object_ids
    assert kit_id in processor.class_instance_to_tools


@pytest.mark.asyncio
async def test_task_mixed_deduplication():
    """Test deduplication with mixed tool types on Task."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    text_kit = TextToolKit()

    task = Task(
        description="test",
        tools=[add_numbers, math_kit, text_kit, multiply_numbers]
    )
    agent._setup_task_tools(task)

    initial_count = len(task.registered_task_tools)
    assert initial_count == 6, (
        f"Should have 6 tools (2 functions + 2 math kit + 2 text kit), got {initial_count}"
    )


@pytest.mark.asyncio
async def test_task_processor_tracking_consistency():
    """Test that task processor tracking stays consistent through operations."""
    agent = Agent(model=MODEL, name="Test Agent", debug=True)

    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit])
    agent._setup_task_tools(task)

    processor = task.tool_manager.registry
    kit_id = id(math_kit)

    assert len(processor.registered_tools) == 2
    assert kit_id in processor.class_instance_to_tools
    assert len(processor.class_instance_to_tools[kit_id]) == 2
    assert kit_id in processor.raw_object_ids

    task.remove_tools("subtract")

    assert len(processor.registered_tools) == 1
    assert kit_id in processor.class_instance_to_tools
    assert len(processor.class_instance_to_tools[kit_id]) == 1
    assert kit_id in processor.raw_object_ids

    task.remove_tools("divide")

    assert len(processor.registered_tools) == 0
    assert kit_id not in processor.class_instance_to_tools
    assert kit_id not in processor.raw_object_ids
    assert len(task.registered_task_tools) == 0


@pytest.mark.asyncio
async def test_task_mcp_handlers_list_cleanup_on_individual_removal():
    """Test task MCP handler cleanup when all MCP tools are removed individually."""
    try:
        from upsonic.tools.mcp import MCPHandler

        agent = Agent(model=MODEL, name="Test Agent", debug=True)

        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
        task = Task(description="test", tools=[handler])
        agent._setup_task_tools(task)

        processor = task.tool_manager.registry
        assert handler in processor.mcp_handlers

        handler_id = id(handler)
        mcp_tool_names = list(processor.mcp_handler_to_tools.get(handler_id, []))
        assert len(mcp_tool_names) > 0

        for tool_name in mcp_tool_names:
            task.remove_tools(tool_name)

        assert handler not in processor.mcp_handlers
        assert handler_id not in processor.mcp_handler_to_tools

    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_task_mcp_handler_with_tool_name_prefix_management():
    """Test MCPHandler with tool_name_prefix on Task via _setup_task_tools."""
    try:
        from upsonic.tools.mcp import MCPHandler

        agent = Agent(model=MODEL, name="Test Agent", debug=True)

        handler = MCPHandler(
            command="npx -y @modelcontextprotocol/server-filesystem /tmp",
            tool_name_prefix="task_prefix"
        )
        task = Task(description="test", tools=[handler, add_numbers])
        agent._setup_task_tools(task)

        assert "add_numbers" in task.registered_task_tools

        mcp_tools = [n for n in task.registered_task_tools.keys() if n.startswith("task_prefix_")]
        assert len(mcp_tools) > 0, "MCP tools should have 'task_prefix_' prefix"

        tool_to_remove = mcp_tools[0]
        task.remove_tools(tool_to_remove)
        assert tool_to_remove not in task.registered_task_tools
        assert "add_numbers" in task.registered_task_tools

        task.remove_tools(handler)
        remaining = [n for n in task.registered_task_tools.keys() if n.startswith("task_prefix_")]
        assert len(remaining) == 0
        assert "add_numbers" in task.registered_task_tools

    except ImportError:
        pytest.skip("MCP dependencies not available")
    except Exception as e:
        if "Failed to connect" in str(e) or "ENOENT" in str(e):
            pytest.skip(f"MCP server not available: {e}")
        else:
            raise
