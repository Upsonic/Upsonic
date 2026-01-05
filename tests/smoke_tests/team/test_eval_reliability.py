"""
Test 26b: ReliabilityEvaluator testing for Task and Team
Success criteria: We check the attributes, what we log and result
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout

from upsonic import Agent, Task, Team
from upsonic.eval import ReliabilityEvaluator
from upsonic.tools import tool

pytestmark = pytest.mark.timeout(240)


@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def calculate_product(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"


@pytest.mark.asyncio
async def test_reliability_evaluator_basic():
    """Test basic ReliabilityEvaluator - verify all attributes and tool call checking."""
    
    # Create agent with tools
    agent = Agent(
        model="openai/gpt-4o",
        tools=[calculate_sum, calculate_product],
        debug=True
    )
    
    # Create task that should use both tools
    task = Task(
        description="First calculate 5 + 3 using calculate_sum, then multiply the result by 2 using calculate_product"
    )
    
    # Execute task
    await agent.do_async(task)
    
    # Create evaluator
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["calculate_sum", "calculate_product"],
        order_matters=False,
        exact_match=False
    )
    
    # Verify evaluator attributes
    assert evaluator.expected_tool_calls == ["calculate_sum", "calculate_product"], \
        "Expected tool calls should be set"
    assert evaluator.order_matters is False, "order_matters should be False"
    assert evaluator.exact_match is False, "exact_match should be False"
    
    # Run evaluation
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = evaluator.run(task, print_results=True)
    
    output = output_buffer.getvalue()
    
    # Verify result attributes
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'passed'), "Result should have passed attribute"
    assert hasattr(result, 'summary'), "Result should have summary attribute"
    assert hasattr(result, 'expected_tool_calls'), "Result should have expected_tool_calls"
    assert hasattr(result, 'actual_tool_calls'), "Result should have actual_tool_calls"
    assert hasattr(result, 'checks'), "Result should have checks"
    assert hasattr(result, 'missing_tool_calls'), "Result should have missing_tool_calls"
    assert hasattr(result, 'unexpected_tool_calls'), "Result should have unexpected_tool_calls"
    
    # Verify result values
    assert isinstance(result.passed, bool), "passed should be boolean"
    assert result.passed is True, "Should pass since both tools were called"
    assert isinstance(result.summary, str), "summary should be string"
    assert len(result.summary) > 0, "summary should not be empty"
    assert "passed" in result.summary.lower(), "Summary should mention passed"
    
    # Verify tool call lists
    assert isinstance(result.expected_tool_calls, list), "expected_tool_calls should be list"
    assert result.expected_tool_calls == ["calculate_sum", "calculate_product"], \
        "expected_tool_calls should match input"
    
    assert isinstance(result.actual_tool_calls, list), "actual_tool_calls should be list"
    assert "calculate_sum" in result.actual_tool_calls, "Should have called calculate_sum"
    assert "calculate_product" in result.actual_tool_calls, "Should have called calculate_product"
    
    # Verify checks
    assert isinstance(result.checks, list), "checks should be list"
    assert len(result.checks) == 2, "Should have 2 checks"
    
    for check in result.checks:
        assert hasattr(check, 'tool_name'), "Check should have tool_name"
        assert hasattr(check, 'was_called'), "Check should have was_called"
        assert hasattr(check, 'times_called'), "Check should have times_called"
        
        assert isinstance(check.tool_name, str), "tool_name should be string"
        assert isinstance(check.was_called, bool), "was_called should be boolean"
        assert isinstance(check.times_called, int), "times_called should be int"
        
        assert check.was_called is True, f"Tool {check.tool_name} should have been called"
        assert check.times_called > 0, f"Tool {check.tool_name} should have been called at least once"
    
    # Verify missing and unexpected lists
    assert isinstance(result.missing_tool_calls, list), "missing_tool_calls should be list"
    assert len(result.missing_tool_calls) == 0, "Should have no missing tool calls"
    
    assert isinstance(result.unexpected_tool_calls, list), "unexpected_tool_calls should be list"
    
    # Verify logging
    assert len(output) > 0, "Should have logging output"
    assert "Reliability" in output or "passed" in output.lower(), "Should log reliability check"


@pytest.mark.asyncio
async def test_reliability_evaluator_order_matters():
    """Test ReliabilityEvaluator with order_matters=True."""
    
    agent = Agent(
        model="openai/gpt-4o",
        tools=[calculate_sum, calculate_product],
        debug=True
    )
    
    task = Task(
        description="First use calculate_sum to add 2 + 3, then use calculate_product to multiply 4 * 5"
    )
    
    await agent.do_async(task)
    
    # Create evaluator with order_matters=True
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["calculate_sum", "calculate_product"],
        order_matters=True,
        exact_match=False
    )
    
    # Verify attributes
    assert evaluator.order_matters is True, "order_matters should be True"
    
    # Run evaluation
    result = evaluator.run(task, print_results=False)
    
    # Verify result
    assert result is not None, "Result should not be None"
    assert isinstance(result.passed, bool), "passed should be boolean"
    
    # If tools were called in correct order, should pass
    # The actual pass/fail depends on execution, but structure should be correct
    assert isinstance(result.summary, str), "summary should be string"


@pytest.mark.asyncio
async def test_reliability_evaluator_exact_match():
    """Test ReliabilityEvaluator with exact_match=True."""
    
    agent = Agent(
        model="openai/gpt-4o",
        tools=[calculate_sum, calculate_product, get_weather],
        debug=True
    )
    
    task = Task(
        description="Use calculate_sum to add 10 + 20"
    )
    
    await agent.do_async(task)
    
    # Create evaluator with exact_match=True (only calculate_sum should be called)
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["calculate_sum"],
        order_matters=False,
        exact_match=True
    )
    
    # Verify attributes
    assert evaluator.exact_match is True, "exact_match should be True"
    
    # Run evaluation
    result = evaluator.run(task, print_results=False)
    
    # Verify result attributes
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'unexpected_tool_calls'), "Should have unexpected_tool_calls"
    assert isinstance(result.unexpected_tool_calls, list), "unexpected_tool_calls should be list"
    
    # If only calculate_sum was called, should pass
    # If other tools were called, they should be in unexpected_tool_calls
    if not result.passed:
        assert len(result.unexpected_tool_calls) > 0, \
            "If failed, should have unexpected tool calls"


@pytest.mark.asyncio
async def test_reliability_evaluator_missing_tools():
    """Test ReliabilityEvaluator when expected tools are not called."""
    
    agent = Agent(
        model="openai/gpt-4o",
        tools=[calculate_sum],
        debug=True
    )
    
    task = Task(
        description="What is 5 + 5? Just tell me the answer."
    )
    
    await agent.do_async(task)
    
    # Expect calculate_sum to be called, but it might not be if LLM answers directly
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["calculate_sum"],
        order_matters=False,
        exact_match=False
    )
    
    result = evaluator.run(task, print_results=False)
    
    # Verify result structure even if it fails
    assert result is not None, "Result should not be None"
    assert isinstance(result.passed, bool), "passed should be boolean"
    assert isinstance(result.missing_tool_calls, list), "missing_tool_calls should be list"
    
    # If tool wasn't called, should be in missing list
    if not result.passed:
        assert "calculate_sum" in result.missing_tool_calls, \
            "calculate_sum should be in missing tools if not called"
        assert "Missing expected tool calls" in result.summary, \
            "Summary should mention missing tools"


@pytest.mark.asyncio
async def test_reliability_evaluator_multiple_calls():
    """Test ReliabilityEvaluator with tools called multiple times."""
    
    agent = Agent(
        model="openai/gpt-4o",
        tools=[calculate_sum],
        debug=True
    )
    
    task = Task(
        description="Calculate: (1 + 2) + (3 + 4) + (5 + 6) using calculate_sum"
    )
    
    await agent.do_async(task)
    
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["calculate_sum"],
        order_matters=False,
        exact_match=False
    )
    
    result = evaluator.run(task, print_results=False)
    
    # Verify times_called is tracked
    assert result is not None, "Result should not be None"
    assert len(result.checks) > 0, "Should have checks"
    
    sum_check = next((c for c in result.checks if c.tool_name == "calculate_sum"), None)
    assert sum_check is not None, "Should have check for calculate_sum"
    assert sum_check.times_called >= 1, "calculate_sum should be called at least once"


def test_reliability_evaluator_validation():
    """Test ReliabilityEvaluator parameter validation."""
    
    # Test valid initialization
    evaluator = ReliabilityEvaluator(expected_tool_calls=["tool1", "tool2"])
    assert evaluator.expected_tool_calls == ["tool1", "tool2"]
    
    # Test invalid expected_tool_calls (not a list)
    with pytest.raises(TypeError, match="must be a list"):
        ReliabilityEvaluator(expected_tool_calls="not_a_list")
    
    # Test invalid expected_tool_calls (not all strings)
    with pytest.raises(TypeError, match="must be a list"):
        ReliabilityEvaluator(expected_tool_calls=["tool1", 123])
    
    # Test empty expected_tool_calls
    with pytest.raises(ValueError, match="cannot be an empty list"):
        ReliabilityEvaluator(expected_tool_calls=[])


def test_reliability_evaluator_defaults():
    """Test ReliabilityEvaluator default values."""
    
    evaluator = ReliabilityEvaluator(expected_tool_calls=["tool1"])
    
    # Verify defaults
    assert evaluator.order_matters is False, "Default order_matters should be False"
    assert evaluator.exact_match is False, "Default exact_match should be False"


def test_reliability_evaluator_attribute_modification():
    """Test that ReliabilityEvaluator attributes can be checked."""
    
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["tool1", "tool2"],
        order_matters=True,
        exact_match=True
    )
    
    # Verify initial values
    assert evaluator.expected_tool_calls == ["tool1", "tool2"]
    assert evaluator.order_matters is True
    assert evaluator.exact_match is True
    
    # Note: These are typically set at initialization and not modified,
    # but we can verify they're accessible


@pytest.mark.asyncio
async def test_reliability_evaluator_result_assertion():
    """Test ReliabilityEvaluationResult.assert_passed() method."""
    
    agent = Agent(
        model="openai/gpt-4o",
        tools=[calculate_sum],
        debug=True
    )
    
    task = Task(
        description="Calculate 7 + 8 using calculate_sum"
    )
    
    await agent.do_async(task)
    
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["calculate_sum"],
        order_matters=False,
        exact_match=False
    )
    
    result = evaluator.run(task, print_results=False)
    
    # If passed, assert_passed should not raise
    if result.passed:
        try:
            result.assert_passed()  # Should not raise
        except AssertionError:
            pytest.fail("assert_passed() should not raise when passed=True")
    else:
        # If failed, assert_passed should raise
        with pytest.raises(AssertionError, match="Reliability evaluation failed"):
            result.assert_passed()


@pytest.mark.asyncio
async def test_reliability_evaluator_logging_output(capsys):
    """Test that ReliabilityEvaluator properly logs results."""
    
    agent = Agent(
        model="openai/gpt-4o",
        tools=[calculate_sum, calculate_product],
        debug=True
    )
    
    task = Task(
        description="Add 3 + 4 using calculate_sum, then multiply by 2 using calculate_product"
    )
    
    await agent.do_async(task)
    
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["calculate_sum", "calculate_product"],
        order_matters=False,
        exact_match=False
    )
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = evaluator.run(task, print_results=True)
    
    output = output_buffer.getvalue()
    
    # Verify logging output contains key information
    assert len(output) > 0, "Should have logging output"
    
    # Should show reliability check status
    if result.passed:
        assert "✅" in output or "Passed" in output or "passed" in output, \
            "Should show passed status"
    else:
        assert "❌" in output or "Failed" in output or "failed" in output, \
            "Should show failed status"
    
    # Should show tool names
    assert "calculate_sum" in output or output.count("sum") > 0, \
        "Should mention expected tools"


@pytest.mark.asyncio
async def test_reliability_evaluator_with_team():
    """Test ReliabilityEvaluator with Team - verify it works with List[Task]."""
    
    # Create team agents with tools
    calculator_agent = Agent(
        model="openai/gpt-4o",
        name="Calculator",
        role="Math Calculator",
        tools=[calculate_sum, calculate_product],
        debug=True
    )
    
    weather_agent = Agent(
        model="openai/gpt-4o",
        name="WeatherAgent",
        role="Weather Provider",
        tools=[get_weather],
        debug=True
    )
    
    # Create team
    team = Team(
        agents=[calculator_agent, weather_agent],
        mode="sequential"
    )
    
    # Create tasks
    tasks = [
        Task(description="Calculate 5 + 7 using calculate_sum"),
        Task(description="Get weather for San Francisco using get_weather")
    ]
    
    # Execute team asynchronously
    await team.multi_agent_async(team.agents, tasks)
    
    # Create evaluator expecting both tools to be called across all tasks
    evaluator = ReliabilityEvaluator(
        expected_tool_calls=["calculate_sum", "get_weather"],
        order_matters=False,
        exact_match=False
    )
    
    # Run evaluation on the tasks list
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = evaluator.run(tasks, print_results=True)
    
    output = output_buffer.getvalue()
    
    # Verify result attributes
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'passed'), "Result should have passed"
    assert hasattr(result, 'checks'), "Result should have checks"
    assert isinstance(result.checks, list), "checks should be list"
    
    # Should have checks for both tools
    tool_names = [check.tool_name for check in result.checks]
    assert "calculate_sum" in tool_names, "Should check calculate_sum"
    assert "get_weather" in tool_names, "Should check get_weather"
    
    # Verify actual_tool_calls contains tools from all tasks
    assert isinstance(result.actual_tool_calls, list), "actual_tool_calls should be list"
    
    # If both tools were called, should pass
    if result.passed:
        assert len(result.missing_tool_calls) == 0, "Should have no missing tools if passed"
    
    # Verify logging
    assert len(output) > 0, "Should have logging output"

