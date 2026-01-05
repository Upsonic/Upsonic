"""
Test 31: Structured input and outputs for function/method tools
Success criteria: Agent calls tools properly without error (we check logs what we print for tools and result output of the Agent)
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout
from pydantic import BaseModel

from upsonic import Agent, Task
from upsonic.tools import tool

pytestmark = pytest.mark.timeout(120)


# Define structured input/output models
class DiscountInput(BaseModel):
    """Input for discount calculation."""
    price: float
    discount_percent: float


class DiscountOutput(BaseModel):
    """Output for discount calculation."""
    original_price: float
    discount_percent: float
    final_price: float
    savings: float


@pytest.mark.asyncio
async def test_structured_tool_input_output():
    """Test tool with structured input and output using Pydantic models."""
    
    @tool
    def calculate_discount(price: float, discount_percent: float) -> float:
        """Calculate the final price after applying a discount."""
        return price * (1 - discount_percent / 100)
    
    agent = Agent(model="openai/gpt-4o", name="Structured Tool Agent", debug=True)
    
    task = Task(
        description="Calculate 20% discount on $100",
        tools=[calculate_discount]
    )
    
    output_buffer = StringIO()
    try:
        with redirect_stdout(output_buffer):
            result = await agent.do_async(task)
        
        output = output_buffer.getvalue()
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        
        # Verify tool was called (check logs)
        assert "calculate_discount" in output.lower() or "discount" in output.lower(), \
            "Should see tool call in logs"
        assert "Tool" in output or "tool" in output.lower(), \
            "Should see tool-related logs"
        assert "Result" in output or "result" in output.lower(), \
            "Should see tool result in logs"
        
        # Verify the calculation is correct (should mention 80 or $80)
        assert "80" in result or "$80" in result or "80.0" in result, \
            "Result should contain the calculated discount amount"
        
        # Verify tool inputs and outputs are properly recorded
        assert hasattr(task, 'tool_calls'), "Task should have tool_calls property"
        tool_calls = task.tool_calls
        assert len(tool_calls) > 0, "Should have recorded tool calls"
        
        # Check first tool call
        tool_call = tool_calls[0]
        assert 'tool_name' in tool_call, "Tool call should have tool_name"
        assert 'params' in tool_call, "Tool call should have params"
        assert 'tool_result' in tool_call, "Tool call should have tool_result"
        
        assert tool_call['tool_name'] == 'calculate_discount', \
            f"Tool name should be 'calculate_discount', got {tool_call['tool_name']}"
        
        # Verify params contains the input values
        params = tool_call['params']
        assert params is not None, "Params should not be None"
        # Params should contain price and discount_percent
        if isinstance(params, dict):
            assert 'price' in params or 'discount_percent' in params, \
                f"Params should contain price and discount_percent, got {params}"
        
        # Verify tool_result is the calculated value
        tool_result = tool_call['tool_result']
        assert tool_result is not None, "Tool result should not be None"
        # Should be 80.0 (100 * (1 - 20/100))
        if isinstance(tool_result, (int, float)):
            assert abs(float(tool_result) - 80.0) < 0.01, \
                f"Tool result should be 80.0, got {tool_result}"
        else:
            assert "80" in str(tool_result), \
                f"Tool result should contain 80, got {tool_result}"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_structured_tool_with_pydantic_input():
    """Test tool that accepts Pydantic model as input."""
    
    @tool
    def process_discount(input_data: DiscountInput) -> DiscountOutput:
        """Process discount calculation with structured input and output."""
        final_price = input_data.price * (1 - input_data.discount_percent / 100)
        savings = input_data.price - final_price
        
        return DiscountOutput(
            original_price=input_data.price,
            discount_percent=input_data.discount_percent,
            final_price=final_price,
            savings=savings
        )
    
    agent = Agent(model="openai/gpt-4o", name="Pydantic Tool Agent", debug=True)
    
    task = Task(
        description="Calculate a 15% discount on a $200 item using the process_discount tool",
        tools=[process_discount]
    )
    
    output_buffer = StringIO()
    try:
        with redirect_stdout(output_buffer):
            result = await agent.do_async(task)
        
        output = output_buffer.getvalue()
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        
        # Verify tool was called or result contains calculation
        tool_called = "process_discount" in output.lower() or "discount" in output.lower()
        has_calculation = "170" in result or "$170" in result or "30" in result or "$30" in result or "15" in result
        
        assert tool_called or has_calculation, \
            f"Should see tool call in logs or calculation in result. Output: {output[:500]}, Result: {result[:200]}"
        
        # Verify tool inputs and outputs are properly recorded
        assert hasattr(task, 'tool_calls'), "Task should have tool_calls property"
        tool_calls = task.tool_calls
        assert len(tool_calls) > 0, "Should have recorded tool calls"
        
        # Check first tool call
        tool_call = tool_calls[0]
        assert 'tool_name' in tool_call, "Tool call should have tool_name"
        assert 'params' in tool_call, "Tool call should have params"
        assert 'tool_result' in tool_call, "Tool call should have tool_result"
        
        # Verify tool name
        assert 'process_discount' in tool_call['tool_name'].lower() or 'discount' in tool_call['tool_name'].lower(), \
            f"Tool name should contain 'process_discount', got {tool_call['tool_name']}"
        
        # Verify params and result exist (even if tool execution failed, they should be recorded)
        assert tool_call['params'] is not None, "Params should not be None"
        assert tool_call['tool_result'] is not None, "Tool result should not be None"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_multiple_structured_tools():
    """Test multiple tools with structured inputs/outputs."""
    
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b
    
    @tool
    def multiply_numbers(a: float, b: float) -> float:
        """Multiplies two numbers."""
        return a * b
    
    @tool
    def calculate_discount(price: float, discount_percent: float) -> float:
        """Calculate the final price after applying a discount."""
        return price * (1 - discount_percent / 100)
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Multi Tool Agent",
        tools=[add_numbers, multiply_numbers, calculate_discount],
        debug=True
    )
    
    task = Task(
        description="First add 10 and 5, then multiply the result by 2, then calculate 10% discount on that result"
    )
    
    output_buffer = StringIO()
    try:
        with redirect_stdout(output_buffer):
            result = await agent.do_async(task)
        
        output = output_buffer.getvalue()
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        
        # Verify multiple tools were called (at least one should be called)
        tool_calls_found = 0
        if "add_numbers" in output.lower() or ("add" in output.lower() and "number" in output.lower()):
            tool_calls_found += 1
        if "multiply_numbers" in output.lower() or ("multiply" in output.lower() and "number" in output.lower()):
            tool_calls_found += 1
        if "calculate_discount" in output.lower() or ("discount" in output.lower() and "calculate" in output.lower()):
            tool_calls_found += 1
        
        # At least one tool should be called (the agent may use different tools or combine operations)
        assert tool_calls_found >= 1 or "Tool" in output or "tool" in output.lower(), \
            f"Should see at least 1 tool call in logs. Found: {tool_calls_found}, Output: {output[:500]}"
        
        # Verify tool execution logs (check for pipeline/step logs)
        assert "Tool" in output or "tool" in output.lower() or "Pipeline" in output or "pipeline" in output.lower(), \
            "Should see tool-related or pipeline logs"
        assert ("Step" in output or "step" in output.lower() or "Pipeline" in output or "pipeline" in output.lower()), \
            "Should see execution logs (Pipeline/Step started/completed)"
        
        # Verify tool inputs and outputs are properly recorded for multiple tools
        assert hasattr(task, 'tool_calls'), "Task should have tool_calls property"
        tool_calls = task.tool_calls
        assert len(tool_calls) > 0, "Should have recorded at least one tool call"
        
        # Check each tool call has proper structure
        for tool_call in tool_calls:
            assert 'tool_name' in tool_call, "Tool call should have tool_name"
            assert 'params' in tool_call, "Tool call should have params"
            assert 'tool_result' in tool_call, "Tool call should have tool_result"
            
            # Verify params and result exist
            assert tool_call['params'] is not None, "Params should not be None"
            assert tool_call['tool_result'] is not None, "Tool result should not be None"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_structured_tool_with_complex_types():
    """Test tool with complex structured types."""
    from typing import List, Dict
    
    @tool
    def process_order(items: List[Dict[str, float]], tax_rate: float) -> Dict[str, float]:
        """Process an order with items and calculate total with tax."""
        subtotal = sum(item.get('price', 0) * item.get('quantity', 0) for item in items)
        tax = subtotal * (tax_rate / 100)
        total = subtotal + tax
        
        return {
            'subtotal': subtotal,
            'tax': tax,
            'total': total
        }
    
    agent = Agent(model="openai/gpt-4o", name="Complex Tool Agent", debug=True)
    
    task = Task(
        description="Process an order with 2 items: item1 price $10 quantity 3, item2 price $5 quantity 2, with 8% tax",
        tools=[process_order]
    )
    
    output_buffer = StringIO()
    try:
        with redirect_stdout(output_buffer):
            result = await agent.do_async(task)
        
        output = output_buffer.getvalue()
        
        # Verify result
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        
        # Verify tool was called
        assert "process_order" in output.lower() or "order" in output.lower(), \
            "Should see tool call in logs"
        
        # Verify calculation results are mentioned
        assert "40" in result or "$40" in result or "subtotal" in result.lower(), \
            "Result should contain order calculation"
        
        # Verify tool inputs and outputs are properly recorded
        assert hasattr(task, 'tool_calls'), "Task should have tool_calls property"
        tool_calls = task.tool_calls
        assert len(tool_calls) > 0, "Should have recorded tool calls"
        
        # Check first tool call
        tool_call = tool_calls[0]
        assert 'tool_name' in tool_call, "Tool call should have tool_name"
        assert 'params' in tool_call, "Tool call should have params"
        assert 'tool_result' in tool_call, "Tool call should have tool_result"
        
        # Verify tool name
        assert 'process_order' in tool_call['tool_name'].lower() or 'order' in tool_call['tool_name'].lower(), \
            f"Tool name should contain 'process_order', got {tool_call['tool_name']}"
        
        # Verify params contains list of items and tax_rate
        params = tool_call['params']
        assert params is not None, "Params should not be None"
        if isinstance(params, dict):
            assert 'items' in params or 'tax_rate' in params, \
                f"Params should contain items and tax_rate, got {params}"
        
        # Verify tool_result contains calculation results
        tool_result = tool_call['tool_result']
        assert tool_result is not None, "Tool result should not be None"
        # Result should be a dict with subtotal, tax, total
        if isinstance(tool_result, dict):
            assert 'subtotal' in tool_result or 'total' in tool_result, \
                f"Tool result should contain subtotal/total, got {tool_result}"
        
    finally:
        pass  # Agent cleanup handled automatically

