"""
Comprehensive test file for Agent with Tools and Pydantic Structured Output
Demonstrates debug level 1 and 2 with tools and structured output

Usage:
    uv run test_debug_agent_tools_pydantic.py

This will show:
- Agent with tools (DuckDuckGo search)
- Tasks with Pydantic structured output
- Tool calls and results in magnificent tables
- Debug level 1 vs 2 comparison
"""

import asyncio
from upsonic import Agent, Task
from pydantic import BaseModel, Field


# Define a Pydantic model for structured output
class WeatherInfo(BaseModel):
    """Weather information structure."""
    location: str = Field(description="The location name")
    temperature: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition (e.g., sunny, rainy)")
    humidity: float = Field(description="Humidity percentage")


class ProductInfo(BaseModel):
    """Product information structure."""
    name: str = Field(description="Product name")
    price: float = Field(description="Product price")
    description: str = Field(description="Product description")
    in_stock: bool = Field(description="Whether product is in stock")


async def test_agent_with_tools_debug_level_1():
    """Test Agent with tools using debug level 1."""
    print("\n" + "=" * 100)
    print("TEST: Agent with Tools - Debug Level 1")
    print("=" * 100)
    
    # Create a simple test tool
    from upsonic.tools.config import tool
    
    @tool
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Research Agent",
        tools=[calculate_sum],
        debug=True,
        debug_level=1
    )
    
    task = Task(
        "Search for information about Python programming language and provide a brief summary in 2 sentences."
    )
    
    print("\n[INFO] Executing with tools and debug_level=1...\n")
    result = await agent.do_async(task)
    
    print(f"\n✅ Result: {result}")
    print("\n" + "=" * 100)


async def test_agent_with_tools_debug_level_2():
    """Test Agent with tools using debug level 2 (magnificent tables)."""
    print("\n" + "=" * 100)
    print("TEST: Agent with Tools - Debug Level 2 (Magnificent Tables)")
    print("=" * 100)
    
    # Create test tools
    from upsonic.tools.config import tool
    
    @tool
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    @tool
    def calculate_product(x: int, y: int) -> int:
        """Calculate the product of two numbers."""
        return x * y
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Research Agent",
        tools=[calculate_sum, calculate_product],
        debug=True,
        debug_level=2
    )
    
    task = Task(
        "First calculate the sum of 10 and 20, then calculate the product of 5 and 6. Compare the two results."
    )
    
    print("\n[INFO] Executing with tools and debug_level=2 (showing magnificent tables)...\n")
    result = await agent.do_async(task)
    
    print(f"\n✅ Result: {result}")
    print("\n" + "=" * 100)


async def test_agent_pydantic_structured_output_debug_level_1():
    """Test Agent with Pydantic structured output using debug level 1."""
    print("\n" + "=" * 100)
    print("TEST: Agent with Pydantic Structured Output - Debug Level 1")
    print("=" * 100)
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Structured Output Agent",
        debug=True,
        debug_level=1
    )
    
    task = Task(
        "Provide weather information for New York. Temperature is 22.5°C, condition is sunny, humidity is 65%.",
        response_format=WeatherInfo
    )
    
    print("\n[INFO] Executing with Pydantic structured output and debug_level=1...\n")
    result = await agent.do_async(task)
    
    print(f"\n✅ Result Type: {type(result)}")
    print(f"✅ Result: {result}")
    if isinstance(result, WeatherInfo):
        print(f"   - Location: {result.location}")
        print(f"   - Temperature: {result.temperature}°C")
        print(f"   - Condition: {result.condition}")
        print(f"   - Humidity: {result.humidity}%")
    print("\n" + "=" * 100)


async def test_agent_pydantic_structured_output_debug_level_2():
    """Test Agent with Pydantic structured output using debug level 2 (magnificent tables)."""
    print("\n" + "=" * 100)
    print("TEST: Agent with Pydantic Structured Output - Debug Level 2 (Magnificent Tables)")
    print("=" * 100)
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Structured Output Agent",
        debug=True,
        debug_level=2
    )
    
    task = Task(
        "Provide weather information for New York. Temperature is 22.5°C, condition is sunny, humidity is 65%.",
        response_format=WeatherInfo
    )
    
    print("\n[INFO] Executing with Pydantic structured output and debug_level=2 (showing magnificent tables)...\n")
    result = await agent.do_async(task)
    
    print(f"\n✅ Result Type: {type(result)}")
    print(f"✅ Result: {result}")
    if isinstance(result, WeatherInfo):
        print(f"   - Location: {result.location}")
        print(f"   - Temperature: {result.temperature}°C")
        print(f"   - Condition: {result.condition}")
        print(f"   - Humidity: {result.humidity}%")
    print("\n" + "=" * 100)


async def test_agent_tools_and_pydantic_debug_level_2():
    """Test Agent with both tools and Pydantic structured output using debug level 2."""
    print("\n" + "=" * 100)
    print("TEST: Agent with Tools + Pydantic Structured Output - Debug Level 2")
    print("=" * 100)
    
    # Create test tools
    from upsonic.tools.config import tool
    
    @tool
    def get_product_price(product_name: str) -> float:
        """Get the price of a product. Returns a mock price."""
        # Mock prices for demonstration
        prices = {
            "iPhone": 999.99,
            "iPad": 499.99,
            "MacBook": 1299.99
        }
        return prices.get(product_name, 0.0)
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Research & Structure Agent",
        tools=[get_product_price],
        debug=True,
        debug_level=2
    )
    
    task = Task(
        "Get the price of iPhone using the tool, then provide structured information including name (iPhone), the price you got, description (Latest iPhone model), and whether it's in stock (true).",
        response_format=ProductInfo
    )
    
    print("\n[INFO] Executing with tools + Pydantic structured output and debug_level=2...\n")
    result = await agent.do_async(task)
    
    print(f"\n✅ Result Type: {type(result)}")
    print(f"✅ Result: {result}")
    if isinstance(result, ProductInfo):
        print(f"   - Name: {result.name}")
        print(f"   - Price: ${result.price}")
        print(f"   - Description: {result.description}")
        print(f"   - In Stock: {result.in_stock}")
    print("\n" + "=" * 100)


async def test_agent_multiple_tool_calls_debug_level_2():
    """Test Agent with multiple tool calls using debug level 2."""
    print("\n" + "=" * 100)
    print("TEST: Agent with Multiple Tool Calls - Debug Level 2")
    print("=" * 100)
    
    # Create test tools
    from upsonic.tools.config import tool
    
    @tool
    def get_number_info(number: int) -> str:
        """Get information about a number."""
        if number % 2 == 0:
            return f"{number} is an even number"
        else:
            return f"{number} is an odd number"
    
    @tool
    def calculate_square(n: int) -> int:
        """Calculate the square of a number."""
        return n * n
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="Multi-Tool Agent",
        tools=[get_number_info, calculate_square],
        debug=True,
        debug_level=2
    )
    
    task = Task(
        "Get information about the number 10, then get information about the number 15, calculate the square of 5, and summarize all the results."
    )
    
    print("\n[INFO] Executing with multiple tool calls and debug_level=2...\n")
    result = await agent.do_async(task)
    
    print(f"\n✅ Result: {result}")
    print("\n" + "=" * 100)


async def main():
    """Run all comprehensive tests."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE AGENT TESTING WITH TOOLS AND PYDANTIC STRUCTURED OUTPUT")
    print("=" * 100)
    print("\nThis test suite demonstrates:")
    print("  • Agent with tools (calculation and info tools)")
    print("  • Tasks with Pydantic structured output")
    print("  • Tool calls and results in magnificent full-width tables")
    print("  • Debug level 1 vs 2 comparison")
    print("  • Multiple tool calls")
    print("  • Combined tools + Pydantic structured output")
    print("\n" + "=" * 100)
    
    # Test 1: Agent with tools - Debug Level 1
    await test_agent_with_tools_debug_level_1()
    
    # Test 2: Agent with tools - Debug Level 2 (magnificent tables)
    await test_agent_with_tools_debug_level_2()
    
    # Test 3: Agent with Pydantic structured output - Debug Level 1
    await test_agent_pydantic_structured_output_debug_level_1()
    
    # Test 4: Agent with Pydantic structured output - Debug Level 2 (magnificent tables)
    await test_agent_pydantic_structured_output_debug_level_2()
    
    # Test 5: Agent with both tools and Pydantic - Debug Level 2
    await test_agent_tools_and_pydantic_debug_level_2()
    
    # Test 6: Agent with multiple tool calls - Debug Level 2
    await test_agent_multiple_tool_calls_debug_level_2()
    
    print("\n" + "=" * 100)
    print("ALL TESTS COMPLETED!")
    print("=" * 100)
    print("\n💡 Key Observations:")
    print("   - Debug Level 1: Shows standard debug information")
    print("   - Debug Level 2: Shows magnificent full-width tables for:")
    print("     * Tool calls with full parameters and results")
    print("     * Pydantic structured output with field-by-field breakdown")
    print("     * LLM results with comprehensive metadata")
    print("     * All tables use full terminal width for maximum visibility")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
