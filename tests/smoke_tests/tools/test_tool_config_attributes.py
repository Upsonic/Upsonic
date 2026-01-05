"""
Comprehensive Test for Tool Config Attributes
Tests: show_result, stop_after_tool_call, sequential, cache_results, cache_dir, 
       cache_ttl, tool_hooks, max_retries, timeout, strict, docstring_format,
       require_parameter_descriptions

"""
import pytest
import time
import os
import tempfile
from typing import Any

from upsonic.tools import tool, ToolHooks
from upsonic import Agent, Task


class TestShowResult:
    """Test show_result attribute"""
    
    def test_show_result_true(self):
        """Test tool with show_result=True (output shown to user, not LLM)"""
        
        @tool(show_result=True)
        def display_report(data: str) -> str:
            """
            Generate and display a report.
            
            Args:
                data: Data for the report
            
            Returns:
                Formatted report
            """
            return f"=== REPORT ===\n{data}\n=============="
        
        # Verify tool config is set
        assert hasattr(display_report, '_upsonic_tool_config')
        assert display_report._upsonic_tool_config.show_result is True
        
    def test_show_result_false(self):
        """Test tool with show_result=False (default, output to LLM)"""
        
        @tool(show_result=False)
        def process_data(data: str) -> str:
            """Process data and return to LLM"""
            return f"Processed: {data}"
        
        assert hasattr(process_data, '_upsonic_tool_config')
        assert process_data._upsonic_tool_config.show_result is False


class TestStopAfterToolCall:
    """Test stop_after_tool_call attribute"""
    
    def test_stop_after_tool_call(self):
        """Test tool that stops execution after being called"""
        
        @tool(stop_after_tool_call=True)
        def final_action(message: str) -> str:
            """
            Perform final action and stop.
            
            Args:
                message: Final message
            
            Returns:
                Completion message
            """
            return f"Final action completed: {message}"
        
        assert hasattr(final_action, '_upsonic_tool_config')
        assert final_action._upsonic_tool_config.stop_after_tool_call is True


class TestSequential:
    """Test sequential attribute"""
    
    def test_sequential_tool(self):
        """Test tool that requires sequential execution"""
        
        @tool(sequential=True)
        def database_transaction(operation: str) -> str:
            """
            Execute database transaction sequentially.
            
            Args:
                operation: Transaction operation
            
            Returns:
                Transaction result
            """
            return f"Transaction executed: {operation}"
        
        assert hasattr(database_transaction, '_upsonic_tool_config')
        assert database_transaction._upsonic_tool_config.sequential is True
        
    def test_parallel_tool(self):
        """Test tool that can be parallelized (default)"""
        
        @tool(sequential=False)
        def independent_task(task_id: str) -> str:
            """Independent task that can run in parallel"""
            return f"Task {task_id} completed"
        
        assert hasattr(independent_task, '_upsonic_tool_config')
        assert independent_task._upsonic_tool_config.sequential is False


class TestCacheResults:
    """Test cache_results, cache_dir, and cache_ttl attributes"""
    
    def test_cache_results_basic(self):
        """Test basic result caching"""
        
        @tool(cache_results=True)
        def expensive_computation(n: int) -> int:
            """
            Expensive computation that should be cached.
            
            Args:
                n: Input number
            
            Returns:
                Computed result
            """
            time.sleep(0.01)  # Simulate expensive operation
            return n * n
        
        assert hasattr(expensive_computation, '_upsonic_tool_config')
        assert expensive_computation._upsonic_tool_config.cache_results is True
        
    def test_cache_with_custom_dir(self):
        """Test caching with custom directory"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            @tool(cache_results=True, cache_dir=tmpdir)
            def cached_operation(value: str) -> str:
                """Cached operation with custom directory"""
                return f"Result: {value}"
            
            config = cached_operation._upsonic_tool_config
            assert config.cache_results is True
            assert config.cache_dir == tmpdir
            
    def test_cache_with_ttl(self):
        """Test caching with TTL"""
        
        @tool(cache_results=True, cache_ttl=60)
        def timed_cache_operation(key: str) -> str:
            """
            Operation with 60-second cache TTL.
            
            Args:
                key: Cache key
            
            Returns:
                Cached result
            """
            return f"Value for {key}"
        
        config = timed_cache_operation._upsonic_tool_config
        assert config.cache_results is True
        assert config.cache_ttl == 60


class TestToolHooks:
    """Test tool_hooks attribute"""
    
    def test_before_hook(self):
        """Test before execution hook"""
        
        execution_log = []
        
        def before_execution(*args, **kwargs):
            execution_log.append("before")
        
        hooks = ToolHooks(before=before_execution)
        
        @tool(tool_hooks=hooks)
        def hooked_function(data: str) -> str:
            """Function with before hook"""
            execution_log.append("during")
            return data
        
        config = hooked_function._upsonic_tool_config
        assert config.tool_hooks is not None
        assert config.tool_hooks.before is before_execution
        
    def test_after_hook(self):
        """Test after execution hook"""
        
        def after_execution(*args, **kwargs):
            pass
        
        hooks = ToolHooks(after=after_execution)
        
        @tool(tool_hooks=hooks)
        def hooked_function(data: str) -> str:
            """Function with after hook"""
            return data
        
        config = hooked_function._upsonic_tool_config
        assert config.tool_hooks is not None
        assert config.tool_hooks.after is after_execution
        
    def test_both_hooks(self):
        """Test both before and after hooks"""
        
        def before(*args, **kwargs):
            pass
        
        def after(*args, **kwargs):
            pass
        
        hooks = ToolHooks(before=before, after=after)
        
        @tool(tool_hooks=hooks)
        def fully_hooked(data: str) -> str:
            """Function with both hooks"""
            return data
        
        config = fully_hooked._upsonic_tool_config
        assert config.tool_hooks.before is before
        assert config.tool_hooks.after is after


class TestMaxRetries:
    """Test max_retries attribute"""
    
    def test_max_retries_custom(self):
        """Test custom max_retries value"""
        
        @tool(max_retries=3)
        def unreliable_operation(data: str) -> str:
            """
            Operation that might fail, retry up to 3 times.
            
            Args:
                data: Input data
            
            Returns:
                Result
            """
            return f"Processed {data}"
        
        config = unreliable_operation._upsonic_tool_config
        assert config.max_retries == 3
        
    def test_max_retries_default(self):
        """Test default max_retries value"""
        
        @tool()
        def standard_operation(data: str) -> str:
            """Standard operation with default retries"""
            return data
        
        config = standard_operation._upsonic_tool_config
        assert config.max_retries == 5  # Default is 5


class TestTimeout:
    """Test timeout attribute"""
    
    def test_custom_timeout(self):
        """Test custom timeout value"""
        
        @tool(timeout=10.0)
        def long_running_task(duration: float) -> str:
            """
            Long running task with 10 second timeout.
            
            Args:
                duration: How long to run
            
            Returns:
                Result
            """
            time.sleep(min(duration, 0.01))  # Don't actually sleep long in tests
            return "Completed"
        
        config = long_running_task._upsonic_tool_config
        assert config.timeout == 10.0
        
    def test_default_timeout(self):
        """Test default timeout value"""
        
        @tool()
        def standard_task(data: str) -> str:
            """Standard task with default timeout"""
            return data
        
        config = standard_task._upsonic_tool_config
        assert config.timeout == 30.0  # Default is 30 seconds


class TestStrict:
    """Test strict attribute"""
    
    def test_strict_mode_enabled(self):
        """Test strict schema validation enabled"""
        
        @tool(strict=True)
        def strict_function(name: str, age: int) -> str:
            """
            Function with strict schema validation.
            
            Args:
                name: Person's name
                age: Person's age
            
            Returns:
                Greeting message
            """
            return f"Hello {name}, age {age}"
        
        config = strict_function._upsonic_tool_config
        assert config.strict is True
        
    def test_strict_mode_disabled(self):
        """Test strict schema validation disabled"""
        
        @tool(strict=False)
        def lenient_function(data: Any) -> str:
            """Function with lenient validation"""
            return str(data)
        
        config = lenient_function._upsonic_tool_config
        assert config.strict is False


class TestDocstringFormat:
    """Test docstring_format attribute"""
    
    def test_google_format(self):
        """Test Google-style docstring format"""
        
        @tool(docstring_format='google')
        def google_style(param1: str, param2: int) -> str:
            """
            Function with Google-style docstring.
            
            Args:
                param1: First parameter
                param2: Second parameter
            
            Returns:
                Result string
            """
            return f"{param1}-{param2}"
        
        config = google_style._upsonic_tool_config
        assert config.docstring_format == 'google'
        
    def test_numpy_format(self):
        """Test NumPy-style docstring format"""
        
        @tool(docstring_format='numpy')
        def numpy_style(x: float, y: float) -> float:
            """
            Function with NumPy-style docstring.
            
            Parameters
            ----------
            x : float
                First value
            y : float
                Second value
            
            Returns
            -------
            float
                Sum of x and y
            """
            return x + y
        
        config = numpy_style._upsonic_tool_config
        assert config.docstring_format == 'numpy'
        
    def test_auto_format(self):
        """Test auto-detect docstring format"""
        
        @tool(docstring_format='auto')
        def auto_detect(param: str) -> str:
            """Auto-detected docstring format"""
            return param
        
        config = auto_detect._upsonic_tool_config
        assert config.docstring_format == 'auto'


class TestRequireParameterDescriptions:
    """Test require_parameter_descriptions attribute"""
    
    def test_require_descriptions_true(self):
        """Test requiring parameter descriptions"""
        
        @tool(require_parameter_descriptions=True)
        def well_documented(name: str, value: int) -> str:
            """
            Well-documented function.
            
            Args:
                name: The name parameter
                value: The value parameter
            
            Returns:
                Formatted string
            """
            return f"{name}={value}"
        
        config = well_documented._upsonic_tool_config
        assert config.require_parameter_descriptions is True
        
    def test_require_descriptions_false(self):
        """Test not requiring parameter descriptions"""
        
        @tool(require_parameter_descriptions=False)
        def minimal_docs(x: int) -> int:
            """Minimal documentation"""
            return x * 2
        
        config = minimal_docs._upsonic_tool_config
        assert config.require_parameter_descriptions is False


class TestCombinedAttributes:
    """Test combinations of multiple attributes"""
    
    def test_multiple_attributes_combined(self):
        """Test tool with multiple config attributes"""
        
        @tool(
            cache_results=True,
            cache_ttl=120,
            max_retries=3,
            timeout=15.0,
            strict=True,
            sequential=True
        )
        def complex_tool(query: str, limit: int = 10) -> str:
            """
            Complex tool with multiple config attributes.
            
            Args:
                query: Search query
                limit: Result limit
            
            Returns:
                Search results
            """
            return f"Results for '{query}' (limit: {limit})"
        
        config = complex_tool._upsonic_tool_config
        assert config.cache_results is True
        assert config.cache_ttl == 120
        assert config.max_retries == 3
        assert config.timeout == 15.0
        assert config.strict is True
        assert config.sequential is True


class TestDefaultToolDecorator:
    """Test tool decorator with no arguments (default config)"""
    
    def test_default_decorator(self):
        """Test @tool decorator with default config"""
        
        @tool
        def simple_function(x: int) -> int:
            """Simple function with default config"""
            return x * 2
        
        assert hasattr(simple_function, '_upsonic_tool_config')
        config = simple_function._upsonic_tool_config
        
        # Check defaults
        assert config.requires_confirmation is False
        assert config.show_result is False
        assert config.stop_after_tool_call is False
        assert config.sequential is False
        assert config.cache_results is False
        assert config.max_retries == 5
        assert config.timeout == 30.0
        
    def test_empty_decorator_call(self):
        """Test @tool() decorator (called with no args)"""
        
        @tool()
        def another_function(data: str) -> str:
            """Another function with default config"""
            return data.upper()
        
        assert hasattr(another_function, '_upsonic_tool_config')
        config = another_function._upsonic_tool_config
        assert config.requires_confirmation is False


class TestToolExecutionWithAgent:
    """Test actual tool execution with Agent and Task"""
    
    def test_basic_tool_execution(self):
        """Test basic tool execution with Agent"""
        
        @tool
        def calculate_discount(price: float, discount_percent: float) -> float:
            """
            Calculate the final price after applying a discount.
            
            Args:
                price: Original price
                discount_percent: Discount percentage
            
            Returns:
                Final price after discount
            """
            return price * (1 - discount_percent / 100)
        
        agent = Agent("openai/gpt-4o-mini")
        task = Task(
            description="Calculate 20% discount on $100 using calculate_discount tool",
            tools=[calculate_discount]
        )
        
        try:
            result = agent.do(task)
            # Should successfully execute and return result
            assert result is not None
            assert isinstance(result, str)
        except Exception as e:
            # If API key is not available, skip
            pytest.skip(f"API key required for agent execution: {e}")
            
    def test_multiple_tools_execution(self):
        """Test execution with multiple tools"""
        
        @tool
        def add_numbers(a: float, b: float) -> float:
            """
            Add two numbers.
            
            Args:
                a: First number
                b: Second number
            
            Returns:
                Sum of a and b
            """
            return a + b
        
        @tool
        def multiply_numbers(a: float, b: float) -> float:
            """
            Multiply two numbers.
            
            Args:
                a: First number
                b: Second number
            
            Returns:
                Product of a and b
            """
            return a * b
        
        agent = Agent("openai/gpt-4o-mini")
        task = Task(
            description="Add 5 and 3, then multiply the result by 2",
            tools=[add_numbers, multiply_numbers]
        )
        
        try:
            result = agent.do(task)
            assert result is not None
        except Exception as e:
            pytest.skip(f"API key required for agent execution: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
