import pytest
import io
import sys
from upsonic import Agent, Task
from upsonic.tools.tool import tool
from upsonic.tools.tool import ToolHooks


class TestHookExecution:
    """Test class for hook execution order and parameters."""
    
    @pytest.fixture
    def execution_tracker(self):
        """Fixture to track execution order and captured parameters."""
        return {
            'execution_order': [],
            'captured_params': {}
        }
    
    @pytest.fixture
    def tracked_hooks(self, execution_tracker):
        """Fixture to create hooks that track their execution."""
        def before_hook(a, b, c):
            execution_tracker['execution_order'].append("before_hook")
            execution_tracker['captured_params']['before'] = {
                'a': a, 
                'b': b, 
                'c': c
            }
            print(f"before_hook called, a={a}, b={b}, c={c}")
            
        def after_hook(result):
            execution_tracker['execution_order'].append("after_hook")
            # Get c from the before_hook parameters for our calculation
            c = execution_tracker['captured_params']['before']['c']
            final_result = result - c
            execution_tracker['captured_params']['after'] = {
                'original_result': result,
                'c': c,
                'final_result': final_result
            }
            print(f"after_hook called, result={result}, c={c}, final_result={final_result}")
            return final_result
            
        return before_hook, after_hook
    
    @pytest.fixture
    def tracked_function(self, tracked_hooks, execution_tracker):
        """Fixture to create the main function with tracking."""
        before_hook, after_hook = tracked_hooks
        
        @tool(tool_hooks=ToolHooks(before=before_hook, after=after_hook))
        def my_sum_function(a: int, b: int, c: int) -> int:
            """
            This function takes three integers and returns the sum of first two.
            Args:
                a: The first integer.
                b: The second integer.
                c: The third integer (used in after_hook for subtraction).
            Returns:
                The sum of a and b.
            """
            execution_tracker['execution_order'].append("main_function")
            result = a + b
            print(f"my_sum_function called with a={a}, b={b}, c={c}, result={result}")
            return result
            
        return my_sum_function
    
    def test_hook_execution_order(self, tracked_function, execution_tracker):
        """Test that hooks are called in the correct order: before -> main -> after."""
        # Create task and agent
        task = Task("What is the sum of 5 and 3, then subtract 2? Use Tool", tools=[tracked_function])
        agent = Agent(name="Sum Agent", model="openai/gpt-4o")
        
        # Execute the task
        result = agent.do(task)
        
        # Assert execution order
        expected_order = ["before_hook", "main_function", "after_hook"]
        assert execution_tracker['execution_order'] == expected_order, \
            f"Expected order: {expected_order}, got: {execution_tracker['execution_order']}"
        
    def test_hook_parameters_structure(self, tracked_function, execution_tracker):
        """Test that hooks receive parameters with correct structure and types."""
        # Create task and agent
        task = Task("Calculate 8 plus 4, then subtract 3 using the tool", tools=[tracked_function])
        agent = Agent(name="Sum Agent", model="openai/gpt-4o")
        
        # Execute the task
        result = agent.do(task)
        
        # Check that before_hook received correct parameter structure
        assert 'before' in execution_tracker['captured_params']
        before_params = execution_tracker['captured_params']['before']
        
        # Verify parameter structure
        assert 'a' in before_params
        assert 'b' in before_params
        assert 'c' in before_params
        
        # Verify parameter types
        assert isinstance(before_params['a'], int)
        assert isinstance(before_params['b'], int)
        assert isinstance(before_params['c'], int)
        
        # Check that after_hook received the result and performed subtraction
        assert 'after' in execution_tracker['captured_params']
        after_params = execution_tracker['captured_params']['after']
        assert 'original_result' in after_params
        assert 'c' in after_params
        assert 'final_result' in after_params
        assert isinstance(after_params['original_result'], int)
        assert isinstance(after_params['final_result'], int)
        
    def test_hook_parameters_values(self, tracked_function, execution_tracker):
        """Test that hooks receive reasonable parameter values and calculations are correct."""
        # Create task and agent
        task = Task("Calculate the sum of 10 and 15, then subtract 5 using the tool", tools=[tracked_function])
        agent = Agent(name="Sum Agent", model="openai/gpt-4o")
        
        # Execute the task
        result = agent.do(task)
        
        # Get the parameters
        before_params = execution_tracker['captured_params']['before']
        after_params = execution_tracker['captured_params']['after']
        
        # The original result should be the sum of a and b
        expected_sum = before_params['a'] + before_params['b']
        assert after_params['original_result'] == expected_sum, \
            f"Expected sum {expected_sum}, got {after_params['original_result']}"
        
        # The final result should be the sum minus c
        expected_final = expected_sum - before_params['c']
        assert after_params['final_result'] == expected_final, \
            f"Expected final result {expected_final}, got {after_params['final_result']}"
        
        # Parameters should be positive integers (based on our prompt)
        assert before_params['a'] > 0
        assert before_params['b'] > 0
        assert before_params['c'] > 0
        
    def test_console_output_order(self, tracked_function, execution_tracker):
        """Test that console output appears in the correct order."""
        # Store original stdout
        original_stdout = sys.stdout
        captured_output = io.StringIO()
        
        try:
            # Redirect stdout manually
            sys.stdout = captured_output
            
            # Create task and agent
            task = Task("Calculate 7 plus 4, then subtract 2 using the tool", tools=[tracked_function])
            agent = Agent(name="Sum Agent", model="openai/gpt-4o")
            
            # Execute the task
            result = agent.do(task)
            
            # Get the output
            output = captured_output.getvalue()
            
            # Find positions of each output
            before_pos = output.find("before_hook called")
            main_pos = output.find("my_sum_function called")
            after_pos = output.find("after_hook called")
            
            # Assert that all outputs were found
            assert before_pos != -1, "before_hook output not found"
            assert main_pos != -1, "main function output not found"  
            assert after_pos != -1, "after_hook output not found"
            
            # Assert correct order
            assert before_pos < main_pos, "before_hook should print before main function"
            assert main_pos < after_pos, "main function should print before after_hook"
            
        finally:
            # Always restore stdout
            sys.stdout = original_stdout
        
    @pytest.mark.parametrize("test_case", [
        {"task": "Sum 6 and 4, subtract 3", "min_a": 3, "min_b": 2, "min_c": 1},
        {"task": "Add 20 and 15, subtract 10", "min_a": 10, "min_b": 10, "min_c": 5},
        {"task": "Calculate 50 plus 30, subtract 20", "min_a": 30, "min_b": 20, "min_c": 10},
    ])
    def test_multiple_scenarios(self, tracked_function, execution_tracker, test_case):
        """Test hooks work correctly across different scenarios with three integers."""
        # Create task and agent
        task = Task(f"{test_case['task']} using the tool", tools=[tracked_function])
        agent = Agent(name="Sum Agent", model="openai/gpt-4o")
        
        # Execute the task
        result = agent.do(task)
        
        # Verify execution order
        expected_order = ["before_hook", "main_function", "after_hook"]
        assert execution_tracker['execution_order'] == expected_order
        
        # Verify parameters are reasonable
        before_params = execution_tracker['captured_params']['before']
        after_params = execution_tracker['captured_params']['after']
        
        # Check that the values make sense for the test case
        assert before_params['a'] >= test_case['min_a'] or before_params['b'] >= test_case['min_b']
        assert before_params['c'] >= test_case['min_c']
        
        # Verify the calculation: (a + b) - c
        expected_sum = before_params['a'] + before_params['b']
        expected_final = expected_sum - before_params['c']
        assert after_params['original_result'] == expected_sum
        assert after_params['final_result'] == expected_final




if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])