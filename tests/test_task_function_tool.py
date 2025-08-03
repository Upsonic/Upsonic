from unittest import TestCase
from upsonic import Agent, Task


class CallTracker:
    """
    This class wraps a function and tracks if it was called and with which arguments.
    """
    def __init__(self):
        self.called_with = None
        self.call_count = 0

    def sum(self, a: int, b: int):
        """
        Custom sum function that also logs its call parameters.
        """
        self.called_with = (a, b)
        self.call_count += 1
        return a + b


class AgentToolTestCase(TestCase):
    """Test cases for Agent tool function calls"""
    
    def test_agent_tool_function_call(self):
        """Test that agent correctly calls tool function with proper arguments"""
        # Test parameters
        num_a = 12
        num_b = 51
        expected_result = num_a + num_b

        tracker = CallTracker()
        task = Task(f"What is the sum of {num_a} and {num_b}? Use Tool", tools=[tracker.sum])
        agent = Agent(name="Sum Agent", model="openai/gpt-4o")

        result = agent.do(task)

        # Use unittest assertions instead of plain assert
        self.assertEqual(tracker.call_count, 1, "The tool function was not called exactly once.")
        self.assertEqual(tracker.called_with, (num_a, num_b), f"Function was called with wrong arguments: {tracker.called_with}")
        self.assertIn(str(expected_result), str(result), f"Expected result '{expected_result}' not found in agent output: {result}")
        
        print("✅ Test passed successfully! Agent correctly called the tool function with proper arguments.")


# If you want to run the test directly
if __name__ == '__main__':
    import unittest
    unittest.main()
    
