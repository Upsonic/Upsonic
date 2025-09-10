from unittest import TestCase
from unittest.mock import patch, AsyncMock
from contextlib import asynccontextmanager
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
    
    @patch('upsonic.models.factory.ModelFactory.create')
    @patch('upsonic.agent.agent.PydanticAgent')
    def test_agent_tool_function_call(self, mock_pydantic_agent, mock_factory_create):
        """Test that agent correctly calls tool function with proper arguments"""
        # Test parameters
        num_a = 12
        num_b = 51
        expected_result = num_a + num_b

        # Mock setup
        mock_provider = AsyncMock()
        mock_model = AsyncMock()
        mock_provider._provision.return_value = (mock_model, None)
        mock_factory_create.return_value = mock_provider

        tracker = CallTracker()
        
        # Mock the agent to call the tool and return expected result
        mock_agent_instance = AsyncMock()
        mock_response = AsyncMock()
        
        # Simulate the agent calling the tool function and getting the result
        def mock_agent_run(*args, **kwargs):
            # Call the actual tool function to test the integration
            result_value = tracker.sum(num_a, num_b)
            mock_response.output = f"The sum of {num_a} and {num_b} is {result_value}."
            return mock_response
        
        mock_response.all_messages = lambda: []
        mock_agent_instance.run.side_effect = mock_agent_run
        
        @asynccontextmanager
        async def mock_run_mcp_servers():
            yield
        mock_agent_instance.run_mcp_servers = mock_run_mcp_servers
        mock_pydantic_agent.return_value = mock_agent_instance

        task = Task(f"What is the sum of {num_a} and {num_b}? Use Tool", tools=[tracker.sum])
        agent = Agent(name="Sum Agent", model="openai/gpt-4o")

        result = agent.do(task)

        # Use unittest assertions instead of plain assert
        self.assertEqual(tracker.call_count, 1, "The tool function was not called exactly once.")
        self.assertEqual(tracker.called_with, (num_a, num_b), f"Function was called with wrong arguments: {tracker.called_with}")
        self.assertIn(str(expected_result), str(result), f"Expected result '{expected_result}' not found in agent output: {result}")
        
        # Test passed successfully


# If you want to run the test directly
if __name__ == '__main__':
    import unittest
    unittest.main()
    
