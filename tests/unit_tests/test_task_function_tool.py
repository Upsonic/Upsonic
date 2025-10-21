from unittest import TestCase
from unittest.mock import patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager
from upsonic import Agent, Task
from upsonic.agent.run_result import RunResult
from upsonic.models import ModelResponse, TextPart
from upsonic.messages import ToolCallPart
from upsonic.tools import tool


class CallTracker:
    """
    This class wraps a function and tracks if it was called and with which arguments.
    """
    def __init__(self):
        self.called_with = None
        self.call_count = 0

    @tool
    def sum(self, a: int, b: int) -> int:
        """
        Custom sum function that also logs its call parameters.
        """
        self.called_with = (a, b)
        self.call_count += 1
        return a + b


class AgentToolTestCase(TestCase):
    """Test cases for Agent tool function calls"""
    
    @patch('upsonic.models.infer_model')
    def test_agent_tool_function_call(self, mock_infer_model):
        """Test that agent correctly calls tool function with proper arguments"""
        # Test parameters
        num_a = 12
        num_b = 51
        expected_result = num_a + num_b

        # Mock the model inference
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model

        tracker = CallTracker()
        
        # Mock the model request to return a proper ModelResponse with tool call
        # First response: tool call
        tool_call_response = ModelResponse(
            parts=[ToolCallPart(
                tool_name="sum",
                args={"a": num_a, "b": num_b},
                tool_call_id="test-tool-call-id"
            )],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="tool_calls"
        )
        
        # Second response: final text after tool execution
        final_response = ModelResponse(
            parts=[TextPart(content=f"The sum of {num_a} and {num_b} is {expected_result}.")],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        
        # Mock the model to return tool call first, then final response
        mock_model.request = AsyncMock(side_effect=[tool_call_response, final_response])

        task = Task(f"What is the sum of {num_a} and {num_b}? Use Tool", tools=[tracker])
        agent = Agent(name="Sum Agent", model=mock_model)

        result = agent.do(task)

        # Check that result is a string (the actual output)
        self.assertIsInstance(result, str)
        
        # Use unittest assertions instead of plain assert
        self.assertEqual(tracker.call_count, 1, "The tool function was not called exactly once.")
        self.assertEqual(tracker.called_with, (num_a, num_b), f"Function was called with wrong arguments: {tracker.called_with}")
        self.assertIn(str(expected_result), str(result), f"Expected result '{expected_result}' not found in agent output: {result}")
        
        # Test passed successfully


# If you want to run the test directly
if __name__ == '__main__':
    import unittest
    unittest.main()
    
