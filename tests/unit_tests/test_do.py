import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager
from upsonic import Task, Agent
from upsonic.agent.run_result import RunResult
from upsonic.models import ModelResponse, TextPart


class TestDo(unittest.TestCase):
    """Test suite for Task, Agent, and do functionality"""
    
    @patch('upsonic.models.infer_model')
    def test_agent_do_basic(self, mock_infer_model):
        """Test basic functionality of Agent.do with a Task"""
        # Mock the model inference
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        
        # Mock the model request to return a proper ModelResponse
        mock_response = ModelResponse(
            parts=[TextPart(content="I was developed by Upsonic, an AI agent framework designed for building reliable AI applications.")],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        mock_model.request = AsyncMock(return_value=mock_response)
        
        # Create a task
        task = Task("Who developed you?")
        
        # Create an agent
        agent = Agent(name="Coder", model=mock_model)
        
        result = agent.do(task)

        # Check that task has a response
        self.assertNotEqual(task.response, None)
        self.assertNotEqual(task.response, "")
        self.assertIsInstance(task.response, str)

        # Check that result is a string (the actual output)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, None)
        self.assertNotEqual(result, "")

        

