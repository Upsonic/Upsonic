import unittest
from unittest.mock import patch, AsyncMock
from contextlib import asynccontextmanager
from upsonic import Task, Agent


class TestDo(unittest.TestCase):
    """Test suite for Task, Agent, and do functionality"""
    
    @patch('upsonic.models.factory.ModelFactory.create')
    @patch('upsonic.agent.agent.PydanticAgent')
    def test_agent_print_do_basic(self, mock_pydantic_agent, mock_factory_create):
        """Test basic functionality of Agent.print_do with a Task"""
        # Mock the factory to return a mock provider that doesn't need API keys
        mock_provider = AsyncMock()
        mock_model = AsyncMock()
        mock_provider._provision.return_value = (mock_model, None)
        mock_factory_create.return_value = mock_provider

        
        mock_agent_instance = AsyncMock()
        
        # Create a mock response object with the expected attributes
        mock_response = AsyncMock()
        mock_response.output = "I was developed by Upsonic, an AI agent framework designed for building reliable AI applications."
        mock_response.all_messages = lambda: []
        mock_agent_instance.run.return_value = mock_response
        
        # Create a proper async context manager for run_mcp_servers
        @asynccontextmanager
        async def mock_run_mcp_servers():
            yield
        
        mock_agent_instance.run_mcp_servers = mock_run_mcp_servers
        mock_pydantic_agent.return_value = mock_agent_instance
        
        # Create a task
        task = Task("Who developed you?")
        
        # Create an agent
        agent = Agent(name="Coder")
        
        result = agent.do(task)

        self.assertNotEqual(task.response, None)
        self.assertNotEqual(task.response, "")
        self.assertIsInstance(task.response, str)

        self.assertNotEqual(result, None)
        self.assertNotEqual(result, "")
        self.assertIsInstance(result, str)

        

