import pytest
from unittest.mock import patch, AsyncMock
from contextlib import asynccontextmanager
from upsonic import Task, Agent
from pydantic import BaseModel

class Names(BaseModel):
    names: list[str]

class TestTaskImageContextHandling:
    
    @patch('upsonic.models.factory.ModelFactory.create')
    @patch('upsonic.agent.agent.PydanticAgent')
    def test_agent_with_multiple_images_returns_combined_names(self, mock_pydantic_agent, mock_factory_create):
        # Mock the factory to return a mock provider that doesn't need API keys
        mock_provider = AsyncMock()
        mock_model = AsyncMock()
        mock_provider._provision.return_value = (mock_model, None)
        mock_factory_create.return_value = mock_provider

        
        mock_agent_instance = AsyncMock()
        
        # Create a mock response object that returns the expected format
        mock_response = AsyncMock()
        mock_response.output = Names(names=["John Smith", "Jane Doe", "Michael Johnson"])
        mock_response.all_messages = lambda: []
        mock_agent_instance.run.return_value = mock_response
        
        # Create a proper async context manager for run_mcp_servers
        @asynccontextmanager
        async def mock_run_mcp_servers():
            yield
        
        mock_agent_instance.run_mcp_servers = mock_run_mcp_servers
        mock_pydantic_agent.return_value = mock_agent_instance
        
        images = ["paper1.png", "paper2.png"]
        
        task = Task(
            "Extract the names in the paper",
            images=images,
            response_format=Names
        )
        
        agent = Agent(name="OCR Agent")
        
        result = agent.print_do(task)
        
        assert isinstance(result, Names)
        assert isinstance(result.names, list)
        assert all(isinstance(name, str) for name in result.names)
