import pytest
from unittest.mock import patch, AsyncMock
from contextlib import asynccontextmanager
from upsonic import Task, Agent
from upsonic.agent.run_result import RunResult
from upsonic.models import ModelResponse, TextPart
from pydantic import BaseModel

class Names(BaseModel):
    names: list[str]

class TestTaskImageContextHandling:
    
    @patch('upsonic.models.infer_model')
    def test_agent_with_multiple_images_returns_combined_names(self, mock_infer_model):
        # Mock the model inference
        mock_model = AsyncMock()
        mock_infer_model.return_value = mock_model
        
        # Mock the model request to return a proper ModelResponse with structured output
        expected_names = Names(names=["John Smith", "Jane Doe", "Michael Johnson"])
        mock_response = ModelResponse(
            parts=[TextPart(content=expected_names.model_dump_json())],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        mock_model.request.return_value = mock_response
        
        images = ["paper1.png", "paper2.png"]
        
        task = Task(
            "Extract the names in the paper",
            images=images,
            response_format=Names
        )
        
        agent = Agent(name="OCR Agent", model=mock_model)
        
        result = agent.print_do(task)
        
        # Check that result is a RunResult with the expected output
        assert isinstance(result, RunResult)
        assert isinstance(result.output, Names)
        assert isinstance(result.output.names, list)
        assert all(isinstance(name, str) for name in result.output.names)
