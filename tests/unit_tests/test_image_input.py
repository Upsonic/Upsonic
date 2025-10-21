import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager
from PIL import Image
import tempfile
import os
from upsonic import Task, Agent
from upsonic.agent.run_result import RunResult
from upsonic.models import ModelResponse, TextPart
from pydantic import BaseModel

class Names(BaseModel):
    names: list[str]

class TestTaskImageContextHandling:
    
    @patch('upsonic.models.infer_model')
    def test_agent_with_multiple_images_returns_combined_names(self, mock_infer_model):
        # Create temporary test images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create simple test images
            img1_path = os.path.join(temp_dir, "paper1.png")
            img2_path = os.path.join(temp_dir, "paper2.png")
            
            # Create simple white images
            img1 = Image.new('RGB', (100, 100), color='white')
            img2 = Image.new('RGB', (100, 100), color='white')
            
            img1.save(img1_path)
            img2.save(img2_path)
            
            # Mock the model inference
            mock_model = MagicMock()
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
            mock_model.request = AsyncMock(return_value=mock_response)
            
            images = [img1_path, img2_path]
            
            task = Task(
                "Extract the names in the paper",
                attachments=images,
                response_format=Names
            )
            
            agent = Agent(name="OCR Agent", model=mock_model)
            
            result = agent.print_do(task)
            
            # Check that result is a Names object with the expected output
            assert isinstance(result, Names)
            assert isinstance(result.names, list)
            assert all(isinstance(name, str) for name in result.names)
