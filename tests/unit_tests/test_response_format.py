import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager
from upsonic import Task, Agent
from upsonic.agent.run_result import RunResult
from upsonic.models import ModelResponse, TextPart
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union


class TravelResponse(BaseModel):
    cities: list[str]


class UserProfile(BaseModel):
    name: str
    age: int
    is_active: bool
    email: Optional[str] = None
    preferences: Dict[str, Any]


class Product(BaseModel):
    id: int
    name: str
    price: float
    in_stock: bool
    tags: list[str]
    metadata: Optional[Dict[str, str]] = None


class MixedTypes(BaseModel):
    string_field: str
    int_field: int
    float_field: float
    bool_field: bool
    list_field: list[Union[str, int]]
    dict_field: Dict[str, Union[str, int, bool]]
    optional_field: Optional[float] = None


class TestTaskResponseFormat:
    """Test suite for Task response_format parameter behavior."""

    @patch('upsonic.models.infer_model')
    def test_task_response_format_behavior(self, mock_infer_model):
        """
        Test response_format parameter behavior:
        1. Without response_format: returns str
        2. With BaseModel response_format: returns BaseModel instance
        3. task.response always matches agent.print_do(task) result
        """
        # Mock the model inference
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        
        # Case 1 Without response_format -> return str
        mock_response_1 = ModelResponse(
            parts=[TextPart(content="I was developed by Upsonic, an AI agent framework.")],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        mock_model.request = AsyncMock(return_value=mock_response_1)
        
        task_no_format = Task("Who developed you?")
        agent = Agent(name="Coder", model=mock_model)
        
        result_no_format = agent.print_do(task_no_format)
        
        # Type check
        assert isinstance(result_no_format, str)
        assert isinstance(task_no_format.response, str) 
        
        # Does results match task.response?
        assert result_no_format == task_no_format.response  
        
        # Case 2 With BaseModel response_format -> return BaseModel instance
        expected_travel = TravelResponse(cities=["Toronto", "Vancouver", "Montreal"])
        mock_response_2 = ModelResponse(
            parts=[TextPart(content=expected_travel.model_dump_json())],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        mock_model.request = AsyncMock(return_value=mock_response_2)
        
        task_with_format = Task(
            "Create a plan to visit cities in Canada", 
            response_format=TravelResponse
        )
        
        result_with_format = agent.print_do(task_with_format)
        
        # Type check
        assert isinstance(result_with_format, TravelResponse)
        assert isinstance(task_with_format.response, TravelResponse)  
        
        # Field structure correctness
        assert isinstance(result_with_format.cities, list)  
        assert all(isinstance(city, str) for city in result_with_format.cities)  
        
        # Does result match task.response?
        assert result_with_format.cities == task_with_format.response.cities  

    @patch('upsonic.models.infer_model')
    def test_diverse_pydantic_types(self, mock_infer_model):
        """
        Test various Pydantic field types to ensure the system handles different data structures correctly.
        """
        # Mock the model inference
        mock_model = MagicMock()
        mock_infer_model.return_value = mock_model
        
        agent = Agent(name="Tester", model=mock_model)
        
        # Case 1 UserProfile with mixed types including Optional fields
        expected_user = UserProfile(
            name="John Doe", 
            age=30, 
            is_active=True, 
            email="john@example.com",
            preferences={"theme": "dark", "notifications": True}
        )
        mock_response_1 = ModelResponse(
            parts=[TextPart(content=expected_user.model_dump_json())],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        mock_model.request = AsyncMock(return_value=mock_response_1)
        
        task_user = Task("Get user profile", response_format=UserProfile)
        result_user = agent.print_do(task_user)
        
        # Type check
        assert isinstance(result_user, UserProfile)
        assert isinstance(result_user.name, str)
        assert isinstance(result_user.age, int)
        assert isinstance(result_user.is_active, bool)
        assert isinstance(result_user.preferences, dict)
        
        # Case 2 Product with float and complex nested structures
        expected_product = Product(
            id=123,
            name="Test Product",
            price=99.99,
            in_stock=True,
            tags=["electronics", "gadget"],
            metadata={"category": "tech"}
        )
        mock_response_2 = ModelResponse(
            parts=[TextPart(content=expected_product.model_dump_json())],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        mock_model.request = AsyncMock(return_value=mock_response_2)
        
        task_product = Task("Get product details", response_format=Product)
        result_product = agent.print_do(task_product)
        
        # Type check
        assert isinstance(result_product, Product)
        assert isinstance(result_product.price, float)
        assert isinstance(result_product.tags, list)
        assert all(isinstance(tag, str) for tag in result_product.tags)
        
        # Case 3 MixedTypes with Union types and complex structures
        expected_mixed = MixedTypes(
            string_field="test",
            int_field=42,
            float_field=3.14,
            bool_field=True,
            list_field=["a", 1, "b", 2],
            dict_field={"key1": "value1", "key2": 123, "key3": True},
            optional_field=2.71
        )
        mock_response_3 = ModelResponse(
            parts=[TextPart(content=expected_mixed.model_dump_json())],
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            usage=None,
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
        mock_model.request = AsyncMock(return_value=mock_response_3)
        
        task_mixed = Task("Get mixed data", response_format=MixedTypes)
        result_mixed = agent.print_do(task_mixed)
        
        # Type check
        assert isinstance(result_mixed, MixedTypes)
        assert isinstance(result_mixed.list_field, list)
        assert isinstance(result_mixed.dict_field, dict)