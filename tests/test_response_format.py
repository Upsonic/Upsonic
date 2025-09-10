import pytest
from unittest.mock import patch, AsyncMock
from contextlib import asynccontextmanager
from upsonic import Task, Agent
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

    @patch('upsonic.models.factory.ModelFactory.create')
    @patch('upsonic.agent.agent.PydanticAgent')
    def test_task_response_format_behavior(self, mock_pydantic_agent, mock_factory_create):
        """
        Test response_format parameter behavior:
        1. Without response_format: returns str
        2. With BaseModel response_format: returns BaseModel instance
        3. task.response always matches agent.print_do(task) result
        """
        # Mock the factory to return a mock provider that doesn't need API keys
        mock_provider = AsyncMock()
        mock_model = AsyncMock()
        mock_provider._provision.return_value = (mock_model, None)
        mock_factory_create.return_value = mock_provider

        # Create a proper async context manager for run_mcp_servers
        @asynccontextmanager
        async def mock_run_mcp_servers():
            yield
        
        # Case 1 Without response_format -> return str
        mock_agent_instance_1 = AsyncMock()
        mock_response_1 = AsyncMock()
        mock_response_1.output = "I was developed by Upsonic, an AI agent framework."
        mock_response_1.all_messages = lambda: []
        mock_agent_instance_1.run.return_value = mock_response_1
        mock_agent_instance_1.run_mcp_servers = mock_run_mcp_servers
        mock_pydantic_agent.return_value = mock_agent_instance_1
        
        task_no_format = Task("Who developed you?")
        agent = Agent(name="Coder")
        
        result_no_format = agent.print_do(task_no_format)
        
        # Type check
        assert isinstance(result_no_format, str)  
        assert isinstance(task_no_format.response, str) 
        
        # Does results match task.response?
        assert result_no_format == task_no_format.response  
        
        # Case 2 With BaseModel response_format -> return BaseModel instance
        mock_agent_instance_2 = AsyncMock()
        mock_response_2 = AsyncMock()
        mock_response_2.output = TravelResponse(cities=["Toronto", "Vancouver", "Montreal"])
        mock_response_2.all_messages = lambda: []
        mock_agent_instance_2.run.return_value = mock_response_2
        mock_agent_instance_2.run_mcp_servers = mock_run_mcp_servers
        mock_pydantic_agent.return_value = mock_agent_instance_2
        
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
        assert result_with_format is task_with_format.response  
        assert result_with_format.cities == task_with_format.response.cities  

    @patch('upsonic.models.factory.ModelFactory.create')
    @patch('upsonic.agent.agent.PydanticAgent')
    def test_diverse_pydantic_types(self, mock_pydantic_agent, mock_factory_create):
        """
        Test various Pydantic field types to ensure the system handles different data structures correctly.
        """
        # Mock the factory to return a mock provider that doesn't need API keys
        mock_provider = AsyncMock()
        mock_model = AsyncMock()
        mock_provider._provision.return_value = (mock_model, None)
        mock_factory_create.return_value = mock_provider

        # Create a proper async context manager for run_mcp_servers
        @asynccontextmanager
        async def mock_run_mcp_servers():
            yield
        
        agent = Agent(name="Tester")
        
        # Case 1 UserProfile with mixed types including Optional fields
        mock_agent_instance_1 = AsyncMock()
        mock_response_1 = AsyncMock()
        mock_response_1.output = UserProfile(
            name="John Doe", 
            age=30, 
            is_active=True, 
            email="john@example.com",
            preferences={"theme": "dark", "notifications": True}
        )
        mock_response_1.all_messages = lambda: []
        mock_agent_instance_1.run.return_value = mock_response_1
        mock_agent_instance_1.run_mcp_servers = mock_run_mcp_servers
        mock_pydantic_agent.return_value = mock_agent_instance_1
        
        task_user = Task("Get user profile", response_format=UserProfile)
        result_user = agent.print_do(task_user)
        
        # Type check
        assert isinstance(result_user, UserProfile)
        assert isinstance(result_user.name, str)
        assert isinstance(result_user.age, int)
        assert isinstance(result_user.is_active, bool)
        assert isinstance(result_user.preferences, dict)
        
        # Case 2 Product with float and complex nested structures
        mock_agent_instance_2 = AsyncMock()
        mock_response_2 = AsyncMock()
        mock_response_2.output = Product(
            id=123,
            name="Test Product",
            price=99.99,
            in_stock=True,
            tags=["electronics", "gadget"],
            metadata={"category": "tech"}
        )
        mock_response_2.all_messages = lambda: []
        mock_agent_instance_2.run.return_value = mock_response_2
        mock_agent_instance_2.run_mcp_servers = mock_run_mcp_servers
        mock_pydantic_agent.return_value = mock_agent_instance_2
        
        task_product = Task("Get product details", response_format=Product)
        result_product = agent.print_do(task_product)
        
        # Type check
        assert isinstance(result_product, Product)
        assert isinstance(result_product.price, float)
        assert isinstance(result_product.tags, list)
        assert all(isinstance(tag, str) for tag in result_product.tags)
        
        # Case 3 MixedTypes with Union types and complex structures
        mock_agent_instance_3 = AsyncMock()
        mock_response_3 = AsyncMock()
        mock_response_3.output = MixedTypes(
            string_field="test",
            int_field=42,
            float_field=3.14,
            bool_field=True,
            list_field=["a", 1, "b", 2],
            dict_field={"key1": "value1", "key2": 123, "key3": True},
            optional_field=2.71
        )
        mock_response_3.all_messages = lambda: []
        mock_agent_instance_3.run.return_value = mock_response_3
        mock_agent_instance_3.run_mcp_servers = mock_run_mcp_servers
        mock_pydantic_agent.return_value = mock_agent_instance_3
        
        task_mixed = Task("Get mixed data", response_format=MixedTypes)
        result_mixed = agent.print_do(task_mixed)
        
        # Type check
        assert isinstance(result_mixed, MixedTypes)
        assert isinstance(result_mixed.list_field, list)
        assert isinstance(result_mixed.dict_field, dict)