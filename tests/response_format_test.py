import pytest
from unittest.mock import patch
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

    def test_task_response_format_behavior(self):
        """
        Test response_format parameter behavior:
        1. Without response_format: returns str
        2. With BaseModel response_format: returns BaseModel instance
        3. task.response always matches agent.print_do(task) result
        """
        
        # Case 1 Without response_format -> return str
        string_response = "My developer is OpenAI"
        
        task_no_format = Task("Who developed you?")
        agent = Agent(name="Coder")
        
        # Mock print_do 
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect_string(task):
                task._response = string_response 
                return string_response
            
            mock_print_do.side_effect = side_effect_string
            
            result_no_format = agent.print_do(task_no_format)
            
            # Type check
            assert isinstance(result_no_format, str)  
            assert isinstance(task_no_format.response, str) 
            
            # Does results match task.response?
            assert result_no_format == task_no_format.response  
            assert result_no_format == string_response  
        
        
        # Case 2 With BaseModel response_format -> return BaseModel instance
        expected_cities = ["Toronto", "Vancouver", "Montreal"]
        basemodel_response = TravelResponse(cities=expected_cities)
        
        task_with_format = Task(
            "Create a plan to visit cities in Canada", 
            response_format=TravelResponse
        )
        
        # Mock print_do 
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect_basemodel(task):
                task._response = basemodel_response  
                return basemodel_response
            
            mock_print_do.side_effect = side_effect_basemodel
            
            result_with_format = agent.print_do(task_with_format)
            
            # Type check
            assert isinstance(result_with_format, TravelResponse)  
            assert isinstance(task_with_format.response, TravelResponse)  
            
            # Field structure correctness
            assert isinstance(result_with_format.cities, list)  
            assert all(isinstance(city, str) for city in result_with_format.cities)  
            assert result_with_format.cities == expected_cities  
            assert len(result_with_format.cities) == 3 		 
            
            # Does result match task.response?
            assert result_with_format is task_with_format.response  
            assert result_with_format.cities == task_with_format.response.cities  

    def test_diverse_pydantic_types(self):
        """
        Test various Pydantic field types to ensure the system handles different data structures correctly.
        """
        agent = Agent(name="Tester")
        
        # Case 1 UserProfile with mixed types including Optional fields
        user_data = {
            "name": "John Doe",
            "age": 30,
            "is_active": True,
            "email": "john@example.com",
            "preferences": {"theme": "dark", "language": "en"}
        }
        user_response = UserProfile(**user_data)
        
        task_user = Task("Get user profile", response_format=UserProfile)
        
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect_user(task):
                task._response = user_response  
                return user_response
            
            mock_print_do.side_effect = side_effect_user
            result_user = agent.print_do(task_user)
            
            # Type check
            assert isinstance(result_user, UserProfile)
            assert isinstance(result_user.name, str)
            assert isinstance(result_user.age, int)
            assert isinstance(result_user.is_active, bool)
            assert isinstance(result_user.preferences, dict)
            assert result_user.name == "John Doe"
            assert result_user.age == 30
            assert result_user.is_active is True
        
        # Case 2 Product with float and complex nested structures
        product_data = {
            "id": 12345,
            "name": "Laptop",
            "price": 999.99,
            "in_stock": True,
            "tags": ["electronics", "computer", "portable"],
            "metadata": {"brand": "TechCorp", "warranty": "2 years"}
        }
        product_response = Product(**product_data)
        
        task_product = Task("Get product details", response_format=Product)
        
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect_product(task):
                task._response = product_response  
                return product_response
            
            mock_print_do.side_effect = side_effect_product
            result_product = agent.print_do(task_product)
            
            # Type check
            assert isinstance(result_product, Product)
            assert isinstance(result_product.price, float)
            assert isinstance(result_product.tags, list)
            assert all(isinstance(tag, str) for tag in result_product.tags)
            assert result_product.price == 999.99
            assert len(result_product.tags) == 3
        
        # Case 3 MixedTypes with Union types and complex structures
        mixed_data = {
            "string_field": "test string",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": False,
            "list_field": ["hello", 123, "world", 456],
            "dict_field": {"key1": "value1", "key2": 789, "key3": True},
            "optional_field": None
        }
        mixed_response = MixedTypes(**mixed_data)
        
        task_mixed = Task("Get mixed data", response_format=MixedTypes)
        
        with patch.object(agent, 'print_do') as mock_print_do:
            def side_effect_mixed(task):
                task._response = mixed_response  
                return mixed_response
            
            mock_print_do.side_effect = side_effect_mixed
            result_mixed = agent.print_do(task_mixed)
            
            # Type check
            assert isinstance(result_mixed, MixedTypes)
            assert isinstance(result_mixed.list_field, list)
            assert isinstance(result_mixed.dict_field, dict)
            assert result_mixed.optional_field is None
            assert len(result_mixed.list_field) == 4
            assert len(result_mixed.dict_field) == 3