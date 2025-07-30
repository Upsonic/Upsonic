import pytest
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
        task_no_format = Task("Who developed you?")
        agent = Agent(name="Coder")
        
        result_no_format = agent.print_do(task_no_format)
        
        # Type check
        assert isinstance(result_no_format, str)  
        assert isinstance(task_no_format.response, str) 
        
        # Does results match task.response?
        assert result_no_format == task_no_format.response  
        
        
        # Case 2 With BaseModel response_format -> return BaseModel instance
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

    def test_diverse_pydantic_types(self):
        """
        Test various Pydantic field types to ensure the system handles different data structures correctly.
        """
        agent = Agent(name="Tester")
        
        # Case 1 UserProfile with mixed types including Optional fields
        task_user = Task("Get user profile", response_format=UserProfile)
        
        result_user = agent.print_do(task_user)
        
        # Type check
        assert isinstance(result_user, UserProfile)
        assert isinstance(result_user.name, str)
        assert isinstance(result_user.age, int)
        assert isinstance(result_user.is_active, bool)
        assert isinstance(result_user.preferences, dict)
        
        # Case 2 Product with float and complex nested structures
        task_product = Task("Get product details", response_format=Product)
        
        result_product = agent.print_do(task_product)
        
        # Type check
        assert isinstance(result_product, Product)
        assert isinstance(result_product.price, float)
        assert isinstance(result_product.tags, list)
        assert all(isinstance(tag, str) for tag in result_product.tags)
        
        # Case 3 MixedTypes with Union types and complex structures
        task_mixed = Task("Get mixed data", response_format=MixedTypes)
        
        result_mixed = agent.print_do(task_mixed)
        
        # Type check
        assert isinstance(result_mixed, MixedTypes)
        assert isinstance(result_mixed.list_field, list)
        assert isinstance(result_mixed.dict_field, dict)