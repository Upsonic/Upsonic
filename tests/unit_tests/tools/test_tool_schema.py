"""Unit tests for tool schema generation."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

from upsonic.tools.schema import (
    function_schema,
    SchemaGenerationError,
    FunctionSchema,
    GenerateToolJsonSchema,
)
from pydantic import BaseModel


class TestToolSchema:
    """Test suite for tool schema generation."""

    def test_tool_schema_generation(self):
        """Test schema generation."""

        def test_function(query: str, limit: int = 10) -> str:
            """Test function.

            Args:
                query: The search query.
                limit: Maximum results.

            Returns:
                The result string.
            """
            return f"Result: {query}"

        schema = function_schema(test_function, GenerateToolJsonSchema)

        assert isinstance(schema, FunctionSchema)
        assert schema.description is not None
        assert "query" in schema.json_schema["properties"]
        assert "limit" in schema.json_schema["properties"]
        assert "query" in schema.json_schema.get("required", [])
        assert "limit" not in schema.json_schema.get("required", [])

    def test_tool_schema_validation(self):
        """Test schema validation."""

        def valid_function(query: str) -> str:
            """Valid function."""
            return f"Result: {query}"

        def invalid_function(query):
            """Invalid function without type hints."""
            return f"Result: {query}"

        # Valid function should generate schema without errors
        schema = function_schema(valid_function, GenerateToolJsonSchema)
        assert isinstance(schema, FunctionSchema)

        # Invalid function should raise SchemaGenerationError
        with pytest.raises(SchemaGenerationError):
            function_schema(invalid_function, GenerateToolJsonSchema)

    def test_tool_schema_from_function(self):
        """Test schema from function."""

        def test_function(
            name: str, age: int, active: bool = True, tags: list[str] = None
        ) -> dict:
            """Test function with various types.

            Args:
                name: Person name.
                age: Person age.
                active: Whether active.
                tags: List of tags.

            Returns:
                Result dictionary.
            """
            return {"name": name, "age": age}

        schema = function_schema(test_function, GenerateToolJsonSchema)

        assert schema.description is not None
        assert "name" in schema.json_schema["properties"]
        assert "age" in schema.json_schema["properties"]
        assert "active" in schema.json_schema["properties"]
        assert "name" in schema.json_schema.get("required", [])
        assert "age" in schema.json_schema.get("required", [])
        assert "active" not in schema.json_schema.get("required", [])

    def test_tool_schema_with_pydantic_model(self):
        """Test schema with Pydantic models."""

        class UserModel(BaseModel):
            name: str
            age: int

        def test_function(user: UserModel) -> str:
            """Test function with Pydantic model.

            Args:
                user: User model.

            Returns:
                Result string.
            """
            return f"User: {user.name}"

        schema = function_schema(test_function, GenerateToolJsonSchema)

        assert schema.description is not None
        # Pydantic models get flattened - check for the model fields instead
        properties = schema.json_schema.get("properties", {})
        # The user parameter should be in properties, or its fields should be
        assert "user" in properties or "name" in properties or "age" in properties

    def test_tool_schema_async_function(self):
        """Test schema for async function."""

        async def async_function(query: str) -> str:
            """Async function.

            Args:
                query: The query.

            Returns:
                Result string.
            """
            return f"Result: {query}"

        schema = function_schema(async_function, GenerateToolJsonSchema)

        assert schema.is_async is True
        assert schema.description is not None

    def test_tool_schema_missing_docstring(self):
        """Test schema with missing docstring."""

        def no_docstring(query: str) -> str:
            return f"Result: {query}"

        # Missing docstring should raise SchemaGenerationError
        with pytest.raises(SchemaGenerationError):
            function_schema(no_docstring, GenerateToolJsonSchema)

    def test_tool_schema_validation_errors(self):
        """Test validation error reporting."""

        def function_without_type_hints(param):
            """Function without type hints."""
            return param

        # Function without type hints should raise SchemaGenerationError
        with pytest.raises(SchemaGenerationError) as exc_info:
            function_schema(function_without_type_hints, GenerateToolJsonSchema)
        assert "type hint" in str(exc_info.value).lower() or "annotation" in str(exc_info.value).lower()

    def test_tool_schema_optional_parameters(self):
        """Test schema with optional parameters."""

        def test_function(required: str, optional: Optional[str] = None) -> str:
            """Test function.

            Args:
                required: Required parameter.
                optional: Optional parameter.

            Returns:
                Result string.
            """
            return f"{required}: {optional}"

        schema = function_schema(test_function, GenerateToolJsonSchema)

        assert "required" in schema.json_schema.get("required", [])
        assert "optional" not in schema.json_schema.get("required", [])
